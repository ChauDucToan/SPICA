from collections.abc import Mapping
from dataclasses import dataclass

from open_clip.transformer import text_global_pool
import torch
from torch import Tensor, nn

from ..models.clip import FrozenClipEncoder, TextTokenizer


@dataclass(frozen=True, slots=True)
class EncodedTextBank:
    embeddings: Tensor
    labels: Tensor
    class_names: tuple[str, ...]
    prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                "Text embeddings must have shape [num_classes, embedding_dim], "
                f"got {tuple(self.embeddings.shape)}"
            )

        num_classes = self.embeddings.shape[0]
        if self.labels.shape != (num_classes,):
            raise ValueError(
                f"Text labels must have shape [{num_classes}], "
                f"got {tuple(self.labels.shape)}"
            )
        if len(self.class_names) != num_classes or len(self.prompts) != num_classes:
            raise ValueError(
                "Text embeddings, labels, class names, and prompts must contain "
                "the same number of classes"
            )
        if torch.unique(self.labels).numel() != num_classes:
            raise ValueError("Text bank labels must be unique")

    def embeddings_for_labels(self, labels: Tensor) -> Tensor:
        if labels.ndim != 1:
            raise ValueError(
                f"Lookup labels must have shape [num_items], got {tuple(labels.shape)}"
            )

        label_to_index = {
            int(label.item()): index for index, label in enumerate(self.labels)
        }
        try:
            indices = [label_to_index[int(label)] for label in labels.tolist()]
        except KeyError as error:
            raise ValueError(
                f"Label {error.args[0]} is missing from the text bank"
            ) from error

        return self.embeddings[torch.tensor(indices, dtype=torch.long)]


class SoftPromptTextBank(nn.Module):
    """Trainable CLIP text prompts used only as a JEPA loss-side classifier.

    The CLIP text tower remains frozen.  A small learned context is inserted
    after the start token and before the class-name tokens, so the resulting
    bank can adapt the supervision geometry without becoming an inference
    input to :class:`~spica.models.jepa.SketchPhotoJepa`.
    """

    def __init__(
        self,
        encoder: FrozenClipEncoder,
        tokenizer: TextTokenizer,
        class_names: Mapping[int, str],
        *,
        prompt_length: int = 4,
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("Cannot create a soft prompt bank from an empty class map")
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")

        sorted_classes = sorted(class_names.items())
        self.class_names = tuple(class_name for _, class_name in sorted_classes)
        self.labels = torch.tensor(
            [class_id for class_id, _ in sorted_classes], dtype=torch.long
        )
        self.prompt_length = prompt_length
        self.prompts = tuple(
            f"[soft_prompt:{prompt_length}] {name.replace('_', ' ')}"
            for name in self.class_names
        )

        names = [name.replace("_", " ") for name in self.class_names]
        class_tokens = tokenizer(names)
        if not isinstance(class_tokens, Tensor) or class_tokens.ndim != 2:
            raise TypeError(
                "The selected OpenCLIP tokenizer must return [classes, context]"
            )
        if class_tokens.shape[0] != len(names):
            raise ValueError("Tokenizer changed the number of class names")

        clip_model = encoder.model
        context_length = int(class_tokens.shape[1])
        if prompt_length + 2 > context_length:
            raise ValueError("prompt_length is too long for the CLIP context window")
        sot_id = int(getattr(tokenizer, "sot_token_id", class_tokens[0, 0].item()))
        eot_id = int(
            getattr(
                tokenizer,
                "eot_token_id",
                int(class_tokens.max().item()),
            )
        )
        token_rows: list[list[int]] = []
        for row in class_tokens.tolist():
            try:
                eot_position = row.index(eot_id)
            except ValueError as error:
                raise ValueError(
                    "Tokenizer output does not contain an EOT token"
                ) from error
            content = row[1:eot_position]
            if len(content) + prompt_length + 2 > context_length:
                raise ValueError(
                    f"Class name {content!r} does not fit after {prompt_length} soft tokens"
                )
            sequence = [sot_id] + [0] * prompt_length + content + [eot_id]
            sequence.extend([0] * (context_length - len(sequence)))
            token_rows.append(sequence)
        self.register_buffer("token_ids", torch.tensor(token_rows, dtype=torch.long))
        self.register_buffer("class_labels", self.labels.clone())

        token_embedding = clip_model.token_embedding
        prefix_tokens = tokenizer(["a photo of a"])
        if not isinstance(prefix_tokens, Tensor):
            raise TypeError("The selected OpenCLIP tokenizer must return a tensor")
        prefix_row = prefix_tokens[0]
        prefix_eot = int(getattr(tokenizer, "eot_token_id", prefix_row.max().item()))
        prefix_position = (prefix_row == prefix_eot).nonzero(as_tuple=False)
        prefix_content = prefix_row[1 : int(prefix_position[0].item())]
        with torch.no_grad():
            prefix_embedding = token_embedding(
                prefix_content.to(encoder.device)
            ).float()
        model_width = int(token_embedding.embedding_dim)
        initialized = torch.empty(
            prompt_length,
            model_width,
            device=encoder.device,
            dtype=torch.float32,
        )
        if prefix_embedding.shape[0] >= prompt_length:
            initialized.copy_(prefix_embedding[:prompt_length])
        else:
            initialized.normal_(
                mean=0.0,
                std=float(prefix_embedding.std(unbiased=False).item()),
            )
            initialized[: prefix_embedding.shape[0]].copy_(prefix_embedding)
        self.context = nn.Parameter(initialized)

        # Do not register the frozen CLIP module as a child: its parameters are
        # owned by the photo encoder and the soft bank should serialize only the
        # small learned context and token metadata.
        object.__setattr__(self, "_clip_model", clip_model)
        clip_model.requires_grad_(False)
        clip_model.eval()

    @property
    def embedding_dim(self) -> int:
        projection = self._clip_model.text_projection
        if isinstance(projection, nn.Linear):
            return int(projection.out_features)
        return int(projection.shape[-1])

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def device(self) -> torch.device:
        return self.context.device

    def forward(self) -> Tensor:
        model = self._clip_model
        tokens = self.token_ids.to(device=self.device)
        cast_dtype = model.transformer.get_cast_dtype()
        fixed = model.token_embedding(tokens).to(cast_dtype)
        learned = (
            self.context.to(dtype=cast_dtype)
            .unsqueeze(0)
            .expand(tokens.shape[0], -1, -1)
        )
        values = torch.cat(
            (fixed[:, :1], learned, fixed[:, 1 + self.prompt_length :]), dim=1
        )
        values = values + model.positional_embedding.to(cast_dtype)
        values = model.transformer(values, attn_mask=model.attn_mask)
        values = model.ln_final(values)
        values = text_global_pool(
            values,
            tokens,
            model.text_pool_type,
            eos_token_id=getattr(model, "text_eos_id", None),
        )
        projection = model.text_projection
        if projection is not None:
            if isinstance(projection, nn.Linear):
                values = projection(values)
            else:
                values = values @ projection
        return torch.nn.functional.normalize(values, dim=-1)


@torch.inference_mode()
def encode_class_text_bank(
    encoder: FrozenClipEncoder,
    tokenizer: TextTokenizer,
    class_names: Mapping[int, str],
    *,
    prompt_template: str = "a photo of a {}",
) -> EncodedTextBank:
    if not class_names:
        raise ValueError("Cannot encode an empty class map")
    if "{}" not in prompt_template:
        raise ValueError("prompt_template must contain one positional '{}' placeholder")

    sorted_classes = sorted(class_names.items())
    labels = torch.tensor(
        [class_id for class_id, _ in sorted_classes],
        dtype=torch.long,
    )
    names = tuple(class_name for _, class_name in sorted_classes)
    prompt_names = tuple(name.replace("_", " ") for name in names)

    try:
        prompts = tuple(prompt_template.format(name) for name in prompt_names)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(
            "prompt_template must contain one positional '{}' placeholder"
        ) from error

    tokens = tokenizer(prompts)
    if not isinstance(tokens, Tensor):
        raise TypeError("The selected OpenCLIP tokenizer must return a tensor")

    embeddings = encoder.encode_text(tokens.to(encoder.device)).float().cpu()
    return EncodedTextBank(
        embeddings=embeddings,
        labels=labels,
        class_names=names,
        prompts=prompts,
    )
