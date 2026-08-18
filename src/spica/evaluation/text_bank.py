from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

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
