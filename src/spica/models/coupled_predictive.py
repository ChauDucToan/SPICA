"""CPU-safe V1 coupled predictive model.

The constructor deliberately accepts already-created CLIP objects.  Loading
weights, choosing transforms, masking, losses, and training orchestration do
not belong here.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..evaluation.text_bank import SoftPromptTextBank, encode_class_text_bank
from ..models.clip import FrozenClipEncoder
from ..semantic_text import clone_fixed_text_bank
from .frozen_prompt import FrozenPromptModel


@dataclass(frozen=True, slots=True)
class CoupledPredictiveOutput:
    """The four query-side representations used by the V1 losses."""

    g: Tensor
    q: Tensor
    mu_i: Tensor
    mu_t: Tensor

    def __getitem__(self, key: str) -> Tensor:
        if key not in {"g", "q", "mu_i", "mu_t"}:
            raise KeyError(key)
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return ("g", "q", "mu_i", "mu_t")


class _PromptPredictor(nn.Module):
    """One small shared cross-attention block; prompts are call-time inputs."""

    def __init__(
        self,
        context_dim: int,
        photo_dim: int,
        text_dim: int,
        *,
        width: int = 256,
        heads: int = 4,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        for name, value in (
            ("context_dim", context_dim),
            ("photo_dim", photo_dim),
            ("text_dim", text_dim),
            ("width", width),
            ("heads", heads),
            ("output_dim", output_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if width % heads:
            raise ValueError("predictor width must be divisible by heads")

        self.context_dim = context_dim
        self.photo_dim = photo_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.context_in = nn.Linear(context_dim, width)
        self.photo_in = nn.Linear(photo_dim, width)
        self.text_in = nn.Linear(text_dim, width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.norm2 = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(
        self,
        context: Tensor,
        photo_prompt: Tensor,
        text_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            context.ndim != 3
            or context.shape[0] == 0
            or context.shape[1] == 0
            or context.shape[2] != self.context_dim
        ):
            raise ValueError(
                "context must have shape [batch>0, tokens>0, context_dim]"
            )
        if (
            photo_prompt.ndim != 2
            or photo_prompt.shape[0] == 0
            or photo_prompt.shape[1] != self.photo_dim
        ):
            raise ValueError("photo_prompt must have shape [tokens>0, photo_dim]")
        if (
            text_context.ndim != 2
            or text_context.shape[0] == 0
            or text_context.shape[1] != self.text_dim
        ):
            raise ValueError("text_context must have shape [tokens>0, text_dim]")
        for name, value in (
            ("context", context),
            ("photo_prompt", photo_prompt),
            ("text_context", text_context),
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite floating-point values")

        memory = self.context_in(context)
        photo_count = photo_prompt.shape[0]
        queries = torch.cat(
            (self.photo_in(photo_prompt), self.text_in(text_context)), dim=0
        )
        queries = queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        attended, _ = self.attention(queries, memory, memory, need_weights=False)
        values = self.norm1(queries + attended)
        values = self.norm2(values + self.ffn(values))
        values = self.output(values)
        photo = values[:, :photo_count].mean(dim=1)
        text = values[:, photo_count:].mean(dim=1)
        return _unit(photo, name="photo prediction"), _unit(text, name="text prediction")


def _unit(value: Tensor, *, name: str) -> Tensor:
    if value.ndim != 2 or value.shape[0] == 0:
        raise RuntimeError(f"{name} must have shape [batch>0, dimension]")
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name} must be finite floating-point values")
    if (value.norm(dim=-1) == 0).any().item():
        raise RuntimeError(f"{name} contains a zero row")
    return F.normalize(value, dim=-1)


class CoupledPredictiveModel(nn.Module):
    """V1 sketch query model with frozen original CLIP teacher.

    ``photo_encoder`` owns the supplied CLIP before construction.  This model
    registers the underlying CLIP exactly once as ``original_clip``; the prompt
    wrappers and text bank keep only non-registering aliases to that teacher.
    """

    def __init__(
        self,
        photo_encoder: FrozenClipEncoder,
        tokenizer: Any,
        class_names: dict[int, str],
        *,
        photo_prompt_length: int = 3,
        text_prompt_length: int = 4,
        predictor_width: int = 256,
        predictor_heads: int = 4,
    ) -> None:
        super().__init__()
        if not isinstance(photo_encoder, FrozenClipEncoder):
            raise TypeError("photo_encoder must be a FrozenClipEncoder")
        if not class_names:
            raise ValueError("class_names must not be empty")
        if any(isinstance(key, bool) or not isinstance(key, int) or key < 0 for key in class_names):
            raise ValueError("class_names keys must be nonnegative integer class IDs")
        if any(not isinstance(name, str) or not name.strip() for name in class_names.values()):
            raise ValueError("class names must be nonempty strings")
        if photo_prompt_length != 3 or text_prompt_length != 4:
            raise ValueError("V1 requires a 3-token photo prompt and 4-token text context")

        original_clip = photo_encoder.model
        if not isinstance(original_clip, nn.Module):
            raise TypeError("photo_encoder.model must be a torch module")
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in original_clip.parameters()):
            raise ValueError("V1 currently requires a float32 CLIP; mixed precision is not gated")
        original_clip.requires_grad_(False)
        original_clip.eval()
        # This is the sole registered owner of all original image/text CLIP state.
        self.original_clip = original_clip
        object.__setattr__(self, "_photo_encoder", photo_encoder)

        visual = getattr(original_clip, "visual", None)
        FrozenPromptModel._validate_visual(visual)
        self.photo_model = FrozenPromptModel(
            visual,
            prompt_length=photo_prompt_length,
            train_sketch_prompt=False,
            train_photo_prompt=True,
        )
        # FrozenPromptModel is reused for its validated prompted-photo path and
        # prompt parameters, but the teacher visual must remain owned by CLIP.
        self.photo_model._modules.pop("visual")
        object.__setattr__(self.photo_model, "visual", visual)

        self.text_bank = SoftPromptTextBank(
            photo_encoder,
            tokenizer,
            class_names,
            prompt_length=text_prompt_length,
            strict_prefix=True,
        )
        hard_bank = encode_class_text_bank(photo_encoder, tokenizer, class_names)
        fixed = clone_fixed_text_bank(hard_bank.embeddings, device=visual.positional_embedding.device)
        if fixed.ndim != 2 or fixed.shape[0] == 0 or fixed.shape[1] == 0:
            raise RuntimeError("fixed text bank has an invalid shape")
        if fixed.shape[1] != int(getattr(visual, "output_dim", fixed.shape[1])):
            raise RuntimeError("CLIP visual and text embedding dimensions disagree")
        self.text_bank.to(device=fixed.device)
        self.register_buffer("T0", fixed)
        with torch.no_grad():
            learned = self.text_bank()
            torch.testing.assert_close(learned.float(), fixed.float(), atol=1e-6, rtol=1e-6)
        self.initial_text_parity_max_error = float((learned - fixed).abs().max())

        self.student_visual = deepcopy(visual)
        FrozenPromptModel._validate_visual(self.student_visual)
        self.student_visual.requires_grad_(True)
        for name in ("ln_post", "proj"):
            component = getattr(self.student_visual, name, None)
            if component is not None and hasattr(component, "requires_grad_"):
                component.requires_grad_(False)
        context_dim = int(self.student_visual.positional_embedding.shape[1])
        photo_dim = int(self.photo_model.photo_prompt.shape[1])
        text_dim = int(self.text_bank.context.shape[1])
        self.output_dim = int(fixed.shape[1])
        self.predictor = _PromptPredictor(
            context_dim,
            photo_dim,
            text_dim,
            width=predictor_width,
            heads=predictor_heads,
            output_dim=self.output_dim,
        )
        self.pooled_head = nn.Linear(context_dim, self.output_dim)
        # Newly constructed heads must follow a supplied teacher's device.
        self.predictor.to(device=fixed.device)
        self.pooled_head.to(device=fixed.device)
        self.train(True)

    @staticmethod
    def _validate_rgb(images: Tensor, *, name: str) -> None:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, 3, height, width]")
        if images.shape[0] == 0 or images.shape[2] == 0 or images.shape[3] == 0:
            raise ValueError(f"{name} must have nonempty batch and spatial dimensions")
        if images.shape[1] != 3:
            raise ValueError(f"{name} must have exactly three RGB channels")
        if not images.is_floating_point():
            raise TypeError(f"{name} must be floating-point")
        if not torch.isfinite(images).all().item():
            raise ValueError(f"{name} must contain only finite values")

    def _context_tokens(self, sketches: Tensor) -> Tensor:
        tokens = self.student_visual._embeds(sketches)
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 3
            or tokens.shape[0] != sketches.shape[0]
            or tokens.shape[1] < 2
            or tokens.shape[2] != self.student_visual.positional_embedding.shape[1]
        ):
            raise RuntimeError("OpenCLIP visual _embeds returned an invalid token tensor")
        values = self.student_visual.transformer(tokens)
        if values.ndim != 3 or values.shape[1] < 2:
            raise RuntimeError("OpenCLIP visual transformer returned invalid tokens")
        return values[:, 1:]

    def forward(self, sketches: Tensor) -> CoupledPredictiveOutput:
        """Encode sketches only; labels and target/gallery inputs are forbidden."""
        self._validate_rgb(sketches, name="sketches")
        if sketches.device != self.student_visual.positional_embedding.device:
            raise ValueError("sketches and model must be on the same device")
        h = self._context_tokens(sketches)
        g = h.mean(dim=1)
        q = _unit(self.pooled_head(g), name="q")
        mu_i, mu_t = self.predictor(
            h,
            self.photo_model.photo_prompt,
            self.text_bank.context,
        )
        return CoupledPredictiveOutput(g=g, q=q, mu_i=mu_i, mu_t=mu_t)

    def encode_photo(self, photos: Tensor) -> Tensor:
        """Live prompted photo embeddings for the loss side."""
        self._validate_rgb(photos, name="photos")
        if photos.device != self.student_visual.positional_embedding.device:
            raise ValueError("photos and model must be on the same device")
        return _unit(self.photo_model.encode_photo(photos), name="photo embedding")

    def photo_reference(self, photos: Tensor) -> Tensor:
        """Normal detached, unprompted original-CLIP photo targets."""
        self._validate_rgb(photos, name="photos")
        if photos.device != self.student_visual.positional_embedding.device:
            raise ValueError("photos and model must be on the same device")
        with torch.no_grad():
            result = self._photo_encoder.encode_image(photos)
        return _unit(result.detach(), name="photo reference").detach()

    @property
    def classids(self) -> Tensor:
        """Use the text bank's sole ordered class-ID buffer."""
        return self.text_bank.class_labels

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # Class/token metadata must match the constructor, not just tensor shapes.
        for name in ("class_labels", "token_ids"):
            key = f"{prefix}text_bank.{name}"
            if key in state_dict and not torch.equal(
                state_dict[key].cpu(), getattr(self.text_bank, name).cpu()
            ):
                raise RuntimeError(f"checkpoint {key} differs from the constructed class bank")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def fixed_text_bank(self) -> Tensor:
        """The immutable hard-token T0 bank, owned by this state dict."""
        return self.T0

    def optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        """Return disjoint AdamW-ready starting groups; no optimizer is created."""
        groups: list[dict[str, Any]] = []
        seen: set[int] = set()

        def add_group(name: str, pairs: list[tuple[str, nn.Parameter]], lr: float, wd: float) -> None:
            for _, parameter in pairs:
                if not parameter.requires_grad:
                    raise RuntimeError("optimizer groups contain a frozen parameter")
                if id(parameter) in seen:
                    raise RuntimeError("optimizer groups contain a duplicate parameter")
                seen.add(id(parameter))
            if pairs:
                groups.append({
                    "name": name,
                    "params": [parameter for _, parameter in pairs],
                    "parameter_names": [parameter_name for parameter_name, _ in pairs],
                    "lr": lr,
                    "weight_decay": wd,
                })

        def split_module(prefix: str, module: nn.Module, lr: float, wd: float) -> None:
            layernorm_names = {
                f"{module_name}.{parameter_name}"
                for module_name, layer in module.named_modules()
                if isinstance(layer, nn.LayerNorm)
                for parameter_name, _ in layer.named_parameters(recurse=False)
            }
            decay, no_decay = [], []
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                (no_decay if parameter.ndim < 2 or name in layernorm_names or name.endswith(".bias") else decay).append((f"{prefix}.{name}", parameter))
            add_group(prefix, decay, lr, wd)
            add_group(f"{prefix}_no_decay", no_decay, lr, 0.0)

        split_module("student_visual", self.student_visual, 1e-5, 1e-2)
        split_module("predictor", self.predictor, 1e-4, 1e-2)
        split_module("pooled_head", self.pooled_head, 1e-4, 1e-2)
        add_group(
            "prompts",
            [("photo_model.photo_prompt", self.photo_model.photo_prompt),
             ("text_bank.context", self.text_bank.context)],
            1e-4,
            1e-4,
        )
        trainable = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if seen != trainable:
            raise RuntimeError("optimizer groups do not consume exactly the trainable parameters")
        return groups

    parameter_groups = optimizer_parameter_groups

    def train(self, mode: bool = True) -> "CoupledPredictiveModel":
        super().train(mode)
        self.original_clip.eval()
        self.photo_model.visual.eval()
        self.student_visual.train(mode)
        return self


__all__ = ["CoupledPredictiveModel", "CoupledPredictiveOutput"]
