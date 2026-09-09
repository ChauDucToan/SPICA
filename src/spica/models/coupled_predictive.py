"""Coupled predictive V1 and explicit contextual-prompt fusion V2."""

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
    """Representations emitted by one sketch view."""

    g: Tensor
    q: Tensor
    mu_i: Tensor | None
    mu_t: Tensor | None


class _PromptPredictor(nn.Module):
    """One shared cross-attention block conditioned by the two prompt banks."""

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
        self.context_dim = context_dim
        self.photo_dim = photo_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.context_in = nn.Linear(context_dim, width)
        self.photo_in = nn.Linear(photo_dim, width)
        self.text_in = nn.Linear(text_dim, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.norm2 = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)
        self.fusion_attention: nn.MultiheadAttention | None = None
        self.fusion_norm: nn.LayerNorm | None = None

    def forward(
        self,
        context: Tensor,
        photo_prompt: Tensor,
        text_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        memory = self.context_in(context)
        photo_count = photo_prompt.shape[0]
        queries = torch.cat(
            (self.photo_in(photo_prompt), self.text_in(text_context)), dim=0
        )
        queries = queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        attended, _ = self.attention(queries, memory, memory, need_weights=False)
        values = self.norm1(queries + attended)
        if self.fusion_attention is not None:
            # Fuse tokens after they have read THIS sketch, preserving each token's context.
            fused, _ = self.fusion_attention(values, values, values, need_weights=False)
            values = self.fusion_norm(values + fused)
        values = self.norm2(values + self.ffn(values))
        values = self.output(values)
        photo = F.normalize(values[:, :photo_count].mean(dim=1), dim=-1)
        text = F.normalize(values[:, photo_count:].mean(dim=1), dim=-1)
        return photo, text


class CoupledPredictiveModel(nn.Module):
    """Fine-tuned sketch context encoder with pooled or predictive outputs.

    The supplied original CLIP is registered only as ``original_clip``.  The
    prompt wrappers retain non-registering aliases to that same frozen teacher.
    """

    def __init__(
        self,
        photo_encoder: FrozenClipEncoder,
        tokenizer: Any,
        class_names: dict[int, str],
        *,
        architecture: str = "predictive",
        photo_prompt_length: int = 3,
        text_prompt_length: int = 4,
        predictor_width: int = 256,
        predictor_heads: int = 4,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        original_clip = photo_encoder.model
        original_clip.requires_grad_(False)
        original_clip.eval()
        # Sole registered owner of original CLIP state; all other references are aliases.
        self.original_clip = original_clip
        object.__setattr__(self, "_photo_encoder", photo_encoder)

        visual = original_clip.visual
        self.photo_model = FrozenPromptModel(
            visual,
            prompt_length=photo_prompt_length,
            train_sketch_prompt=False,
            train_photo_prompt=True,
        )
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
        fixed = clone_fixed_text_bank(
            hard_bank.embeddings,
            device=visual.positional_embedding.device,
        )
        self.text_bank.to(device=fixed.device)
        self.register_buffer("T0", fixed)
        with torch.no_grad():
            learned = self.text_bank()
        self.initial_text_parity_max_error = float((learned - fixed).abs().max())

        self.student_visual = deepcopy(visual)
        self.student_visual.requires_grad_(True)
        for name in ("ln_post", "proj"):
            component = getattr(self.student_visual, name, None)
            if component is not None:
                component.requires_grad_(False)

        context_dim = int(self.student_visual.positional_embedding.shape[1])
        photo_dim = int(self.photo_model.photo_prompt.shape[1])
        text_dim = int(self.text_bank.context.shape[1])
        self.output_dim = int(fixed.shape[1])
        predictor = _PromptPredictor(
            context_dim,
            photo_dim,
            text_dim,
            width=predictor_width,
            heads=predictor_heads,
            output_dim=self.output_dim,
        )
        # Construct predictor before pooled_head in every arm for identical RNG draws.
        self.pooled_head = nn.Linear(context_dim, self.output_dim)
        predictor.to(device=fixed.device)
        self.pooled_head.to(device=fixed.device)
        self.predictor = predictor
        # R0 still pays the predictor construction RNG cost, but has no predictor state.
        if architecture == "pooled":
            self.predictor = None
        elif architecture == "predictive_fusion_v2":
            # Add modules after all V1/common initialization; old parameter draws stay identical.
            self.predictor.fusion_attention = nn.MultiheadAttention(
                predictor_width, predictor_heads, dropout=0.0, batch_first=True,
            ).to(device=fixed.device)
            self.predictor.fusion_norm = nn.LayerNorm(predictor_width).to(device=fixed.device)
        self.train(True)

    @property
    def device(self) -> torch.device:
        return next(self.original_clip.parameters()).device

    def _context_tokens(self, sketches: Tensor) -> Tensor:
        tokens = self.student_visual._embeds(sketches)
        return self.student_visual.transformer(tokens)[:, 1:]

    def forward(self, sketches: Tensor) -> CoupledPredictiveOutput:
        """Encode sketches without labels, targets, or gallery inputs."""
        h = self._context_tokens(sketches)
        g = h.mean(dim=1)
        q = F.normalize(self.pooled_head(g), dim=-1)
        if self.predictor is None:
            return CoupledPredictiveOutput(g=g, q=q, mu_i=None, mu_t=None)
        mu_i, mu_t = self.predictor(
            h,
            self.photo_model.photo_prompt,
            self.text_bank.context,
        )
        return CoupledPredictiveOutput(g=g, q=q, mu_i=mu_i, mu_t=mu_t)

    def encode_photo(self, photos: Tensor) -> Tensor:
        """Encode the live prompted photo bank."""
        return self.photo_model.encode_photo(photos)

    def photo_reference(self, photos: Tensor) -> Tensor:
        """Encode detached, unprompted original-CLIP photo targets."""
        with torch.no_grad():
            return self._photo_encoder.encode_image(photos).detach()

    @property
    def classids(self) -> Tensor:
        return self.text_bank.class_labels

    def optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        """Return AdamW groups for exactly the active trainable modules."""
        groups: list[dict[str, Any]] = []

        def add_group(
            name: str,
            pairs: list[tuple[str, nn.Parameter]],
            lr: float,
            wd: float,
        ) -> None:
            if pairs:
                groups.append(
                    {
                        "name": name,
                        "params": [parameter for _, parameter in pairs],
                        "parameter_names": [parameter_name for parameter_name, _ in pairs],
                        "lr": lr,
                        "weight_decay": wd,
                    }
                )

        def split_module(prefix: str, module: nn.Module, lr: float, wd: float) -> None:
            layernorm_names = {
                f"{module_name}.{parameter_name}"
                for module_name, layer in module.named_modules()
                if isinstance(layer, nn.LayerNorm)
                for parameter_name, _ in layer.named_parameters(recurse=False)
            }
            decay: list[tuple[str, nn.Parameter]] = []
            no_decay: list[tuple[str, nn.Parameter]] = []
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                target = no_decay if (
                    parameter.ndim < 2
                    or name in layernorm_names
                    or name.endswith(".bias")
                ) else decay
                target.append((f"{prefix}.{name}", parameter))
            add_group(prefix, decay, lr, wd)
            add_group(f"{prefix}_no_decay", no_decay, lr, 0.0)

        split_module("student_visual", self.student_visual, 1e-5, 1e-2)
        if self.predictor is not None:
            split_module("predictor", self.predictor, 1e-4, 1e-2)
        split_module("pooled_head", self.pooled_head, 1e-4, 1e-2)
        add_group(
            "prompts",
            [
                ("photo_model.photo_prompt", self.photo_model.photo_prompt),
                ("text_bank.context", self.text_bank.context),
            ],
            1e-4,
            1e-4,
        )
        return groups

    def train(self, mode: bool = True) -> "CoupledPredictiveModel":
        super().train(mode)
        self.original_clip.eval()
        self.photo_model.visual.eval()
        self.student_visual.train(mode)
        return self


__all__ = ["CoupledPredictiveModel", "CoupledPredictiveOutput"]
