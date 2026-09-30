"""Minimal SketchLVM visual-prompt adaptation of an OpenCLIP visual encoder."""

from copy import deepcopy

import torch
from jaxtyping import Float
from torch import Tensor, nn

RawImages = Float[Tensor, "batch 3 224 224"]
Embeddings = Float[Tensor, "batch embedding"]
Tokens = Float[Tensor, "batch length width"]
PromptedTokens = Float[Tensor, "batch length+3 width"]


class AppendPrompt(nn.Module):
    """Append shallow visual tokens after native CLS and patch tokens."""

    def __init__(self, width: int) -> None:
        super().__init__()
        # Paper does not specify initialization; follow released code's randn.
        self.prompts = nn.Parameter(torch.randn(3, width))

    def forward(self, tokens: Tokens) -> PromptedTokens:
        prompts = (
            self.prompts.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        )
        return torch.cat((tokens, prompts), dim=1)


class PromptedVisual(nn.Module):
    """One independent CLIP visual branch with trainable prompts and LayerNorms."""

    def __init__(
        self,
        base_visual: nn.Module,
        mean: tuple[float, float, float] | list[float],
        std: tuple[float, float, float] | list[float],
    ) -> None:
        super().__init__()
        self.visual = deepcopy(base_visual)
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.visual.requires_grad_(False)
        width = int(self.visual.ln_pre.normalized_shape[0])
        self.visual.ln_pre = nn.Sequential(AppendPrompt(width), self.visual.ln_pre)
        for module in self.visual.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
        self.train(False)

    def train(self, mode: bool = True) -> "PromptedVisual":
        super().train(mode)
        # Frozen CLIP stays in inference mode; LayerNorm affine parameters still
        # receive gradients, and the branch contains no trainable dropout.
        self.visual.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, images: RawImages) -> Embeddings:
        images = (images - self.mean.to(images)) / self.std.to(images)
        return torch.nn.functional.normalize(self.visual(images), dim=-1)


class SketchLVM(nn.Module):
    """Separate sketch/photo shallow-prompt CLIP branches."""

    def __init__(
        self,
        visual: nn.Module,
        mean: tuple[float, float, float] | list[float],
        std: tuple[float, float, float] | list[float],
    ) -> None:
        super().__init__()
        self.sketch = PromptedVisual(visual, mean, std)
        self.photo = PromptedVisual(visual, mean, std)

    @property
    def device(self) -> torch.device:
        return self.sketch.device

    def forward(self, images: RawImages) -> Embeddings:
        return self.sketch(images)

    def adaptation_state_dict(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.named_parameters()
            if value.requires_grad
        }
