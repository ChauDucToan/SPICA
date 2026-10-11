"""MambaSketchStem: Trainable CNN-Mamba stem replacing frozen CLIP ViT patch projection."""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from .sketch_student import HybridBlock, conv_block

GridFeatures = Float[Tensor, "batch channels height width"]
PatchTokens = Float[Tensor, "batch 49 768"]
Embeddings = Float[Tensor, "batch 512"]
SketchImages = Float[Tensor, "batch 3 224 224"]


class MambaSketchStem(nn.Module):
    """Extract stroke features and project to 49 patch tokens matching CLIP ViT."""

    def __init__(self, in_channels: int = 192, clip_dim: int = 768) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.clip_dim = clip_dim
        self.stem = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64, 2),
            conv_block(64, 96, 2),
            conv_block(96, 96),
            conv_block(96, in_channels, 2),
        )
        self.blocks = nn.Sequential(HybridBlock(in_channels), HybridBlock(in_channels))
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(in_channels, clip_dim, 1, bias=False),
            nn.GroupNorm(8, clip_dim),
        )

    def forward(self, sketches: SketchImages) -> PatchTokens:
        ink = 1.0 - sketches.mean(dim=1, keepdim=True)
        feat = self.blocks(self.stem(ink))
        patches = self.proj(feat).flatten(2).transpose(1, 2)
        return patches


class CLIPWithSketchStem(nn.Module):
    """Wrap frozen CLIP visual transformer with trainable MambaSketchStem."""

    def __init__(
        self,
        clip_visual: nn.Module,
        mean: tuple[float, float, float] | list[float] = (0.48145466, 0.4578275, 0.40821073),
        std: tuple[float, float, float] | list[float] = (0.26862954, 0.26130258, 0.27577711),
        stem_channels: int = 192,
    ) -> None:
        super().__init__()
        clip_dim = clip_visual.conv1.out_channels
        self.stem = MambaSketchStem(in_channels=stem_channels, clip_dim=clip_dim)
        self.visual = clip_visual.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def device(self) -> torch.device:
        return next(self.stem.parameters()).device

    def target_patch_tokens(self, sketches: SketchImages) -> PatchTokens:
        """Extract standard patch tokens using frozen CLIP conv1 as anchor."""
        norm_sketches = (sketches - self.mean) / self.std
        with torch.no_grad():
            return (
                self.visual.conv1(norm_sketches)
                .reshape(sketches.shape[0], self.visual.conv1.out_channels, -1)
                .permute(0, 2, 1)
            )

    def forward(
        self, sketches: SketchImages, return_patches: bool = False
    ) -> Embeddings | tuple[Embeddings, PatchTokens]:
        patches = self.stem(sketches)
        batch = patches.shape[0]

        cls_token = (
            self.visual.class_embedding.to(patches.dtype)
            .view(1, 1, -1)
            .expand(batch, -1, -1)
        )
        pos_embed = self.visual.positional_embedding.to(patches.dtype)

        tokens = torch.cat((cls_token, patches), dim=1) + pos_embed
        tokens = self.visual.ln_pre(tokens)
        tokens = self.visual.transformer(tokens)
        pooled = self.visual.ln_post(tokens[:, 0])

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        query = F.normalize(pooled, dim=-1)
        if return_patches:
            return query, patches
        return query
