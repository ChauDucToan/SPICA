"""Random-init sketch encoder: local CNN + four-direction Mamba-2.

Input: RGB pixels in [0, 1], normally B×3×224×224 (not CLIP-normalized).
Output: unit-length B×512 embeddings for a frozen CLIP photo gallery.
This is a shared-weight four-scan design, not the original VMamba SS2D block.
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from mamba_ssm import Mamba2
from torch import Tensor, nn

GridFeatures = Float[Tensor, "batch channels height width"]
SketchImages = Float[Tensor, "batch 3 224 224"]
Embeddings = Float[Tensor, "batch 512"]


def conv_block(inputs: int, outputs: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, outputs),
        nn.SiLU(),
    )


class FourWayMamba(nn.Module):
    """Scan row-major and column-major grids, forward and backward; restore XY."""

    def __init__(self, channels: int = 384) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        # Official Triton SSD scan + PyTorch conv; no causal-conv1d extension.
        self.mamba = Mamba2(
            d_model=channels,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            chunk_size=64,
            use_mem_eff_path=False,
        )

    def forward(self, features: GridFeatures) -> GridFeatures:
        batch, channels, height, width = features.shape
        rows = features.flatten(2).transpose(1, 2)
        columns = features.transpose(2, 3).flatten(2).transpose(1, 2)
        scans = torch.cat((rows, rows.flip(1), columns, columns.flip(1)), dim=0)
        right, left, down, up = self.mamba(self.norm(scans)).chunk(4, dim=0)
        horizontal = (
            (right + left.flip(1))
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        vertical = (
            (down + up.flip(1))
            .transpose(1, 2)
            .reshape(batch, channels, width, height)
            .transpose(2, 3)
        )
        return (horizontal + vertical) * 0.25


class HybridBlock(nn.Module):
    def __init__(self, channels: int = 384) -> None:
        super().__init__()
        self.local = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.global_context = FourWayMamba(channels)
        self.gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def forward(self, features: GridFeatures) -> GridFeatures:
        # Both branches return deltas; add the identity exactly once.
        return (
            features
            + self.local(features)
            + self.gate.sigmoid() * self.global_context(features)
        )


class StudentPatchProjector(nn.Module):
    """Project Student feature grid to 49 768-dim patch tokens matching CLIP ViT."""

    def __init__(self, in_channels: int = 384, out_channels: int = 768) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, features: GridFeatures) -> Tensor:
        return F.normalize(self.proj(features).flatten(2).transpose(1, 2), dim=-1)


class SketchStudent(nn.Module):
    def __init__(self, channels: int = 384) -> None:
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            conv_block(1, 48),  # 224×224: learn edges before downsampling
            conv_block(48, 96, 2),  # 112×112
            conv_block(96, 192, 2),  # 56×56
            conv_block(192, 192),
            conv_block(192, channels, 2),  # 28×28
        )
        self.blocks = nn.Sequential(HybridBlock(channels), HybridBlock(channels))
        self.projection = nn.Linear(channels, 512)
        self.last_features: GridFeatures | None = None

    @property
    def device(self) -> torch.device:
        return self.projection.weight.device

    def forward(self, sketches: SketchImages) -> Embeddings:
        ink = 1.0 - sketches.mean(dim=1, keepdim=True)
        features = self.blocks(self.stem(ink))
        self.last_features = features
        # Max pooling keeps thin ink represented; retain 10% background context.
        weights = 0.1 + 0.9 * F.adaptive_max_pool2d(ink, features.shape[-2:])
        pooled = (features * weights).sum(dim=(-2, -1)) / weights.sum(dim=(-2, -1))
        return F.normalize(self.projection(pooled), dim=-1)
