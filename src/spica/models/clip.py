from dataclasses import dataclass
from pathlib import Path
from typing import Self

import open_clip
import torch
from torch import Tensor, nn

from ..data.datasets import ImageTransform


class FrozenClipImageEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.train(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True) -> Self:
        """Keep the wrapped CLIP model in evaluation mode."""
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "CLIP images must have shape [batch, channels, height, width], "
                f"got {tuple(images.shape)}"
            )
        if images.shape[1] != 3:
            raise ValueError(f"CLIP expects three RGB channels, got {images.shape[1]}")
        if not images.is_floating_point():
            raise TypeError(f"CLIP expects floating-point images, got {images.dtype}")
        if not isinstance(self.model, open_clip.CLIP):
            raise TypeError("CLIP model expects encode_image() method")

        embeddings = self.model.encode_image(images, normalize=True)

        if embeddings.ndim != 2:
            raise RuntimeError(
                "OpenCLIP encode_image must return [batch, embedding_dim], "
                f"got {tuple(embeddings.shape)}"
            )

        return embeddings


@dataclass(frozen=True, slots=True)
class FrozenClipImageBundle:
    encoder: FrozenClipImageEncoder
    transform: ImageTransform
    model_name: str
    pretrained: str | None


def load_frozen_clip_image_encoder(
    *,
    model_name: str = "ViT-B-32",
    pretrained: str | None = "openai",
    device: str | torch.device = "cpu",
    cache_dir: Path | None = None,
) -> FrozenClipImageBundle:
    model, _, eval_transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        precision="fp32",
        device=device,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    return FrozenClipImageBundle(
        encoder=FrozenClipImageEncoder(model),
        transform=eval_transform,
        model_name=model_name,
        pretrained=pretrained,
    )
