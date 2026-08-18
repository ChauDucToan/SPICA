from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import open_clip
import torch
from torch import Tensor, nn

from ..data.datasets import ImageTransform

TextTokenizer = Callable[[Sequence[str]], Tensor]


class FrozenClipEncoder(nn.Module):
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
        return self.encode_image(images)

    def encode_image(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "CLIP images must have shape [batch, channels, height, width], "
                f"got {tuple(images.shape)}"
            )
        if images.shape[1] != 3:
            raise ValueError(f"CLIP expects three RGB channels, got {images.shape[1]}")
        if not images.is_floating_point():
            raise TypeError(f"CLIP expects floating-point images, got {images.dtype}")

        embeddings = self.model.encode_image(images, normalize=True)
        self._validate_embeddings(embeddings, expected_batch_size=images.shape[0])
        return embeddings

    def encode_text(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(
                "CLIP text tokens must have shape [batch, context_length], "
                f"got {tuple(tokens.shape)}"
            )
        if tokens.is_floating_point():
            raise TypeError(f"CLIP text tokens must be integer IDs, got {tokens.dtype}")

        embeddings = self.model.encode_text(tokens, normalize=True)
        self._validate_embeddings(embeddings, expected_batch_size=tokens.shape[0])
        return embeddings

    @staticmethod
    def _validate_embeddings(
        embeddings: Tensor,
        *,
        expected_batch_size: int,
    ) -> None:
        if embeddings.ndim != 2:
            raise RuntimeError(
                "OpenCLIP encoder must return [batch, embedding_dim], "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[0] != expected_batch_size:
            raise RuntimeError(
                "OpenCLIP changed the batch size from "
                f"{expected_batch_size} to {embeddings.shape[0]}"
            )


@dataclass(frozen=True, slots=True)
class FrozenClipBundle:
    encoder: FrozenClipEncoder
    transform: ImageTransform
    tokenizer: TextTokenizer
    model_name: str
    pretrained: str | None


def load_frozen_clip(
    *,
    # OpenAI ViT-B/32 was pretrained with QuickGELU, not standard GELU.
    model_name: str = "ViT-B-32-quickgelu",
    pretrained: str | None = "openai",
    device: str | torch.device = "cpu",
    cache_dir: Path | None = None,
) -> FrozenClipBundle:
    model, _, eval_transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        precision="fp32",
        device=device,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    tokenizer = open_clip.get_tokenizer(
        model_name,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    return FrozenClipBundle(
        encoder=FrozenClipEncoder(model),
        transform=eval_transform,
        tokenizer=tokenizer,
        model_name=model_name,
        pretrained=pretrained,
    )
