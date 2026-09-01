"""Small training utilities shared by frozen-CLIP retrieval trainers."""

import torch
from torch import Tensor

from .models.clip import FrozenClipEncoder


def encode_multi_positive_images(
    encoder: FrozenClipEncoder,
    sketch_images: Tensor,
    positive_images: Tensor,
    negative_images: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode one multi-positive batch with the frozen CLIP encoder.

    The returned tensors are ``[B, D]``, ``[B, M, D]`` and ``[B, D]``.  Keeping
    this helper outside of any particular objective makes the deterministic
    controls independent of the vMF implementation.
    """
    if sketch_images.ndim != 4:
        raise ValueError(
            f"sketch_images must have shape [B, C, H, W], got {sketch_images.shape}"
        )
    if positive_images.ndim != 5:
        raise ValueError(
            "positive_images must have shape [B, M, C, H, W], got "
            f"{positive_images.shape}"
        )
    if negative_images.shape != sketch_images.shape:
        raise ValueError("negative_images and sketch_images shapes must match")
    if positive_images.shape[0] != sketch_images.shape[0]:
        raise ValueError("Positive and sketch batch sizes must match")
    if positive_images.shape[2:] != sketch_images.shape[1:]:
        raise ValueError("Positive and sketch image dimensions must match")

    batch_size, num_positives = positive_images.shape[:2]
    flattened_positives = positive_images.flatten(0, 1)
    images = torch.cat(
        (sketch_images, flattened_positives, negative_images),
        dim=0,
    )
    with torch.no_grad():
        embeddings = encoder(images)
    sketch_embeddings, positive_embeddings, negative_embeddings = embeddings.split(
        (batch_size, batch_size * num_positives, batch_size),
        dim=0,
    )
    return (
        sketch_embeddings,
        positive_embeddings.reshape(batch_size, num_positives, -1),
        negative_embeddings,
    )
