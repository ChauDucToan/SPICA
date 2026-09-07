"""Deterministic CPU-side ink-region masking for normalized CLIP images."""
from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import Sequence

import torch
from torch import Tensor

from .transforms import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD

MASK_POLICY_VERSION = "ink_centered_square_v1"


def mask_seed(base_seed: int, sample_key: str, view: int = 0) -> int:
    """Derive a stable per-sample seed without touching process RNG state."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError("base_seed must be an integer")
    if not isinstance(sample_key, str):
        raise TypeError("sample_key must be a string")
    if not isinstance(view, int) or isinstance(view, bool):
        raise TypeError("view must be an integer")
    payload = f"{base_seed}\0{sample_key}\0{view}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _channels(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result  # type: ignore[return-value]


def _tensor_sha256(image: Tensor) -> str:
    """Hash canonical float32 CPU bytes with the tensor shape included."""
    canonical = image.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(b"spica-ink-mask-tensor-v1\0")
    digest.update(struct.pack(">I", canonical.ndim))
    for size in canonical.shape:
        digest.update(struct.pack(">Q", int(size)))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_image(image: Tensor) -> tuple[int, int]:
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.device.type != "cpu":
        raise ValueError("image must be a CPU tensor")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must have shape [3, H, W], got {tuple(image.shape)}")
    height, width = (int(image.shape[1]), int(image.shape[2]))
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be nonzero")
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image must contain only finite values")
    return height, width


def _box(row: int, col: int, radius: int, height: int, width: int) -> tuple[int, int, int, int]:
    return (
        max(0, col - radius),
        max(0, row - radius),
        min(width, col + radius + 1),
        min(height, row + radius + 1),
    )


def apply_ink_mask(
    image: Tensor,
    *,
    fraction: float,
    seed: int,
    mean: Sequence[float] = CLIP_IMAGE_MEAN,
    std: Sequence[float] = CLIP_IMAGE_STD,
    ink_threshold: float = 0.9,
) -> tuple[Tensor, dict[str, object]]:
    """Erase one deterministic, centered square from a normalized CPU image.

    ``fraction`` is a target fraction of thresholded ink pixels, not square
    area. The selected center is sampled once from all ink pixels; increasing
    severity therefore produces nested squares for a fixed seed and image.
    """
    height, width = _validate_image(image)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("fraction must be finite and in [0, 1)")
    mean_values = _channels(mean, "mean")
    std_values = _channels(std, "std")
    if any(value <= 0 for value in std_values):
        raise ValueError("std values must be positive")
    ink_threshold = float(ink_threshold)
    if not math.isfinite(ink_threshold):
        raise ValueError("ink_threshold must be finite")

    input_sha256 = _tensor_sha256(image)
    metadata: dict[str, object] = {
        "policy_version": MASK_POLICY_VERSION,
        "seed": seed,
        "requested_fraction": fraction,
        "realized_fraction": 0.0,
        "ink_pixels_before": 0,
        "ink_pixels_erased": 0,
        "ink_pixels_after": 0,
        "bbox": None,
        "status": "zero_fraction" if fraction == 0 else "ok",
        "input_sha256": input_sha256,
        "output_sha256": input_sha256,
    }
    output = image.clone()
    mean_tensor = torch.tensor(mean_values, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std_values, dtype=torch.float32).view(3, 1, 1)
    rgb = image.detach().to(dtype=torch.float32) * std_tensor + mean_tensor
    ink = rgb.mean(dim=0) < ink_threshold
    ink_pixels = int(ink.sum().item())
    metadata["ink_pixels_before"] = ink_pixels
    metadata["ink_pixels_after"] = ink_pixels
    if fraction == 0:
        return output, metadata
    if ink_pixels == 0:
        metadata["status"] = "blank_input"
        return output, metadata

    target = max(1, math.ceil(fraction * ink_pixels))
    coordinates = ink.nonzero(as_tuple=False).tolist()
    row, col = coordinates[random.Random(seed).randrange(len(coordinates))]

    selected: tuple[int, int, int, int] | None = None
    best_nonblank: tuple[int, int, int, int] | None = None
    erased = 0
    for radius in range(max(height, width)):
        candidate = _box(row, col, radius, height, width)
        left, top, right, bottom = candidate
        candidate_erased = int(ink[top:bottom, left:right].sum().item())
        if candidate_erased < ink_pixels:
            best_nonblank = candidate
            if candidate_erased >= target:
                selected = candidate
                erased = candidate_erased
                break

    if selected is None:
        selected = best_nonblank
        metadata["status"] = "target_unreachable"
        if selected is None:
            metadata["output_sha256"] = input_sha256
            return output, metadata
        left, top, right, bottom = selected
        erased = int(ink[top:bottom, left:right].sum().item())
    else:
        left, top, right, bottom = selected

    white = (1.0 - torch.tensor(mean_values, dtype=image.dtype)) / torch.tensor(
        std_values, dtype=image.dtype
    )
    output[:, top:bottom, left:right] = white.view(3, 1, 1)
    after = ink_pixels - erased
    metadata.update(
        {
            "realized_fraction": erased / ink_pixels,
            "ink_pixels_erased": erased,
            "ink_pixels_after": after,
            "bbox": [left, top, right, bottom],
            "output_sha256": _tensor_sha256(output),
        }
    )
    return output, metadata
