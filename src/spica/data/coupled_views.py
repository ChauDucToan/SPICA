"""Deterministic clean/region-corrupted view pairs for coupled training."""
from __future__ import annotations

import random

from torch import Tensor

from .masking import _tensor_sha256, _validate_image, apply_ink_mask, mask_seed


def region_pair(
    normalized_cpu_image: Tensor,
    relative_sample_key: str,
    zero_based_update: int,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Return an untouched clean clone and the deterministic masked view."""
    _validate_image(normalized_cpu_image)
    if (
        not isinstance(relative_sample_key, str)
        or not relative_sample_key
        or "\x00" in relative_sample_key
        or "\\" in relative_sample_key
        or relative_sample_key.startswith("/")
        or relative_sample_key.endswith("/")
        or "//" in relative_sample_key
        or any(part in ("", ".", "..") for part in relative_sample_key.split("/"))
    ):
        raise ValueError("relative_sample_key must be a canonical relative POSIX path")
    if not isinstance(zero_based_update, int) or isinstance(zero_based_update, bool):
        raise TypeError("zero_based_update must be an integer")
    if zero_based_update < 0:
        raise ValueError("zero_based_update must be nonnegative")

    clean = normalized_cpu_image.clone()
    seed = mask_seed(4242 + zero_based_update, relative_sample_key, view=1)
    severity = random.Random(seed).choice((0.25, 0.5, 0.75))
    masked, mask_metadata = apply_ink_mask(
        clean,
        fraction=severity,
        seed=seed,
        ink_threshold=0.9,
    )
    metadata = dict(mask_metadata)
    metadata.update(
        {
            "relative_sample_key": relative_sample_key,
            "zero_based_update": zero_based_update,
            "step_convention": "zero_based_update",
            "mask_view": 1,
            "severity": severity,
            "clean_record": {
                "view": 0,
                "generated": False,
                "fraction": 0.0,
                "sha256": _tensor_sha256(clean),
            },
        }
    )
    return clean, masked, metadata


__all__ = ["region_pair"]
