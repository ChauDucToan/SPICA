"""Design-only CPU masking/thinning sketch; never used by a trainer."""
from __future__ import annotations

import random

import torch
from torch import Tensor

from spica.data.masking import (
    _tensor_sha256,
    _validate_image,
    apply_ink_mask,
    mask_seed,
)
from spica.data.transforms import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD

_PREVIEW_VERSION = "binary_ink_erosion_preview_v0"


def region_pair(
    normalized_cpu_image: Tensor,
    relative_sample_key: str,
    zero_based_update: int,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Return an untouched clean clone and the historical view-1 masked clone.

    This is an illustration only: view 0 is the clean record, not a generated
    mask.  In particular, this function never calls ``thinning_preview``.
    """
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


def thinning_preview(
    normalized_cpu_image: Tensor,
    iterations: int,
) -> tuple[Tensor, dict[str, object]]:
    """Standalone binary 3x3 ink erosion preview, explicitly not a policy.

    Ink is thresholded after CLIP denormalization.  A binary erosion can drop
    components and is not topology-preserving thinning or a real-stroke model.
    Source antialias values remain unchanged except where deleted ink becomes
    normalized white. Pixels outside the image are treated as background,
    so ink touching the image boundary is eroded there as well.
    """
    height, width = _validate_image(normalized_cpu_image)
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("iterations must be an integer")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    mean = torch.tensor(CLIP_IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    rgb = normalized_cpu_image.detach().to(dtype=torch.float32) * std + mean
    ink = rgb.mean(dim=0) < 0.9
    before = int(ink.sum().item())
    output = normalized_cpu_image.clone()
    input_sha = _tensor_sha256(normalized_cpu_image)
    status = "blank_input" if before == 0 else ("zero_iterations" if iterations == 0 else "ok")
    applied = 0

    if before:
        white = ((1.0 - mean) / std).to(dtype=output.dtype)[:, 0, 0][:, None]
        for _ in range(iterations):
            background = ~ink
            padded = torch.ones((height + 2, width + 2), dtype=torch.bool)
            padded[1:-1, 1:-1] = background
            next_ink = ~padded.unfold(0, 3, 1).unfold(1, 3, 1).any(dim=(-1, -2))
            if not bool(next_ink.any()):
                status = "would_blank_skipped"
                break
            output[:, ink & ~next_ink] = white
            ink = next_ink
            applied += 1

    after = int(ink.sum().item()) if before else 0
    metadata: dict[str, object] = {
        "version": _PREVIEW_VERSION,
        "policy_version": _PREVIEW_VERSION,
        "ink_pixels_before": before,
        "ink_pixels_deleted": before - after,
        "ink_pixels_after": after,
        "realized_fraction": (before - after) / before if before else 0.0,
        "requested_iterations": iterations,
        "iterations_applied": applied,
        "status": status,
        "input_sha256": input_sha,
        "output_sha256": _tensor_sha256(output),
        "enabled_in_region_trial": False,
    }
    return output, metadata


def _demo() -> None:
    """Small CPU-only contract check; no model, gradients, optimizer, or data."""
    mean = torch.tensor(CLIP_IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD).view(3, 1, 1)
    rgb = torch.ones(3, 15, 15)
    rgb[:, 2:7, 2:7] = 0.0  # thick component
    rgb[:, 9, 2:8] = 0.25  # thin component
    rgb[:, 12, 12] = 0.0  # one pixel
    rgb[:, 0, 14] = 0.0  # edge ink
    image = (rgb - mean) / std
    original = image.clone()

    py_state, torch_state = random.getstate(), torch.random.get_rng_state()
    clean, masked, metadata = region_pair(image, "class/sample.png", 3)
    assert torch.equal(image, original) and torch.equal(clean, image)
    assert clean.data_ptr() != image.data_ptr()
    assert random.getstate() == py_state and torch.equal(torch.random.get_rng_state(), torch_state)
    seed = mask_seed(4242 + 3, "class/sample.png", view=1)
    severity = random.Random(seed).choice((0.25, 0.5, 0.75))
    expected, expected_meta = apply_ink_mask(image, fraction=severity, seed=seed, ink_threshold=0.9)
    assert torch.equal(masked, expected) and metadata["severity"] == severity
    for key in ("ink_pixels_before", "ink_pixels_erased", "ink_pixels_after", "bbox", "status", "input_sha256", "output_sha256"):
        assert metadata[key] == expected_meta[key]

    original_preview = thinning_preview
    try:
        globals()["thinning_preview"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("called"))
        region_pair(image, "class/sample.png", 3)
    finally:
        globals()["thinning_preview"] = original_preview

    preview, preview_meta = thinning_preview(image, 1)
    assert torch.equal(image, original)
    assert preview_meta["ink_pixels_after"] <= preview_meta["ink_pixels_before"]
    changed = preview != image
    source_ink = ((image.detach().to(torch.float32) * std + mean).mean(dim=0) < 0.9)
    assert bool((changed.any(dim=0) <= source_ink).all())
    assert bool(changed[:, 0, 14].all())
    white = ((1.0 - mean) / std).to(image.dtype)
    assert torch.equal(preview[:, changed.any(dim=0)], white[:, 0, 0][:, None].expand(-1, int(changed.any(dim=0).sum())))
    zero, zero_meta = thinning_preview(image, 0)
    assert torch.equal(zero, image) and zero_meta["iterations_applied"] == 0

    blank = torch.ones_like(image) * white
    blank_out, blank_meta = thinning_preview(blank, 2)
    assert torch.equal(blank_out, blank) and blank_meta["status"] == "blank_input"
    one = blank.clone()
    one[:, 7, 7] = (0.0 - mean[:, 0, 0]) / std[:, 0, 0]
    one_out, one_meta = thinning_preview(one, 1)
    assert torch.equal(one_out, one) and one_meta["status"] == "would_blank_skipped"
    print("MASKING_SKETCH_SELF_CHECK_PASS")
    print("region_view=1 clean_view_0_generated=False")
    print("thinning_policy=standalone enabled_in_region_trial=False")


if __name__ == "__main__":
    _demo()
