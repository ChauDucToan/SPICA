from __future__ import annotations

import random

import pytest
import torch

from spica.data.masking import apply_ink_mask, mask_seed
from spica.data.transforms import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD


def _normalized(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD).view(3, 1, 1)
    return (rgb - mean) / std


def _fixture() -> torch.Tensor:
    rgb = torch.ones(3, 15, 17)
    rgb[:, 3:12, 4:13] = 0.0
    return _normalized(rgb)


def test_replay_is_nested_and_does_not_mutate_input() -> None:
    image = _fixture()
    original = image.clone()
    outputs = [apply_ink_mask(image, fraction=f, seed=19) for f in (0.25, 0.5, 0.75)]

    assert torch.equal(image, original)
    boxes = [meta["bbox"] for _, meta in outputs]
    assert boxes[0] is not None and boxes[1] is not None and boxes[2] is not None
    assert boxes[0][0] >= boxes[1][0] >= boxes[2][0]
    assert boxes[0][1] >= boxes[1][1] >= boxes[2][1]
    assert boxes[0][2] <= boxes[1][2] <= boxes[2][2]
    assert boxes[0][3] <= boxes[1][3] <= boxes[2][3]
    assert all(torch.equal(out, apply_ink_mask(image, fraction=f, seed=19)[0]) for out, f in zip((x[0] for x in outputs), (0.25, 0.5, 0.75)))
    erased = [out != image for out, _ in outputs]
    assert torch.all(~erased[0] | erased[1])
    assert torch.all(~erased[1] | erased[2])
    assert outputs[0][1]["realized_fraction"] <= outputs[1][1]["realized_fraction"] <= outputs[2][1]["realized_fraction"]


def test_mask_seed_and_mask_do_not_touch_global_rng() -> None:
    image = _fixture()
    random.seed(101)
    torch.manual_seed(101)
    expected_random = random.random()
    expected_torch = torch.rand(1)
    random.seed(101)
    torch.manual_seed(101)

    first = mask_seed(4242, "class/sketch.png", 7)
    second = mask_seed(4242, "class/sketch.png", 7)
    apply_ink_mask(image, fraction=0.5, seed=first)

    assert first == second
    assert random.random() == expected_random
    assert torch.equal(torch.rand(1), expected_torch)


def test_blank_zero_and_tiny_inputs_are_safe() -> None:
    blank = _normalized(torch.ones(3, 1, 1))
    masked, metadata = apply_ink_mask(blank, fraction=0.75, seed=1)
    assert torch.equal(masked, blank)
    assert metadata["status"] == "blank_input"
    assert metadata["bbox"] is None

    one_ink = _normalized(torch.zeros(3, 1, 1))
    masked, metadata = apply_ink_mask(one_ink, fraction=0.75, seed=1)
    assert torch.equal(masked, one_ink)
    assert metadata["status"] == "target_unreachable"
    assert metadata["realized_fraction"] == 0.0

    untouched, metadata = apply_ink_mask(one_ink, fraction=0.0, seed=1)
    assert torch.equal(untouched, one_ink)
    assert metadata["status"] == "zero_fraction"


def test_metadata_hashes_and_validation() -> None:
    image = _fixture()
    masked, metadata = apply_ink_mask(image, fraction=0.5, seed=5)
    assert set(metadata) == {
        "policy_version", "seed", "requested_fraction", "realized_fraction",
        "ink_pixels_before", "ink_pixels_erased", "ink_pixels_after", "bbox",
        "status", "input_sha256", "output_sha256",
    }
    assert metadata["output_sha256"] != metadata["input_sha256"]
    assert metadata["ink_pixels_before"] == metadata["ink_pixels_erased"] + metadata["ink_pixels_after"]
    assert metadata["realized_fraction"] == pytest.approx(
        metadata["ink_pixels_erased"] / metadata["ink_pixels_before"]
    )
    assert torch.isfinite(masked).all()

    with pytest.raises(ValueError):
        apply_ink_mask(image, fraction=1.0, seed=1)
    with pytest.raises(ValueError):
        apply_ink_mask(image, fraction=0.5, seed=1, std=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        apply_ink_mask(torch.ones(3, 0, 2), fraction=0.5, seed=1)
