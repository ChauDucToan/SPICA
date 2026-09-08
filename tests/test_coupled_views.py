import random

import pytest
import torch

from docs.designs.coupled_predictive_region_v1.masking_sketch import (
    region_pair as original_region_pair,
)
from spica.coupled_predictive_losses import coupled_region_loss
from spica.data.coupled_views import region_pair
from spica.data.masking import mask_seed
from spica.data.transforms import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD


def _image() -> torch.Tensor:
    mean = torch.tensor(CLIP_IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD).view(3, 1, 1)
    rgb = torch.ones(3, 8, 8)
    rgb[:, 2:6, 2:6] = 0
    return (rgb - mean) / std


def test_region_pair_is_immutable_rng_safe_deterministic_and_matches_original() -> None:
    image = _image()
    before = image.clone()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()

    actual = region_pair(image, "class/sample.png", 3)
    expected = original_region_pair(image, "class/sample.png", 3)

    assert torch.equal(image, before)
    assert torch.equal(actual[0], image)
    assert actual[0].data_ptr() != image.data_ptr()
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    assert random.getstate() == py_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert actual[2]["seed"] == mask_seed(4245, "class/sample.png", view=1)
    assert actual[2]["mask_view"] == 1


def test_region_pair_rejects_noncanonical_keys_and_invalid_updates() -> None:
    for key in ("", "/sample.png", "a/../sample.png", "a//sample.png", "a\\sample.png", "a/"):
        with pytest.raises(ValueError, match="canonical relative POSIX"):
            region_pair(_image(), key, 0)
    with pytest.raises(TypeError):
        region_pair(_image(), "sample.png", True)
    with pytest.raises(TypeError):
        region_pair(_image(), "sample.png", 1.5)
    with pytest.raises(ValueError, match="nonnegative"):
        region_pair(_image(), "sample.png", -1)


def test_blank_input_preserves_helper_status_and_pixels() -> None:
    mean = torch.tensor(CLIP_IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD).view(3, 1, 1)
    blank = ((torch.ones(3, 8, 8) - mean) / std).float()
    clean, masked, metadata = region_pair(blank, "blank.png", 0)

    assert torch.equal(clean, blank)
    assert torch.equal(masked, blank)
    assert metadata["status"] == "blank_input"
    assert metadata["ink_pixels_before"] == 0
    assert metadata["ink_pixels_erased"] == 0


def test_actual_tiny_openclip_loss_uses_two_region_views_without_updates() -> None:
    from test_coupled_predictive_model import _model

    model = _model()
    images = [_image(), torch.roll(_image(), 1, 2)]
    pairs = [region_pair(image, f"class/{i}.png", 0) for i, image in enumerate(images)]
    clean = torch.stack([pair[0] for pair in pairs])
    corrupted = torch.stack([pair[1] for pair in pairs])
    photos = torch.randn(3, 3, 8, 8)
    positive = torch.tensor([0, 1], dtype=torch.long)
    negative = torch.tensor([[1], [0]], dtype=torch.long)
    labels = torch.tensor([2, 7], dtype=torch.long)
    photo_labels = torch.tensor([2, 7, 2], dtype=torch.long)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    teacher = {name: value.detach().clone() for name, value in model.original_clip.state_dict().items()}

    result = coupled_region_loss(
        model,
        clean,
        corrupted,
        photos,
        positive,
        negative,
        labels,
        photo_labels,
        ["photo-a", "photo-b", "photo-c"],
        lambda_sig=0.0,
    )
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert model.student_visual.transformer.resblocks[0].mlp.c_fc.weight.grad is not None
    assert model.predictor.output.weight.grad is not None
    assert model.photo_model.photo_prompt.grad is not None
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value, teacher[name]) for name, value in model.original_clip.state_dict().items())
