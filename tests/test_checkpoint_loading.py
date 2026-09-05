"""Fail-closed validation for trainable-only prompt checkpoints."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spica.models.checkpoint import load_prompt_checkpoint


class _FakeTransformer(nn.Module):
    batch_first = True

    def __init__(self) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([nn.Identity()])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


class _FakeVisual(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(2, width))
        self.proj = nn.Parameter(torch.eye(width))
        self.ln_pre = nn.Identity()
        self.transformer = _FakeTransformer()
        self.pool_type = "tok"
        self.attn_pool = None
        self.output_tokens = False

    def _embeds(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], 2, self.positional_embedding.shape[1])

    def _pool(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tokens[:, 0], tokens


def _make_model():
    from spica.models.frozen_prompt import FrozenPromptModel

    return FrozenPromptModel(
        _FakeVisual(),
        prompt_length=3,
        train_visual_layernorm=False,
        train_sketch_prompt=True,
        train_photo_prompt=True,
    )


def _payload(model: nn.Module) -> dict:
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name in {"sketch_prompt", "photo_prompt"}
    }
    return {
        "model_type": "frozen_prompt_alignment",
        "resolved_config": {
            "model_name": "fake",
            "pretrained": "fake",
            "visual_prompt_length": 3,
            "train_visual_layernorm": False,
            "train_sketch_prompt": True,
            "train_photo_prompt": True,
        },
        "model_state_dict": state,
    }


def test_trainable_only_load_succeeds_and_allows_frozen_missing() -> None:
    model = _make_model()
    info = load_prompt_checkpoint(
        model,
        _payload(model),
        expected_config={"model_name": "fake", "pretrained": "fake"},
    )
    assert info["loaded_prompt_keys"] == ["photo_prompt", "sketch_prompt"]
    assert info["missing_frozen_backbone_keys"]
    assert info["unexpected_keys"] == []
    assert info["backbone_identity_status"] == "UNVERIFIED_MISSING"


def test_missing_prompt_key_raises() -> None:
    model = _make_model()
    payload = _payload(model)
    del payload["model_state_dict"]["sketch_prompt"]
    with pytest.raises(ValueError, match="required.*prompt|missing.*prompt"):
        load_prompt_checkpoint(model, payload)


def test_shape_mismatch_raises() -> None:
    model = _make_model()
    payload = _payload(model)
    payload["model_state_dict"]["sketch_prompt"] = torch.randn(99, 8)
    with pytest.raises(ValueError, match="tensor mismatch"):
        load_prompt_checkpoint(model, payload)


def test_unexpected_key_raises() -> None:
    model = _make_model()
    payload = _payload(model)
    payload["model_state_dict"]["not_a_model_key"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="unexpected keys"):
        load_prompt_checkpoint(model, payload)


def test_config_mismatch_raises() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="config mismatch"):
        load_prompt_checkpoint(
            model, _payload(model), expected_config={"model_name": "wrong"}
        )


if __name__ == "__main__":
    test_trainable_only_load_succeeds_and_allows_frozen_missing()
    print("PASS: trainable_only_load")
    test_missing_prompt_key_raises()
    print("PASS: missing_prompt")
    test_shape_mismatch_raises()
    print("PASS: shape_mismatch")
    test_unexpected_key_raises()
    print("PASS: unexpected_key")
    test_config_mismatch_raises()
    print("PASS: config_mismatch")
    print("ALL CHECKPOINT LOADING TESTS PASSED")
