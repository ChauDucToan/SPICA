"""Validation helpers for prompt-only checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .frozen_prompt import FrozenPromptModel


def visual_backbone_identity(model: FrozenPromptModel) -> dict[str, Any]:
    """Return structural identity that can be checked without serializing CLIP."""
    visual = model.visual
    positional = getattr(visual, "positional_embedding")
    projection = getattr(visual, "proj")
    blocks = getattr(getattr(visual, "transformer"), "resblocks")
    return {
        "visual_class": f"{type(visual).__module__}.{type(visual).__qualname__}",
        "visual_width": int(positional.shape[1]),
        "positional_embedding_shape": list(positional.shape),
        "projection_shape": list(projection.shape),
        "transformer_block_count": len(blocks),
        "pool_type": getattr(visual, "pool_type", None),
        "attn_pool": getattr(visual, "attn_pool", None) is not None,
        "output_tokens": bool(getattr(visual, "output_tokens", False)),
    }


def load_trainable_state(
    model: nn.Module,
    state: Mapping[str, Any],
    *,
    required_keys: set[str] = frozenset(),
) -> dict[str, Any]:
    """Load a trainable-only state after checking every key and shape."""
    if not isinstance(state, Mapping):
        raise ValueError("model state must be a mapping")
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state)
    unexpected = sorted(checkpoint_keys - model_keys)
    if unexpected:
        raise ValueError(f"checkpoint has unexpected keys: {unexpected}")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    required = trainable | set(required_keys)
    missing = model_keys - checkpoint_keys
    allowed_frozen = {
        name for name in missing if name not in required
    }
    disallowed_missing = sorted(missing - allowed_frozen)
    if disallowed_missing:
        raise ValueError(f"checkpoint is missing required keys: {disallowed_missing}")
    shape_errors: list[str] = []
    for name in sorted(checkpoint_keys):
        value = state[name]
        if not isinstance(value, Tensor):
            shape_errors.append(f"{name}: value is not a tensor")
        elif tuple(value.shape) != tuple(model_state[name].shape):
            shape_errors.append(
                f"{name}: checkpoint {tuple(value.shape)} != model {tuple(model_state[name].shape)}"
            )
    if shape_errors:
        raise ValueError("checkpoint tensor mismatch: " + "; ".join(shape_errors))
    result = model.load_state_dict(dict(state), strict=False)
    if set(result.unexpected_keys) or set(result.missing_keys) != allowed_frozen:
        raise ValueError(
            "checkpoint load mismatch: "
            f"missing={sorted(result.missing_keys)}, unexpected={sorted(result.unexpected_keys)}"
        )
    return {
        "missing_frozen_keys": sorted(allowed_frozen),
        "unexpected_keys": [],
        "required_keys": sorted(required),
    }


def load_prompt_checkpoint(
    model: FrozenPromptModel,
    payload: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a trainable-only prompt checkpoint with fail-closed validation.

    Missing frozen visual parameters are allowed because historical checkpoints
    intentionally store only trainable state. Missing prompts, trainable visual
    parameters, unexpected keys, shape mismatches, and incompatible config are
    errors. The returned metadata records exactly what was verified.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if payload.get("model_type") not in {"frozen_prompt_alignment", "frozen_prompt_v2"}:
        raise ValueError("checkpoint is not a supported frozen-prompt checkpoint")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("checkpoint resolved_config is missing")
    for key in ("model_name", "pretrained", "visual_prompt_length"):
        if key not in resolved:
            raise ValueError(f"checkpoint resolved_config missing {key}")

    config_keys = (
        "model_name",
        "pretrained",
        "visual_prompt_length",
        "train_visual_layernorm",
        "train_sketch_prompt",
        "train_photo_prompt",
    )
    for key in config_keys:
        if expected_config is not None and key in expected_config:
            if resolved.get(key) != expected_config[key]:
                raise ValueError(
                    f"checkpoint config mismatch for {key}: "
                    f"{resolved.get(key)!r} != {expected_config[key]!r}"
                )
    if int(resolved["visual_prompt_length"]) != model.prompt_length:
        raise ValueError("checkpoint prompt length does not match model")
    for key, actual in (
        ("train_visual_layernorm", model.train_visual_layernorm),
        ("train_sketch_prompt", model.train_sketch_prompt),
        ("train_photo_prompt", model.train_photo_prompt),
    ):
        if key in resolved and bool(resolved[key]) != actual:
            raise ValueError(f"checkpoint {key} does not match model")

    prompt_keys = {"sketch_prompt", "photo_prompt"}
    load_info = load_trainable_state(
        model,
        state,
        required_keys=prompt_keys,
    )
    non_backbone_missing = [
        name
        for name in load_info["missing_frozen_keys"]
        if not name.startswith("visual.")
    ]
    if non_backbone_missing:
        raise ValueError(
            "checkpoint omits non-backbone state: "
            f"{sorted(non_backbone_missing)}"
        )
    for name in sorted(prompt_keys):
        if not torch.equal(model.state_dict()[name].detach().cpu(), state[name].detach().cpu()):
            raise ValueError(f"checkpoint prompt {name} did not load correctly")

    actual_identity = visual_backbone_identity(model)
    embedded_identity = payload.get("backbone_identity")
    if embedded_identity is not None and embedded_identity != actual_identity:
        raise ValueError("checkpoint backbone identity does not match model")

    return {
        "loaded_prompt_keys": sorted(prompt_keys),
        "missing_frozen_backbone_keys": load_info["missing_frozen_keys"],
        "unexpected_keys": [],
        "backbone_identity": actual_identity,
        "embedded_backbone_identity_verified": embedded_identity is not None,
        "backbone_identity_status": "VERIFIED"
        if embedded_identity is not None
        else "UNVERIFIED_MISSING",
        "resolved_model_name": str(resolved["model_name"]),
        "resolved_pretrained": resolved.get("pretrained"),
    }
