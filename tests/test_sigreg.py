"""CPU gates for the local pinned-minimal SIGReg implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from spica.models.sigreg import SIGReg

ROOT = Path(__file__).parents[1]


def reference_minimal(values: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """The pinned MINIMAL.md formula, kept deliberately explicit for parity."""
    t = torch.linspace(0.0, 3.0, 17, dtype=torch.float32)
    dt = 3.0 / 16.0
    weights = torch.full((17,), 2.0 * dt)
    weights[[0, -1]] = dt
    weights *= torch.exp(-t.square() / 2.0)
    phi = torch.exp(-t.square() / 2.0)
    projected = values.float() @ directions.float()
    x_t = projected.unsqueeze(-1) * t
    err = (torch.cos(x_t).mean(-3) - phi).square() + torch.sin(x_t).mean(-3).square()
    return ((err @ weights) * values.shape[-2]).mean()


def test_fixed_directions_match_pinned_minimal_formula_and_views() -> None:
    torch.manual_seed(12)
    values = torch.randn(2, 31, 5)
    directions = torch.randn(5, 23)
    loss = SIGReg(num_projections=23)(values, directions=directions)
    expected = reference_minimal(values, directions / directions.norm(dim=0))
    torch.testing.assert_close(loss, expected, rtol=0, atol=2e-6)

    model_values = values.detach().clone().requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    model_loss = SIGReg(num_projections=23)(
        model_values, directions=directions
    )
    reference_loss = reference_minimal(
        reference_values, directions / directions.norm(dim=0)
    )
    (model_gradient,) = torch.autograd.grad(model_loss, model_values)
    (reference_gradient,) = torch.autograd.grad(reference_loss, reference_values)
    torch.testing.assert_close(model_gradient, reference_gradient, rtol=0, atol=2e-6)


def test_zero_shift_and_scale_are_finite_and_shift_scale_are_penalized() -> None:
    torch.manual_seed(4)
    values = torch.randn(96, 7)
    directions = torch.randn(7, 256)
    regularizer = SIGReg(num_projections=256)
    zero = regularizer(torch.zeros_like(values), directions=directions)
    normal = regularizer(values, directions=directions)
    shifted_scaled = regularizer(values * 1.8 + 2.0, directions=directions)
    assert torch.isfinite(zero) and zero.item() > 0
    assert torch.isfinite(normal) and torch.isfinite(shifted_scaled)
    assert shifted_scaled.item() > normal.item() * 1.2


def test_gradients_are_finite_and_nonzero() -> None:
    torch.manual_seed(8)
    values = torch.randn(64, 11, requires_grad=True)
    directions = torch.randn(11, 32)
    loss = SIGReg(num_projections=32)(values, directions=directions)
    (gradient,) = torch.autograd.grad(loss, values)
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.norm().item() > 0


def test_private_generator_replay_state_restore_and_global_rng_is_untouched() -> None:
    values = torch.randn(32, 4)
    first = SIGReg(num_projections=9, seed=123)
    second = SIGReg(num_projections=9, seed=123)

    torch.manual_seed(91)
    expected_next = torch.rand(5)
    torch.manual_seed(91)
    first_value = first(values)
    actual_next = torch.rand(5)
    torch.testing.assert_close(actual_next, expected_next)

    second_value = second(values)
    torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)

    checkpoint = first.state_dict()
    continued = first(values)
    restored = SIGReg(num_projections=9, seed=999)
    restored.load_state_dict(checkpoint)
    torch.testing.assert_close(restored(values), continued, rtol=0, atol=0)


def test_validation_and_singleton_batch() -> None:
    regularizer = SIGReg(num_projections=3)
    with pytest.raises(ValueError):
        regularizer(torch.randn(4))
    with pytest.raises(ValueError):
        regularizer(torch.empty(0, 2, 3))
    with pytest.raises(ValueError, match="float32"):
        SIGReg().half()(torch.randn(2, 3))
    with pytest.raises(ValueError):
        regularizer(torch.tensor([[float("nan")]]))
    with pytest.raises(TypeError):
        regularizer(torch.ones(2, 1, dtype=torch.int64))
    with pytest.raises(ValueError):
        regularizer(torch.randn(1, 2), directions=torch.zeros(2, 3))
    value = regularizer(torch.tensor([[0.25, -0.5]]), directions=torch.ones(2, 3))
    assert torch.isfinite(value)


def test_tracked_source_receipt_is_immutable_and_pinned() -> None:
    receipt = json.loads(
        (ROOT / "docs/sigreg_source_identity_2026-09-08.json").read_text()
    )
    source = receipt["primary_source"]
    assert source["immutable_commit"] == "c293d291ca87cd4fddee9d3fffe4e914c7272052"
    assert source["sha256"]["MINIMAL.md"] == "5fdbd73825ae2e49e25a63cdc1f0caba17438181a817de78f08c11f58dd4224f"
    for url in source["source_urls"].values():
        assert f'/{source["immutable_commit"]}/' in url


def test_optional_local_source_bundle_matches_tracked_receipt() -> None:
    receipt = json.loads((ROOT / "docs/sigreg_source_identity_2026-09-08.json").read_text())
    source = receipt["primary_source"]
    bundle = ROOT / source["local_receipt_directory"]
    if not bundle.exists():
        pytest.skip("local-only primary source receipt absent; numerical gates still run")
    manifest = json.loads((bundle / "FETCH_MANIFEST.json").read_text())
    assert source["immutable_commit"] == manifest["commit"]
    assert {item["path"]: item["sha256"] for item in manifest["files"]} == source["sha256"]
    for path, digest in source["sha256"].items():
        assert hashlib.sha256((bundle / path).read_bytes()).hexdigest() == digest
