"""Numerical tests for the stable spherical_log_map implementation."""

from __future__ import annotations

import torch
from spica.models.alignment import spherical_log_map, _normalize


def _random_unit(n: int, dim: int, dtype=torch.float64) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(n, dim, dtype=dtype), dim=-1)


def test_log_at_anchor_is_zero() -> None:
    """Log_t(t) = 0 exactly."""
    t = _random_unit(5, 512)
    result = spherical_log_map(t, t)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_tangent_orthogonality() -> None:
    """Tangent vectors must be orthogonal to anchor."""
    t = _random_unit(10, 512)
    z = _random_unit(10, 512)
    log_z = spherical_log_map(t, z)
    dot = (log_z * _normalize(t)).sum(dim=-1).abs()
    assert (dot < 1e-4).all(), f"max orthogonality violation: {dot.max().item()}"


def test_tangent_norm_matches_geodesic_angle() -> None:
    """||Log_t(z)|| should equal the geodesic angle arccos(dot(t,z))."""
    t = _random_unit(20, 512)
    z = _random_unit(20, 512)
    cosine = (t * z).sum(dim=-1)
    mask = cosine.abs() < 0.99
    t, z = t[mask], z[mask]
    assert t.shape[0] > 5

    log_z = spherical_log_map(t, z)
    tangent_norm = log_z.norm(dim=-1)
    expected_angle = torch.acos(cosine[mask].clamp(-1, 1))
    assert torch.allclose(tangent_norm.double(), expected_angle.double(), atol=1e-4)


def test_small_angle_gradient_finite() -> None:
    """Gradient must be finite and non-zero near the anchor."""
    t = _random_unit(1, 64, dtype=torch.float64)
    z_near = t + 1e-4 * _random_unit(1, 64, dtype=torch.float64)
    z_near = torch.nn.functional.normalize(z_near, dim=-1)
    z_near.requires_grad_(True)

    log_z = spherical_log_map(t, z_near)
    loss = log_z.square().sum()
    loss.backward()
    assert z_near.grad is not None
    assert torch.isfinite(z_near.grad).all()
    assert z_near.grad.norm() > 0


def test_gradcheck_regular_points() -> None:
    """torch.autograd.gradcheck at regular (non-singular) points."""
    t = _random_unit(3, 16, dtype=torch.float64)
    z = _random_unit(3, 16, dtype=torch.float64)
    cosine = (t * z).sum(dim=-1)
    assert (cosine.abs() < 0.95).all()
    z.requires_grad_(True)

    def fn(z_in):
        return spherical_log_map(t, z_in)

    assert torch.autograd.gradcheck(fn, z, eps=1e-6, atol=1e-4, rtol=1e-3)


def test_no_nan_or_inf() -> None:
    """No NaN/Inf for any input including edge cases."""
    t = _random_unit(5, 64)
    z_regular = _random_unit(5, 64)
    z_anchor = t.clone()
    z_antipode = -t.clone()
    z_near_anchor = torch.nn.functional.normalize(t + 1e-7 * _random_unit(5, 64), dim=-1)

    for z in [z_regular, z_anchor, z_antipode, z_near_anchor]:
        result = spherical_log_map(t, z)
        assert torch.isfinite(result).all(), "non-finite output"


def test_float32_output_dtype_preserved() -> None:
    """Output dtype matches input dtype."""
    t = _random_unit(3, 32, dtype=torch.float32)
    z = _random_unit(3, 32, dtype=torch.float32)
    result = spherical_log_map(t, z)
    assert result.dtype == torch.float32


if __name__ == "__main__":
    test_log_at_anchor_is_zero()
    print("PASS: log_at_anchor")
    test_tangent_orthogonality()
    print("PASS: tangent_orthogonality")
    test_tangent_norm_matches_geodesic_angle()
    print("PASS: tangent_norm_matches_angle")
    test_small_angle_gradient_finite()
    print("PASS: small_angle_gradient")
    test_gradcheck_regular_points()
    print("PASS: gradcheck")
    test_no_nan_or_inf()
    print("PASS: no_nan_inf")
    test_float32_output_dtype_preserved()
    print("PASS: dtype_preserved")
    print("\nALL SPHERICAL LOG MAP TESTS PASSED")
