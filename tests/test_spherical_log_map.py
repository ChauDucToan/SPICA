"""Numerical tests for the corrected spherical_log_map and tangent_moments."""

from __future__ import annotations

import pytest
import torch
from spica.models.alignment import spherical_log_map, tangent_moments, _normalize


def _random_unit(n: int, dim: int, dtype=torch.float64) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(n, dim, dtype=dtype), dim=-1)


# --- Log map tests ---

def test_log_at_anchor_is_zero() -> None:
    t = _random_unit(5, 512)
    result, valid = spherical_log_map(t, t)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)
    assert valid.all()


def test_near_anchor_nonzero_output() -> None:
    """t=[1,0,0], z=normalize([1,1e-7,0]): output near [0,1e-7,0], NOT zero."""
    t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    z_raw = torch.tensor([[1.0, 1e-7, 0.0]], dtype=torch.float64)
    z = torch.nn.functional.normalize(z_raw, dim=-1)
    result, valid = spherical_log_map(t, z)
    assert valid.all()
    expected = torch.tensor([[0.0, 1e-7, 0.0]], dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-9, rtol=1e-4)


def test_tangent_orthogonality() -> None:
    t = _random_unit(10, 512)
    z = _random_unit(10, 512)
    log_z, valid = spherical_log_map(t, z)
    dot = (log_z[valid] * _normalize(t)[valid]).sum(dim=-1).abs()
    assert (dot < 1e-4).all(), f"max orthogonality violation: {dot.max().item()}"


def test_tangent_norm_matches_geodesic_angle() -> None:
    t = _random_unit(20, 512)
    z = _random_unit(20, 512)
    cosine = (t * z).sum(dim=-1)
    mask = cosine.abs() < 0.99
    t, z = t[mask], z[mask]
    assert t.shape[0] > 5
    log_z, valid = spherical_log_map(t, z)
    tangent_norm = log_z[valid].norm(dim=-1)
    expected_angle = torch.acos(cosine[mask][valid].clamp(-1, 1))
    assert torch.allclose(tangent_norm.double(), expected_angle.double(), atol=1e-4)


def test_small_angle_gradient_finite() -> None:
    t = _random_unit(1, 64, dtype=torch.float64)
    z_near = t + 1e-4 * _random_unit(1, 64, dtype=torch.float64)
    z_near = torch.nn.functional.normalize(z_near, dim=-1)
    z_near.requires_grad_(True)
    log_z, valid = spherical_log_map(t, z_near)
    assert valid.all()
    direction = torch.zeros(64, dtype=torch.float64)
    direction[1] = 1.0
    loss = (log_z * direction).sum()
    loss.backward()
    assert z_near.grad is not None
    assert torch.isfinite(z_near.grad).all()


def test_exact_anchor_squared_norm_gradient_is_zero() -> None:
    """Zero gradient of ||Log_t(t)||² at the exact anchor is correct."""
    t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    z = t.clone().requires_grad_(True)
    log_z, valid = spherical_log_map(t, z)
    assert valid.all()
    log_z.square().sum().backward()
    assert z.grad is not None
    assert torch.allclose(z.grad, torch.zeros_like(z.grad), atol=1e-12)


def test_directional_derivative_near_anchor() -> None:
    t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    delta = 1e-5
    z = torch.nn.functional.normalize(
        torch.tensor([[1.0, delta, 0.0]], dtype=torch.float64), dim=-1
    )
    z.requires_grad_(True)
    log_z, valid = spherical_log_map(t, z)
    assert valid.all()
    log_z[0, 1].backward()
    assert z.grad is not None
    assert abs(z.grad[0, 1].item() - 1.0) < 0.1


def test_exact_antipode_strict_raises() -> None:
    t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    z = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="near-antipodal"):
        spherical_log_map(t, z, strict_antipode=True)


def test_exact_antipode_nonstrict_invalid_mask() -> None:
    t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    z = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
    result, valid = spherical_log_map(t, z, strict_antipode=False)
    assert not valid.any()
    assert torch.allclose(result, torch.zeros_like(result))


def test_near_antipode_handled_by_threshold() -> None:
    t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    z = torch.nn.functional.normalize(
        torch.tensor([[-1.0, 1e-6, 0.0]], dtype=torch.float64), dim=-1
    )
    result, valid = spherical_log_map(t, z, antipode_eps=1e-4, strict_antipode=False)
    assert not valid.any()


def test_no_nan_or_inf() -> None:
    t = _random_unit(5, 64)
    z_regular = _random_unit(5, 64)
    z_anchor = t.clone()
    z_near_anchor = torch.nn.functional.normalize(t + 1e-7 * _random_unit(5, 64), dim=-1)
    for z in [z_regular, z_anchor, z_near_anchor]:
        result, valid = spherical_log_map(t, z)
        assert torch.isfinite(result).all()


def test_nonfinite_input_raises() -> None:
    t = _random_unit(1, 8, dtype=torch.float32)
    z = t.clone()
    z[0, 0] = float('inf')
    with pytest.raises(ValueError, match="non-finite"):
        spherical_log_map(t, z)


def test_zero_vector_input_raises() -> None:
    t = torch.zeros(1, 8, dtype=torch.float32)
    z = torch.ones(1, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="non-zero"):
        spherical_log_map(t, z)
    with pytest.raises(ValueError, match="non-zero"):
        spherical_log_map(z, t)


def test_gradcheck_regular_points() -> None:
    t = _random_unit(3, 16, dtype=torch.float64)
    z = _random_unit(3, 16, dtype=torch.float64)
    cosine = (t * z).sum(dim=-1)
    assert (cosine.abs() < 0.95).all()
    z.requires_grad_(True)

    def fn(z_in):
        out, _ = spherical_log_map(t, z_in)
        return out

    assert torch.autograd.gradcheck(fn, z, eps=1e-6, atol=1e-4, rtol=1e-3)


def test_float32_dtype_preserved() -> None:
    t = _random_unit(3, 32, dtype=torch.float32)
    z = _random_unit(3, 32, dtype=torch.float32)
    result, valid = spherical_log_map(t, z)
    assert result.dtype == torch.float32
    assert valid.dtype == torch.bool


# --- Covariance n-1 tests ---

def test_covariance_n_minus_1() -> None:
    data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    mean, cov = tangent_moments(data)
    assert torch.allclose(mean, torch.tensor([2.0 / 3, 2.0 / 3], dtype=torch.float64), atol=1e-10)
    expected = torch.tensor([[1.0 / 3, -1.0 / 6], [-1.0 / 6, 1.0 / 3]], dtype=torch.float64)
    assert torch.allclose(cov, expected, atol=1e-10)


def test_covariance_valid_mask() -> None:
    data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [99.0, 99.0], [1.0, 1.0]], dtype=torch.float64)
    mask = torch.tensor([True, True, False, True])
    mean, cov = tangent_moments(data, mask)
    expected_mean = torch.tensor([2.0 / 3, 2.0 / 3], dtype=torch.float64)
    assert torch.allclose(mean, expected_mean, atol=1e-10)


def test_covariance_singleton_raises() -> None:
    data = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 2"):
        tangent_moments(data)


def test_finite_sample_normalization_differs_from_population() -> None:
    data = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    _, corrected = tangent_moments(data)
    population = data.var(dim=0, unbiased=False).reshape(1, 1)
    assert torch.equal(corrected, torch.tensor([[2.0]], dtype=torch.float64))
    assert torch.equal(population, torch.tensor([[1.0]], dtype=torch.float64))
    assert not torch.equal(corrected, population)


if __name__ == "__main__":
    test_log_at_anchor_is_zero()
    print("PASS: log_at_anchor")
    test_near_anchor_nonzero_output()
    print("PASS: near_anchor_nonzero")
    test_tangent_orthogonality()
    print("PASS: tangent_orthogonality")
    test_tangent_norm_matches_geodesic_angle()
    print("PASS: tangent_norm_matches_angle")
    test_small_angle_gradient_finite()
    print("PASS: small_angle_gradient")
    test_directional_derivative_near_anchor()
    print("PASS: directional_derivative")
    test_exact_antipode_strict_raises()
    print("PASS: antipode_strict_raises")
    test_exact_antipode_nonstrict_invalid_mask()
    print("PASS: antipode_nonstrict_mask")
    test_near_antipode_handled_by_threshold()
    print("PASS: near_antipode_threshold")
    test_no_nan_or_inf()
    print("PASS: no_nan_inf")
    test_nonfinite_input_raises()
    print("PASS: nonfinite_input_raises")
    test_gradcheck_regular_points()
    print("PASS: gradcheck")
    test_float32_dtype_preserved()
    print("PASS: dtype_preserved")
    test_covariance_n_minus_1()
    print("PASS: covariance_n_minus_1")
    test_covariance_valid_mask()
    print("PASS: covariance_valid_mask")
    test_covariance_singleton_raises()
    print("PASS: covariance_singleton_raises")
    print("\nALL TESTS PASSED")
