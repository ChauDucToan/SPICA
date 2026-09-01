import math

import pytest
import torch
import torch.nn.functional as F

from spica.models.retrieval import K1VmfPhotoPredictor, VmfPrediction
from spica.models.vmf import k1_vmf_retrieval_loss, log_vmf_normalizer


def test_log_vmf_normalizer_matches_high_precision_references() -> None:
    concentrations = torch.tensor(
        [0.0, 1.0, 10.0, 100.0, 512.0, 1000.0, 2048.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected_values = torch.tensor(
        [
            0.0,
            -0.0009765606446118611,
            -0.09763770538223744,
            -9.588837707103689,
            -193.32201551407566,
            -540.2589158204465,
            -1421.5969264878145,
        ],
        dtype=torch.float64,
    )
    expected_gradients = torch.tensor(
        [
            0.0,
            -0.0019531175784661718,
            -0.019523834023025135,
            -0.1884047640148357,
            -0.6183325436672318,
            -0.7765309329025389,
            -0.8829696120621198,
        ],
        dtype=torch.float64,
    )

    values = log_vmf_normalizer(
        concentrations,
        dimension=512,
        relative_to_uniform=True,
    )
    gradients = torch.autograd.grad(values.sum(), concentrations)[0]

    assert torch.allclose(values, expected_values, atol=2e-9, rtol=0)
    assert torch.allclose(gradients, expected_gradients, atol=1e-10, rtol=0)


def test_log_vmf_normalizer_small_concentration_series() -> None:
    concentration = torch.tensor(1e-2, dtype=torch.float64, requires_grad=True)
    value = log_vmf_normalizer(
        concentration,
        dimension=512,
        relative_to_uniform=True,
    )
    gradient = torch.autograd.grad(value, concentration)[0]

    assert value.item() == pytest.approx(-((1e-2) ** 2) / (2 * 512), abs=1e-14)
    assert gradient.item() == pytest.approx(-1e-2 / 512, abs=1e-12)


def test_k1_predictor_has_identity_direction_and_constant_initial_kappa() -> None:
    torch.manual_seed(0)
    inputs = F.normalize(torch.randn(8, 512), dim=-1)
    predictor = K1VmfPhotoPredictor(
        embedding_dim=512,
        hidden_dim=64,
        min_concentration=1e-4,
        max_concentration=2048.0,
        initial_concentration=512.0,
    )

    prediction = predictor(inputs)

    assert prediction.mean_direction.shape == (8, 512)
    assert prediction.concentration.shape == (8,)
    assert torch.allclose(prediction.mean_direction, inputs, atol=1e-6, rtol=0)
    assert torch.allclose(
        prediction.mean_direction.norm(dim=-1),
        torch.ones(8),
        atol=1e-6,
        rtol=0,
    )
    assert torch.allclose(
        prediction.concentration,
        torch.full((8,), 512.0),
        atol=1e-4,
        rtol=0,
    )


def test_k1_vmf_loss_is_finite_and_trains_both_heads() -> None:
    torch.manual_seed(1)
    inputs = F.normalize(torch.randn(8, 512), dim=-1)
    positives = F.normalize(inputs + 0.2 * torch.randn_like(inputs), dim=-1)
    negatives = F.normalize(torch.randn_like(inputs), dim=-1)
    predictor = K1VmfPhotoPredictor(hidden_dim=64)

    prediction = predictor(inputs)
    losses = k1_vmf_retrieval_loss(
        prediction,
        positives,
        negatives,
        margin=0.2,
        nll_weight=1 / 512,
        ranking_weight=1.0,
    )
    losses.total.backward()

    direction_output = predictor.direction_predictor.stack[-1]
    concentration_output = predictor.concentration_head[-1]
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.positive_nll)
    assert torch.isfinite(losses.cosine_ranking)
    assert direction_output.weight.grad is not None
    assert concentration_output.weight.grad is not None
    assert direction_output.weight.grad.abs().sum() > 0
    assert concentration_output.weight.grad.abs().sum() > 0


def test_k1_vmf_gallery_ranking_equals_cosine_ranking() -> None:
    torch.manual_seed(2)
    directions = F.normalize(torch.randn(4, 512, dtype=torch.float64), dim=-1)
    gallery = F.normalize(torch.randn(31, 512, dtype=torch.float64), dim=-1)
    concentrations = torch.tensor([1.0, 100.0, 512.0, 2048.0], dtype=torch.float64)

    cosine_scores = directions @ gallery.T
    log_normalizers = log_vmf_normalizer(
        concentrations,
        dimension=512,
        relative_to_uniform=True,
    )
    vmf_scores = log_normalizers[:, None] + concentrations[:, None] * cosine_scores

    assert torch.equal(
        torch.argsort(cosine_scores, dim=1, descending=True, stable=True),
        torch.argsort(vmf_scores, dim=1, descending=True, stable=True),
    )


def test_k1_vmf_loss_rejects_nonpositive_concentration() -> None:
    direction = F.normalize(torch.randn(2, 512), dim=-1)
    prediction = VmfPrediction(
        mean_direction=direction,
        concentration=torch.tensor([1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="strictly positive"):
        k1_vmf_retrieval_loss(
            prediction,
            direction,
            -direction,
        )


def test_k1_predictor_validates_concentration_bounds() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        K1VmfPhotoPredictor(
            min_concentration=0.0,
            max_concentration=10.0,
            initial_concentration=10.0,
        )

    with pytest.raises(ValueError, match="greater than"):
        K1VmfPhotoPredictor(
            min_concentration=10.0,
            max_concentration=10.0,
            initial_concentration=10.0,
        )

    assert math.isfinite(K1VmfPhotoPredictor().initial_concentration)
