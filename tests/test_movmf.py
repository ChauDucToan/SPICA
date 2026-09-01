import pytest
import torch
import torch.nn.functional as F

from spica.models.retrieval import (
    MoVmfPhotoPredictor,
    MoVmfPrediction,
    VmfPrediction,
)
from spica.models.vmf import (
    k1_vmf_retrieval_loss,
    mo_vmf_gallery_scores,
    mo_vmf_retrieval_loss,
)


def test_movmf_predictor_initialization_and_shapes() -> None:
    torch.manual_seed(10)
    inputs = F.normalize(torch.randn(5, 512), dim=-1)
    predictor = MoVmfPhotoPredictor(
        embedding_dim=512,
        hidden_dim=64,
        num_components=4,
        min_concentration=1e-4,
        max_concentration=2048.0,
        initial_concentration=512.0,
        component_init_std=1e-4,
    )

    prediction = predictor(inputs)
    probabilities = prediction.mixture_logits.softmax(dim=-1)

    assert prediction.mean_directions.shape == (5, 4, 512)
    assert prediction.concentrations.shape == (5, 4)
    assert prediction.mixture_logits.shape == (5, 4)
    assert torch.allclose(
        prediction.mean_directions.norm(dim=-1),
        torch.ones(5, 4),
        atol=1e-6,
        rtol=0,
    )
    assert torch.allclose(
        prediction.concentrations,
        torch.full((5, 4), 512.0),
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        probabilities,
        torch.full((5, 4), 0.25),
        atol=1e-7,
        rtol=0,
    )
    assert not torch.equal(
        prediction.mean_directions[:, 0],
        prediction.mean_directions[:, 1],
    )


def test_movmf_k1_reduces_to_k1_vmf_contract() -> None:
    torch.manual_seed(11)
    directions = F.normalize(torch.randn(6, 512, dtype=torch.float64), dim=-1)
    positives = F.normalize(
        directions + 0.2 * torch.randn_like(directions),
        dim=-1,
    )
    negatives = F.normalize(torch.randn_like(directions), dim=-1)
    concentrations = torch.linspace(100.0, 1000.0, 6, dtype=torch.float64)
    k1_prediction = VmfPrediction(
        mean_direction=directions,
        concentration=concentrations,
    )
    mo_prediction = MoVmfPrediction(
        mean_directions=directions[:, None, :],
        concentrations=concentrations[:, None],
        mixture_logits=torch.zeros(6, 1, dtype=torch.float64),
    )

    k1_loss = k1_vmf_retrieval_loss(
        k1_prediction,
        positives,
        negatives,
        margin=0.2,
        nll_weight=1 / 512,
        ranking_weight=1.0,
    )
    mo_loss = mo_vmf_retrieval_loss(
        mo_prediction,
        positives,
        negatives,
        margin=0.2,
        nll_weight=1 / 512,
        ranking_weight=1.0,
    )

    assert torch.allclose(mo_loss.positive_nll, k1_loss.positive_nll, atol=1e-12)
    assert torch.allclose(
        mo_loss.density_ranking,
        k1_loss.cosine_ranking,
        atol=1e-12,
    )
    assert torch.allclose(mo_loss.total, k1_loss.total, atol=1e-12)
    assert torch.allclose(
        mo_loss.normalized_positive_score,
        k1_loss.positive_cosine,
        atol=1e-14,
    )
    assert torch.allclose(
        mo_loss.normalized_negative_score,
        k1_loss.negative_cosine,
        atol=1e-14,
    )


def test_movmf_scores_are_permutation_invariant() -> None:
    torch.manual_seed(12)
    prediction = MoVmfPrediction(
        mean_directions=F.normalize(
            torch.randn(3, 4, 512, dtype=torch.float64),
            dim=-1,
        ),
        concentrations=torch.rand(3, 4, dtype=torch.float64) * 900 + 100,
        mixture_logits=torch.randn(3, 4, dtype=torch.float64),
    )
    gallery = F.normalize(torch.randn(17, 512, dtype=torch.float64), dim=-1)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = MoVmfPrediction(
        mean_directions=prediction.mean_directions[:, permutation],
        concentrations=prediction.concentrations[:, permutation],
        mixture_logits=prediction.mixture_logits[:, permutation],
    )

    scores = mo_vmf_gallery_scores(prediction, gallery)
    permuted_scores = mo_vmf_gallery_scores(permuted, gallery)

    assert torch.allclose(scores, permuted_scores, atol=1e-12, rtol=0)


def test_movmf_normalized_scores_preserve_density_ranking() -> None:
    torch.manual_seed(13)
    prediction = MoVmfPrediction(
        mean_directions=F.normalize(torch.randn(4, 3, 512), dim=-1),
        concentrations=torch.rand(4, 3) * 1000 + 1,
        mixture_logits=torch.randn(4, 3),
    )
    gallery = F.normalize(torch.randn(29, 512), dim=-1)

    raw_scores = mo_vmf_gallery_scores(prediction, gallery, normalized=False)
    normalized_scores = mo_vmf_gallery_scores(prediction, gallery, normalized=True)

    assert torch.equal(
        torch.argsort(raw_scores, dim=1, descending=True, stable=True),
        torch.argsort(normalized_scores, dim=1, descending=True, stable=True),
    )


def test_movmf_loss_trains_all_parameter_heads() -> None:
    torch.manual_seed(14)
    inputs = F.normalize(torch.randn(8, 512), dim=-1)
    positives = F.normalize(inputs + 0.2 * torch.randn_like(inputs), dim=-1)
    negatives = F.normalize(torch.randn_like(inputs), dim=-1)
    predictor = MoVmfPhotoPredictor(hidden_dim=64, num_components=4)

    losses = mo_vmf_retrieval_loss(
        predictor(inputs),
        positives,
        negatives,
        nll_weight=1 / 512,
    )
    losses.total.backward()

    assert torch.isfinite(losses.total)
    for head in (
        predictor.direction_head,
        predictor.concentration_head,
        predictor.mixture_head,
    ):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_movmf_predictor_validates_component_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        MoVmfPhotoPredictor(num_components=0)
