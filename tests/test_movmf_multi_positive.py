import torch
import torch.nn.functional as F

from spica.models.retrieval import MoVmfPhotoPredictor, MoVmfPrediction
from spica.models.vmf import (
    dominant_satellite_regularization,
    mo_vmf_multi_positive_retrieval_loss,
)
from spica.train_movmf_ablation import _scheduled_prediction


def test_fixed_kappa_warmup_uses_fixed_value():
    raw = MoVmfPrediction(
        mean_directions=F.normalize(torch.randn(2, 3, 8), dim=-1),
        concentrations=torch.full((2, 3), 64.0),
        mixture_logits=torch.zeros(2, 3),
    )
    scheduled, temperature, in_warmup = _scheduled_prediction(
        raw,
        step=0,
        warmup_steps=20,
        warmup_concentration=128.0,
        gate_temperature_start=1.0,
        gate_temperature_anneal_steps=50,
        warmup_dominant_weight=0.8,
    )
    assert in_warmup
    assert temperature == float("inf")
    assert torch.allclose(scheduled.concentrations, torch.full((2, 3), 128.0))
    assert torch.allclose(
        scheduled.mixture_logits.softmax(dim=-1),
        torch.tensor([[0.8, 0.1, 0.1]]).expand(2, -1),
    )


def test_multi_positive_loss_is_finite_and_trains_all_heads() -> None:
    torch.manual_seed(20)
    inputs = F.normalize(torch.randn(6, 512), dim=-1)
    positives = F.normalize(
        inputs[:, None, :] + 0.2 * torch.randn(6, 3, 512),
        dim=-1,
    )
    negatives = F.normalize(torch.randn(6, 512), dim=-1)
    predictor = MoVmfPhotoPredictor(
        hidden_dim=64,
        num_components=3,
        initial_concentration=64.0,
        max_concentration=1024.0,
    )

    losses = mo_vmf_multi_positive_retrieval_loss(
        predictor(inputs),
        positives,
        negatives,
        nll_weight=1 / 512,
        balance_weight=1.0,
        sharpness_weight=0.05,
        assignment_weight=1.0,
        diversity_weight=1.0,
        ranking_score_transform="tanh",
    )
    losses.total.backward()

    for value in (
        losses.total,
        losses.positive_nll,
        losses.density_ranking,
        losses.posterior_balance,
        losses.posterior_sharpness,
        losses.balanced_assignment,
        losses.direction_diversity,
    ):
        assert torch.isfinite(value)
    assert losses.posterior_responsibilities.shape == (6, 3, 3)
    for head in (
        predictor.direction_head,
        predictor.concentration_head,
        predictor.mixture_head,
    ):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_per_query_balance_rewards_using_all_components() -> None:
    directions = torch.zeros(2, 3, 512, dtype=torch.float64)
    directions[:, 0, 0] = 1
    directions[:, 1, 1] = 1
    directions[:, 2, 2] = 1
    balanced_positives = directions.clone()
    collapsed_positives = directions[:, :1, :].expand(-1, 3, -1).clone()
    negatives = torch.zeros(2, 512, dtype=torch.float64)
    negatives[:, 3] = 1
    prediction = MoVmfPrediction(
        mean_directions=directions,
        concentrations=torch.full((2, 3), 100.0, dtype=torch.float64),
        mixture_logits=torch.zeros(2, 3, dtype=torch.float64),
    )

    balanced = mo_vmf_multi_positive_retrieval_loss(
        prediction,
        balanced_positives,
        negatives,
        balance_weight=1.0,
    )
    collapsed = mo_vmf_multi_positive_retrieval_loss(
        prediction,
        collapsed_positives,
        negatives,
        balance_weight=1.0,
    )
    balanced_matching = mo_vmf_multi_positive_retrieval_loss(
        prediction,
        balanced_positives,
        negatives,
        assignment_weight=1.0,
    )
    collapsed_matching = mo_vmf_multi_positive_retrieval_loss(
        prediction,
        collapsed_positives,
        negatives,
        assignment_weight=1.0,
    )

    assert balanced.posterior_balance < 1e-10
    assert balanced.posterior_sharpness < 1e-10
    assert collapsed.posterior_balance > 1.0
    assert collapsed.posterior_sharpness < 1e-10
    assert balanced_matching.balanced_assignment < 1e-10
    assert collapsed_matching.balanced_assignment > 5.0


def test_dominant_satellite_initialization_and_regularization() -> None:
    torch.manual_seed(22)
    sketches = F.normalize(torch.randn(5, 512), dim=-1)
    positives = F.normalize(
        sketches[:, None, :] + 0.2 * torch.randn(5, 3, 512),
        dim=-1,
    )
    predictor = MoVmfPhotoPredictor(
        hidden_dim=64,
        num_components=3,
        initial_concentration=64.0,
        max_concentration=1024.0,
        initial_dominant_weight=0.8,
    )
    prediction = predictor(sketches)

    regularization = dominant_satellite_regularization(
        prediction,
        sketches,
        positives,
        target_dominant_weight=0.8,
    )
    total = (
        regularization.gate_prior
        + regularization.dominant_sketch_anchor
        + regularization.dominant_photo_anchor
        + regularization.semantic_consistency
        + regularization.satellite_coverage
        + regularization.spread_matching
        + regularization.satellite_concentration_floor
    )
    total.backward()

    expected_weights = torch.tensor([0.8, 0.1, 0.1]).expand(5, -1)
    assert torch.allclose(
        prediction.mixture_logits.softmax(dim=-1),
        expected_weights,
        atol=1e-7,
        rtol=0,
    )
    assert regularization.gate_prior < 1e-7
    assert regularization.semantic_center.shape == (5, 512)
    assert torch.isfinite(total)
    assert predictor.direction_head[-1].weight.grad is not None
    assert predictor.concentration_head[-1].weight.grad is not None
    assert predictor.mixture_head[-1].weight.grad is not None


def test_semantic_barycenter_ranking_uses_gate_weighted_direction() -> None:
    torch.manual_seed(23)
    prediction = MoVmfPrediction(
        mean_directions=F.normalize(torch.randn(4, 3, 512), dim=-1),
        concentrations=torch.rand(4, 3) * 500 + 10,
        mixture_logits=torch.randn(4, 3),
    )
    positives = F.normalize(torch.randn(4, 3, 512), dim=-1)
    negatives = F.normalize(torch.randn(4, 512), dim=-1)

    losses = mo_vmf_multi_positive_retrieval_loss(
        prediction,
        positives,
        negatives,
        ranking_score_transform="semantic_barycenter",
    )
    probabilities = prediction.mixture_logits.softmax(dim=-1)
    semantic_direction = F.normalize(
        (probabilities[:, :, None] * prediction.mean_directions).sum(dim=1),
        dim=-1,
    )

    assert torch.allclose(
        losses.ranking_positive_scores,
        (semantic_direction[:, None] * positives).sum(dim=-1),
    )
    assert torch.allclose(
        losses.ranking_negative_score,
        (semantic_direction * negatives).sum(dim=-1),
    )


def test_tanh_ranking_transform_keeps_loss_finite_for_large_score_gaps() -> None:
    torch.manual_seed(21)
    prediction = MoVmfPrediction(
        mean_directions=F.normalize(torch.randn(4, 3, 512), dim=-1),
        concentrations=torch.tensor(
            [[1.0, 1000.0, 2000.0]] * 4,
        ),
        mixture_logits=torch.randn(4, 3) * 20,
    )
    positives = F.normalize(torch.randn(4, 3, 512), dim=-1)
    negatives = F.normalize(torch.randn(4, 512), dim=-1)

    losses = mo_vmf_multi_positive_retrieval_loss(
        prediction,
        positives,
        negatives,
        ranking_score_transform="tanh",
    )

    expected_ranking = F.softplus(
        0.2
        - losses.normalized_positive_scores.tanh()
        + losses.normalized_negative_score.tanh()[:, None]
    ).mean()
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.density_ranking)
    assert torch.allclose(losses.density_ranking, expected_ranking)
