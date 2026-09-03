import inspect
import math

import torch
from torch import nn
import torch.nn.functional as F

from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.transport import (
    TransportFeatureSet,
    evaluate_transport_features,
    hidden_space_compatibility,
    multi_photo_component_alignment,
    training_angle_summary,
    transport_probe_dict,
)
from spica.models.clip import FrozenVisualProjection, TrainableSketchHiddenEncoder
from spica.train_transport import _freeze_encoder
from spica.models.transport import (
    SpicaPredictiveTransport,
    deterministic_direction_mixture_loss,
    directional_mixture_loss,
    fixed_origin_transport_target,
    parallel_transport_tangent,
    photo_transport_target,
    tangent_projection,
    transport_geometry_loss,
)


class _DummyVisual(nn.Module):
    output_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.eye(4, 3))
        self.transformer = nn.Identity()
        self.lift = nn.Linear(3, 4)

    def _embeds(self, images: torch.Tensor) -> torch.Tensor:
        return self.lift(images.mean(dim=(2, 3))).unsqueeze(1)

    def _pool(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return values[:, 0], values


def _model(*, mode: str = "tangent", k: int = 1, use_vmf: bool = False):
    visual = _DummyVisual()
    encoder = TrainableSketchHiddenEncoder(visual, hidden_dim=4, mode="full")
    projection = FrozenVisualProjection(torch.randn(4, 3))
    return SpicaPredictiveTransport(
        encoder,
        projection,
        transport_mode=mode,
        predictor_hidden_dim=8,
        num_components=k,
        use_vmf=use_vmf,
        rho_max=math.pi / 3,
        initial_rho=math.radians(1),
    )


def test_transport_uses_pre_projection_hidden_and_stable_origin() -> None:
    model = _model(mode="residual")
    images = torch.randn(3, 3, 8, 8)
    prediction = model(images)
    assert prediction.h.shape == (3, 4)
    assert prediction.z0.shape == (3, 3)
    assert prediction.q.shape == (3, 3)
    assert torch.allclose(prediction.q, prediction.z0, atol=1e-5)
    assert model.photo_projection.matrix.requires_grad is False
    assert not hasattr(model, "text")


def test_nonlearned_rho_controls_have_expected_schedule_and_no_head() -> None:
    for strategy, expected in (("zero", 0.0), ("fixed", math.radians(15.0))):
        model = SpicaPredictiveTransport(
            TrainableSketchHiddenEncoder(_DummyVisual(), hidden_dim=4, mode="full"),
            FrozenVisualProjection(torch.randn(4, 3)),
            predictor_hidden_dim=8,
            rho_max=math.radians(15.0),
            fixed_rho=math.radians(15.0),
            rho_strategy=strategy,
        )
        assert model.transport_head.rho_head is None  # type: ignore[attr-defined]
        output = model(torch.randn(2, 3, 8, 8))
        assert torch.allclose(output.rho, torch.full((2,), expected), atol=1e-6)
    model = SpicaPredictiveTransport(
        TrainableSketchHiddenEncoder(_DummyVisual(), hidden_dim=4, mode="full"),
        FrozenVisualProjection(torch.randn(4, 3)),
        predictor_hidden_dim=8,
        rho_max=math.radians(15.0),
        rho_strategy="linear_warmup",
        rho_warmup_steps=75,
    )
    model.set_schedule_step(37)
    output = model(torch.randn(2, 3, 8, 8))
    assert torch.allclose(
        output.rho, torch.full((2,), math.radians(15.0 * 37 / 75)), atol=1e-6
    )


def test_hidden_compatibility_and_training_angle_summary_are_finite() -> None:
    projection = FrozenVisualProjection(torch.randn(5, 3))
    reference = torch.randn(12, 5)
    adapted = reference @ torch.linalg.qr(torch.randn(5, 5)).Q
    compatibility = hidden_space_compatibility(reference, adapted, projection)
    assert 0.0 <= compatibility["linear_cka"] <= 1.00001
    assert compatibility["procrustes_residual"] < 1e-4
    assert -1.0 <= compatibility["frozen_projection_mean_cosine"] <= 1.0
    summary = training_angle_summary(torch.tensor([0.1, 0.2, 0.8]))
    assert summary["count"] == 3
    assert abs(summary["fraction_gt_15_degrees"] - 1.0 / 3.0) < 1e-6


def test_multi_photo_residual_alignment_has_one_value_per_component() -> None:
    rows, components, dimension = 6, 3, 4
    z0 = F.normalize(torch.randn(rows, dimension), dim=-1)
    directions = torch.randn(rows, components, dimension)
    directions = (
        directions
        - (directions * z0[:, None, :]).sum(dim=-1, keepdim=True) * z0[:, None, :]
    )
    directions = F.normalize(directions, dim=-1)
    features = TransportFeatureSet(
        h=torch.randn(rows, dimension),
        z0=z0,
        directions=directions,
        concentrations=None,
        gate_logits=torch.randn(rows, components),
        rho=torch.rand(rows),
        q_hypotheses=F.normalize(z0[:, None, :] + 0.1 * directions, dim=-1),
        q=F.normalize(z0 + 0.1 * directions[:, 0, :], dim=-1),
        labels=torch.tensor([0, 0, 0, 1, 1, 1]),
        paths=tuple(f"sketch-{index}" for index in range(rows)),
    )
    photo_embeddings = F.normalize(torch.randn(8, dimension), dim=-1)
    photo_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    alignment = multi_photo_component_alignment(
        features, photo_embeddings, photo_labels, photos_per_class=3
    )
    assert len(alignment["class_alignment_by_component"]) == components
    assert len(alignment["instance_residual_alignment_by_component"]) == components
    assert isinstance(alignment["instance_residual_alignment_mean"], float)


def test_tangent_transport_is_unit_and_orthogonal() -> None:
    prediction = _model(k=2)(torch.randn(5, 3, 8, 8))
    assert prediction.directions.shape == (5, 2, 3)
    assert prediction.q_hypotheses.shape == (5, 2, 3)
    assert torch.allclose(
        prediction.directions.norm(dim=-1), torch.ones(5, 2), atol=1e-5
    )
    assert torch.allclose(
        prediction.q_hypotheses.norm(dim=-1), torch.ones(5, 2), atol=1e-5
    )
    tangent_cosine = (prediction.directions * prediction.z0[:, None, :]).sum(dim=-1)
    assert tangent_cosine.abs().max() < 1e-5


def test_sphere_tangent_projection_and_parallel_transport() -> None:
    x = F.normalize(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    y = F.normalize(torch.tensor([[2.0, -1.0, 4.0]]), dim=-1)
    v = tangent_projection(torch.tensor([[0.4, -0.2, 0.7]]), x)
    transported = parallel_transport_tangent(v, x, y)
    assert torch.allclose((transported * y).sum(dim=-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(transported.norm(dim=-1), v.norm(dim=-1), atol=1e-5)


def test_parallel_transport_identity_and_fixed_origin_diagnostic() -> None:
    x = F.normalize(torch.randn(4, 8), dim=-1)
    v = tangent_projection(torch.randn(4, 8), x)
    assert torch.allclose(parallel_transport_tangent(v, x, x), v, atol=1e-6)
    z0 = F.normalize(torch.randn(4, 8), dim=-1)
    target = F.normalize(torch.randn(4, 8), dim=-1)
    fixed = fixed_origin_transport_target(x, target, z0)
    assert torch.allclose((fixed.direction * z0).sum(dim=-1), torch.zeros(4), atol=1e-5)


def test_freeze_after_warmup_disables_encoder_gradients_and_state_updates() -> None:
    model = _model()
    model.train()
    _freeze_encoder(model)
    assert not any(
        parameter.requires_grad
        for parameter in model.sketch_context_encoder.parameters()
    )
    assert not model.sketch_context_encoder.training


def test_text_never_enters_predictor_signature() -> None:
    assert list(inspect.signature(SpicaPredictiveTransport.forward).parameters) == [
        "self",
        "sketch_images",
    ]


def test_base_query_control_is_exactly_z0_and_has_no_transport_gradients() -> None:
    model = _model()
    model.transport_enabled = False
    model.transport_head.requires_grad_(False)
    output = model(torch.randn(3, 3, 8, 8))
    assert torch.allclose(output.q, output.z0)
    assert torch.allclose(output.q_hypotheses[:, 0], output.z0)
    assert not any(
        parameter.requires_grad for parameter in model.transport_head.parameters()
    )


def test_photo_transport_target_handles_zero_angle() -> None:
    base = F.normalize(torch.randn(4, 8), dim=-1)
    target = photo_transport_target(base, base)
    assert torch.all(target.near_zero)
    assert torch.allclose(target.theta, torch.zeros(4))
    assert torch.allclose(target.direction.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(
        (target.direction * base).sum(dim=-1), torch.zeros(4), atol=1e-5
    )


def test_transport_losses_and_vmf_direction_mixture_are_finite() -> None:
    torch.manual_seed(8)
    prediction = _model(k=2, use_vmf=True)(torch.randn(6, 3, 8, 8))
    target = photo_transport_target(
        prediction.z0, F.normalize(torch.randn(6, 3), dim=-1)
    )
    positives = F.normalize(torch.randn(6, 1, 3), dim=-1)
    negatives = F.normalize(torch.randn(6, 3), dim=-1)
    loss = directional_mixture_loss(prediction, target.direction, positives, negatives)
    assert torch.isfinite(loss.total)
    assert torch.allclose(loss.posterior_responsibilities.sum(dim=-1), torch.ones(6))
    control_prediction = _model(k=2)(torch.randn(6, 3, 8, 8))
    control = deterministic_direction_mixture_loss(
        control_prediction, target.direction, positives, negatives
    )
    assert torch.isfinite(control.total)
    assert torch.isfinite(
        transport_geometry_loss(prediction.q, torch.randn_like(prediction.q))
    )


def test_transport_retrieval_modes_and_probes() -> None:
    torch.manual_seed(9)
    model = _model(k=2)
    images = torch.randn(6, 3, 8, 8)
    output = model(images)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    features = TransportFeatureSet(
        h=output.h.detach(),
        z0=output.z0.detach(),
        directions=output.directions.detach(),
        rho=output.rho.detach(),
        q_hypotheses=output.q_hypotheses.detach(),
        q=output.q.detach(),
        labels=labels,
        paths=tuple(f"q{i}" for i in range(6)),
        gate_logits=None,
    )
    gallery_embeddings = F.normalize(torch.randn(9, 3), dim=-1)
    gallery = EncodedRetrievalSet(
        embeddings=gallery_embeddings,
        labels=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        paths=tuple(f"p{i}" for i in range(9)),
    )
    evaluations = evaluate_transport_features(
        features,
        gallery,
        precision_at_k=(1, 3),
        map_at_k=(3,),
        query_chunk_size=2,
    )
    assert set(evaluations) == {"barycentric", "angular_logsumexp", "max"}
    assert all(
        torch.isfinite(torch.tensor(e.metrics.mean_average_precision))
        for e in evaluations.values()
    )
    probe = transport_probe_dict(features, gallery, frozen_reference=features.z0)
    assert "transport" in probe
    assert "mixture" in probe
    assert "effective_rank" in probe["q"]
