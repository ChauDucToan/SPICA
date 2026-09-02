"""Predictive Semantic Transport models and losses for SPICA.

The transport family keeps a stable semantic origin in the frozen photo CLIP
coordinate system.  A raw sketch is encoded to the visual hidden state before
CLIP's projection, projected with a *frozen photo* projection, and only then
adapted by a residual or tangent-space transport head.

No class text, photo image, or gallery feature is accepted by a model forward
pass.  Those values belong to training losses and evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .clip import (
    FrozenVisualProjection,
    TrainableSketchHiddenEncoder,
)

TransportMode = Literal["residual", "bounded_residual", "tangent"]


def _validate_batch_features(name: str, value: Tensor, ndim: int | None = None) -> None:
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating-point")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _inverse_sigmoid(value: float, *, eps: float = 1e-6) -> float:
    bounded = min(max(value, eps), 1.0 - eps)
    return math.log(bounded / (1.0 - bounded))


def _feature_input(h: Tensor, z0: Tensor, use_z0: bool) -> Tensor:
    _validate_batch_features("h", h, 2)
    _validate_batch_features("z0", z0, 2)
    if h.shape[0] != z0.shape[0]:
        raise ValueError("h and z0 must have the same batch size")
    return torch.cat((h, z0), dim=-1) if use_z0 else h


class _TransportMLP(nn.Module):
    """A rich-feature MLP with an explicitly accessible final layer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("transport MLP dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @property
    def output(self) -> nn.Linear:
        return self.network[-1]

    def zero_initialize_output(self) -> None:
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initialize_output(self, std: float) -> None:
        if not math.isfinite(std) or std <= 0:
            raise ValueError("output initialization std must be finite and positive")
        nn.init.normal_(self.output.weight, std=std)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}], got "
                f"{tuple(features.shape)}"
            )
        return self.network(features)


def _safe_normalize(value: Tensor, *, dim: int = -1, eps: float = 1e-8) -> Tensor:
    return F.normalize(value, dim=dim, eps=eps)


def tangent_projection(vector: Tensor, base: Tensor) -> Tensor:
    """Project ``vector`` onto the tangent plane at unit ``base``."""
    _validate_batch_features("vector", vector)
    _validate_batch_features("base", base, 2)
    if vector.shape[-1] != base.shape[-1] or vector.shape[0] != base.shape[0]:
        raise ValueError("vector and base dimensions must agree")
    base_expanded = base if vector.ndim == 2 else base.unsqueeze(1)
    return vector - (vector * base_expanded).sum(dim=-1, keepdim=True) * base_expanded


def parallel_transport_tangent(
    vector: Tensor,
    source: Tensor,
    destination: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Parallel-transport a tangent vector between points on a unit sphere.

    The closed form is valid away from antipodal points:
    ``v - (v·y)/(1+x·y) * (x+y)``.  At the antipodal singularity there is no
    unique shortest geodesic, so we use the destination tangent projection as
    a deterministic, numerically safe fallback.
    """
    _validate_batch_features("vector", vector, 2)
    _validate_batch_features("source", source, 2)
    _validate_batch_features("destination", destination, 2)
    if vector.shape != source.shape or source.shape != destination.shape:
        raise ValueError("vector, source, and destination shapes must match")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    x = _safe_normalize(source)
    y = _safe_normalize(destination)
    v = tangent_projection(vector, x)
    denominator = 1.0 + (x * y).sum(dim=-1, keepdim=True)
    transported = v - ((v * y).sum(dim=-1, keepdim=True) / denominator.clamp_min(eps)) * (x + y)
    fallback = tangent_projection(v, y)
    return torch.where(denominator > eps, transported, fallback)


def _stable_tangent_direction(
    vector: Tensor,
    base: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Normalize a tangent vector and use a deterministic safe fallback.

    At zero transport angle the target direction is mathematically undefined.
    A least-aligned coordinate basis gives a unit tangent fallback without
    injecting a learned semantic direction or producing NaNs.
    """
    if vector.ndim not in {2, 3} or base.ndim != 2:
        raise ValueError("vector must be [B,D] or [B,K,D], base must be [B,D]")
    if vector.shape[0] != base.shape[0] or vector.shape[-1] != base.shape[-1]:
        raise ValueError("vector and base dimensions must agree")
    if vector.shape[-1] < 2:
        raise ValueError("a tangent direction requires embedding dimension >= 2")
    tangent = tangent_projection(vector, base)
    norm = tangent.norm(dim=-1, keepdim=True)

    # Select the coordinate axis least aligned with the base.  This avoids the
    # unstable subtraction that would result from choosing a fixed axis that is
    # nearly parallel to z0.
    coordinate = base.abs().argmin(dim=-1)
    basis = F.one_hot(coordinate, num_classes=base.shape[-1]).to(dtype=base.dtype)
    if vector.ndim == 3:
        basis = basis.unsqueeze(1).expand(-1, vector.shape[1], -1)
        base_for_basis = base.unsqueeze(1)
    else:
        base_for_basis = base
    fallback = basis - (basis * base_for_basis).sum(dim=-1, keepdim=True) * base_for_basis
    fallback = _safe_normalize(fallback, eps=eps)
    normalized = tangent / norm.clamp_min(eps)
    return torch.where(norm > eps, normalized, fallback)


@dataclass(frozen=True, slots=True)
class SemanticTransportPrediction:
    """Raw-sketch transport output.

    ``q_hypotheses`` is always ``[B,K,D]``.  ``q`` is the single retrieval
    endpoint for K=1 and the gate-weighted barycentric endpoint for K>1.
    ``rho`` is ``[B]`` for shared distance and ``[B,K]`` for component distance.
    """

    h: Tensor
    z0: Tensor
    directions: Tensor
    rho: Tensor
    q_hypotheses: Tensor
    q: Tensor
    delta_raw: Tensor | None = None
    alpha: Tensor | None = None
    gate_logits: Tensor | None = None
    concentrations: Tensor | None = None

    @property
    def direction(self) -> Tensor:
        if self.directions.shape[1] != 1:
            raise ValueError("direction is only unambiguous for K=1")
        return self.directions[:, 0]

    @property
    def pi(self) -> Tensor:
        if self.gate_logits is None:
            return self.directions.new_ones(
                self.directions.shape[0], self.directions.shape[1]
            ) / self.directions.shape[1]
        return self.gate_logits.softmax(dim=-1)

    @property
    def num_components(self) -> int:
        return int(self.directions.shape[1])


class ResidualTransportHead(nn.Module):
    """Simple residual ``normalize(z0 + alpha * delta_raw)`` transport."""

    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        predictor_hidden_dim: int = 512,
        *,
        use_z0: bool = False,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.use_z0 = use_z0
        self.alpha_value = alpha
        input_dim = hidden_dim + (embedding_dim if use_z0 else 0)
        self.delta_head = _TransportMLP(input_dim, predictor_hidden_dim, embedding_dim)
        # The residual starts exactly at the stable semantic origin.
        self.delta_head.zero_initialize_output()

    def forward(self, h: Tensor, z0: Tensor) -> SemanticTransportPrediction:
        features = _feature_input(h, z0, self.use_z0)
        delta = self.delta_head(features)
        q = _safe_normalize(z0 + self.alpha_value * delta)
        directions = _safe_normalize(delta).unsqueeze(1)
        q_hypotheses = q.unsqueeze(1)
        rho = torch.acos(
            (z0 * q).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        )
        return SemanticTransportPrediction(
            h=h,
            z0=z0,
            directions=directions,
            rho=rho,
            q_hypotheses=q_hypotheses,
            q=q,
            delta_raw=delta,
            alpha=z0.new_full((z0.shape[0],), self.alpha_value),
            gate_logits=None,
            concentrations=None,
        )


class BoundedResidualTransportHead(nn.Module):
    """Residual transport with a learned bounded correction scale."""

    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        predictor_hidden_dim: int = 512,
        *,
        use_z0: bool = False,
        alpha_max: float = 0.5,
        initial_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError("alpha_max must be finite and positive")
        if not math.isfinite(initial_alpha) or not 0 <= initial_alpha < alpha_max:
            raise ValueError("initial_alpha must be in [0, alpha_max)")
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.use_z0 = use_z0
        self.alpha_max = alpha_max
        self.initial_alpha = initial_alpha
        input_dim = hidden_dim + (embedding_dim if use_z0 else 0)
        self.delta_head = _TransportMLP(input_dim, predictor_hidden_dim, embedding_dim)
        self.delta_head.initialize_output(1e-3)
        self.alpha_head = _TransportMLP(input_dim, predictor_hidden_dim, 1)
        self.alpha_head.zero_initialize_output()
        self.alpha_head.output.bias.data.fill_(
            _inverse_sigmoid(initial_alpha / alpha_max)
        )

    def forward(self, h: Tensor, z0: Tensor) -> SemanticTransportPrediction:
        features = _feature_input(h, z0, self.use_z0)
        delta = self.delta_head(features)
        delta_unit = _safe_normalize(delta)
        alpha = self.alpha_max * self.alpha_head(features).squeeze(-1).sigmoid()
        q = _safe_normalize(z0 + alpha[:, None] * delta_unit)
        directions = delta_unit.unsqueeze(1)
        rho = torch.acos(
            (z0 * q).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        )
        return SemanticTransportPrediction(
            h=h,
            z0=z0,
            directions=directions,
            rho=rho,
            q_hypotheses=q.unsqueeze(1),
            q=q,
            delta_raw=delta,
            alpha=alpha,
            gate_logits=None,
            concentrations=None,
        )


class TangentTransportHead(nn.Module):
    """Spherical transport direction and distance predictor.

    Directions are projected into the tangent plane at z0.  The endpoint is
    produced by the exponential map ``cos(rho) z0 + sin(rho) d`` and therefore
    cannot change the semantic origin by an unconstrained full-vector rewrite.
    """

    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        predictor_hidden_dim: int = 512,
        *,
        num_components: int = 1,
        use_z0: bool = False,
        rho_max: float = math.pi / 4.0,
        initial_rho: float = 0.0,
        shared_rho: bool = True,
        use_vmf: bool = False,
        min_kappa: float = 1e-4,
        max_kappa: float = 2048.0,
        initial_kappa: float = 64.0,
        direction_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or embedding_dim < 2 or predictor_hidden_dim <= 0:
            raise ValueError("invalid tangent transport dimensions")
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        if not math.isfinite(rho_max) or not 0 < rho_max <= math.pi:
            raise ValueError("rho_max must be in (0, pi]")
        if not math.isfinite(initial_rho) or not 0 <= initial_rho < rho_max:
            raise ValueError("initial_rho must be in [0, rho_max)")
        if not math.isfinite(min_kappa) or min_kappa < 0:
            raise ValueError("min_kappa must be finite and non-negative")
        if not math.isfinite(max_kappa) or max_kappa <= min_kappa:
            raise ValueError("max_kappa must exceed min_kappa")
        if not math.isfinite(initial_kappa) or not min_kappa < initial_kappa < max_kappa:
            raise ValueError("initial_kappa must be strictly within kappa bounds")
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.num_components = num_components
        self.use_z0 = use_z0
        self.rho_max = rho_max
        self.initial_rho = initial_rho
        self.shared_rho = shared_rho
        self.use_vmf = use_vmf
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        self.initial_kappa = initial_kappa
        input_dim = hidden_dim + (embedding_dim if use_z0 else 0)

        self.direction_head = _TransportMLP(
            input_dim,
            predictor_hidden_dim,
            num_components * embedding_dim,
        )
        self.direction_head.initialize_output(direction_init_std)

        rho_outputs = 1 if shared_rho else num_components
        self.rho_head = _TransportMLP(input_dim, predictor_hidden_dim, rho_outputs)
        self.rho_head.zero_initialize_output()
        self.rho_head.output.bias.data.fill_(
            _inverse_sigmoid(initial_rho / rho_max)
        )

        if num_components > 1:
            self.gate_head = _TransportMLP(input_dim, predictor_hidden_dim, num_components)
            self.gate_head.zero_initialize_output()
        else:
            self.gate_head = None

        if use_vmf:
            self.kappa_head = _TransportMLP(
                input_dim, predictor_hidden_dim, num_components
            )
            self.kappa_head.zero_initialize_output()
            self.kappa_head.output.bias.data.fill_(
                _inverse_sigmoid(
                    (initial_kappa - min_kappa) / (max_kappa - min_kappa)
                )
            )
        else:
            self.kappa_head = None

    def forward(self, h: Tensor, z0: Tensor) -> SemanticTransportPrediction:
        features = _feature_input(h, z0, self.use_z0)
        batch_size = h.shape[0]
        raw = self.direction_head(features).reshape(
            batch_size, self.num_components, self.embedding_dim
        )
        directions = _stable_tangent_direction(raw, z0)
        raw_rho = self.rho_head(features)
        if self.shared_rho:
            rho = self.rho_max * raw_rho.squeeze(-1).sigmoid()
            rho_for_endpoint = rho[:, None].expand(-1, self.num_components)
        else:
            rho = self.rho_max * raw_rho.sigmoid()
            rho_for_endpoint = rho
        q_hypotheses = (
            rho_for_endpoint.cos()[..., None] * z0[:, None, :]
            + rho_for_endpoint.sin()[..., None] * directions
        )
        # Normalization only removes floating point error; the expression above
        # is the hypersphere exponential map when d and z0 are unit/orthogonal.
        q_hypotheses = _safe_normalize(q_hypotheses)
        if self.gate_head is None:
            gate_logits = None
            q = q_hypotheses[:, 0]
        else:
            gate_logits = self.gate_head(features)
            q = _safe_normalize(
                (gate_logits.softmax(dim=-1)[..., None] * q_hypotheses).sum(dim=1)
            )

        concentrations: Tensor | None = None
        if self.kappa_head is not None:
            concentrations = self.min_kappa + (
                self.max_kappa - self.min_kappa
            ) * self.kappa_head(features).sigmoid()

        return SemanticTransportPrediction(
            h=h,
            z0=z0,
            directions=directions,
            rho=rho,
            q_hypotheses=q_hypotheses,
            q=q,
            delta_raw=raw,
            alpha=None,
            gate_logits=gate_logits,
            concentrations=concentrations,
        )


class SpicaPredictiveTransport(nn.Module):
    """Raw-sketch Predictive Semantic Transport model.

    The model has no text/photo arguments.  ``photo_projection`` is a frozen
    parameter buffer copied from the photo CLIP image encoder at construction.
    """

    def __init__(
        self,
        sketch_context_encoder: TrainableSketchHiddenEncoder,
        photo_projection: FrozenVisualProjection,
        *,
        transport_mode: TransportMode = "tangent",
        predictor_hidden_dim: int = 512,
        num_components: int = 1,
        use_z0: bool = False,
        alpha: float = 1.0,
        alpha_max: float = 0.5,
        initial_alpha: float = 0.0,
        rho_max: float = math.pi / 4.0,
        initial_rho: float = 0.0,
        shared_rho: bool = True,
        use_vmf: bool = False,
        min_kappa: float = 1e-4,
        max_kappa: float = 2048.0,
        initial_kappa: float = 64.0,
        direction_init_std: float = 1e-3,
        transport_enabled: bool = True,
    ) -> None:
        super().__init__()
        if transport_mode not in {"residual", "bounded_residual", "tangent"}:
            raise ValueError(f"Unsupported transport_mode: {transport_mode!r}")
        if sketch_context_encoder.hidden_dim != photo_projection.hidden_dim:
            raise ValueError(
                "Sketch hidden dimension and photo projection input dimension must "
                f"match, got {sketch_context_encoder.hidden_dim} and "
                f"{photo_projection.hidden_dim}"
            )
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        if transport_mode != "tangent" and num_components != 1:
            raise ValueError("residual transport currently supports only K=1")
        if transport_mode != "tangent" and use_vmf:
            raise ValueError("vMF is only defined for tangent direction transport")
        self.sketch_context_encoder = sketch_context_encoder
        self.photo_projection = photo_projection
        self.transport_mode = transport_mode
        self.embedding_dim = photo_projection.embedding_dim
        self.hidden_dim = sketch_context_encoder.hidden_dim
        self.num_components = num_components
        self.shared_rho = shared_rho
        self.use_vmf = use_vmf
        self.transport_enabled = transport_enabled

        if transport_mode == "residual":
            self.transport_head: nn.Module = ResidualTransportHead(
                self.hidden_dim,
                self.embedding_dim,
                predictor_hidden_dim,
                use_z0=use_z0,
                alpha=alpha,
            )
        elif transport_mode == "bounded_residual":
            self.transport_head = BoundedResidualTransportHead(
                self.hidden_dim,
                self.embedding_dim,
                predictor_hidden_dim,
                use_z0=use_z0,
                alpha_max=alpha_max,
                initial_alpha=initial_alpha,
            )
        else:
            self.transport_head = TangentTransportHead(
                self.hidden_dim,
                self.embedding_dim,
                predictor_hidden_dim,
                num_components=num_components,
                use_z0=use_z0,
                rho_max=rho_max,
                initial_rho=initial_rho,
                shared_rho=shared_rho,
                use_vmf=use_vmf,
                min_kappa=min_kappa,
                max_kappa=max_kappa,
                initial_kappa=initial_kappa,
                direction_init_std=direction_init_std,
            )
        if not transport_enabled:
            self.transport_head.requires_grad_(False)

    @property
    def predictor(self) -> nn.Module:
        """Compatibility alias for optimizer/checkpoint callers."""
        return self.transport_head

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def transport_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.transport_head.parameters())

    @property
    def sketch_encoder_trainable_parameter_count(self) -> int:
        return self.sketch_context_encoder.trainable_parameter_count

    def forward(self, sketch_images: Tensor) -> SemanticTransportPrediction:
        h = self.sketch_context_encoder(sketch_images)
        z0 = _safe_normalize(self.photo_projection(h))
        if not self.transport_enabled:
            # Matched text/no-text base controls still use the same model
            # wrapper and encoder, but the transport head is not part of the
            # forward graph or optimizer objective.
            return SemanticTransportPrediction(
                h=h,
                z0=z0,
                directions=z0[:, None, :],
                rho=z0.new_zeros((z0.shape[0],)),
                q_hypotheses=z0[:, None, :],
                q=z0,
            )
        return self.transport_head(h, z0)


# Public descriptive aliases used by experiment scripts and downstream code.
SketchPhotoTransportPredictor = SpicaPredictiveTransport
SketchPhotoTransport = SpicaPredictiveTransport
SpicaSemanticTransport = SpicaPredictiveTransport
PredictiveSemanticTransport = SpicaPredictiveTransport


@dataclass(frozen=True, slots=True)
class GeodesicTransportTarget:
    z_photo: Tensor
    cosine: Tensor
    theta: Tensor
    direction: Tensor
    near_zero: Tensor


def photo_transport_target(
    z0: Tensor,
    photo_embedding: Tensor,
    *,
    eps: float = 1e-6,
) -> GeodesicTransportTarget:
    """Decompose a frozen photo target into tangent direction and angle."""
    _validate_batch_features("z0", z0, 2)
    _validate_batch_features("photo_embedding", photo_embedding, 2)
    if z0.shape != photo_embedding.shape:
        raise ValueError("z0 and photo_embedding must have matching shapes")
    if not 0 < eps < 0.5 or not math.isfinite(eps):
        raise ValueError("eps must be finite and in (0, .5)")
    base = _safe_normalize(z0).detach()
    target = _safe_normalize(photo_embedding).detach()
    raw_cosine = (base * target).sum(dim=-1).clamp(-1.0, 1.0)
    cosine = raw_cosine.clamp(min=-1.0 + eps, max=1.0 - eps)
    theta = torch.acos(cosine)
    near_zero = (1.0 - raw_cosine).abs() <= eps
    theta = torch.where(near_zero, torch.zeros_like(theta), theta)
    tangent = target - cosine[:, None] * base
    direction = _stable_tangent_direction(tangent, base, eps=eps).detach()
    return GeodesicTransportTarget(
        z_photo=target,
        cosine=cosine.detach(),
        theta=theta.detach(),
        direction=direction,
        near_zero=near_zero.detach(),
    )


def fixed_origin_transport_target(
    z_ref: Tensor,
    photo_embedding: Tensor,
    z0: Tensor,
    *,
    eps: float = 1e-6,
) -> GeodesicTransportTarget:
    """Build a diagnostic target at a frozen origin and move it to ``z0``.

    ``z_ref`` is intentionally diagnostic-only: this function must never be
    used as a model input.  The angle and cosine remain those measured at the
    frozen origin; only the tangent direction is parallel-transported to the
    current model origin so that its cosine with a prediction is well-defined.
    """
    fixed = photo_transport_target(z_ref, photo_embedding, eps=eps)
    moved_direction = parallel_transport_tangent(
        fixed.direction, z_ref, z0, eps=eps
    ).detach()
    return GeodesicTransportTarget(
        z_photo=fixed.z_photo,
        cosine=fixed.cosine,
        theta=fixed.theta,
        direction=moved_direction,
        near_zero=fixed.near_zero,
    )


# Descriptive aliases for experiment notebooks and downstream callers.
compute_photo_transport_target = photo_transport_target
geodesic_transport_target = photo_transport_target


def transport_direction_loss(
    predicted_direction: Tensor,
    target_direction: Tensor,
) -> Tensor:
    """Cosine loss for K=1 tangent directions."""
    _validate_batch_features("predicted_direction", predicted_direction, 2)
    _validate_batch_features("target_direction", target_direction, 2)
    if predicted_direction.shape != target_direction.shape:
        raise ValueError("predicted and target direction shapes must match")
    return 1.0 - (
        _safe_normalize(predicted_direction)
        * _safe_normalize(target_direction.detach())
    ).sum(dim=-1).mean()


def transport_distance_loss(predicted_rho: Tensor, target_theta: Tensor) -> Tensor:
    if predicted_rho.ndim != 1 or target_theta.ndim != 1:
        raise ValueError("predicted_rho and target_theta must have shape [B]")
    if predicted_rho.shape != target_theta.shape:
        raise ValueError("predicted_rho and target_theta shapes must match")
    _validate_batch_features("predicted_rho", predicted_rho, 1)
    _validate_batch_features("target_theta", target_theta, 1)
    return F.smooth_l1_loss(predicted_rho, target_theta.detach())


def transport_endpoint_loss(queries: Tensor, photo_target: Tensor) -> Tensor:
    _validate_batch_features("queries", queries, 2)
    _validate_batch_features("photo_target", photo_target, 2)
    if queries.shape != photo_target.shape:
        raise ValueError("queries and photo_target shapes must match")
    return 1.0 - (
        _safe_normalize(queries) * _safe_normalize(photo_target.detach())
    ).sum(dim=-1).mean()


def transport_ranking_loss(
    queries: Tensor,
    positive_target: Tensor,
    negative_embedding: Tensor,
    *,
    margin: float = 0.2,
) -> Tensor:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    _validate_batch_features("queries", queries, 2)
    _validate_batch_features("positive_target", positive_target, 2)
    _validate_batch_features("negative_embedding", negative_embedding, 2)
    if queries.shape != positive_target.shape or queries.shape != negative_embedding.shape:
        raise ValueError("ranking tensors must have matching shapes")
    query = _safe_normalize(queries)
    positive = _safe_normalize(positive_target.detach())
    negative = _safe_normalize(negative_embedding.detach())
    return F.softplus(
        margin - (query * positive).sum(dim=-1) + (query * negative).sum(dim=-1)
    ).mean()


def transport_geometry_loss(
    queries: Tensor,
    reference: Tensor,
    *,
    off_diagonal: bool = True,
) -> Tensor:
    """Preserve relational geometry without pinning q to the reference point."""
    _validate_batch_features("queries", queries, 2)
    _validate_batch_features("reference", reference, 2)
    if queries.shape != reference.shape or queries.shape[0] < 2:
        raise ValueError("queries/reference must match and contain at least two rows")
    query_gram = _safe_normalize(queries) @ _safe_normalize(queries).T
    reference_gram = _safe_normalize(reference.detach()) @ _safe_normalize(reference.detach()).T
    difference = (query_gram - reference_gram).square()
    if off_diagonal:
        mask = ~torch.eye(difference.shape[0], dtype=torch.bool, device=difference.device)
        return difference[mask].mean()
    return difference.mean()


def transport_multi_hypothesis_ranking_loss(
    q_hypotheses: Tensor,
    gate_logits: Tensor | None,
    positive_targets: Tensor,
    negative_embedding: Tensor,
    *,
    margin: float = 0.2,
    temperature: float = 0.07,
) -> Tensor:
    """Repository-native ranking loss over K angular transport endpoints."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    _validate_batch_features("q_hypotheses", q_hypotheses, 3)
    _validate_batch_features("positive_targets", positive_targets, 3)
    _validate_batch_features("negative_embedding", negative_embedding, 2)
    b, k, d = q_hypotheses.shape
    if positive_targets.shape[0] != b or positive_targets.shape[2] != d:
        raise ValueError("positive target dimensions must match hypotheses")
    if negative_embedding.shape != (b, d):
        raise ValueError("negative embedding dimensions must match hypotheses")
    if gate_logits is None:
        log_pi = q_hypotheses.new_full((b, k), -math.log(k))
    else:
        if gate_logits.shape != (b, k):
            raise ValueError("gate_logits must have shape [B,K]")
        log_pi = gate_logits.log_softmax(dim=-1)
    hypotheses = _safe_normalize(q_hypotheses)
    positives = _safe_normalize(positive_targets.detach())
    negative = _safe_normalize(negative_embedding.detach())
    # Elementwise products avoid PyTorch's bmm_outer_product Triton dispatch
    # on systems where the optional ldconfig utility is unavailable.
    positive_cosines = (
        hypotheses[:, None, :, :] * positives[:, :, None, :]
    ).sum(dim=-1)
    negative_cosines = (hypotheses * negative[:, None, :]).sum(dim=-1)
    positive_scores = temperature * torch.logsumexp(
        log_pi[:, None, :] + positive_cosines / temperature, dim=-1
    )
    negative_score = temperature * torch.logsumexp(
        log_pi + negative_cosines / temperature, dim=-1
    )
    return F.softplus(
        margin - positive_scores + negative_score[:, None]
    ).mean()


@dataclass(frozen=True, slots=True)
class DirectionMixtureLoss:
    total: Tensor
    direction_nll: Tensor
    ranking: Tensor
    posterior_responsibilities: Tensor
    mixture_probabilities: Tensor
    positive_direction_cosine: Tensor
    mean_kappa: Tensor


def _relative_log_vmf_normalizer(
    concentration: Tensor,
    *,
    dimension: int,
) -> Tensor:
    """Relative log normalizer, with a small-dimension exact fallback.

    The repository's high-dimensional approximation is used for the actual
    512-D CLIP experiments.  The short series/asymptotic fallback makes the
    transport loss usable in small synthetic unit tests as well.
    """
    if dimension < 2:
        raise ValueError("vMF dimension must be at least 2")
    if dimension >= 32:
        from .vmf import log_vmf_normalizer

        return log_vmf_normalizer(
            concentration,
            dimension=dimension,
            relative_to_uniform=True,
        )
    kappa = concentration.clamp_min(1e-6).to(torch.float64)
    nu = dimension / 2.0 - 1.0
    small = kappa < 40.0
    terms_count = 256
    m = torch.arange(terms_count, device=kappa.device, dtype=torch.float64)
    log_k_half = torch.log(kappa / 2.0)[..., None]
    terms = (
        (2.0 * m + nu) * log_k_half
        - torch.lgamma(m + 1.0)
        - torch.lgamma(m + nu + 1.0)
    )
    log_bessel_series = torch.logsumexp(terms, dim=-1)
    correction = (4.0 * nu * nu - 1.0) / (8.0 * kappa)
    log_bessel_asymptotic = kappa - 0.5 * math.log(2.0 * math.pi * 1.0) - 0.5 * torch.log(kappa)
    log_bessel = torch.where(
        small,
        log_bessel_series,
        log_bessel_asymptotic + correction,
    )
    log_c = nu * torch.log(kappa) - (dimension / 2.0) * math.log(2.0 * math.pi) - log_bessel
    log_uniform = (
        math.lgamma(dimension / 2.0)
        - math.log(2.0)
        - (dimension / 2.0) * math.log(math.pi)
    )
    return (log_c - log_uniform).to(dtype=concentration.dtype)


def directional_mixture_loss(
    prediction: SemanticTransportPrediction,
    target_direction: Tensor,
    positive_targets: Tensor,
    negative_embedding: Tensor,
    *,
    margin: float = 0.2,
    ranking_weight: float = 1.0,
    direction_nll_weight: float = 1.0,
    assignment_temperature: float = 0.05,
) -> DirectionMixtureLoss:
    """Mo-vMF objective over tangent transport directions, not photo endpoints."""
    if prediction.concentrations is None:
        raise ValueError("directional_mixture_loss requires predicted concentrations")
    directions = prediction.directions
    b, k, d = directions.shape
    _validate_batch_features("target_direction", target_direction, 2)
    if target_direction.shape != (b, d):
        raise ValueError("target_direction must have shape [B,D]")
    if not math.isfinite(ranking_weight) or ranking_weight < 0:
        raise ValueError("ranking_weight must be finite and non-negative")
    if not math.isfinite(direction_nll_weight) or direction_nll_weight < 0:
        raise ValueError("direction_nll_weight must be finite and non-negative")
    if ranking_weight == 0 and direction_nll_weight == 0:
        raise ValueError("at least one directional mixture loss weight is positive")
    if not math.isfinite(assignment_temperature) or assignment_temperature <= 0:
        raise ValueError("assignment_temperature must be finite and positive")
    if prediction.concentrations.shape != (b, k):
        raise ValueError("concentrations must have shape [B,K]")
    directions = _safe_normalize(directions)
    target = _safe_normalize(target_direction.detach())
    if prediction.gate_logits is None:
        log_pi = directions.new_full((b, k), -math.log(k))
    else:
        log_pi = prediction.gate_logits.log_softmax(dim=-1)
    concentrations = prediction.concentrations
    log_normalizer = _relative_log_vmf_normalizer(
        concentrations,
        dimension=d - 1,
    )
    cosine = (directions * target[:, None, :]).sum(dim=-1)
    component_log_likelihood = log_pi + log_normalizer + concentrations * cosine
    positive_log_likelihood = torch.logsumexp(component_log_likelihood, dim=-1)
    direction_nll = -positive_log_likelihood.mean()
    posterior = component_log_likelihood.softmax(dim=-1)
    ranking = transport_multi_hypothesis_ranking_loss(
        prediction.q_hypotheses,
        prediction.gate_logits,
        positive_targets,
        negative_embedding,
        margin=margin,
        temperature=assignment_temperature,
    )
    total = direction_nll_weight * direction_nll + ranking_weight * ranking
    return DirectionMixtureLoss(
        total=total,
        direction_nll=direction_nll,
        ranking=ranking,
        posterior_responsibilities=posterior,
        mixture_probabilities=log_pi.exp(),
        positive_direction_cosine=cosine,
        mean_kappa=concentrations.mean(),
    )


def deterministic_direction_mixture_loss(
    prediction: SemanticTransportPrediction,
    target_direction: Tensor,
    positive_targets: Tensor,
    negative_embedding: Tensor,
    *,
    margin: float = 0.2,
    ranking_weight: float = 1.0,
    direction_weight: float = 1.0,
    assignment_temperature: float = 0.05,
) -> DirectionMixtureLoss:
    """Matched K>1 control without concentrations or vMF likelihood."""
    directions = prediction.directions
    b, k, d = directions.shape
    if target_direction.shape != (b, d):
        raise ValueError("target_direction must have shape [B,D]")
    if not math.isfinite(assignment_temperature) or assignment_temperature <= 0:
        raise ValueError("assignment_temperature must be finite and positive")
    if prediction.gate_logits is None:
        log_pi = directions.new_full((b, k), -math.log(k))
    else:
        log_pi = prediction.gate_logits.log_softmax(dim=-1)
    directions = _safe_normalize(directions)
    target = _safe_normalize(target_direction.detach())
    cosine = (directions * target[:, None, :]).sum(dim=-1)
    responsibilities = (
        log_pi + cosine / assignment_temperature
    ).softmax(dim=-1)
    direction_loss = 1.0 - (responsibilities * cosine).sum(dim=-1).mean()
    ranking = transport_multi_hypothesis_ranking_loss(
        prediction.q_hypotheses,
        prediction.gate_logits,
        positive_targets,
        negative_embedding,
        margin=margin,
        temperature=assignment_temperature,
    )
    total = direction_weight * direction_loss + ranking_weight * ranking
    return DirectionMixtureLoss(
        total=total,
        direction_nll=direction_loss,
        ranking=ranking,
        posterior_responsibilities=responsibilities,
        mixture_probabilities=log_pi.exp(),
        positive_direction_cosine=cosine,
        mean_kappa=cosine.new_zeros(()),
    )


def barycentric_transport_query(
    q_hypotheses: Tensor,
    gate_logits: Tensor | None,
) -> Tensor:
    _validate_batch_features("q_hypotheses", q_hypotheses, 3)
    b, k, _ = q_hypotheses.shape
    if gate_logits is None:
        weights = q_hypotheses.new_full((b, k), 1.0 / k)
    else:
        if gate_logits.shape != (b, k):
            raise ValueError("gate_logits must have shape [B,K]")
        weights = gate_logits.softmax(dim=-1)
    return _safe_normalize((weights[..., None] * q_hypotheses).sum(dim=1))


def transport_gallery_scores(
    q_hypotheses: Tensor,
    gate_logits: Tensor | None,
    gallery_embeddings: Tensor,
    *,
    mode: Literal["barycentric", "angular_logsumexp", "max"] = "angular_logsumexp",
    temperature: float = 0.07,
) -> Tensor:
    """Compute text/photo-free K>1 retrieval scores."""
    _validate_batch_features("q_hypotheses", q_hypotheses, 3)
    _validate_batch_features("gallery_embeddings", gallery_embeddings, 2)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    b, k, d = q_hypotheses.shape
    if gallery_embeddings.shape[1] != d:
        raise ValueError("gallery and hypothesis dimensions must match")
    gallery = _safe_normalize(gallery_embeddings)
    if mode == "barycentric":
        return barycentric_transport_query(q_hypotheses, gate_logits) @ gallery.T
    normalized_hypotheses = _safe_normalize(q_hypotheses)
    component_scores = (
        normalized_hypotheses.reshape(b * k, d) @ gallery.T
    ).reshape(b, k, gallery.shape[0])
    if mode == "max":
        return component_scores.max(dim=1).values
    if mode != "angular_logsumexp":
        raise ValueError(f"Unknown transport score mode: {mode!r}")
    if gate_logits is None:
        log_pi = q_hypotheses.new_full((b, k), -math.log(k))
    else:
        if gate_logits.shape != (b, k):
            raise ValueError("gate_logits must have shape [B,K]")
        log_pi = gate_logits.log_softmax(dim=-1)
    return temperature * torch.logsumexp(
        log_pi[:, :, None] + component_scores / temperature,
        dim=1,
    )
