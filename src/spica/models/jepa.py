"""Predictive cross-modal JEPA components for SPICA.

The module intentionally has no text or photo inputs in the predictor API.  A
sketch image is mapped to an internal context representation, predicted into a
512-D photo-semantic latent, and normalized only at the retrieval boundary.
Frozen CLIP text embeddings are consumed by a training-only classification loss
in this module, never by :class:`SketchPhotoJepa`.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .clip import TrainableSketchContextEncoder


@dataclass(frozen=True, slots=True)
class JepaPrediction:
    """Intermediate and retrieval representations produced from a sketch."""

    h: Tensor
    u: Tensor
    q: Tensor


class SpicaJepaPredictor(nn.Module):
    """Predict a photo-semantic latent from a sketch context latent."""

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_norm = nn.LayerNorm(embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Start from the CLIP-initialized context direction while giving the
        # predictor a well-conditioned residual path to learn immediately.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context_latent: Tensor) -> JepaPrediction:
        if context_latent.ndim != 2:
            raise ValueError(
                "context_latent must have shape [batch, embedding_dim], "
                f"got {tuple(context_latent.shape)}"
            )
        if context_latent.shape[1] != self.embedding_dim:
            raise ValueError(
                "context_latent embedding dimension must equal embedding_dim, "
                f"got {context_latent.shape[1]} and {self.embedding_dim}"
            )
        if not context_latent.is_floating_point():
            raise TypeError("context_latent must be floating-point")
        if not torch.isfinite(context_latent).all().item():
            raise ValueError("context_latent must contain only finite values")
        u = context_latent + self.network(self.context_norm(context_latent))
        q = F.normalize(u, dim=-1)
        return JepaPrediction(h=context_latent, u=u, q=q)


class SketchPhotoJepa(nn.Module):
    """Raw-sketch to photo-semantic query model.

    The only forward input is a batch of sketch images.  Text banks and photo
    targets are deliberately kept outside this module and are used solely by
    training losses.
    """

    def __init__(
        self,
        sketch_context_encoder: TrainableSketchContextEncoder,
        predictor: SpicaJepaPredictor,
    ) -> None:
        super().__init__()
        if sketch_context_encoder.embedding_dim != predictor.embedding_dim:
            raise ValueError(
                "Sketch context and predictor dimensions must match, got "
                f"{sketch_context_encoder.embedding_dim} and {predictor.embedding_dim}"
            )
        self.sketch_context_encoder = sketch_context_encoder
        self.predictor = predictor
        self.embedding_dim = predictor.embedding_dim

    def forward(self, sketch_images: Tensor) -> JepaPrediction:
        context_latent = self.sketch_context_encoder(sketch_images)
        return self.predictor(context_latent)

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
    def predictor_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.predictor.parameters())

    @property
    def sketch_encoder_trainable_parameter_count(self) -> int:
        return self.sketch_context_encoder.trainable_parameter_count


def photo_semantic_target(photo_embeddings: Tensor) -> Tensor:
    """Build a stop-gradient normalized centroid from ``[B, M, D]`` photos."""
    if photo_embeddings.ndim != 3:
        raise ValueError(
            "photo_embeddings must have shape [batch, num_positives, dimension], "
            f"got {tuple(photo_embeddings.shape)}"
        )
    if photo_embeddings.shape[1] <= 0:
        raise ValueError("photo_embeddings must contain at least one positive")
    if not photo_embeddings.is_floating_point():
        raise TypeError("photo_embeddings must be floating-point")
    if not torch.isfinite(photo_embeddings).all().item():
        raise ValueError("photo_embeddings must contain only finite values")
    normalized = F.normalize(photo_embeddings, dim=-1)
    # Targets are externally stable frozen-CLIP quantities.  Detaching here
    # makes that invariant explicit even if a caller did not use no_grad.
    return F.normalize(normalized.mean(dim=1), dim=-1).detach()


def _validate_query_target_pair(
    queries: Tensor,
    targets: Tensor,
    *,
    name: str = "targets",
) -> None:
    if queries.ndim != 2 or targets.ndim != 2:
        raise ValueError(
            f"queries and {name} must both have shape [batch, dimension]"
        )
    if queries.shape != targets.shape:
        raise ValueError(
            f"queries and {name} must have matching shapes, got "
            f"{tuple(queries.shape)} and {tuple(targets.shape)}"
        )
    for value_name, value in (("queries", queries), (name, targets)):
        if not value.is_floating_point():
            raise TypeError(f"{value_name} must be floating-point")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{value_name} must contain only finite values")


def jepa_prediction_loss(queries: Tensor, target: Tensor) -> Tensor:
    """Cosine prediction loss against a detached photo-semantic target."""
    _validate_query_target_pair(queries, target, name="target")
    return 1.0 - (
        F.normalize(queries, dim=-1) * F.normalize(target.detach(), dim=-1)
    ).sum(dim=-1).mean()


def jepa_ranking_loss(
    queries: Tensor,
    positive_target: Tensor,
    negative_embeddings: Tensor,
    *,
    margin: float = 0.2,
) -> Tensor:
    """Repository-native softplus ranking loss in the frozen photo space."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    _validate_query_target_pair(queries, positive_target, name="positive_target")
    _validate_query_target_pair(queries, negative_embeddings, name="negative_embeddings")
    query = F.normalize(queries, dim=-1)
    positive = F.normalize(positive_target.detach(), dim=-1)
    negative = F.normalize(negative_embeddings.detach(), dim=-1)
    positive_score = (query * positive).sum(dim=-1)
    negative_score = (query * negative).sum(dim=-1)
    return F.softplus(margin - positive_score + negative_score).mean()


@dataclass(frozen=True, slots=True)
class VicRegLoss:
    variance: Tensor
    covariance: Tensor
    total: Tensor


def vicreg_latent_regularization(
    latent: Tensor,
    *,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> VicRegLoss:
    """VICReg-style variance/covariance penalty on an internal latent.

    This function intentionally accepts ``h`` or ``u`` and is not intended for
    the normalized retrieval query ``q``.  Statistics are estimated across the
    current training batch.
    """
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("latent must have shape [batch >= 2, dimension]")
    if not latent.is_floating_point() or not torch.isfinite(latent).all().item():
        raise ValueError("latent must be finite floating-point values")
    for name, value in (
        ("variance_weight", variance_weight),
        ("covariance_weight", covariance_weight),
        ("target_std", target_std),
        ("eps", eps),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if target_std == 0:
        raise ValueError("target_std must be positive")
    centered = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    variance = F.relu(target_std - std).square().mean()
    covariance_matrix = centered.T @ centered / (latent.shape[0] - 1)
    off_diagonal = covariance_matrix - torch.diag_embed(
        torch.diagonal(covariance_matrix)
    )
    covariance = off_diagonal.square().mean()
    total = variance_weight * variance + covariance_weight * covariance
    return VicRegLoss(variance=variance, covariance=covariance, total=total)


class SignatureRegularizer(nn.Module):
    """A lightweight SIGReg-style random-projection signature penalty.

    Each batch is standardized only for this shape regularizer.  Fixed random
    unit projections compare empirical characteristic functions with a standard
    normal characteristic function over a small frequency grid.  It has no
    learnable parameters and is applied to an internal JEPA latent, never q.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        num_projections: int = 32,
        num_frequencies: int = 16,
        frequency_max: float = 5.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_projections <= 0 or num_frequencies <= 0:
            raise ValueError("num_projections and num_frequencies must be positive")
        if not math.isfinite(frequency_max) or frequency_max <= 0:
            raise ValueError("frequency_max must be finite and positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projections = torch.randn(
            num_projections,
            embedding_dim,
            generator=generator,
            dtype=torch.float32,
        )
        projections = F.normalize(projections, dim=-1)
        frequencies = torch.linspace(
            frequency_max / num_frequencies,
            frequency_max,
            num_frequencies,
            dtype=torch.float32,
        )
        self.register_buffer("projections", projections)
        self.register_buffer("frequencies", frequencies)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[0] < 2:
            raise ValueError("latent must have shape [batch >= 2, dimension]")
        if latent.shape[1] != self.projections.shape[1]:
            raise ValueError(
                "latent dimension does not match signature projections: "
                f"{latent.shape[1]} != {self.projections.shape[1]}"
            )
        if not latent.is_floating_point() or not torch.isfinite(latent).all().item():
            raise ValueError("latent must be finite floating-point values")
        centered = latent - latent.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False).clamp_min(1e-4)
        standardized = centered / std
        projected = standardized @ self.projections.to(
            device=latent.device, dtype=latent.dtype
        ).T
        frequencies = self.frequencies.to(device=latent.device, dtype=latent.dtype)
        arguments = projected[..., None] * frequencies
        empirical_real = arguments.cos().mean(dim=0)
        empirical_imag = arguments.sin().mean(dim=0)
        target_real = torch.exp(-0.5 * frequencies.square())
        return (
            (empirical_real - target_real[None, :]).square()
            + empirical_imag.square()
        ).mean()


def jepa_text_classification_loss(
    queries: Tensor,
    text_embeddings: Tensor,
    text_labels: Tensor,
    class_labels: Tensor,
    *,
    temperature: float = 0.07,
    detach_text: bool = True,
) -> tuple[Tensor, Tensor]:
    """Classify predicted queries with a frozen seen-class CLIP text bank.

    Returns ``(cross_entropy, logits)``.  By default only ``queries``
    participate in autograd; ``detach_text=False`` supports an explicitly
    trainable soft-prompt bank while keeping the predictor API text-free.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if queries.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("queries and text_embeddings must be two-dimensional")
    if queries.shape[1] != text_embeddings.shape[1]:
        raise ValueError("query and text embedding dimensions must match")
    if text_embeddings.shape[0] < 2:
        raise ValueError("text bank must contain at least two classes")
    if text_labels.ndim != 1 or text_labels.shape[0] != text_embeddings.shape[0]:
        raise ValueError("text_labels must contain one label per text embedding")
    if class_labels.ndim != 1 or class_labels.shape[0] != queries.shape[0]:
        raise ValueError("class_labels must contain one label per query")
    if text_labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("text_labels must be integer labels")
    if class_labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("class_labels must be integer labels")
    if not torch.equal(text_labels, text_labels.sort().values):
        raise ValueError("text_labels must be sorted for deterministic label lookup")
    positions = torch.searchsorted(text_labels.to(class_labels.device), class_labels)
    in_range = positions < text_labels.shape[0]
    if not in_range.all().item():
        raise ValueError("A training class label is missing from the text bank")
    observed = text_labels.to(class_labels.device)[positions]
    if not torch.equal(observed, class_labels):
        raise ValueError("A training class label is missing from the text bank")
    query = F.normalize(queries, dim=-1)
    bank_input = text_embeddings.detach() if detach_text else text_embeddings
    bank = F.normalize(bank_input, dim=-1).to(
        device=queries.device, dtype=queries.dtype
    )
    logits = query @ bank.T / temperature
    return F.cross_entropy(logits, positions), logits


def classification_accuracy(
    logits: Tensor,
    text_labels: Tensor,
    class_labels: Tensor,
) -> Tensor:
    if logits.ndim != 2 or text_labels.ndim != 1 or class_labels.ndim != 1:
        raise ValueError("logits and labels have invalid dimensions")
    if logits.shape != (class_labels.shape[0], text_labels.shape[0]):
        raise ValueError("logits shape does not match class and text labels")
    predicted = text_labels.to(logits.device)[logits.argmax(dim=-1)]
    return predicted.eq(class_labels.to(logits.device)).float().mean()


# Descriptive aliases used by experiment scripts and downstream callers.
CrossModalJepaRetriever = SketchPhotoJepa
SketchPhotoJepaPredictor = SpicaJepaPredictor

# Transport-family exports live in their own module so the previous full-vector
# JEPA implementation remains an untouched T0 control.  These aliases make the
# new public family discoverable from the historical model module as well.
from .transport import (  # noqa: E402, F401  (intentional compatibility exports)
    PredictiveSemanticTransport,
    SketchPhotoTransport,
    SketchPhotoTransportPredictor,
    SpicaPredictiveTransport,
    SpicaSemanticTransport,
)
