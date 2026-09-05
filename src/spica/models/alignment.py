"""Class-conditional distribution alignment on the unit-sphere embedding space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

AlignmentGeometry = Literal["log_map", "chordal"]
AlignmentAnchor = Literal["text", "photo_mean"]


@dataclass(frozen=True, slots=True)
class AlignmentLoss:
    """Reduced alignment losses and the number of usable class groups."""

    total: Tensor
    mean: Tensor
    covariance: Tensor
    num_classes: int
    num_sketches: int
    num_photos: int


def _validate_points(name: str, points: Tensor) -> None:
    if points.ndim < 1 or points.shape[-1] < 1:
        raise ValueError(f"{name} must have shape [..., dimension]")
    if not points.is_floating_point():
        raise TypeError(f"{name} must be floating-point")


def _normalize(points: Tensor) -> Tensor:
    return F.normalize(points, dim=-1, eps=1e-12)


def spherical_log_map(anchor: Tensor, points: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Map unit-sphere points to the tangent plane at ``anchor``.

    Uses the numerically stable formulation:
        a = dot(t, z); v = z - a*t; theta = atan2(||v||, a)
        Log_t(z) = (theta / ||v||) * v

    This avoids the acos/sin ratio singularity at both the anchor (a~1)
    and antipode (a~-1). At the anchor itself the result is exactly zero.
    Antipodal points produce a finite-magnitude tangent with direction
    determined by the residual v.

    ``anchor`` may be one vector or broadcast over ``points``.  The returned
    vectors have the same leading shape as ``points`` and are orthogonal to the
    normalized anchor up to floating-point error.
    """
    _validate_points("anchor", anchor)
    _validate_points("points", points)
    if anchor.shape[-1] != points.shape[-1]:
        raise ValueError("anchor and points must have the same embedding dimension")
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must be in (0, 1)")

    base = _normalize(anchor)
    values = _normalize(points)
    # Compute in at least float32 for numerical stability
    base_f = base.float() if base.dtype != torch.float64 else base
    values_f = values.float() if values.dtype != torch.float64 else values

    a = (values_f * base_f).sum(dim=-1)  # cos(theta), unclamped
    v = values_f - a.unsqueeze(-1) * base_f  # component orthogonal to anchor
    v_norm = v.norm(dim=-1)

    # atan2 gives correct angle in [0, pi] without acos singularity
    theta = torch.atan2(v_norm, a)

    # theta/v_norm -> 1.0 as v_norm->0 (small-angle limit preserves gradient)
    # At exact anchor: v_norm=0, theta=0, result=0 (correct)
    factor = theta / v_norm.clamp_min(eps)

    tangent = v * factor.unsqueeze(-1)

    # Exact anchor case: return clean zeros
    at_anchor = v_norm.le(eps)
    tangent = torch.where(at_anchor.unsqueeze(-1), torch.zeros_like(tangent), tangent)

    return tangent.to(values.dtype)


def spherical_exp_map(anchor: Tensor, tangent: Tensor) -> Tensor:
    """Map tangent vectors back to the unit sphere at ``anchor``."""
    _validate_points("anchor", anchor)
    _validate_points("tangent", tangent)
    if anchor.shape[-1] != tangent.shape[-1]:
        raise ValueError("anchor and tangent must have the same embedding dimension")
    base = _normalize(anchor)
    norm = tangent.norm(dim=-1)
    direction = tangent / norm.clamp_min(1e-12).unsqueeze(-1)
    return _normalize(
        torch.cos(norm).unsqueeze(-1) * base
        + torch.sin(norm).unsqueeze(-1) * direction
    )


def tangent_moments(values: Tensor) -> tuple[Tensor, Tensor]:
    """Return population mean and covariance for at least two observations."""
    _validate_points("values", values)
    if values.ndim != 2:
        raise ValueError("values must have shape [observations, dimension]")
    if values.shape[0] < 2:
        raise ValueError("at least two observations are required for covariance")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    return mean, covariance


def _lookup_anchors(
    labels: Tensor, text_embeddings: Tensor, text_labels: Tensor
) -> Tensor:
    if labels.ndim != 1 or text_labels.ndim != 1:
        raise ValueError("labels and text_labels must be one-dimensional")
    if text_embeddings.ndim != 2 or text_embeddings.shape[0] != text_labels.shape[0]:
        raise ValueError("text embeddings and labels have incompatible shapes")
    if text_labels.numel() == 0 or torch.unique(text_labels).numel() != text_labels.numel():
        raise ValueError("text labels must be non-empty and unique")
    positions = {int(label): index for index, label in enumerate(text_labels.tolist())}
    try:
        indices = [positions[int(label)] for label in labels.tolist()]
    except KeyError as error:
        raise ValueError(f"label {error.args[0]} is missing from the text bank") from error
    return text_embeddings[torch.tensor(indices, device=text_embeddings.device)]


def class_conditional_alignment_loss(
    sketch_embeddings: Tensor,
    positive_photo_embeddings: Tensor,
    labels: Tensor,
    *,
    text_embeddings: Tensor | None = None,
    text_labels: Tensor | None = None,
    mean_weight: float = 1.0,
    covariance_weight: float = 1.0,
    geometry: AlignmentGeometry = "log_map",
    anchor: AlignmentAnchor = "text",
) -> AlignmentLoss:
    """Match sketch/photo class moments, using only the positive train batch.

    The source is the sketch distribution in each matched class group.  The
    photo distribution is formed from that group's positive photos and is
    detached before becoming a target.  Thus the loss cannot update the target
    branch or a text anchor, while the ordinary ranking loss can still update
    both visual prompts.
    """
    _validate_points("sketch_embeddings", sketch_embeddings)
    if positive_photo_embeddings.ndim != 3:
        raise ValueError("positive_photo_embeddings must have shape [batch, positives, dim]")
    _validate_points("positive_photo_embeddings", positive_photo_embeddings)
    if positive_photo_embeddings.shape[-1] != sketch_embeddings.shape[-1]:
        raise ValueError("sketch and photo embeddings must have the same dimension")
    if labels.ndim != 1 or labels.shape[0] != sketch_embeddings.shape[0]:
        raise ValueError("labels must align with sketch_embeddings")
    if positive_photo_embeddings.shape[0] != labels.shape[0]:
        raise ValueError("positive photos must align with labels")
    if geometry not in {"log_map", "chordal"}:
        raise ValueError("geometry must be 'log_map' or 'chordal'")
    if anchor not in {"text", "photo_mean"}:
        raise ValueError("anchor must be 'text' or 'photo_mean'")
    if not torch.isfinite(sketch_embeddings).all() or not torch.isfinite(
        positive_photo_embeddings
    ).all():
        raise ValueError("alignment inputs must be finite")
    if mean_weight < 0 or covariance_weight < 0:
        raise ValueError("alignment weights must be non-negative")

    zero = sketch_embeddings.new_zeros(())
    if mean_weight == 0 and covariance_weight == 0:
        return AlignmentLoss(zero, zero, zero, 0, 0, 0)
    if anchor == "text" and (text_embeddings is None or text_labels is None):
        raise ValueError("text embeddings and labels are required for a text anchor")
    if anchor == "text":
        assert text_embeddings is not None
        assert text_labels is not None
        text_embeddings = _normalize(text_embeddings).detach()

    sketch_values = _normalize(sketch_embeddings)
    photo_values = _normalize(positive_photo_embeddings)
    mean_losses: list[Tensor] = []
    covariance_losses: list[Tensor] = []
    total_sketches = 0
    total_photos = 0
    unique_labels = torch.unique(labels, sorted=True)
    for class_label in unique_labels:
        class_mask = labels == class_label
        class_sketches = sketch_values[class_mask]
        class_photos = photo_values[class_mask].reshape(-1, photo_values.shape[-1])
        if class_sketches.shape[0] < 2:
            continue
        if anchor == "text":
            assert text_embeddings is not None
            assert text_labels is not None
            class_anchor = _lookup_anchors(
                class_label.reshape(1), text_embeddings, text_labels
            )[0]
        else:
            class_anchor = _normalize(class_photos.mean(dim=0)).detach()

        if geometry == "log_map":
            source = spherical_log_map(class_anchor, class_sketches)
            target = spherical_log_map(class_anchor, class_photos).detach()
        else:
            source = class_sketches
            target = class_photos.detach()
        source_mean, source_covariance = tangent_moments(source)
        target_mean, target_covariance = tangent_moments(target)
        mean_losses.append((source_mean - target_mean).square().sum())
        covariance_losses.append((source_covariance - target_covariance).square().sum())
        total_sketches += int(class_sketches.shape[0])
        total_photos += int(class_photos.shape[0])

    if not mean_losses:
        return AlignmentLoss(zero, zero, zero, 0, 0, 0)
    mean_loss = torch.stack(mean_losses).mean()
    covariance_loss = torch.stack(covariance_losses).mean()
    total = mean_weight * mean_loss + covariance_weight * covariance_loss
    return AlignmentLoss(
        total=total,
        mean=mean_loss,
        covariance=covariance_loss,
        num_classes=len(mean_losses),
        num_sketches=total_sketches,
        num_photos=total_photos,
    )


def class_conditional_geometry_diagnostics(
    sketch_embeddings: Tensor,
    photo_embeddings: Tensor,
    labels: Tensor,
    *,
    photo_labels: Tensor | None = None,
    text_embeddings: Tensor | None = None,
    text_labels: Tensor | None = None,
) -> dict[str, float | int | None]:
    """Summarize class-wise tangent moments without constructing gradients."""
    _validate_points("sketch_embeddings", sketch_embeddings)
    _validate_points("photo_embeddings", photo_embeddings)
    if labels.ndim != 1 or labels.shape[0] != sketch_embeddings.shape[0]:
        raise ValueError("labels must align with sketch_embeddings")
    if photo_labels is None or photo_labels.ndim != 1 or photo_labels.shape[0] != photo_embeddings.shape[0]:
        raise ValueError("photo_labels must align with photo_embeddings")
    if text_embeddings is None or text_labels is None:
        raise ValueError("text embeddings and labels are required for diagnostics")
    text_embeddings = _normalize(text_embeddings).detach()
    sketch_values = _normalize(sketch_embeddings)
    photo_values = _normalize(photo_embeddings)
    mean_distances: list[Tensor] = []
    covariance_norms: list[Tensor] = []
    sketch_angles: list[Tensor] = []
    photo_angles: list[Tensor] = []
    orthogonality: list[Tensor] = []
    class_count = 0
    for class_label in torch.unique(labels, sorted=True):
        sketches = sketch_values[labels == class_label]
        photos = photo_values[photo_labels == class_label]
        if sketches.shape[0] < 2 or photos.shape[0] < 2:
            continue
        anchor = _lookup_anchors(
            class_label.reshape(1), text_embeddings, text_labels
        )[0]
        sketch_log = spherical_log_map(anchor, sketches)
        photo_log = spherical_log_map(anchor, photos)
        sketch_mean, sketch_covariance = tangent_moments(sketch_log)
        photo_mean, photo_covariance = tangent_moments(photo_log)
        mean_distances.append((sketch_mean - photo_mean).norm())
        covariance_norms.append((sketch_covariance - photo_covariance).norm())
        sketch_angles.append(sketch_log.norm(dim=-1).mean())
        photo_angles.append(photo_log.norm(dim=-1).mean())
        orthogonality.append((sketch_log * _normalize(anchor)).sum(dim=-1).abs().mean())
        class_count += 1
    if not class_count:
        return {
            "class_count": 0,
            "mean_distance": None,
            "covariance_frobenius_distance": None,
            "sketch_log_angle": None,
            "photo_log_angle": None,
            "sketch_anchor_orthogonality": None,
        }
    return {
        "class_count": class_count,
        "mean_distance": float(torch.stack(mean_distances).mean().item()),
        "covariance_frobenius_distance": float(torch.stack(covariance_norms).mean().item()),
        "sketch_log_angle": float(torch.stack(sketch_angles).mean().item()),
        "photo_log_angle": float(torch.stack(photo_angles).mean().item()),
        "sketch_anchor_orthogonality": float(torch.stack(orthogonality).mean().item()),
    }
