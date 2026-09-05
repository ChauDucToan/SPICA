"""Class-conditional distribution alignment on the unit-sphere embedding space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

AlignmentGeometry = Literal["log_map", "chordal"]
AlignmentAnchor = Literal["text", "photo_mean"]
AlignmentTargetGradient = Literal["detached", "symmetric"]


@dataclass(frozen=True, slots=True)
class AlignmentLoss:
    """Reduced alignment losses and the number of usable class groups."""

    total: Tensor
    mean: Tensor
    covariance: Tensor
    num_classes: int
    num_sketches: int
    num_photos: int
    invalid_sketches: int = 0
    invalid_photos: int = 0
    skipped_classes: int = 0


def _validate_points(name: str, points: Tensor) -> None:
    if points.ndim < 1 or points.shape[-1] < 1:
        raise ValueError(f"{name} must have shape [..., dimension]")
    if not points.is_floating_point():
        raise TypeError(f"{name} must be floating-point")


def _normalize(points: Tensor) -> Tensor:
    """Normalize geometry inputs in float32 or float64, never lower precision."""
    work_dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    values = points.to(work_dtype)
    if not torch.isfinite(values).all():
        raise ValueError("geometry inputs must be finite")
    if (values.norm(dim=-1) <= 1e-12).any():
        raise ValueError("geometry inputs must have non-zero norm")
    return F.normalize(values, dim=-1, eps=1e-12)


def spherical_log_map(
    anchor: Tensor,
    points: Tensor,
    *,
    eps: float = 1e-6,
    antipode_eps: float = 1e-4,
    strict_antipode: bool = False,
    return_validity: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    """Map unit-sphere points to the tangent plane at ``anchor``.

    Uses the numerically stable formulation:
        a = dot(t, z); v = z - a*t; theta = atan2(||v||, a)
        Log_t(z) = (theta / ||v||) * v

    ``atan2`` improves numerical stability over the acos/sin ratio, but does
    not remove the geometric non-uniqueness at the antipode.

    Near the anchor (a~1, ||v||~0), ``theta / ||v|| -> 1`` via the
    small-angle limit, so the output is the residual ``v`` and local gradient
    is preserved. At the exact anchor the correct output is zero.

    Near the antipode (a~-1, ||v||~0), the sample is invalid because the Log
    map has no unique direction. When ``||v|| <= antipode_eps`` and ``a < 0``,
    strict mode raises. With ``return_validity=True`` the function returns a
    validity mask so callers can exclude and count invalid samples. Otherwise
    invalid input raises rather than silently returning a zero tangent.

    ``anchor`` may be one vector or broadcast over ``points``. With
    ``return_validity=True`` the return value is ``(tangent, valid_mask)``;
    otherwise it is the tangent tensor.
    """
    _validate_points("anchor", anchor)
    _validate_points("points", points)
    if anchor.shape[-1] != points.shape[-1]:
        raise ValueError("anchor and points must have the same embedding dimension")
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must be in (0, 1)")
    if antipode_eps <= 0 or antipode_eps >= 1:
        raise ValueError("antipode_eps must be in (0, 1)")
    if antipode_eps < eps:
        raise ValueError("antipode_eps must be at least eps")
    if not isinstance(strict_antipode, bool):
        raise TypeError("strict_antipode must be a bool")
    if not isinstance(return_validity, bool):
        raise TypeError("return_validity must be a bool")
    if not torch.isfinite(anchor).all():
        raise ValueError("anchor contains non-finite values")
    if not torch.isfinite(points).all():
        raise ValueError("points contains non-finite values")

    # Cast to at least float32 BEFORE normalization and geometry ops.
    work_dtype = (
        torch.float64
        if anchor.dtype == torch.float64 or points.dtype == torch.float64
        else torch.float32
    )
    anchor_work = anchor.to(work_dtype)
    points_work = points.to(work_dtype)
    if (anchor_work.norm(dim=-1) <= 1e-12).any():
        raise ValueError("anchor must have non-zero norm")
    if (points_work.norm(dim=-1) <= 1e-12).any():
        raise ValueError("points must have non-zero norm")
    base = F.normalize(anchor_work, dim=-1, eps=1e-12)
    values = F.normalize(points_work, dim=-1, eps=1e-12)

    a = (values * base).sum(dim=-1)  # cos(theta), unclamped
    v = values - a.unsqueeze(-1) * base  # component orthogonal to anchor
    v_norm = v.norm(dim=-1)

    # Both residual norm and dot sign are required to classify the region.
    near_anchor = (v_norm <= eps) & (a >= 0)
    is_antipodal = (v_norm <= antipode_eps) & (a < 0)
    if strict_antipode and is_antipodal.any():
        n_bad = int(is_antipodal.sum().item())
        raise ValueError(
            f"{n_bad} near-antipodal sample(s) detected; Log map is non-unique."
        )

    # Clamp the divisor so even the unselected branch is finite and creates no
    # NaN gradient. The selected near-anchor branch uses the exact limit 1.
    theta = torch.atan2(v_norm, a)
    regular_factor = theta / v_norm.clamp_min(eps)
    factor = torch.where(near_anchor, torch.ones_like(regular_factor), regular_factor)
    tangent = v * factor.unsqueeze(-1)

    valid_mask = ~is_antipodal
    if is_antipodal.any():
        # This value is deliberately unusable; callers must use the mask.
        tangent = torch.where(
            is_antipodal.unsqueeze(-1), torch.zeros_like(tangent), tangent
        )
        if not return_validity:
            raise ValueError(
                "near-antipodal samples detected; call with return_validity=True "
                "to exclude and count them"
            )

    if return_validity:
        return tangent, valid_mask
    return tangent


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


def tangent_moments(
    values: Tensor, valid_mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Return sample mean and covariance (Bessel-corrected, n-1 denominator).

    When ``valid_mask`` is provided, only valid rows participate. Groups
    with fewer than 2 valid observations raise an error.
    """
    _validate_points("values", values)
    if values.ndim != 2:
        raise ValueError("values must have shape [observations, dimension]")
    if valid_mask is not None:
        if valid_mask.ndim != 1 or valid_mask.shape[0] != values.shape[0]:
            raise ValueError("valid_mask must be one-dimensional and match values")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        values = values[valid_mask]
    if not torch.isfinite(values).all():
        raise ValueError("moment values must be finite")
    n = values.shape[0]
    if n < 2:
        raise ValueError(f"at least 2 valid observations required for covariance, got {n}")
    mean = values.mean(dim=0)
    centered = values - mean
    # ponytail: Bessel-corrected sample covariance (n-1). Finite-sample noise
    # remains; this only removes the (n-1)/n bias under IID assumption.
    covariance = centered.T @ centered / (n - 1)
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
    target_gradient: AlignmentTargetGradient = "detached",
) -> AlignmentLoss:
    """Match sketch/photo class moments using the positive train batch.

    ``target_gradient='detached'`` is the legacy policy: alignment updates only
    sketch prompts. ``'symmetric'`` lets alignment update both sketch and photo
    prompts. Text anchors are always detached; ranking gradients are unaffected
    in either policy. Invalid near-antipodal samples are excluded and counted.
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
    if target_gradient not in {"detached", "symmetric"}:
        raise ValueError("target_gradient must be 'detached' or 'symmetric'")
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
    invalid_sketches = 0
    invalid_photos = 0
    skipped_classes = 0
    unique_labels = torch.unique(labels, sorted=True)
    for class_label in unique_labels:
        class_mask = labels == class_label
        class_sketches = sketch_values[class_mask]
        class_photos = photo_values[class_mask].reshape(-1, photo_values.shape[-1])
        if class_sketches.shape[0] < 2:
            skipped_classes += 1
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
            source, source_valid = spherical_log_map(
                class_anchor, class_sketches, return_validity=True
            )
            target_raw, target_valid = spherical_log_map(
                class_anchor, class_photos, return_validity=True
            )
            invalid_sketches += int((~source_valid).sum().item())
            invalid_photos += int((~target_valid).sum().item())
            target = (
                target_raw.detach()
                if target_gradient == "detached"
                else target_raw
            )
        else:
            source = class_sketches
            target = (
                class_photos.detach()
                if target_gradient == "detached"
                else class_photos
            )
            source_valid = torch.ones(
                class_sketches.shape[0], dtype=torch.bool, device=class_sketches.device
            )
            target_valid = torch.ones(
                class_photos.shape[0], dtype=torch.bool, device=class_photos.device
            )
        if int(source_valid.sum()) < 2 or int(target_valid.sum()) < 2:
            skipped_classes += 1
            continue
        source_mean, source_covariance = tangent_moments(source, source_valid)
        target_mean, target_covariance = tangent_moments(target, target_valid)
        mean_losses.append((source_mean - target_mean).square().sum())
        covariance_losses.append((source_covariance - target_covariance).square().sum())
        total_sketches += int(source_valid.sum().item())
        total_photos += int(target_valid.sum().item())

    if not mean_losses:
        return AlignmentLoss(
            zero,
            zero,
            zero,
            0,
            total_sketches,
            total_photos,
            invalid_sketches,
            invalid_photos,
            skipped_classes,
        )
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
        invalid_sketches=invalid_sketches,
        invalid_photos=invalid_photos,
        skipped_classes=skipped_classes,
    )


def class_conditional_geometry_diagnostics(
    sketch_embeddings: Tensor,
    photo_embeddings: Tensor,
    labels: Tensor,
    *,
    photo_labels: Tensor | None = None,
    text_embeddings: Tensor | None = None,
    text_labels: Tensor | None = None,
) -> dict[str, float | int | str | None]:
    """Summarize corrected class-wise tangent moments without gradients."""
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
    sketch_ranks: list[int] = []
    photo_ranks: list[int] = []
    invalid_sketch_count = 0
    invalid_photo_count = 0
    skipped_class_count = 0
    valid_sketch_count = 0
    valid_photo_count = 0
    class_count = 0
    anchor_norm = None
    for class_label in torch.unique(labels, sorted=True):
        sketches = sketch_values[labels == class_label]
        photos = photo_values[photo_labels == class_label]
        if sketches.shape[0] < 2 or photos.shape[0] < 2:
            skipped_class_count += 1
            continue
        anchor = _lookup_anchors(
            class_label.reshape(1), text_embeddings, text_labels
        )[0]
        sketch_log, sketch_valid = spherical_log_map(
            anchor, sketches, return_validity=True
        )
        photo_log, photo_valid = spherical_log_map(
            anchor, photos, return_validity=True
        )
        invalid_sketch_count += int((~sketch_valid).sum().item())
        invalid_photo_count += int((~photo_valid).sum().item())
        if int(sketch_valid.sum().item()) < 2 or int(photo_valid.sum().item()) < 2:
            skipped_class_count += 1
            continue
        sketch_mean, sketch_covariance = tangent_moments(sketch_log, sketch_valid)
        photo_mean, photo_covariance = tangent_moments(photo_log, photo_valid)
        valid_sketch = sketch_log[sketch_valid]
        valid_photo = photo_log[photo_valid]
        mean_distances.append((sketch_mean - photo_mean).norm())
        covariance_norms.append((sketch_covariance - photo_covariance).norm())
        sketch_angles.append(valid_sketch.norm(dim=-1).mean())
        photo_angles.append(valid_photo.norm(dim=-1).mean())
        anchor_norm = _normalize(anchor)
        orthogonality.append((valid_sketch * anchor_norm).sum(dim=-1).abs().mean())
        sketch_ranks.append(
            int(torch.linalg.matrix_rank(valid_sketch - sketch_mean).item())
        )
        photo_ranks.append(
            int(torch.linalg.matrix_rank(valid_photo - photo_mean).item())
        )
        valid_sketch_count += int(sketch_valid.sum().item())
        valid_photo_count += int(photo_valid.sum().item())
        class_count += 1
    base_result: dict[str, float | int | None] = {
        "class_count": class_count,
        "valid_sketch_count": valid_sketch_count,
        "valid_photo_count": valid_photo_count,
        "invalid_sketch_count": invalid_sketch_count,
        "invalid_photo_count": invalid_photo_count,
        "skipped_class_count": skipped_class_count,
        "estimator": "sample_covariance_n_minus_1",
        "covariance_distance": "frobenius_norm",
        "sketch_covariance_rank_mean": None,
        "sketch_covariance_rank_max": None,
        "photo_covariance_rank_mean": None,
        "photo_covariance_rank_max": None,
        "mean_distance": None,
        "covariance_frobenius_distance": None,
        "sketch_log_angle": None,
        "photo_log_angle": None,
        "sketch_anchor_orthogonality": None,
    }
    if not class_count:
        return base_result
    base_result.update(
        {
            "mean_distance": float(torch.stack(mean_distances).mean().item()),
            "covariance_frobenius_distance": float(torch.stack(covariance_norms).mean().item()),
            "sketch_log_angle": float(torch.stack(sketch_angles).mean().item()),
            "photo_log_angle": float(torch.stack(photo_angles).mean().item()),
            "sketch_anchor_orthogonality": float(torch.stack(orthogonality).mean().item()),
            "sketch_covariance_rank_mean": sum(sketch_ranks) / len(sketch_ranks),
            "sketch_covariance_rank_max": max(sketch_ranks),
            "photo_covariance_rank_mean": sum(photo_ranks) / len(photo_ranks),
            "photo_covariance_rank_max": max(photo_ranks),
        }
    )
    return base_result
