"""Evaluation, retrieval modes, and geometry probes for semantic transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.clip import FrozenVisualProjection
from ..models.transport import (
    SemanticTransportPrediction,
    SpicaPredictiveTransport,
    _relative_log_vmf_normalizer,
    fixed_origin_transport_target,
    photo_transport_target,
    transport_gallery_scores,
)
from .embeddings import EncodedRetrievalSet
from .jepa import feature_geometry, photo_target_alignment_diagnostics, semantic_query_diagnostics
from .metrics import (
    CategoryRetrievalEvaluation,
    CategoryRetrievalMetrics,
    _average_precision_from_relevance,
)

TransportScoreMode = Literal["barycentric", "angular_logsumexp", "max"]


def _effective_rank_from_matrix(values: Tensor) -> float:
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    denominator = energy.square().sum().clamp_min(1e-12)
    return float(energy.sum().square().div(denominator).item())


def hidden_space_compatibility(
    h_ref: Tensor,
    h_t: Tensor,
    projection: FrozenVisualProjection,
) -> dict[str, float]:
    """Compare adapted and original pre-projection CLIP hidden spaces.

    Rows must correspond to the same raw sketches.  CKA and Procrustes are
    invariant/agnostic to a global orthogonal change of basis; the projection
    cosine tests the stronger claim that the *frozen* CLIP projection remains
    usable after adaptation.
    """
    if h_ref.ndim != 2 or h_t.ndim != 2 or h_ref.shape != h_t.shape:
        raise ValueError("h_ref and h_t must have matching shape [N,D]")
    if h_ref.shape[0] < 2 or not h_ref.is_floating_point() or not h_t.is_floating_point():
        raise ValueError("hidden features must be floating point with at least two rows")
    if not torch.isfinite(h_ref).all().item() or not torch.isfinite(h_t).all().item():
        raise ValueError("hidden features must be finite")
    x = h_ref.float() - h_ref.float().mean(dim=0, keepdim=True)
    y = h_t.float() - h_t.float().mean(dim=0, keepdim=True)
    xx = x.T @ x
    yy = y.T @ y
    xy = x.T @ y
    cka_denominator = (xx.square().sum() * yy.square().sum()).sqrt().clamp_min(1e-12)
    cka = xy.square().sum().div(cka_denominator)
    # Orthogonal Procrustes for h_t R ~= h_ref, using the uncentered states
    # exactly as specified by the probe.  CKA above remains centered.
    ref = h_ref.float()
    adapted = h_t.float()
    u, _, vh = torch.linalg.svd(adapted.T @ ref, full_matrices=False)
    rotation = u @ vh
    residual = (adapted @ rotation - ref).norm().div(ref.norm().clamp_min(1e-12))
    projected_ref = projection(ref)
    projected_t = projection(h_t.float())
    projection_cosine = F.cosine_similarity(
        F.normalize(projected_ref, dim=-1), F.normalize(projected_t, dim=-1), dim=-1
    ).mean()
    return {
        "linear_cka": float(cka.item()),
        "procrustes_residual": float(residual.item()),
        "frozen_projection_mean_cosine": float(projection_cosine.item()),
        "effective_rank_h_ref": _effective_rank_from_matrix(h_ref),
        "effective_rank_h_t": _effective_rank_from_matrix(h_t),
        "effective_rank_W_h_t": _effective_rank_from_matrix(projected_t),
    }


def training_angle_summary(theta: Tensor) -> dict[str, object]:
    """Summarize actual sampled sketch/photo angles in radians/degrees."""
    if theta.ndim != 1 or theta.numel() == 0:
        raise ValueError("theta must be a non-empty [N] tensor")
    if not theta.is_floating_point() or not torch.isfinite(theta).all().item():
        raise ValueError("theta must be finite floating point")
    degrees = theta.detach().float().clamp_min(0.0) * (180.0 / math.pi)
    quantiles = torch.quantile(degrees, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
    return {
        "count": int(degrees.numel()),
        "mean_degrees": float(degrees.mean().item()),
        "std_degrees": float(degrees.std(unbiased=False).item()),
        "p05_degrees": float(quantiles[0].item()),
        "p25_degrees": float(quantiles[1].item()),
        "p50_degrees": float(quantiles[2].item()),
        "p75_degrees": float(quantiles[3].item()),
        "p95_degrees": float(quantiles[4].item()),
        "fraction_gt_5_degrees": float((degrees > 5).float().mean().item()),
        "fraction_gt_10_degrees": float((degrees > 10).float().mean().item()),
        "fraction_gt_15_degrees": float((degrees > 15).float().mean().item()),
        "fraction_gt_20_degrees": float((degrees > 20).float().mean().item()),
        "fraction_gt_30_degrees": float((degrees > 30).float().mean().item()),
        "fraction_gt_45_degrees": float((degrees > 45).float().mean().item()),
    }


@dataclass(frozen=True, slots=True)
class TransportFeatureSet:
    h: Tensor
    z0: Tensor
    directions: Tensor
    rho: Tensor
    q_hypotheses: Tensor
    q: Tensor
    labels: Tensor
    paths: tuple[str, ...]
    gate_logits: Tensor | None = None
    concentrations: Tensor | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("h", self.h),
            ("z0", self.z0),
            ("q", self.q),
            ("directions", self.directions),
            ("q_hypotheses", self.q_hypotheses),
            ("rho", self.rho),
        ):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains non-finite values")
        if self.h.ndim != 2 or self.z0.ndim != 2 or self.q.ndim != 2:
            raise ValueError("h, z0, and q must have shape [N,D]")
        if self.directions.ndim != 3 or self.q_hypotheses.ndim != 3:
            raise ValueError("directions and q_hypotheses must have shape [N,K,D]")
        if self.directions.shape != self.q_hypotheses.shape:
            raise ValueError("directions and q_hypotheses shapes must match")
        if self.h.shape[0] != self.z0.shape[0] or self.q.shape[0] != self.h.shape[0]:
            raise ValueError("all feature collections must have the same row count")
        if self.directions.shape[0] != self.h.shape[0] or self.q.shape[1] != self.directions.shape[2]:
            raise ValueError("transport feature dimensions do not match")
        if self.rho.shape not in {(self.h.shape[0],), (self.h.shape[0], self.directions.shape[1])}:
            raise ValueError("rho must have shape [N] or [N,K]")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.h.shape[0]:
            raise ValueError("labels must contain one value per feature row")
        if len(self.paths) != self.h.shape[0]:
            raise ValueError("paths must contain one value per feature row")
        if self.gate_logits is not None and self.gate_logits.shape != self.directions.shape[:2]:
            raise ValueError("gate_logits must have shape [N,K]")
        if self.concentrations is not None and self.concentrations.shape != self.directions.shape[:2]:
            raise ValueError("concentrations must have shape [N,K]")

    @property
    def num_components(self) -> int:
        return int(self.directions.shape[1])

    @property
    def queries(self) -> EncodedRetrievalSet:
        return EncodedRetrievalSet(
            embeddings=self.q,
            labels=self.labels,
            paths=self.paths,
            metadata={"representation": "predictive_semantic_transport_query"},
        )

    @property
    def probabilities(self) -> Tensor:
        if self.gate_logits is None:
            return self.directions.new_full(
                (self.h.shape[0], self.num_components),
                1.0 / self.num_components,
            )
        return self.gate_logits.softmax(dim=-1)


@torch.inference_mode()
def encode_transport_loader(
    model: SpicaPredictiveTransport,
    loader: DataLoader,
    *,
    device: torch.device,
    max_items: int | None = None,
) -> TransportFeatureSet:
    """Encode raw sketches and retain all transport quantities for probes."""
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive when provided")
    model.eval()
    collections: dict[str, list[Tensor]] = {
        "h": [],
        "z0": [],
        "directions": [],
        "rho": [],
        "q_hypotheses": [],
        "q": [],
    }
    labels: list[Tensor] = []
    paths: list[str] = []
    gate_logits: list[Tensor] = []
    concentrations: list[Tensor] = []
    remaining = max_items
    for batch in loader:
        images = batch["image"]
        take = images.shape[0] if remaining is None else min(remaining, images.shape[0])
        if take <= 0:
            break
        prediction: SemanticTransportPrediction = model(
            images[:take].to(device=device, non_blocking=device.type == "cuda")
        )
        for name in collections:
            value = getattr(prediction, name)
            if not isinstance(value, Tensor):
                raise RuntimeError(f"transport prediction did not provide {name}")
            collections[name].append(value.float().cpu())
        if prediction.gate_logits is not None:
            gate_logits.append(prediction.gate_logits.float().cpu())
        if prediction.concentrations is not None:
            concentrations.append(prediction.concentrations.float().cpu())
        labels.append(batch["label"][:take].long().cpu())
        paths.extend(str(path) for path in batch["path"][:take])
        if remaining is not None:
            remaining -= take
            if remaining == 0:
                break

    if not collections["q"]:
        raise ValueError("Cannot encode an empty transport loader")
    return TransportFeatureSet(
        **{name: torch.cat(values, dim=0) for name, values in collections.items()},
        labels=torch.cat(labels, dim=0),
        paths=tuple(paths),
        gate_logits=torch.cat(gate_logits, dim=0) if gate_logits else None,
        concentrations=torch.cat(concentrations, dim=0) if concentrations else None,
    )


def _evaluate_score_chunks(
    features: TransportFeatureSet,
    gallery: EncodedRetrievalSet,
    *,
    mode: TransportScoreMode,
    temperature: float,
    precision_at_k: tuple[int, ...],
    map_at_k: tuple[int, ...],
    map_at_k_denominator: str,
    query_chunk_size: int,
    device: torch.device,
) -> CategoryRetrievalEvaluation:
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    ks = tuple(sorted(set(precision_at_k)))
    map_ks = tuple(sorted(set(map_at_k)))
    num_queries = features.q.shape[0]
    num_gallery = gallery.embeddings.shape[0]
    if num_queries == 0 or num_gallery == 0:
        raise ValueError("queries and gallery cannot be empty")
    top_k = max(max(ks), *map_ks) if map_ks else max(ks)
    if top_k > num_gallery:
        raise ValueError("requested retrieval cutoff exceeds gallery size")
    compute_device = torch.device(device)
    gallery_embeddings = F.normalize(gallery.embeddings.to(compute_device), dim=-1)
    gallery_labels = gallery.labels.to(compute_device)
    rank_positions = torch.arange(
        1, num_gallery + 1, device=compute_device, dtype=torch.float32
    )
    ap_batches: list[Tensor] = []
    precision_batches: dict[int, list[Tensor]] = {k: [] for k in ks}
    map_batches: dict[int, list[Tensor]] = {k: [] for k in map_ks}
    top_indices: list[Tensor] = []
    top_scores: list[Tensor] = []
    for start in range(0, num_queries, query_chunk_size):
        stop = min(start + query_chunk_size, num_queries)
        q_hyp = features.q_hypotheses[start:stop].to(compute_device)
        gates = None if features.gate_logits is None else features.gate_logits[start:stop].to(compute_device)
        scores = transport_gallery_scores(
            q_hyp,
            gates,
            gallery_embeddings,
            mode=mode,
            temperature=temperature,
        )
        query_labels = features.labels[start:stop].to(compute_device)
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
        relevant = gallery_labels[ranked].eq(query_labels[:, None])
        positives = relevant.sum(dim=1)
        if torch.any(positives == 0).item():
            raise ValueError("Every transport query must have a positive gallery item")
        average_precision, truncated = _average_precision_from_relevance(
            relevant,
            rank_positions,
            map_ks,
            map_at_k_denominator=map_at_k_denominator,
        )
        ap_batches.append(average_precision.cpu())
        for k, values in truncated.items():
            map_batches[k].append(values.cpu())
        relevant_float = relevant.float()
        for k in ks:
            precision_batches[k].append(relevant_float[:, :k].mean(dim=1).cpu())
        top = ranked[:, :top_k]
        top_indices.append(top.cpu())
        top_scores.append(scores.gather(dim=1, index=top).float().cpu())
    ap = torch.cat(ap_batches)
    return CategoryRetrievalEvaluation(
        metrics=CategoryRetrievalMetrics(
            mean_average_precision=ap.double().mean().item(),
            precision_at_k={k: torch.cat(values).double().mean().item() for k, values in precision_batches.items()},
            mean_average_precision_at_k={k: torch.cat(values).double().mean().item() for k, values in map_batches.items()},
            num_queries=num_queries,
            num_gallery_items=num_gallery,
        ),
        average_precision_per_query=ap,
        top_indices=torch.cat(top_indices),
        top_scores=torch.cat(top_scores),
    )


def evaluate_transport_features(
    features: TransportFeatureSet,
    gallery: EncodedRetrievalSet,
    *,
    modes: tuple[TransportScoreMode, ...] = ("barycentric", "angular_logsumexp", "max"),
    temperature: float = 0.07,
    precision_at_k: tuple[int, ...] = (1, 5, 10, 100, 200),
    map_at_k: tuple[int, ...] = (200,),
    map_at_k_denominator: str = "prefix_positive",
    query_chunk_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> dict[str, CategoryRetrievalEvaluation]:
    """Evaluate barycentric, soft multi-hypothesis, and max-query modes."""
    results: dict[str, CategoryRetrievalEvaluation] = {}
    for mode in dict.fromkeys(modes):
        results[mode] = _evaluate_score_chunks(
            features,
            gallery,
            mode=mode,
            temperature=temperature,
            precision_at_k=precision_at_k,
            map_at_k=map_at_k,
            map_at_k_denominator=map_at_k_denominator,
            query_chunk_size=query_chunk_size,
            device=device,
        )
    return results


def evaluate_base_queries(
    features: TransportFeatureSet,
    gallery: EncodedRetrievalSet,
    *,
    temperature: float = 0.07,
    precision_at_k: tuple[int, ...] = (1, 5, 10, 100, 200),
    map_at_k: tuple[int, ...] = (200,),
    map_at_k_denominator: str = "prefix_positive",
    query_chunk_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> CategoryRetrievalEvaluation:
    """Evaluate the untransported ``z0`` on exactly the same gallery.

    This is deliberately a feature-level control rather than a second model
    forward pass.  It makes ``mAP(q) - mAP(z0)`` an apples-to-apples causal
    transport gain at every checkpoint.
    """
    base = TransportFeatureSet(
        h=features.h,
        z0=features.z0,
        directions=features.z0[:, None, :],
        rho=features.z0.new_zeros(features.z0.shape[0]),
        q_hypotheses=features.z0[:, None, :],
        q=features.z0,
        labels=features.labels,
        paths=features.paths,
    )
    return _evaluate_score_chunks(
        base,
        gallery,
        mode="barycentric",
        temperature=temperature,
        precision_at_k=precision_at_k,
        map_at_k=map_at_k,
        map_at_k_denominator=map_at_k_denominator,
        query_chunk_size=query_chunk_size,
        device=device,
    )


def _pearson(first: Tensor, second: Tensor) -> float | None:
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape or first.numel() < 2:
        return None
    first = first.float()
    second = second.float()
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = (first_centered.square().sum() * second_centered.square().sum()).sqrt()
    if denominator <= 1e-12:
        return None
    return float((first_centered * second_centered).sum().div(denominator).item())


def transport_query_correlations(
    features: TransportFeatureSet,
    gallery: EncodedRetrievalSet,
    average_precision: Tensor,
) -> dict[str, float | None]:
    """Correlate learned rho with per-query retrieval and target diagnostics."""
    if average_precision.ndim != 1 or average_precision.shape[0] != features.q.shape[0]:
        raise ValueError("average_precision must contain one value per query")
    target = _class_centroid_targets(features, gallery)
    target_angle = photo_transport_target(features.z0, target).theta
    query = F.normalize(features.q, dim=-1)
    query_labels = torch.unique(features.labels, sorted=True)
    query_centroids = torch.stack([
        F.normalize(query[features.labels == label].mean(dim=0), dim=-1)
        for label in query_labels
    ])
    positions = torch.searchsorted(query_labels, features.labels)
    own = (query * query_centroids[positions]).sum(dim=-1)
    all_cosines = query_centroids[positions] @ query_centroids.T
    own_class = torch.arange(query_labels.numel(), device=query_labels.device)[positions]
    other_mask = torch.ones_like(all_cosines, dtype=torch.bool)
    other_mask[torch.arange(all_cosines.shape[0], device=all_cosines.device), own_class] = False
    other = all_cosines.masked_fill(~other_mask, 0.0).sum(dim=-1) / other_mask.sum(dim=-1).clamp_min(1)
    class_margin = own - other
    rho = features.rho if features.rho.ndim == 1 else features.rho.mean(dim=-1)
    return {
        "rho_vs_per_query_ap": _pearson(rho, average_precision),
        "rho_vs_class_margin": _pearson(rho, class_margin),
        "rho_vs_target_angle": _pearson(rho, target_angle),
    }


def _class_centroid_targets(
    features: TransportFeatureSet,
    gallery: EncodedRetrievalSet,
) -> Tensor:
    labels = torch.unique(gallery.labels, sorted=True)
    centroids = torch.stack(
        [F.normalize(gallery.embeddings[gallery.labels == label].mean(dim=0), dim=-1) for label in labels]
    )
    positions = torch.searchsorted(labels, features.labels)
    if torch.any(positions >= labels.shape[0]).item() or not torch.equal(labels[positions], features.labels):
        raise ValueError("A transport query class is missing from the gallery")
    return centroids[positions]


def _rho_summary(rho: Tensor) -> dict[str, float]:
    values = rho if rho.ndim == 1 else rho.mean(dim=-1)
    degrees = values * (180.0 / math.pi)
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95]))
    degree_quantiles = quantiles * (180.0 / math.pi)
    return {
        "mean_rho": values.mean().item(),
        "std_rho": values.std(unbiased=False).item(),
        "p05_rho": quantiles[0].item(),
        "p50_rho": quantiles[1].item(),
        "p95_rho": quantiles[2].item(),
        "mean_rho_degrees": degrees.mean().item(),
        "std_rho_degrees": degrees.std(unbiased=False).item(),
        "p05_rho_degrees": degree_quantiles[0].item(),
        "p50_rho_degrees": degree_quantiles[1].item(),
        "p95_rho_degrees": degree_quantiles[2].item(),
    }


def _angle_summary(theta: Tensor) -> dict[str, float]:
    degrees = theta.detach().float() * (180.0 / math.pi)
    quantiles = torch.quantile(degrees, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
    return {
        "mean_degrees": degrees.mean().item(),
        "std_degrees": degrees.std(unbiased=False).item(),
        "p05_degrees": quantiles[0].item(),
        "p25_degrees": quantiles[1].item(),
        "p50_degrees": quantiles[2].item(),
        "p75_degrees": quantiles[3].item(),
        "p95_degrees": quantiles[4].item(),
    }


def _target_angle_probe(
    features: TransportFeatureSet,
    target: Tensor,
    frozen_reference: Tensor | None,
) -> dict[str, object]:
    moving = photo_transport_target(features.z0, target)
    rho = features.rho if features.rho.ndim == 1 else features.rho.mean(dim=-1)
    valid_ratio = rho[moving.theta > 1e-6] / moving.theta[moving.theta > 1e-6]
    result: dict[str, object] = {
        "moving": _angle_summary(moving.theta),
        "rho_over_moving_theta": float(valid_ratio.median().item()) if valid_ratio.numel() else None,
        "rho_over_moving_theta_mean": float(valid_ratio.mean().item()) if valid_ratio.numel() else None,
        "moving_fraction_beyond_cap": {
            str(cap): float((moving.theta * (180.0 / math.pi) > cap).float().mean().item())
            for cap in (5, 10, 15, 20, 30, 45)
        },
    }
    if frozen_reference is None:
        return result
    if frozen_reference.shape != features.z0.shape:
        raise ValueError("frozen_reference must match [N,D] transport features")
    fixed = fixed_origin_transport_target(frozen_reference, target, features.z0)
    result["fixed"] = _angle_summary(fixed.theta)
    result["fixed_fraction_beyond_cap"] = {
        str(cap): float((fixed.theta * (180.0 / math.pi) > cap).float().mean().item())
        for cap in (5, 10, 15, 20, 30, 45)
    }
    moving_directions = F.normalize(features.directions, dim=-1)
    fixed_alignment = (moving_directions * fixed.direction[:, None, :]).sum(dim=-1)
    moving_alignment = (moving_directions * moving.direction[:, None, :]).sum(dim=-1)
    frame_agreement = (moving.direction * fixed.direction).sum(dim=-1)
    result["moving_target_alignment"] = float(moving_alignment.mean().item())
    result["fixed_target_alignment"] = float(fixed_alignment.mean().item())
    result["target_frame_agreement"] = float(frame_agreement.mean().item())
    result["fixed_tangent_destination_max_abs_dot"] = float(
        (fixed.direction * F.normalize(features.z0, dim=-1)).sum(dim=-1).abs().max().item()
    )
    return result


def component_direction_alignment(
    features: TransportFeatureSet,
    instance_targets: Tensor,
    class_targets: Tensor,
) -> dict[str, object]:
    """Compare K directions with instance and class-semantic tangent targets.

    Both targets are built at the same current ``z0``.  Callers are
    responsible for constructing ``class_targets`` from training photos only;
    this function never reads labels or gallery data to create a prototype.
    """
    if instance_targets.shape != class_targets.shape or instance_targets.shape != features.z0.shape:
        raise ValueError("instance/class targets must match feature shape [N,D]")
    instance = photo_transport_target(features.z0, instance_targets)
    semantic = photo_transport_target(features.z0, class_targets)
    directions = F.normalize(features.directions, dim=-1)
    instance_cos = (directions * instance.direction[:, None, :]).sum(dim=-1)
    class_cos = (directions * semantic.direction[:, None, :]).sum(dim=-1)
    weights = features.probabilities
    result: dict[str, object] = {
        "instance_alignment_by_component": instance_cos.mean(dim=0).tolist(),
        "class_alignment_by_component": class_cos.mean(dim=0).tolist(),
        "instance_alignment_max": instance_cos.max(dim=-1).values.mean().item(),
        "class_alignment_max": class_cos.max(dim=-1).values.mean().item(),
        "instance_alignment_gate_weighted": (weights * instance_cos).sum(dim=-1).mean().item(),
        "class_alignment_gate_weighted": (weights * class_cos).sum(dim=-1).mean().item(),
    }
    if features.concentrations is not None:
        # The vMF posterior is a useful responsibility-selected diagnostic,
        # but is computed from the supplied targets and never fed to training.
        log_pi = weights.clamp_min(1e-12).log()
        posterior = (
            log_pi + features.concentrations * instance_cos
        ).softmax(dim=-1)
        result["instance_alignment_responsibility_selected"] = (posterior * instance_cos).sum(dim=-1).mean().item()
        result["class_alignment_responsibility_selected"] = (posterior * class_cos).sum(dim=-1).mean().item()
    return result


def multi_photo_component_alignment(
    features: TransportFeatureSet,
    train_photo_embeddings: Tensor,
    train_photo_labels: Tensor,
    *,
    photos_per_class: int = 8,
    seed: int = 3407,
) -> dict[str, object]:
    """Measure class-semantic versus instance-residual transport directions.

    ``train_photo_embeddings`` must contain training photos only.  The class
    prototype and all R sampled photos are formed in the tangent space at each
    current z0 using the spherical logarithm; no validation/test photo can
    enter this diagnostic unless the caller supplies it explicitly.
    """
    if photos_per_class <= 0:
        raise ValueError("photos_per_class must be positive")
    if train_photo_embeddings.ndim != 2 or train_photo_labels.ndim != 1:
        raise ValueError("training photos/labels have invalid dimensions")
    if train_photo_embeddings.shape[0] != train_photo_labels.shape[0]:
        raise ValueError("training photo embeddings and labels must align")
    if train_photo_embeddings.shape[1] != features.z0.shape[1]:
        raise ValueError("training photo dimension must match z0")
    if not torch.isfinite(train_photo_embeddings).all().item():
        raise ValueError("training photo embeddings must be finite")
    photos = F.normalize(train_photo_embeddings, dim=-1)
    labels = torch.unique(train_photo_labels, sorted=True)
    by_label = {int(label): torch.where(train_photo_labels == label)[0] for label in labels}
    if any(int(label) not in by_label for label in features.labels.unique()):
        raise ValueError("a query class is missing from training photos")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction_values: list[Tensor] = []
    residual_values: list[Tensor] = []
    class_values: list[Tensor] = []
    for index, label in enumerate(features.labels.tolist()):
        candidates = by_label[int(label)]
        if candidates.numel() >= photos_per_class:
            chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:photos_per_class]]
        else:
            chosen = candidates[torch.randint(candidates.numel(), (photos_per_class,), generator=generator)]
        base = F.normalize(features.z0[index], dim=-1)
        selected = photos[chosen]
        prototype = F.normalize(selected.mean(dim=0), dim=-1)
        # The prototype is deliberately computed from the complete train-photo
        # class, while the R instances are sampled from that same train set.
        full_class_prototype = F.normalize(photos[candidates].mean(dim=0), dim=-1)
        prototype = full_class_prototype

        def sphere_log(destination: Tensor) -> Tensor:
            cosine = (base * destination).sum().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            theta = torch.acos(cosine)
            tangent = destination - cosine * base
            return tangent * (theta / tangent.norm().clamp_min(1e-8))

        class_log = sphere_log(prototype)
        class_unit = F.normalize(class_log, dim=-1)
        instance_logs = torch.stack([sphere_log(photo) for photo in selected])
        residuals = instance_logs - (instance_logs * class_unit).sum(dim=-1, keepdim=True) * class_unit
        residuals = F.normalize(residuals, dim=-1, eps=1e-8)
        predicted = F.normalize(features.directions[index], dim=-1)
        class_values.append((predicted * class_unit).sum(dim=-1))
        residual_cosines = predicted @ residuals.T
        direction_values.append(residual_cosines)
        residual_values.append(residual_cosines.max(dim=-1).values)
    class_alignment = torch.stack(class_values)
    all_residual = torch.stack(direction_values)
    max_residual = torch.stack(residual_values)
    weights = features.probabilities
    return {
        "photos_per_class": photos_per_class,
        "seed": seed,
        "class_alignment_by_component": class_alignment.mean(dim=0).tolist(),
        "instance_residual_alignment_by_component": all_residual.mean(dim=(0, 2)).tolist(),
        "instance_residual_alignment_max_by_component": max_residual.mean(dim=0).tolist(),
        "class_alignment_max": class_alignment.max(dim=-1).values.mean().item(),
        "instance_residual_alignment_mean": all_residual.mean().item(),
        "instance_residual_alignment_max": max_residual.max(dim=-1).values.mean().item(),
        "class_alignment_gate_weighted": (weights * class_alignment).sum(dim=-1).mean().item(),
        "instance_residual_alignment_gate_weighted": (weights * max_residual).sum(dim=-1).mean().item(),
    }


def train_photo_class_prototypes(
    encoded_photos: EncodedRetrievalSet,
) -> tuple[Tensor, Tensor]:
    """Return normalized class prototypes from an explicitly train-only set."""
    labels = torch.unique(encoded_photos.labels, sorted=True)
    prototypes = torch.stack([
        F.normalize(encoded_photos.embeddings[encoded_photos.labels == label].mean(dim=0), dim=-1)
        for label in labels
    ])
    return labels, prototypes


def transport_probe_dict(
    features: TransportFeatureSet,
    gallery: EncodedRetrievalSet,
    *,
    frozen_reference: Tensor | None = None,
    kappa_max: float | None = None,
) -> dict[str, object]:
    """Return retrieval, transport, stability, and mixture geometry probes."""
    target = _class_centroid_targets(features, gallery)
    target_transport = photo_transport_target(features.z0, target)
    directions = F.normalize(features.directions, dim=-1)
    direction_cosines = (directions * target_transport.direction[:, None, :]).sum(dim=-1)
    probabilities = features.probabilities
    posterior = None
    if features.concentrations is not None:
        # This is a diagnostic posterior against the class photo centroid, not
        # a training input or an inference-time signal.
        log_pi = probabilities.clamp_min(1e-12).log()
        log_normalizer = _relative_log_vmf_normalizer(
            features.concentrations,
            dimension=features.directions.shape[-1] - 1,
        )
        posterior = (
            log_pi
            + log_normalizer
            + features.concentrations * direction_cosines
        ).softmax(dim=-1)
    # Reuse the tested JEPA geometry routines without making a transport model
    # look like a full-vector predictor at inference.
    from .jepa import JepaFeatureSet

    # The legacy probe dataclass assumes all representations share one
    # dimension.  h is pre-projection (768 for ViT-B/32), so use q for its
    # compatibility fields and report h/z0 geometry separately above.
    jepa_features = JepaFeatureSet(
        h=features.q,
        u=features.q,
        q=features.q,
        labels=features.labels,
        paths=features.paths,
    )
    semantic = semantic_query_diagnostics(jepa_features, gallery)
    photo_alignment = photo_target_alignment_diagnostics(jepa_features, gallery)
    query_geometry = feature_geometry(features.q).to_dict()
    base_geometry = feature_geometry(features.z0).to_dict()
    target_angles = _target_angle_probe(features, target, frozen_reference)
    result: dict[str, object] = {
        "h": feature_geometry(features.h).to_dict(),
        "z0": base_geometry,
        "q": query_geometry,
        "semantic": semantic,
        "photo_targets": photo_alignment,
        "transport": {
            **_rho_summary(features.rho),
            "mean_direction_cosine": direction_cosines.mean().item(),
            "direction_cosine_std": direction_cosines.std(unbiased=False).item(),
            "endpoint_photo_cosine": (F.normalize(features.q, dim=-1) * target).sum(dim=-1).mean().item(),
            "base_photo_cosine": (F.normalize(features.z0, dim=-1) * target).sum(dim=-1).mean().item(),
            "mean_distance_error": (
                (features.rho if features.rho.ndim == 1 else features.rho.mean(dim=-1))
                - target_transport.theta
            ).abs().mean().item(),
            "near_zero_target_fraction": target_transport.near_zero.float().mean().item(),
            "target_angles": target_angles,
        },
        "mixture": {
            "num_components": features.num_components,
            "gate_entropy": (
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
            ),
            "mean_kappa": 0.0 if features.concentrations is None else features.concentrations.mean().item(),
            "kappa_saturation_fraction": 0.0
            if features.concentrations is None or kappa_max is None
            else float((features.concentrations >= 0.99 * kappa_max).float().mean().item()),
            "component_usage": probabilities.mean(dim=0).tolist(),
            "component_pairwise_direction_cosine": _component_pairwise_cosines(directions),
            "responsibility_entropy": None
            if posterior is None
            else float(
                -(posterior * posterior.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
            ),
            "mean_direction_cosine_by_component": direction_cosines.mean(dim=0).tolist(),
        },
    }
    if frozen_reference is not None:
        if frozen_reference.shape != features.z0.shape:
            raise ValueError("frozen_reference must match [N,D] transport features")
        reference = F.normalize(frozen_reference, dim=-1)
        result["reference"] = {
            "base_reference_cosine": (F.normalize(features.z0, dim=-1) * reference).sum(dim=-1).mean().item(),
            "query_reference_cosine": (F.normalize(features.q, dim=-1) * reference).sum(dim=-1).mean().item(),
        }
    return result


def _component_pairwise_cosines(directions: Tensor) -> list[float]:
    if directions.shape[1] < 2:
        return []
    pairs = torch.triu_indices(
        directions.shape[1], directions.shape[1], offset=1, device=directions.device
    )
    values = (
        directions[:, pairs[0], :] * directions[:, pairs[1], :]
    ).sum(dim=-1).mean(dim=0)
    return [float(value) for value in values.reshape(-1)]
