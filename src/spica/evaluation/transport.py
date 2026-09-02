"""Evaluation, retrieval modes, and geometry probes for semantic transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.transport import (
    SemanticTransportPrediction,
    SpicaPredictiveTransport,
    _relative_log_vmf_normalizer,
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
