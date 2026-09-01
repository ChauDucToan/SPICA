"""Evaluation and feature-geometry probes for predictive SPICA JEPA runs."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.jepa import JepaPrediction, SketchPhotoJepa
from .embeddings import EncodedRetrievalSet
from .metrics import CategoryRetrievalEvaluation, evaluate_category_retrieval


@dataclass(frozen=True, slots=True)
class JepaFeatureSet:
    h: Tensor
    u: Tensor
    q: Tensor
    labels: Tensor
    paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.h.ndim != 2 or self.u.ndim != 2 or self.q.ndim != 2:
            raise ValueError("h, u, and q must have shape [num_items, dimension]")
        if self.h.shape != self.u.shape or self.h.shape != self.q.shape:
            raise ValueError("h, u, and q must have matching shapes")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.q.shape[0]:
            raise ValueError("labels must contain one value per feature row")
        if len(self.paths) != self.q.shape[0]:
            raise ValueError("paths must contain one value per feature row")

    @property
    def queries(self) -> EncodedRetrievalSet:
        return EncodedRetrievalSet(
            embeddings=self.q,
            labels=self.labels,
            paths=self.paths,
            metadata={"representation": "predicted_photo_semantic_query"},
        )


@dataclass(frozen=True, slots=True)
class FeatureGeometry:
    effective_rank: float
    mean_variance: float
    minimum_variance: float
    near_zero_variance_fraction: float
    covariance_offdiag: float
    mean_pairwise_cosine: float
    global_anisotropy: float
    mean_norm: float
    singular_values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "effective_rank": self.effective_rank,
            "mean_feature_variance": self.mean_variance,
            "minimum_feature_variance": self.minimum_variance,
            "near_zero_variance_fraction": self.near_zero_variance_fraction,
            "covariance_offdiag": self.covariance_offdiag,
            "mean_pairwise_cosine": self.mean_pairwise_cosine,
            "global_anisotropy": self.global_anisotropy,
            "mean_norm": self.mean_norm,
            "singular_values": list(self.singular_values),
        }


@torch.inference_mode()
def encode_jepa_loader(
    model: SketchPhotoJepa,
    loader: DataLoader,
    *,
    device: torch.device,
    max_items: int | None = None,
) -> JepaFeatureSet:
    """Run the raw-sketch model and retain h/u/q for geometry probes."""
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive when provided")
    model.eval()
    h_batches: list[Tensor] = []
    u_batches: list[Tensor] = []
    q_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    paths: list[str] = []
    remaining = max_items

    for batch in loader:
        images = batch["image"]
        if remaining is not None:
            take = min(remaining, images.shape[0])
            images = images[:take]
        if images.shape[0] == 0:
            break
        prediction: JepaPrediction = model(
            images.to(device=device, non_blocking=device.type == "cuda")
        )
        for values, collection in (
            (prediction.h, h_batches),
            (prediction.u, u_batches),
            (prediction.q, q_batches),
        ):
            if not torch.isfinite(values).all().item():
                raise FloatingPointError("JEPA model returned non-finite features")
            collection.append(values.float().cpu())
        norms = prediction.q.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
            raise RuntimeError("JEPA predicted queries must have unit norm")
        label_batches.append(batch["label"][: images.shape[0]].long().cpu())
        paths.extend(str(path) for path in batch["path"][: images.shape[0]])
        if remaining is not None:
            remaining -= images.shape[0]
            if remaining == 0:
                break

    if not q_batches:
        raise ValueError("Cannot encode an empty JEPA loader")
    return JepaFeatureSet(
        h=torch.cat(h_batches, dim=0),
        u=torch.cat(u_batches, dim=0),
        q=torch.cat(q_batches, dim=0),
        labels=torch.cat(label_batches, dim=0),
        paths=tuple(paths),
    )


def evaluate_jepa_features(
    features: JepaFeatureSet,
    gallery: EncodedRetrievalSet,
    *,
    precision_at_k: tuple[int, ...] = (1, 5, 10, 100, 200),
    map_at_k: tuple[int, ...] = (200,),
    map_at_k_denominator: str = "prefix_positive",
    query_chunk_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> CategoryRetrievalEvaluation:
    return evaluate_category_retrieval(
        features.queries,
        gallery,
        precision_at_k=precision_at_k,
        map_at_k=map_at_k,
        map_at_k_denominator=map_at_k_denominator,
        query_chunk_size=query_chunk_size,
        top_k=max(max(precision_at_k), *map_at_k),
        device=device,
    )


def feature_geometry(
    features: Tensor,
    *,
    max_samples: int = 4096,
    near_zero_threshold: float = 1e-6,
    pair_samples: int = 16384,
) -> FeatureGeometry:
    """Measure rank, variance, covariance, spectrum, and anisotropy.

    The returned statistics use a deterministic random subset when a caller
    supplies more than ``max_samples`` rows.  No regularizer is applied here;
    this is a diagnostic probe only.
    """
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must have shape [num_items >= 2, dimension]")
    if not features.is_floating_point() or not torch.isfinite(features).all().item():
        raise ValueError("features must be finite floating-point values")
    if max_samples < 2 or pair_samples <= 0 or near_zero_threshold < 0:
        raise ValueError("invalid feature geometry sampling arguments")
    if features.shape[0] > max_samples:
        sample_generator = torch.Generator(device="cpu")
        sample_generator.manual_seed(17)
        sample_indices = torch.randperm(
            features.shape[0], generator=sample_generator
        )[:max_samples]
        sampled = features[sample_indices].float()
    else:
        sampled = features.float()
    centered = sampled - sampled.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (sampled.shape[0] - 1)
    variances = covariance.diagonal().clamp_min(0)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    eigenvalue_sum = eigenvalues.sum()
    if eigenvalue_sum <= 0:
        effective_rank = 0.0
        anisotropy = 0.0
    else:
        effective_rank = (eigenvalue_sum.square() / eigenvalues.square().sum().clamp_min(1e-12)).item()
        anisotropy = (eigenvalues[0] / eigenvalue_sum).item()
    mask = ~torch.eye(
        covariance.shape[0], dtype=torch.bool, device=covariance.device
    )
    off_diagonal = covariance[mask]

    normalized = F.normalize(sampled, dim=-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    first = torch.randint(
        0,
        normalized.shape[0],
        (min(pair_samples, max(1, normalized.shape[0] * 4)),),
        generator=generator,
    )
    second = torch.randint(
        0,
        normalized.shape[0],
        first.shape,
        generator=generator,
    )
    pairwise = (normalized[first] * normalized[second]).sum(dim=-1)

    return FeatureGeometry(
        effective_rank=effective_rank,
        mean_variance=variances.mean().item(),
        minimum_variance=variances.min().item(),
        near_zero_variance_fraction=(variances <= near_zero_threshold).float().mean().item(),
        covariance_offdiag=off_diagonal.abs().mean().item(),
        mean_pairwise_cosine=pairwise.mean().item(),
        global_anisotropy=anisotropy,
        mean_norm=sampled.norm(dim=-1).mean().item(),
        singular_values=tuple(torch.sqrt(eigenvalues).tolist()),
    )


def _class_centroids(embeddings: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    unique = torch.unique(labels, sorted=True)
    centroids: list[Tensor] = []
    for label in unique:
        values = embeddings[labels == label]
        centroids.append(F.normalize(values.mean(dim=0), dim=-1))
    return unique, torch.stack(centroids, dim=0)


def semantic_query_diagnostics(
    features: JepaFeatureSet,
    gallery: EncodedRetrievalSet,
) -> dict[str, float | int]:
    """Compute target alignment and predicted class-centroid geometry."""
    if features.q.shape[1] != gallery.embeddings.shape[1]:
        raise ValueError("Feature and gallery dimensions must match")
    gallery_labels, gallery_centroids = _class_centroids(
        F.normalize(gallery.embeddings, dim=-1), gallery.labels
    )
    positions = torch.searchsorted(gallery_labels, features.labels)
    if not torch.all(positions < gallery_labels.shape[0]).item():
        raise ValueError("A query class is missing from the photo gallery")
    if not torch.equal(gallery_labels[positions], features.labels):
        raise ValueError("A query class is missing from the photo gallery")
    target = gallery_centroids[positions]
    query = F.normalize(features.q, dim=-1)
    target_alignment = (query * target).sum(dim=-1)

    query_labels, query_centroids = _class_centroids(query, features.labels)
    own_centroid = query_centroids[torch.searchsorted(query_labels, features.labels)]
    intra = (query * own_centroid).sum(dim=-1).mean()
    if query_centroids.shape[0] > 1:
        centroid_cosines = query_centroids @ query_centroids.T
        off_diagonal = centroid_cosines[~torch.eye(
            centroid_cosines.shape[0], dtype=torch.bool
        )]
        inter = off_diagonal.mean()
    else:
        inter = query_centroids.new_tensor(0.0)
    return {
        "predicted_target_cosine": target_alignment.mean().item(),
        "predicted_target_cosine_std": target_alignment.std(unbiased=False).item(),
        "intra_class_cosine": intra.item(),
        "inter_class_cosine": inter.item(),
        "semantic_margin": (intra - inter).item(),
        "num_classes": int(query_labels.numel()),
    }


def photo_target_alignment_diagnostics(
    features: JepaFeatureSet,
    gallery: EncodedRetrievalSet,
) -> dict[str, float]:
    """Compare each query with individual positives, centroids, and negatives."""
    query = F.normalize(features.q, dim=-1)
    photos = F.normalize(gallery.embeddings, dim=-1)
    positive_values: list[Tensor] = []
    centroid_values: list[Tensor] = []
    negative_values: list[Tensor] = []
    for label in torch.unique(features.labels, sorted=True):
        query_mask = features.labels == label
        positive_mask = gallery.labels == label
        negative_mask = ~positive_mask
        if not positive_mask.any().item() or not negative_mask.any().item():
            raise ValueError("Every query class needs positive and negative gallery items")
        class_queries = query[query_mask]
        class_positives = photos[positive_mask]
        positive_cosines = class_queries @ class_positives.T
        centroid = F.normalize(class_positives.mean(dim=0), dim=-1)
        positive_values.append(positive_cosines.mean(dim=-1))
        centroid_values.append((class_queries * centroid).sum(dim=-1))
        negative_values.append((class_queries @ photos[negative_mask].T).mean(dim=-1))
    individual = torch.cat(positive_values)
    centroid = torch.cat(centroid_values)
    negative = torch.cat(negative_values)
    return {
        "individual_positive_cosine": individual.mean().item(),
        "individual_positive_cosine_std": individual.std(unbiased=False).item(),
        "positive_centroid_cosine": centroid.mean().item(),
        "negative_gallery_cosine": negative.mean().item(),
        "positive_negative_margin": (centroid - negative).mean().item(),
    }


def feature_probe_dict(
    features: JepaFeatureSet,
    gallery: EncodedRetrievalSet,
) -> dict[str, object]:
    return {
        "h": feature_geometry(features.h).to_dict(),
        "u": feature_geometry(features.u).to_dict(),
        "q": feature_geometry(features.q).to_dict(),
        "semantic": semantic_query_diagnostics(features, gallery),
        "photo_targets": photo_target_alignment_diagnostics(features, gallery),
    }
