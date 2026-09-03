from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .embeddings import EncodedRetrievalSet


@dataclass(frozen=True, slots=True)
class CategoryRetrievalMetrics:
    mean_average_precision: float
    precision_at_k: dict[int, float]
    num_queries: int
    num_gallery_items: int
    mean_average_precision_at_k: dict[int, float] = field(default_factory=dict)

    def to_log_dict(
        self,
        prefix: str = "retrieval",
        *,
        map_at_k_denominator: str | None = None,
    ) -> dict[str, float | int]:
        if map_at_k_denominator is not None and map_at_k_denominator not in {
            "prefix_positive",
            "all_relevant",
            "min_relevant_k",
        }:
            raise ValueError(
                "map_at_k_denominator must be one of prefix_positive, "
                "all_relevant, or min_relevant_k"
            )
        normalized_prefix = prefix.rstrip("/")
        metrics: dict[str, float | int] = {
            f"{normalized_prefix}/mAP": self.mean_average_precision,
            f"{normalized_prefix}/num_queries": self.num_queries,
            f"{normalized_prefix}/num_gallery_items": self.num_gallery_items,
        }
        metrics.update(
            {
                f"{normalized_prefix}/P@{k}": value
                for k, value in self.precision_at_k.items()
            }
        )
        metrics.update(
            {
                f"{normalized_prefix}/mAP@{k}": value
                for k, value in self.mean_average_precision_at_k.items()
            }
        )
        if map_at_k_denominator is not None:
            metrics.update(
                {
                    f"{normalized_prefix}/mAP@{k}_{map_at_k_denominator}": value
                    for k, value in self.mean_average_precision_at_k.items()
                }
            )
        return metrics


@dataclass(frozen=True, slots=True)
class CategoryRetrievalEvaluation:
    metrics: CategoryRetrievalMetrics
    average_precision_per_query: Tensor
    top_indices: Tensor
    top_scores: Tensor


MapAtKDenominator = Literal["prefix_positive", "all_relevant", "min_relevant_k"]


def _average_precision_from_relevance(
    relevant: Tensor,
    rank_positions: Tensor,
    cutoffs: tuple[int, ...],
    *,
    map_at_k_denominator: MapAtKDenominator = "prefix_positive",
) -> tuple[Tensor, dict[int, Tensor]]:
    """Compute full AP and explicitly named AP@K conventions.

    ``prefix_positive`` is the historical repository metric: AP is computed
    on the returned prefix and divided by positives found in that prefix.
    ``all_relevant`` keeps the full-gallery number of relevant items in the
    denominator, while ``min_relevant_k`` uses ``min(R, K)``.  The latter two
    variants are useful for auditing benchmark protocols; the default is kept
    for backward compatibility with existing artifacts.
    """
    if map_at_k_denominator not in {
        "prefix_positive",
        "all_relevant",
        "min_relevant_k",
    }:
        raise ValueError(
            "map_at_k_denominator must be 'prefix_positive', 'all_relevant', "
            f"or 'min_relevant_k', got {map_at_k_denominator!r}"
        )
    relevant_float = relevant.float()
    precision_at_rank = relevant_float.cumsum(dim=1) / rank_positions
    num_positives = relevant_float.sum(dim=1)
    full = (precision_at_rank * relevant_float).sum(dim=1) / num_positives.clamp_min(1)
    truncated: dict[int, Tensor] = {}
    for k in cutoffs:
        prefix_relevance = relevant_float[:, :k]
        prefix_precision = precision_at_rank[:, :k]
        if map_at_k_denominator == "prefix_positive":
            denominator = prefix_relevance.sum(dim=1)
        elif map_at_k_denominator == "all_relevant":
            denominator = num_positives
        else:
            denominator = torch.minimum(
                num_positives,
                torch.full_like(num_positives, k),
            )
        truncated[k] = (prefix_precision * prefix_relevance).sum(
            dim=1
        ) / denominator.clamp_min(1)
    return full, truncated


@torch.inference_mode()
def evaluate_category_retrieval(
    queries: EncodedRetrievalSet,
    gallery: EncodedRetrievalSet,
    *,
    precision_at_k: tuple[int, ...] = (1, 5, 10, 100),
    map_at_k: tuple[int, ...] = (),
    query_chunk_size: int = 256,
    top_k: int | None = None,
    map_at_k_denominator: MapAtKDenominator = "prefix_positive",
    device: str | torch.device = "cpu",
) -> CategoryRetrievalEvaluation:
    if queries.embeddings.shape[1] != gallery.embeddings.shape[1]:
        raise ValueError(
            "Query and gallery embedding dimensions must match, got "
            f"{queries.embeddings.shape[1]} and {gallery.embeddings.shape[1]}"
        )
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    ks = tuple(sorted(set(precision_at_k)))
    map_ks = tuple(sorted(set(map_at_k)))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"precision_at_k must contain positive values, got {ks}")
    if any(k <= 0 for k in map_ks):
        raise ValueError(f"map_at_k must contain positive values, got {map_ks}")

    num_queries = queries.embeddings.shape[0]
    num_gallery_items = gallery.embeddings.shape[0]
    if num_queries == 0:
        raise ValueError("Cannot evaluate an empty query set")
    if num_gallery_items == 0:
        raise ValueError("Cannot evaluate an empty gallery")
    if max(ks) > num_gallery_items:
        raise ValueError(
            f"P@{max(ks)} requires at least {max(ks)} gallery items, but the gallery contains {num_gallery_items}"
        )
    if map_ks and max(map_ks) > num_gallery_items:
        raise ValueError(
            f"mAP@{max(map_ks)} requires at least {max(map_ks)} gallery items, but the gallery contains {num_gallery_items}"
        )
    if top_k is None:
        top_k = max(ks)
    if not 0 < top_k <= num_gallery_items:
        raise ValueError(
            f"top_k must be between 1 and {num_gallery_items}, got {top_k}"
        )

    compute_device = torch.device(device)
    gallery_embeddings = F.normalize(gallery.embeddings.to(compute_device), dim=-1)
    gallery_labels = gallery.labels.to(compute_device)
    rank_positions = torch.arange(
        1, num_gallery_items + 1, device=compute_device, dtype=torch.float32
    )
    average_precision_batches: list[Tensor] = []
    precision_batches: dict[int, list[Tensor]] = {k: [] for k in ks}
    map_batches: dict[int, list[Tensor]] = {k: [] for k in map_ks}
    top_index_batches: list[Tensor] = []
    top_score_batches: list[Tensor] = []

    for start in range(0, num_queries, query_chunk_size):
        stop = min(start + query_chunk_size, num_queries)
        query_embeddings = F.normalize(
            queries.embeddings[start:stop].to(compute_device), dim=-1
        )
        query_labels = queries.labels[start:stop].to(compute_device)
        scores = query_embeddings @ gallery_embeddings.T
        ranked_indices = torch.argsort(scores, dim=1, descending=True, stable=True)
        relevant = gallery_labels[ranked_indices].eq(query_labels[:, None])
        num_positives = relevant.sum(dim=1)
        if torch.any(num_positives == 0):
            local_indices = torch.nonzero(num_positives == 0).flatten()
            raise ValueError(
                "Every query must have at least one positive gallery item; queries without positives: "
                f"{(local_indices + start).tolist()}"
            )
        average_precision, truncated = _average_precision_from_relevance(
            relevant,
            rank_positions,
            map_ks,
            map_at_k_denominator=map_at_k_denominator,
        )
        average_precision_batches.append(average_precision.cpu())
        for k, values in truncated.items():
            map_batches[k].append(values.cpu())
        relevant_float = relevant.float()
        for k in ks:
            precision_batches[k].append(relevant_float[:, :k].mean(dim=1).cpu())
        chunk_top_indices = ranked_indices[:, :top_k]
        top_index_batches.append(chunk_top_indices.cpu())
        top_score_batches.append(
            scores.gather(dim=1, index=chunk_top_indices).float().cpu()
        )

    average_precision_per_query = torch.cat(average_precision_batches, dim=0)
    return CategoryRetrievalEvaluation(
        metrics=CategoryRetrievalMetrics(
            mean_average_precision=average_precision_per_query.double().mean().item(),
            precision_at_k={
                k: torch.cat(values, dim=0).double().mean().item()
                for k, values in precision_batches.items()
            },
            num_queries=num_queries,
            num_gallery_items=num_gallery_items,
            mean_average_precision_at_k={
                k: torch.cat(values, dim=0).double().mean().item()
                for k, values in map_batches.items()
            },
        ),
        average_precision_per_query=average_precision_per_query,
        top_indices=torch.cat(top_index_batches, dim=0),
        top_scores=torch.cat(top_score_batches, dim=0),
    )
