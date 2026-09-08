"""Evaluation primitives for the coupled-predictive V1 campaign."""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..evaluation.embeddings import EncodedRetrievalSet
from ..evaluation.metrics import evaluate_category_retrieval_all_denominators
from ..evaluation.frozen_prompt import encode_prompted_loader
from ..evaluation.masked_view import _encode_masked_loader

EVAL_FRACTIONS = (0.25, 0.5, 0.75)
EVAL_SEEDS = (101, 202, 303)
EVAL_PRECISION = (1, 5, 10, 100, 200)
_RETRIEVAL_NAMES = (
    "full_mAP", "P@200", "mAP@200_prefix_positive",
    "mAP@200_all_relevant", "mAP@200_min_relevant_k",
)


class CoupledPredictiveAdapter:
    """Expose the adapter expected by the existing evaluation helpers."""

    def __init__(self, model: Any, *, query: str = "mu_i") -> None:
        self.model = model
        self.query = query

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def training(self) -> bool:
        return bool(self.model.training)

    def train(self, mode: bool = True) -> "CoupledPredictiveAdapter":
        self.model.train(mode)
        return self

    def eval(self) -> "CoupledPredictiveAdapter":
        return self.train(False)

    def __call__(self, images: Tensor) -> Tensor:
        return getattr(self.model(images), self.query)

    def encode_photo(self, photos: Tensor) -> Tensor:
        return self.model.encode_photo(photos)


@contextmanager
def preserve_rng_and_mode(model: Any):
    """Probes cannot consume training RNG or leave the model in eval mode."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = bool(model.training)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train(was_training)


def _metrics(
    queries: EncodedRetrievalSet,
    gallery: EncodedRetrievalSet,
    *,
    device: torch.device,
    query_chunk: int,
) -> dict[str, Any]:
    evaluations = evaluate_category_retrieval_all_denominators(
        queries, gallery, precision_at_k=EVAL_PRECISION, map_at_k=(200,),
        query_chunk_size=query_chunk, top_k=200, device=device,
    )
    prefix = evaluations["prefix_positive"]
    top = prefix.top_indices
    relevant = gallery.labels[top] == queries.labels[:, None]
    p200 = relevant.float().mean(dim=1)
    result: dict[str, Any] = {
        "full_mAP": float(prefix.metrics.mean_average_precision),
        "P@200": float(prefix.metrics.precision_at_k[200]),
        "mAP@200_prefix_positive": float(prefix.metrics.mean_average_precision_at_k[200]),
        "mAP@200_all_relevant": float(evaluations["all_relevant"].metrics.mean_average_precision_at_k[200]),
        "mAP@200_min_relevant_k": float(evaluations["min_relevant_k"].metrics.mean_average_precision_at_k[200]),
        "average_precision_per_query": prefix.average_precision_per_query.tolist(),
        "average_precision_at_k_per_query": {
            name: value.average_precision_at_k_per_query[200].tolist()
            for name, value in evaluations.items()
        },
        "P@200_per_query": p200.tolist(),
        "top_indices": top.tolist(),
    }
    return result


def _status_counts(metas: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(meta.get("status", "unknown")) for meta in metas).items()))


def evaluate_views(
    adapter: CoupledPredictiveAdapter,
    clean_loader: DataLoader,
    gallery_loader: DataLoader,
    masked_loader_factory: Callable[[float, int], DataLoader],
    *,
    query_entries: tuple[Any, ...] | None = None,
    fractions: tuple[float, ...] = EVAL_FRACTIONS,
    seeds: tuple[int, ...] = EVAL_SEEDS,
    query_chunk_size: int = 256,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate clean plus the fixed nine masked views in one probe."""
    del query_entries, fractions, seeds  # the approved constants are the protocol
    device = device or adapter.device
    with preserve_rng_and_mode(adapter.model):
        adapter.eval()
        clean = encode_prompted_loader(adapter, clean_loader)
        gallery = encode_prompted_loader(adapter, gallery_loader, photo=True)
        clean_metrics = _metrics(clean, gallery, device=device, query_chunk=query_chunk_size)
        conditions: list[dict[str, Any]] = []
        for fraction in EVAL_FRACTIONS:
            for seed in EVAL_SEEDS:
                encoded, metas = _encode_masked_loader(
                    adapter, masked_loader_factory(fraction, seed)
                )
                row = _metrics(encoded, gallery, device=device, query_chunk=query_chunk_size)
                row.update({
                    "fraction": fraction,
                    "seed": seed,
                    "status_counts": _status_counts(metas),
                    "mask_metadata": {"count": len(metas), "metas": metas},
                })
                conditions.append(row)

    macro = {
        name: float(np.mean([row[name] for row in conditions]))
        for name in _RETRIEVAL_NAMES
    }
    by_fraction = {
        str(fraction): {
            name: float(np.mean([row[name] for row in conditions if row["fraction"] == fraction]))
            for name in _RETRIEVAL_NAMES
        }
        for fraction in EVAL_FRACTIONS
    }
    return {
        "benchmark": "SPICA_category_retrieval",
        "benchmark_status": "official_not_verified",
        "evaluation_scope": "pseudo_validation",
        "identities": {
            "query_ids": list(clean.paths),
            "query_labels": clean.labels.tolist(),
            "gallery_ids": list(gallery.paths),
            "gallery_labels": gallery.labels.tolist(),
        },
        "clean": clean_metrics,
        "conditions": conditions,
        "masked_macro": macro,
        "masked_by_fraction": by_fraction,
        "condition_count": len(conditions),
        "query_count": len(clean.paths),
        "gallery_count": len(gallery.paths),
        "query_identity_sha256": hashlib.sha256(json.dumps(clean.paths, separators=(",", ":")).encode()).hexdigest(),
        "gallery_identity_sha256": hashlib.sha256(json.dumps(gallery.paths, separators=(",", ":")).encode()).hexdigest(),
    }


def write_probe(path: Path, result: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "CoupledPredictiveAdapter", "EVAL_FRACTIONS", "EVAL_SEEDS",
    "evaluate_views", "preserve_rng_and_mode", "write_probe",
]
