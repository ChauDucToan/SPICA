"""Small, deterministic fixed-seen retrieval probe for training."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import random
from typing import Any, Callable, Sequence
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from ..data.datasets import _load_rgb_image
from .coupled_predictive import QOnlyAdapter
from .embeddings import EncodedRetrievalSet
from .masked_view import _mask_image, _transform_stats
from .metrics import evaluate_category_retrieval
FRACTIONS = (0.25, 0.5, 0.75)
SEEDS = (101, 202, 303)

@contextmanager
def _preserved(model: Any = None):
    states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = tuple((module, module.training) for module in model.modules()) if model is not None else ()
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.random.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for module, training in modes:
            module.training = training

class TrainingProbe:
    def __init__(
        self,
        query_images: Tensor,
        gallery_images: Tensor,
        query_labels: Tensor | Sequence[int],
        gallery_labels: Tensor | Sequence[int],
        *,
        query_ids: Sequence[str] | None = None,
        gallery_ids: Sequence[str] | None = None,
        masked_queries: dict[tuple[float, int], Tensor] | None = None,
    ) -> None:
        self.query_images = self._images(query_images, "query_images")
        self.clean_queries = self.query_images
        self.gallery_images = self._images(gallery_images, "gallery_images")
        self.query_labels = torch.as_tensor(query_labels, dtype=torch.long).flatten().clone()
        self.gallery_labels = torch.as_tensor(gallery_labels, dtype=torch.long).flatten().clone()
        if (len(self.query_labels), len(self.gallery_labels)) != (32, 256):
            raise ValueError("TrainingProbe requires exactly 32 queries and 256 gallery labels")
        if (len(self.clean_queries), len(self.gallery_images)) != (32, 256):
            raise ValueError("TrainingProbe requires exactly 32 queries and 256 gallery images")
        query_values = tuple(self.query_labels.tolist())
        gallery_values = tuple(self.gallery_labels.tolist())
        if len(set(query_values)) != 32 or tuple(sorted(query_values)) != query_values:
            raise ValueError("queries must contain one sorted item per class")
        if any(gallery_values.count(label) != 8 for label in query_values):
            raise ValueError("gallery must contain eight photos for every query class")
        self.query_ids = tuple(query_ids or (f"query/{i}" for i in range(32)))
        self.gallery_ids = tuple(gallery_ids or (f"gallery/{i}" for i in range(256)))
        if (len(self.query_ids), len(self.gallery_ids)) != (32, 256):
            raise ValueError("query_ids/gallery_ids do not match image counts")
        if len(set(self.query_ids)) != 32 or len(set(self.gallery_ids)) != 256:
            raise ValueError("query and gallery IDs must be unique")
        if masked_queries is None:
            raise ValueError("masked_queries must be supplied explicitly")
        self.masked_queries = self._make_masks(masked_queries)
        self._p_all = sum(gallery_values.count(label) for label in query_values) / (32 * 256)
        manifest = {"query_ids": self.query_ids, "query_labels": self.query_labels.tolist(),
                    "gallery_ids": self.gallery_ids, "gallery_labels": self.gallery_labels.tolist()}
        self.manifest_hash = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    @classmethod
    def from_protocol(cls, protocol: Any, transform: Callable[[Image.Image], Tensor]) -> "TrainingProbe":
        split = getattr(protocol, "train", None)
        if split is None:
            raise TypeError("protocol must expose a train split")
        by_sketch: dict[int, list[Any]] = {}
        by_photo: dict[int, list[Any]] = {}
        for entry in split.sketch_entries:
            by_sketch.setdefault(int(entry.label), []).append(entry)
        for entry in split.photo_entries:
            by_photo.setdefault(int(entry.label), []).append(entry)
        classes = sorted(set(by_sketch) & set(by_photo))[:32]
        if len(classes) != 32 or any(len(by_photo[label]) < 8 for label in classes):
            raise ValueError("training protocol must contain 32 classes and eight photos per class")
        root = getattr(protocol, "root", None)
        def ident(entry: Any) -> str:
            path = entry.path
            if root is not None:
                try:
                    return path.resolve().relative_to(root.resolve()).as_posix()
                except ValueError:
                    pass
            return str(path)
        with _preserved():
            selected_q = [sorted(by_sketch[label], key=lambda e: str(e.path))[0] for label in classes]
            selected_g = [e for label in classes for e in
                          sorted(by_photo[label], key=lambda e: str(e.path))[:8]]
            q = torch.stack([transform(_load_rgb_image(e)).detach().cpu() for e in selected_q])
            g = torch.stack([transform(_load_rgb_image(e)).detach().cpu() for e in selected_g])
            query_ids = [ident(e) for e in selected_q]
            mean, std = _transform_stats(transform)
            masked = {(fraction, seed): torch.stack([
                _mask_image(image, fraction=fraction, base_seed=seed, sample_key=key,
                            mean=mean, std=std, ink_threshold=0.9)[0]
                for image, key in zip(q, query_ids, strict=True)])
                for fraction in FRACTIONS for seed in SEEDS}
            return cls(q, g, torch.tensor(classes),
                       torch.tensor([e.label for e in selected_g]),
                       query_ids=query_ids, gallery_ids=[ident(e) for e in selected_g],
                       masked_queries=masked)
    @staticmethod
    def _images(value: Tensor, name: str) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(f"{name} must have shape [N, 3, H, W]")
        if (value.device.type != "cpu" or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{name} must be finite floating-point CPU tensors")
        return value.detach().clone().contiguous()
    def _make_masks(self, supplied: dict[tuple[float, int], Tensor]) -> dict[tuple[float, int], Tensor]:
        result = {key: self._images(value, f"masked_queries[{key}]")
                  for key, value in supplied.items()}
        expected = {(fraction, seed) for fraction in FRACTIONS for seed in SEEDS}
        if set(result) != expected:
            raise ValueError("masked_queries must contain the fixed nine conditions")
        if any(value.shape != self.clean_queries.shape for value in result.values()):
            raise ValueError("masked query tensors must match clean query shape")
        return result
    @staticmethod
    def _metrics(query: Tensor, qlabels: Tensor, gallery: Tensor, glabels: Tensor,
                 device: torch.device) -> dict[str, float]:
        evaluation = evaluate_category_retrieval(
            EncodedRetrievalSet(query, qlabels, tuple(map(str, range(len(query))))),
            EncodedRetrievalSet(gallery, glabels, tuple(map(str, range(len(gallery))))),
            precision_at_k=(200,), map_at_k=(200,), top_k=200,
            map_at_k_denominator="min_relevant_k", device=device)
        metrics = evaluation.metrics
        return {"mAP@200": float(metrics.mean_average_precision_at_k[200]),
                "mAP@all": float(metrics.mean_average_precision),
                "P@200": float(metrics.precision_at_k[200])}
    def __call__(self, model: Any, device: str | torch.device) -> dict[str, Any]:
        device = torch.device(device)
        with _preserved(model), torch.inference_mode():
            adapter = QOnlyAdapter(model)
            model.eval()
            queries = adapter(self.clean_queries.to(device)).float().cpu()
            masked_outputs = {key: adapter(images.to(device)).float().cpu()
                              for key, images in self.masked_queries.items()}
            gallery = torch.cat([adapter.encode_photo(batch.to(device)).float().cpu()
                                 for batch in self.gallery_images.split(64)])
            clean = self._metrics(queries, self.query_labels, gallery, self.gallery_labels, device)
            rows = [self._metrics(masked_outputs[(f, s)], self.query_labels, gallery,
                                  self.gallery_labels, device)
                    for f in FRACTIONS for s in SEEDS]
            masked = {name: float(sum(row[name] for row in rows) / len(rows)) for name in clean}
        return {"cleaned": clean, "masked": masked, "summary": self.summary}
    @property
    def summary(self) -> dict[str, Any]:
        return {"scope": "seen_train_probe_not_validation", "query_count": 32, "gallery_count": 256,
                "query_ids": list(self.query_ids), "gallery_ids": list(self.gallery_ids),
                "manifest_hash": self.manifest_hash,
                "P@all": {"clean": self._p_all, "masked": self._p_all}}

__all__ = ["TrainingProbe", "FRACTIONS", "SEEDS"]
