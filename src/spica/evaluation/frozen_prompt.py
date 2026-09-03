"""Evaluation and cache helpers for frozen-backbone prompt experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .embeddings import EncodedRetrievalSet
from .jepa import feature_geometry
from .metrics import CategoryRetrievalEvaluation, evaluate_category_retrieval


def hash_state(values: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(values[name].detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def cache_identity(
    *,
    prompt_checkpoint_hash: str,
    prompt_length: int,
    prompt_mode: str,
    modality: str,
    model_name: str,
    pretrained: str | None,
    data_manifest_identity: dict[str, Any],
) -> dict[str, Any]:
    if modality not in {"sketch", "photo"}:
        raise ValueError("modality must be sketch or photo")
    return {
        "prompt_checkpoint_hash": prompt_checkpoint_hash,
        "prompt_length": prompt_length,
        "prompt_mode": prompt_mode,
        "modality": modality,
        "model_name": model_name,
        "pretrained": pretrained,
        "data_manifest_identity": data_manifest_identity,
    }


def save_prompt_cache(
    encoded: EncodedRetrievalSet,
    path: Path,
    *,
    identity: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "embeddings": encoded.embeddings,
            "labels": encoded.labels,
            "paths": encoded.paths,
            "identity": identity,
        },
        path,
    )


def load_prompt_cache(path: Path, *, expected_identity: dict[str, Any]) -> EncodedRetrievalSet:
    if not path.is_file():
        raise FileNotFoundError(f"prompted gallery cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported prompted gallery cache")
    if payload.get("identity") != expected_identity:
        raise ValueError("prompted gallery cache identity does not match this run")
    return EncodedRetrievalSet(
        embeddings=payload["embeddings"].float(),
        labels=payload["labels"].long(),
        paths=tuple(str(value) for value in payload["paths"]),
        metadata={"prompt_cache_identity": json.dumps(expected_identity, sort_keys=True)},
    )


def encode_prompted_loader(model: Any, loader: DataLoader, *, photo: bool = False) -> EncodedRetrievalSet:
    model.eval()
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(model.device, non_blocking=model.device.type == "cuda")
            values = model.encode_photo(images) if photo else model(images)
            embeddings.append(values.float().cpu())
            labels.append(batch["label"].long().cpu())
            paths.extend(str(path) for path in batch["path"])
    if not embeddings:
        raise ValueError("cannot encode an empty loader")
    return EncodedRetrievalSet(torch.cat(embeddings), torch.cat(labels), tuple(paths))


def evaluate_prompted(
    queries: EncodedRetrievalSet,
    gallery: EncodedRetrievalSet,
    *,
    query_chunk_size: int,
    device: torch.device,
) -> CategoryRetrievalEvaluation:
    return evaluate_category_retrieval(
        queries,
        gallery,
        precision_at_k=(1, 5, 10, 100, 200),
        map_at_k=(200,),
        map_at_k_denominator="prefix_positive",
        query_chunk_size=query_chunk_size,
        top_k=200,
        device=device,
    )


def cross_modal_geometry(
    sketches: EncodedRetrievalSet,
    photos: EncodedRetrievalSet,
) -> dict[str, float]:
    sketch = F.normalize(sketches.embeddings, dim=-1)
    photo = F.normalize(photos.embeddings, dim=-1)
    photo_labels = torch.unique(photos.labels, sorted=True)
    centroids = torch.stack(
        [F.normalize(photo[photos.labels == label].mean(0), dim=-1) for label in photo_labels]
    )
    positions = torch.searchsorted(photo_labels, sketches.labels)
    same = (sketch * centroids[positions]).sum(-1)
    different_values: list[Tensor] = []
    for label in torch.unique(sketches.labels, sorted=True):
        query = sketch[sketches.labels == label]
        negative = photo[photos.labels != label]
        different_values.append((query @ negative.T).mean(-1))
    different = torch.cat(different_values)
    return {
        "same_class_sketch_photo_cosine": same.mean().item(),
        "different_class_sketch_photo_cosine": different.mean().item(),
        "semantic_margin": (same.mean() - different.mean()).item(),
    }


def reference_preservation(
    current: EncodedRetrievalSet,
    reference: EncodedRetrievalSet,
) -> float:
    if current.paths != reference.paths or current.labels.shape != reference.labels.shape:
        raise ValueError("reference and current retrieval sets are not aligned")
    return F.cosine_similarity(current.embeddings, reference.embeddings, dim=-1).mean().item()


def geometry_payload(
    sketches: EncodedRetrievalSet,
    photos: EncodedRetrievalSet,
    *,
    sketch_reference: EncodedRetrievalSet,
    photo_reference: EncodedRetrievalSet,
    max_samples: int = 512,
) -> dict[str, Any]:
    return {
        "sketch": feature_geometry(sketches.embeddings, max_samples=max_samples, pair_samples=2048).to_dict(),
        "photo": feature_geometry(photos.embeddings, max_samples=max_samples, pair_samples=2048).to_dict(),
        "cross_modal": cross_modal_geometry(sketches, photos),
        "reference_preservation": {
            "sketch": reference_preservation(sketches, sketch_reference),
            "photo": reference_preservation(photos, photo_reference),
        },
    }
