"""Final-only official benchmark evaluation for coupled predictive runs.

The public entry point accepts an already constructed model and explicit loaders.
It never trains, selects a checkpoint, or materializes an ``N_query x N_gallery``
score matrix.  The command-line wrapper is intentionally responsible for the
run/checkpoint/data provenance gates; this module contains the reusable
bounded-memory evaluator and a CPU-only synthetic self-check.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..data.masking import MASK_POLICY_VERSION
from .coupled_predictive import QOnlyAdapter
from ..evaluation.embeddings import EncodedRetrievalSet
from ..evaluation.metrics import evaluate_category_retrieval_all_denominators

EVAL_FRACTIONS = (0.25, 0.5, 0.75)
EVAL_SEEDS = (101, 202, 303)
EVAL_CONDITIONS = tuple((fraction, seed) for fraction in EVAL_FRACTIONS for seed in EVAL_SEEDS)
QUERY_CHUNK_SIZE = 256
TOP_K = 200
METRICS = (
    "full_mAP",
    "P@200",
    "mAP@200_prefix_positive",
    "mAP@200_all_relevant",
    "mAP@200_min_relevant_k",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npy(path: Path, value: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value))
        handle.flush()
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(b"\0")
    digest.update(str(value.dtype).encode())
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def module_state_hash(module: nn.Module, *, prefix: str | None = None) -> str:
    """Reuse trainer state identity and independently reject nonfinite state."""
    from ..train_coupled_predictive import _state_hash

    for name, value in module.state_dict().items():
        if prefix is not None and not name.startswith(prefix):
            continue
        if not isinstance(value, Tensor):
            raise TypeError(f"state entry {name!r} contains a non-tensor")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"state entry {name!r} contains NaN or Inf")
    return _state_hash(module, prefix=prefix)


@contextmanager
def _preserve_runtime(model: Any):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = {module: bool(module.training) for module in model.modules()}
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for module, mode in modes.items():
            module.training = mode


def _batch_images(batch: Mapping[str, Any], device: torch.device) -> Tensor:
    images = batch.get("image")
    if not isinstance(images, Tensor) or images.ndim != 4:
        raise ValueError("loader batches must contain image [B,3,H,W] tensors")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("loader images contain NaN or Inf")
    return images.to(device=device, non_blocking=device.type == "cuda")


def _batch_labels_paths(batch: Mapping[str, Any], size: int) -> tuple[Tensor, tuple[str, ...]]:
    labels = batch.get("label")
    paths = batch.get("path")
    if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.shape[0] != size:
        raise ValueError("loader labels must be a [B] tensor")
    if not isinstance(paths, (list, tuple)) or len(paths) != size:
        raise ValueError("loader paths must contain one item per image")
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise ValueError("loader labels must use an integer dtype")
    labels = labels.detach().cpu().long()
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("loader labels are not finite")
    return labels, tuple(str(path) for path in paths)


def _encode_loader(
    adapter: QOnlyAdapter,
    loader: Any,
    *,
    device: torch.device,
    gallery: bool,
    metadata_path: Path,
    require_mask_metadata: bool = False,
) -> EncodedRetrievalSet:
    """Encode one loader; only embeddings are retained and score chunks stay bounded."""
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    paths: list[str] = []
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as metadata_file:
            with torch.inference_mode():
                for batch in loader:
                    if not isinstance(batch, Mapping):
                        raise TypeError("loader batch must be a mapping")
                    images = _batch_images(batch, device)
                    if gallery:
                        encoded = adapter.encode_photo(images)
                    else:
                        encoded = adapter(images)
                    if not isinstance(encoded, Tensor) or encoded.ndim != 2 or encoded.shape[0] != images.shape[0]:
                        raise ValueError("model encoder must return [B,D] embeddings")
                    encoded = encoded.float()
                    if not bool(torch.isfinite(encoded).all()):
                        raise ValueError("model encoder returned NaN or Inf")
                    batch_labels, batch_paths = _batch_labels_paths(batch, images.shape[0])
                    embeddings.append(encoded.detach().cpu())
                    labels.append(batch_labels)
                    paths.extend(batch_paths)
                    raw_meta = batch.get("mask_meta")
                    if raw_meta is None:
                        raw_meta = batch.get("mask_metadata")
                    if require_mask_metadata and raw_meta is None:
                        raise ValueError("masked loader must provide mask metadata")
                    if raw_meta is not None:
                        if not isinstance(raw_meta, (list, tuple)) or len(raw_meta) != len(batch_paths):
                            raise ValueError("mask metadata must align with the query batch")
                        for meta in raw_meta:
                            if not isinstance(meta, Mapping) or str(meta.get("status", "")) not in {"ok", "blank_input", "target_unreachable", "zero_fraction"}:
                                raise ValueError("mask metadata has an unknown status")
                    for index, (path, label) in enumerate(zip(batch_paths, batch_labels.tolist(), strict=True)):
                        row: dict[str, Any] = {"index": count + index, "path": path, "label": int(label)}
                        if raw_meta is not None:
                            row["mask"] = raw_meta[index]
                        metadata_file.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
                    count += len(batch_paths)
            metadata_file.flush()
        if not embeddings:
            raise ValueError("cannot encode an empty loader")
        temporary.replace(metadata_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return EncodedRetrievalSet(
        embeddings=torch.cat(embeddings, dim=0).float(),
        labels=torch.cat(labels, dim=0).long(),
        paths=tuple(paths),
    )


def _validate_same_order(reference: EncodedRetrievalSet, candidate: EncodedRetrievalSet, name: str) -> None:
    if candidate.paths != reference.paths or not torch.equal(candidate.labels, reference.labels):
        raise ValueError(f"{name} query order or labels differ from clean control")


def _metric_record(evaluations: Mapping[str, Any], gallery: EncodedRetrievalSet, queries: EncodedRetrievalSet) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    prefix = evaluations["prefix_positive"]
    metrics = prefix.metrics
    top = prefix.top_indices.detach().cpu().long()
    relevant = gallery.labels[top].eq(queries.labels[:, None])
    p200 = relevant.float().mean(dim=1)
    scalar = {
        "full_mAP": float(metrics.mean_average_precision),
        "P@200": float(metrics.precision_at_k[200]),
        "mAP@200_prefix_positive": float(metrics.mean_average_precision_at_k[200]),
        "mAP@200_all_relevant": float(evaluations["all_relevant"].metrics.mean_average_precision_at_k[200]),
        "mAP@200_min_relevant_k": float(evaluations["min_relevant_k"].metrics.mean_average_precision_at_k[200]),
    }
    arrays = {
        "average_precision": prefix.average_precision_per_query.numpy().astype(np.float32, copy=False),
        "P200": p200.numpy().astype(np.float32, copy=False),
        "AP200_prefix_positive": prefix.average_precision_at_k_per_query[200].numpy().astype(np.float32, copy=False),
        "AP200_all_relevant": evaluations["all_relevant"].average_precision_at_k_per_query[200].numpy().astype(np.float32, copy=False),
        "AP200_min_relevant_k": evaluations["min_relevant_k"].average_precision_at_k_per_query[200].numpy().astype(np.float32, copy=False),
        "top200": top.numpy().astype(np.int32, copy=False),
    }
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("retrieval metrics contain NaN or Inf")
    return scalar, arrays


def _evaluate_condition(
    adapter: QOnlyAdapter,
    queries: EncodedRetrievalSet,
    gallery: EncodedRetrievalSet,
    *,
    device: torch.device,
    output_dir: Path,
    condition_name: str,
    mask_metadata: Path,
) -> dict[str, Any]:
    query_path = output_dir / f"{condition_name}_query_embeddings.npy"
    arrays_path = output_dir / f"{condition_name}_per_query.npz"
    metadata_hash = _sha256_file(mask_metadata)
    _write_npy(query_path, queries.embeddings.numpy().astype(np.float32, copy=False))
    evaluations = evaluate_category_retrieval_all_denominators(
        queries,
        gallery,
        precision_at_k=(200,),
        map_at_k=(200,),
        query_chunk_size=QUERY_CHUNK_SIZE,
        top_k=TOP_K,
        device=device,
    )
    scalar, arrays = _metric_record(evaluations, gallery, queries)
    _write_npz(arrays_path, **arrays)
    with mask_metadata.open(encoding='utf-8') as handle:
        status_counts = dict(sorted(Counter(json.loads(line).get('mask', {}).get('status', 'unmasked') for line in handle).items()))
    assert sum(status_counts.values()) == len(queries.paths)
    artifact = output_dir / f"{condition_name}_summary.json"
    artifact_value = {"condition": condition_name, "metrics": scalar, "query_count": len(queries.paths), "gallery_count": len(gallery.paths), "mask_metadata_sha256": metadata_hash, "query_embeddings_sha256": _sha256_file(query_path), "per_query_sha256": _sha256_file(arrays_path), "mask_status_counts": status_counts}
    _write_json(artifact, artifact_value)
    return {
        "condition": condition_name,
        "metrics": scalar,
        "summary_artifact": artifact.name,
        "summary_artifact_sha256": _sha256_file(artifact),
        "query_count": len(queries.paths),
        "gallery_count": len(gallery.paths),
        "query_embeddings": str(query_path.name),
        "per_query": str(arrays_path.name),
        "mask_metadata": str(mask_metadata.name),
        "mask_metadata_sha256": metadata_hash,
        "mask_status_counts": status_counts,
    }


def evaluate_coupled_benchmark(
    model: Any,
    clean_query_loader: Any,
    gallery_loader: Any,
    masked_loader_factory: Callable[[float, int], Any],
    output_dir: Path,
    *,
    device: str | torch.device | None = None,
    benchmark: str,
    protocol_identity: Mapping[str, Any],
    class_names: Mapping[int, str],
    source_hash: str,
    checkpoint_selection: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    status: str = "COMPLETE",
) -> dict[str, Any]:
    """Evaluate clean and the fixed nine masks using an explicit loader API.

    ``gallery_loader`` is consumed exactly once.  Each query loader is consumed
    once, in the fixed condition order.  The returned object is a compact
    summary; embeddings, per-query arrays, and mask JSONL are files in
    ``output_dir``.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    adapter = QOnlyAdapter(model)
    actual_device = torch.device(device or adapter.device)
    expected_device = torch.device(adapter.device)
    if actual_device.type == "cuda" and actual_device.index is None:
        actual_device = torch.device("cuda", torch.cuda.current_device())
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())
    if actual_device != expected_device:
        raise ValueError(f"model is on {adapter.device}, but evaluator device is {actual_device}")
    if status not in {"COMPLETE", "TRAIN_ONLY_PREFLIGHT_NOT_OFFICIAL"}:
        raise ValueError(f"unsupported evaluation status: {status!r}")
    if not source_hash or not isinstance(data_identity, Mapping):
        raise ValueError("source_hash and data_identity are required provenance fields")
    before = module_state_hash(model)
    runtime = _preserve_runtime(model)
    runtime.__enter__()
    predictor_calls: list[int] = []
    hook = None
    predictor = getattr(model, "predictor", None)
    if isinstance(predictor, nn.Module):
        hook = predictor.register_forward_hook(lambda *_args: predictor_calls.append(1))
    if actual_device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(actual_device)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "RUNNING",
        "official_unseen_used_for_selection": False,
        "official_unseen_used_for_training": False,
        "official_test_evaluated": status == "COMPLETE",
        "official_test_loader_opened": status == "COMPLETE",
        "evaluation_scope": "official_unseen_final" if status == "COMPLETE" else "train_only_preflight",
        "benchmark": benchmark,
        "source_hash": str(source_hash),
        "architecture": "predictive_fusion_v2",
        "class_names": {str(key): str(value) for key, value in sorted(class_names.items())},
        "data_identity": data_identity,
        "config": {
            "architecture": "predictive_fusion_v2",
            "class_names": {str(key): str(value) for key, value in sorted(class_names.items())},
            "data_identity": data_identity,
        },
        "protocol_identity": protocol_identity,
        "checkpoint_selection": dict(checkpoint_selection),
        "checkpoint_selections": {"latest": dict(checkpoint_selection.get("latest", checkpoint_selection))},
        "mask_policy": {
            "version": MASK_POLICY_VERSION,
            "fractions": list(EVAL_FRACTIONS),
            "seeds": list(EVAL_SEEDS),
            "order": "fraction-major then seed-major",
        },
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "gallery_encoded_once": True,
        "conditions": [],
    }
    try:
        adapter.eval()
        gallery = _encode_loader(
            adapter, gallery_loader, device=actual_device, gallery=True,
            metadata_path=output_dir / "gallery_metadata.jsonl",
        )
        if len(gallery.paths) < TOP_K:
            raise ValueError(f"gallery must contain at least {TOP_K} items")
        _write_npy(output_dir / "gallery_embeddings.npy", gallery.embeddings.numpy().astype(np.float32, copy=False))
        _write_npy(output_dir / "gallery_labels.npy", gallery.labels.numpy().astype(np.int64, copy=False))
        clean = _encode_loader(
            adapter, clean_query_loader, device=actual_device, gallery=False,
            metadata_path=output_dir / "clean_mask_metadata.jsonl",
        )
        if len(clean.paths) == 0:
            raise ValueError("clean query loader is empty")
        if not set(clean.labels.tolist()) <= set(gallery.labels.tolist()):
            raise ValueError('every query class must have gallery positives')
        summary['gallery_embeddings_sha256'] = _sha256_file(output_dir / 'gallery_embeddings.npy')
        summary['gallery_labels_sha256'] = _sha256_file(output_dir / 'gallery_labels.npy')
        summary['gallery_metadata_sha256'] = _sha256_file(output_dir / 'gallery_metadata.jsonl')
        summary["query_count"] = len(clean.paths)
        summary["gallery_count"] = len(gallery.paths)
        summary["conditions"].append(_evaluate_condition(
            adapter, clean, gallery, device=actual_device, output_dir=output_dir,
            condition_name="clean", mask_metadata=output_dir / "clean_mask_metadata.jsonl",
        ))
        for fraction, seed in EVAL_CONDITIONS:
            condition_name = f"mask_f{int(fraction * 100):02d}_s{seed}"
            masked = _encode_loader(
                adapter, masked_loader_factory(fraction, seed), device=actual_device, gallery=False,
                metadata_path=output_dir / f"{condition_name}_mask_metadata.jsonl",
                require_mask_metadata=True,
            )
            _validate_same_order(clean, masked, condition_name)
            summary["conditions"].append(_evaluate_condition(
                adapter, masked, gallery, device=actual_device, output_dir=output_dir,
                condition_name=condition_name,
                mask_metadata=output_dir / f"{condition_name}_mask_metadata.jsonl",
            ))
        if len(summary["conditions"]) != 10:
            raise AssertionError("wrong number of evaluation conditions")
        after = module_state_hash(model)
        if before != after:
            raise RuntimeError("model state changed during evaluation")
        if predictor_calls:
            raise RuntimeError(f"q-only adapter invoked predictor {len(predictor_calls)} times")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("evaluation created parameter gradients")
        clean_metrics = summary["conditions"][0]["metrics"]
        masked_rows = summary["conditions"][1:]
        summary.update({
            "status": status,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_allocated": int(torch.cuda.max_memory_allocated(actual_device)) if actual_device.type == "cuda" else 0,
            "peak_gpu_memory_reserved": int(torch.cuda.max_memory_reserved(actual_device)) if actual_device.type == "cuda" else 0,
            "clean": clean_metrics,
            "masked_macro": {name: float(np.mean([row["metrics"][name] for row in masked_rows])) for name in METRICS},
            "masked_by_fraction": {str(fraction): {name: float(np.mean([row["metrics"][name] for row in masked_rows if row["condition"].startswith(f"mask_f{int(fraction * 100):02d}_")])) for name in METRICS} for fraction in EVAL_FRACTIONS},
            "source_snapshot_hash": str(source_hash),
            "step": int(checkpoint_selection.get("step", checkpoint_selection.get("latest", {}).get("step", -1))),
            "model_state_before": before,
            "model_state_after": after,
            "predictor_forwards": len(predictor_calls),
            "query_order": "clean then fraction-major/seed-major masks",
        })
        _write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "source_hash": str(source_hash),
        }
        if not (output_dir / "failure.json").exists():
            _write_json(output_dir / "failure.json", failure)
        raise
    finally:
        if hook is not None:
            hook.remove()
        runtime.__exit__(None, None, None)


def _numpy_metrics(query: np.ndarray, query_labels: np.ndarray, gallery: np.ndarray, gallery_labels: np.ndarray) -> dict[str, np.ndarray | float]:
    scores = query @ gallery.T
    order = np.argsort(-scores, axis=1, kind="stable")
    relevant = gallery_labels[order] == query_labels[:, None]
    positions = np.arange(1, gallery.shape[0] + 1, dtype=np.float64)
    precision = np.cumsum(relevant, axis=1) / positions
    positives = relevant.sum(axis=1)
    ap = (precision * relevant).sum(axis=1) / positives
    top = relevant[:, :200]
    p200 = top.mean(axis=1)
    ap_values: dict[str, np.ndarray] = {}
    for name in ("prefix_positive", "all_relevant", "min_relevant_k"):
        prefix = relevant[:, :200]
        numerator = (precision[:, :200] * prefix).sum(axis=1)
        if name == "prefix_positive":
            denominator = prefix.sum(axis=1)
        elif name == "all_relevant":
            denominator = positives
        else:
            denominator = np.minimum(positives, 200)
        ap_values[name] = numerator / np.maximum(denominator, 1)
    return {
        "full_mAP": float(ap.mean()),
        "P@200": float(p200.mean()),
        "mAP@200_prefix_positive": float(ap_values["prefix_positive"].mean()),
        "mAP@200_all_relevant": float(ap_values["all_relevant"].mean()),
        "mAP@200_min_relevant_k": float(ap_values["min_relevant_k"].mean()),
        "average_precision": ap,
        "P200": p200,
        "AP200_prefix_positive": ap_values["prefix_positive"],
        "AP200_all_relevant": ap_values["all_relevant"],
        "AP200_min_relevant_k": ap_values["min_relevant_k"],
        "top200": order[:, :200],
    }


class _SelfCheckModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.pooled_head = nn.Identity()
        self.predictor = nn.Linear(width, width)

    def _context_tokens(self, images: Tensor) -> Tensor:
        return images[:, :1, 0, :8]

    def encode_photo(self, images: Tensor) -> Tensor:
        return F.normalize(images[:, 0, 0, :8], dim=-1)


def _fixture_loaders() -> tuple[_SelfCheckModel, DataLoader, DataLoader, Callable[[float, int], DataLoader]]:
    model = _SelfCheckModel().eval()
    query_labels = torch.arange(32, dtype=torch.long) % 8
    gallery_labels = torch.arange(256, dtype=torch.long) % 8
    query_images = torch.zeros(32, 3, 1, 8)
    gallery_images = torch.zeros(256, 3, 1, 8)
    query_images[torch.arange(32), 0, 0, query_labels] = 1.0
    gallery_images[torch.arange(256), 0, 0, gallery_labels] = 1.0

    def loader(images: Tensor, labels: Tensor) -> DataLoader:
        rows = [{"image": images[i], "label": int(labels[i]), "path": f"fixture/{i}.png"} for i in range(len(labels))]
        def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "image": torch.stack([item["image"] for item in items]),
                "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
                "path": tuple(item["path"] for item in items),
            }
        return DataLoader(rows, batch_size=7, shuffle=False, collate_fn=collate)

    clean = loader(query_images, query_labels)
    gallery = loader(gallery_images, gallery_labels)

    def masked(fraction: float, seed: int) -> DataLoader:
        del fraction
        changed = query_images.clone()
        if seed % 2:
            changed[:, 0, 0, 0] = 0.0
        rows = [
            {"image": changed[i], "label": int(query_labels[i]), "path": f"fixture/{i}.png", "mask_meta": {"seed": seed, "status": "ok"}}
            for i in range(len(query_labels))
        ]
        def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "image": torch.stack([item["image"] for item in items]),
                "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
                "path": tuple(item["path"] for item in items),
                "mask_meta": [item["mask_meta"] for item in items],
            }
        return DataLoader(rows, batch_size=7, shuffle=False, collate_fn=collate)

    return model, clean, gallery, masked


def cpu_self_check() -> dict[str, Any]:
    """Run a no-data/no-network check of ordering, formulas, and q bypass."""
    with tempfile.TemporaryDirectory(prefix="spica_coupled_benchmark_selfcheck_") as raw:
        output = Path(raw) / "evaluation"
        model, clean, gallery, masked = _fixture_loaders()
        before = module_state_hash(model)
        result = evaluate_coupled_benchmark(
            model,
            clean,
            gallery,
            masked,
            output,
            device="cpu",
            benchmark="synthetic_fixture",
            protocol_identity={"fixture": True},
            class_names={i: f"class_{i}" for i in range(8)},
            source_hash="fixture-source-hash",
            checkpoint_selection={"step": 2, "path": "checkpoint_step2.pt", "sha256": "fixture"},
            data_identity={"fixture": True},
            status="TRAIN_ONLY_PREFLIGHT_NOT_OFFICIAL",
        )
        assert result["status"] == "TRAIN_ONLY_PREFLIGHT_NOT_OFFICIAL"
        assert result["predictor_forwards"] == 0
        assert module_state_hash(model) == before
        with np.load(output / "clean_per_query.npz") as saved:
            query = np.load(output / "clean_query_embeddings.npy")
            gallery_array = np.load(output / "gallery_embeddings.npy")
            expected = _numpy_metrics(
                query, np.arange(32) % 8, gallery_array, np.arange(256) % 8
            )
            clean_metrics = result["conditions"][0]["metrics"]
            for name in METRICS:
                assert abs(float(clean_metrics[name]) - float(expected[name])) < 1e-6, name
            for name in ("average_precision", "P200", "AP200_prefix_positive", "AP200_all_relevant", "AP200_min_relevant_k"):
                assert np.allclose(saved[name], expected[name], atol=1e-6, rtol=1e-6)
            assert np.array_equal(saved["top200"], expected["top200"])
        # The independent order guard must reject a reordered masked loader.
        def bad_mask(fraction: float, seed: int) -> DataLoader:
            del fraction, seed
            rows = list(reversed([{"image": torch.zeros(3, 1, 8), "label": int(i % 8), "path": f"fixture/{i}.png"} for i in range(32)]))
            def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
                return {
                    "image": torch.stack([item["image"] for item in items]),
                    "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
                    "path": tuple(item["path"] for item in items),
                }
            return DataLoader(rows, batch_size=8, shuffle=False, collate_fn=collate)
        try:
            evaluate_coupled_benchmark(
                model, clean, gallery, bad_mask, Path(raw) / "bad",
                device="cpu", benchmark="synthetic_fixture", protocol_identity={"fixture": True},
                class_names={i: f"class_{i}" for i in range(8)}, source_hash="fixture",
                checkpoint_selection={"step": 2}, data_identity={"fixture": True},
                status="TRAIN_ONLY_PREFLIGHT_NOT_OFFICIAL",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("reordered masked loader was accepted")
        return {
            "status": "PASS",
            "official_image_loader_opened": False,
            "fullsort_independent_numpy_match": True,
            "conditions": 10,
            "query_chunk_size": QUERY_CHUNK_SIZE,
            "predictor_forwards": 0,
            "model_state_unchanged": True,
            "order_guard": True,
        }


__all__ = [
    "EVAL_FRACTIONS",
    "EVAL_SEEDS",
    "QOnlyAdapter",
    "cpu_self_check",
    "evaluate_coupled_benchmark",
    "module_state_hash",
]
