"""REAL-CLIP coupled-predictive V1, no-update GPU preflight.

This script is intentionally a preflight, not a trainer: it loads one real
batch from the approved S0 data path, builds V1, backpropagates once, and
writes auditable phase receipts.  It never creates an optimizer or evaluates
an official gallery.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

# These are part of the execution contract, and must be set before torch or
# OpenCLIP is imported.  No credentials are collected or emitted.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = "ViT-B-32-quickgelu"
EXPECTED_CLIP_SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
EXPECTED_CLIP_BYTES = 605143284
CLIP_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai"
    / "snapshots/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors"
)
EXPECTED_TRAIN_CLASSES = tuple(
    value
    for value in range(104)
    if value not in {7, 13, 16, 27, 28, 31, 33, 34, 39, 42, 45, 51, 52, 53, 60, 65, 75, 86, 90, 99}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor_hash expects a tensor")
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(b"\0")
    digest.update(str(value.dtype).encode())
    digest.update(b"\0")
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def snapshot_state(module: Any) -> dict[str, dict[str, Any]]:
    """Hash state one tensor at a time; do not retain a CPU state copy."""
    import torch

    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_hash(value),
        }
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def snapshot_parameters(module: Any) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "sha256": tensor_hash(parameter),
        }
        for name, parameter in module.named_parameters()
    }


def _same_snapshot(before: Mapping[str, Any], after: Mapping[str, Any], name: str) -> None:
    if before != after:
        changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
        raise AssertionError(f"{name} changed without an optimizer step: {changed[:5]}")


def _finite(value: Any, name: str) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
        raise AssertionError(f"{name} is nonfinite")


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _safe_git_state() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD").strip(),
        "status": _git("status", "--porcelain", "--untracked-files=all").splitlines(),
    }


def _archive_source_snapshot(out: Path, snapshot: Mapping[str, Any]) -> None:
    files = snapshot["manifest"]
    if not files or len(files) != snapshot["file_count"]:
        raise AssertionError("source inventory is empty or incomplete")
    destination_root = out / "source_snapshot" / "files"
    index: list[dict[str, Any]] = []
    for item in files:
        relative = Path(str(item["path"]))
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"source disappeared while archiving: {source}")
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        actual = sha256_file(destination)
        if actual != str(item["sha256"]):
            raise AssertionError(f"source archive hash mismatch: {relative}")
        index.append({"path": str(relative), "sha256": actual, "bytes": destination.stat().st_size})
    _json_write(out / "source_snapshot" / "index.json", {
        "sha256": snapshot["sha256"],
        "file_count": len(index),
        "files": index,
    })


def _phase(out: Path, name: str, function: Callable[[], Any]) -> Any:
    report = out / f"phase_{name}.json"
    _json_write(report, {"phase": name, "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        value = function()
    except Exception as error:
        import torch
        memory = {}
        if torch.cuda.is_initialized():
            memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "allocated_bytes": torch.cuda.memory_allocated(),
                      "reserved_bytes": torch.cuda.memory_reserved(),
                      "device_index": torch.cuda.current_device()}
        _json_write(report, {
            "phase": name,
            "failure_memory": memory,
            "status": "FAIL",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        })
        raise
    _json_write(report, {
        "phase": name,
        "status": "PASS",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "result": value,
    })
    return value


def _relative(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _duplicate_check(existing: Mapping[str, Any], tensor: Any, label: int, path: str) -> None:
    import torch

    if int(existing["label"]) != label:
        raise AssertionError(f"duplicate photo has conflicting labels: {path}")
    if not torch.equal(existing["tensor"], tensor):
        raise AssertionError(f"duplicate photo has different transformed image: {path}")


def pack_unique_photos(
    batch: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    data_root: Path,
) -> dict[str, Any]:
    """Pack exactly the batch's positive/negative photos, first occurrence wins.

    Returned ``bank_tensors`` is the only photo tensor bank; indices point into
    it.  This pure helper deliberately does not sample, pair, or read files.
    """
    import torch

    if len(trace) != 32:
        raise ValueError("trace must contain exactly 32 rows")
    positives = batch["positive_photos"]
    negatives = batch["negative_photo"]
    if not isinstance(positives, torch.Tensor) or positives.ndim != 5 or tuple(positives.shape[:2]) != (32, 1):
        raise ValueError("positive_photos must have shape [32, 1, C, H, W]")
    if not isinstance(negatives, torch.Tensor) or negatives.ndim != 4 or negatives.shape[0] != 32:
        raise ValueError("negative_photo must have shape [32, C, H, W]")

    ordered: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    positive_indices: list[int] = []
    negative_indices: list[list[int]] = []

    def add(path: str, tensor: Any, label: int) -> int:
        resolved = str(Path(path).resolve())
        previous = by_path.get(resolved)
        if previous is not None:
            _duplicate_check(previous, tensor, label, resolved)
            return int(previous["index"])
        row = {"index": len(ordered), "path": resolved, "label": int(label), "tensor": tensor.detach().clone()}
        by_path[resolved] = row
        ordered.append(row)
        return int(row["index"])

    for index, row in enumerate(trace):
        positives_for_row = row.get("positive_photo_paths")
        if not isinstance(positives_for_row, Sequence) or isinstance(positives_for_row, (str, bytes)) or len(positives_for_row) != 1:
            raise ValueError("trace must contain one positive path per query")
        positive_indices.append(add(str(positives_for_row[0]), positives[index, 0], int(row["label"])))
    for index, row in enumerate(trace):
        negative_indices.append([add(str(row["negative_photo_path"]), negatives[index], int(row["negative_label"]))])

    bank_paths = tuple(row["path"] for row in ordered)
    if len(set(bank_paths)) != len(bank_paths):
        raise AssertionError("unique photo bank contains duplicate paths")
    return {
        "bank_tensors": torch.stack([row["tensor"] for row in ordered]),
        "bank_labels": torch.tensor([row["label"] for row in ordered], dtype=torch.long),
        "positive_indices": torch.tensor(positive_indices, dtype=torch.long),
        "negative_indices": torch.tensor(negative_indices, dtype=torch.long),
        "photo_ids": tuple(_relative(path, data_root) for path in bank_paths),
        "photo_paths": bank_paths,
        "positive_paths": tuple(str(row["positive_photo_paths"][0]) for row in trace),
        "negative_paths": tuple(str(row["negative_photo_path"]) for row in trace),
    }


def _output_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"--output-dir must be fresh and nonexistent: {path}")
    path.mkdir(parents=True)
    return path


def _imports() -> dict[str, Any]:
    try:
        from preflight_semantic_text import (
            CLIP_PATH as semantic_clip_path,
            EXPECTED_CLIP_BYTES as semantic_clip_bytes,
            EXPECTED_CLIP_SHA256 as semantic_clip_sha256,
            batch_trace as semantic_batch_trace,
            compose_arm as semantic_compose_arm,
            seed42 as semantic_seed42,
            validate_data_and_pairing as semantic_validate_data_and_pairing,
        )
    except ImportError:
        from scripts.preflight_semantic_text import (
            CLIP_PATH as semantic_clip_path,
            EXPECTED_CLIP_BYTES as semantic_clip_bytes,
            EXPECTED_CLIP_SHA256 as semantic_clip_sha256,
            batch_trace as semantic_batch_trace,
            compose_arm as semantic_compose_arm,
            seed42 as semantic_seed42,
            validate_data_and_pairing as semantic_validate_data_and_pairing,
        )
    from spica.coupled_predictive_losses import coupled_region_loss as v1_coupled_region_loss
    from spica.data.coupled_views import region_pair
    from spica.models.coupled_predictive import CoupledPredictiveModel
    from spica.models.clip import load_frozen_clip
    from spica.provenance import source_snapshot
    import spica.train_frozen_prompt as trainer
    return {
        "semantic_clip_path": semantic_clip_path,
        "semantic_clip_bytes": semantic_clip_bytes,
        "semantic_clip_sha256": semantic_clip_sha256,
        "batch_trace": semantic_batch_trace,
        "compose_arm": semantic_compose_arm,
        "seed42": semantic_seed42,
        "validate_data_and_pairing": semantic_validate_data_and_pairing,
        "coupled_region_loss": v1_coupled_region_loss,
        "region_pair": region_pair,
        "CoupledPredictiveModel": CoupledPredictiveModel,
        "load_frozen_clip": load_frozen_clip,
        "source_snapshot": source_snapshot,
        "trainer": trainer,
    }


def _build_contract(args: Any, data: Any, split: Any, pairing: Mapping[str, Any]) -> dict[str, Any]:
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(args, resolve=True)
    return {
        "reuse_s0_data_and_loader_config_only": {
            "data_config": str(args.data_config),
            "positive_sampling": str(args.positive_sampling),
            "pairing_manifest_path": str(args.pairing_manifest_path),
            "pseudo_val_seed": int(args.pseudo_val_seed),
            "pseudo_val_num_classes": int(args.pseudo_val_num_classes),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "drop_last": bool(args.drop_last),
            "seed": 42,
            "resolved_source": resolved,
        },
        "v1_effective_preflight_contract": {
            "device": "cuda",
            "model_name": EXPECTED_MODEL,
            "batch_size": 32,
            "query_views": "clean+region_corrupted",
            "region_zero_based_update": 0,
            "unique_photo_bank": True,
            "positive_sampling": "same_class",
            "lambda_sig": 0.0,
            "no_optimizer": True,
            "no_sigreg": True,
            "no_training_steps": True,
            "no_calibration": True,
            "official_test_manifest_loaded": False,
            "wandb": "disabled",
            "train_class_count": len(split.train_class_ids),
            "canonical_positive_pool": int(pairing["unique_photo_pool"]),
        },
    }


def _make_region_views(batch: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], root: Path, region_pair: Callable[..., Any]) -> tuple[Any, Any, list[dict[str, Any]]]:
    import torch

    clean_rows: list[Any] = []
    corrupt_rows: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for index, row in enumerate(trace):
        relative = str(row["sketch_relative"])
        clean, corrupt, details = region_pair(batch["sketch"][index].cpu(), relative, zero_based_update=0)
        clean_rows.append(clean)
        corrupt_rows.append(corrupt)
        metadata.append({
            **details,
            "sketch_path": str(row["sketch_path"]),
            "sketch_relative": relative,
            "label": int(row["label"]),
            "clean_tensor_sha256": tensor_hash(clean),
            "corrupted_tensor_sha256": tensor_hash(corrupt),
        })
    return torch.stack(clean_rows), torch.stack(corrupt_rows), metadata


def _state_storage_separation(teacher: Any, student: Any) -> dict[str, Any]:
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if set(teacher_state) != set(student_state):
        raise AssertionError("student visual state keys differ from teacher visual state keys")
    shared: list[str] = []
    for name in teacher_state:
        if not torch_equal(teacher_state[name], student_state[name]):
            raise AssertionError(f"student visual initialization differs: {name}")
        if teacher_state[name].untyped_storage().data_ptr() == student_state[name].untyped_storage().data_ptr():
            shared.append(name)
    if shared:
        raise AssertionError(f"student and teacher share storage: {shared[:5]}")
    return {"state_keys": len(teacher_state), "byte_equal": True, "storage_shared": False}


def torch_equal(left: Any, right: Any) -> bool:
    import torch

    return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)


@contextmanager
def _method_wrappers(model: Any, capture: dict[str, Any]):
    original_photo = model.encode_photo
    original_reference = model.photo_reference

    @functools.wraps(original_photo)
    def encode_photo(photos: Any) -> Any:
        capture["photo_reference_calls"] = capture.get("photo_reference_calls", 0)
        capture["photo_live_calls"] = capture.get("photo_live_calls", 0) + 1
        value = original_photo(photos)
        capture["live_photo"] = value
        value.retain_grad()
        return value

    @functools.wraps(original_reference)
    def photo_reference(photos: Any) -> Any:
        capture["photo_reference_calls"] = capture.get("photo_reference_calls", 0) + 1
        return original_reference(photos)

    object.__setattr__(model, "encode_photo", encode_photo)
    object.__setattr__(model, "photo_reference", photo_reference)
    try:
        yield
    finally:
        object.__setattr__(model, "encode_photo", original_photo)
        object.__setattr__(model, "photo_reference", original_reference)


def _stats(value: Any) -> dict[str, Any]:
    import torch

    detached = value.detach().float()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(detached).all().item()),
        "element_std": float(detached.std(unbiased=False).item()),
        "norm_mean": float(detached.norm(dim=-1).mean().item()) if detached.ndim >= 2 else float(detached.norm().item()),
        "norm_std": float(detached.norm(dim=-1).std(unbiased=False).item()) if detached.ndim >= 2 else 0.0,
        "norm_min": float(detached.norm(dim=-1).min().item()) if detached.ndim >= 2 else float(detached.norm().item()),
        "norm_max": float(detached.norm(dim=-1).max().item()) if detached.ndim >= 2 else float(detached.norm().item()),
    }


def _run_graph(model: Any, clean: Any, corrupt: Any, bank: Mapping[str, Any], labels: Any, device: Any, loss_fn: Callable[..., Any]) -> dict[str, Any]:
    import torch

    hooks: list[Any] = []
    seen: dict[str, Any] = {"query_forward_calls": 0, "student_transformer_calls": 0, "predictor_calls": 0, "text_bank_calls": 0}

    def query_hook(_module: Any, _inputs: Any, output: Any) -> None:
        seen["query_forward_calls"] += 1
        for name in ("q", "mu_i", "mu_t"):
            value = getattr(output, name)
            value.retain_grad()
            seen[name] = value
        seen["g"] = output.g
        output.g.retain_grad()

    def predictor_pre(_module: Any, inputs: Any) -> None:
        seen["predictor_calls"] += 1
        seen["predictor_context"] = inputs[0].detach()
        if inputs[1].data_ptr() != model.photo_model.photo_prompt.data_ptr():
            raise AssertionError("predictor photo prompt is not the original owner")
        if inputs[2].data_ptr() != model.text_bank.context.data_ptr():
            raise AssertionError("predictor text context is not the original owner")

    def student_hook(_module: Any, _inputs: Any, output: Any) -> None:
        seen["student_transformer_calls"] += 1
        value = output[0] if isinstance(output, tuple) else output
        seen["student_transformer_output"] = value.detach()

    def text_hook(_module: Any, _inputs: Any, output: Any) -> None:
        seen["text_bank_calls"] += 1
        seen["text_bank"] = output
        output.retain_grad()

    hooks.extend([
        model.register_forward_hook(query_hook),
        model.student_visual.transformer.register_forward_hook(student_hook),
        model.predictor.register_forward_pre_hook(predictor_pre),
        model.text_bank.register_forward_hook(text_hook),
    ])
    capture: dict[str, Any] = {}
    torch.cuda.synchronize(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    graph_start = time.perf_counter()
    try:
        with _method_wrappers(model, capture):
            terms = loss_fn(
                model,
                clean,
                corrupt,
                bank["bank_tensors"],
                bank["positive_indices"],
                bank["negative_indices"],
                labels,
                bank["bank_labels"],
                bank["photo_ids"],
                lambda_sig=0.0,
            )
        if any(seen[name] != 1 for name in ("query_forward_calls", "student_transformer_calls", "predictor_calls")):
            raise AssertionError("query/predictor forward count or trace is wrong")
        if tuple(seen["student_transformer_output"].shape) != (64, 50, 768):
            raise AssertionError(f"unexpected student transformer output shape: {tuple(seen['student_transformer_output'].shape)}")
        if tuple(seen["predictor_context"].shape) != (64, 49, 768):
            raise AssertionError(f"unexpected predictor context shape: {tuple(seen['predictor_context'].shape)}")
        if not torch.equal(seen["predictor_context"], seen["student_transformer_output"][:, 1:]):
            raise AssertionError("predictor context is not exactly the student patch-token slice")
        if not torch.equal(seen["g"].detach(), seen["predictor_context"].mean(dim=1)):
            raise AssertionError("g is not the unnormalized patch-token mean")
        for name in ("q", "mu_i", "mu_t"):
            norms = seen[name].detach().float().norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=2e-5):
                raise AssertionError(f"{name} is not unit normalized")
        for name in ("total", "clean_rank_i", "masked_rank_i", "clean_ce_t", "masked_ce_t"):
            _finite(terms[name], name)
        if terms["sigreg"].detach().item() != 0.0:
            raise AssertionError("lambda_sig=0 did not produce an exact zero SIGReg term")
        if capture.get("photo_live_calls") != 1 or capture.get("photo_reference_calls") != 1:
            raise AssertionError("live photo/reference call count is not one")
        if seen.get("text_bank_calls") != 1:
            raise AssertionError("text bank was not encoded exactly once")
        if tuple(capture["live_photo"].shape) != (int(bank["bank_tensors"].shape[0]), 512):
            raise AssertionError(f"unexpected live photo shape: {tuple(capture['live_photo'].shape)}")
        if tuple(seen["text_bank"].shape) != (84, 512):
            raise AssertionError(f"unexpected text bank shape: {tuple(seen['text_bank'].shape)}")

        live_photo = capture["live_photo"]
        text_bank = seen["text_bank"]
        rank_photo_grad = torch.autograd.grad(
            terms["clean_rank_i"] + terms["masked_rank_i"], live_photo, retain_graph=True
        )[0]
        text_ce_grad = torch.autograd.grad(
            terms["clean_ce_t"] + terms["masked_ce_t"], text_bank, retain_graph=True
        )[0]
        _finite(rank_photo_grad, "rank live photo gradient")
        _finite(text_ce_grad, "CE live text-bank gradient")
        if rank_photo_grad.norm().item() <= 0 or text_ce_grad.norm().item() <= 0:
            raise AssertionError("live ranking or CE bank edge has zero gradient")

        # Edge probes can populate retained .grad fields: report only total-loss gradients.
        for value in (seen["g"], seen["q"], seen["mu_i"], seen["mu_t"], live_photo, text_bank):
            value.grad = None
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        terms["total"].backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        graph_elapsed = time.perf_counter() - graph_start
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

        for name in ("g", "q", "mu_i", "mu_t", "live_photo", "text_bank"):
            value = seen.get(name) if name in seen else capture.get(name)
            if value is None:
                value = capture.get(name)
            if value is None or value.grad is None:
                raise AssertionError(f"missing retained gradient for {name}")
            _finite(value.grad, f"{name} retained gradient")

        gradients: dict[str, dict[str, Any]] = {}
        trainable_count = 0
        trainable_elements = 0
        frozen_count = 0
        frozen_elements = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                trainable_count += 1
                trainable_elements += parameter.numel()
                if parameter.grad is None:
                    raise AssertionError(f"missing trainable gradient: {name}")
                _finite(parameter.grad, f"gradient {name}")
                norm = float(parameter.grad.detach().float().norm().item())
                if name in ("photo_model.photo_prompt", "text_bank.context") and norm <= 0:
                    raise AssertionError(f"shared prompt has zero total gradient: {name}")
                gradients[name] = {"norm": norm, "finite": True, "missing": False, "zero": norm == 0.0}
            else:
                frozen_count += 1
                frozen_elements += parameter.numel()
                if parameter.grad is not None:
                    raise AssertionError(f"frozen parameter received a gradient: {name}")
        groups: dict[str, dict[str, Any]] = {}
        for group in model.optimizer_parameter_groups():
            norms = [parameter.grad.detach().float().norm() for parameter in group["params"] if parameter.grad is not None]
            l2 = float(torch.stack(norms).pow(2).sum().sqrt().item()) if norms else 0.0
            if l2 <= 0.0 or not torch.isfinite(torch.tensor(l2)).item():
                raise AssertionError(f"optimizer group has no finite nonzero gradient: {group['name']}")
            groups[group["name"]] = {"parameter_count": len(group["params"]), "element_count": sum(parameter.numel() for parameter in group["params"]), "l2": l2}

        g_stats = _stats(seen["g"])
        if abs(g_stats["norm_mean"] - 1.0) < 1e-4 or g_stats["norm_std"] <= 0.0:
            raise AssertionError("g unexpectedly has unit-normalized or constant output")
        return {
            "loss": {name: float(value.detach().item()) for name, value in terms.items()},
            "forward_counts": {
                "query_forward_64": seen["query_forward_calls"],
                "student_transformer": seen["student_transformer_calls"],
                "predictor": seen["predictor_calls"],
                "text_bank": seen["text_bank_calls"],
                "photo_live": capture["photo_live_calls"],
                "teacher_reference": capture["photo_reference_calls"],
            },
            "hook_outputs": {
                "student_transformer": _stats(seen["student_transformer_output"]),
                "predictor_context": _stats(seen["predictor_context"]),
                "text_bank": _stats(seen["text_bank"]),
                "live_photo": _stats(capture["live_photo"]),
                "q": _stats(seen["q"]),
                "mu_i": _stats(seen["mu_i"]),
                "mu_t": _stats(seen["mu_t"]),
                "g": g_stats,
            },
            "retained_gradient_stats": {
                name: _stats((seen[name] if name in seen else capture[name]).grad)
                for name in ("g", "q", "mu_i", "mu_t", "live_photo", "text_bank")
            },
            "edge_gradients": {"rank_wrt_live_photo_norm": float(rank_photo_grad.detach().float().norm().item()), "ce_wrt_text_bank_norm": float(text_ce_grad.detach().float().norm().item())},
            "parameter_gradients": gradients,
            "gradient_group_l2": groups,
            "trainable_parameter_count": trainable_count,
            "trainable_parameter_elements": trainable_elements,
            "frozen_parameter_count": frozen_count,
            "frozen_parameter_elements": frozen_elements,
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "full_graph_delta_allocated_bytes": peak_allocated - baseline_allocated,
                "full_graph_delta_reserved_bytes": peak_reserved - baseline_reserved,
                "full_graph_and_edge_checks_seconds": graph_elapsed,
                "backward_seconds": elapsed,
                "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
                "scope": "full forward + intermediate edge checks + total backward; no optimizer states/step",
            },
        }
    finally:
        for hook in hooks:
            hook.remove()


def run_preflight(output_dir: Path) -> dict[str, Any]:
    """Run the authorized GPU-only preflight into an already-fresh directory."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("run_preflight requires a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    modules = _imports()
    import torch
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "preflight_kind": "REAL_CLIP_COUPLED_PREDICTIVE_V1_NO_UPDATE",
        "output_dir": str(output_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "WANDB_MODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "flags": {"device": "cuda", "batch_size": 32, "lambda_sig": 0.0, "no_optimizer": True, "no_sigreg": True, "no_wandb": True, "official_test_manifest_loaded": False, "official_unseen": False, "evaluation_gallery": False},
    }
    _json_write(output_dir / "preflight_receipt.json", receipt)

    def archive_phase() -> dict[str, Any]:
        snapshot = modules["source_snapshot"](ROOT)
        _archive_source_snapshot(output_dir, snapshot)
        return snapshot

    source_before = _phase(output_dir, "source_archive", archive_phase)
    receipt["source_before"] = {"sha256": source_before["sha256"], "file_count": source_before["file_count"], "git": _safe_git_state()}

    def cache_phase() -> dict[str, Any]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
        if str(modules["semantic_clip_path"]) != str(CLIP_PATH):
            raise AssertionError("CLIP helper path differs from the approved local cache path")
        if not CLIP_PATH.is_file():
            raise FileNotFoundError(f"verified local CLIP cache is missing: {CLIP_PATH}")
        actual_bytes, actual_sha = CLIP_PATH.stat().st_size, sha256_file(CLIP_PATH)
        if actual_bytes != EXPECTED_CLIP_BYTES or actual_sha != EXPECTED_CLIP_SHA256:
            raise AssertionError(f"CLIP cache identity mismatch: {actual_bytes}, {actual_sha}")
        if (actual_bytes, actual_sha) != (modules["semantic_clip_bytes"], modules["semantic_clip_sha256"]):
            raise AssertionError("CLIP identity differs from semantic preflight helper")
        device = torch.device("cuda")
        return {"path": str(CLIP_PATH), "bytes": actual_bytes, "sha256": actual_sha, "device": torch.cuda.get_device_name(device), "device_index": torch.cuda.current_device(), "torch": str(torch.__version__), "cuda": str(torch.version.cuda)}

    cache = _phase(output_dir, "environment_and_cache", cache_phase)
    receipt["clip"] = cache

    state: dict[str, Any] = {}

    # _loader requires the verified local CLIP eval transform, so data identity
    # is validated first and the actual loader is made after model construction.
    args = modules["compose_arm"]("semantic_text_S0")
    if int(args.batch_size) != 32 or str(args.positive_sampling) != "same_class" or not args.drop_last:
        raise AssertionError("preflight requires batch32/drop_last/same_class canonical pool")
    def data_phase() -> dict[str, Any]:
        state["data_bundle"] = modules["validate_data_and_pairing"](modules["trainer"], args)
        _, _, _, split_id, manifest_id, pair = state["data_bundle"]
        return {"split_identity": split_id, "manifest_identity": manifest_id,
                "pairing_sha256": pair["sha256"], "pairing_records": pair["records"],
                "canonical_positive_photo_pool": pair["unique_photo_pool"]}

    data_result = _phase(output_dir, "data_identity", data_phase)
    data, names, split, split_identity, manifest_identity, pairing = state["data_bundle"]
    _json_write(output_dir / "resolved_preflight_config.json", _build_contract(args, data, split, pairing))

    def model_phase() -> dict[str, Any]:
        modules["seed42"]()
        bundle = modules["load_frozen_clip"](model_name=EXPECTED_MODEL, pretrained=str(CLIP_PATH), device="cuda")
        teacher = bundle.encoder.model
        teacher_before = snapshot_state(teacher)
        teacher_parameter_before = snapshot_parameters(teacher)
        model = modules["CoupledPredictiveModel"](bundle.encoder, bundle.tokenizer, {class_id: names[class_id] for class_id in split.train_class_ids})
        model.train(True)
        teacher_after = snapshot_state(teacher)
        teacher_parameter_after = snapshot_parameters(teacher)
        if teacher_after != teacher_before or teacher_parameter_after != teacher_parameter_before:
            raise AssertionError("teacher state changed during model construction")
        if any(parameter.requires_grad for parameter in teacher.parameters()) or teacher.training:
            raise AssertionError("teacher is not frozen/eval")
        devices = {name: str(tensor.device) for name, tensor in
                   list(model.named_parameters()) + list(model.named_buffers())}
        expected_device = f"cuda:{torch.cuda.current_device()}"
        if any(device != expected_device for device in devices.values()):
            raise AssertionError("not all model parameters/buffers are on the selected CUDA device")
        separation = _state_storage_separation(teacher.visual, model.student_visual)
        if model.photo_model.visual is not teacher.visual or model.original_clip is not teacher:
            raise AssertionError("teacher ownership is not shared exactly once")
        if model.text_bank.context.shape != (4, 512) or model.photo_model.photo_prompt.shape != (3, 768) or model.T0.shape != (84, 512):
            raise AssertionError("V1 prompt/text shapes are wrong")
        if model.T0.requires_grad or model.T0.is_inference() or not torch.isfinite(model.T0).all().item():
            raise AssertionError("T0 is not a normal finite detached tensor")
        if tuple(int(x) for x in model.text_bank.prefix_token_ids) != (320, 1125, 539, 320):
            raise AssertionError("V1 prefix token IDs are wrong")
        if model.initial_text_parity_max_error > 1e-6:
            raise AssertionError(f"initial T0 parity error is {model.initial_text_parity_max_error}")
        if tuple(model.text_bank.token_ids.shape) != (84, 77) or tuple(model.classids.shape) != (84,):
            raise AssertionError("V1 text token/class-ID metadata shapes are wrong")
        if tuple(int(x) for x in model.classids.tolist()) != EXPECTED_TRAIN_CLASSES:
            raise AssertionError("V1 class IDs are not the canonical 84 train classes")
        state.update({"bundle": bundle, "model": model, "teacher_before": teacher_before, "teacher_after_constructor": teacher_after})
        return {"parameter_and_buffer_devices": devices,
                "teacher_parameters_before_constructor": teacher_parameter_before,
                "teacher_parameters_after_constructor": teacher_parameter_after,
                "teacher_state_before_constructor": teacher_before, "teacher_state_after_constructor": teacher_after, "teacher_state_unchanged": True, "student_visual": separation, "prompt_shapes": {"photo_prompt": list(model.photo_model.photo_prompt.shape), "text_context": list(model.text_bank.context.shape), "T0": list(model.T0.shape)}, "prefix_token_ids": list(model.text_bank.prefix_token_ids), "initial_text_parity_max_error": float(model.initial_text_parity_max_error), "token_ids_shape": list(model.text_bank.token_ids.shape), "class_ids": list(model.classids.tolist()), "class_ids_shape": list(model.classids.shape), "teacher_eval": not teacher.training}

    model_result = _phase(output_dir, "model_initialization", model_phase)
    bundle, model = state["bundle"], state["model"]

    def batch_phase() -> dict[str, Any]:
        loader = modules["trainer"]._loader((split.train_sketch_entries, split.train_photo_entries), bundle.transform, args, train=True, seed=42, positive_pairing=pairing["mapping"])
        if len(loader) != 1457:
            raise AssertionError(f"unexpected train loader length: {len(loader)}")
        batch = next(iter(loader))
        trace = modules["batch_trace"](batch, pairing["mapping"], Path(data.root))
        from PIL import Image
        from spica.data.transforms import build_clip_eval_transform
        expected_transform = build_clip_eval_transform()
        for index, row in enumerate(trace):
            row["sketch_sha256"] = sha256_file(Path(row["sketch_path"]))
            with Image.open(row["sketch_path"]) as image:
                expected = expected_transform(image.convert("RGB"))
            if not torch.equal(expected, batch["sketch"][index]):
                raise AssertionError("local-checkpoint preprocessing differs from approved CLIP transform")
        if len({row["sketch_path"] for row in trace}) != 32:
            raise AssertionError("batch does not contain 32 unique sketch IDs")
        train_photo_labels = {str(entry.path.resolve()): entry.label for entry in split.train_photo_entries}
        train_photo_paths = set(train_photo_labels)
        train_classes = set(split.train_class_ids)
        for row in trace:
            if int(row["label"]) not in train_classes or int(row["negative_label"]) not in train_classes:
                raise AssertionError("batch label escaped pseudo-train classes")
            paths = [*row["positive_photo_paths"], row["negative_photo_path"]]
            if any(str(Path(path).resolve()) not in train_photo_paths for path in paths):
                raise AssertionError("batch photo escaped the train photo manifest")
            if any(train_photo_labels[str(Path(path).resolve())] != row["label"] for path in row["positive_photo_paths"]):
                raise AssertionError("positive photo label differs from train manifest")
            if train_photo_labels[str(Path(row["negative_photo_path"]).resolve())] != row["negative_label"]:
                raise AssertionError("negative photo label differs from train manifest")
            if any(path not in pairing["pool_paths"] for path in row["positive_photo_paths"]):
                raise AssertionError("positive photo escaped the canonical 8400-photo pool")
        packed = pack_unique_photos(batch, trace, Path(data.root))
        clean, corrupt, metadata = _make_region_views(batch, trace, Path(data.root), modules["region_pair"])
        state.update({"batch": batch, "trace": trace, "packed": packed, "clean": clean, "corrupt": corrupt, "region_metadata": metadata})
        tensor_hashes = {
            "sketch": [tensor_hash(batch["sketch"][index]) for index in range(32)],
            "positive": [tensor_hash(batch["positive_photos"][index, 0]) for index in range(32)],
            "negative": [tensor_hash(batch["negative_photo"][index]) for index in range(32)],
            "unique_photo_bank": [tensor_hash(value) for value in packed["bank_tensors"]],
        }
        _json_write(output_dir / "batch_trace.json", {"rows": trace, "region_metadata": metadata, "photo_bank_ids": list(packed["photo_ids"]), "normalized_tensor_sha256": tensor_hashes})
        return {"loader_length": len(loader), "batch_size": len(trace), "unique_sketch_ids": len({row["sketch_path"] for row in trace}), "unique_photo_bank": int(packed["bank_tensors"].shape[0]), "all_train_classes": sorted(train_classes), "photo_bank_ids": list(packed["photo_ids"]), "positive_indices": packed["positive_indices"].tolist(), "negative_indices": packed["negative_indices"].tolist(), "normalized_tensor_sha256": tensor_hashes}

    batch_result = _phase(output_dir, "batch_and_region_views", batch_phase)

    def graph_phase() -> dict[str, Any]:
        device = torch.device("cuda")
        clean = state["clean"].to(device, non_blocking=True)
        corrupt = state["corrupt"].to(device, non_blocking=True)
        packed = {key: value for key, value in state["packed"].items()}
        packed["bank_tensors"] = packed["bank_tensors"].to(device, non_blocking=True)
        packed["bank_labels"] = packed["bank_labels"].to(device)
        packed["positive_indices"] = packed["positive_indices"].to(device)
        packed["negative_indices"] = packed["negative_indices"].to(device)
        labels = state["batch"]["label"].long().to(device)
        param_before = snapshot_parameters(model)
        state_before = snapshot_state(model)
        if model.original_clip.training or model.photo_model.visual.training or not model.student_visual.training:
            raise AssertionError("teacher/student train-eval modes violate V1 contract")
        result = _run_graph(model, clean, corrupt, packed, labels, device, modules["coupled_region_loss"])
        teacher_after = snapshot_state(model.original_clip)
        state_after = snapshot_state(model)
        _same_snapshot(state["teacher_before"], teacher_after, "teacher state")
        _same_snapshot(state_before, state_after, "all model state")
        param_after = snapshot_parameters(model)
        _same_snapshot(param_before, param_after, "all model parameters")
        result["teacher_parameters"] = {
            "before": {name: value for name, value in param_before.items() if name.startswith("original_clip.")},
            "after": {name: value for name, value in param_after.items() if name.startswith("original_clip.")},
        }
        if any(parameter.grad is not None for parameter in model.original_clip.parameters()):
            raise AssertionError("teacher received a gradient")
        for name in ("ln_post", "proj"):
            component = getattr(model.student_visual, name, None)
            if isinstance(component, torch.Tensor):
                if component.grad is not None:
                    raise AssertionError(f"unused student {name} received a gradient")
            elif component is not None and any(parameter.grad is not None for parameter in component.parameters()):
                raise AssertionError(f"unused student {name} received a gradient")
        result["state_identity"] = {"before": state_before, "after": state_after, "byte_identical": True, "parameter_flags_before": {name: value["requires_grad"] for name, value in param_before.items()}}
        result["device"] = {"name": torch.cuda.get_device_name(device), "index": torch.cuda.current_device()}
        return result

    graph_result = _phase(output_dir, "full_graph_backward", graph_phase)
    source_after = _phase(output_dir, "source_unchanged", lambda: modules["source_snapshot"](ROOT))
    if source_after["sha256"] != source_before["sha256"]:
        raise AssertionError("source snapshot changed during preflight")
    receipt.update({"data": data_result, "model": model_result, "batch": batch_result, "graph": graph_result, "source_after": {"sha256": source_after["sha256"], "file_count": source_after["file_count"]}, "status": "PASS", "finished_utc": datetime.now(timezone.utc).isoformat()})
    _json_write(output_dir / "resolved_preflight_config.json", _build_contract(args, data, split, pairing))
    _json_write(output_dir / "preflight_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="REAL-CLIP V1 coupled predictive no-update GPU preflight")
    parser.add_argument("--output-dir", required=True, help="fresh, nonexistent output directory")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parsed = parser.parse_args()
    output = _output_dir(parsed.output_dir)
    print(f"preflight output: {output}", flush=True)
    try:
        run_preflight(output)
    except Exception as error:
        receipt_path = output / "preflight_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        receipt.update({"status": "FAIL", "output_dir": str(output), "finished_utc": datetime.now(timezone.utc).isoformat(), "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}})
        _json_write(output / "preflight_receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 1
    print(json.dumps({"status": "PASS", "output_dir": str(output)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
