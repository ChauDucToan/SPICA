"""Measure the SIGReg/task gradient scale on four real train batches.

This is deliberately a diagnostic, not a trainer: it creates no optimizer,
does no update, and does not create a W&B run.  The resulting lambda is a
measurement pending parent review; this script does not mark it verified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import traceback
from typing import Any, Iterator, Mapping

# Set these before importing torch/open_clip or project modules.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["WANDB_MODE"] = "disabled"

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = "ViT-B-32-quickgelu"
BATCHES = 4
BATCH_SIZE = 32
RHO = 0.1
SEED = 42


def _seed42() -> None:
    """Seed every process RNG before loading the local CLIP/model."""
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _fresh_output(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)
    return path


def _state_digest(state: Mapping[str, Any]) -> str:
    """Digest a serializable tensor-state mapping without retaining tensors."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(str(name).encode())
        digest.update(b"\0")
        if hasattr(value, "detach"):
            digest.update(_tensor_sha256(value).encode())
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    from spica.semantic_text import tensor_sha256

    return tensor_sha256(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(output: Path, name: str) -> dict[str, Any]:
    path = output / name
    return {"path": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _tensor_record(value: Any, *, requires_grad: bool | None = None) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _tensor_sha256(value),
        **({"requires_grad": bool(requires_grad)} if requires_grad is not None else {}),
    }


def _named_parameter_records(module: Any) -> dict[str, dict[str, Any]]:
    return {
        name: _tensor_record(parameter, requires_grad=parameter.requires_grad)
        for name, parameter in module.named_parameters()
    }


def _state_records(module: Any) -> dict[str, dict[str, Any]]:
    return {
        name: _tensor_record(value)
        for name, value in module.state_dict().items()
        if hasattr(value, "detach")
    }


def _module_hash(module: Any) -> str:
    return _state_digest(
        {name: value for name, value in module.state_dict().items() if hasattr(value, "detach")}
    )


def _assert_same(before: Any, after: Any, label: str) -> None:
    if before != after:
        changed = [
            name
            for name in sorted(set(before) | set(after))
            if before.get(name) != after.get(name)
        ]
        raise AssertionError(f"{label} changed without an optimizer step: {changed[:8]}")


def _all_float32(module: Any, label: str) -> None:
    import torch

    for name, value in module.named_parameters():
        if value.dtype != torch.float32:
            raise TypeError(f"{label} parameter {name} is {value.dtype}, not float32")
    # Metadata buffers (class/token IDs) are intentionally integer tensors;
    # every floating-point buffer must still remain FP32.
    for name, value in module.named_buffers():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise TypeError(f"{label} buffer {name} is {value.dtype}, not float32")


def _global_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    numpy_state = repr(np.random.get_state()).encode()
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(numpy_state).hexdigest(),
        "torch_cpu": _tensor_sha256(torch.get_rng_state()),
        "torch_cuda": [
            _tensor_sha256(value) for value in torch.cuda.get_rng_state_all()
        ] if torch.cuda.is_available() else [],
    }


def _sigreg_state(sigreg: Any) -> Any:
    return sigreg.get_extra_state()["generator_state"].detach().cpu().clone()


def _sigreg_state_record(sigreg: Any) -> dict[str, Any]:
    value = _sigreg_state(sigreg)
    return {"sha256": _tensor_sha256(value), "numel": int(value.numel()), "dtype": str(value.dtype)}


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _last_block_layout(model: Any) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    block = model.student_visual.transformer.resblocks[-1]
    parameters = tuple(block.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise AssertionError("last student transformer block is not entirely trainable")
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names = [names_by_id.get(id(parameter)) for parameter in parameters]
    if any(name is None for name in names):
        raise AssertionError("last-block parameter is not registered in the model")
    offset = 0
    layout: list[dict[str, Any]] = []
    for order, (name, parameter) in enumerate(zip(names, parameters, strict=True)):
        if parameter.dtype != __import__("torch").float32:
            raise TypeError(f"last-block parameter {name} is not float32")
        end = offset + parameter.numel()
        layout.append({
            "order": order,
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
            "offset_start": offset,
            "offset_end": end,
        })
        offset = end
    return parameters, layout


def _flat_gradient(gradients: tuple[Any, ...], parameters: tuple[Any, ...], label: str) -> Any:
    import torch

    if len(gradients) != len(parameters):
        raise AssertionError(f"{label} gradient count does not match parameter count")
    pieces = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            raise AssertionError(f"{label} gradient is None for a last-block parameter")
        if gradient.dtype != torch.float32 or not torch.isfinite(gradient).all().item():
            raise FloatingPointError(f"{label} gradient is non-finite or not float32")
        if gradient.shape != parameter.shape:
            raise AssertionError(f"{label} gradient shape mismatch for {tuple(parameter.shape)}")
        pieces.append(gradient.detach().reshape(-1))
    flat = torch.cat(pieces).to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(flat).all().item() or flat.numel() == 0:
        raise FloatingPointError(f"{label} flattened gradient is non-finite or empty")
    return flat


def _double_stats(task_gradient: Any, sig_gradient: Any) -> dict[str, float]:
    import torch

    task = task_gradient.to(dtype=torch.float64)
    sig = sig_gradient.to(dtype=torch.float64)
    task_norm = torch.linalg.vector_norm(task)
    sig_norm = torch.linalg.vector_norm(sig)
    if not torch.isfinite(task_norm).item() or not torch.isfinite(sig_norm).item():
        raise FloatingPointError("gradient norm is non-finite")
    if task_norm.item() == 0.0 or sig_norm.item() == 0.0:
        raise FloatingPointError("task or SIGReg gradient norm is zero")
    dot = torch.dot(task, sig)
    cosine = dot / (task_norm * sig_norm)
    ratio = task_norm / sig_norm
    values = {
        "task_norm_l2_float64": task_norm.item(),
        "sigreg_norm_l2_float64": sig_norm.item(),
        "dot_float64": dot.item(),
        "cosine_float64": cosine.item(),
        "ratio_task_over_sigreg_float64": ratio.item(),
    }
    if not all(torch.isfinite(torch.tensor(value, dtype=torch.float64)).item() for value in values.values()):
        raise FloatingPointError("gradient diagnostic statistic is non-finite")
    return {name: float(value) for name, value in values.items()}


@contextmanager
def _capture_model_output(model: Any) -> Iterator[dict[str, Any]]:
    captured: dict[str, Any] = {"calls": 0}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["calls"] += 1
        captured["g"] = output.g

    handle = model.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _build_initialization_receipt(model: Any, names: Mapping[int, str], split: Any) -> dict[str, Any]:
    import torch

    teacher = model.original_clip
    teacher_visual = teacher.visual
    student = model.student_visual
    teacher_state = teacher_visual.state_dict()
    student_state = student.state_dict()
    if set(teacher_state) != set(student_state):
        raise AssertionError("teacher visual and student visual state keys differ")
    for name in teacher_state:
        if not torch.equal(teacher_state[name], student_state[name]):
            raise AssertionError(f"student initialization differs from teacher visual: {name}")
        if teacher_state[name].untyped_storage().data_ptr() == student_state[name].untyped_storage().data_ptr():
            raise AssertionError(f"student shares storage with teacher visual: {name}")
    if any(parameter.requires_grad for parameter in teacher.parameters()) or teacher.training:
        raise AssertionError("original CLIP teacher is not frozen/eval")
    _all_float32(model, "model")
    return {
        "model_architecture": model.architecture,
        "predictor_width": 256,
        "predictor_heads": 4,
        "classmap_count": len(split.train_class_ids),
        "classmap_ids": [int(value) for value in split.train_class_ids],
        "context_names_count": len(names),
        "context_names_ids": sorted(int(value) for value in names),
        "classmap_is_subset_of_context_names": set(model.classids.detach().cpu().tolist()).issubset(set(names)),
        "original_teacher_frozen": all(not parameter.requires_grad for parameter in teacher.parameters()),
        "original_teacher_eval": not teacher.training,
        "original_teacher_parameters": _named_parameter_records(teacher),
        "original_teacher_state": _state_records(teacher),
        "student_initial_parameters": _named_parameter_records(student),
        "student_initial_state": _state_records(student),
        "teacher_visual_student_initial_state_equal": True,
        "teacher_student_storage_separate": True,
        "model_initial_state_hash": _module_hash(model),
    }


def run(output: Path, device_name: str = "cuda", *, architecture: str = "predictive") -> dict[str, Any]:
    import torch
    from spica.coupled_predictive_losses import coupled_region_loss, task_loss
    from spica.data.coupled_training import (
        load_protocol_data,
        make_train_loader,
        prepare_batch,
        verify_clip_cache,
    )
    from spica.models.clip import load_frozen_clip
    from spica.models.coupled_predictive import CoupledPredictiveModel
    from spica.models.sigreg import SIGReg

    if architecture not in {"predictive", "predictive_fusion_v2"}:
        raise ValueError("unsupported diagnostic architecture")
    positive_pool = "full" if architecture == "predictive_fusion_v2" else "canonical"
    diagnostic_config = {"architecture": architecture, "positive_pool": positive_pool,
                         "task_identity": "mean_views(rank_i+ce_i)" if positive_pool == "full" else "mean_views(rank_i+ce_t)",
                         "batch_size": BATCH_SIZE}
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this authorized diagnostic requires CUDA; refusing CPU fallback")
    device = torch.device(device_name)

    from spica.provenance import source_snapshot
    from spica.train_coupled_predictive import _initialization_hashes
    import shutil
    source_before = source_snapshot(ROOT)
    for item in source_before["manifest"]:
        target = output / "source_snapshot" / "files" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["path"], target)
        assert _file_sha256(target) == item["sha256"]
    _json(output / "source_snapshot" / "index.json", source_before)
    clip_identity = verify_clip_cache()
    protocol = load_protocol_data()
    _seed42()
    split = protocol["split"]
    names = protocol["names"]
    data_root = Path(str(protocol["data"].root))

    # This is the same local-cache load and default predictive model init as the
    # trainer, with the approved 84-class train map rather than all 104 names.
    bundle = load_frozen_clip(
        model_name=EXPECTED_MODEL,
        pretrained=str(clip_identity["path"]),
        device=device,
    )
    classmap = {
        int(class_id): str(names[int(class_id)])
        for class_id in split.train_class_ids
    }
    if len(classmap) != 84 or not set(classmap).issubset(set(int(key) for key in names)):
        raise AssertionError("classmap must be the 84-class subset of the 104-name context map")
    model = CoupledPredictiveModel(
        bundle.encoder,
        bundle.tokenizer,
        classmap,
        architecture=architecture,
    ).to(device)
    model.train(True)
    if model.predictor is None or model.predictor.context_in.out_features != 256:
        raise AssertionError("diagnostic requires the default width-256 predictive model")
    _all_float32(model, "model")
    init_receipt = _build_initialization_receipt(model, names, split)
    initialization_hashes = _initialization_hashes(model)
    model_before = _state_records(model)
    teacher_before = _state_records(model.original_clip)
    model_hash_before = _module_hash(model)
    teacher_hash_before = _state_digest(
        {name: value for name, value in model.original_clip.state_dict().items() if hasattr(value, "detach")}
    )

    loader = make_train_loader(protocol, bundle.transform, positive_pool=positive_pool)
    if loader.generator is None:
        raise AssertionError("train loader has no generator")
    expected_generator = torch.Generator().manual_seed(SEED).get_state()
    if not torch.equal(loader.generator.get_state(), expected_generator):
        raise AssertionError("train loader generator was not initialized with seed 42")
    iterator = iter(loader)
    parameters, layout = _last_block_layout(model)
    parameter_names = [row["name"] for row in layout]
    sigreg = SIGReg(knots=17, num_projections=256, t_max=3.0, seed=SEED).to(device)
    _all_float32(sigreg, "SIGReg")
    sigreg_states: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []

    _json(output / "gradient_layout.json", {
        "parameter_scope": "model.student_visual.transformer.resblocks[-1]",
        "parameters": layout,
        "flat_numel": layout[-1]["offset_end"],
        "raw_gradient_dtype": "torch.float32",
        "parameter_names_ordered": parameter_names,
    })

    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(BATCHES):
        raw_batch = next(iterator)
        if raw_batch["sketch"].shape[0] != BATCH_SIZE:
            raise AssertionError(f"batch {batch_index} is not B32")
        loader_generator_before = _tensor_sha256(loader.generator.get_state())
        batch = prepare_batch(
            raw_batch,
            data_root=data_root,
            step=batch_index,
            classids=tuple(int(value) for value in model.classids.detach().cpu().tolist()),
        )
        batch = _move_batch(batch, device)
        if batch["clean"].shape[0] != BATCH_SIZE or batch["corrupted"].shape[0] != BATCH_SIZE:
            raise AssertionError(f"prepared batch {batch_index} is not B32")
        batch_records.append({
            "batch_index": batch_index,
            "trace": batch["trace"],
            "mask_metadata": batch["mask_metadata"],
            "clean_sha256": _tensor_sha256(batch["clean"]),
            "corrupted_sha256": _tensor_sha256(batch["corrupted"]),
            "photo_bank_sha256": _tensor_sha256(batch["photos"]),
            "photo_ids": list(batch["photo_ids"]),
            "loader_generator_before_sha256": loader_generator_before,
            "loader_generator_after_fetch_sha256": _tensor_sha256(loader.generator.get_state()),
        })

        model_before_batch = _state_records(model)
        teacher_before_batch = _state_records(model.original_clip)
        global_rng_before = _global_rng_state()
        sigreg_states[f"batch_{batch_index:02d}_before"] = _sigreg_state(sigreg)
        capture: dict[str, Any]
        with _capture_model_output(model) as capture:
            terms = coupled_region_loss(
                model,
                batch["clean"],
                batch["corrupted"],
                batch["photos"],
                batch["positive_indices"],
                batch["negative_indices"],
                batch["labels"],
                batch["photo_labels"],
                batch["photo_ids"],
                lambda_sig=0.0,
                sigreg=None,
            )
        if capture.get("calls") != 1 or "g" not in capture:
            raise AssertionError("coupled_region_loss did not produce exactly one captured model output")
        if terms["sigreg"].detach().item() != 0.0:
            raise AssertionError("lambda_sig=0 control computed a nonzero SIGReg term")
        task = task_loss(terms, architecture)
        if not torch.isfinite(task).item():
            raise FloatingPointError(f"task loss is non-finite at batch {batch_index}")
        g = capture["g"]
        if g.shape[0] != 2 * BATCH_SIZE or g.dtype != torch.float32:
            raise AssertionError(f"captured g has wrong shape/dtype: {tuple(g.shape)}, {g.dtype}")

        task_gradients = torch.autograd.grad(
            task, parameters, retain_graph=True, create_graph=False, allow_unused=True
        )
        task_gradient = _flat_gradient(task_gradients, parameters, "task")

        sigreg_states[f"batch_{batch_index:02d}_before_clean"] = _sigreg_state(sigreg)
        sig_clean = sigreg(g[:BATCH_SIZE])
        sigreg_states[f"batch_{batch_index:02d}_after_clean"] = _sigreg_state(sigreg)
        sig_corrupted = sigreg(g[BATCH_SIZE:])
        sigreg_states[f"batch_{batch_index:02d}_after_corrupted"] = _sigreg_state(sigreg)
        sig = 0.5 * (sig_clean + sig_corrupted)
        if not torch.isfinite(sig).item():
            raise FloatingPointError(f"SIGReg loss is non-finite at batch {batch_index}")
        sig_gradients = torch.autograd.grad(
            sig, parameters, retain_graph=False, create_graph=False, allow_unused=True
        )
        sig_gradient = _flat_gradient(sig_gradients, parameters, "SIGReg")
        sigreg_states[f"batch_{batch_index:02d}_after"] = _sigreg_state(sigreg)

        stats = _double_stats(task_gradient, sig_gradient)
        raw_path = output / f"raw_gradients_batch{batch_index:02d}.pt"
        torch.save({
            "batch_index": batch_index,
            "parameter_scope": "model.student_visual.transformer.resblocks[-1]",
            "parameter_names": parameter_names,
            "layout": layout,
            "task_gradient_flat_cpu_fp32": task_gradient,
            "sigreg_gradient_flat_cpu_fp32": sig_gradient,
            "task_loss": float(task.detach().cpu()),
            "sigreg_loss": float(sig.detach().cpu()),
            "stats_float64": stats,
        }, raw_path)

        model_after_batch = _state_records(model)
        teacher_after_batch = _state_records(model.original_clip)
        _assert_same(model_before_batch, model_after_batch, f"model at batch {batch_index}")
        _assert_same(teacher_before_batch, teacher_after_batch, f"teacher at batch {batch_index}")
        global_rng_after = _global_rng_state()
        if global_rng_before != global_rng_after:
            raise AssertionError(f"global RNG changed during batch {batch_index}; only SIGReg private RNG may advance")
        row = {
            "batch_index": batch_index,
            "batch_size": BATCH_SIZE,
            "task_loss": float(task.detach().cpu()),
            "sigreg_loss": float(sig.detach().cpu()),
            "raw_gradients": str(raw_path.name),
            "sigreg_state_before_sha256": _tensor_sha256(sigreg_states[f"batch_{batch_index:02d}_before"]),
            "sigreg_state_before_clean_sha256": _tensor_sha256(sigreg_states[f"batch_{batch_index:02d}_before_clean"]),
            "sigreg_state_after_clean_sha256": _tensor_sha256(sigreg_states[f"batch_{batch_index:02d}_after_clean"]),
            "sigreg_state_after_corrupted_sha256": _tensor_sha256(sigreg_states[f"batch_{batch_index:02d}_after_corrupted"]),
            "sigreg_state_after_sha256": _tensor_sha256(sigreg_states[f"batch_{batch_index:02d}_after"]),
            **stats,
        }
        rows.append(row)
        del task_gradients, sig_gradients, task_gradient, sig_gradient, terms, task, sig, sig_clean, sig_corrupted, g, capture, batch

    torch.save(sigreg_states, output / "sigreg_rng_states.pt")
    _json(output / "batch_records.json", batch_records)
    model_after = _state_records(model)
    teacher_after = _state_records(model.original_clip)
    _assert_same(model_before, model_after, "entire model")
    _assert_same(teacher_before, teacher_after, "frozen original CLIP teacher")
    if _module_hash(model) != model_hash_before:
        raise AssertionError("entire model hash changed despite no optimizer/model update")
    if not all(not parameter.requires_grad for parameter in model.original_clip.parameters()):
        raise AssertionError("original CLIP teacher is no longer fully frozen")
    if model.original_clip.training:
        raise AssertionError("original CLIP teacher is no longer eval")

    ratios = [row["ratio_task_over_sigreg_float64"] for row in rows]
    import numpy as np

    ratio_array = np.asarray(ratios, dtype=np.float64)
    if not np.isfinite(ratio_array).all() or (ratio_array <= 0).any():
        raise FloatingPointError("ratio statistics are non-finite or non-positive")
    median_ratio = float(np.median(ratio_array))
    lambda_sig = float(RHO * median_ratio)
    ratio_mean = float(np.mean(ratio_array))
    ratio_std = float(np.std(ratio_array, ddof=0))
    stability = {
        "min_ratio": float(np.min(ratio_array)),
        "max_ratio": float(np.max(ratio_array)),
        "max_over_min_ratio": float(np.max(ratio_array) / np.min(ratio_array)),
        "mean_ratio": ratio_mean,
        "std_ratio": ratio_std,
        "cv_ratio": float(ratio_std / ratio_mean),
        "threshold_gate_applied": False,
        "interpretation": "reported for parent review; no numerical stability threshold is locked here",
    }
    from spica.provenance import capture_provenance

    provenance = capture_provenance(
        ROOT,
        resolved_config={
            "diagnostic": "coupled_sigreg_gradient_scale",
            **diagnostic_config,
            "device": str(device),
            "seed": SEED,
            "batches": BATCHES,
            "batch_size": BATCH_SIZE,
            "rho": RHO,
            "model_name": EXPECTED_MODEL,
            "classmap_ids": [int(value) for value in split.train_class_ids],
            "context_names_count": len(names),
            "parameter_scope": "model.student_visual.transformer.resblocks[-1]",
            "sigreg": {"knots": 17, "num_projections": 256, "t_max": 3.0, "seed": SEED},
            "no_optimizer": True,
            "no_model_update": True,
            "wandb": "disabled",
        },
    )
    _json(output / "provenance.json", provenance)
    assert provenance["source_snapshot"]["sha256"] == source_before["sha256"]
    component_paths = (
        "src/spica/models/coupled_predictive.py", "src/spica/models/sigreg.py",
        "src/spica/coupled_predictive_losses.py", "src/spica/data/coupled_training.py",
        "src/spica/data/coupled_views.py", "src/spica/data/masking.py",
        "src/spica/models/clip.py", "src/spica/models/frozen_prompt.py",
        "src/spica/evaluation/text_bank.py",
    )
    result = {
        "status": "MEASURED_PENDING_REVIEW",
        "verified": False,
        "diagnostic": "coupled_sigreg_gradient_scale",
        "formula_identity": "SIGReg pinned MINIMAL: 17 knots [0,3], 256 unit Gaussian projections, FP32 Epps-Pulley statistic; separate clean/corrupted calls",
        "architecture_identity": f"CoupledPredictiveModel {architecture} default width=256, heads=4, float32",
        "resolved_config": diagnostic_config,
        "gradient_batch_source": f"load_protocol_data -> make_train_loader(seed=42,positive_pool={positive_pool}) -> prepare_batch(step=0..3)",
        "started_seed": SEED,
        "batches": BATCHES,
        "batch_size": BATCH_SIZE,
        "rho": RHO,
        "lambda_sig": lambda_sig,
        "selection": {"lambda_sig": lambda_sig, "formula": "rho * median(task_norm / sigreg_norm)"},
        "parameter_scope": "model.student_visual.transformer.resblocks[-1]",
        "parameter_names_ordered": parameter_names,
        "sigreg": {"knots": 17, "num_projections": 256, "t_max": 3.0, "seed": SEED, "private_cpu_rng": True},
        "rows": rows,
        "stability": stability,
        "clip_identity": clip_identity,
        "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"),
        "provenance_file": "provenance.json",
        "initialization_hashes": initialization_hashes,
        "component_sha256": {path: _file_sha256(ROOT / path) for path in component_paths},
        "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
        "train_class_ids": [int(value) for value in split.train_class_ids],
        "context_name_count": len(names),
        "data_identity": {
            "split": protocol["split_identity"],
            "manifest": protocol["manifest_identity"],
            "pairing": {
                "sha256": protocol["pairing"]["sha256"],
                "records": protocol["pairing"]["records"],
                "unique_photo_pool": protocol["pairing"]["unique_photo_pool"],
            },
            "active_positive_pool": positive_pool,
            "positive_pool_count": 58950 if positive_pool == "full" else 8400,
            "pairing_role": "audit_only" if positive_pool == "full" else "positive_sampling_source",
            "loader_length": len(loader),
            "batch_records_file": "batch_records.json",
        },
        "model_initialization": init_receipt,
        "model_hash_before": model_hash_before,
        "model_hash_after": _module_hash(model),
        "teacher_state_hash_before": teacher_hash_before,
        "teacher_state_hash_after": _state_digest({name: value for name, value in model.original_clip.state_dict().items() if hasattr(value, "detach")}),
        "entire_model_unchanged": True,
        "teacher_frozen_and_unchanged": True,
        "loader_generator_seed": SEED,
        "raw_gradient_files": [row["raw_gradients"] for row in rows],
        "sigreg_rng_state_file": "sigreg_rng_states.pt",
        "batch_records_file": "batch_records.json",
        "artifacts": [
            _artifact_record(output, "provenance.json"),
            _artifact_record(output, "gradient_layout.json"),
            _artifact_record(output, "batch_records.json"),
            _artifact_record(output, "sigreg_rng_states.pt"),
            *[_artifact_record(output, row["raw_gradients"]) for row in rows],
        ],
        "gradient_layout_file": "gradient_layout.json",
        "wandb": "disabled; no run created",
        "note": "Parent must review the measured spread and create a separate diagnostic_verified.json before any fixed-lambda arm.",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json(output / "diagnostic_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--architecture", choices=("predictive", "predictive_fusion_v2"), default="predictive")
    args = parser.parse_args()
    output: Path | None = None
    try:
        output = _fresh_output(args.output_dir)
        result = run(output, args.device, architecture=args.architecture)
        print(json.dumps({"status": result["status"], "output_dir": str(output), "lambda_sig": result["lambda_sig"]}, sort_keys=True))
        return 0
    except Exception as error:
        if output is not None:
            _json(output / "diagnostic_result.json", {
                "status": "FAIL",
                "verified": False,
                "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            })
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
