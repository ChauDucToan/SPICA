#!/usr/bin/env python3
"""Fixed, no-update sketch-reference calibration for the official TU train split.

The production loss owns only the scalar ``sketch_ref`` term.  This diagnostic
captures the teacher and clean-q actually used by that loss, then reconstructs
the detached 4xB32 logits locally for independent algebra checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import traceback
import types
from statistics import median
from typing import Any, Iterator, Mapping

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from diagnose_coupled_sigreg import (  # noqa: E402
    _all_float32, _file_sha256, _fresh_output, _global_rng_state,
    _module_hash, _move_batch, _seed42, _tensor_sha256,
)
from diagnose_fusion_photo_ce import _cosine, _norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RUN = ROOT / "outputs/coupled_benchmark_execution_20260910T164500Z/runs/tuberlin_220_30"
BATCHES, BATCH_SIZE, SEED, TAU, RHO = 4, 32, 42, 0.07, 0.1
METHOD = "coupled_predictive_mp_sketch_ref_official_v1"
ARM = "F2_MP_Q_SREF_OFFICIAL"
LAST_SCOPE = "model.student_visual.transformer.resblocks[-1]"
COMPONENTS = (
    "scripts/diagnose_coupled_sketch_ref.py",
    "scripts/diagnose_coupled_sigreg.py",
    "scripts/diagnose_fusion_photo_ce.py",
    "scripts/run_coupled_campaign.py",
    "src/spica/coupled_predictive_losses.py",
    "src/spica/models/coupled_predictive.py",
    "src/spica/models/clip.py",
    "src/spica/models/checkpoint.py",
    "src/spica/train_coupled_predictive.py",
    "src/spica/train_coupled_benchmark.py",
    "src/spica/data/coupled_benchmark.py",
    "src/spica/data/coupled_training.py",
    "src/spica/data/coupled_views.py",
    "src/spica/data/datasets.py",
    "src/spica/data/masking.py",
    "src/spica/evaluation/coupled_predictive.py",
    "src/spica/provenance.py",
)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _artifact(output: Path, name: str) -> dict[str, Any]:
    path = output / name
    return {"path": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _first_jsonl(path: Path, count: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) == count:
                break
            rows.append(json.loads(line))
    if len(rows) != count:
        raise AssertionError(f"{path} has {len(rows)} rows, expected {count}")
    return rows


def _canonical(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): _plain(v) for k, v in row.items() if k not in {"step", "actual_lr"}}


def _historical_bindings(run_dir: Path) -> dict[str, str]:
    names = ("initialization.json", "resolved_config.json", "observation_trace.jsonl", "mask_metadata.jsonl", "run_result.json", "checkpoint_step0.pt")
    result = {}
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = _file_sha256(path)
    return result


def _last_block_parameters(model: Any, *, expected: tuple[int, int] | None = None) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Enumerate the module itself; never sort or infer its parameters."""
    import torch

    block = model.student_visual.transformer.resblocks[-1]
    parameters = tuple(block.parameters())
    if not parameters or any(not p.requires_grad for p in parameters):
        raise AssertionError("last student transformer block is not entirely trainable")
    names = {id(p): name for name, p in model.named_parameters()}
    layout, offset = [], 0
    for order, parameter in enumerate(parameters):
        name = names.get(id(parameter))
        if name is None or parameter.dtype != torch.float32:
            raise AssertionError("last-block parameter is unregistered or not float32")
        end = offset + parameter.numel()
        layout.append({"order": order, "name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype), "numel": int(parameter.numel()), "offset_start": offset, "offset_end": end})
        offset = end
    if expected is not None and (len(parameters), offset) != expected:
        raise AssertionError(f"last-block layout mismatch: {len(parameters)} tensors/{offset} coordinates, expected {expected}")
    return parameters, layout


def _scope_parameters(model: Any) -> dict[str, list[tuple[str, Any]]]:
    last, _ = _last_block_parameters(model)
    names = {id(p): name for name, p in model.named_parameters()}
    scopes = {
        "student_last_block": [(names[id(p)], p) for p in last],
        "pooled_head": [(n, p) for n, p in model.named_parameters() if n.startswith("pooled_head.") and p.requires_grad],
    }
    if any(not rows for rows in scopes.values()):
        raise AssertionError("calibration scopes must not be empty")
    ids = [id(p) for rows in scopes.values() for _, p in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("calibration scopes overlap")
    return scopes


def _layout(scopes: Mapping[str, list[tuple[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": 1, "scopes": {}}
    for scope, rows in scopes.items():
        offset, parameters = 0, []
        for order, (name, parameter) in enumerate(rows):
            end = offset + parameter.numel()
            parameters.append({"order": order, "name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype), "numel": int(parameter.numel()), "offset_start": offset, "offset_end": end})
            offset = end
        result["scopes"][scope] = {
            "parameter_scope": LAST_SCOPE if scope == "student_last_block" else "model.pooled_head",
            "parameters": parameters,
            "parameter_names_ordered": [row["name"] for row in parameters],
            "flat_numel": offset,
            "raw_gradient_dtype": "torch.float32",
        }
    return result


def _gradients(loss: Any, scopes: Mapping[str, list[tuple[str, Any]]], *, retain_graph: bool) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, list[str]]]:
    import torch

    rows = [(scope, name, parameter) for scope, values in scopes.items() for name, parameter in values]
    values = iter(torch.autograd.grad(loss, [p for _, _, p in rows], retain_graph=retain_graph, allow_unused=True))
    per_parameter: dict[str, dict[str, Any]] = {scope: {} for scope in scopes}
    flat: dict[str, Any] = {}
    unused: dict[str, list[str]] = {scope: [] for scope in scopes}
    for scope, name, parameter in rows:
        gradient = next(values)
        if gradient is None:
            unused[scope].append(name)
            raise AssertionError(f"unused gradient in calibration scope: {scope}/{name}")
        if gradient.dtype != torch.float32 or gradient.shape != parameter.shape or not torch.isfinite(gradient).all().item():
            raise FloatingPointError(f"invalid gradient in {scope}/{name}")
        cpu = gradient.detach().to(device="cpu", dtype=torch.float32).contiguous()
        per_parameter[scope][name] = cpu
    for scope, values in per_parameter.items():
        flat[scope] = torch.cat([value.reshape(-1) for value in values.values()])
    return per_parameter, flat, unused


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != BATCHES:
        raise ValueError(f"selection requires exactly {BATCHES} batches")
    ratios = {}
    candidates = {}
    for scope in ("student_last_block", "pooled_head"):
        values = [_norm(row["old_total"][scope], positive=True) / _norm(row["sketch_ref"][scope], positive=True) for row in rows]
        ratios[scope] = values
        candidates[scope] = RHO * float(median(values))
    binding = min(candidates, key=candidates.get)
    return {"rho": RHO, "lambda_sketch_ref": float(candidates[binding]), "lambda_candidates": candidates, "binding_scope": binding, "per_batch_old_over_sketch_ref_norm": ratios, "formula": ".1 * min_scope(median_batches(||grad_full_old_MP||_2 / ||grad_sketch_ref||_2))", "scope": "fixed initialization, four B32 batches; not an optimum, cap, significance test, or AdamW update ratio"}


def _summaries(rows: list[dict[str, Any]], selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    coefficient = float(selection["lambda_sketch_ref"])
    result = []
    for row in rows:
        stats = {}
        for scope in ("student_last_block", "pooled_head"):
            old, reference = row["old_total"][scope].double(), row["sketch_ref"][scope].double()
            old_norm, ref_norm = _norm(old, positive=True), _norm(reference, positive=True)
            stats[scope] = {"old_total_norm_l2_float64": old_norm, "sketch_ref_norm_l2_float64": ref_norm, "old_over_sketch_ref_norm_ratio_float64": old_norm / ref_norm, "cosine_old_total_sketch_ref_float64": _cosine(old, reference), "combined_norm_l2_float64": _norm(old + coefficient * reference, positive=True), "realized_sketch_ref_over_old_total": coefficient * ref_norm / old_norm}
        result.append({"batch": row["batch"], "losses": row["losses"], "scope_stats": stats, "matrix": row["matrix"], "forward_counts": row["forward_counts"], "teacher_capture": row["teacher_capture"], "unused": row["unused"], "raw_artifact": row["raw_artifact"]})
    return result


@contextmanager
def _forward_counts(model: Any) -> Iterator[dict[str, int]]:
    counts = {"model": 0, "student_visual": 0, "predictor": 0, "original_visual": 0}
    handles = []
    for module, name in ((model, "model"), (model.student_visual, "student_visual"), (model.predictor, "predictor"), (model.original_clip.visual, "original_visual")):
        if module is not None:
            handles.append(module.register_forward_hook(lambda *_args, _name=name, **_kwargs: counts.__setitem__(_name, counts[_name] + 1)))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def _captured_loss(model: Any, batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Run exactly one loss graph and restore the temporary teacher wrapper."""
    from spica.coupled_predictive_losses import coupled_region_loss

    captured: dict[str, Any] = {}
    had_instance_reference = "photo_reference" in model.__dict__
    original_instance_reference = model.__dict__.get("photo_reference")
    original_reference = model.photo_reference
    photos, corrupted = batch["photos"], batch["corrupted"]

    def reference(self: Any, values: Any, *args: Any, **kwargs: Any) -> Any:
        role = "photos" if values.data_ptr() == photos.data_ptr() else "corrupted" if values.data_ptr() == corrupted.data_ptr() else "unknown"
        if role == "unknown":
            raise AssertionError("photo_reference received an unbound input tensor")
        output = original_reference(values, *args, **kwargs)
        captured.setdefault("teacher", []).append({"role": role, "input": values.detach().cpu().clone(), "output": output.detach().cpu().clone(), "input_record": {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": _tensor_sha256(values)}, "output_record": {"shape": list(output.shape), "dtype": str(output.dtype), "sha256": _tensor_sha256(output), "requires_grad": bool(output.requires_grad)}})
        return output

    def output_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["output"] = output

    model.photo_reference = types.MethodType(reference, model)
    try:
        with _forward_counts(model) as counts:
            handle = model.register_forward_hook(output_hook)
            try:
                terms = coupled_region_loss(model, batch["clean"], corrupted, photos, batch["positive_indices"], batch["negative_indices"], batch["labels"], batch["photo_labels"], batch["photo_ids"], lambda_sig=0.0, sigreg=None, main_photo_objective="multi_positive_supervised_contrastive", lambda_sketch_ref=0.0)
            finally:
                handle.remove()
    finally:
        if had_instance_reference:
            model.photo_reference = original_instance_reference
        else:
            model.__dict__.pop("photo_reference", None)
    if len(captured.get("teacher", [])) != 2 or [x["role"] for x in captured["teacher"]] != ["photos", "corrupted"]:
        raise AssertionError(f"unexpected photo_reference calls: {captured.get('teacher')}")
    return terms, captured, counts


def _reference_oracle(terms: Mapping[str, Any], captured: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    output = captured.get("output")
    if output is None or output.q is None:
        raise AssertionError("model forward hook did not capture q")
    teacher = captured["teacher"][1]["output"].to(output.q.device)
    clean_q = output.q[: teacher.shape[0]]
    matrix = F.normalize(teacher, dim=-1) @ F.normalize(clean_q.detach(), dim=-1).T / TAU
    expected = F.cross_entropy(matrix, torch.arange(teacher.shape[0], device=matrix.device))
    if tuple(matrix.shape) != (teacher.shape[0], teacher.shape[0]):
        raise AssertionError(f"unexpected sketch-reference matrix shape: {tuple(matrix.shape)}")
    torch.testing.assert_close(terms["sketch_ref"].detach(), expected, rtol=2e-5, atol=2e-6)
    if not terms["sketch_ref"].requires_grad:
        raise AssertionError("sketch_ref lost clean-q autograd")
    return {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "sha256": _tensor_sha256(matrix), "temperature": TAU, "teacher_view": "corrupted_existing_masked_view", "student_view": "clean_q", "orientation": "teacher_rows_student_columns", "diagonal_targets": list(range(teacher.shape[0]))}, matrix.detach().cpu().float(), captured["teacher"][1]["output"], clean_q.detach().cpu().float()


def _load_official_setup(run_dir: Path, device: Any) -> tuple[Any, Any, Any, Any, dict[str, Any], dict[str, str]]:
    from spica.data import coupled_benchmark as benchmark_data
    from spica.models.clip import load_frozen_clip
    from spica.models.coupled_predictive import CoupledPredictiveModel
    from spica.train_coupled_benchmark import CLIP_MODEL, _verify_clip
    from spica.train_coupled_predictive import _initialization_hashes, _state_hash

    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if config.get("arm") != "F2_MP_Q_OFFICIAL" or config.get("method_version") != "coupled_predictive_mp_official_v1":
        raise AssertionError("historical run is not the locked official F2 MP-Q arm")
    required_config = {
        "dataset": "tuberlin_220_30", "batch_size": BATCH_SIZE, "seed": SEED,
        "num_workers": 4, "drop_last": True, "pin_memory": True,
        "architecture": "predictive_fusion_v2", "positive_pool": "full_official_train_photo_manifest",
        "main_photo_objective": "multi_positive_supervised_contrastive", "main_photo_temperature": TAU,
        "main_query": "mu_i", "training_main_query": "mu_i", "evaluation_query": "q",
        "evaluation_adapter": "QOnlyAdapter;predictor_forwards=0", "selection_policy": "none;final_only",
        "official_unseen_used_for_training": False, "official_unseen_used_for_selection": False,
    }
    if any(config.get(key) != value for key, value in required_config.items()):
        raise AssertionError("historical TU config is not the locked official F2 MP-Q identity")
    if config.get("batch_identity") != {"batch_size": 32, "drop_last": True, "full_batch_required": True, "num_workers": 4, "pin_memory": True}:
        raise AssertionError("historical batch identity does not match")
    if config.get("official_unseen_used_for_training") or config.get("official_unseen_used_for_selection"):
        raise AssertionError("official unseen data was marked as used")
    historical_init = json.loads((run_dir / "initialization.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints or checkpoints[0].get("step") != 0:
        raise AssertionError("run result lacks historical step-0 checkpoint metadata")
    step0 = checkpoints[0]
    protocol = benchmark_data.load_benchmark_protocol("configs/data/tuberlin_220_30.yaml", split="train")
    if protocol.identity["image_decode_scope"] != "none":
        raise AssertionError("protocol validation unexpectedly decoded official images")
    clip_identity = _verify_clip()
    if config.get("clip_identity") != clip_identity or config.get("protocol_identity") != _plain(protocol.identity) or config.get("dataset_config_sha256") != protocol.config_sha256:
        raise AssertionError("historical official setup identity mismatch")
    _seed42()
    bundle = load_frozen_clip(model_name=CLIP_MODEL, pretrained=str(clip_identity["path"]), device=device)
    classmap = {int(k): str(v) for k, v in protocol.train.class_names.items()}
    if len(classmap) != 220 or set(classmap) != set(protocol.train.class_ids):
        raise AssertionError("diagnostic must use the 220-class train map")
    model = CoupledPredictiveModel(bundle.encoder, bundle.tokenizer, classmap, architecture="predictive_fusion_v2").to(device)
    model.train(True)
    _all_float32(model, "model")
    initialization = _initialization_hashes(model)
    if initialization != historical_init or step0.get("initialization_hashes") != historical_init or step0.get("model_state_hash") != _state_hash(model):
        raise AssertionError("fresh initialization/current state differs from historical step-0 metadata")
    if model.original_clip.training or any(p.requires_grad for p in model.original_clip.parameters()):
        raise AssertionError("original frozen CLIP teacher is not frozen/eval")
    bindings = _historical_bindings(run_dir)
    return benchmark_data, protocol, bundle, model, {"historical": historical_init, "current": initialization, "config": config, "clip": clip_identity}, bindings


def _config(run_dir: Path, protocol: Any, scopes: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "diagnostic": "coupled_sketch_reference_gradient_scale", "arm": ARM, "method_version": METHOD, "dataset": "tuberlin_220_30", "seed": SEED, "batch_size": BATCH_SIZE, "batches": BATCHES, "temperature": TAU, "rho": RHO, "num_workers": 4, "drop_last": True, "optimizer_updates": 0, "official_unseen_used_for_training": False, "official_unseen_used_for_selection": False, "official_test_loader_created": False, "noeval": True, "no_eval": True, "no_retrieval_evaluation": True, "mask_policy": "existing_region_pair_seed4242_plus_update_severity_choice", "baseline_initialization": "historical_F2_MP_Q_OFFICIAL_step0", "protocol_identity": _plain(protocol.identity), "teacher_view": "corrupted_existing_masked_view", "student_view": "clean_q", "direction": "teacher_rows_student_columns", "lambda_sketch_ref": 0.0, "parameter_grad_all_none": True, "scopes": list(scopes), "parameter_scope": LAST_SCOPE, "historical_run": str(run_dir.relative_to(ROOT))}


def run(output: Path, run_dir: Path) -> dict[str, Any]:
    import torch
    from spica.provenance import capture_provenance
    from spica.train_coupled_predictive import _tensor_hash
    from run_coupled_campaign import _copy_source_archive

    if not torch.cuda.is_available():
        raise RuntimeError("production diagnostic requires CUDA; no GPU measurement was run")
    device = torch.device("cuda")
    source_config = {"diagnostic": METHOD, "arm": ARM, "dataset": "tuberlin_220_30", "seed": SEED, "batch_size": BATCH_SIZE, "batches": BATCHES, "temperature": TAU, "rho": RHO, "optimizer_updates": 0}
    provenance = capture_provenance(ROOT, resolved_config=source_config)
    manifest = {"schema_version": 1, "diagnostic": METHOD, "arm": ARM, "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"), "optimizer_updates": 0}
    _copy_source_archive(output, manifest, provenance)
    component_hashes = {}
    snapshot = {str(x["path"]): str(x["sha256"]) for x in provenance["source_snapshot"]["manifest"]}
    for relative in COMPONENTS:
        digest = _file_sha256(ROOT / relative)
        if snapshot.get(relative) != digest:
            raise AssertionError(f"component is not bound by source snapshot: {relative}")
        component_hashes[relative] = digest
    _json(output / "component_bindings.json", {"schema_version": 1, "components": list(COMPONENTS), "sha256": component_hashes})
    traces, masks = _first_jsonl(run_dir / "observation_trace.jsonl", BATCHES * BATCH_SIZE), _first_jsonl(run_dir / "mask_metadata.jsonl", BATCHES * BATCH_SIZE)
    benchmark_data, protocol, bundle, model, setup, bindings = _load_official_setup(run_dir, device)
    _last_block_parameters(model, expected=(12, 7_087_872))
    scopes, layout = _scope_parameters(model), None
    layout = _layout(scopes)
    _json(output / "gradient_layout.json", layout)
    config = _config(run_dir, protocol, scopes)
    _json(output / "resolved_config.json", config)
    _json(output / "provenance.json", {"source_snapshot": provenance["source_snapshot"], "historical_bindings": bindings, "component_sha256": component_hashes})
    model_before, teacher_before = _module_hash(model), _module_hash(model.original_clip)
    rows, records = [], []
    loader = benchmark_data.make_train_loader(protocol, bundle.transform, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=True, seed=SEED)
    expected_generator = torch.Generator().manual_seed(SEED).get_state()
    if loader.generator is None or not torch.equal(loader.generator.get_state(), expected_generator):
        raise AssertionError("official train loader does not expose seed-42 generator")
    iterator = iter(loader)
    for batch_index in range(BATCHES):
        raw = next(iterator)
        batch_cpu = benchmark_data.prepare_batch(protocol, raw, step=batch_index)
        lo, hi = batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE
        if len(batch_cpu["trace"]) != BATCH_SIZE or len(batch_cpu["mask_metadata"]["rows"]) != BATCH_SIZE:
            raise AssertionError(f"batch {batch_index} is not exactly B32 trace/mask")
        if batch_cpu["clean"].shape[0] != BATCH_SIZE or batch_cpu["corrupted"].shape[0] != BATCH_SIZE:
            raise AssertionError(f"batch {batch_index} has unexpected view batch dimensions")
        if [_canonical(x) for x in batch_cpu["trace"]] != [_canonical(x) for x in traces[lo:hi]] or [_canonical(x) for x in batch_cpu["mask_metadata"]["rows"]] != [_canonical(x) for x in masks[lo:hi]]:
            raise AssertionError(f"historical trace/mask mismatch at batch {batch_index}")
        records.append({"batch": batch_index, "trace": batch_cpu["trace"], "mask_metadata": batch_cpu["mask_metadata"], "photo_ids": list(batch_cpu["photo_ids"]), "labels": batch_cpu["labels"].tolist(), "photo_labels": batch_cpu["photo_labels"].tolist(), "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": _tensor_hash(v)} for k, v in batch_cpu.items() if isinstance(v, torch.Tensor)}})
        batch = _move_batch(batch_cpu, device)
        state_before, teacher_state_before, rng_before = _module_hash(model), _module_hash(model.original_clip), _global_rng_state()
        terms, captured, counts = _captured_loss(model, batch)
        matrix_meta, matrix, teacher, clean_q = _reference_oracle(terms, captured)
        expected_counts = {"model": 1, "student_visual": 0, "predictor": 1, "original_visual": 2}
        if counts != expected_counts:
            raise AssertionError(f"unexpected forward counts at batch {batch_index}: {counts} != {expected_counts}")
        old_per, old_flat, old_unused = _gradients(terms["total"], scopes, retain_graph=True)
        ref_per, ref_flat, ref_unused = _gradients(terms["sketch_ref"], scopes, retain_graph=False)
        if rng_before != _global_rng_state() or _module_hash(model) != state_before or _module_hash(model.original_clip) != teacher_state_before:
            raise AssertionError("no-update state/RNG contract failed")
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError("diagnostic populated model.grad")
        raw_name = f"raw_gradients_batch{batch_index:02d}.pt"
        torch.save({"schema_version": 1, "format": 1, "batch": batch_index, "losses": {"old_total": float(terms["total"].detach().cpu()), "sketch_ref": float(terms["sketch_ref"].detach().cpu())}, "gradients": {"old_total": old_per, "sketch_ref": ref_per}, "flat_gradients": {"old_total": old_flat, "sketch_ref": ref_flat}, "unused": {"old_total": old_unused, "sketch_ref": ref_unused}, "sketch_ref_matrix": matrix, "teacher": teacher, "clean_q": clean_q, "layout": layout}, output / raw_name)
        loaded = torch.load(output / raw_name, map_location="cpu", weights_only=True)
        if loaded["schema_version"] != 1 or loaded["format"] != 1 or loaded["unused"] != {"old_total": old_unused, "sketch_ref": ref_unused}:
            raise AssertionError("raw artifact schema rejected")
        rows.append({"batch": batch_index, "losses": {"old_total": float(terms["total"].detach().cpu()), "sketch_ref": float(terms["sketch_ref"].detach().cpu())}, "old_total": old_flat, "sketch_ref": ref_flat, "matrix": matrix_meta, "forward_counts": counts, "teacher_capture": [{k: v for k, v in item.items() if k.endswith("record") or k == "role"} for item in captured["teacher"]], "unused": {"old_total": old_unused, "sketch_ref": ref_unused}, "raw_artifact": _artifact(output, raw_name)})
        del terms, batch, raw, loaded, captured, batch_cpu
    selection = _select(rows)
    measured = _summaries(rows, selection)
    _json(output / "batch_records.json", records)
    _json(output / "lambda_selection.json", selection)
    _json(output / "measured_rows.json", measured)
    source_after = capture_provenance(ROOT, resolved_config=source_config)["source_snapshot"]
    if source_after["sha256"] != provenance["source_snapshot"]["sha256"] or _module_hash(model) != model_before or _module_hash(model.original_clip) != teacher_before:
        raise AssertionError("source/model changed during diagnostic")
    generated = ["source_snapshot/index.json", "source_snapshot/provenance.json", "source_snapshot/execution_manifest.json", "component_bindings.json", "gradient_layout.json", "resolved_config.json", "provenance.json", "batch_records.json", "lambda_selection.json", "measured_rows.json"]
    result = {"schema_version": 1, "status": "MEASURED_PENDING_REVIEW", "verified": False, "diagnostic": METHOD, "campaign": METHOD, "arm": ARM, "config": {**config, "parameter_grad_all_none": True}, "selection": selection, "rows": measured, "source_snapshot_hash": provenance["source_snapshot"]["sha256"], "source_snapshot_hash_after": source_after["sha256"], "component_sha256": component_hashes, "component_bindings": list(COMPONENTS), "historical_bindings": bindings, "initialization_hashes": setup["current"], "initialization_matches_historical": setup["current"] == setup["historical"], "clip_identity": setup["clip"], "protocol_identity": _plain(protocol.identity), "model_state_hash_before": model_before, "model_state_hash_after": _module_hash(model), "teacher_state_hash_before": teacher_before, "teacher_state_hash_after": _module_hash(model.original_clip), "parameter_grad_all_none": True, "optimizer_updates": 0, "artifacts": [_artifact(output, name) for name in generated], "finished_utc": datetime.now(timezone.utc).isoformat()}
    result["artifacts"].extend(row["raw_artifact"] for row in rows)
    _json(output / "provenance_after.json", {"source_snapshot": source_after, "component_sha256": component_hashes})
    result["artifacts"].append(_artifact(output, "provenance_after.json"))
    _json(output / "diagnostic_result.json", result)
    return result


def _archived_loss_module() -> Any:
    import importlib.util
    path = ROOT / "outputs/coupled_benchmark_execution_20260910T164500Z/source_snapshot/files/src/spica/coupled_predictive_losses.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("spica._historical_f2_mp_loss", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "spica"
    spec.loader.exec_module(module)
    return module


def _self_check(output: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from check_coupled_benchmark_cpu import synthetic_args, tiny_model

    model = tiny_model(42)
    clean, corrupted, photos, positive, negative, labels, photo_labels, photo_ids = synthetic_args()
    if clean.shape[0] != 2 or corrupted.shape[0] != 2:
        raise AssertionError("tiny self-check fixture is not the expected B2 graph")
    batch = {"clean": clean, "corrupted": corrupted, "photos": photos, "positive_indices": positive, "negative_indices": negative, "labels": labels, "photo_labels": photo_labels, "photo_ids": photo_ids}
    terms, captured, counts = _captured_loss(model, batch)
    if counts != {"model": 1, "student_visual": 0, "predictor": 1, "original_visual": 2}:
        raise AssertionError(f"tiny forward graph changed: {counts}")
    matrix_meta, matrix, teacher, clean_q = _reference_oracle(terms, captured)
    if matrix_meta["orientation"] != "teacher_rows_student_columns":
        raise AssertionError("tiny oracle orientation changed")
    oracle = F.cross_entropy(F.normalize(teacher, dim=-1) @ F.normalize(clean_q, dim=-1).T / TAU, torch.arange(clean.shape[0]))
    torch.testing.assert_close(terms["sketch_ref"].detach(), oracle, rtol=2e-5, atol=2e-6)
    archived = _archived_loss_module()
    old = archived.coupled_region_loss(model, clean, corrupted, photos, positive, negative, labels, photo_labels, photo_ids, lambda_sig=0.0, sigreg=None, main_photo_objective="multi_positive_supervised_contrastive")
    torch.testing.assert_close(terms["total"], old["total"], rtol=0, atol=0)
    _last_block_parameters(model, expected=(12, 872))
    scopes = _scope_parameters(model)
    _gradients(terms["total"], scopes, retain_graph=True)
    _gradients(terms["sketch_ref"], scopes, retain_graph=False)
    toy = [{"old_total": {"student_last_block": torch.tensor([3., 4.]), "pooled_head": torch.tensor([6., 8.])}, "sketch_ref": {"student_last_block": torch.tensor([1., 0.]), "pooled_head": torch.tensor([2., 0.])}}] * BATCHES
    selected = _select(toy)
    if selected["binding_scope"] != "student_last_block" or selected["lambda_sketch_ref"] != 0.5:
        raise AssertionError("tiny lambda selection changed")
    result = {"schema_version": 1, "status": "PASS", "verified": False, "scope": "synthetic CPU contract only; no official data/model claim", "source_provenance": "SYNTHETIC_UNVERIFIED", "checks": ["production loss graph captures two teacher calls and one q output", "detached reconstructed teacher-row/student-column algebra", "historical total exact at explicit zero coefficient", "finite non-unused gradients in both calibration scopes", "standard-library median toy selection", "fresh output root"], "selection": selected, "source_sha256": _file_sha256(Path(__file__)), "optimizer_updates": 0}
    _json(output / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    output = None
    try:
        output = _fresh_output(args.output_dir)
        result = _self_check(output) if args.self_check else run(output, HISTORICAL_RUN)
        print(json.dumps({"status": result["status"], "output": str(output), "selection": result.get("selection")}, sort_keys=True))
        return 0
    except Exception as error:
        if output is not None:
            _json(output / ("receipt.json" if args.self_check else "diagnostic_result.json"), {"schema_version": 1, "status": "FAIL", "verified": False, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
