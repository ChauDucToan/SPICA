#!/usr/bin/env python3
"""Inert-by-default TU-only SREF campaign runner.

The runner validates a reviewed, still-unverified calibration and a real two-step
SREF smoke before it can launch one trainer child followed by the shared
benchmark evaluator child.  It owns no model or evaluator implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from run_coupled_benchmarks import COMPONENTS as LEGACY_COMPONENTS  # noqa: E402
from run_coupled_campaign import _copy_source_archive, _json_atomic, _now, _start_child  # noqa: E402

METHOD = "coupled_predictive_mp_sketch_ref_official_v1"
ARM = "F2_MP_Q_SREF_OFFICIAL"
DATASET = "tuberlin_220_30"
ARCHITECTURE = "predictive_fusion_v2"
TOTAL_STEPS, WARMUP_STEPS, BATCH_SIZE, SMOKE_STEPS = 1189, 59, 32, 2
TEMPERATURE, RHO = 0.07, 0.1
BASELINE_ROOT = ROOT / "outputs/coupled_benchmark_execution_20260910T164500Z"
BASELINE_RUN = BASELINE_ROOT / "runs" / DATASET
BASELINE_CHECKPOINT_SHA256 = "a39a0a6a6107566afa43af44410f25d40ab8b36d28d75ac6a4bf0c7901752460"

# Copied deliberately instead of importing the diagnostic: importing it changes
# WANDB/HF environment defaults before this runner has decided to launch.
DIAGNOSTIC_COMPONENTS = (
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
COMPONENTS = tuple(sorted(set(LEGACY_COMPONENTS) | set(DIAGNOSTIC_COMPONENTS) | {
    "scripts/check_coupled_sketch_ref_cpu.py",
    "scripts/run_coupled_sketch_ref.py",
    "scripts/evaluate_coupled_benchmark.py",
}))
# The worker's raw receipt must authenticate its diagnostic inputs.  The
# launch gate separately authenticates the complete production union below.
REQUIRED_RAW_COMPONENTS = frozenset(DIAGNOSTIC_COMPONENTS)
BASELINE_FILES = (
    BASELINE_RUN / "run_result.json",
    BASELINE_RUN / "resolved_config.json",
    BASELINE_RUN / "initialization.json",
    BASELINE_RUN / "checkpoint_step1189.pt",
    BASELINE_ROOT / "results.json",
    BASELINE_ROOT / "evaluation" / DATASET / "summary.json",
    BASELINE_RUN / "observation_trace.jsonl",
    BASELINE_RUN / "mask_metadata.jsonl",
    BASELINE_RUN / "lr_history_every_step.jsonl",
)
_HASH = 64


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != _HASH:
        raise ValueError(f"{label} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA256 hex digest") from error
    return value


def _file(value: object, label: str = "file", *, base: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    raw = Path(value)
    path = (base / raw if not raw.is_absolute() else raw).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _hash_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    result: dict[str, str] = {}
    for name, digest in value.items():
        path = _file(name, f"{label} path")
        result[str(name)] = _digest(digest, f"{label}[{name!r}]")
        if _sha(path) != result[str(name)]:
            raise ValueError(f"{label} hash mismatch: {name}")
    return result


def _records(value: object, label: str, *, base: Path = ROOT) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError(f"{label} entries must contain paths")
        name = str(row["path"])
        if not name or Path(name).is_absolute() or ".." in Path(name).parts or name in seen:
            raise ValueError(f"invalid {label} path: {name!r}")
        path = _file(name, f"{label} path", base=base)
        digest = _digest(row.get("sha256"), f"{label} {name} hash")
        if _sha(path) != digest:
            raise ValueError(f"{label} hash mismatch: {name}")
        if "bytes" in row and row["bytes"] != path.stat().st_size:
            raise ValueError(f"{label} byte count mismatch: {name}")
        seen.add(name)
        result.append(dict(row))
    return result


def _source_hash() -> str:
    from spica.provenance import source_snapshot
    return str(source_snapshot(ROOT)["sha256"])


def _baseline() -> dict[str, Any]:
    for path in BASELINE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"fixed baseline file is missing: {path}")
    result, config = _json(BASELINE_FILES[0]), _json(BASELINE_FILES[1])
    if result.get("status") != "COMPLETE" or result.get("campaign") != "coupled_predictive_mp_official_v1" or result.get("arm") != "F2_MP_Q_OFFICIAL":
        raise ValueError("fixed TU baseline is not the official F2 MP-Q run")
    if result.get("dataset") != DATASET or result.get("step") != TOTAL_STEPS or result.get("source_snapshot_hash_after") != result.get("source_snapshot_hash"):
        raise ValueError("fixed TU baseline horizon/source binding is invalid")
    if config.get("architecture") != ARCHITECTURE or config.get("method_version") != result["campaign"] or config.get("arm") != result["arm"]:
        raise ValueError("fixed TU baseline config identity is invalid")
    if config.get("selection_policy") != "none;final_only" or config.get("official_unseen_used_for_training") is not False or config.get("official_unseen_used_for_selection") is not False:
        raise ValueError("fixed TU baseline has an unsafe selection/holdout policy")
    if result.get("frozen_original_state_hash_before") != result.get("frozen_original_state_hash_after"):
        raise ValueError("fixed TU baseline changed the frozen teacher")
    selections = result.get("selections")
    if not isinstance(selections, Mapping) or set(selections) != {"latest"} or selections["latest"].get("step") != TOTAL_STEPS:
        raise ValueError("fixed TU baseline is not final-only")
    latest = dict(selections["latest"])
    checkpoint = _file(latest.get("path"), "fixed baseline checkpoint", base=BASELINE_RUN)
    if _sha(checkpoint) != BASELINE_CHECKPOINT_SHA256 or latest.get("sha256") != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("fixed TU baseline checkpoint SHA256 mismatch")
    summary = _json(BASELINE_ROOT / "evaluation" / DATASET / "summary.json")
    if summary.get("status") != "COMPLETE" or len(summary.get("conditions", [])) != 10:
        raise ValueError("fixed TU baseline summary is not the ten-condition final evaluation")
    return {"result": result, "config": config, "summary": summary,
            "checkpoint": checkpoint, "files": {_relative(path): _sha(path) for path in BASELINE_FILES}}


def _validate_raw_and_receipt(path_value: object, baseline: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = _file(path_value, "reviewed SREF receipt")
    receipt = _json(receipt_path)
    if receipt.get("status") != "PASS" or receipt.get("verified") is not True or receipt.get("scope") != "SKETCH_REF_INITIALIZATION_CALIBRATION_ONLY":
        raise ValueError("reviewed receipt must be PASS/verified and calibration-only")
    raw_ref = receipt.get("raw_result")
    if not isinstance(raw_ref, Mapping):
        raise ValueError("reviewed receipt raw_result is required")
    # The receipt stores repository paths, not paths relative to the receipt.
    raw_path = _file(raw_ref.get("path"), "raw diagnostic", base=ROOT)
    raw_sha = _digest(raw_ref.get("sha256"), "raw diagnostic SHA256")
    if _sha(raw_path) != raw_sha:
        raise ValueError("raw diagnostic SHA256 mismatch")
    raw = _json(raw_path)
    if raw.get("status") != "MEASURED_PENDING_REVIEW" or raw.get("verified") is not False or raw.get("campaign") != METHOD or raw.get("arm") != ARM:
        raise ValueError("raw diagnostic is not the required unverified SREF result")
    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("raw diagnostic config is required")
    expected = {"dataset": DATASET, "seed": 42, "batch_size": BATCH_SIZE, "batches": 4,
                "temperature": TEMPERATURE, "rho": RHO, "optimizer_updates": 0,
                "official_unseen_used_for_training": False, "official_unseen_used_for_selection": False,
                "official_test_loader_created": False, "no_retrieval_evaluation": True,
                "teacher_view": "corrupted_existing_masked_view", "student_view": "clean_q"}
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(f"raw diagnostic config mismatch at {key}")
    direction = raw.get("direction", config.get("direction"))
    if direction is None and isinstance(raw.get("rows"), list):
        directions = {row.get("matrix", {}).get("orientation") for row in raw["rows"] if isinstance(row, Mapping) and isinstance(row.get("matrix"), Mapping)}
        direction = directions.pop() if len(directions) == 1 else None
    if direction != "teacher_rows_student_columns":
        raise ValueError("raw diagnostic direction is not teacher_rows_student_columns")
    if raw.get("blind_holdout_claim", config.get("blind_holdout_claim", False)) is True:
        raise ValueError("SREF calibration must not make a blind-holdout claim")
    for key in ("initialization_hashes", "protocol_identity", "clip_identity"):
        if raw.get(key) != baseline["config"].get("initialization_hashes" if key == "initialization_hashes" else key):
            raise ValueError(f"raw diagnostic {key} is not the fixed baseline identity")
    if raw.get("model_state_hash_before") != raw.get("model_state_hash_after") or raw.get("teacher_state_hash_before") != raw.get("teacher_state_hash_after"):
        raise ValueError("raw diagnostic state hashes changed")
    if raw.get("parameter_grad_all_none") is not True or raw.get("optimizer_updates") != 0:
        raise ValueError("raw diagnostic is not a no-update measurement")
    if not isinstance(raw.get("rows"), list) or len(raw["rows"]) != 4:
        raise ValueError("raw diagnostic must contain all four measured batches")
    selection = raw.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("raw diagnostic selection is required")
    candidates = selection.get("lambda_candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != {"student_last_block", "pooled_head"}:
        raise ValueError("raw diagnostic must contain exactly the two required selection scopes")
    for name, value in candidates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"invalid positive lambda candidate: {name}")
    chosen_scope = selection.get("binding_scope")
    chosen = selection.get("lambda_sketch_ref")
    if chosen_scope not in candidates or chosen != candidates[chosen_scope] or float(chosen) != min(map(float, candidates.values())):
        raise ValueError("selected lambda is not the positive minimum candidate")
    if selection.get("direction", direction) != "teacher_rows_student_columns":
        raise ValueError("selection direction mismatch")
    artifacts = _records(raw.get("artifacts"), "raw diagnostic artifacts", base=raw_path.parent)
    expected_artifacts = {"source_snapshot/index.json", "source_snapshot/provenance.json", "source_snapshot/execution_manifest.json", "component_bindings.json", "gradient_layout.json", "resolved_config.json", "provenance.json", "provenance_after.json", "batch_records.json", "lambda_selection.json", "measured_rows.json", *(f"raw_gradients_batch{i:02d}.pt" for i in range(4))}
    if {row["path"] for row in artifacts} != expected_artifacts:
        raise ValueError("raw diagnostic artifact coverage mismatch")
    raw_components = _hash_map(raw.get("component_sha256"), "raw diagnostic component_sha256")
    if REQUIRED_RAW_COMPONENTS != set(raw_components):
        raise ValueError("raw diagnostic component coverage is incomplete")
    if dict(receipt.get("selection", {})) != dict(selection) or receipt.get("component_sha256") != raw_components:
        raise ValueError("reviewed receipt does not exactly bind raw selection/components")
    evidence = _records(receipt.get("evidence"), "independent review evidence")
    required_inputs = {_relative(raw_path), *(_relative(raw_path.parent / row["path"]) for row in artifacts), *(_relative(_file(row["path"])) for row in evidence), *baseline["files"]}
    inputs = _hash_map(receipt.get("input_sha256"), "reviewed receipt input_sha256")
    if not required_inputs <= set(inputs):
        raise ValueError("reviewed receipt input_sha256 does not bind every raw artifact, review, and baseline file")
    if not raw.get("source_snapshot_hash") or raw.get("source_snapshot_hash_after") != raw.get("source_snapshot_hash"):
        raise ValueError("raw diagnostic source binding is missing or changed")
    return {"path": receipt_path, "sha256": _sha(receipt_path), "raw_path": raw_path,
            "raw_sha256": raw_sha, "raw": raw, "selection": dict(selection),
            "lambda": float(chosen), "components": raw_components, "evidence": evidence,
            "inputs": inputs}


def _validate_smoke(root: Path, baseline: Mapping[str, Any], lambda_value: float) -> dict[str, Any]:
    result, config = _json(root / "run_result.json"), _json(root / "resolved_config.json")
    from evaluate_coupled_benchmark import _run_result
    _run_result(root, arm=ARM)
    if result.get("status") != "COMPLETE" or result.get("step") != SMOKE_STEPS or result.get("completed_steps") != SMOKE_STEPS or result.get("arm") != ARM or result.get("campaign") != METHOD:
        raise ValueError("SREF smoke is not a complete two-update run")
    if config.get("arm") != ARM or config.get("method_version") != METHOD or config.get("dataset") != DATASET or config.get("smoke") is not True:
        raise ValueError("SREF smoke config identity is invalid")
    if config.get("sketch_ref_lambda") != lambda_value or config.get("lambda_sketch_ref") != lambda_value:
        raise ValueError("SREF smoke does not serialize the selected lambda in both fields")
    if config.get("total_steps") != TOTAL_STEPS or config.get("warmup_steps") != WARMUP_STEPS or config.get("actual_updates") != SMOKE_STEPS:
        raise ValueError("SREF smoke schedule mismatch")
    if config.get("initialization_hashes") != baseline["config"].get("initialization_hashes") or config.get("protocol_identity") != baseline["config"].get("protocol_identity") or config.get("clip_identity") != baseline["config"].get("clip_identity"):
        raise ValueError("SREF smoke initialization/protocol/CLIP identity differs from baseline")
    if config.get("frozen_original_state_hash") != baseline["config"].get("frozen_original_state_hash"):
        raise ValueError("SREF smoke frozen teacher identity differs from baseline")
    if config.get("architecture") != baseline["config"].get("architecture") or config.get("model_trainable_parameters") != baseline["config"].get("model_trainable_parameters") or config.get("model_total_parameters") != baseline["config"].get("model_total_parameters"):
        raise ValueError("SREF smoke model identity differs from baseline")
    if config.get("optimizer") != baseline["config"].get("optimizer"):
        raise ValueError("SREF smoke optimizer differs from baseline")
    coeff = dict(baseline["config"].get("loss_coefficient_identity", {}))
    coeff["sketch_ref"] = lambda_value
    if config.get("loss_coefficient_identity") != coeff:
        raise ValueError("SREF smoke coefficients differ from baseline plus selected lambda")
    if result.get("source_snapshot_hash") != config.get("source_snapshot_hash") or result.get("source_snapshot_hash_after") != result.get("source_snapshot_hash"):
        raise ValueError("SREF smoke source/result self-binding failed")
    if result.get("frozen_original_state_hash_before") != result.get("frozen_original_state_hash_after"):
        raise ValueError("SREF smoke changed the frozen teacher")
    gate = result.get("optimizer_moment_gate", {})
    if gate.get("state_tensor_count") != 358 or gate.get("finite") is not True or gate.get("nonzero_moment") is not True:
        raise ValueError("SREF smoke optimizer moments are not the expected finite nonzero 2-update gate")
    if result.get("trace_count") != 64 or result.get("mask_count") != 64 or result.get("expected_trace_count") != 64 or result.get("expected_mask_count") != 64:
        raise ValueError("SREF smoke does not contain exactly 64 trace/mask rows")
    latest = result.get("selections", {}).get("latest")
    if not isinstance(latest, Mapping) or latest.get("step") != SMOKE_STEPS:
        raise ValueError("SREF smoke latest checkpoint is not step 2")
    checkpoint = _file(latest.get("path"), "SREF smoke latest checkpoint", base=root)
    if _sha(checkpoint) != latest.get("sha256"):
        raise ValueError("SREF smoke latest checkpoint SHA mismatch")
    for name, count in (("observation_trace.jsonl", 64), ("mask_metadata.jsonl", 64), ("lr_history_every_step.jsonl", 3)):
        if len((root / name).read_text(encoding="utf-8").splitlines()) < count:
            raise ValueError(f"SREF smoke lacks the required {count} rows in {name}")
    for name, count in (("observation_trace.jsonl", 64), ("mask_metadata.jsonl", 64), ("lr_history_every_step.jsonl", 3)):
        smoke_lines = (root / name).read_text(encoding="utf-8").splitlines(keepends=True)[:count]
        baseline_lines = (BASELINE_RUN / name).read_text(encoding="utf-8").splitlines(keepends=True)[:count]
        if smoke_lines != baseline_lines:
            raise ValueError(f"SREF smoke first-window differs from fixed baseline: {name}")
    return {"path": root, "result": result, "config": config, "initialization_hashes": config["initialization_hashes"]}


def _validate_gate(path_value: object, diagnostic: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = _file(path_value, "SREF launch gate")
    gate = _json(path)
    if gate.get("status") != "PASS" or gate.get("method") != METHOD or gate.get("arm") != ARM:
        raise ValueError("SREF launch gate method/arm/status mismatch")
    if gate.get("diagnostic_sha256", gate.get("reviewed_receipt_sha256")) != diagnostic["sha256"]:
        raise ValueError("launch gate does not bind the reviewed receipt")
    components = gate.get("component_sha256")
    if not isinstance(components, Mapping) or set(components) != set(COMPONENTS):
        raise ValueError("launch gate component set is not the exact SREF union")
    for name in COMPONENTS:
        if _sha(_file(name)) != _digest(components[name], f"gate component {name}"):
            raise ValueError(f"launch gate component changed: {name}")
    records = _records(gate.get("evidence"), "launch gate evidence")
    evidence_paths = {row["path"] for row in records}
    for role in ("cpu_review", "optimizer_review", "train_probe"):
        name = gate.get("required_evidence", {}).get(role)
        if name not in evidence_paths:
            raise ValueError(f"missing named gate evidence: {role}")
        evidence = _json(_file(name))
        if evidence.get("status") not in {"PASS", "TRAIN_ONLY_PREFLIGHT_NOT_OFFICIAL"}:
            raise ValueError(f"gate evidence not accepted: {role}")
    smoke_roots = gate.get("smoke_roots")
    if not isinstance(smoke_roots, Mapping) or set(smoke_roots) != {DATASET}:
        raise ValueError("launch gate must contain exactly the TU SREF smoke root")
    smoke = _file(smoke_roots[DATASET] + "/run_result.json", "SREF smoke result") if isinstance(smoke_roots[DATASET], str) else None
    if smoke is None:
        raise ValueError("invalid SREF smoke root")
    smoke_root = smoke.parent
    datasets = {DATASET: {"config": "configs/data/tuberlin_220_30.yaml", "total_steps": TOTAL_STEPS, "warmup_steps": WARMUP_STEPS}}
    if gate.get("datasets") != datasets or gate.get("no_retry_or_resume") is not True:
        raise ValueError("launch gate dataset horizon/retry policy is invalid")
    smoke = _validate_smoke(smoke_root, baseline, diagnostic["lambda"])
    if smoke["result"]["source_snapshot_hash"] != gate.get("smoke_source_snapshot_hash"):
        raise ValueError("gate does not bind reviewed smoke source")
    # Full source may differ only through later docs/receipts; production components must match.
    index = _json(smoke_root / "source_snapshot/index.json")
    archived = {row["path"]: row["sha256"] for row in index["manifest"]}
    if index["sha256"] != gate["smoke_source_snapshot_hash"] or any(archived.get(name) != components[name] for name in COMPONENTS):
        raise ValueError("smoke production component/source binding mismatch")
    return {"path": path, "sha256": _sha(path), "payload": gate, "components": dict(components)}, smoke_root


def _prefix_equal(run: Path, smoke: Path) -> None:
    for name, count in (("observation_trace.jsonl", 64), ("mask_metadata.jsonl", 64), ("lr_history_every_step.jsonl", 3)):
        if (run / name).read_text(encoding="utf-8").splitlines(keepends=True)[:count] != (smoke / name).read_text(encoding="utf-8").splitlines(keepends=True)[:count]:
            raise ValueError(f"final SREF first-window differs from smoke: {name}")


def _validate_train(run: Path, smoke: Mapping[str, Any], baseline: Mapping[str, Any], source_hash: str, lambda_value: float) -> dict[str, Any]:
    result, config = _json(run / "run_result.json"), _json(run / "resolved_config.json")
    from evaluate_coupled_benchmark import _run_result
    _run_result(run, arm=ARM)
    if result.get("status") != "COMPLETE" or result.get("campaign") != METHOD or result.get("arm") != ARM or result.get("dataset") != DATASET or result.get("step") != TOTAL_STEPS:
        raise ValueError("SREF final trainer result is not the locked complete TU run")
    if result.get("source_snapshot_hash") != source_hash or result.get("source_snapshot_hash_after") != source_hash or config.get("source_snapshot_hash") != source_hash:
        raise ValueError("SREF final source binding failed")
    if config.get("method_version") != METHOD or config.get("arm") != ARM or config.get("dataset") != DATASET or config.get("smoke") is not False or config.get("selection_policy") != "none;final_only":
        raise ValueError("SREF final config identity/selection policy is invalid")
    if config.get("official_unseen_used_for_training") is not False or config.get("official_unseen_used_for_selection") is not False:
        raise ValueError("SREF final config has an unsafe holdout policy")
    if config.get("total_steps") != TOTAL_STEPS or config.get("warmup_steps") != WARMUP_STEPS or config.get("actual_updates") != TOTAL_STEPS:
        raise ValueError("SREF final schedule mismatch")
    for name in ("observation_trace.jsonl", "mask_metadata.jsonl"):
        with (run / name).open() as handle:
            if sum(1 for _ in handle) != TOTAL_STEPS * BATCH_SIZE:
                raise ValueError(f"SREF actual record count mismatch: {name}")
    if config.get("sketch_ref_lambda") != lambda_value or config.get("lambda_sketch_ref") != lambda_value:
        raise ValueError("SREF final config does not serialize both lambda fields")
    coeff = dict(baseline["config"]["loss_coefficient_identity"])
    coeff["sketch_ref"] = lambda_value
    if config.get("loss_coefficient_identity") != coeff or config.get("optimizer") != baseline["config"].get("optimizer"):
        raise ValueError("SREF final coefficients or optimizer differ from baseline")
    if config.get("architecture") != baseline["config"].get("architecture") or config.get("model_trainable_parameters") != baseline["config"].get("model_trainable_parameters") or config.get("model_total_parameters") != baseline["config"].get("model_total_parameters"):
        raise ValueError("SREF final model identity differs from baseline")
    for key in ("initialization_hashes", "protocol_identity", "clip_identity"):
        if config.get(key) != baseline["config"].get(key):
            raise ValueError(f"SREF final {key} differs from baseline")
    if result.get("trace_count") != TOTAL_STEPS * BATCH_SIZE or result.get("mask_count") != TOTAL_STEPS * BATCH_SIZE or result.get("expected_trace_count") != TOTAL_STEPS * BATCH_SIZE or result.get("expected_mask_count") != TOTAL_STEPS * BATCH_SIZE:
        raise ValueError("SREF final trace/mask counts are not 38048")
    if result.get("metrics") is not None or config.get("metrics") is not None:
        raise ValueError("SREF trainer must not select on metrics")
    if result.get("frozen_original_state_hash_before") != result.get("frozen_original_state_hash_after"):
        raise ValueError("SREF final changed the frozen teacher")
    gate = result.get("optimizer_moment_gate", {})
    if gate.get("state_tensor_count") != 358 or gate.get("finite") is not True or gate.get("nonzero_moment") is not True:
        raise ValueError("SREF final optimizer moment gate failed")
    if result.get("model_state_hash_before_updates") == result.get("model_state_hash_after_updates"):
        raise ValueError("SREF final performed no update")
    selections = result.get("selections")
    if not isinstance(selections, Mapping) or set(selections) != {"latest"} or selections["latest"].get("step") != TOTAL_STEPS:
        raise ValueError("SREF final checkpoint selection is not final-only")
    latest = dict(selections["latest"])
    checkpoint = _file(latest.get("path"), "SREF final checkpoint", base=run)
    if _sha(checkpoint) != latest.get("sha256"):
        raise ValueError("SREF final checkpoint SHA mismatch")
    _prefix_equal(run, Path(smoke["path"]))
    return {"result": result, "config": config, "checkpoint": checkpoint, "checkpoint_sha256": latest["sha256"]}


def _validate_evaluation(path: Path, train: Mapping[str, Any], baseline: Mapping[str, Any], source_hash: str) -> dict[str, Any]:
    summary = _json(path / "summary.json")
    if summary.get("status") != "COMPLETE" or summary.get("official_test_evaluated") is not True or summary.get("step") != TOTAL_STEPS or summary.get("source_snapshot_hash") != source_hash:
        raise ValueError("SREF official evaluation is incomplete or source-unbound")
    if summary.get("predictor_forwards") != 0 or summary.get("model_state_before") != summary.get("model_state_after"):
        raise ValueError("SREF evaluation query/state gate failed")
    if summary.get("model_state_before") != train["result"]["model_state_hash_after_updates"] or summary.get("checkpoint_selection", {}).get("latest", {}).get("sha256") != train["checkpoint_sha256"]:
        raise ValueError("SREF evaluation final checkpoint/state mismatch")
    if summary.get("query_count") != 2400 or summary.get("gallery_count") != 27989:
        raise ValueError("SREF official query/gallery count mismatch")
    rows, base_rows = summary.get("conditions"), baseline["summary"].get("conditions")
    if not isinstance(rows, list) or not isinstance(base_rows, list) or len(rows) != 10 or len(base_rows) != 10:
        raise ValueError("SREF evaluation must expose clean plus nine mask conditions")
    deltas: dict[str, dict[str, float]] = {}
    for row, base in zip(rows, base_rows, strict=True):
        if row.get("condition") != base.get("condition"):
            raise ValueError("SREF evaluation condition order differs from fixed baseline")
        metrics, base_metrics = row.get("metrics"), base.get("metrics")
        if not isinstance(metrics, Mapping) or not isinstance(base_metrics, Mapping):
            raise ValueError("evaluation condition metrics are missing")
        deltas[str(row["condition"])] = {name: float(metrics[name]) - float(base_metrics[name]) for name in ("full_mAP", "P@200", "mAP@200_prefix_positive", "mAP@200_all_relevant", "mAP@200_min_relevant_k")}
    return {"summary": summary, "summary_sha256": _sha(path / "summary.json"), "metric_deltas": deltas}


def launch(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.parent != (ROOT / "outputs").resolve():
        raise ValueError("--output must be a fresh direct child of outputs/")
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise RuntimeError("at least 24GiB free is required; no artifact cleanup is permitted")
    baseline = _baseline()
    diagnostic = _validate_raw_and_receipt(args.diagnostic, baseline)
    gate, smoke_root = _validate_gate(args.gate, diagnostic, baseline)
    resolved = {"method": METHOD, "arm": ARM, "dataset": DATASET, "total_steps": TOTAL_STEPS,
                "warmup_steps": WARMUP_STEPS, "lambda_sketch_ref": diagnostic["lambda"],
                "sketch_ref_lambda": diagnostic["lambda"], "gate_sha256": gate["sha256"],
                "diagnostic_sha256": diagnostic["sha256"], "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
                "official_unseen_used_for_training": False, "official_unseen_used_for_selection": False,
                "selection_policy": "none;final_only", "no_retry_or_resume": True,
                "official_baseline_seen_before_design": True, "blind_holdout_claim": False}
    from spica.provenance import capture_provenance
    provenance = capture_provenance(ROOT, resolved_config=resolved)
    source_hash = str(provenance["source_snapshot"]["sha256"])
    manifest = {"head_commit": provenance["head_commit"], "source_snapshot_hash": source_hash,
                "source_snapshot": provenance["source_snapshot"], "resolved_config": resolved}
    output.mkdir()
    runtime = {"status": "LAUNCHING", "method": METHOD, "arm": ARM, "dataset": DATASET,
               "started_at": _now(), "source_snapshot_hash": source_hash, "child_pid": None,
               "phase": None, "pids": {}}
    _json_atomic(output / "runtime.json", runtime)
    try:
        _json_atomic(output / "execution_manifest.json", manifest)
        _copy_source_archive(output, manifest, provenance)
        shutil.copyfile(args.gate, output / "launch_gate.json")
        shutil.copyfile(args.diagnostic, output / "diagnostic_verified.json")
        shutil.copyfile(BASELINE_ROOT / "evaluation" / DATASET / "summary.json", output / "baseline_tuberlin_summary.json")
        if _sha(output / "launch_gate.json") != gate["sha256"] or _sha(output / "diagnostic_verified.json") != diagnostic["sha256"]:
            raise RuntimeError("archived launch inputs changed during copy")
        run = output / "runs" / DATASET
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "WANDB_MODE": "online", "PYTHONPATH": str(ROOT / "src")}
        commands = {
            "train": [sys.executable, "-m", "spica.train_coupled_benchmark", "--dataset", DATASET,
                      "--output-dir", str(run), "--campaign-root", str(output), "--device", "cuda",
                      "--wandb-mode", "online", "--lambda-sketch-ref", str(diagnostic["lambda"])],
            "evaluate": [sys.executable, "scripts/evaluate_coupled_benchmark.py", "--run-dir", str(run),
                         "--output-dir", str(output / "evaluation" / DATASET), "--device", "cuda", "--arm", ARM],
        }
        train = None
        for phase, command in commands.items():
            if _source_hash() != source_hash:
                raise RuntimeError(f"source changed before {phase}; refusing launch")
            runtime.update(status="RUNNING", phase=phase, command=command, child_pid=None)
            _json_atomic(output / "runtime.json", runtime)
            child, stdout, stderr = _start_child(command, cwd=ROOT, env=env,
                stdout=output / "logs" / f"{phase}.stdout.log", stderr=output / "logs" / f"{phase}.stderr.log")
            runtime["child_pid"] = child.pid
            runtime["pids"][phase] = child.pid
            _json_atomic(output / "runtime.json", runtime)
            try:
                code = child.wait()
            finally:
                stdout.close()
                stderr.close()
            runtime.update(child_pid=None, exit_code=code)
            _json_atomic(output / "runtime.json", runtime)
            if code:
                raise RuntimeError(f"SREF {phase} child exited {code}; no retry")
            if _source_hash() != source_hash:
                raise RuntimeError(f"source changed during {phase}; refusing to certify")
            if phase == "train":
                train = _validate_train(run, _validate_smoke(smoke_root, baseline, diagnostic["lambda"]), baseline, source_hash, diagnostic["lambda"])
                _json_atomic(run / "parent_final_gate.json", {"status": "PASS", "scope": "final_checkpoint_final_only_before_shared_evaluation", "checkpoint_sha256": train["checkpoint_sha256"], "source_snapshot_hash": source_hash})
        if train is None:
            raise RuntimeError("trainer result was not validated")
        evaluation = _validate_evaluation(output / "evaluation" / DATASET, train, baseline, source_hash)
        results = {"status": "TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED", "raw_status": "UNVERIFIED",
                   "method": METHOD, "arm": ARM, "dataset": DATASET, "step": TOTAL_STEPS,
                   "lambda_sketch_ref": diagnostic["lambda"], "source_snapshot_hash": source_hash,
                   "checkpoint_sha256": train["checkpoint_sha256"], "evaluation_summary_sha256": evaluation["summary_sha256"],
                   "clean_and_nine_condition_metric_deltas": evaluation["metric_deltas"],
                   "clean": evaluation["summary"]["clean"], "masked_macro": evaluation["summary"]["masked_macro"],
                   "masked_by_fraction": evaluation["summary"]["masked_by_fraction"],
                   "masked_macro_delta": {k: evaluation["summary"]["masked_macro"][k] - baseline["summary"]["masked_macro"][k] for k in baseline["summary"]["masked_macro"]},
                   "official_baseline_seen_before_design": True, "blind_holdout_claim": False,
                   "official_unseen_used_for_training": False, "official_unseen_used_for_selection": False}
        _json_atomic(output / "results.json", results)
        runtime.update(status="TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED", finished_at=_now())
        _json_atomic(output / "runtime.json", runtime)
        return 0
    except BaseException:
        runtime.update(status="FAILED_NO_RETRY", finished_at=_now(), traceback=traceback.format_exc(), child_pid=None)
        _json_atomic(output / "runtime.json", runtime)
        raise


def check_gate(args: argparse.Namespace) -> int:
    baseline = _baseline()
    diagnostic = _validate_raw_and_receipt(args.diagnostic, baseline)
    gate, smoke = _validate_gate(args.gate, diagnostic, baseline)
    print(json.dumps({"status": "PASS", "device": "cpu/filesystem", "gate": gate["path"].as_posix(), "diagnostic": diagnostic["path"].as_posix(), "smoke": smoke.as_posix(), "lambda_sketch_ref": diagnostic["lambda"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--check-gate", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--gate")
    parser.add_argument("--diagnostic")
    args = parser.parse_args()
    if not args.launch and not args.check_gate:
        print(json.dumps({"status": "INERT", "method": METHOD, "arm": ARM, "would_launch": False}, sort_keys=True))
        return 0
    missing = [name for name in ("gate", "diagnostic") if not getattr(args, name)]
    if missing or (args.launch and not args.output):
        parser.error("--launch/--check-gate require --gate and --diagnostic; --launch also requires --output")
    return check_gate(args) if args.check_gate else launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
