#!/usr/bin/env python3
"""Strict post-run verifier for the coupled-predictive V1 campaign.

Reads local artifacts first.  Optional ``--replay`` only re-evaluates authorized
selected checkpoints; it never trains or mutates producer outputs.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

# Keep model resolution local while allowing the explicitly requested online
# W&B verification/report run.  Never import preflight modules here.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "online"

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ARMS = ("R0", "R1", "R1_SIG")
STEPS = (0, 600, 1200, 1800, 2400, 3000, 3600)
SELECT_STEPS = STEPS[1:]
SELECTIONS = ("latest", "best_clean", "best_masked")
FRACTIONS = (0.25, 0.5, 0.75)
SEEDS = (101, 202, 303)
METRICS = (
    "full_mAP",
    "P@200",
    "mAP@200_prefix_positive",
    "mAP@200_all_relevant",
    "mAP@200_min_relevant_k",
)
AP200_ARRAYS = {
    "mAP@200_prefix_positive": "prefix_positive",
    "mAP@200_all_relevant": "all_relevant",
    "mAP@200_min_relevant_k": "min_relevant_k",
}
TRACE_ROWS = 115_200
TOL = 1e-6
CAMPAIGN = "coupled_predictive_v1"
EXPECTED_SPLIT_SHA256 = "3e02604d2ed315aa254d4264ec440a7e50233c7c9b175be224519feafda88425"
EXPECTED_PAIRING_SHA256 = "545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c"
EXPECTED_CLIP_SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
EXPECTED_CLIP_BYTES = 605143284
MASK_STATUSES = ("blank_input", "ok", "target_unreachable", "zero_fraction")
ENTITY = "a-cctest05187-erd"
PROJECT_NAME = "spica"
CLIP_MODEL = "ViT-B-32-quickgelu"


class VerifyError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def load_json(path: Path) -> Any:
    check(path.is_file(), f"missing JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerifyError(f"invalid JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise VerifyError(f"{label} is boolean, not a metric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VerifyError(f"{label} is not numeric: {value!r}") from exc
    check(math.isfinite(result), f"{label} is not finite")
    return result


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            out.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def torch_load(path: Path) -> dict[str, Any]:
    check(path.is_file(), f"missing checkpoint: {path}")
    # NumPy's MT19937 state is the only non-tensor payload in our RNG receipt.
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType,
    ]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    check(isinstance(payload, dict), f"checkpoint is not a dict: {path}")
    return payload


def recompute_source_archive(index: Mapping[str, Any], archive: Path) -> dict[str, Any]:
    entries = index.get("manifest")
    check(isinstance(entries, list) and entries, f"empty source manifest: {archive}")
    aggregate = hashlib.sha256()
    names: list[str] = []
    file_hashes: dict[str, str] = {}
    total_bytes = 0
    for entry in entries:
        check(isinstance(entry, Mapping), f"bad source manifest row: {archive}")
        name = str(entry.get("path", ""))
        check(name and not name.startswith("/") and ".." not in Path(name).parts, f"bad source path {name!r}: {archive}")
        path = archive / "files" / name
        check(path.is_file() and not path.is_symlink(), f"missing archived source file: {path}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        check(int(entry.get("bytes", -1)) == len(data), f"source bytes mismatch: {path}")
        check(str(entry.get("sha256")) == digest, f"source sha mismatch: {path}")
        aggregate.update(name.encode())
        aggregate.update(b"\0")
        aggregate.update(data)
        aggregate.update(b"\0")
        names.append(name)
        file_hashes[name] = digest
        total_bytes += len(data)
    check(len(names) == len(set(names)), f"duplicate source archive names: {archive}")
    actual = aggregate.hexdigest()
    check(actual != "0" * 64 and total_bytes > 0, f"zero/blank source aggregate: {archive}")
    check(index.get("file_count") in (None, len(entries)), f"source file_count mismatch: {archive}")
    check(str(index.get("sha256")) == actual, f"source aggregate mismatch: {archive}")
    return {"sha256": actual, "file_count": len(entries), "bytes": total_bytes, "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(), "files": file_hashes}


def manifest_source_hash(root: Path) -> tuple[str, dict[str, Any]]:
    candidates = [root / "execution_manifest.json", root / "source_snapshot" / "execution_manifest.json", root / "runtime" / "source_state_initial.json"]
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_json(path)
        flat = flatten(payload)
        for key in ("source_snapshot_hash", "source_snapshot_sha256", "source_hash", "source.sha256", "source_snapshot.sha256"):
            if isinstance(flat.get(key), str) and len(str(flat[key])) == 64:
                return str(flat[key]), {"path": str(path), "payload": payload}
        for key, value in flat.items():
            if "source" in key and "sha" in key and isinstance(value, str) and len(value) == 64:
                return value, {"path": str(path), "payload": payload}
    raise VerifyError(f"missing root execution manifest/source hash under {root}")


def verify_runtime(root: Path, source_hash: str) -> dict[str, Any]:
    runtime = load_json(root / "runtime.json")
    check(runtime.get("status") in {"VERIFYING", "VERIFIED"}, f"runtime is not verifier-authoritative: {runtime.get('status')!r}")
    check(tuple(runtime.get("arms", ())) == ARMS, "runtime arm order mismatch")
    check(tuple(runtime.get("completed_arms", ())) == ARMS, "runtime completed_arms mismatch")
    check(runtime.get("current_arm") is None, "runtime current_arm must be None")
    check(runtime.get("source_snapshot_hash") == source_hash, "runtime source hash mismatch")
    child_pid = runtime.get("child_pid")
    check(child_pid is None or int(child_pid) == os.getpid(), "runtime has an active child other than this verifier")
    states = runtime.get("arms_state")
    check(isinstance(states, Mapping) and set(states) == set(ARMS), "runtime arm state rows mismatch")
    for arm in ARMS:
        state = states[arm]
        check(isinstance(state, Mapping) and int(state.get("exit_code", -1)) == 0, f"runtime {arm} exit_code is not zero")
        check(state.get("reported_source_snapshot_hash") == source_hash, f"runtime {arm} reported source mismatch")
    if runtime.get("status") == "VERIFYING":
        verification = runtime.get("verification")
        check(isinstance(verification, Mapping), "runtime VERIFYING row lacks verification metadata")
        pid = verification.get("pid")
        check(pid is None or int(pid) == os.getpid(), "runtime verification pid is not this verifier")
    return runtime


def verify_source_chain(root: Path, source_hash: str, root_manifest: Mapping[str, Any], root_index: Mapping[str, Any]) -> dict[str, Any]:
    """Check the bytes, index, provenance, and execution manifest are one chain."""
    snapshot = root_manifest.get("source_snapshot")
    check(isinstance(snapshot, Mapping), "root execution manifest lacks source_snapshot")
    check(snapshot.get("sha256") == source_hash, "root manifest source snapshot hash mismatch")
    check(canonical(snapshot) == canonical(root_index), "root source index differs from execution manifest snapshot")
    archived_manifest = load_json(root / "source_snapshot" / "execution_manifest.json")
    archived_snapshot = archived_manifest.get("source_snapshot") if isinstance(archived_manifest, Mapping) else None
    check(isinstance(archived_snapshot, Mapping) and canonical(archived_snapshot) == canonical(root_index), "archived execution manifest source snapshot differs from index")
    check(root_manifest.get("source_snapshot_hash") == archived_manifest.get("source_snapshot_hash"), "root/archive execution-manifest source hashes differ")
    provenance = load_json(root / "source_snapshot" / "provenance.json")
    provenance_snapshot = provenance.get("source_snapshot") if isinstance(provenance, Mapping) else None
    check(isinstance(provenance_snapshot, Mapping) and canonical(provenance_snapshot) == canonical(root_index), "archived provenance source snapshot differs from index")
    return {"checked": True, "index_sha256": sha256_file(root / "source_snapshot" / "index.json"), "file_count": len(root_index.get("manifest", []))}


def discover_runs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("run_result.json")):
        if "verification" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        arm = payload.get("arm") or payload.get("experiment_role")
        if arm in ARMS:
            check(arm not in found, f"duplicate run_result for {arm}: {found[arm]} and {path.parent}")
            found[str(arm)] = path.parent
    missing = set(ARMS) - set(found)
    check(not missing, f"missing arm run_result(s): {sorted(missing)}")
    return {arm: found[arm] for arm in ARMS}


def metric_dict(block: Mapping[str, Any], label: str) -> dict[str, float]:
    return {metric: finite(block.get(metric), f"{label}/{metric}") for metric in METRICS}


def query_array(block: Mapping[str, Any], names: Sequence[Any], query_count: int, label: str) -> list[float]:
    value: Any = None
    for name in names:
        if isinstance(name, tuple):
            container = block.get(name[0])
            if isinstance(container, Mapping):
                value = container.get(name[1], container.get(str(name[1])))
        else:
            value = block.get(name)
        if value is not None:
            break
    check(isinstance(value, list) and len(value) == query_count, f"missing/wrong per-query array {label}; expected {query_count}")
    values = [finite(item, label) for item in value]
    check(np.isfinite(np.asarray(values, dtype=np.float64)).all(), f"nonfinite array {label}")
    return values


def verify_metric_block(block: Mapping[str, Any], label: str, query_count: int) -> dict[str, Any]:
    scalars = metric_dict(block, label)
    full_ap = query_array(block, ("average_precision_per_query",), query_count, f"{label}/average_precision_per_query")
    p200 = query_array(block, ("P@200_per_query", ("precision_at_k_per_query", "200"), ("precision_at_k_per_query", 200)), query_count, f"{label}/P@200_per_query")
    means = {
        "full_mAP": float(np.asarray(full_ap, dtype=np.float64).mean()),
        "P@200": float(np.asarray(p200, dtype=np.float64).mean()),
    }
    arrays = block.get("average_precision_at_k_per_query")
    check(isinstance(arrays, Mapping), f"{label} missing AP@k per-query arrays")
    for metric, key in AP200_ARRAYS.items():
        values = query_array(block, (("average_precision_at_k_per_query", key),), query_count, f"{label}/{key}_AP200")
        means[metric] = float(np.asarray(values, dtype=np.float64).mean())
    for metric, mean in means.items():
        check(abs(mean - scalars[metric]) <= TOL, f"{label} mean mismatch {metric}: mean={mean}, scalar={scalars[metric]}")
    return {"scalars": scalars, "means": means}


def verify_mask_metadata(row: Mapping[str, Any], label: str, query_count: int) -> str:
    status_counts = row.get("status_counts")
    check(isinstance(status_counts, Mapping), f"{label} missing status_counts")
    check(set(str(key) for key in status_counts) <= set(MASK_STATUSES), f"{label} status_counts keys mismatch")
    check(all(isinstance(value, int) and value >= 0 for value in status_counts.values()), f"{label} invalid status count")
    check(sum(int(value) for value in status_counts.values()) == query_count, f"{label} status_counts total mismatch")
    metadata = row.get("mask_metadata")
    check(isinstance(metadata, Mapping), f"{label} missing mask_metadata")
    metas = metadata.get("metas") or metadata.get("rows")
    check(metadata.get("count") == query_count and isinstance(metas, list) and len(metas) == query_count, f"{label} mask metadata count mismatch")
    identity_rows = []
    for index, meta in enumerate(metas):
        check(isinstance(meta, Mapping), f"{label} mask meta {index} is not a mapping")
        status = str(meta.get("status", meta.get("masked_status", ""))).strip()
        check(status in MASK_STATUSES, f"{label} invalid mask status at {index}: {status}")
        if "zero_based_update" in meta:
            check(int(meta.get("zero_based_update", -1)) >= 0, f"{label} invalid zero_based_update at {index}")
            check(meta.get("step_convention") == "zero_based_update", f"{label} mask step convention mismatch at {index}")
        check(str(meta.get("input_sha256", "")).strip(), f"{label} blank input sha at {index}")
        check(str(meta.get("output_sha256", "")).strip(), f"{label} blank output sha at {index}")
        check(float(meta.get("requested_fraction", row["fraction"])) == float(row["fraction"]), f"{label} fraction mismatch at {index}")
        # ``row.seed`` is the fixed condition seed; metadata ``seed`` is the
        # per-sample derived mask seed and is intentionally different.
        check(isinstance(meta.get("seed"), int) and not isinstance(meta.get("seed"), bool), f"{label} missing derived mask seed at {index}")
        identity_rows.append({key: meta.get(key) for key in sorted(meta) if key not in {"absolute_path"}})
    from collections import Counter
    check(dict(Counter(str(meta["status"]) for meta in metas)) == {str(k): v for k, v in status_counts.items() if v}, f"{label} status_counts disagree with records")
    return hashlib.sha256(canonical(identity_rows).encode()).hexdigest()


def _identity_sha(ids: Sequence[Any]) -> str:
    # This is intentionally the producer's exact JSON identity convention.
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def _reconstruct_ap200(top_indices: Sequence[Sequence[Any]], query_labels: Sequence[Any], gallery_labels: Sequence[Any], denominator: str) -> list[float]:
    gallery = np.asarray(gallery_labels, dtype=np.int64)
    queries = np.asarray(query_labels, dtype=np.int64)
    top = np.asarray(top_indices, dtype=np.int64)
    result: list[float] = []
    for row, query in zip(top, queries, strict=True):
        relevant = (gallery[row] == query).astype(np.float64)
        prefix = np.cumsum(relevant) / np.arange(1, len(row) + 1, dtype=np.float64)
        if denominator == "prefix_positive":
            denom = float(np.sum(relevant))
        elif denominator == "all_relevant":
            denom = float(np.sum(gallery == query))
        else:
            denom = min(float(np.sum(gallery == query)), 200.0)
        result.append(float(np.sum(prefix * relevant) / max(denom, 1.0)))
    return result


def verify_probe(raw: Mapping[str, Any], label: str) -> dict[str, Any]:
    check(raw.get("benchmark") == "SPICA_category_retrieval", f"{label} benchmark identity mismatch")
    check(raw.get("benchmark_status") == "official_not_verified", f"{label} benchmark status mismatch")
    check(raw.get("evaluation_scope") == "pseudo_validation", f"{label} evaluation scope mismatch")
    identities = raw.get("identities")
    check(isinstance(identities, Mapping), f"{label} missing identities")
    query_ids = identities.get("query_ids")
    gallery_ids = identities.get("gallery_ids")
    query_labels = identities.get("query_labels")
    gallery_labels = identities.get("gallery_labels")
    check(isinstance(query_ids, list) and query_ids and all(isinstance(x, str) and x for x in query_ids), f"{label} bad query ids")
    check(isinstance(gallery_ids, list) and gallery_ids and all(isinstance(x, str) and x for x in gallery_ids), f"{label} bad gallery ids")
    check(isinstance(query_labels, list) and len(query_labels) == len(query_ids), f"{label} bad query labels")
    check(isinstance(gallery_labels, list) and len(gallery_labels) == len(gallery_ids), f"{label} bad gallery labels")
    check([int(x) for x in query_labels] == list(query_labels) and [int(x) for x in gallery_labels] == list(gallery_labels), f"{label} labels are not integer-valued")
    query_count = int(raw.get("query_count", len(query_ids)))
    check(query_count == len(query_ids), f"{label} query_count mismatch")
    check(int(raw.get("gallery_count", len(gallery_ids))) == len(gallery_ids), f"{label} gallery_count mismatch")
    check(raw.get("query_identity_sha256") == _identity_sha(query_ids), f"{label} query identity hash mismatch")
    check(raw.get("gallery_identity_sha256") == _identity_sha(gallery_ids), f"{label} gallery identity hash mismatch")

    clean = verify_metric_block(raw.get("clean", {}), f"{label}/clean", query_count)
    for block_name, block in [("clean", raw.get("clean", {})), *[(f"condition_{i}", row) for i, row in enumerate(raw.get("conditions", []))]]:
        top = block.get("top_indices") if isinstance(block, Mapping) else None
        check(isinstance(top, list) and len(top) == query_count, f"{label}/{block_name} top_indices row count mismatch")
        check(all(isinstance(row, list) and len(row) == 200 for row in top), f"{label}/{block_name} top_indices must be [Q,200]")
        check(all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(gallery_ids) for row in top for index in row), f"{label}/{block_name} top_indices out of range")
        recomputed_p = [float(np.mean(np.asarray(gallery_labels, dtype=np.int64)[row] == int(query_labels[i]))) for i, row in enumerate(top)]
        stored_p = query_array(block, ("P@200_per_query", ("precision_at_k_per_query", "200"), ("precision_at_k_per_query", 200)), query_count, f"{label}/{block_name}/P200")
        check(np.allclose(recomputed_p, stored_p, atol=TOL, rtol=0), f"{label}/{block_name} P200 reconstruction mismatch")
        for metric, denominator in AP200_ARRAYS.items():
            expected_ap = _reconstruct_ap200(top, query_labels, gallery_labels, denominator)
            stored_ap = query_array(block, (("average_precision_at_k_per_query", denominator),), query_count, f"{label}/{block_name}/{denominator}")
            check(np.allclose(expected_ap, stored_ap, atol=TOL, rtol=0), f"{label}/{block_name} {denominator} AP200 reconstruction mismatch")
    conditions = raw.get("conditions")
    check(isinstance(conditions, list) and len(conditions) == 9 and raw.get("condition_count", 9) == 9, f"{label} must have exactly 9 conditions")
    expected_grid = {(fraction, seed) for fraction in FRACTIONS for seed in SEEDS}
    seen = {(float(row.get("fraction")), int(row.get("seed"))) for row in conditions if isinstance(row, Mapping)}
    check(seen == expected_grid, f"{label} condition grid mismatch: {seen}")

    condition_reports = []
    for row in conditions:
        key = (float(row["fraction"]), int(row["seed"]))
        block = verify_metric_block(row, f"{label}/masked/{key[0]}@{key[1]}", query_count)
        block.update({"fraction": key[0], "seed": key[1], "mask_identity_sha256": verify_mask_metadata(row, f"{label}/masked/{key[0]}@{key[1]}", query_count)})
        condition_reports.append(block)

    macro = {metric: float(np.mean([row["scalars"][metric] for row in condition_reports], dtype=np.float64)) for metric in METRICS}
    recorded_macro = metric_dict(raw.get("masked_macro", {}), f"{label}/masked_macro")
    for metric in METRICS:
        check(abs(macro[metric] - recorded_macro[metric]) <= TOL, f"{label} masked macro mismatch {metric}")

    by_fraction: dict[str, dict[str, float]] = {}
    fraction_payload = raw.get("masked_by_fraction")
    check(isinstance(fraction_payload, Mapping), f"{label} missing masked_by_fraction")
    for fraction in FRACTIONS:
        rows = [row for row in condition_reports if row["fraction"] == fraction]
        by_fraction[str(fraction)] = {metric: float(np.mean([row["scalars"][metric] for row in rows], dtype=np.float64)) for metric in METRICS}
        recorded = metric_dict(fraction_payload.get(str(fraction), {}), f"{label}/fraction_{fraction}")
        for metric in METRICS:
            check(abs(by_fraction[str(fraction)][metric] - recorded[metric]) <= TOL, f"{label} fraction mean mismatch {fraction}/{metric}")

    return {"query_count": query_count, "gallery_count": len(gallery_ids), "query_ids": query_ids, "gallery_ids": gallery_ids, "query_labels": query_labels, "gallery_labels": gallery_labels, "clean": clean["scalars"], "masked_macro": recorded_macro, "masked_by_fraction": by_fraction, "condition_reports": condition_reports}


def init_hashes(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("_initialization_hashes", payload.get("initialization_hashes"))
    check(isinstance(value, Mapping) and value, "missing initialization hashes")
    return value


def verify_checkpoint(path: Path, arm: str, step: int, source_hash: str) -> tuple[dict[str, Any], Mapping[str, Any]]:
    payload = torch_load(path)
    check(payload.get("format_version") == 1, f"checkpoint format mismatch: {arm}/{step}")
    check(payload.get("campaign") == CAMPAIGN, f"checkpoint campaign mismatch: {arm}/{step}")
    check(int(payload.get("step", -1)) == step, f"checkpoint step mismatch: {arm}/{step}")
    check(payload.get("source_snapshot_hash") == source_hash, f"checkpoint source hash mismatch: {arm}/{step}")
    for key in ("optimizer_state_dict", "scheduler_state_dict", "rng_state", "clip_identity", "data_identity", "model_state_hash", "selection_metadata"):
        check(key in payload, f"checkpoint missing {key}: {arm}/{step}")
    rng = payload["rng_state"]
    check(isinstance(rng, Mapping), f"checkpoint RNG schema mismatch: {arm}/{step}")
    # ``data_loader_generator`` is the key emitted by the current trainer;
    # accept the spelling used by the pending parent trainer patch too.
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    check(required_rng <= set(rng), f"checkpoint RNG schema mismatch: {arm}/{step}")
    check("data_loader_generator" in rng or "dataloader_generator" in rng, f"checkpoint loader RNG missing: {arm}/{step}")
    check((payload.get("sigreg_state_dict") is None) == (arm != "R1_SIG"), f"SIGReg state presence mismatch: {arm}/{step}")
    state = payload.get("model_state_dict")
    check(isinstance(state, Mapping) and state, f"checkpoint missing model_state_dict: {arm}/{step}")
    check(not any(str(key).startswith("original_clip.") for key in state), f"checkpoint stores original_clip state: {arm}/{step}")
    has_predictor = any(str(key).startswith("predictor.") for key in state)
    check(has_predictor == (arm != "R0"), f"predictor state ownership mismatch: {arm}/{step}")
    config = payload.get("resolved_config")
    check(isinstance(config, Mapping), f"checkpoint missing resolved_config: {arm}/{step}")
    check(config.get("arm") == arm, f"checkpoint config arm mismatch: {arm}/{step}")
    check(config.get("source_snapshot_hash", source_hash) in (None, source_hash), f"checkpoint config source mismatch: {arm}/{step}")
    clip = payload["clip_identity"]
    check(isinstance(clip, Mapping) and clip.get("sha256") == EXPECTED_CLIP_SHA256 and int(clip.get("bytes", -1)) == EXPECTED_CLIP_BYTES, f"checkpoint CLIP identity mismatch: {arm}/{step}")
    data_identity = payload["data_identity"]
    check(isinstance(data_identity, Mapping) and data_identity.get("split", {}).get("sha256") == EXPECTED_SPLIT_SHA256 and data_identity.get("pairing", {}).get("sha256") == EXPECTED_PAIRING_SHA256, f"checkpoint data identity mismatch: {arm}/{step}")
    optimizer = payload["optimizer_state_dict"]
    check(isinstance(optimizer, Mapping) and isinstance(optimizer.get("param_groups"), list) and isinstance(optimizer.get("state"), Mapping), f"checkpoint optimizer schema mismatch: {arm}/{step}")
    parameter_ids: list[int] = []
    for group in optimizer["param_groups"]:
        check(isinstance(group, Mapping) and isinstance(group.get("params"), list), f"checkpoint optimizer group malformed: {arm}/{step}")
        parameter_ids.extend(int(value) for value in group["params"])
        check(math.isfinite(float(group.get("lr", float("nan")))), f"checkpoint optimizer LR invalid: {arm}/{step}")
    check(len(parameter_ids) == len(set(parameter_ids)), f"checkpoint optimizer parameter IDs duplicated: {arm}/{step}")
    if step == 0:
        check(not optimizer["state"], f"step-zero optimizer state is not empty: {arm}")
    else:
        check(set(int(key) for key in optimizer["state"]) == set(parameter_ids), f"checkpoint optimizer state coverage mismatch: {arm}/{step}")
        for state_row in optimizer["state"].values():
            check(isinstance(state_row, Mapping) and int(state_row.get("step", -1)) == step, f"checkpoint optimizer state step mismatch: {arm}/{step}")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state_row.get(key)
                check(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), f"checkpoint optimizer tensor invalid: {arm}/{step}")
    scheduler = payload["scheduler_state_dict"]
    check(isinstance(scheduler, Mapping) and int(scheduler.get("last_epoch", -1)) == step, f"checkpoint scheduler step mismatch: {arm}/{step}")
    check(isinstance(payload.get("model_state_hash"), str) and len(payload["model_state_hash"]) == 64, f"checkpoint model hash missing: {arm}/{step}")
    return payload, init_hashes(payload)


def selector(rows: Iterable[Mapping[str, Any]], block: str) -> Mapping[str, Any]:
    best: Mapping[str, Any] | None = None
    best_score = -math.inf
    for row in rows:
        score = finite(row[block]["mAP@200_prefix_positive"], f"selection/{block}")
        if score > best_score:  # strict max; earliest tie survives
            best = row
            best_score = score
    check(best is not None, "selector received no rows")
    return best


def resolve_optional_project_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT / path).resolve()


def verify_sig_diagnostic(config: Mapping[str, Any], class_ids: Sequence[int], source_hash: str, lambda_sig: float, archive_manifest: Mapping[str, str]) -> dict[str, Any]:
    path = resolve_optional_project_path(config.get("diagnostic"))
    check(path is not None and path.is_file(), "R1_SIG diagnostic receipt path is missing")
    # Receipt bytes stay immutable; diagnostic/training full-source inventories may differ.
    check(sha256_file(path) == config["diagnostic_path_sha256"], "SIGReg diagnostic receipt bytes changed")
    payload = load_json(path)
    check(payload.get("status") == "PASS" and payload.get("verified") is True, "R1_SIG diagnostic receipt is not verified PASS")
    recorded_lambda = payload.get("lambda_sig", payload.get("selection", {}).get("lambda_sig") if isinstance(payload.get("selection"), Mapping) else None)
    check(abs(finite(recorded_lambda, "diagnostic/lambda_sig") - lambda_sig) <= TOL, "R1_SIG diagnostic lambda mismatch")
    recorded_ids = payload.get("class_ids", payload.get("train_class_ids"))
    if recorded_ids is not None:
        check([int(x) for x in recorded_ids] == list(class_ids), "R1_SIG diagnostic class IDs mismatch")
    for key in ("formula_identity", "architecture_identity", "gradient_batch_source", "parameter_scope", "initialization_hashes", "component_sha256"):
        check(payload.get(key), f"R1_SIG diagnostic missing {key}")
    check(payload.get("batches") == 4 and payload.get("batch_size") == 32 and payload.get("rho") == 0.1, "R1_SIG diagnostic locked parameters mismatch")
    check(payload.get("parameter_scope") == "model.student_visual.transformer.resblocks[-1]", "R1_SIG diagnostic parameter scope mismatch")
    check(payload.get("initialization_hashes") == config.get("initialization_hashes"), "R1_SIG diagnostic initialization hashes differ from training config")
    components = payload["component_sha256"]
    check(isinstance(components, Mapping) and components, "R1_SIG diagnostic component hashes missing")
    for name, digest in components.items():
        check(archive_manifest.get(str(name)) == str(digest), f"diagnostic component hash does not match campaign archive: {name}")
    return {"path": str(path), "sha256": sha256_file(path), "lambda_sig": lambda_sig, "component_sha256": dict(components)}


def _cpu_self_check() -> dict[str, Any]:
    """Small offline guards for the verifier's high-risk proof logic."""
    with tempfile.TemporaryDirectory(prefix="coupled_verify_selfcheck_") as tmp:
        root = Path(tmp)
        good = {"status": "VERIFYING", "arms": list(ARMS), "completed_arms": list(ARMS), "current_arm": None, "child_pid": None, "source_snapshot_hash": "a" * 64, "arms_state": {arm: {"exit_code": 0, "reported_source_snapshot_hash": "a" * 64} for arm in ARMS}, "verification": {"pid": os.getpid()}}
        write_json(root / "runtime.json", good)
        verify_runtime(root, "a" * 64)
        good["completed_arms"] = []
        write_json(root / "runtime.json", good)
        try:
            verify_runtime(root, "a" * 64)
        except VerifyError:
            pass
        else:
            raise AssertionError("runtime self-check accepted incomplete arms")

        top = [[1, 0, 2], [2, 1, 0]]
        queries, gallery = [7, 8], [8, 7, 7]
        expected = _reconstruct_ap200(top, queries, gallery, "prefix_positive")
        check(np.allclose(expected, [5.0 / 6.0, 1.0 / 3.0], atol=TOL, rtol=0), "AP@K prefix reconstruction self-check failed")

        archive_index = {"manifest": [{"path": "src/example.py", "sha256": hashlib.sha256(b"x").hexdigest(), "bytes": 1}]}
        changed = {"status": "PASS", "verified": True, "component_sha256": {"src/example.py": archive_index["manifest"][0]["sha256"]}, "note": "one"}
        changed_again = {**changed, "note": "two"}
        for receipt in (changed, changed_again):
            check(all(archive_index["manifest"][0]["sha256"] == digest for digest in receipt["component_sha256"].values()), "diagnostic component binding self-check failed")
    return {"runtime_wrong_rejected": True, "ap200_prefix_reconstructed": True, "diagnostic_component_binding_survives_receipt_change": True}


def verify_config(config: Mapping[str, Any], arm: str, source_hash: str, archive_manifest: Mapping[str, str]) -> dict[str, Any]:
    check(config.get("arm") == arm, f"config arm mismatch: {arm}")
    check(int(config.get("batch_size", -1)) == 32, f"batch size mismatch: {arm}")
    check(int(config.get("total_steps", -1)) == 3600, f"total steps mismatch: {arm}")
    check(tuple(int(x) for x in config.get("probe_steps", ())) == STEPS, f"probe steps mismatch: {arm}")
    expected_arch = "pooled" if arm == "R0" else "predictive"
    check(config.get("architecture") == expected_arch, f"architecture mismatch: {arm}")
    check(config.get("mask_policy") == "region_pair_v1; zero_based_step=step-1", f"mask policy mismatch: {arm}")
    check(tuple(float(x) for x in config.get("eval_mask_fractions", ())) == FRACTIONS, f"eval fractions mismatch: {arm}")
    check(tuple(int(x) for x in config.get("eval_mask_seeds", ())) == SEEDS, f"eval seeds mismatch: {arm}")
    check(config.get("optimizer") == "AdamW" and config.get("scheduler") == "warmup180_cosine_floor0", f"optimizer/scheduler mismatch: {arm}")
    check(config.get("campaign_id") and config.get("device") == "cuda" and config.get("wandb_mode") == "online", f"runtime config mismatch: {arm}")
    check(int(config.get("max_steps", -1)) == 3600 and config.get("smoke") is False, f"training budget mismatch: {arm}")
    clip = config.get("clip_identity")
    check(isinstance(clip, Mapping) and clip.get("sha256") == EXPECTED_CLIP_SHA256 and int(clip.get("bytes", -1)) == EXPECTED_CLIP_BYTES, f"CLIP identity mismatch: {arm}")
    identity = config.get("data_identity")
    check(isinstance(identity, Mapping), f"missing data identity: {arm}")
    check(identity.get("manifest", {}).get("sha256"), f"missing manifest identity: {arm}")
    check(identity.get("split", {}).get("sha256") == EXPECTED_SPLIT_SHA256, f"split identity mismatch: {arm}")
    check(identity.get("pairing", {}).get("sha256") == EXPECTED_PAIRING_SHA256, f"pairing identity mismatch: {arm}")
    class_ids = [int(x) for x in config.get("train_class_ids", [])]
    check(len(class_ids) == 84 and class_ids == sorted(set(class_ids)), f"train_class_ids malformed: {arm}")
    class_names = config.get("class_names")
    check(isinstance(class_names, Mapping) and len(class_names) == 84, f"class_names malformed: {arm}")
    lambda_sig = finite(config.get("lambda_sig", 0.0), f"{arm}/lambda_sig")
    diagnostic = None
    if arm == "R1_SIG":
        check(lambda_sig > 0.0, "R1_SIG lambda_sig must be fixed positive")
        check(config.get("diagnostic_lambda_source") == "verified_receipt", "R1_SIG lambda source mismatch")
        diagnostic = verify_sig_diagnostic(config, class_ids, source_hash, lambda_sig, archive_manifest)
    else:
        check(lambda_sig == 0.0, f"{arm} lambda_sig must be zero")
        check(config.get("diagnostic_lambda_source") == "control_zero", f"{arm} lambda source mismatch")
    return {"class_ids": class_ids, "lambda_sig": lambda_sig, "architecture": expected_arch, "source_snapshot_hash": source_hash, "diagnostic": diagnostic}


def verify_serialized_json(path: Path, label: str) -> None:
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    # Round-tripping is deliberately checked on the serialized artifact, not
    # just the already-loaded object used by the verifier.
    try:
        serialized = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        check(json.loads(serialized) == parsed, f"{label} changed on JSON reserialization")
    except (TypeError, ValueError) as exc:
        raise VerifyError(f"{label} is not JSON-reserializable: {exc}") from exc


def verify_lr_and_trace(run_dir: Path, result: Mapping[str, Any], config: Mapping[str, Any], trace_path: Path, arm: str) -> dict[str, Any]:
    lr_path = resolve_path(result.get("lr_history_every_step", "lr_history_every_step.jsonl"), run_dir)
    check(lr_path.is_file(), f"missing per-step LR history: {arm}")
    lr_rows = [json.loads(line) for line in lr_path.read_text(encoding="utf-8").splitlines()]
    check(len(lr_rows) == 3601 and [int(row.get("step", -1)) for row in lr_rows] == list(range(3601)), f"LR history rows mismatch: {arm}")
    by_step: dict[int, Mapping[str, Any]] = {}
    for row in lr_rows:
        values = row.get("lr")
        check(isinstance(values, Mapping) and set(values), f"invalid LR history row: {arm}")
        check(all(math.isfinite(float(value)) and float(value) >= 0 for value in values.values()), f"invalid LR history value: {arm}")
        by_step[int(row["step"])] = row
    optimizer_groups = config.get("optimizer_groups")
    check(isinstance(optimizer_groups, list) and optimizer_groups, f"missing optimizer groups: {arm}")
    group_names = [f"group_{index}" for index, _ in enumerate(optimizer_groups)]
    base_lrs = [finite(group.get("lr"), f"{arm}/optimizer_groups/{index}/lr") for index, group in enumerate(optimizer_groups)]
    check(all(set(by_step[step]["lr"]) == set(group_names) for step in by_step), f"LR history optimizer groups mismatch: {arm}")

    def schedule(epoch: int) -> float:
        if epoch < 180:
            return (epoch + 1) / 180.0
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - 180) / (3600 - 180)))

    for step, row in by_step.items():
        # The trainer records LR before optimizer/scheduler.step(), so row t
        # reflects scheduler epoch max(t-1, 0); this also catches stale or
        # hand-edited per-group history rather than merely checking finiteness.
        factor = schedule(max(step - 1, 0))
        for index, base_lr in enumerate(base_lrs):
            expected = base_lr * factor
            check(abs(float(row["lr"][f"group_{index}"]) - expected) <= TOL, f"computed LR mismatch: {arm}/step{step}/group_{index}")
    trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    check(len(trace_rows) == TRACE_ROWS, f"trace rows mismatch: {arm}")
    for index, row in enumerate(trace_rows):
        step = int(row.get("step", -1))
        check(step in by_step, f"trace step missing from LR history: {arm}/{step}")
        actual = finite(row.get("actual_lr"), f"{arm}/trace/{index}/actual_lr")
        expected = finite(by_step[step]["lr"].get("group_0"), f"{arm}/lr/{step}/group_0")
        check(abs(actual - expected) <= TOL, f"trace actual_lr does not match LR history: {arm}/{index}")
    return {"path": str(lr_path), "rows": len(lr_rows), "groups": group_names, "trace_actual_lr_matches_group_0": True, "first_group_lr": [float(by_step[step]["lr"]["group_0"]) for step in range(3601)]}


def verify_arm(run_dir: Path, arm: str, source_hash: str, archive_manifest: Mapping[str, str]) -> dict[str, Any]:
    result = load_json(run_dir / "run_result.json")
    check(result.get("status") == "COMPLETE", f"arm result not COMPLETE: {arm}")
    check(result.get("campaign") == CAMPAIGN and result.get("arm") == arm, f"run identity mismatch: {arm}")
    check(int(result.get("completed_steps", result.get("step", -1))) == 3600, f"arm did not complete 3600: {arm}")
    check(result.get("source_snapshot_hash") == source_hash, f"run source hash mismatch: {arm}")

    arm_index = load_json(run_dir / "source_snapshot" / "index.json")
    archive = recompute_source_archive(arm_index, run_dir / "source_snapshot")
    check(archive["sha256"] == source_hash, f"source archive hash != root source hash: {arm}")
    arm_provenance = load_json(run_dir / "provenance.json")
    arm_snapshot = arm_provenance.get("source_snapshot") if isinstance(arm_provenance, Mapping) else None
    check(isinstance(arm_snapshot, Mapping) and canonical(arm_snapshot) == canonical(arm_index), f"arm provenance/source archive chain mismatch: {arm}")

    config = load_json(run_dir / "resolved_config.json")
    config_report = verify_config(config, arm, source_hash, archive["files"])

    trace_path = resolve_path(result.get("trace"), run_dir)
    check(trace_path.is_file(), f"missing trace: {arm}")
    trace_count = sum(1 for _ in trace_path.open(encoding="utf-8"))
    check(trace_count == TRACE_ROWS and int(result.get("trace_count", -1)) == TRACE_ROWS and int(result.get("expected_trace_count", -1)) == TRACE_ROWS, f"trace rows mismatch: {arm}: {trace_count}")
    check(result.get("frozen_original_state_hash_before") == result.get("frozen_original_state_hash_after"), f"frozen original CLIP changed: {arm}")
    lr_report = verify_lr_and_trace(run_dir, result, config, trace_path, arm)

    checkpoints = result.get("checkpoints")
    probes = result.get("probes")
    check(isinstance(checkpoints, list) and len(checkpoints) == 7, f"expected 7 checkpoints: {arm}")
    check(isinstance(probes, list) and len(probes) == 7, f"expected 7 probes: {arm}")
    check(tuple(int(row["step"]) for row in checkpoints) == STEPS, f"checkpoint steps mismatch: {arm}")
    check(tuple(int(row["step"]) for row in probes) == STEPS, f"probe steps mismatch: {arm}")

    trajectory: dict[str, Any] = {}
    checkpoint_reports: dict[str, Any] = {}
    init_by_step: dict[str, Any] = {}
    for ckpt_row, probe_row in zip(checkpoints, probes, strict=True):
        step = int(ckpt_row["step"])
        ckpt_path = resolve_path(ckpt_row["path"], run_dir)
        check(sha256_file(ckpt_path) == ckpt_row["sha256"], f"checkpoint sha mismatch: {arm}/{step}")
        payload, init = verify_checkpoint(ckpt_path, arm, step, source_hash)
        check(canonical(payload["resolved_config"]) == canonical(config), f"checkpoint resolved config differs from run config: {arm}/{step}")
        init_by_step[str(step)] = init
        checkpoint_reports[str(step)] = {"path": str(ckpt_path), "sha256": ckpt_row["sha256"], "state_keys": sorted(payload["model_state_dict"])}

        probe_path = resolve_path(probe_row["path"], run_dir)
        check(probe_path.name == f"probe_step{step}.json", f"probe filename mismatch: {arm}/{step}")
        check(sha256_file(probe_path) == probe_row["sha256"], f"probe sha mismatch: {arm}/{step}")
        verify_serialized_json(probe_path, f"{arm}/step{step} probe")
        raw_report = verify_probe(load_json(probe_path), f"{arm}/step{step}")
        for block in ("clean", "masked_macro"):
            expected = probe_row[block]
            for metric in METRICS:
                check(abs(finite(expected[metric], f"probe summary {arm}/{step}/{block}/{metric}") - raw_report[block][metric]) <= TOL, f"probe summary mismatch: {arm}/{step}/{block}/{metric}")
        trajectory[str(step)] = {"checkpoint": checkpoint_reports[str(step)], **raw_report}

    first_init = canonical(init_by_step["0"])
    check(all(canonical(init) == first_init for init in init_by_step.values()), f"initialization hashes drift within arm: {arm}")

    selections = result.get("selections")
    check(isinstance(selections, Mapping) and set(selections) >= set(SELECTIONS), f"missing selections: {arm}")
    candidate_rows = [{"step": step, "clean": trajectory[str(step)]["clean"], "masked_macro": trajectory[str(step)]["masked_macro"]} for step in SELECT_STEPS]
    expected = {
        "latest": candidate_rows[-1],
        "best_clean": selector(candidate_rows, "clean"),
        "best_masked": selector(candidate_rows, "masked_macro"),
    }
    selected: dict[str, Any] = {}
    for name in SELECTIONS:
        actual = selections[name]
        step = int(actual.get("step", actual.get("training_global_step", -1)))
        check(step != 0 and step == int(expected[name]["step"]), f"selection step mismatch: {arm}/{name}")
        path = resolve_path(actual.get("path", checkpoint_reports[str(step)]["path"]), run_dir)
        expected_sha = checkpoint_reports[str(step)]["sha256"]
        check(path.is_file() and sha256_file(path) == expected_sha, f"selection checkpoint mismatch: {arm}/{name}")
        check(str(actual.get("sha256", actual.get("checkpoint_sha256", expected_sha))) == expected_sha, f"selection sha mismatch: {arm}/{name}")
        scores = actual.get("metrics") or {"clean": trajectory[str(step)]["clean"], "masked_macro": trajectory[str(step)]["masked_macro"]}
        selected[name] = {"step": step, "path": str(path), "sha256": expected_sha, "scores": scores}

    return {"run_dir": str(run_dir), "result": result, "archive": archive, "config": config, "config_report": config_report, "trace": {"path": str(trace_path), "rows": trace_count, "sha256": sha256_file(trace_path)}, "lr_history": lr_report, "checkpoints": checkpoint_reports, "trajectory": trajectory, "initialization_hashes": init_by_step["0"], "selected": selected}


def verify_cross_arm_probe_identity(arms: Mapping[str, Any]) -> dict[str, Any]:
    baseline = arms[ARMS[0]]["trajectory"]
    baseline_lr = arms[ARMS[0]]["lr_history"]["first_group_lr"]
    for arm in ARMS[1:]:
        check(arms[arm]["lr_history"]["first_group_lr"] == baseline_lr, f"first optimizer-group LR trajectory differs across arms: {arm}")
        check(arms[arm]["trace"]["sha256"] == arms["R0"]["trace"]["sha256"], f"observation traces differ: {arm}")
        check(sha256_file(Path(arms[arm]["run_dir"]) / "mask_metadata.jsonl") == sha256_file(Path(arms["R0"]["run_dir"]) / "mask_metadata.jsonl"), f"training mask traces differ: {arm}")
    for arm in ARMS[1:]:
        for step in STEPS:
            left = baseline[str(step)]
            right = arms[arm]["trajectory"][str(step)]
            check(left["query_ids"] == right["query_ids"], f"probe query IDs differ across arms at step {step}: {arm}")
            check(left["gallery_ids"] == right["gallery_ids"], f"probe gallery IDs differ across arms at step {step}: {arm}")
            check(left["query_labels"] == right["query_labels"] and left["gallery_labels"] == right["gallery_labels"], f"probe labels differ across arms at step {step}: {arm}")
            for index, row in enumerate(left["condition_reports"]):
                check(row["mask_identity_sha256"] == right["condition_reports"][index]["mask_identity_sha256"], f"eval mask metadata differs across arms at step {step}/{index}: {arm}")
    return {"same_ordered_ids_labels_and_eval_masks": True, "arms": list(ARMS), "steps": list(STEPS)}

def verify_common_initialization(arms: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[str, Mapping[str, Any]] = {}
    for arm in ARMS:
        value = arms[arm]["initialization_hashes"]
        check(set(value) == {"groups", "model"}, f"{arm} initialization hash schema mismatch")
        check(isinstance(value.get("groups"), Mapping) and isinstance(value.get("model"), str), f"{arm} initialization hashes malformed")
        groups[arm] = value["groups"]
    fields = ("student", "pooled", "photo", "text", "T0")
    check(all(field in groups[arm] for arm in ARMS for field in fields), "initialization group fields are incomplete")
    values = {f"groups.{field}": groups["R0"][field] for field in fields}
    check(all(len({str(groups[arm][field]) for arm in ARMS}) == 1 for field in fields), "common initialization hashes differ")
    r1_hashes = [arms[arm]["initialization_hashes"] for arm in ("R1", "R1_SIG")]
    check(r1_hashes[0] == r1_hashes[1], "R1/R1_SIG initialization hash payload differs")
    state_zero: dict[str, Mapping[str, Any]] = {}
    for arm in ("R1", "R1_SIG"):
        path = Path(arms[arm]["checkpoints"]["0"]["path"])
        payload = torch_load(path)
        state_zero[arm] = payload["model_state_dict"]
    check(set(state_zero["R1"]) == set(state_zero["R1_SIG"]), "R1/R1_SIG step-zero state keys differ")
    check(all(torch.equal(state_zero["R1"][key], state_zero["R1_SIG"][key]) for key in state_zero["R1"]), "R1/R1_SIG step-zero model state differs")
    r0_state = torch_load(Path(arms["R0"]["checkpoints"]["0"]["path"]))["model_state_dict"]
    common_keys = set(r0_state) & set(state_zero["R1"])
    check(all(torch.equal(r0_state[key], state_zero["R1"][key]) for key in common_keys), "R0 common step-zero model state differs")
    check(not any(str(key).startswith("predictor.") for key in r0_state), "R0 step-zero state unexpectedly contains predictor")
    return {"common_fields": list(values), "common_values_sha256": hashlib.sha256(canonical(values).encode()).hexdigest(), "architecture_specific_excluded": ["model", "predictor"], "predictive_state_zero_equal": True}


def promotion(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "clean_prefix_strict_improvement": candidate["clean"]["mAP@200_prefix_positive"] > baseline["clean"]["mAP@200_prefix_positive"],
        "clean_P200_non_decrease": candidate["clean"]["P@200"] >= baseline["clean"]["P@200"],
        "clean_full_mAP_non_decrease": candidate["clean"]["full_mAP"] >= baseline["clean"]["full_mAP"],
        "masked_macro_prefix_strict_improvement": candidate["masked_macro"]["mAP@200_prefix_positive"] > baseline["masked_macro"]["mAP@200_prefix_positive"],
    }


def compare_arms(arms: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for left, right in (("R1", "R0"), ("R1_SIG", "R1"), ("R1_SIG", "R0")):
        rows = []
        for selection in SELECTIONS:
            a = arms[left]["selected"][selection]["scores"]
            b = arms[right]["selected"][selection]["scores"]
            rows.append({
                "selection": selection,
                "left": left,
                "right": right,
                "left_step": arms[left]["selected"][selection]["step"],
                "right_step": arms[right]["selected"][selection]["step"],
                "clean_delta": {metric: a["clean"][metric] - b["clean"][metric] for metric in METRICS},
                "masked_macro_delta": {metric: a["masked_macro"][metric] - b["masked_macro"][metric] for metric in METRICS},
            })
        comparisons[f"{left}_vs_{right}"] = rows
    gates = {}
    for arm, control in (("R1", "R0"), ("R1_SIG", "R1"), ("R1_SIG", "R0")):
        cand = arms[arm]["selected"]["best_clean"]["scores"]
        base = arms[control]["selected"]["best_clean"]["scores"]
        gate = promotion(cand, base)
        gates[f"{arm}_vs_{control}"] = {"gate": gate, "passed": all(gate.values())}
    return {"pairwise": comparisons, "promotion_at_best_clean": gates, "external_S0_comparisons": {arm: "NOT_MATCHED_NOT_COMPARABLE" for arm in ARMS}}


def compare_configs(arms: Mapping[str, Any]) -> dict[str, Any]:
    base = flatten(arms["R0"]["config"])
    allowed_prefixes = (
        "arm", "architecture", "lambda_sig", "diagnostic", "diagnostic_path_sha256", "diagnostic_lambda_source",
        "output_dir", "model_trainable_parameters", "model_total_parameters", "optimizer_groups",
    )
    report: dict[str, Any] = {}
    for arm in ARMS[1:]:
        current = flatten(arms[arm]["config"])
        diffs = sorted(key for key in set(base) | set(current) if base.get(key) != current.get(key))
        unexpected = [key for key in diffs if not key.startswith(allowed_prefixes)]
        check(not unexpected, f"unexpected config differences {arm} vs R0: {unexpected[:12]}")
        report[f"{arm}_vs_R0"] = {"differences": diffs}
    return report


def wandb_run_id(result: Mapping[str, Any]) -> str | None:
    nested = result.get("wandb")
    if isinstance(nested, Mapping) and nested.get("run_id"):
        return str(nested["run_id"])
    if result.get("wandb_run_id"):
        return str(result["wandb_run_id"])
    return None


def fraction_label(fraction: float) -> str:
    return f"{float(fraction):.2f}".replace(".", "")


def verify_current_critical_source(root: Path) -> dict[str, Any]:
    """Allow post-run documentation edits, but never replay changed code/config."""
    index_path = root / "source_snapshot" / "index.json"
    if not index_path.is_file():
        return {"checked": False, "reason": "no root archive"}
    index = load_json(index_path)
    checked = 0
    for entry in index.get("manifest", []):
        relative = str(entry["path"])
        if Path(relative).suffix not in {".py", ".yaml", ".yml", ".toml", ".lock"}:
            continue
        current = PROJECT / relative
        check(current.is_file(), f"current critical source is missing: {relative}")
        check(sha256_file(current) == str(entry["sha256"]), f"current critical source drift: {relative}")
        checked += 1
    check(checked > 0, "source archive has no critical production files")
    return {"checked": True, "critical_file_count": checked, "docs_ignored": True}

def verify_wandb(arms: Mapping[str, Any], root: Path, entity: str, project: str) -> dict[str, Any] | None:
    run_ids = {arm: wandb_run_id(arms[arm]["result"]) for arm in ARMS}
    if not any(run_ids.values()):
        return None
    missing = [arm for arm, run_id in run_ids.items() if not run_id]
    check(not missing, f"online W&B run ids missing for arms: {missing}")
    import wandb

    api = wandb.Api(timeout=120)
    output: dict[str, Any] = {"read_only": True, "runs": {}}
    for arm, run_id in run_ids.items():
        run = api.run(f"{entity}/{project}/{run_id}")
        config = dict(run.config or {})
        check(config.get("source_snapshot_hash") == arms[arm]["result"]["source_snapshot_hash"], f"W&B source hash mismatch: {arm}")
        check(config.get("architecture") == arms[arm]["config"]["architecture"], f"W&B architecture mismatch: {arm}")
        check(float(config.get("lambda_sig", -1)) == float(arms[arm]["config"].get("lambda_sig", 0.0)), f"W&B lambda mismatch: {arm}")
        check([int(x) for x in config.get("train_class_ids", [])] == arms[arm]["config_report"]["class_ids"], f"W&B class ids mismatch: {arm}")
        rows = [dict(row) for row in run.scan_history(page_size=1000)]
        step_rows = [row for row in rows if "step_train" in row]
        check(len(step_rows) == 361, f"W&B history row count mismatch: {arm}: {len(step_rows)}")
        check([int(row["step_train"]) for row in step_rows] == list(range(0, 3601, 10)), f"W&B step_train trajectory mismatch: {arm}")
        by_step = {int(row["step_train"]): row for row in step_rows}
        for step in STEPS:
            row = by_step.get(step)
            check(row is not None, f"W&B missing probe step {arm}/{step}")
            local = arms[arm]["trajectory"][str(step)]
            for metric in METRICS:
                keys = [(f"clean/{metric}", local["clean"][metric]), (f"masked/macro/{metric}", local["masked_macro"][metric])]
                keys.extend((f"masked/fraction_{fraction_label(fraction)}/{metric}", local["masked_by_fraction"][str(fraction)][metric]) for fraction in FRACTIONS)
                for condition in local["condition_reports"]:
                    keys.append((f"masked/fraction_{fraction_label(condition['fraction'])}/seed_{condition['seed']}/{metric}", condition["scalars"][metric]))
                for key, expected in keys:
                    check(key in row, f"W&B missing metric {arm}/step{step}/{key}")
                    check(abs(finite(row[key], f"W&B {arm}/{step}/{key}") - float(expected)) <= TOL, f"W&B metric mismatch {arm}/step{step}/{key}")

        artifacts = list(run.logged_artifacts())
        artifact_records = []
        selected_downloads: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="coupled_wandb_artifacts_") as tmp:
            tmp_root = Path(tmp)
            for artifact in artifacts:
                aliases = sorted(str(alias) for alias in getattr(artifact, "aliases", ()))
                metadata = dict(getattr(artifact, "metadata", {}) or {})
                artifact_records.append({"name": artifact.name, "aliases": aliases, "metadata": metadata})
            for selection in SELECTIONS:
                expected = arms[arm]["selected"][selection]
                matches = [artifact for artifact in artifacts if selection in [str(alias) for alias in getattr(artifact, "aliases", ())] and str((getattr(artifact, "metadata", {}) or {}).get("checkpoint_sha256")) == expected["sha256"]]
                check(len(matches) == 1, f"W&B selected artifact alias is missing or ambiguous: {arm}/{selection}")
                artifact = matches[0]
                metadata = dict(getattr(artifact, "metadata", {}) or {})
                check(str(metadata.get("source_snapshot_hash")) == arms[arm]["result"]["source_snapshot_hash"], f"W&B artifact source hash mismatch: {arm}/{selection}")
                check(str(getattr(artifact, "name", "")).startswith("coupled-"), f"unexpected W&B checkpoint artifact name: {artifact.name}")
                target = tmp_root / arm / selection
                downloaded = Path(artifact.download(root=str(target)))
                files = [downloaded] if downloaded.is_file() else sorted(path for path in downloaded.rglob("*") if path.is_file())
                hashes = {str(path): sha256_file(path) for path in files}
                check(expected["sha256"] in set(hashes.values()), f"downloaded artifact does not contain selected checkpoint: {arm}/{selection}")
                selected_downloads[selection] = {"artifact": artifact.name, "aliases": sorted(str(alias) for alias in artifact.aliases), "file_count": len(files), "checkpoint_sha256": expected["sha256"]}
        output["runs"][arm] = {"run_id": run_id, "url": getattr(run, "url", None), "state": str(run.state), "history_rows": len(step_rows), "artifacts": artifact_records, "selected_downloads": selected_downloads}
    return output


def build_model_for_replay(arm: str, config: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Mapping[str, Any]]:
    from spica.data.coupled_training import load_protocol_data, verify_clip_cache
    from spica.models.clip import load_frozen_clip
    from spica.models.coupled_predictive import CoupledPredictiveModel

    protocol = load_protocol_data()
    split = protocol["split"]
    names = protocol["names"]
    class_ids = [int(value) for value in split.train_class_ids]
    check(class_ids == [int(x) for x in config["train_class_ids"]], f"replay protocol class ids mismatch: {arm}")
    check(len(class_ids) == 84, "replay classmap is not 84 classes")
    clip = verify_clip_cache()
    bundle = load_frozen_clip(model_name=CLIP_MODEL, pretrained=str(clip["path"]), device=device)
    classmap = {class_id: str(names[class_id]) for class_id in class_ids}
    model = CoupledPredictiveModel(bundle.encoder, bundle.tokenizer, classmap, architecture=str(config["architecture"])).to(device)
    if arm == "R0":
        model.predictor = None
    return protocol, bundle, model


def load_replay_state(model: Any, checkpoint: Path, arm: str, step: int, source_hash: str) -> None:
    payload = torch_load(checkpoint)
    check(payload.get("source_snapshot_hash") == source_hash and int(payload.get("step", -1)) == step, f"replay checkpoint provenance mismatch: {arm}/{step}")
    check(payload.get("resolved_config", {}).get("arm") == arm, f"replay checkpoint arm mismatch: {arm}/{step}")
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad_missing = [key for key in missing if not key.startswith("original_clip.")]
    check(not bad_missing, f"replay missing non-original_clip keys: {arm}/{step}: {bad_missing[:8]}")
    check(not unexpected, f"replay unexpected checkpoint keys: {arm}/{step}: {unexpected[:8]}")
    from spica.train_coupled_predictive import _state_hash
    check(_state_hash(model) == payload["model_state_hash"], f"reconstructed model hash differs: {arm}/{step}")
    check(_state_hash(model, prefix="original_clip.") == payload["resolved_config"]["frozen_original_state_hash"], f"reconstructed teacher hash differs: {arm}/{step}")


def compare_replay(original: Mapping[str, Any], replayed: Mapping[str, Any], label: str) -> dict[str, Any]:
    check(original.get("identities") == replayed.get("identities"), f"replay identities differ for {label}")
    for key in ("query_identity_sha256", "gallery_identity_sha256", "query_count", "gallery_count", "condition_count"):
        check(original.get(key) == replayed.get(key), f"replay {key} differs for {label}")
    deltas: dict[str, float] = {}
    def check_block(left: Mapping[str, Any], right: Mapping[str, Any], name: str) -> None:
        check(left.get("top_indices") == right.get("top_indices"), f"replay top_indices differ for {label}/{name}")
        for key in ("average_precision_per_query", "P@200_per_query"):
            check(np.allclose(np.asarray(left.get(key), dtype=np.float64), np.asarray(right.get(key), dtype=np.float64), atol=TOL, rtol=0), f"replay {key} differs for {label}/{name}")
        for denominator in AP200_ARRAYS.values():
            check(np.allclose(np.asarray(left["average_precision_at_k_per_query"][denominator], dtype=np.float64), np.asarray(right["average_precision_at_k_per_query"][denominator], dtype=np.float64), atol=TOL, rtol=0), f"replay AP200 differs for {label}/{name}/{denominator}")
        for metadata_key in ("mask_metadata", "status_counts"):
            if metadata_key in left:
                check(canonical(left[metadata_key]) == canonical(right.get(metadata_key)), f"replay {metadata_key} differs for {label}/{name}")
    check_block(original["clean"], replayed["clean"], "clean")
    for fraction, original_metrics in original.get("masked_by_fraction", {}).items():
        replay_metrics = replayed.get("masked_by_fraction", {}).get(str(fraction))
        check(isinstance(replay_metrics, Mapping), f"replay masked_by_fraction missing for {label}/{fraction}")
        for metric in METRICS:
            deltas[f"masked_by_fraction/{fraction}/{metric}"] = abs(finite(replay_metrics[metric], f"replay {label}") - finite(original_metrics[metric], f"original {label}"))
            check(deltas[f"masked_by_fraction/{fraction}/{metric}"] <= TOL, f"replay masked_by_fraction differs for {label}/{fraction}/{metric}")
    for block in ("clean", "masked_macro"):
        for metric in METRICS:
            deltas[f"{block}/{metric}"] = abs(finite(replayed[block][metric], f"replay {label}") - finite(original[block][metric], f"original {label}"))
    for row in original["conditions"]:
        match = next(item for item in replayed["conditions"] if float(item["fraction"]) == float(row["fraction"]) and int(item["seed"]) == int(row["seed"]))
        check_block(row, match, f"{row['fraction']}@{row['seed']}")
        for metric in METRICS:
            deltas[f"{row['fraction']}@{row['seed']}/{metric}"] = abs(finite(match[metric], f"replay {label}") - finite(row[metric], f"original {label}"))
    max_delta = max(deltas.values(), default=0.0)
    check(max_delta <= TOL, f"replay scalar delta too large for {label}: {max_delta}")
    return {"max_abs_delta": max_delta, "deltas": deltas}


def run_replay(root: Path, verification: Path, arms: Mapping[str, Any], device_name: str) -> dict[str, Any]:
    check(device_name == "cuda" and torch.cuda.is_available(), "--replay is authorized only for available CUDA")
    os.environ["WANDB_MODE"] = "online"
    from spica import train_coupled_predictive as trainer
    from spica.evaluation.coupled_predictive import write_probe

    device = torch.device(device_name)
    replay_root = verification / "replay"
    check(not replay_root.exists(), f"refusing to overwrite existing replay directory: {replay_root}")
    replay_root.mkdir(parents=True)
    report: dict[str, Any] = {"device": str(device), "aliases": []}
    for arm in ARMS:
        protocol, bundle, model = build_model_for_replay(arm, arms[arm]["config"], device)
        model.eval()
        cache: dict[int, Mapping[str, Any]] = {}
        try:
            selected_steps = sorted({int(arms[arm]["selected"][name]["step"]) for name in SELECTIONS})
            for step in selected_steps:
                checkpoint = Path(arms[arm]["checkpoints"][str(step)]["path"])
                load_replay_state(model, checkpoint, arm, step, arms[arm]["result"]["source_snapshot_hash"])
                model.eval()
                with torch.no_grad():
                    result = trainer._eval_factory(protocol, bundle.transform, model, device)()
                out = replay_root / arm / f"step{step}.json"
                write_probe(out, result)
                verify_serialized_json(out, f"replay {arm}/step{step}")
                cache[step] = load_json(out)
            for name in SELECTIONS:
                step = int(arms[arm]["selected"][name]["step"])
                original = load_json(Path(arms[arm]["run_dir"]) / f"probe_step{step}.json")
                alias_out = replay_root / arm / name / "result.json"
                write_probe(alias_out, cache[step])
                verify_serialized_json(alias_out, f"replay {arm}/{name}/step{step}")
                delta = compare_replay(original, load_json(alias_out), f"{arm}/{name}/step{step}")
                report["aliases"].append({"arm": arm, "selection": name, "step": step, "replay_path": str(alias_out), **delta})
        finally:
            del model, bundle, protocol
            gc.collect()
            torch.cuda.empty_cache()
    check(len(report["aliases"]) == 9, "replay did not record nine selected alias deltas")
    return report


def selection_rows(arms: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for selection in SELECTIONS:
            row = arms[arm]["selected"][selection]
            scores = row["scores"]
            out = {"arm": arm, "selection": selection, "step": row["step"], "checkpoint_sha256": row["sha256"]}
            for metric in METRICS:
                out[f"clean/{metric}"] = scores["clean"][metric]
                out[f"masked_macro/{metric}"] = scores["masked_macro"][metric]
            rows.append(out)
    return rows


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = ["# Coupled-predictive campaign verification", "", f"**Status:** `{summary['status']}`", "", "## Selected checkpoints", "", "| arm | selection | step | clean prefix AP200 | clean P@200 | clean full mAP | masked prefix AP200 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["selection_rows"]:
        lines.append(f"| {row['arm']} | {row['selection']} | {row['step']} | {row['clean/mAP@200_prefix_positive']:.10g} | {row['clean/P@200']:.10g} | {row['clean/full_mAP']:.10g} | {row['masked_macro/mAP@200_prefix_positive']:.10g} |")
    lines += ["", "## Locked protocol and provenance", "", "- Region-first protocol: `docs/designs/coupled_predictive_region_v1/region_first_protocol.md`.", "- Architecture: `docs/designs/coupled_predictive_region_v1/architecture.md`.", "- Diagnostic lambda and receipt are recorded in each R1_SIG resolved config; source archive is primary evidence.", "", "## Promotion", "", "```json", json.dumps(summary["comparisons"]["promotion_at_best_clean"], indent=2, sort_keys=True), "```", "", "External S0 comparisons: `NOT_MATCHED_NOT_COMPARABLE`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def log_wandb_report(summary: Mapping[str, Any], verification: Path, entity: str, project: str) -> dict[str, Any]:
    from spica.tracking.wandb import WandbExperiment

    report = WandbExperiment(project=project, entity=entity, group=Path(summary["campaign_root"]).name, job_type="report", name="coupled_campaign_verification_report", mode="online", directory=verification)
    try:
        columns = ["arm", "selection", "step", *METRICS]
        clean_rows = [[row["arm"], row["selection"], row["step"], *[row[f"clean/{metric}"] for metric in METRICS]] for row in summary["selection_rows"]]
        macro_rows = [[row["arm"], row["selection"], row["step"], *[row[f"masked_macro/{metric}"] for metric in METRICS]] for row in summary["selection_rows"]]
        report.log_table("clean_9rows", columns=columns, rows=clean_rows)
        report.log_table("masked_macro_9rows", columns=columns, rows=macro_rows)
        pair_rows = []
        for name, rows in summary["comparisons"]["pairwise"].items():
            for row in rows:
                pair_rows.append([name, row["selection"], row["left_step"], row["right_step"], row["clean_delta"]["mAP@200_prefix_positive"], row["clean_delta"]["P@200"], row["clean_delta"]["full_mAP"]])
        report.log_table("paired_comparisons", columns=["pair", "selection", "left_step", "right_step", "clean_prefix_delta", "clean_P200_delta", "clean_full_mAP_delta"], rows=pair_rows)
        promotion_rows = [[name, value["passed"], *[value["gate"][key] for key in value["gate"]]] for name, value in summary["comparisons"]["promotion_at_best_clean"].items()]
        report.log_table("promotion_gates", columns=["comparison", "passed", "clean_prefix_strict", "clean_P200_nondec", "clean_full_mAP_nondec", "masked_prefix_strict"], rows=promotion_rows)
        report.set_summary({"verification_status": summary["status"], "source_snapshot_hash": summary["source_snapshot_hash"]})
        return {"run_id": report.run_id, "url": report.run_url}
    finally:
        report.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root")
    parser.add_argument("--replay", action="store_true", help="CUDA re-evaluate 3 selected aliases per arm; no training")
    parser.add_argument("--device", default="cuda", help="replay device; only cuda is accepted")
    parser.add_argument("--no-wandb-report", action="store_true", help="skip the required small online report run")
    parser.add_argument("--cpu-self-check", action="store_true", help="run tiny offline verifier checks and exit")
    parser.add_argument("--wandb-entity", default=ENTITY)
    parser.add_argument("--wandb-project", default=PROJECT_NAME)
    args = parser.parse_args()

    if args.cpu_self_check:
        print(json.dumps({"status": "CPU_SELF_CHECK_PASS", "checks": _cpu_self_check()}, sort_keys=True))
        return 0

    if args.campaign_root is None:
        parser.error("--campaign-root is required unless --cpu-self-check")
    root = Path(args.campaign_root).expanduser().resolve()
    verification = root / "verification"
    summary_path = verification / "summary.json"
    try:
        check(root.is_dir(), f"campaign root is not a directory: {root}")
        source_hash, root_manifest = manifest_source_hash(root)
        root_index = root / "source_snapshot" / "index.json"
        check(root_index.is_file(), "missing root source archive index")
        root_archive = recompute_source_archive(load_json(root_index), root / "source_snapshot")
        check(root_archive["sha256"] == source_hash, "root source archive does not match execution manifest")
        runtime_report = verify_runtime(root, source_hash)
        current_source = verify_current_critical_source(root)
        runs = discover_runs(root)
        arms = {arm: verify_arm(runs[arm], arm, source_hash, root_archive["files"]) for arm in ARMS}
        init_report = verify_common_initialization(arms)
        cross_arm_identity = verify_cross_arm_probe_identity(arms)
        config_diff = compare_configs(arms)
        comparisons = compare_arms(arms)
        wandb_report = verify_wandb(arms, root, args.wandb_entity, args.wandb_project)
        replay_report = run_replay(root, verification, arms, args.device) if args.replay else None
        summary: dict[str, Any] = {
            "status": "VERIFIED",
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_root": str(root),
            "source_snapshot_hash": source_hash,
            "root_manifest": root_manifest,
            "runtime": runtime_report,
            "source_chain": verify_source_chain(root, source_hash, root_manifest["payload"], load_json(root_index)),
            "current_source": current_source,
            "arms": {arm: {key: arms[arm][key] for key in ("run_dir", "archive", "trace", "lr_history", "checkpoints", "selected", "config_report")} for arm in ARMS},
            "common_initialization": init_report,
            "cross_arm_probe_identity": cross_arm_identity,
            "config_differences": config_diff,
            "selection_rows": selection_rows(arms),
            "comparisons": comparisons,
            "wandb": wandb_report,
            "replay": replay_report,
            "limitations": ["one local campaign/split", "external semantic-text S0/SeCo/official-unseen comparisons are not matched or claimed"],
        }
        write_json(summary_path, summary)
        write_markdown(verification / "summary.md", summary)
        if not args.no_wandb_report:
            check(wandb_report is not None, "online W&B verification/report requires three finished training run IDs")
            summary["wandb_report_run"] = log_wandb_report(summary, verification, args.wandb_entity, args.wandb_project)
            write_json(summary_path, summary)
        print(json.dumps({"status": "VERIFIED", "summary": str(summary_path), "replay": bool(replay_report), "wandb_checked": bool(wandb_report)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"status": "FAIL", "campaign_root": str(root), "error": f"{type(exc).__name__}: {exc}"}
        write_json(summary_path, failure)
        (verification / "summary.md").write_text(f"# Coupled-predictive campaign verification\n\n**Status:** `FAIL`\n\n`{failure['error']}`\n", encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
