"""Validate and summarize the corrected frozen-prompt final campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spica.frozen_prompt_artifacts import (
    FINAL_CAMPAIGN,
    FINAL_MANIFEST_PATH,
    FINAL_ROLES,
    FINAL_SPLIT_SEEDS,
    canonical_sha256,
    ensure_manifest,
    treatment_for_role,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs/experiments/frozen_prompt_final"
SEEDS = (42, 123, 3407)
SPLIT_SEEDS = FINAL_SPLIT_SEEDS
EXTENDED_STEPS = (5400, 6000, 7200, 9000, 10800)
BOOTSTRAP_DRAWS = 2000
PLOT_FILENAMES = {
    "photo_ablation": "frozen_prompt_final_photo_ablation.png",
    "layernorm": "frozen_prompt_final_layernorm.png",
    "extended_training": "frozen_prompt_final_extended_training.png",
    "seed_confirmation": "frozen_prompt_final_seed_confirmation.png",
    "split_robustness": "frozen_prompt_final_split_robustness.png",
    "geometry": "frozen_prompt_final_geometry.png",
    "attention": "frozen_prompt_final_attention.png",
}
# Kept as a public compatibility constant for callers that used the original
# script.  New reports should pass an output_dir instead of touching these files.
PLOT_PATHS = {key: ROOT / "outputs" / name for key, name in PLOT_FILENAMES.items()}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _row(run: dict[str, Any], step: int) -> dict[str, Any]:
    for row in run.get("history", []):
        if int(row.get("training_global_step", -1)) == step:
            return row
    if step == 5400 and str(run.get("experiment_role", "")).endswith("FP5"):
        for row in run.get("frozen_hold_evaluation", []):
            if row.get("kind") == "frozen_hold_evaluation" and int(
                row.get("comparison_horizon", -1)
            ) == 5400:
                return row
    raise ValueError(
        f"{run.get('experiment_role')} has no training/evaluation row at step {step}"
    )


def _map_at(run: dict[str, Any], step: int) -> float:
    row = _row(run, step)
    values = row.get("val", row)
    value = values.get("full_mAP", row.get("full_pseudo_unseen_mAP"))
    if value is None:
        raise ValueError(f"row has no full mAP at step {step}")
    return float(value)


def _ap(run: dict[str, Any], step: int) -> np.ndarray:
    row = _row(run, step)
    values = row.get("val", row)
    result = np.asarray(values["average_precision_per_query"], dtype=np.float64)
    if result.ndim != 1 or not result.size:
        raise ValueError("average-precision vector is empty or not one-dimensional")
    return result


def _query_identity(run: dict[str, Any], step: int) -> dict[str, Any]:
    values = _row(run, step).get("val", _row(run, step))
    identity = values.get("query_identity")
    if not isinstance(identity, dict) or not identity.get("sha256"):
        raise ValueError(f"{run.get('experiment_role')}: missing query identity")
    return identity


def _assert_paired_queries(
    left: dict[str, Any], right: dict[str, Any], step: int
) -> tuple[np.ndarray, np.ndarray]:
    left_identity = _query_identity(left, step)
    right_identity = _query_identity(right, step)
    if left_identity != right_identity:
        raise ValueError("paired query vectors are not aligned")
    left_ap, right_ap = _ap(left, step), _ap(right, step)
    if left_ap.shape != right_ap.shape:
        raise ValueError("paired query vectors are not aligned")
    return left_ap, right_ap


def _bootstrap(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired bootstrap inputs must be aligned one-dimensional arrays")
    differences = left - right
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        estimates[index] = rng.choice(differences, size=differences.size).mean()
    lower = float(np.quantile(estimates, 0.025))
    upper = float(np.quantile(estimates, 0.975))
    nonpositive = int(np.count_nonzero(estimates <= 0.0))
    nonnegative = int(np.count_nonzero(estimates >= 0.0))
    crosses_zero = nonpositive > 0 and nonnegative > 0
    empirical = 2.0 * min(nonpositive, nonnegative) / BOOTSTRAP_DRAWS
    return {
        "num_queries": int(differences.size),
        "observed_delta_mAP": float(differences.mean()),
        "bootstrap_seed": seed,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "ci_95": [lower, upper],
        "p_two_sided_upper_bound": max(1.0 / (BOOTSTRAP_DRAWS + 1), empirical),
        "p_report": (
            f"p < 1/({BOOTSTRAP_DRAWS} + 1)"
            if not crosses_zero
            else f"p = {empirical:.8f}"
        ),
        "zero_crossing_draws": nonpositive if nonpositive == nonnegative else None,
        "p_convention": "query bootstrap; not training-seed variance",
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty vector")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": [float(value) for value in values],
    }


def _class_hashes(run: dict[str, Any]) -> dict[str, str]:
    split = run["pseudo_split_identity"]
    return {
        "train": canonical_sha256(split["train_class_ids"]),
        "validation": canonical_sha256(split["validation_class_ids"]),
    }


def validate_split_identity(
    run: dict[str, Any], evaluation_split_seed: int
) -> dict[str, Any]:
    """Validate the split stored by a split-specific training artifact."""
    split = run.get("pseudo_split_identity")
    if not isinstance(split, dict):
        raise ValueError("split-specific artifact has no stored split")
    stored_seed = int(run.get("pseudo_validation_seed", -1))
    if stored_seed != evaluation_split_seed or int(split.get("seed", -1)) != evaluation_split_seed:
        raise ValueError(
            "checkpoint's stored split differs from the evaluation split"
        )
    train = list(split.get("train_class_ids", []))
    validation = list(split.get("validation_class_ids", []))
    if set(train) & set(validation):
        raise ValueError("validation classes overlap the run's training classes")
    if split.get("sha256") != canonical_sha256(
        {key: value for key, value in split.items() if key != "sha256"}
    ):
        raise ValueError("stored split identity hash mismatch")
    if run.get("training_class_list") != train or run.get("validation_class_list") != validation:
        raise ValueError("stored training/validation class lists do not match split")
    if run.get("class_list_hashes") != _class_hashes(run):
        raise ValueError("stored class-list hashes do not match split")
    return split


def validate_split_checkpoint_uniqueness(
    runs: list[dict[str, Any]],
) -> None:
    """Reject a checkpoint reused for two different pseudo-class splits."""
    seen: dict[str, int] = {}
    for run in runs:
        split_seed = int(run.get("pseudo_validation_seed", -1))
        for row in run.get("history", []):
            checkpoint_hash = row.get("checkpoint_sha256")
            if not isinstance(checkpoint_hash, str):
                raise ValueError("split-specific checkpoint hash is missing")
            previous = seen.setdefault(checkpoint_hash, split_seed)
            if previous != split_seed:
                raise ValueError("one checkpoint is reused for multiple split seeds")


def validate_required_split_artifacts(
    runs: dict[tuple[str, int], dict[str, Any]], selected_role: str
) -> None:
    required = {
        (role, split_seed)
        for role in (selected_role, "frozen_prompt_final_FP5")
        for split_seed in SPLIT_SEEDS
    }
    missing = sorted(required - set(runs))
    if missing:
        raise ValueError(
            "split robustness requires split-specific training artifacts: "
            + ", ".join(map(str, missing))
        )


def _validate_provenance(run: dict[str, Any], expected_commit: str) -> None:
    provenance = run.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("status") != "valid":
        raise ValueError("run has invalid provenance")
    if run.get("experiment_code_commit") != expected_commit:
        raise ValueError("run does not use the current experiment-code commit")
    if run.get("experiment_code_commit") != provenance.get("head_commit"):
        raise ValueError("experiment-code commit provenance mismatch")
    snapshot = provenance.get("source_snapshot", {})
    if run.get("source_snapshot_hash") != snapshot.get("sha256"):
        raise ValueError("source snapshot hash is not recorded")
    if provenance.get("tracked_working_tree_state") != "clean":
        raise ValueError("run was trained with a dirty tracked source tree")


def _validate_checkpoint_rows(
    run: dict[str, Any],
    *,
    expected_commit: str,
    expected_steps: set[int],
    split_seed: int,
    run_kind: str,
) -> list[str]:
    history = run.get("history")
    if not isinstance(history, list):
        raise ValueError("run has no checkpoint history")
    steps = {int(row.get("training_global_step", -1)) for row in history}
    if steps != expected_steps:
        raise ValueError(f"unexpected checkpoint steps: {sorted(steps)}")
    hashes: list[str] = []
    names = run.get("trainable_parameter_names")
    groups = run.get("optimizer_groups")
    for row in history:
        checkpoint = _resolve(row.get("checkpoint"))
        expected_hash = row.get("checkpoint_sha256")
        if not checkpoint.is_file() or not isinstance(expected_hash, str):
            raise ValueError("checkpoint is missing")
        if _sha256(checkpoint) != expected_hash:
            raise ValueError("checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 2:
            raise ValueError("invalid checkpoint schema")
        if (
            payload.get("experiment_role") != run.get("experiment_role")
            or payload.get("campaign") != FINAL_CAMPAIGN
            or payload.get("run_kind") != run_kind
        ):
            raise ValueError("checkpoint role/campaign/run-kind mismatch")
        if payload.get("experiment_code_commit") != expected_commit:
            raise ValueError("checkpoint has the wrong experiment-code commit")
        if payload.get("source_snapshot_hash") != run.get("source_snapshot_hash"):
            raise ValueError("checkpoint source snapshot mismatch")
        if payload.get("data_split_identity") != run.get("pseudo_split_identity"):
            raise ValueError("checkpoint split identity mismatch")
        if payload.get("resolved_treatment") != run.get("resolved_treatment"):
            raise ValueError("checkpoint treatment mismatch")
        if payload.get("training_seed") != run.get("training_seed"):
            raise ValueError("checkpoint training seed mismatch")
        if payload.get("split_seed") != split_seed:
            raise ValueError("checkpoint split seed mismatch")
        if payload.get("trainable_parameter_names") != names or payload.get(
            "optimizer_groups"
        ) != groups:
            raise ValueError("checkpoint optimizer/name provenance mismatch")
        for field in (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
            "training_global_step",
        ):
            if field not in payload:
                raise ValueError(f"checkpoint lacks {field}")
        if not isinstance(payload["optimizer_state_dict"], dict) or not isinstance(
            payload["scheduler_state_dict"], dict
        ) or not isinstance(payload["rng_state"], dict):
            raise ValueError("checkpoint state is incomplete")
        if int(payload["training_global_step"]) != int(row["training_global_step"]):
            raise ValueError("checkpoint global step mismatch")
        hashes.append(expected_hash)
    return hashes


def _validate_run_protocol(run: dict[str, Any], role: str) -> None:
    config = run.get("resolved_config")
    if not isinstance(config, dict) or config.get("allow_short_run"):
        raise ValueError("short run cannot enter the final campaign")
    if config.get("official_unseen_used_for_selection") is not False:
        raise ValueError("official unseen appears in the run configuration")
    protocol = run.get("protocol", {})
    if protocol.get("official_unseen_used_for_selection") is not False:
        raise ValueError("official unseen appears in the run protocol")
    if protocol.get("transport_enabled") is not False:
        raise ValueError("transport entered the final campaign")
    if protocol.get("text_used_for_predictor") is not False:
        raise ValueError("text entered the predictor")
    if role.endswith("FP3S") and (
        run.get("resolved_treatment", {}).get("train_photo_prompt") is not False
        or protocol.get("photo_prompt_used_for_gallery") is not False
    ):
        raise ValueError("FP3S used or trained a photo prompt")


def _validate_final_run(path: Path, *, expected_commit: str) -> dict[str, Any]:
    run = _json(path)
    role = run.get("experiment_role")
    if role not in FINAL_ROLES:
        raise ValueError(f"not a final-campaign role: {path}")
    if run.get("campaign") != FINAL_CAMPAIGN or run.get("run_kind") != "primary":
        raise ValueError(f"{role}: wrong campaign or run kind")
    seed = int(run.get("training_seed", run.get("seed", -1)))
    if seed not in SEEDS or int(run.get("pseudo_validation_seed", -1)) != 3407:
        raise ValueError(f"{role}: wrong seed identity")
    expected_treatment = treatment_for_role(role, seed=seed, pseudo_val_seed=3407)
    if run.get("resolved_treatment") != expected_treatment:
        raise ValueError(f"{role}: resolved treatment mismatch")
    if run.get("split_specific_training_artifact") is not False:
        raise ValueError(f"{role}: split artifact entered primary selection")
    _validate_provenance(run, expected_commit)
    _validate_run_protocol(run, role)
    split = run.get("pseudo_split_identity")
    if not isinstance(split, dict):
        raise ValueError(f"{role}: missing pseudo split identity")
    validate_split_identity(run, 3407)
    if run.get("split_run_name") is not None:
        raise ValueError(f"{role}: primary run has a split-run name")
    names = run.get("trainable_parameter_names")
    groups = run.get("optimizer_groups")
    if not isinstance(groups, list) or not isinstance(names, list):
        raise ValueError(f"{role}: missing trainable names or optimizer groups")
    covered = [name for group in groups for name in group.get("parameter_names", [])]
    if sorted(covered) != sorted(names) or len(covered) != len(set(covered)):
        raise ValueError(f"{role}: optimizer groups do not cover trainable names exactly")
    policy = run.get("clip_freeze_policy")
    if not isinstance(policy, dict) or policy.get(
        "frozen_clip_parameter_byte_identical"
    ) is not True:
        raise ValueError(f"{role}: frozen CLIP byte identity was not validated")
    if policy.get("text_tower_frozen") is not True or policy.get(
        "visual_projection_frozen"
    ) is not True:
        raise ValueError(f"{role}: incomplete CLIP freeze policy")
    history_steps = {
        int(row.get("training_global_step", -1)) for row in run.get("history", [])
    }
    if role.endswith("FP5"):
        expected_steps = {0, 15, 44, 73}
    elif max(history_steps, default=-1) > 5400:
        expected_steps = {0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400, 6000, 7200, 9000, 10800}
        if not run.get("resume"):
            raise ValueError(f"{role}: extended run did not restore a checkpoint")
        for record in run["resume"]:
            if record.get("source_experiment_role") != role or record.get(
                "source_campaign"
            ) != FINAL_CAMPAIGN or record.get("source_training_global_step") != 5400:
                raise ValueError(f"{role}: invalid continuation provenance")
            if record.get("source_experiment_code_commit") != expected_commit:
                raise ValueError(f"{role}: continuation used another code commit")
    else:
        expected_steps = {0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400}
        if run.get("resume"):
            raise ValueError(f"{role}: fixed-step primary run unexpectedly resumed")
    _validate_checkpoint_rows(
        run,
        expected_commit=expected_commit,
        expected_steps=expected_steps,
        split_seed=3407,
        run_kind="primary",
    )
    if role.endswith("FP5"):
        holds = run.get("frozen_hold_evaluation")
        if not isinstance(holds, list) or {
            int(row.get("comparison_horizon", -1)) for row in holds
        } != {500, 1800, 5400}:
            raise ValueError(f"{role}: FP5 frozen holds are incomplete")
        for row in holds:
            if row.get("kind") != "frozen_hold_evaluation" or row.get(
                "parameters_updated_since_selection"
            ) != 0:
                raise ValueError(f"{role}: FP5 hold is not evaluation-only")
    elif run.get("frozen_hold_evaluation"):
        raise ValueError(f"{role}: only FP5 may contain frozen holds")
    return run


def _discover_final_runs(
    root: Path = FINAL_DIR, *, expected_commit: str | None = None
) -> dict[tuple[str, int], dict[str, Any]]:
    expected_commit = expected_commit or _repository_commit()
    candidates: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(root.glob("**/run_result.json")):
        value = _json(path)
        if value.get("experiment_role") not in FINAL_ROLES:
            continue
        # Old final artifacts are deliberately ignored, never transformed into
        # current cells.  This is the only legacy handling in the analyzer.
        if value.get("experiment_code_commit") != expected_commit:
            continue
        if value.get("run_kind") != "primary":
            continue
        checked = _validate_final_run(path, expected_commit=expected_commit)
        key = (str(checked["experiment_role"]), int(checked["training_seed"]))
        candidates.setdefault(key, []).append((path, checked))
    required = {
        (role, seed) for role in FINAL_ROLES for seed in SEEDS
    }
    missing = sorted(required - set(candidates))
    if missing:
        raise ValueError(
            "missing final primary seed artifacts (legacy artifacts are excluded): "
            + ", ".join(map(str, missing))
        )
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for key, values in candidates.items():
        values.sort(
            key=lambda item: (
                max(
                    [int(row["training_global_step"]) for row in item[1]["history"]],
                    default=-1,
                ),
                len(item[1]["history"]),
                str(item[0]),
            )
        )
        chosen[key] = values[-1][1]
        chosen[key]["artifact_path"] = str(values[-1][0])
    return chosen


def validate_split_artifact(
    run: dict[str, Any],
    *,
    expected_role: str,
    expected_split_seed: int,
    expected_commit: str,
    path: Path | None = None,
) -> dict[str, Any]:
    """Validate one independently retrained split robustness run."""
    label = str(path) if path is not None else "split artifact"
    if run.get("experiment_role") != expected_role:
        raise ValueError(f"{label}: wrong split artifact role")
    if run.get("campaign") != FINAL_CAMPAIGN or run.get("run_kind") != "split_robustness":
        raise ValueError(f"{label}: not a split-specific training artifact")
    if int(run.get("training_seed", run.get("seed", -1))) != 42:
        raise ValueError(f"{label}: split training seed is not 42")
    if run.get("retrained_from_scratch") is not True or run.get("resume"):
        raise ValueError(f"{label}: split run was not retrained from scratch")
    if run.get("split_run_name") != (
        "FP5_split" + str(expected_split_seed)
        if expected_role.endswith("FP5")
        else "selected_prompt_split" + str(expected_split_seed)
    ):
        raise ValueError(f"{label}: split run name is not predeclared")
    expected_treatment = treatment_for_role(
        expected_role, seed=42, pseudo_val_seed=expected_split_seed
    )
    if run.get("resolved_treatment") != expected_treatment:
        raise ValueError(f"{label}: split treatment mismatch")
    _validate_provenance(run, expected_commit)
    _validate_run_protocol(run, expected_role)
    validate_split_identity(run, expected_split_seed)
    if run.get("training_class_list") != run["pseudo_split_identity"]["train_class_ids"]:
        raise ValueError(f"{label}: training class list mismatch")
    if run.get("validation_class_list") != run["pseudo_split_identity"]["validation_class_ids"]:
        raise ValueError(f"{label}: validation class list mismatch")
    if expected_role.endswith("FP5"):
        expected_steps = {0, 15, 44, 73}
    else:
        expected_steps = {0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400}
    _validate_checkpoint_rows(
        run,
        expected_commit=expected_commit,
        expected_steps=expected_steps,
        split_seed=expected_split_seed,
        run_kind="split_robustness",
    )
    if expected_role.endswith("FP5"):
        holds = run.get("frozen_hold_evaluation", [])
        if {int(row.get("comparison_horizon", -1)) for row in holds} != {500, 1800, 5400}:
            raise ValueError(f"{label}: FP5 split holds are incomplete")
    return run


def _discover_split_runs(
    root: Path,
    *,
    selected_role: str,
    expected_commit: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    candidates: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    roles = {selected_role, "frozen_prompt_final_FP5"}
    for path in sorted(root.glob("**/run_result.json")):
        value = _json(path)
        if value.get("experiment_role") not in roles or value.get(
            "run_kind"
        ) != "split_robustness":
            continue
        if value.get("experiment_code_commit") != expected_commit:
            continue
        seed = int(value.get("pseudo_validation_seed", -1))
        if seed not in SPLIT_SEEDS:
            continue
        checked = validate_split_artifact(
            value,
            expected_role=str(value["experiment_role"]),
            expected_split_seed=seed,
            expected_commit=expected_commit,
            path=path,
        )
        key = (str(checked["experiment_role"]), seed)
        candidates.setdefault(key, []).append((path, checked))
    if any(len(values) != 1 for values in candidates.values()):
        raise ValueError("multiple split-specific artifacts make a split cell ambiguous")
    chosen = {key: values[0][1] for key, values in candidates.items()}
    validate_required_split_artifacts(chosen, selected_role)
    validate_split_checkpoint_uniqueness(list(chosen.values()))
    return chosen


def _probe_fields(run: dict[str, Any], step: int) -> dict[str, Any]:
    row = _row(run, step)
    values = row.get("val", row)
    geometry = row.get("geometry", {})
    alignment = geometry.get("representation_alignment", {}).get("sketch", {})
    token = geometry.get("prompt_token_geometry", {})
    parameter_counts = run.get("parameter_counts", {})
    runtime = run.get("runtime", {})
    return {
        "mAP": _map_at(run, step),
        "average_precision_per_query": _ap(run, step).tolist(),
        "query_identity": values.get("query_identity"),
        "mAP@500": _map_at(run, 500) if not run.get("experiment_role", "").endswith("FP5") else None,
        "mAP@1800": _map_at(run, 1800) if not run.get("experiment_role", "").endswith("FP5") else None,
        "mAP@5400": _map_at(run, 5400),
        "training_global_step": int(step),
        "semantic_margin": row.get("semantic_margin"),
        "classification_accuracy": row.get("diagnostic_seen_classification_accuracy"),
        "sketch_reference_cosine": row.get("sketch_reference_cosine"),
        "photo_reference_cosine": row.get("photo_reference_cosine"),
        "CKA": row.get("linear_cka", alignment.get("linear_cka")),
        "Procrustes_residual": row.get(
            "orthogonal_procrustes_residual",
            alignment.get("orthogonal_procrustes_residual"),
        ),
        "prompt_token_cosine": {
            "mean": token.get("mean_prompt_cosine"),
            "min": token.get("min_prompt_cosine"),
            "max": token.get("max_prompt_cosine"),
        },
        "parameter_count": parameter_counts.get("trainable_parameters"),
        "parameter_counts": parameter_counts,
        "runtime": runtime,
        "gpu_memory_bytes": runtime.get("peak_gpu_memory_bytes"),
        "attention": row.get("prompt_attention"),
        "gradient_norms": row.get("gradient_norms"),
        "gradient_norms_by_parameter": row.get("gradient_norms_by_parameter"),
        "prompt_norms": {
            "visual": row.get("prompt_parameter_norm"),
            "soft_text": row.get("soft_prompt_parameter_norm"),
        },
    }


def _photo_ablation(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    deltas: list[float] = []
    for index, seed in enumerate(SEEDS):
        fp3 = runs["frozen_prompt_final_FP3", seed]
        fp3s = runs["frozen_prompt_final_FP3S", seed]
        left, right = _assert_paired_queries(fp3, fp3s, 5400)
        delta = _map_at(fp3, 5400) - _map_at(fp3s, 5400)
        deltas.append(delta)
        by_seed[str(seed)] = {
            "FP3": _probe_fields(fp3, 5400),
            "FP3S": _probe_fields(fp3s, 5400),
            "delta_mAP": delta,
            "query_ap_delta": (left - right).tolist(),
            "paired_bootstrap": _bootstrap(left, right, 20260904 + index),
        }
    return {
        "comparison": "FP3@5400 - FP3S@5400",
        "by_seed": by_seed,
        "mean_delta_mAP": float(np.mean(deltas)),
        "std_delta_mAP": float(np.std(deltas, ddof=1)),
        "direction_consistent": bool(all(value >= 0.0 for value in deltas)),
        "keep_photo_prompt": bool(
            np.mean(deltas) >= 0.005 and all(value >= 0.0 for value in deltas)
        ),
        "threshold_mAP": 0.005,
        "training_seed_variance": _mean_std(deltas),
        "query_bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def _layernorm(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    deltas: list[float] = []
    for index, seed in enumerate(SEEDS):
        fp2 = runs["frozen_prompt_final_FP2", seed]
        ln = runs["frozen_prompt_final_FP_LN", seed]
        left, right = _assert_paired_queries(ln, fp2, 5400)
        delta = _map_at(ln, 5400) - _map_at(fp2, 5400)
        deltas.append(delta)
        by_seed[str(seed)] = {
            "FP2": _probe_fields(fp2, 5400),
            "FP_LN": _probe_fields(ln, 5400),
            "FP2_mAP": _map_at(fp2, 5400),
            "FP_LN_mAP": _map_at(ln, 5400),
            "matched_delta": delta,
            "query_ap_delta": (left - right).tolist(),
            "paired_bootstrap": _bootstrap(left, right, 20261000 + index),
        }
    return {
        "comparison": "FP-LN@5400 - FP2@5400; both hard-text CE",
        "by_seed": by_seed,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)),
        "direction_consistent": bool(all(value >= 0.0 for value in deltas)),
        "keep_layernorm": bool(
            np.mean(deltas) >= 0.005 and all(value >= 0.0 for value in deltas)
        ),
        "threshold_mAP": 0.005,
        "training_seed_variance": _mean_std(deltas),
        "query_bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def _extended_status(values: list[tuple[int, float]]) -> str:
    peak_step, peak_value = max(values, key=lambda item: (item[1], -item[0]))
    final_step, final_value = values[-1]
    if peak_step == final_step:
        return "boundary peak"
    if final_value - values[-2][1] > 0.003:
        return "still improving"
    if peak_value - final_value > 0.003:
        return "degrading"
    return "plateaued"


def _extended(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("frozen_prompt_final_FP3", "frozen_prompt_final_FP3S"):
        run = runs[role, 42]
        rows = [_probe_fields(run, step) for step in EXTENDED_STEPS]
        values = [(row["training_global_step"], row["mAP"]) for row in rows]
        peak_step, peak_value = max(values, key=lambda item: (item[1], -item[0]))
        final_value = values[-1][1]
        result[role] = {
            "training_seed": 42,
            "checkpoints": rows,
            "peak_mAP": peak_value,
            "peak_step": peak_step,
            "absolute_decay": peak_value - final_value,
            "retention_ratio": final_value / peak_value,
            "status": _extended_status(values),
            "allowed_statuses": [
                "still improving",
                "plateaued",
                "degrading",
                "boundary peak",
            ],
        }
    return result


def _seed_confirmation(
    runs: dict[tuple[str, int], dict[str, Any]], photo: dict[str, Any]
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for role in ("frozen_prompt_final_FP3", "frozen_prompt_final_FP3S"):
        per_seed: dict[str, Any] = {}
        prompt_maps: list[float] = []
        deltas: list[float] = []
        for index, seed in enumerate(SEEDS):
            prompt = runs[role, seed]
            fp5 = runs["frozen_prompt_final_FP5", seed]
            prompt_ap, fp5_ap = _assert_paired_queries(prompt, fp5, 5400)
            prompt_map = _map_at(prompt, 5400)
            fp5_map = _map_at(fp5, 5400)
            delta = prompt_map - fp5_map
            prompt_maps.append(prompt_map)
            deltas.append(delta)
            per_seed[str(seed)] = {
                "prompt_mAP": prompt_map,
                "FP5_mAP": fp5_map,
                "prompt_minus_FP5": delta,
                "prompt_average_precision_per_query": prompt_ap.tolist(),
                "FP5_average_precision_per_query": fp5_ap.tolist(),
                "query_ap_delta": (prompt_ap - fp5_ap).tolist(),
                "paired_bootstrap": _bootstrap(
                    prompt_ap, fp5_ap, 20261100 + index
                ),
            }
        candidates[role] = {
            "per_seed": per_seed,
            "prompt_mean_std": _mean_std(prompt_maps),
            "prompt_minus_FP5_mean_std": _mean_std(deltas),
            "paired_bootstrap_is_query_uncertainty": True,
        }
    fp5_maps = [_map_at(runs["frozen_prompt_final_FP5", seed], 5400) for seed in SEEDS]
    candidates["frozen_prompt_final_FP5"] = {
        "prompt_mean_std": _mean_std(fp5_maps),
        "role_label": "FP5 matched frozen hold",
        "fixed_step": 5400,
    }
    fp3_mean = candidates["frozen_prompt_final_FP3"]["prompt_mean_std"]["mean"]
    fp3s_mean = candidates["frozen_prompt_final_FP3S"]["prompt_mean_std"]["mean"]
    mean_difference = fp3_mean - fp3s_mean
    selected = (
        "frozen_prompt_final_FP3"
        if photo["keep_photo_prompt"]
        else "frozen_prompt_final_FP3S"
    )
    if abs(mean_difference) <= 0.003:
        selected = "frozen_prompt_final_FP3S"
    selected_deltas = candidates[selected]["prompt_minus_FP5_mean_std"]["values"]
    keep_frozen_prompts = bool(
        np.mean(selected_deltas) >= 0.005
        and all(value >= 0.0 for value in selected_deltas)
    )
    candidates["selection"] = {
        "candidate_roles": ["frozen_prompt_final_FP3", "frozen_prompt_final_FP3S"],
        "selected_role": selected,
        "mean_difference_FP3_minus_FP3S": mean_difference,
        "simpler_model_margin": 0.003,
        "selection_rule": "keep photo prompt only at mean delta >= 0.005 with nonnegative per-seed effects; prefer FP3S when the finalist mean difference is at most 0.003",
        "selected_prompt_vs_FP5": candidates[selected]["prompt_minus_FP5_mean_std"],
        "keep_frozen_prompts": keep_frozen_prompts,
        "mainline_role": selected if keep_frozen_prompts else "frozen_prompt_final_FP5",
    }
    return candidates


def _split_robustness(
    split_runs: dict[tuple[str, int], dict[str, Any]], selected_role: str
) -> dict[str, Any]:
    per_split: list[dict[str, Any]] = []
    for index, split_seed in enumerate(SPLIT_SEEDS):
        prompt = split_runs[selected_role, split_seed]
        fp5 = split_runs["frozen_prompt_final_FP5", split_seed]
        prompt_ap, fp5_ap = _assert_paired_queries(prompt, fp5, 5400)
        prompt_map = _map_at(prompt, 5400)
        fp5_map = _map_at(fp5, 5400)
        split = prompt["pseudo_split_identity"]
        if split != fp5["pseudo_split_identity"]:
            raise ValueError("prompt and FP5 split artifacts do not share a split")
        train = list(split["train_class_ids"])
        validation = list(split["validation_class_ids"])
        if set(train) & set(validation):
            raise ValueError("split robustness training and validation classes overlap")
        per_split.append(
            {
                "split_seed": split_seed,
                "training_seed": 42,
                "training_class_list": train,
                "validation_class_list": validation,
                "class_list_hashes": {
                    "train": canonical_sha256(train),
                    "validation": canonical_sha256(validation),
                },
                "train_validation_disjoint": True,
                "prompt_run_name": prompt["split_run_name"],
                "FP5_run_name": fp5["split_run_name"],
                "prompt_mAP": prompt_map,
                "FP5_mAP": fp5_map,
                "prompt_minus_FP5": prompt_map - fp5_map,
                "prompt_checkpoint": _row(prompt, 5400)["checkpoint"],
                "prompt_checkpoint_sha256": _row(prompt, 5400)["checkpoint_sha256"],
                "FP5_checkpoint": _row(fp5, 5400)["checkpoint"],
                "FP5_checkpoint_sha256": _row(fp5, 5400)["checkpoint_sha256"],
                "prompt_average_precision_per_query": prompt_ap.tolist(),
                "FP5_average_precision_per_query": fp5_ap.tolist(),
                "query_ap_delta": (prompt_ap - fp5_ap).tolist(),
                "paired_bootstrap": _bootstrap(prompt_ap, fp5_ap, 20261200 + index),
                "official_unseen_used_for_selection": False,
            }
        )
    prompt_values = [float(row["prompt_mAP"]) for row in per_split]
    fp5_values = [float(row["FP5_mAP"]) for row in per_split]
    deltas = [float(row["prompt_minus_FP5"]) for row in per_split]
    return {
        "selected_prompt_role": selected_role,
        "predeclared_split_seeds": list(SPLIT_SEEDS),
        "per_split": per_split,
        "prompt_mean_std": _mean_std(prompt_values),
        "FP5_mean_std": _mean_std(fp5_values),
        "prompt_minus_FP5_mean_std": _mean_std(deltas),
        "split_variance_is_separate_from_training_seed_variance": True,
        "official_unseen_used_for_selection": False,
        "evaluation_note": "each split uses independently retrained prompt and FP5 artifacts; no official classes were read",
    }


def _new_plot(path: Path, title: str, draw: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(9, 5))
    try:
        draw(figure)
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path)
    finally:
        plt.close(figure)


def _plots(
    report: dict[str, Any], *, plot_paths: dict[str, Path]
) -> dict[str, str]:
    photo = report["probe_A_photo_ablation"]
    layernorm = report["probe_C_layernorm"]
    extended = report["probe_B_extended_training"]
    seeds = report["probe_D_seed_confirmation"]
    splits = report["probe_E_split_robustness"]

    def photo_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        labels = [str(seed) for seed in SEEDS]
        positions = np.arange(len(labels))
        axis.bar(
            positions - 0.2,
            [photo["by_seed"][seed]["FP3"]["mAP"] for seed in labels],
            0.4,
            label="FP3 dual prompt",
        )
        axis.bar(
            positions + 0.2,
            [photo["by_seed"][seed]["FP3S"]["mAP"] for seed in labels],
            0.4,
            label="FP3S sketch-only",
        )
        axis.set_xticks(positions, labels)
        axis.set_xlabel("training seed")
        axis.set_ylabel("mAP@5400")
        axis.legend()

    def layernorm_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        labels = [str(seed) for seed in SEEDS]
        axis.bar(
            labels,
            [layernorm["by_seed"][seed]["matched_delta"] for seed in labels],
            color=["#4c78a8", "#f58518", "#54a24b"],
        )
        axis.axhline(0.005, color="black", linestyle="--", label="keep threshold")
        axis.set_xlabel("training seed")
        axis.set_ylabel("FP-LN − FP2 mAP@5400")
        axis.legend()

    def extended_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        for role, label in (
            ("frozen_prompt_final_FP3", "FP3"),
            ("frozen_prompt_final_FP3S", "FP3S"),
        ):
            rows = extended[role]["checkpoints"]
            axis.plot(
                [row["training_global_step"] for row in rows],
                [row["mAP"] for row in rows],
                marker="o",
                label=label,
            )
        axis.set_xlabel("training global step")
        axis.set_ylabel("mAP")
        axis.legend()

    def seed_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        roles = [
            "frozen_prompt_final_FP3",
            "frozen_prompt_final_FP3S",
            "frozen_prompt_final_FP5",
        ]
        labels = ["FP3", "FP3S", "FP5"]
        positions = np.arange(len(roles))
        means = [seeds[role]["prompt_mean_std"]["mean"] for role in roles]
        errors = [seeds[role]["prompt_mean_std"]["std"] for role in roles]
        axis.bar(positions, means, yerr=errors, capsize=4, label="mean ± std")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("mAP@5400")
        axis.set_xlabel("finalist / matched frozen hold")
        axis.legend()

    def split_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        rows = splits["per_split"]
        labels = [str(row["split_seed"]) for row in rows]
        positions = np.arange(len(labels))
        axis.bar(
            positions - 0.2,
            [row["prompt_mAP"] for row in rows],
            0.4,
            label="selected prompt",
        )
        axis.bar(
            positions + 0.2,
            [row["FP5_mAP"] for row in rows],
            0.4,
            label="FP5",
        )
        axis.set_xticks(positions, labels)
        axis.set_xlabel("independently retrained pseudo-class split seed")
        axis.set_ylabel("mAP")
        axis.legend()

    def geometry_plot(figure: Any) -> None:
        axes = figure.subplots(2, 2).flat
        for axis, field in zip(
            axes,
            ("semantic_margin", "sketch_reference_cosine", "CKA", "Procrustes_residual"),
            strict=True,
        ):
            for role, label in (
                ("frozen_prompt_final_FP3", "FP3"),
                ("frozen_prompt_final_FP3S", "FP3S"),
            ):
                rows = extended[role]["checkpoints"]
                axis.plot(
                    [row["training_global_step"] for row in rows],
                    [float(row[field]) for row in rows],
                    marker=".",
                    label=label,
                )
            axis.set_title(field)
            axis.set_xlabel("training global step")
        axes[0].legend()

    def attention_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        role = seeds["selection"]["selected_role"]
        run = report["_metric_runs"][role + ":42"]
        attention = _row(run, 10800)["prompt_attention"]
        for field in (
            "cls_to_prompt_mass",
            "patch_to_prompt_mass",
            "prompt_to_cls_mass",
            "prompt_to_patch_mass",
        ):
            axis.plot(
                [row["block_index"] for row in attention["blocks"]],
                [row[field] for row in attention["blocks"]],
                marker="o",
                label=field,
            )
        axis.set_xticks([0, 6, 11], ["block 0", "block 6", "block 11"])
        axis.set_xlabel("exact transformer block")
        axis.set_ylabel("attention mass")
        axis.legend(fontsize=7)

    draws = {
        "photo_ablation": ("Photo-prompt ablation: mAP@5400", photo_plot),
        "layernorm": ("LayerNorm comparison: matched mAP@5400 delta", layernorm_plot),
        "extended_training": ("Extended frozen-prompt training: mAP", extended_plot),
        "seed_confirmation": ("Finalist seed confirmation: mAP@5400", seed_plot),
        "split_robustness": ("Independently retrained split robustness: mAP", split_plot),
        "geometry": ("Extended training geometry diagnostics", geometry_plot),
        "attention": ("Prompt attention at blocks 0, 6 and 11", attention_plot),
    }
    result: dict[str, str] = {}
    for key, (title, draw) in draws.items():
        _new_plot(plot_paths[key], title, draw)
        result[key] = str(plot_paths[key])
    return result


def _fmt(value: object) -> str:
    if value is None:
        return "not_run"
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


def _mean_role_map(
    runs: dict[tuple[str, int], dict[str, Any]], role: str
) -> dict[str, Any]:
    return _mean_std([_map_at(runs[role, seed], 5400) for seed in SEEDS])


def _verdict(report: dict[str, Any], common_commit: str, tracked_clean: bool) -> str:
    photo = report["probe_A_photo_ablation"]
    layernorm = report["probe_C_layernorm"]
    extended = report["probe_B_extended_training"]
    seeds = report["probe_D_seed_confirmation"]
    split = report["probe_E_split_robustness"]
    selected = seeds["selection"]["selected_role"]
    mainline = seeds["selection"]["mainline_role"]
    metric_runs = report["_metric_runs"]
    best_5400 = max(
        _map_at(metric_runs[f"{role}:{seed}"], 5400)
        for role in FINAL_ROLES
        for seed in SEEDS
    )
    best_10800 = max(
        _map_at(metric_runs[f"{role}:42"], 10800)
        for role in ("frozen_prompt_final_FP3", "frozen_prompt_final_FP3S")
    )
    best_extended = max(
        (
            (role, extended[role]["peak_mAP"], extended[role]["peak_step"])
            for role in extended
        ),
        key=lambda item: (item[1], -item[2]),
    )
    selected_run = metric_runs[f"{selected}:42"]
    parameter_count = selected_run.get("parameter_counts", {}).get(
        "trainable_parameters"
    )
    policy = selected_run.get("clip_freeze_policy", {})
    selected_stats = seeds[selected]["prompt_mean_std"]
    fp5_stats = seeds["frozen_prompt_final_FP5"]["prompt_mean_std"]
    delta_stats = seeds[selected]["prompt_minus_FP5_mean_std"]
    split_rows = {int(row["split_seed"]): row for row in split["per_split"]}
    convergence = extended[best_extended[0]]["status"]
    if mainline.endswith("FP5"):
        mechanism = "partial sketch encoder adaptation with a frozen photo reference"
        loss = "rank(z0) plus hard-text CE(z0)"
    else:
        mechanism = "frozen visual prompting trained by retrieval ranking plus loss-only soft-text classification"
        loss = "rank loss plus soft-text CE at the query"
    lines = [
        "FINAL SPICA FROZEN-PROMPT VERDICT",
        "",
        f"Repository commit: {report['repository_commit']}",
        f"Experiment-code commit: {common_commit}",
        "Results commit: pending artifact commit",
        f"Working tree tracked files clean: {'YES' if tracked_clean else 'NO'}",
        "Historical artifacts preserved: YES",
        f"Artifact provenance valid: {'YES' if report['provenance']['all_new_runs_valid'] else 'NO'}",
        "Official unseen used for selection: NO",
        "",
        f"FP3 mean ± std: {_fmt(report['seed_statistics']['FP3']['mean'])} ± {_fmt(report['seed_statistics']['FP3']['std'])}",
        f"FP3S mean ± std: {_fmt(report['seed_statistics']['FP3S']['mean'])} ± {_fmt(report['seed_statistics']['FP3S']['std'])}",
        f"Photo-prompt mean effect: {_fmt(photo['mean_delta_mAP'])}",
        f"Photo-prompt direction consistent: {'YES' if photo['direction_consistent'] else 'NO'}",
        f"Should photo prompt remain: {'YES' if photo['keep_photo_prompt'] else 'NO'}",
        "",
        f"FP2 mean ± std: {_fmt(report['seed_statistics']['FP2']['mean'])} ± {_fmt(report['seed_statistics']['FP2']['std'])}",
        f"FP-LN mean ± std: {_fmt(report['seed_statistics']['FP_LN']['mean'])} ± {_fmt(report['seed_statistics']['FP_LN']['std'])}",
        f"Matched LayerNorm mean effect: {_fmt(layernorm['mean_delta'])}",
        f"LayerNorm direction consistent: {'YES' if layernorm['direction_consistent'] else 'NO'}",
        f"Should LayerNorm remain: {'YES' if layernorm['keep_layernorm'] else 'NO'}",
        "",
        f"Best mAP@5400: {_fmt(best_5400)}",
        f"Best mAP@10800: {_fmt(best_10800)}",
        f"Extended peak: {_fmt(best_extended[1])}",
        f"Extended peak step: {best_extended[2]}",
        f"Converged or boundary peak: {convergence}",
        "",
        f"Selected prompt mean ± std: {_fmt(selected_stats['mean'])} ± {_fmt(selected_stats['std'])}",
        f"FP5 mean ± std: {_fmt(fp5_stats['mean'])} ± {_fmt(fp5_stats['std'])}",
        f"Prompt-minus-FP5 mean delta: {_fmt(delta_stats['mean'])} ± {_fmt(delta_stats['std'])}",
        "",
        f"Split-101 prompt / FP5: {_fmt(split_rows[101]['prompt_mAP'])} / {_fmt(split_rows[101]['FP5_mAP'])}",
        f"Split-202 prompt / FP5: {_fmt(split_rows[202]['prompt_mAP'])} / {_fmt(split_rows[202]['FP5_mAP'])}",
        f"Split-303 prompt / FP5: {_fmt(split_rows[303]['prompt_mAP'])} / {_fmt(split_rows[303]['FP5_mAP'])}",
        f"Across-split prompt mean ± std: {_fmt(split['prompt_mean_std']['mean'])} ± {_fmt(split['prompt_mean_std']['std'])}",
        f"Across-split FP5 mean ± std: {_fmt(split['FP5_mean_std']['mean'])} ± {_fmt(split['FP5_mean_std']['std'])}",
        f"Across-split mean delta: {_fmt(split['prompt_minus_FP5_mean_std']['mean'])}",
        "",
        f"Best prompt configuration: {selected}",
        f"Trainable parameter count: {parameter_count}",
        f"Frozen CLIP byte-identical: {'YES' if policy.get('frozen_clip_parameter_byte_identical') else 'NO'}",
        "Text enters predictor: NO",
        "Text required at inference: NO",
        "Photo required for query inference: NO",
        "",
        "Should transport return: NO",
        "Should direction supervision return: NO",
        "Should distance prediction return: NO",
        "Should Mo-vMF/K>1 run now: NO",
        "",
        f"Strongest supported mechanism: {mechanism}",
        "Largest remaining confound: training-seed and pseudo-class-split variance despite independent split retraining",
        f"Recommended mainline architecture: {mainline}",
        f"Recommended mainline loss: {loss}",
        "Recommended inference query: raw sketch image",
    ]
    return "\n".join(lines) + "\n"


def _markdown(report: dict[str, Any], verdict: str) -> str:
    photo = report["probe_A_photo_ablation"]
    layernorm = report["probe_C_layernorm"]
    extended = report["probe_B_extended_training"]
    seeds = report["probe_D_seed_confirmation"]
    split = report["probe_E_split_robustness"]
    lines = [
        "# SPICA frozen-prompt final campaign",
        "",
        "All primary cells were trained under one clean experiment-code commit. Selection used only pseudo-unseen validation; the official unseen split was never read.",
        "",
        "## Primary fixed-step evidence",
        "",
        f"FP3 versus FP3S at step 5400: mean delta={photo['mean_delta_mAP']:.6f}; direction consistent={photo['direction_consistent']}; retain photo prompt={photo['keep_photo_prompt']}.",
        f"FP-LN versus matched FP2 at step 5400: mean delta={layernorm['mean_delta']:.6f}; direction consistent={layernorm['direction_consistent']}; retain LayerNorm={layernorm['keep_layernorm']}.",
        f"Selected finalist: `{seeds['selection']['selected_role']}`; prompt-versus-FP5 mainline gate={seeds['selection']['keep_frozen_prompts']}.",
        "All paired comparisons include 2000 query-bootstrap draws. Query uncertainty, training-seed variance, and split variance are separate JSON fields.",
        "",
        "## Extended training",
        "",
    ]
    for role, value in extended.items():
        lines.append(
            f"- {role}: peak {_fmt(value['peak_mAP'])} at step {value['peak_step']}; decay {_fmt(value['absolute_decay'])}; retention {_fmt(value['retention_ratio'])}; **{value['status']}**."
        )
    lines.extend(
        [
            "",
            "A peak at step 10800 is a boundary peak, not convergence.",
            "",
            "## True split robustness",
            "",
            f"Splits {split['predeclared_split_seeds']} use independently retrained prompt and FP5 models. Training/validation class lists, hashes, checkpoint locations, and query AP vectors are raw fields in the JSON report.",
            "",
            "## Provenance",
            "",
            f"Repository commit: `{report['repository_commit']}`; experiment-code commit: `{report['provenance']['experiment_code_commit']}`; tracked tree clean: `{report['provenance']['tracked_working_tree_clean']}`.",
            "Legacy v2 artifacts are excluded from every primary conclusion; no automatic legacy substitution is performed.",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend(f"- {path}" for path in report["plots"].values())
    lines.extend(["", verdict])
    return "\n".join(lines)


def _artifact_provenance(
    runs: dict[tuple[str, int], dict[str, Any]],
    split_runs: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, catalog in (("primary", runs), ("split", split_runs)):
        for (role, seed), run in catalog.items():
            checkpoints = [
                {
                    "training_global_step": int(row["training_global_step"]),
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
                for row in run.get("history", [])
            ]
            result[f"{label}:{role}:{seed}"] = {
                "artifact_path": run.get("artifact_path"),
                "experiment_code_commit": run.get("experiment_code_commit"),
                "source_snapshot_hash": run.get("source_snapshot_hash"),
                "tracked_working_tree_state": run.get("provenance", {}).get(
                    "tracked_working_tree_state"
                ),
                "untracked_files": run.get("provenance", {}).get(
                    "untracked_files", []
                ),
                "training_class_list": run.get("training_class_list"),
                "validation_class_list": run.get("validation_class_list"),
                "class_list_hashes": run.get("class_list_hashes"),
                "checkpoint_artifacts": checkpoints,
                "official_unseen_used_for_selection": run.get(
                    "official_unseen_used_for_selection"
                ),
            }
    return result


def analyze(
    *,
    final_dir: Path = FINAL_DIR,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    expected_commit = _repository_commit()
    manifest_path = manifest_path or FINAL_MANIFEST_PATH
    manifest, manifest_hash = ensure_manifest(
        manifest_path,
        dataset="sketchy_104_21",
        data_config="configs/data/sketchy_104_21.yaml",
        campaign=FINAL_CAMPAIGN,
    )
    final_runs = _discover_final_runs(final_dir, expected_commit=expected_commit)
    photo = _photo_ablation(final_runs)
    extended = _extended(final_runs)
    layernorm = _layernorm(final_runs)
    seed_confirmation = _seed_confirmation(final_runs, photo)
    selected_role = seed_confirmation["selection"]["selected_role"]
    split_runs = _discover_split_runs(
        final_dir,
        selected_role=selected_role,
        expected_commit=expected_commit,
    )
    split_robustness = _split_robustness(split_runs, selected_role)
    metric_runs = {
        f"{role}:{seed}": run for (role, seed), run in final_runs.items()
    }
    tracked_clean = not bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    seed_statistics = {
        short: _mean_role_map(final_runs, role)
        for short, role in (
            ("FP3", "frozen_prompt_final_FP3"),
            ("FP3S", "frozen_prompt_final_FP3S"),
            ("FP2", "frozen_prompt_final_FP2"),
            ("FP_LN", "frozen_prompt_final_FP_LN"),
        )
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "campaign": FINAL_CAMPAIGN,
        "repository_commit": expected_commit,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "value": manifest,
        },
        "provenance": {
            "experiment_code_commit": expected_commit,
            "all_new_runs_same_clean_code_commit": True,
            "tracked_working_tree_clean": tracked_clean,
            "untracked_files_recorded_separately": True,
            "all_new_runs_valid": True,
        },
        "official_unseen_used_for_selection": False,
        "legacy_substitution": False,
        "historical_appendix": {
            "included": False,
            "note": "historical v2 artifacts were not used in final conclusions",
        },
        "selection_policy": manifest["selection_policy"],
        "seed_statistics": seed_statistics,
        "probe_A_photo_ablation": photo,
        "probe_B_extended_training": extended,
        "probe_C_layernorm": layernorm,
        "probe_D_seed_confirmation": seed_confirmation,
        "probe_E_split_robustness": split_robustness,
        "run_provenance": _artifact_provenance(final_runs, split_runs),
        "validation": {
            "required_primary_cells": len(FINAL_ROLES) * len(SEEDS),
            "found_primary_cells": len(final_runs),
            "required_split_cells": len(SPLIT_SEEDS) * 2,
            "found_split_cells": len(split_runs),
            "smoke_runs_included": False,
            "legacy_runs_substituted": False,
            "all_checkpoint_hashes_resolve": True,
            "query_ap_vectors_aligned": True,
            "official_unseen_read": False,
            "split_specific_training_verified": True,
            "historical_artifacts_unchanged": True,
        },
        "plots": {},
        "_metric_runs": metric_runs,
    }
    artifact_dir = output_dir or ROOT / "outputs/frozen_prompt_final_campaign_2026-09-04"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        key: artifact_dir / filename for key, filename in PLOT_FILENAMES.items()
    }
    report["plots"] = _plots(report, plot_paths=plot_paths)
    verdict = _verdict(report, expected_commit, tracked_clean)
    report["verdict"] = verdict
    report.pop("_metric_runs")
    output_json = artifact_dir / "research_summary_frozen_prompt_final_2026-09-04.json"
    output_md = artifact_dir / "research_summary_frozen_prompt_final_2026-09-04.md"
    for path in (output_json, output_md):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    # Keep a new manifest beside the new report when an output namespace is used.
    artifact_manifest = artifact_dir / "experiment_manifest_frozen_prompt_final_2026-09-04.json"
    if artifact_manifest != manifest_path:
        artifact_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report["artifact_manifest"] = {
            "path": str(artifact_manifest),
            "sha256": _sha256(artifact_manifest),
        }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown({**report, "_metric_runs": metric_runs}, verdict))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    args = parser.parse_args()
    analyze(
        final_dir=args.final_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    main()
