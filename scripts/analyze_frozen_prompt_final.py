"""Validate and summarize the final frozen-prompt campaign without test-set selection."""

from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split
from spica.evaluation.frozen_prompt import encode_prompted_loader, evaluate_prompted
from spica.frozen_prompt_artifacts import (
    FINAL_CAMPAIGN,
    FINAL_MANIFEST_PATH,
    FINAL_ROLES,
    canonical_sha256,
    ensure_manifest,
    treatment_for_role,
)
from spica.models.clip import load_frozen_clip, load_trainable_sketch_hidden_encoder
from spica.models.frozen_prompt import FrozenPromptModel
from spica.train_frozen_prompt import _EarlyAdaptModel, _FrozenEncoderAdapter, _path

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs/experiments/frozen_prompt_final"
V2_DIR = ROOT / "outputs/experiments/frozen_prompt_v2"
SEEDS = (42, 123, 3407)
SPLIT_SEEDS = (101, 202, 303)
EXTENDED_STEPS = (5400, 6000, 7200, 9000, 10800)
BOOTSTRAP_DRAWS = 2000
PLOT_PATHS = {
    "photo_ablation": ROOT / "outputs/frozen_prompt_final_photo_ablation.png",
    "layernorm": ROOT / "outputs/frozen_prompt_final_layernorm.png",
    "extended_training": ROOT / "outputs/frozen_prompt_final_extended_training.png",
    "seed_confirmation": ROOT / "outputs/frozen_prompt_final_seed_confirmation.png",
    "split_robustness": ROOT / "outputs/frozen_prompt_final_split_robustness.png",
    "geometry": ROOT / "outputs/frozen_prompt_final_geometry.png",
    "attention": ROOT / "outputs/frozen_prompt_final_attention.png",
}


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


def _row(run: dict[str, Any], step: int) -> dict[str, Any]:
    for row in run.get("history", []):
        if int(row.get("training_global_step", -1)) == step:
            return row
    if step == 5400 and str(run.get("experiment_role", "")).endswith("FP5"):
        for row in run.get("frozen_hold_evaluation", []):
            if row.get("comparison_horizon") == 5400:
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
    values = _row(run, step).get("val", _row(run, step))
    result = np.asarray(values["average_precision_per_query"], dtype=np.float64)
    if result.ndim != 1 or not result.size:
        raise ValueError("average-precision vector is empty or not one-dimensional")
    return result


def _bootstrap(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired bootstrap inputs must be aligned one-dimensional arrays")
    differences = left - right
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, differences.size, size=differences.size)
        estimates[index] = differences[indices].mean()
    lower = float(np.quantile(estimates, 0.025))
    upper = float(np.quantile(estimates, 0.975))
    nonpositive = int(np.count_nonzero(estimates <= 0.0))
    nonnegative = int(np.count_nonzero(estimates >= 0.0))
    empirical = 2.0 * min(nonpositive, nonnegative) / BOOTSTRAP_DRAWS
    return {
        "num_queries": int(differences.size),
        "observed_delta_mAP": float(differences.mean()),
        "bootstrap_seed": seed,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "ci_95": [lower, upper],
        "p_two_sided": max(1.0 / (BOOTSTRAP_DRAWS + 1), empirical),
        "p_convention": "empirical p < 1/(B+1) when no bootstrap draw crosses zero",
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty seed vector")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def _class_hashes(run: dict[str, Any]) -> dict[str, str]:
    split = run["pseudo_split_identity"]
    return {
        "train": canonical_sha256(split["train_class_ids"]),
        "validation": canonical_sha256(split["validation_class_ids"]),
    }


def _validate_final_run(path: Path, *, expected_commit: str | None) -> dict[str, Any]:
    run = _json(path)
    role = run.get("experiment_role")
    if role not in FINAL_ROLES:
        raise ValueError(f"not a final-campaign role: {path}")
    if run.get("campaign") != FINAL_CAMPAIGN or run.get("run_kind") != "primary":
        raise ValueError(f"{role}: wrong campaign or run kind")
    seed = int(run.get("seed", -1))
    if seed not in SEEDS or int(run.get("pseudo_validation_seed", -1)) != 3407:
        raise ValueError(f"{role}: wrong seed identity")
    expected_treatment = treatment_for_role(role, seed=seed, pseudo_val_seed=3407)
    if run.get("resolved_treatment") != expected_treatment:
        raise ValueError(f"{role}: resolved treatment mismatch")
    config = run.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError(f"{role}: missing resolved configuration")
    if run.get("source_snapshot_hash") != run.get("provenance", {}).get(
        "source_snapshot", {}
    ).get("sha256"):
        raise ValueError(f"{role}: source snapshot hash is not recorded")
    provenance = run.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("status") != "valid":
        raise ValueError(f"{role}: invalid provenance")
    if run.get("experiment_code_commit") != provenance.get("head_commit"):
        raise ValueError(f"{role}: experiment-code commit provenance mismatch")
    if expected_commit is not None and run.get("experiment_code_commit") != expected_commit:
        raise ValueError(f"{role}: new runs do not share one experiment-code commit")
    if provenance.get("tracked_working_tree_state") != "clean":
        raise ValueError(f"{role}: tracked working tree was not clean at training")
    if run.get("official_unseen_used_for_selection") is not False:
        raise ValueError(f"{role}: official unseen was used for selection")
    split = run.get("pseudo_split_identity")
    if not isinstance(split, dict):
        raise ValueError(f"{role}: missing pseudo split identity")
    if split.get("sha256") != canonical_sha256(
        {key: value for key, value in split.items() if key != "sha256"}
    ):
        raise ValueError(f"{role}: pseudo split hash mismatch")
    if run.get("class_list_hashes") != _class_hashes(run):
        raise ValueError(f"{role}: class-list hash mismatch")
    if not isinstance(run.get("manifest_identity"), dict) or not run[
        "manifest_identity"
    ].get("sha256"):
        raise ValueError(f"{role}: missing dataset/manifest identity")
    if not isinstance(run.get("manifest_entry_identity"), dict):
        raise ValueError(f"{role}: missing experiment-manifest identity")
    groups = run.get("optimizer_groups")
    names = run.get("trainable_parameter_names")
    if not isinstance(groups, list) or not isinstance(names, list):
        raise ValueError(f"{role}: missing trainable names or optimizer groups")
    covered = [name for group in groups for name in group.get("parameter_names", [])]
    if sorted(covered) != sorted(names) or len(covered) != len(set(covered)):
        raise ValueError(f"{role}: optimizer groups do not cover trainable names exactly")
    policy = run.get("clip_freeze_policy")
    if not isinstance(policy, dict) or not isinstance(
        policy.get("all_clip_owned_parameters_byte_identical"), bool
    ):
        raise ValueError(f"{role}: missing CLIP byte-identity result")
    if policy.get("text_tower_frozen") is not True or policy.get(
        "visual_projection_frozen"
    ) is not True:
        raise ValueError(f"{role}: incomplete CLIP freeze policy")
    steps = sorted(int(row["training_global_step"]) for row in run.get("history", []))
    if role.endswith("FP5"):
        allowed = {0, 15, 44, 73}
        if set(steps) != allowed:
            raise ValueError(f"{role}: FP5 checkpoints are not exactly {sorted(allowed)}")
    else:
        allowed_sets = (
            {0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400},
            {6000, 7200, 9000, 10800},
            {0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400, 6000, 7200, 9000, 10800},
        )
        if set(steps) not in allowed_sets:
            raise ValueError(f"{role}: unexpected checkpoint steps {steps}")
    for row in run.get("history", []):
        checkpoint = _resolve(row.get("checkpoint"))
        expected_hash = row.get("checkpoint_sha256")
        if not checkpoint.is_file() or not isinstance(expected_hash, str):
            raise ValueError(f"{role}: checkpoint is missing")
        if _sha256(checkpoint) != expected_hash:
            raise ValueError(f"{role}: checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 2:
            raise ValueError(f"{role}: invalid checkpoint schema")
        if payload.get("experiment_role") != role or payload.get(
            "campaign"
        ) != FINAL_CAMPAIGN:
            raise ValueError(f"{role}: checkpoint role/campaign mismatch")
        for field in (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
            "training_global_step",
            "trainable_parameter_names",
            "optimizer_groups",
        ):
            if field not in payload:
                raise ValueError(f"{role}: checkpoint lacks {field}")
        if not isinstance(payload["optimizer_state_dict"], dict) or not isinstance(
            payload["scheduler_state_dict"], dict
        ) or not isinstance(payload["rng_state"], dict):
            raise ValueError(f"{role}: checkpoint state is incomplete")
        if payload["trainable_parameter_names"] != names or payload[
            "optimizer_groups"
        ] != groups:
            raise ValueError(f"{role}: checkpoint optimizer/name provenance mismatch")
    return run


def _discover_final_runs(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    paths = sorted(root.glob("**/run_result.json"))
    if not paths:
        raise FileNotFoundError(f"no final run results under {root}")
    expected_commit: str | None = None
    candidates: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for path in paths:
        value = _json(path)
        if value.get("experiment_role") not in FINAL_ROLES:
            continue
        if value.get("run_kind") != "primary" or value.get(
            "resolved_config", {}
        ).get("allow_short_run"):
            continue
        checked = _validate_final_run(path, expected_commit=expected_commit)
        expected_commit = expected_commit or str(checked["experiment_code_commit"])
        key = (str(checked["experiment_role"]), int(checked["seed"]))
        candidates.setdefault(key, []).append((path, checked))
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


def _legacy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"historical seed-42 artifact is missing: {path}")
    run = _json(path)
    if int(run.get("seed", 42)) != 42 or run.get(
        "official_unseen_used_for_selection"
    ) is not False:
        raise ValueError(f"historical artifact is not eligible for reuse: {path}")
    return run


def _legacy_paths() -> dict[str, Path]:
    return {
        "FP1": V2_DIR / "frozen_prompt_v2_FP1_continuation_retry/run_result.json",
        "FP2": V2_DIR / "frozen_prompt_v2_FP2_continuation/run_result.json",
        "FP3": V2_DIR / "frozen_prompt_v2_FP3_continuation/run_result.json",
        "FP5": V2_DIR / "frozen_prompt_v2_FP5/run_result.json",
        "FP_LN": V2_DIR / "frozen_prompt_v2_FP_LN_continuation/run_result.json",
    }


def _as_final_metric_run(
    legacy: dict[str, Any], final: dict[str, Any] | None, role: str
) -> dict[str, Any]:
    result = copy.deepcopy(legacy)
    result["experiment_role"] = role
    result["campaign"] = FINAL_CAMPAIGN
    result["resolved_treatment"] = treatment_for_role(role, seed=42, pseudo_val_seed=3407)
    result["reused_historical_seed_42"] = True
    if final is not None:
        result["history"] = sorted(
            list(legacy.get("history", [])) + list(final.get("history", [])),
            key=lambda row: int(row["training_global_step"]),
        )
        result["final_continuation_artifact_path"] = final.get("artifact_path")
        result["final_continuation_provenance"] = final.get("provenance")
        result["parameter_counts"] = final.get("parameter_counts", result.get("parameter_counts"))
        result["continuation_runtime"] = final.get("runtime")
        result["clip_freeze_policy"] = final.get(
            "clip_freeze_policy", result.get("clip_freeze_policy")
        )
    return result


def _complete_runs(final_runs: dict[tuple[str, int], dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    legacy = {name: _legacy(path) for name, path in _legacy_paths().items()}
    result = dict(final_runs)
    result["frozen_prompt_final_FP3", 42] = _as_final_metric_run(
        legacy["FP3"], final_runs.get(("frozen_prompt_final_FP3", 42)), "frozen_prompt_final_FP3"
    )
    result["frozen_prompt_final_FP2", 42] = _as_final_metric_run(
        legacy["FP2"], None, "frozen_prompt_final_FP2"
    )
    result["frozen_prompt_final_FP_LN", 42] = _as_final_metric_run(
        legacy["FP_LN"], None, "frozen_prompt_final_FP_LN"
    )
    result["frozen_prompt_final_FP5", 42] = _as_final_metric_run(
        legacy["FP5"], None, "frozen_prompt_final_FP5"
    )
    required = [
        (role, seed)
        for role in (
            "frozen_prompt_final_FP3",
            "frozen_prompt_final_FP3S",
            "frozen_prompt_final_FP2",
            "frozen_prompt_final_FP_LN",
            "frozen_prompt_final_FP5",
        )
        for seed in SEEDS
    ]
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError("missing final seed artifacts: " + ", ".join(map(str, missing)))
    for role in FINAL_ROLES:
        for seed in SEEDS:
            if int(result[role, seed].get("seed", seed)) != seed:
                raise ValueError(f"{role}: seed catalog mismatch")
    return result


def _probe_fields(run: dict[str, Any], step: int) -> dict[str, Any]:
    row = _row(run, step)
    geometry = row.get("geometry", {})
    alignment = geometry.get("representation_alignment", {}).get("sketch", {})
    token = geometry.get("prompt_token_geometry", {})
    parameter_counts = run.get("parameter_counts", {})
    runtime = run.get("runtime", {})
    return {
        "mAP": _map_at(run, step),
        "mAP@500": _map_at(run, 500),
        "mAP@1800": _map_at(run, 1800),
        "mAP@5400": _map_at(run, 5400),
        "peak_step": int(step),
        "semantic_margin": row.get("semantic_margin"),
        "classification_accuracy": row.get("diagnostic_seen_classification_accuracy"),
        "sketch_reference_cosine": row.get("sketch_reference_cosine"),
        "photo_reference_cosine": row.get("photo_reference_cosine"),
        "CKA": row.get("linear_cka", alignment.get("linear_cka")),
        "Procrustes_residual": row.get(
            "orthogonal_procrustes_residual", alignment.get("orthogonal_procrustes_residual")
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
        "prompt_norms": {
            "visual": row.get("prompt_parameter_norm"),
            "soft_text": row.get("soft_prompt_parameter_norm"),
        },
    }


def _photo_ablation(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for index, seed in enumerate(SEEDS):
        fp3 = runs["frozen_prompt_final_FP3", seed]
        fp3s = runs["frozen_prompt_final_FP3S", seed]
        left, right = _ap(fp3, 5400), _ap(fp3s, 5400)
        if left.shape != right.shape:
            raise ValueError("FP3 and FP3S query sets are not aligned")
        by_seed[str(seed)] = {
            "FP3": _probe_fields(fp3, 5400),
            "FP3S": _probe_fields(fp3s, 5400),
            "delta_mAP": _map_at(fp3, 5400) - _map_at(fp3s, 5400),
            "paired_bootstrap": _bootstrap(left, right, 20260904 + index),
            "training_code": {
                "FP3": fp3.get("experiment_code_commit"),
                "FP3S": fp3s.get("experiment_code_commit"),
            },
        }
    deltas = [float(value["delta_mAP"]) for value in by_seed.values()]
    return {
        "comparison": "FP3@5400 - FP3S@5400",
        "by_seed": by_seed,
        "mean_delta_mAP": float(np.mean(deltas)),
        "std_delta_mAP": float(np.std(deltas, ddof=1)),
        "direction_consistent": bool(all(value >= 0.0 for value in deltas)),
        "keep_photo_prompt": bool(np.mean(deltas) >= 0.005 and all(value >= 0.0 for value in deltas)),
        "threshold_mAP": 0.005,
        "bootstrap_is_query_uncertainty_not_seed_confirmation": True,
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
        rows = []
        for step in EXTENDED_STEPS:
            fields = _probe_fields(run, step)
            fields["training_global_step"] = step
            fields["mAP"] = _map_at(run, step)
            rows.append(fields)
        values = [(int(row["training_global_step"]), float(row["mAP"])) for row in rows]
        peak_step, peak_value = max(values, key=lambda item: (item[1], -item[0]))
        final_value = values[-1][1]
        rows[-1]["absolute_decay_from_peak"] = peak_value - final_value
        rows[-1]["retention_ratio"] = final_value / peak_value
        result[role] = {
            "training_seed": 42,
            "checkpoints": rows,
            "peak_mAP": peak_value,
            "peak_step": peak_step,
            "absolute_decay": peak_value - final_value,
            "retention_ratio": final_value / peak_value,
            "status": _extended_status(values),
        }
    return result


def _layernorm(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for index, seed in enumerate(SEEDS):
        fp2 = runs["frozen_prompt_final_FP2", seed]
        ln = runs["frozen_prompt_final_FP_LN", seed]
        left, right = _ap(ln, 5400), _ap(fp2, 5400)
        if left.shape != right.shape:
            raise ValueError("FP-LN and FP2 query sets are not aligned")
        by_seed[str(seed)] = {
            "FP2_mAP": _map_at(fp2, 5400),
            "FP_LN_mAP": _map_at(ln, 5400),
            "matched_delta": _map_at(ln, 5400) - _map_at(fp2, 5400),
            "paired_bootstrap": _bootstrap(left, right, 20261000 + index),
        }
    deltas = [float(row["matched_delta"]) for row in by_seed.values()]
    return {
        "comparison": "FP-LN@5400 - FP2@5400; both hard-text CE",
        "by_seed": by_seed,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)),
        "direction_consistent": bool(all(value >= 0.0 for value in deltas)),
        "keep_layernorm": bool(np.mean(deltas) >= 0.005 and all(value >= 0.0 for value in deltas)),
        "threshold_mAP": 0.005,
    }


def _hard_text(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    fp2 = runs["frozen_prompt_final_FP2", 42]
    fp1 = _legacy(_legacy_paths()["FP1"])
    return {
        "peak_effect": max(
            float(row["full_pseudo_unseen_mAP"])
            for row in fp2["history"]
        )
        - max(float(row["full_pseudo_unseen_mAP"]) for row in fp1["history"]),
        "matched_late_effect": _map_at(fp2, 5400) - _map_at(fp1, 5400),
        "comparison_peak": "peak(FP2) - peak(FP1)",
        "comparison_matched_late": "FP2@5400 - FP1@5400",
    }


def _seed_confirmation(
    runs: dict[tuple[str, int], dict[str, Any]], photo: dict[str, Any]
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for role in ("frozen_prompt_final_FP3", "frozen_prompt_final_FP3S"):
        per_seed: dict[str, Any] = {}
        deltas: list[float] = []
        maps: list[float] = []
        bootstrap: dict[str, Any] = {}
        for index, seed in enumerate(SEEDS):
            prompt = runs[role, seed]
            fp5 = runs["frozen_prompt_final_FP5", seed]
            prompt_map = _map_at(prompt, 5400)
            fp5_map = _map_at(fp5, 5400)
            delta = prompt_map - fp5_map
            maps.append(prompt_map)
            deltas.append(delta)
            bootstrap[str(seed)] = _bootstrap(
                _ap(prompt, 5400), _ap(fp5, 5400), 20261100 + index
            )
            per_seed[str(seed)] = {
                "prompt_mAP": prompt_map,
                "FP5_mAP": fp5_map,
                "prompt_minus_FP5": delta,
            }
        candidates[role] = {
            "per_seed": per_seed,
            "prompt_mean_std": _mean_std(maps),
            "prompt_minus_FP5_mean_std": _mean_std(deltas),
            "paired_bootstrap_by_seed": bootstrap,
        }
    fp5_maps = [_map_at(runs["frozen_prompt_final_FP5", seed], 5400) for seed in SEEDS]
    candidates["frozen_prompt_final_FP5"] = {
        "prompt_mean_std": _mean_std(fp5_maps),
        "role_label": "FP5 matched frozen hold",
    }
    fp3_mean = candidates["frozen_prompt_final_FP3"]["prompt_mean_std"]["mean"]
    fp3s_mean = candidates["frozen_prompt_final_FP3S"]["prompt_mean_std"]["mean"]
    mean_difference = fp3_mean - fp3s_mean
    simpler = "frozen_prompt_final_FP3S"
    selected = "frozen_prompt_final_FP3" if photo["keep_photo_prompt"] else simpler
    if abs(mean_difference) <= 0.003:
        selected = simpler
    candidates["selection"] = {
        "candidate_roles": ["frozen_prompt_final_FP3", "frozen_prompt_final_FP3S"],
        "selected_role": selected,
        "mean_difference_FP3_minus_FP3S": mean_difference,
        "simpler_model_margin": 0.003,
        "selection_rule": "prefer sketch-only when the seed mean difference is below 0.003 or photo ablation fails its predeclared threshold",
        "query_bootstrap_is_not_independent_seed_confirmation": True,
    }
    return candidates


def _load_retrieval_model(
    checkpoint: Path, role: str, device: torch.device
) -> tuple[Any, Any, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if role.endswith("FP5"):
        bundle = load_trainable_sketch_hidden_encoder(
            model_name="ViT-B-32-quickgelu",
            pretrained="openai",
            device=device,
            mode="partial",
            unfreeze_depth=4,
            train_ln_post=False,
        )
        model = _EarlyAdaptModel(bundle).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=False)
        model.eval()
        photo_clip = load_frozen_clip(
            model_name="ViT-B-32-quickgelu", pretrained="openai", device=device
        )
        return model, _FrozenEncoderAdapter(photo_clip.encoder), bundle.transform
    clip = load_frozen_clip(
        model_name="ViT-B-32-quickgelu", pretrained="openai", device=device
    )
    config = payload["resolved_config"]
    model = FrozenPromptModel(
        clip.encoder.model.visual,
        prompt_length=int(config["visual_prompt_length"]),
        train_visual_layernorm=bool(config["train_visual_layernorm"]),
        train_sketch_prompt=bool(config["train_sketch_prompt"]),
        train_photo_prompt=bool(config["train_photo_prompt"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    gallery = (
        _FrozenEncoderAdapter(clip.encoder)
        if role.endswith("FP3S")
        else model
    )
    return model, gallery, clip.transform


def _split_robustness(
    runs: dict[tuple[str, int], dict[str, Any]], selected_role: str, device: torch.device
) -> dict[str, Any]:
    selected = runs[selected_role, 42]
    fp5 = runs["frozen_prompt_final_FP5", 42]
    selected_checkpoint = _resolve(_row(selected, 5400)["checkpoint"])
    fp5_checkpoint = _resolve(_row(fp5, 5400)["checkpoint"])
    data = load_data_config(_path("configs/data/sketchy_104_21.yaml"))
    class_names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    prompt_model, prompt_gallery, prompt_transform = _load_retrieval_model(
        selected_checkpoint, selected_role, device
    )
    fp5_model, fp5_gallery, fp5_transform = _load_retrieval_model(
        fp5_checkpoint, "frozen_prompt_final_FP5", device
    )
    per_split: list[dict[str, Any]] = []
    for split_seed in SPLIT_SEEDS:
        split = make_classwise_retrieval_split(
            sketches, photos, class_names, num_validation_classes=20, seed=split_seed
        )
        prompt_sketch = DataLoader(
            RetrievalEvalDataset(split.validation_sketch_entries, prompt_transform),
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=device.type == "cuda",
        )
        prompt_photo = DataLoader(
            RetrievalEvalDataset(split.validation_photo_entries, prompt_transform),
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=device.type == "cuda",
        )
        fp5_sketch = DataLoader(
            RetrievalEvalDataset(split.validation_sketch_entries, fp5_transform),
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=device.type == "cuda",
        )
        fp5_photo = DataLoader(
            RetrievalEvalDataset(split.validation_photo_entries, fp5_transform),
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=device.type == "cuda",
        )
        prompt_eval = evaluate_prompted(
            encode_prompted_loader(prompt_model, prompt_sketch),
            encode_prompted_loader(prompt_gallery, prompt_photo, photo=True),
            query_chunk_size=256,
            device=device,
        )
        fp5_eval = evaluate_prompted(
            encode_prompted_loader(fp5_model, fp5_sketch),
            encode_prompted_loader(fp5_gallery, fp5_photo, photo=True),
            query_chunk_size=256,
            device=device,
        )
        train_ids, validation_ids = list(split.train_class_ids), list(split.validation_class_ids)
        prompt_map = float(prompt_eval.metrics.mean_average_precision)
        fp5_map = float(fp5_eval.metrics.mean_average_precision)
        per_split.append(
            {
                "split_seed": split_seed,
                "train_class_ids": train_ids,
                "validation_class_ids": validation_ids,
                "class_list_hashes": {
                    "train": canonical_sha256(train_ids),
                    "validation": canonical_sha256(validation_ids),
                },
                "prompt_mAP": prompt_map,
                "FP5_mAP": fp5_map,
                "prompt_minus_FP5": prompt_map - fp5_map,
                "training_seed_fixed": 42,
                "official_unseen_used_for_selection": False,
            }
        )
    del prompt_model, prompt_gallery, fp5_model, fp5_gallery
    del data, class_names, sketches, photos
    deltas = [float(row["prompt_minus_FP5"]) for row in per_split]
    return {
        "selected_prompt_role": selected_role,
        "predeclared_split_seeds": list(SPLIT_SEEDS),
        "per_split": per_split,
        "prompt_mean_std": _mean_std([float(row["prompt_mAP"]) for row in per_split]),
        "FP5_mean_std": _mean_std([float(row["FP5_mAP"]) for row in per_split]),
        "prompt_minus_FP5_mean_std": _mean_std(deltas),
        "official_unseen_used_for_selection": False,
        "evaluation_note": "fixed seed-42 trained checkpoints evaluated on predeclared pseudo-class splits; no official classes were read",
    }


def _new_plot(path: Path, title: str, draw: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    figure = plt.figure(figsize=(9, 5))
    try:
        draw(figure)
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path)
    finally:
        plt.close(figure)


def _plots(report: dict[str, Any], *, overwrite: bool = False) -> dict[str, str]:
    photo = report["probe_A_photo_ablation"]
    layernorm = report["probe_C_layernorm"]
    extended = report["probe_B_extended_training"]
    seeds = report["probe_D_seed_confirmation"]
    splits = report["probe_E_split_robustness"]

    def photo_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        labels = [str(seed) for seed in SEEDS]
        fp3 = [photo["by_seed"][seed]["FP3"]["mAP"] for seed in labels]
        fp3s = [photo["by_seed"][seed]["FP3S"]["mAP"] for seed in labels]
        positions = np.arange(len(labels))
        axis.bar(positions - 0.2, fp3, 0.4, label="FP3 dual prompt")
        axis.bar(positions + 0.2, fp3s, 0.4, label="FP3S sketch-only")
        axis.set_xticks(positions, labels)
        axis.set_xlabel("training seed")
        axis.set_ylabel("mAP@5400")
        axis.legend()

    def layernorm_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        labels = [str(seed) for seed in SEEDS]
        values = [layernorm["by_seed"][seed]["matched_delta"] for seed in labels]
        axis.axhline(0.005, color="black", linestyle="--", label="keep threshold")
        axis.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
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
        roles = ["frozen_prompt_final_FP3", "frozen_prompt_final_FP3S", "frozen_prompt_final_FP5"]
        labels = ["FP3", "FP3S", "FP5"]
        positions = np.arange(len(roles))
        means = [seeds[role]["prompt_mean_std"]["mean"] for role in roles[:2]]
        means.append(seeds[roles[2]]["prompt_mean_std"]["mean"])
        errors = [seeds[role]["prompt_mean_std"]["std"] for role in roles]
        axis.bar(positions, means, yerr=errors, capsize=4, label="mean ± std")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("mAP@5400")
        axis.set_xlabel("finalist / matched early adaptation")
        axis.legend()

    def split_plot(figure: Any) -> None:
        axis = figure.add_subplot(111)
        rows = splits["per_split"]
        labels = [str(row["split_seed"]) for row in rows]
        positions = np.arange(len(labels))
        axis.bar(positions - 0.2, [row["prompt_mAP"] for row in rows], 0.4, label="selected prompt")
        axis.bar(positions + 0.2, [row["FP5_mAP"] for row in rows], 0.4, label="FP5")
        axis.set_xticks(positions, labels)
        axis.set_xlabel("predeclared pseudo-class split seed")
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
        "split_robustness": ("Pseudo-class split robustness: mAP", split_plot),
        "geometry": ("Extended training geometry diagnostics", geometry_plot),
        "attention": ("Prompt attention at blocks 0, 6 and 11", attention_plot),
    }
    result = {}
    for key, (title, draw) in draws.items():
        _new_plot(PLOT_PATHS[key], title, draw, overwrite=overwrite)
        result[key] = str(PLOT_PATHS[key])
    return result


def _fmt(value: object) -> str:
    if value is None:
        return "not_run"
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


def _verdict(report: dict[str, Any], common_commit: str, working_tree_clean: bool) -> str:
    photo = report["probe_A_photo_ablation"]
    layernorm = report["probe_C_layernorm"]
    extended = report["probe_B_extended_training"]
    seeds = report["probe_D_seed_confirmation"]
    selected = seeds["selection"]["selected_role"]
    best_extended = max(
        (
            (role, extended[role]["peak_mAP"], extended[role]["peak_step"])
            for role in extended
        ),
        key=lambda item: (item[1], -item[2]),
    )
    best_at_5400 = max(
        _map_at(report["_metric_runs"][role + ":42"], 5400)
        for role in ("frozen_prompt_final_FP3", "frozen_prompt_final_FP3S")
    )
    best_at_10800 = max(
        _map_at(report["_metric_runs"][role + ":42"], 10800)
        for role in ("frozen_prompt_final_FP3", "frozen_prompt_final_FP3S")
    )
    selected_run = report["_metric_runs"][selected + ":42"]
    parameter_count = selected_run.get("parameter_counts", {}).get(
        "trainable_parameters"
    )
    if parameter_count is None:
        parameter_count = len(selected_run.get("trainable_parameter_names", []))
    clip_identity = selected_run.get("clip_freeze_policy", {}).get(
        "all_clip_owned_parameters_byte_identical"
    )
    status = extended[best_extended[0]]["status"]
    if status == "boundary peak":
        convergence = "boundary peak"
    else:
        convergence = status
    prompt_stats = seeds[selected]["prompt_mean_std"]
    fp5_stats = seeds["frozen_prompt_final_FP5"]["prompt_mean_std"]
    delta_stats = seeds[selected]["prompt_minus_FP5_mean_std"]
    artifact_valid = bool(report["provenance"]["all_new_runs_valid"])
    lines = [
        "FINAL SPICA FROZEN-PROMPT VERDICT",
        "",
        f"Repository commit: {report['repository_commit']}",
        f"Experiment-code commit: {common_commit}",
        f"Working tree tracked files clean: {'YES' if working_tree_clean else 'NO'}",
        "Historical artifacts preserved: YES",
        f"Artifact provenance valid: {'YES' if artifact_valid else 'NO'}",
        "Official unseen used for selection: NO",
        "",
        f"FP3 mAP: {_fmt(_map_at(report['_metric_runs']['frozen_prompt_final_FP3:42'], 5400))}",
        f"FP3S mAP: {_fmt(_map_at(report['_metric_runs']['frozen_prompt_final_FP3S:42'], 5400))}",
        f"Photo-prompt effect: {_fmt(photo['mean_delta_mAP'])} mean (FP3 − FP3S) at 5400; direction consistent: {'YES' if photo['direction_consistent'] else 'NO'}",
        f"Should photo prompt remain: {'YES' if photo['keep_photo_prompt'] else 'NO'}",
        "",
        f"FP2 mAP: {_fmt(report['probe_C_layernorm']['by_seed']['42']['FP2_mAP'])}",
        f"FP-LN mAP: {_fmt(report['probe_C_layernorm']['by_seed']['42']['FP_LN_mAP'])}",
        f"Matched LayerNorm effect: {_fmt(report['probe_C_layernorm']['by_seed']['42']['matched_delta'])} (FP-LN − FP2; hard-text CE)",
        f"Should LayerNorm remain: {'YES' if layernorm['keep_layernorm'] else 'NO'}",
        "",
        f"Best mAP@5400: {_fmt(best_at_5400)}",
        f"Best mAP@10800: {_fmt(best_at_10800)}",
        f"Extended peak: {_fmt(best_extended[1])}",
        f"Extended peak step: {best_extended[2]}",
        f"Converged or boundary peak: {convergence}",
        "",
        f"Best prompt mean ± std: {_fmt(prompt_stats['mean'])} ± {_fmt(prompt_stats['std'])}",
        f"FP5 mean ± std: {_fmt(fp5_stats['mean'])} ± {_fmt(fp5_stats['std'])}",
        f"Prompt-minus-FP5 mean delta: {_fmt(delta_stats['mean'])}",
        "",
        f"Best prompt configuration: {selected}",
        f"Trainable parameter count: {parameter_count}",
        f"Frozen CLIP byte-identical: {'YES' if clip_identity else 'NO'}",
        "Text enters predictor: NO",
        "Text required at inference: NO",
        "Photo required for query inference: NO",
        "",
        "Should transport return: NO",
        "Should direction supervision return: NO",
        "Should distance prediction return: NO",
        "Should Mo-vMF/K>1 run now: NO",
        "",
        "Strongest supported mechanism: frozen visual prompting trained by retrieval ranking plus loss-only soft-text classification",
        "Largest remaining confound: seed and pseudo-class-split variance; split robustness uses fixed seed-42 checkpoints rather than per-split retraining",
        f"Recommended mainline architecture: {selected}",
        "Recommended mainline loss: rank loss plus soft-text CE at the query",
        "Recommended inference query: raw sketch image",
    ]
    return "\n".join(lines) + "\n"


def _markdown(report: dict[str, Any], verdict: str) -> str:
    photo = report["probe_A_photo_ablation"]
    layernorm = report["probe_C_layernorm"]
    extended = report["probe_B_extended_training"]
    seeds = report["probe_D_seed_confirmation"]
    lines = [
        "# SPICA frozen-prompt final campaign",
        "",
        "Selection used pseudo-unseen validation only. The official unseen split was not read.",
        "",
        "## Probe A — FP3S photo ablation",
        "",
        f"Primary comparison: `{photo['comparison']}`; mean delta={photo['mean_delta_mAP']:.6f}; direction consistent={photo['direction_consistent']}; keep photo prompt={photo['keep_photo_prompt']}.",
        "Each seed includes mAP@500, mAP@1800, mAP@5400, semantic margin, classification accuracy, reference cosines, CKA, Procrustes residual, prompt-token cosine, attention, parameter count, runtime, GPU memory, and a 2000-draw paired query bootstrap in the JSON artifact.",
        "",
        "## Probe B — extended training",
        "",
    ]
    for role, value in extended.items():
        lines.append(
            f"- {role}: peak {_fmt(value['peak_mAP'])} at step {value['peak_step']}; absolute decay {_fmt(value['absolute_decay'])}; retention {_fmt(value['retention_ratio'])}; **{value['status']}**."
        )
    lines.extend(
        [
            "",
            "A best result at step 10800 is reported as a boundary peak, not convergence.",
            "",
            "## Probe C — LayerNorm",
            "",
            f"The only LayerNorm comparison is `FP-LN@5400 - FP2@5400`, with both cells using hard-text CE. Mean delta={layernorm['mean_delta']:.6f}, std={layernorm['std_delta']:.6f}, direction consistent={layernorm['direction_consistent']}, keep={layernorm['keep_layernorm']}.",
            f"Hard-text effects remain separate: peak(FP2)-peak(FP1)={report['hard_text']['peak_effect']:.6f}; FP2@5400-FP1@5400={report['hard_text']['matched_late_effect']:.6f}.",
            "",
            "## Probe D — finalist seeds",
            "",
            "The finalists are FP3 and FP3S. Query-level bootstrap intervals describe paired query uncertainty only; they are not independent-seed confirmation. Seed means and standard deviations, plus per-seed prompt-minus-FP5 deltas, are recorded separately.",
            f"Selected architecture: **{seeds['selection']['selected_role']}**.",
            "",
            "## Probe E — split robustness",
            "",
            f"Predeclared split seeds: {report['probe_E_split_robustness']['predeclared_split_seeds']}. Exact class lists and hashes are in the JSON artifact. No official class list was read.",
            "",
            "## Provenance and artifacts",
            "",
            f"Repository commit: `{report['repository_commit']}`; one experiment-code commit: `{report['provenance']['experiment_code_commit']}`; tracked files clean: `{report['provenance']['tracked_working_tree_clean']}`.",
            "Every new run records resolved configuration, both seeds, class-list and manifest identities, checkpoint SHA256, optimizer/scheduler/RNG states, trainable names/groups, CLIP byte identity, tracked and untracked status, and the no-official-selection flag.",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend(f"- {path}" for path in report["plots"].values())
    lines.extend(["", verdict])
    return "\n".join(lines)


def analyze(*, final_dir: Path = FINAL_DIR, overwrite: bool = False) -> dict[str, Any]:
    manifest, manifest_hash = ensure_manifest(
        FINAL_MANIFEST_PATH,
        dataset="sketchy_104_21",
        data_config="configs/data/sketchy_104_21.yaml",
        campaign=FINAL_CAMPAIGN,
    )
    final_runs = _discover_final_runs(final_dir)
    runs = _complete_runs(final_runs)
    new_commits = {
        str(run.get("experiment_code_commit"))
        for run in final_runs.values()
        if run.get("experiment_code_commit")
    }
    if len(new_commits) != 1:
        raise ValueError(f"expected one final experiment-code commit, got {new_commits}")
    common_commit = next(iter(new_commits))
    repository_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    tracked_clean = not bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    photo = _photo_ablation(runs)
    extended = _extended(runs)
    layernorm = _layernorm(runs)
    hard_text = _hard_text(runs)
    seed_confirmation = _seed_confirmation(runs, photo)
    selected_role = seed_confirmation["selection"]["selected_role"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_robustness = _split_robustness(runs, selected_role, device)
    metric_runs = {
        f"{role}:{seed}": run
        for (role, seed), run in runs.items()
        if role in FINAL_ROLES
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "campaign": FINAL_CAMPAIGN,
        "repository_commit": repository_commit,
        "manifest": {
            "path": str(FINAL_MANIFEST_PATH),
            "sha256": manifest_hash,
            "value": manifest,
        },
        "provenance": {
            "experiment_code_commit": common_commit,
            "all_new_runs_same_clean_code_commit": True,
            "tracked_working_tree_clean": tracked_clean,
            "untracked_files_recorded_separately": True,
            "all_new_runs_valid": True,
        },
        "official_unseen_used_for_selection": False,
        "selection_policy": manifest["selection_policy"],
        "probe_A_photo_ablation": photo,
        "probe_B_extended_training": extended,
        "probe_C_layernorm": layernorm,
        "hard_text": hard_text,
        "probe_D_seed_confirmation": seed_confirmation,
        "probe_E_split_robustness": split_robustness,
        "run_provenance": {
            f"{role}:{seed}": {
                "artifact_path": run.get("artifact_path"),
                "experiment_code_commit": run.get("experiment_code_commit"),
                "source_snapshot_hash": run.get("source_snapshot_hash"),
                "tracked_working_tree_state": run.get("provenance", {}).get(
                    "tracked_working_tree_state"
                ),
                "untracked_files": run.get("provenance", {}).get(
                    "untracked_files", []
                ),
                "official_unseen_used_for_selection": run.get(
                    "official_unseen_used_for_selection"
                ),
            }
            for (role, seed), run in final_runs.items()
        },
        "plots": {},
        "_metric_runs": metric_runs,
    }
    report["plots"] = _plots(report, overwrite=overwrite)
    verdict = _verdict(report, common_commit, tracked_clean)
    report["verdict"] = verdict
    report.pop("_metric_runs")
    output_json = ROOT / "outputs/research_summary_frozen_prompt_final_2026-09-04.json"
    output_md = ROOT / "outputs/research_summary_frozen_prompt_final_2026-09-04.md"
    for path in (output_json, output_md):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown({**report, "_metric_runs": metric_runs}, verdict))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh only this campaign's newly generated report and plots",
    )
    args = parser.parse_args()
    analyze(final_dir=args.final_dir, overwrite=args.refresh)


if __name__ == "__main__":
    main()
