"""Fail-closed fixed-step and peak reports for alignment campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

TREATMENT_ONLY_KEYS = {
    "visual_prompt_length",
    "prompt_mode",
    "text_mode",
    "train_visual_layernorm",
    "train_sketch_prompt",
    "train_photo_prompt",
    "lambda_rank",
    "lambda_cls",
    "classification_location",
    "encoder_mode",
    "encoder_unfreeze_depth",
    "encoder_train_ln_post",
    "transport_enabled",
    "transport_mode",
    "num_positive_photos",
    "batch_size",
    "classes_per_batch",
    "sketches_per_class",
    "alignment_geometry",
    "alignment_anchor",
    "alignment_target_gradient",
    "lambda_alignment_mean",
    "lambda_alignment_covariance",
    "seed",
    "pseudo_val_seed",
    "official_unseen_used_for_selection",
}
NON_TRAINING_CONFIG_KEYS = {"alignment_calibration_artifact"}
PROTOCOL_KEYS = (
    "selection_metric",
    "train_class_scope",
    "alignment_fit_scope",
    "validation_used_for_alignment",
    "test_used_for_alignment",
    "text_used_for_predictor",
    "photo_used_for_predictor",
    "ranking_positive_reduction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolve_path(value: object) -> Path:
    return Path(str(value)).expanduser()


def _runs(campaign_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(campaign_dir.glob("**/run_result.json")):
        result = json.loads(path.read_text())
        result["_artifact_path"] = str(path.resolve())
        found.append(result)
    return found


def _step0_metadata(result: dict[str, Any]) -> dict[str, Any]:
    history = result.get("history", [])
    rows = [row for row in history if int(row.get("training_global_step", -1)) == 0]
    if not rows:
        return {"status": "UNVERIFIED_NO_STEP0"}
    checkpoint = _resolve_path(rows[0].get("checkpoint", ""))
    if not checkpoint.is_file():
        return {"status": "MISSING_STEP0_CHECKPOINT", "checkpoint": str(checkpoint)}
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - artifact is an external input
        return {"status": "UNREADABLE_STEP0_CHECKPOINT", "error": str(error)}
    if not isinstance(payload, dict):
        return {"status": "INVALID_STEP0_CHECKPOINT"}
    return {
        "status": "VERIFIED",
        "checkpoint": str(checkpoint),
        "initial_model_state_hash": payload.get("initial_model_state_hash"),
        "source_snapshot_hash": payload.get("source_snapshot_hash"),
        "backbone_identity": payload.get("backbone_identity"),
    }


def _protocol_identity(result: dict[str, Any]) -> dict[str, Any]:
    protocol = result.get("protocol", {})
    return {key: protocol.get(key) for key in PROTOCOL_KEYS}


def _validate(result: dict[str, Any], requested_horizon: int) -> dict[str, Any]:
    artifact_path = Path(result["_artifact_path"])
    history = result.get("history", [])
    errors: list[str] = []
    if result.get("official_unseen_used_for_selection") is not False:
        errors.append("official unseen data was used for selection")
    protocol = result.get("protocol", {})
    for key in (
        "validation_used_for_alignment",
        "test_used_for_alignment",
        "text_used_for_predictor",
        "photo_used_for_predictor",
    ):
        if protocol.get(key) is not False:
            errors.append(f"protocol violation: {key}")
    if not history:
        errors.append("run has no history")
    steps: list[int] = []
    valid_rows: list[dict[str, Any]] = []
    hash_checks: list[dict[str, Any]] = []
    for row in history:
        try:
            step = int(row["training_global_step"])
            checkpoint = _resolve_path(row["checkpoint"])
            expected_hash = str(row.get("checkpoint_sha256", ""))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid history row: {error}")
            continue
        steps.append(step)
        if not checkpoint.is_file():
            hash_checks.append({"step": step, "status": "MISSING", "path": str(checkpoint)})
            errors.append(f"missing checkpoint at step {step}: {checkpoint}")
            continue
        actual_hash = _sha256(checkpoint)
        hash_status = "VERIFIED" if expected_hash == actual_hash else "MISMATCH"
        hash_checks.append(
            {
                "step": step,
                "path": str(checkpoint),
                "metadata_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": hash_status,
            }
        )
        if not expected_hash:
            errors.append(f"checkpoint hash is missing at step {step}")
        elif expected_hash != actual_hash:
            errors.append(f"checkpoint hash mismatch at step {step}")
        if hash_status != "VERIFIED":
            continue
        try:
            value = float(row["full_pseudo_unseen_mAP"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"raw mAP is missing at step {step}")
            continue
        if not (value == value and abs(value) != float("inf")):
            errors.append(f"raw mAP is non-finite at step {step}")
            continue
        valid_rows.append({"step": step, "mAP": value, "checkpoint": str(checkpoint)})
    duplicate_steps = sorted({step for step in steps if steps.count(step) > 1})
    if duplicate_steps:
        errors.append(f"duplicate history steps: {duplicate_steps}")
    resolved = result.get("resolved_config", {})
    configured_horizon = resolved.get("max_steps")
    if configured_horizon is None or int(configured_horizon) != requested_horizon:
        errors.append(
            f"configured horizon {configured_horizon!r} != requested {requested_horizon}"
        )
    fixed_rows = [row for row in valid_rows if row["step"] == requested_horizon]
    if not fixed_rows:
        fixed_status = "INCOMPLETE"
        fixed_value = None
    elif len(fixed_rows) > 1:
        fixed_status = "DUPLICATE"
        fixed_value = None
    else:
        fixed_status = "VALID"
        fixed_value = fixed_rows[0]["mAP"]
    peak_row = min(
        valid_rows,
        key=lambda row: (-row["mAP"], row["step"]),
        default=None,
    )
    peak_value = None if peak_row is None else peak_row["mAP"]
    peak_step = None if peak_row is None else peak_row["step"]
    initialization = _step0_metadata(result)
    if initialization.get("status") != "VERIFIED":
        errors.append(
            f"step-0 initialization is not verified: {initialization.get('status')}"
        )
    source_hash = result.get("source_snapshot_hash")
    if source_hash is None:
        source_hash = result.get("provenance", {}).get("source_snapshot", {}).get("sha256")
    if not source_hash:
        errors.append("source snapshot hash is missing")
    split = result.get("pseudo_split_identity", {})
    if not isinstance(split, dict) or not split.get("sha256"):
        errors.append("pseudo split identity hash is missing")
    split_hash = (
        split.get("sha256") or _canonical_hash(split)
        if isinstance(split, dict)
        else _canonical_hash(split)
    )
    treatment = result.get("resolved_treatment", {})
    config_hash = _canonical_hash(resolved)
    seed = result.get("training_seed", result.get("seed"))
    if seed is None:
        errors.append("training seed is missing")
    run_status = "VALID"
    if any("mismatch" in error or "missing checkpoint" in error for error in errors):
        run_status = "ARTIFACT_INVALID"
    elif errors:
        run_status = "INCOMPLETE"
    if run_status != "VALID":
        fixed_value = None
        fixed_status = "INCOMPLETE"
        peak_value = None
        peak_step = None
    return {
        "artifact_path": str(artifact_path),
        "campaign": result.get("campaign"),
        "experiment_role": result.get("experiment_role"),
        "run_kind": result.get("run_kind"),
        "training_seed": None if seed is None else int(seed),
        "split_identity": split,
        "split_identity_hash": split_hash,
        "initialization": initialization,
        "initialization_regime": initialization.get("initial_model_state_hash", "UNVERIFIED"),
        "source_hash": source_hash,
        "config_hash": config_hash,
        "resolved_config": resolved,
        "resolved_treatment": treatment,
        "evaluation_protocol": _protocol_identity(result),
        "evaluation_protocol_hash": _canonical_hash(_protocol_identity(result)),
        "training_horizon": configured_horizon,
        "requested_horizon": requested_horizon,
        "status": run_status,
        "errors": errors,
        "hash_checks": hash_checks,
        "history_steps": sorted(steps),
        "fixed_step": {"status": fixed_status, "mAP": fixed_value},
        "peak": {"mAP": peak_value, "step": peak_step},
        "retention": None
        if fixed_value is None or peak_value in (None, 0)
        else fixed_value / peak_value,
        "absolute_decay": None
        if fixed_value is None or peak_value is None
        else peak_value - fixed_value,
    }


def _matching_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        run["campaign"],
        run["training_seed"],
        run["split_identity_hash"],
        run["initialization_regime"],
        run["training_horizon"],
        run["evaluation_protocol_hash"],
        run["run_kind"],
    )


def _config_differences(
    control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    keys = (
        set(control["resolved_config"]) | set(candidate["resolved_config"])
    ) - NON_TRAINING_CONFIG_KEYS
    return {
        key: (
            control["resolved_config"].get(key),
            candidate["resolved_config"].get(key),
        )
        for key in sorted(keys)
        if control["resolved_config"].get(key) != candidate["resolved_config"].get(key)
    }


def _provenance_complete(run: dict[str, Any]) -> bool:
    split = run.get("split_identity", {})
    protocol = run.get("evaluation_protocol", {})
    return bool(
        run.get("source_hash")
        and run.get("training_seed") is not None
        and run.get("training_horizon") is not None
        and isinstance(split, dict)
        and split.get("sha256")
        and run.get("initialization_regime") not in {None, "UNVERIFIED"}
        and all(value is not None for value in protocol.values())
    )


def _pair(
    candidate: dict[str, Any], controls: list[dict[str, Any]]
) -> dict[str, Any]:
    same_identity = [control for control in controls if _matching_key(control) == _matching_key(candidate)]
    if not same_identity:
        return {
            "status": "UNMATCHED",
            "control_artifact": None,
            "control_seed": None,
            "paired_delta": None,
            "reason": "no control with identical campaign/seed/split/init/horizon/protocol",
        }
    same_identity.sort(key=lambda run: run["artifact_path"])
    control = same_identity[0]
    duplicate = len(same_identity) > 1
    differences = _config_differences(control, candidate)
    allowed = TREATMENT_ONLY_KEYS | {"experiment_role", "experiment_name"}
    disallowed = sorted(set(differences) - allowed)
    if duplicate:
        return {
            "status": "DUPLICATE_CONTROL",
            "control_artifact": control["artifact_path"],
            "control_seed": control["training_seed"],
            "paired_delta": None,
            "peak_delta": None,
            "duplicate_control_count": len(same_identity),
            "reason": "ambiguous duplicate controls; delta suppressed",
            "config_differences": differences,
            "disallowed_config_differences": disallowed,
            "source_hashes": [run["source_hash"] for run in same_identity]
            + [candidate["source_hash"]],
        }
    if not _provenance_complete(control) or not _provenance_complete(candidate):
        return {
            "status": "UNMATCHED_PROVENANCE",
            "control_artifact": control["artifact_path"],
            "control_seed": control["training_seed"],
            "paired_delta": None,
            "peak_delta": None,
            "reason": "source/split/initialization/protocol evidence is incomplete",
            "config_differences": differences,
            "disallowed_config_differences": disallowed,
            "source_hashes": [control["source_hash"], candidate["source_hash"]],
        }
    if control["status"] == "ARTIFACT_INVALID" or candidate["status"] == "ARTIFACT_INVALID":
        return {
            "status": "ARTIFACT_INVALID",
            "control_artifact": control["artifact_path"],
            "control_seed": control["training_seed"],
            "paired_delta": None,
            "peak_delta": None,
            "reason": "invalid run artifact; delta suppressed",
            "config_differences": differences,
            "disallowed_config_differences": disallowed,
            "source_hashes": [control["source_hash"], candidate["source_hash"]],
        }
    if control["status"] != "VALID" or candidate["status"] != "VALID":
        return {
            "status": "INCOMPLETE",
            "control_artifact": control["artifact_path"],
            "control_seed": control["training_seed"],
            "paired_delta": None,
            "peak_delta": None,
            "reason": "run is incomplete; delta suppressed",
            "config_differences": differences,
            "disallowed_config_differences": disallowed,
            "source_hashes": [control["source_hash"], candidate["source_hash"]],
        }
    if control["source_hash"] != candidate["source_hash"]:
        return {
            "status": "UNMATCHED_SOURCE",
            "control_artifact": control["artifact_path"],
            "control_seed": control["training_seed"],
            "paired_delta": None,
            "reason": "source snapshot hashes differ",
            "source_hashes": [control["source_hash"], candidate["source_hash"]],
            "config_differences": differences,
            "disallowed_config_differences": disallowed,
        }
    if disallowed:
        return {
            "status": "UNMATCHED_CONFIG",
            "control_artifact": control["artifact_path"],
            "control_seed": control["training_seed"],
            "paired_delta": None,
            "reason": "non-treatment config differences",
            "config_differences": differences,
            "disallowed_config_differences": disallowed,
        }
    if control["fixed_step"]["mAP"] is None or candidate["fixed_step"]["mAP"] is None:
        status = "INCOMPLETE"
        delta = None
    else:
        status = "MATCHED"
        delta = candidate["fixed_step"]["mAP"] - control["fixed_step"]["mAP"]
    peak_delta = None
    if control["peak"]["mAP"] is not None and candidate["peak"]["mAP"] is not None:
        peak_delta = candidate["peak"]["mAP"] - control["peak"]["mAP"]
    return {
        "status": status,
        "control_artifact": control["artifact_path"],
        "control_seed": control["training_seed"],
        "paired_delta": delta,
        "peak_delta": peak_delta,
        "duplicate_control_count": len(same_identity),
        "config_differences": differences,
        "disallowed_config_differences": disallowed,
        "source_hashes": [control["source_hash"], candidate["source_hash"]],
    }


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": None if not values else statistics.fmean(values),
        "sample_std": None if len(values) < 2 else statistics.stdev(values),
        "values": values,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _report_campaign(runs: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    controls = [run for run in runs if run["experiment_role"] == "alignment_control"]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_role.setdefault(str(run["experiment_role"]), []).append(run)
    for values in by_role.values():
        values.sort(key=lambda run: run["artifact_path"])
    pairs: dict[str, list[dict[str, Any]]] = {}
    for role, values in by_role.items():
        pairs[role] = []
        if role == "alignment_control":
            continue
        for run in values:
            pairs[role].append(
                {
                    "artifact_path": run["artifact_path"],
                    "training_seed": run["training_seed"],
                    **_pair(run, controls),
                }
            )
    summaries: list[dict[str, Any]] = []
    for role in sorted(by_role):
        values = by_role[role]
        deltas = [
            pair["paired_delta"]
            for pair in pairs.get(role, [])
            if pair.get("paired_delta") is not None
        ]
        summaries.append(
            {
                "role": role,
                "run_count": len(values),
                "unique_training_seeds": sorted(
                    {run["training_seed"] for run in values if run["training_seed"] is not None}
                ),
                "fixed_mAP_mean": _stats(
                    [run["fixed_step"]["mAP"] for run in values if run["fixed_step"]["mAP"] is not None]
                ),
                "peak_mAP_mean": _stats(
                    [run["peak"]["mAP"] for run in values if run["peak"]["mAP"] is not None]
                ),
                "paired_fixed_delta": _stats(deltas),
                "missing_or_incomplete_runs": [
                    run["artifact_path"]
                    for run in values
                    if run["status"] != "VALID" or run["fixed_step"]["status"] != "VALID"
                ],
            }
        )
    return {
        "campaign": runs[0]["campaign"] if runs else None,
        "requested_horizon": horizon,
        "run_count": len(runs),
        "runs": runs,
        "pairs": pairs,
        "summaries": summaries,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Corrected alignment historical results",
        "",
        "All values below are recomputed from raw `run_result.json` histories; no Markdown value was copied into recomputed fields.",
        "Fixed-step and peak tables are separate. Matching fails closed on campaign, seed, split, initialization, horizon, protocol, source, and non-treatment config differences.",
        "",
    ]
    for campaign in payload["campaigns"]:
        lines += [
            f"## Campaign `{campaign['campaign']}` — requested horizon `{campaign['requested_horizon']}`",
            "",
            "### Fixed-step per run",
            "",
            "| Role | Seed | Status | mAP@horizon | Paired delta |",
            "|---|---:|---|---:|---:|",
        ]
        pair_by_artifact = {
            (role, pair["artifact_path"]): pair
            for role, pairs in campaign["pairs"].items()
            for pair in pairs
        }
        for run in campaign["runs"]:
            pair = pair_by_artifact.get((run["experiment_role"], run["artifact_path"]))
            delta = None if pair is None else pair.get("paired_delta")
            status = run["fixed_step"]["status"]
            if pair is not None and pair["status"] != "MATCHED":
                status = pair["status"]
            lines.append(
                f"| `{run['experiment_role']}` | {run['training_seed']} | {status} | "
                f"{_fmt(run['fixed_step']['mAP'])} | {_fmt(delta)} |"
            )
        lines += [
            "",
            "### Peak per run",
            "",
            "| Role | Seed | Peak mAP | Peak step | Retention | Absolute decay | Peak delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for run in campaign["runs"]:
            pair = pair_by_artifact.get((run["experiment_role"], run["artifact_path"]))
            lines.append(
                f"| `{run['experiment_role']}` | {run['training_seed']} | "
                f"{_fmt(run['peak']['mAP'])} | {run['peak']['step'] or '—'} | "
                f"{_fmt(run['retention'])} | {_fmt(run['absolute_decay'])} | "
                f"{_fmt(None if pair is None else pair.get('peak_delta'))} |"
            )
        lines += ["", "### Aggregate", "", "| Role | Unique seeds | Paired delta mean | Sample std | n |", "|---|---:|---:|---:|---:|"]
        for summary in campaign["summaries"]:
            stats = summary["paired_fixed_delta"]
            lines.append(
                f"| `{summary['role']}` | {summary['unique_training_seeds']} | "
                f"{_fmt(stats['mean'])} | {_fmt(stats['sample_std'])} | {stats['count']} |"
            )
        lines += ["", "### Matching/provenance notes", ""]
        for run in campaign["runs"]:
            if run["errors"]:
                lines.append(
                    f"- `{run['artifact_path']}`: { '; '.join(run['errors']) }"
                )
        for role, pairs in campaign["pairs"].items():
            for pair in pairs:
                if pair["status"] not in {"MATCHED", "INCOMPLETE"}:
                    lines.append(
                        f"- `{role}` seed {pair['training_seed']}: {pair['status']} — {pair.get('reason', '')}"
                    )
        lines.append("")
    return "\n".join(lines)


def write_report(
    campaign_dir: Path,
    output_path: Path,
    *,
    horizon: int,
    campaigns: set[str] | None = None,
    include_smoke: bool = False,
) -> dict[str, Any]:
    discovered = _runs(campaign_dir)
    if campaigns is not None:
        discovered = [run for run in discovered if run.get("campaign") in campaigns]
    if not include_smoke:
        discovered = [run for run in discovered if run.get("run_kind") != "smoke"]
    if not discovered:
        raise ValueError(f"no matching run_result.json files found below {campaign_dir}")
    validated = [_validate(result, horizon) for result in discovered]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in validated:
        grouped.setdefault(str(run["campaign"]), []).append(run)
    campaigns = [
        _report_campaign(values, horizon)
        for _, values in sorted(grouped.items())
    ]
    payload = {
        "schema_version": 2,
        "requested_horizon": horizon,
        "campaign_root": str(campaign_dir.resolve()),
        "matching_policy": {
            "same_seed_only": True,
            "fallback_control_mean": False,
            "duplicate_rule": "ambiguous duplicate controls are reported and paired deltas are suppressed",
            "horizon_rule": "explicit --horizon; missing final step is INCOMPLETE",
            "allowed_config_differences": sorted(TREATMENT_ONLY_KEYS | {"experiment_role", "experiment_name"}),
            "ignored_non_training_config_keys": sorted(NON_TRAINING_CONFIG_KEYS),
            "primary_excludes_smoke_pilot": "campaigns are reported separately; no cross-campaign matching",
        },
        "campaigns": campaigns,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown(payload))
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--campaign", action="append", dest="campaigns")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    write_report(
        args.campaign_dir,
        args.output,
        horizon=args.horizon,
        campaigns=None if args.campaigns is None else set(args.campaigns),
        include_smoke=args.include_smoke,
    )


if __name__ == "__main__":
    main()
