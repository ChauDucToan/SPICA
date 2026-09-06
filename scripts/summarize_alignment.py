"""Fail-closed fixed-step and peak reports for alignment campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from spica.alignment_artifacts import (
    ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
    CORRECTED_MANIFEST_MARKER,
    CORRECTED_PILOT_ROLES,
    corrected_pilot_inference_contract_mismatches,
    manifest_identity_sha256,
    treatment_for_role,
)
from spica.config.data import load_data_config
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split, split_manifest_identity

CALIBRATION_CONFIG_KEY = "alignment_calibration_artifact"
CALIBRATION_SCHEMA_VERSION = 2
CALIBRATION_SKETCH_TOLERANCE = 1e-6
CALIBRATION_RULE = (
    "lambda_alignment_mean = target_ratio * median("
    "base.sketch_gradient_norms / detached.sketch_gradient_norms)"
)

# Historical reports intentionally retain their old, broad treatment policy.
# The corrected pilot below uses pair-specific rules instead.
HISTORICAL_ALIGNMENT_ALLOWED_CONFIG_KEYS = frozenset(
    {
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
)

CORRECTED_PILOT_PAIR_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("alignment_control", "alignment_mean_text_log"): {
        "pair_id": "R-MD",
        "allowed_config_differences": frozenset(
            {"experiment_role", "experiment_name", CALIBRATION_CONFIG_KEY, "lambda_alignment_mean"}
        ),
    },
    ("alignment_control", "alignment_mean_text_log_symmetric"): {
        "pair_id": "R-MS",
        "allowed_config_differences": frozenset(
            {
                "experiment_role",
                "experiment_name",
                CALIBRATION_CONFIG_KEY,
                "lambda_alignment_mean",
                "alignment_target_gradient",
            }
        ),
    },
    ("alignment_mean_text_log", "alignment_mean_text_log_symmetric"): {
        "pair_id": "MD-MS",
        "allowed_config_differences": frozenset(
            {"experiment_role", "experiment_name", "alignment_target_gradient"}
        ),
        "equal_config_keys": frozenset({"lambda_alignment_mean"}),
    },
}

# These are the resolved fields that must be present before a corrected pair is
# called matched. Other resolved fields are also compared when present.
CORRECTED_PILOT_REQUIRED_CONFIG_KEYS = (
    "experiment_campaign",
    "experiment_role",
    "run_kind",
    "experiment_manifest_path",
    "data_config",
    "model_name",
    "pretrained",
    "visual_prompt_length",
    "prompt_mode",
    "text_mode",
    "soft_prompt_length",
    "prompt_template",
    "train_visual_layernorm",
    "train_sketch_prompt",
    "train_photo_prompt",
    "classification_location",
    "encoder_mode",
    "encoder_unfreeze_depth",
    "encoder_train_ln_post",
    "transport_enabled",
    "transport_mode",
    "batch_size",
    "classes_per_batch",
    "sketches_per_class",
    "num_positive_photos",
    "lambda_rank",
    "lambda_cls",
    "margin",
    "tau_cls",
    "visual_prompt_learning_rate",
    "soft_prompt_learning_rate",
    "visual_layernorm_learning_rate",
    "encoder_learning_rate",
    "visual_prompt_weight_decay",
    "soft_prompt_weight_decay",
    "visual_layernorm_weight_decay",
    "encoder_weight_decay",
    "alignment_geometry",
    "alignment_anchor",
    "alignment_target_gradient",
    "lambda_alignment_mean",
    "lambda_alignment_covariance",
    "calibration_batches",
    "calibration_target_ratio",
    "max_steps",
    "probe_steps",
    "eval_batch_size",
    "query_chunk_size",
    "pseudo_val_seed",
    "pseudo_val_num_classes",
    "seed",
    "train_class_scope",
    "official_unseen_used_for_selection",
)
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


def _split_identity_hash(value: Any) -> str:
    if not isinstance(value, dict):
        return _canonical_hash(value)
    return _canonical_hash({key: item for key, item in value.items() if key != "sha256"})


def _resolve_path(value: object) -> Path:
    return Path(str(value)).expanduser()


def _runs(campaign_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(campaign_dir.glob("**/run_result.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            result = {"_load_error": str(error)}
        if not isinstance(result, dict):
            result = {"_load_error": "run_result is not a JSON object"}
        result["_artifact_path"] = str(path.resolve())
        found.append(result)
    return found


def _step0_metadata(
    result: dict[str, Any], *, strict_steps: bool = False
) -> dict[str, Any]:
    history = result.get("history", [])
    rows = []
    for row in history if isinstance(history, list) else []:
        if not isinstance(row, dict):
            continue
        raw_step = row.get("training_global_step", -1)
        if strict_steps:
            if isinstance(raw_step, bool) or not isinstance(raw_step, int):
                continue
        else:
            try:
                raw_step = int(raw_step)
            except (TypeError, ValueError):
                continue
        if raw_step == 0:
            rows.append(row)
    if not rows:
        return {"status": "UNVERIFIED_NO_STEP0"}
    checkpoint = _resolve_run_path(
        result, rows[0].get("checkpoint", ""), prefer_artifact=True
    )
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
        "campaign": payload.get("campaign"),
        "experiment_role": payload.get("experiment_role"),
        "run_kind": payload.get("run_kind"),
        "initial_model_state_hash": payload.get("initial_model_state_hash"),
        "initial_text_bank_state_hash": payload.get("initial_text_bank_state_hash"),
        "source_snapshot_hash": payload.get("source_snapshot_hash"),
        "backbone_identity": payload.get("backbone_identity"),
        "scheduler_identity": payload.get("scheduler_state_dict"),
        "resolved_config": payload.get("resolved_config"),
        "resolved_treatment": payload.get("resolved_treatment"),
        "gradient_calibration_identity": payload.get("gradient_calibration_identity"),
        "training_seed": payload.get("training_seed"),
        "split_identity": payload.get("data_split_identity"),
        "manifest_identity": payload.get("data_manifest_identity"),
        "manifest_entry_identity": payload.get("manifest_entry_identity"),
        "optimizer_identity": payload.get("optimizer_groups"),
    }


def _protocol_identity(result: dict[str, Any]) -> dict[str, Any]:
    protocol = result.get("protocol", {})
    if not isinstance(protocol, dict):
        protocol = {}
    identity = {key: protocol.get(key) for key in PROTOCOL_KEYS}
    if _matching_mode(result) == "corrected_v2":
        identity["inference_contract"] = result.get("inference_contract")
    return identity


def _mismatch(
    field: str, expected: Any, actual: Any, message: str
) -> dict[str, Any]:
    return {
        "field": field,
        "expected": expected,
        "actual": actual,
        "message": message,
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _same_float(left: Any, right: Any) -> bool:
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _same_config_value(left: Any, right: Any) -> bool:
    if _finite_number(left) or _finite_number(right):
        return _same_float(left, right)
    return left == right


def _same_json_value(left: Any, right: Any) -> bool:
    return _canonical_hash(left) == _canonical_hash(right)


def _resolve_run_path(
    run: dict[str, Any],
    value: Any,
    *,
    resolve: bool = True,
    prefer_artifact: bool = False,
) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve() if resolve else path
    candidates = (
        (Path(run["_artifact_path"]).parent / path, Path.cwd() / path)
        if prefer_artifact
        else (Path.cwd() / path, Path(run["_artifact_path"]).parent / path)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve() if resolve else candidate
    unresolved = Path.cwd() / path
    return unresolved.resolve() if resolve else unresolved


def _matching_mode(result: dict[str, Any]) -> str:
    resolved = result.get("resolved_config")
    identity = result.get("manifest_entry_identity")
    resolved_campaign = (
        resolved.get("experiment_campaign") if isinstance(resolved, dict) else None
    )
    declared_values = (
        result.get("manifest_path"),
        resolved.get("experiment_manifest_path")
        if isinstance(resolved, dict)
        else None,
        identity.get("manifest_path") if isinstance(identity, dict) else None,
    )
    corrected_signal = (
        result.get("campaign") == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        or resolved_campaign == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        or result.get("experiment_role") == "alignment_mean_text_log_symmetric"
        or any(
            value is not None
            and Path(str(value)).name == "corrected_pilot_manifest_v2.json"
            for value in declared_values
        )
    )
    if any(value is None for value in declared_values):
        return "corrected_v2" if corrected_signal else "historical"
    paths = [
        _resolve_run_path(result, value, resolve=False, prefer_artifact=True)
        for value in declared_values
    ]
    if any(path.absolute() != paths[0].absolute() for path in paths[1:]):
        return "corrected_v2" if corrected_signal else "historical"
    for manifest_path in paths:
        try:
            if manifest_path.is_symlink() or (
                manifest_path.exists() and manifest_path.lstat().st_nlink != 1
            ):
                return "corrected_v2" if corrected_signal else "historical"
            if not manifest_path.is_file():
                return "corrected_v2" if corrected_signal else "historical"
            manifest = json.loads(manifest_path.read_bytes())
        except (OSError, ValueError, json.JSONDecodeError):
            return "corrected_v2" if corrected_signal else "historical"
        if manifest_path.name != "corrected_pilot_manifest.json":
            return "corrected_v2" if corrected_signal else "historical"
        protocol = manifest.get("protocol") if isinstance(manifest, dict) else None
        legacy_protocol = {
            "selection_metric": "full_pseudo_unseen_mAP",
            "official_unseen_used_for_selection": False,
            "text_used_for_predictor": False,
            "photo_used_for_predictor": False,
            "train_only_alignment_targets": True,
        }
        if not (
            isinstance(manifest, dict)
            and manifest.get("manifest_marker") is None
            and type(manifest.get("schema_version")) is int
            and manifest.get("schema_version") == 2
            and manifest.get("campaign") == result.get("campaign")
            and manifest.get("roles") == list(CORRECTED_PILOT_ROLES)
            and isinstance(manifest.get("entries"), list)
            and all(isinstance(item, dict) for item in manifest["entries"])
            and _same_json_value(protocol, legacy_protocol)
        ):
            return "corrected_v2"
    return "historical"


def _series_is_finite(
    value: Any,
    length: int,
    *,
    allow_none: bool,
    nonnegative: bool = False,
) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    return all(
        (item is None if allow_none else False)
        or (
            _finite_number(item)
            and (not nonnegative or float(item) >= 0.0)
        )
        for item in value
    )


def _series_matches(actual: Any, expected: list[float | None]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for observed, target in zip(actual, expected, strict=True):
        if target is None:
            if observed is not None:
                return False
        elif not _same_float(observed, target):
            return False
    return True


def _reasons_are_explicit(values: Any, reasons: Any, length: int) -> bool:
    if not isinstance(values, list) or not isinstance(reasons, list):
        return False
    if len(values) != length or len(reasons) != length:
        return False
    return all(
        (value is None and isinstance(reason, str) and bool(reason))
        or (value is not None and reason is None)
        for value, reason in zip(values, reasons, strict=True)
    )


def _calibration_validation(result: dict[str, Any]) -> dict[str, Any]:
    campaign = result.get("campaign")
    role = result.get("experiment_role")
    if campaign != ALIGNMENT_CORRECTED_PILOT_CAMPAIGN or role not in CORRECTED_PILOT_ROLES:
        return {"status": "NOT_APPLICABLE", "mismatches": []}

    resolved = result.get("resolved_config")
    if not isinstance(resolved, dict):
        resolved = {}
    evidence = result.get("gradient_calibration")
    configured_reference = resolved.get(CALIBRATION_CONFIG_KEY)
    mismatches: list[dict[str, Any]] = []
    is_control = role == "alignment_control"
    if is_control:
        if configured_reference not in (None, ""):
            mismatches.append(
                _mismatch(
                    CALIBRATION_CONFIG_KEY,
                    None,
                    configured_reference,
                    "R must not reference a calibration artifact",
                )
            )
        if evidence is not None:
            mismatches.append(
                _mismatch(
                    "gradient_calibration",
                    None,
                    evidence,
                    "R must not contain gradient_calibration evidence",
                )
            )
        return {
            "status": "ABSENT" if not mismatches else "UNVERIFIED",
            "mismatches": mismatches,
        }

    arm = "MD" if role == "alignment_mean_text_log" else "MS"
    if configured_reference in (None, ""):
        mismatches.append(
            _mismatch(
                CALIBRATION_CONFIG_KEY,
                "verified calibration artifact path",
                configured_reference,
                f"{arm} requires a calibration artifact",
            )
        )
    if not isinstance(evidence, dict):
        mismatches.append(
            _mismatch(
                "gradient_calibration",
                "complete calibration evidence",
                evidence,
                f"{arm} calibration evidence is incomplete",
            )
        )
        evidence = {}
    required_evidence = {
        "artifact",
        "artifact_sha256",
        "lambda_alignment_mean",
        "campaign",
        "experiment_role",
        "training_seed",
        "source_snapshot_hash",
        "split_identity_hash",
        "initial_model_state_hash",
        "initial_text_bank_state_hash",
        "config_hash",
        "schema_version",
        "fixed_batch_identity_sha256",
        "first_batch_replay_verified",
        "calibration_batch_replay_verified",
        "calibration_batch_replay_count",
        "calibration_batch_replay_expected_count",
        "calibration_batch_replay_prefix_sha256",
        "worker_lifecycle_verified",
    }
    if not required_evidence.issubset(evidence):
        mismatches.append(
            _mismatch(
                "gradient_calibration",
                sorted(required_evidence),
                sorted(evidence),
                f"{arm} calibration evidence is incomplete",
            )
        )
    if not isinstance(evidence.get("config_hash"), str) or not evidence.get("config_hash"):
        mismatches.append(
            _mismatch(
                "gradient_calibration.config_hash",
                "non-empty hash",
                evidence.get("config_hash"),
                f"{arm} calibration evidence is incomplete",
            )
        )
    if evidence.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        mismatches.append(
            _mismatch(
                "gradient_calibration.schema_version",
                CALIBRATION_SCHEMA_VERSION,
                evidence.get("schema_version"),
                f"{arm} calibration evidence is insufficient",
            )
        )
    if evidence.get("first_batch_replay_verified") is not True:
        mismatches.append(
            _mismatch(
                "gradient_calibration.first_batch_replay_verified",
                True,
                evidence.get("first_batch_replay_verified"),
                f"{arm} calibration batch replay is unverified",
            )
        )
    if evidence.get("calibration_batch_replay_verified") is not True:
        mismatches.append(
            _mismatch(
                "gradient_calibration.calibration_batch_replay_verified",
                True,
                evidence.get("calibration_batch_replay_verified"),
                f"{arm} calibration batch sequence replay is unverified",
            )
        )
    if evidence.get("worker_lifecycle_verified") is not True:
        mismatches.append(
            _mismatch(
                "gradient_calibration.worker_lifecycle_verified",
                True,
                evidence.get("worker_lifecycle_verified"),
                f"{arm} calibration worker lifecycle is unverified",
            )
        )

    artifact_path: Path | None = None
    if configured_reference not in (None, ""):
        artifact_path = _resolve_run_path(result, configured_reference)
        recorded_path_value = evidence.get("artifact")
        if recorded_path_value in (None, ""):
            mismatches.append(
                _mismatch(
                    "gradient_calibration.artifact",
                    str(artifact_path),
                    recorded_path_value,
                    f"{arm} calibration artifact reference does not match resolved config",
                )
            )
        elif _resolve_run_path(result, recorded_path_value) != artifact_path:
            mismatches.append(
                _mismatch(
                    "gradient_calibration.artifact",
                    str(artifact_path),
                    recorded_path_value,
                    f"{arm} calibration artifact reference does not match resolved config",
                )
            )

    artifact: dict[str, Any] | None = None
    artifact_bytes: bytes | None = None
    if artifact_path is None or not artifact_path.is_file():
        mismatches.append(
            _mismatch(
                "calibration_artifact.readability",
                "readable regular file",
                None if artifact_path is None else str(artifact_path),
                f"{arm} calibration artifact is missing",
            )
        )
    else:
        try:
            artifact_bytes = artifact_path.read_bytes()
            artifact_value = json.loads(artifact_bytes)
            if not isinstance(artifact_value, dict):
                raise ValueError("calibration artifact is not a JSON object")
            artifact = artifact_value
        except (OSError, ValueError, json.JSONDecodeError) as error:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.readability",
                    "readable JSON object",
                    str(error),
                    f"{arm} calibration artifact is unreadable",
                )
            )

    if isinstance(artifact, dict):
        try:
            fixed_batches = artifact["calibration"]["fixed_batch_identity"]["batches"]
        except (KeyError, TypeError):
            fixed_batches = None
        if not isinstance(fixed_batches, list):
            mismatches.append(
                _mismatch(
                    "gradient_calibration.replay_batches",
                    "calibration batch list",
                    fixed_batches,
                    f"{arm} calibration replay evidence is incomplete",
                )
            )
        else:
            expected_count = len(fixed_batches)
            if evidence.get("calibration_batch_replay_expected_count") != expected_count:
                mismatches.append(
                    _mismatch(
                        "gradient_calibration.calibration_batch_replay_expected_count",
                        expected_count,
                        evidence.get("calibration_batch_replay_expected_count"),
                        f"{arm} calibration replay count is stale",
                    )
                )
            if evidence.get("calibration_batch_replay_count") != expected_count:
                mismatches.append(
                    _mismatch(
                        "gradient_calibration.calibration_batch_replay_count",
                        expected_count,
                        evidence.get("calibration_batch_replay_count"),
                        f"{arm} calibration batch sequence replay is incomplete",
                    )
                )
            if evidence.get("calibration_batch_replay_prefix_sha256") != _canonical_hash(
                fixed_batches
            ):
                mismatches.append(
                    _mismatch(
                        "gradient_calibration.calibration_batch_replay_prefix_sha256",
                        _canonical_hash(fixed_batches),
                        evidence.get("calibration_batch_replay_prefix_sha256"),
                        f"{arm} calibration replay prefix is stale",
                    )
                )

    if artifact_bytes is not None:
        actual_hash = hashlib.sha256(artifact_bytes).hexdigest()
        expected_hash = evidence.get("artifact_sha256")
        if expected_hash != actual_hash:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.sha256",
                    expected_hash,
                    actual_hash,
                    f"{arm} calibration artifact SHA-256 mismatch",
                )
            )

    config_lambda = resolved.get("lambda_alignment_mean")
    recorded_lambda = evidence.get("lambda_alignment_mean")
    if not _same_float(recorded_lambda, config_lambda):
        mismatches.append(
            _mismatch(
                "gradient_calibration.lambda_alignment_mean",
                config_lambda,
                recorded_lambda,
                f"{arm} calibration metadata lambda does not match resolved lambda",
            )
        )

    calibration: dict[str, Any] = {}
    if artifact is not None:
        if artifact.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.schema_version",
                    CALIBRATION_SCHEMA_VERSION,
                    artifact.get("schema_version"),
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if artifact.get("schema_name") != "corrected_mean_alignment_calibration":
            mismatches.append(
                _mismatch(
                    "calibration_artifact.schema_name",
                    "corrected_mean_alignment_calibration",
                    artifact.get("schema_name"),
                    f"{arm} calibration evidence is insufficient",
                )
            )
        calibration_value = artifact.get("calibration")
        if isinstance(calibration_value, dict):
            calibration = calibration_value
        else:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.evidence",
                    "calibration object",
                    calibration_value,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if "weighted_photo_gradient_ratios_if_symmetric" in artifact:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.evidence",
                    "separate symmetric photo diagnostics",
                    "weighted_photo_gradient_ratios_if_symmetric",
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if artifact.get("status") != "VALID" or calibration.get("status") != "VALID":
            mismatches.append(
                _mismatch(
                    "calibration_artifact.evidence",
                    "VALID",
                    {"status": artifact.get("status"), "calibration_status": calibration.get("status")},
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if calibration.get("lambda_selection_status") != "VALID":
            mismatches.append(
                _mismatch(
                    "calibration_artifact.lambda_selection_status",
                    "VALID",
                    calibration.get("lambda_selection_status"),
                    f"{arm} calibration evidence is insufficient",
                )
            )
        fixed_batch_identity = calibration.get("fixed_batch_identity")
        if evidence.get("fixed_batch_identity_sha256") != (
            fixed_batch_identity.get("sha256")
            if isinstance(fixed_batch_identity, dict)
            else None
        ):
            mismatches.append(
                _mismatch(
                    "gradient_calibration.fixed_batch_identity_sha256",
                    fixed_batch_identity.get("sha256")
                    if isinstance(fixed_batch_identity, dict)
                    else None,
                    evidence.get("fixed_batch_identity_sha256"),
                    f"{arm} calibration fixed-batch identity is stale",
                )
            )
        sketch_comparison = calibration.get("sketch_gradient_comparison")
        comparison_difference = (
            sketch_comparison.get("detached_vs_symmetric_max_abs_difference")
            if isinstance(sketch_comparison, dict)
            else None
        )
        comparison_valid = (
            isinstance(sketch_comparison, dict)
            and sketch_comparison.get("match") is True
            and _same_float(
                sketch_comparison.get("tolerance"), CALIBRATION_SKETCH_TOLERANCE
            )
            and _finite_number(
                sketch_comparison.get("detached_vs_symmetric_max_abs_difference")
            )
            and float(
                sketch_comparison["detached_vs_symmetric_max_abs_difference"]
            )
            >= 0.0
            and float(
                sketch_comparison["detached_vs_symmetric_max_abs_difference"]
            )
            <= CALIBRATION_SKETCH_TOLERANCE
        )
        if not comparison_valid:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.sketch_gradient_comparison",
                    {"match": True, "max_difference": "<= tolerance"},
                    sketch_comparison,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        else:
            base_diagnostics = artifact.get("base")
            detached_diagnostics = artifact.get("detached")
            symmetric_diagnostics = artifact.get("symmetric")
            batches_count = calibration.get("batches")
            if (
                isinstance(base_diagnostics, dict)
                and isinstance(detached_diagnostics, dict)
                and isinstance(symmetric_diagnostics, dict)
                and isinstance(batches_count, int)
                and not isinstance(batches_count, bool)
                and batches_count > 0
                and _series_is_finite(
                    base_diagnostics.get("sketch_gradient_norms"),
                    batches_count,
                    allow_none=False,
                )
                and _series_is_finite(
                    detached_diagnostics.get("sketch_gradient_norms"),
                    batches_count,
                    allow_none=False,
                )
                and _series_is_finite(
                    symmetric_diagnostics.get("sketch_gradient_norms"),
                    batches_count,
                    allow_none=False,
                )
                and float(comparison_difference)
                + 1e-12
                < max(
                    abs(float(detached_value) - float(symmetric_value))
                    for detached_value, symmetric_value in zip(
                        detached_diagnostics["sketch_gradient_norms"],
                        symmetric_diagnostics["sketch_gradient_norms"],
                        strict=True,
                    )
                )
            ):
                mismatches.append(
                    _mismatch(
                        "calibration_artifact.sketch_gradient_comparison",
                        "max difference at least the diagnostic norm gap",
                        sketch_comparison,
                        f"{arm} calibration evidence is insufficient",
                    )
                )

        step0 = _step0_metadata(result)
        expected_identity = result.get("initial_model_state_hash")
        if expected_identity is None:
            expected_identity = step0.get("initial_model_state_hash")
        expected_text_identity = result.get("initial_text_bank_state_hash")
        if expected_text_identity is None:
            expected_text_identity = step0.get("initial_text_bank_state_hash")
        identity = calibration.get("initialization_identity")
        actual_identity = (
            identity.get("model_state_hash")
            if isinstance(identity, dict)
            else None
        )
        actual_text_identity = (
            identity.get("text_bank_state_hash")
            if isinstance(identity, dict)
            else None
        )
        if not isinstance(identity, dict) or not actual_identity:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.initialization_identity",
                    "identity with model_state_hash",
                    identity,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if not actual_identity or expected_identity in (None, "UNVERIFIED"):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.initial_model_state_hash",
                    expected_identity,
                    actual_identity,
                    f"{arm} calibration artifact identity is stale",
                )
            )
        elif actual_identity != expected_identity:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.initial_model_state_hash",
                    expected_identity,
                    actual_identity,
                    f"{arm} calibration artifact identity is stale",
                )
            )
        if not actual_text_identity or not expected_text_identity:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.initialization_identity.text_bank_state_hash",
                    expected_text_identity,
                    actual_text_identity,
                    f"{arm} calibration artifact identity is stale",
                )
            )
        elif actual_text_identity != expected_text_identity:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.initialization_identity.text_bank_state_hash",
                    expected_text_identity,
                    actual_text_identity,
                    f"{arm} calibration artifact identity is stale",
                )
            )

        expected_campaign = result.get("campaign")
        actual_campaign = artifact.get("campaign")
        artifact_role = artifact.get("experiment_role")
        if artifact_role not in CORRECTED_PILOT_ROLES[1:]:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.experiment_role",
                    list(CORRECTED_PILOT_ROLES[1:]),
                    artifact_role,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if actual_campaign != expected_campaign:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.campaign",
                    expected_campaign,
                    actual_campaign,
                    f"{arm} calibration artifact identity is stale",
                )
            )
        expected_seed = result.get("training_seed")
        actual_seed = artifact.get("training_seed")
        seed_matches = type(actual_seed) is type(expected_seed) and actual_seed == expected_seed
        if not seed_matches:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.training_seed",
                    expected_seed,
                    actual_seed,
                    f"{arm} calibration artifact identity is stale",
                )
            )
        for field, expected in (
            ("campaign", expected_campaign),
            ("training_seed", expected_seed),
            ("initial_model_state_hash", expected_identity),
            ("initial_text_bank_state_hash", expected_text_identity),
        ):
            if artifact.get(field) != expected:
                mismatches.append(
                    _mismatch(
                        f"calibration_artifact.{field}",
                        expected,
                        artifact.get(field),
                        f"{arm} calibration artifact identity is stale",
                    )
                )
        nested_identity = {
            "experiment_role": artifact_role,
            "campaign": expected_campaign,
            "training_seed": expected_seed,
            "initial_model_state_hash": actual_identity,
        }
        for field, expected in nested_identity.items():
            if calibration.get(field) != expected:
                mismatches.append(
                    _mismatch(
                        f"calibration_artifact.calibration.{field}",
                        expected,
                        calibration.get(field),
                        f"{arm} calibration artifact identity is stale",
                    )
                )

        source_hash = result.get("source_snapshot_hash")
        if source_hash is None:
            provenance = result.get("provenance")
            source_snapshot = (
                provenance.get("source_snapshot")
                if isinstance(provenance, dict)
                else None
            )
            source_hash = (
                source_snapshot.get("sha256")
                if isinstance(source_snapshot, dict)
                else None
            )
        split_value = result.get("pseudo_split_identity", {})
        split_hash = (
            split_value.get("sha256")
            if isinstance(split_value, dict)
            else None
        )
        for field, expected in (
            ("campaign", expected_campaign),
            ("experiment_role", role),
            ("training_seed", expected_seed),
            ("source_snapshot_hash", source_hash),
            ("split_identity_hash", split_hash),
        ):
            if evidence.get(field) != expected:
                mismatches.append(
                    _mismatch(
                        f"gradient_calibration.{field}",
                        expected,
                        evidence.get(field),
                        f"{arm} calibration evidence is stale",
                    )
                )
        split_value = artifact.get("split_identity")
        expected_split = result.get("pseudo_split_identity")
        if (
            not isinstance(split_value, dict)
            or split_value.get("sha256") != _split_identity_hash(split_value)
            or not isinstance(expected_split, dict)
            or split_value != expected_split
        ):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.split_identity",
                    expected_split,
                    split_value,
                    f"{arm} calibration artifact split identity is stale",
                )
            )
        optional_identity_fields = (
            ("source_snapshot_hash", source_hash),
            ("split_identity_hash", split_hash),
            ("initial_text_bank_state_hash", expected_text_identity),
        )
        for field, expected in optional_identity_fields:
            actual = artifact.get(field)
            if actual != expected:
                mismatches.append(
                    _mismatch(
                        f"calibration_artifact.{field}",
                        expected,
                        actual,
                        f"{arm} calibration artifact identity is stale",
                    )
                )
            if calibration.get(field) != expected:
                mismatches.append(
                    _mismatch(
                        f"calibration_artifact.calibration.{field}",
                        expected,
                        calibration.get(field),
                        f"{arm} calibration artifact identity is stale",
                    )
                )
        recorded_identity_fields = (
            ("initial_model_state_hash", actual_identity),
            ("initial_text_bank_state_hash", actual_text_identity),
            ("source_snapshot_hash", source_hash),
            ("split_identity_hash", split_hash),
            ("config_hash", artifact.get("config_hash")),
        )
        for field, expected in recorded_identity_fields:
            actual = evidence.get(field)
            if actual is not None and actual != expected:
                mismatches.append(
                    _mismatch(
                        f"gradient_calibration.{field}",
                        expected,
                        actual,
                        f"{arm} calibration evidence is stale",
                    )
                )

        target = calibration.get("target_ratio")
        configured_target = resolved.get("calibration_target_ratio")
        if not _same_float(target, configured_target):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.target_ratio",
                    configured_target,
                    target,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        batches = calibration.get("batches")
        configured_batches = resolved.get("calibration_batches")
        if batches != configured_batches:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.batches",
                    configured_batches,
                    batches,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if calibration.get("rule") != CALIBRATION_RULE:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.rule",
                    CALIBRATION_RULE,
                    calibration.get("rule"),
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if calibration.get("state_restoration_verified") is not True:
            mismatches.append(
                _mismatch(
                    "calibration_artifact.state_restoration_verified",
                    True,
                    calibration.get("state_restoration_verified"),
                    f"{arm} calibration evidence is insufficient",
                )
            )
        calibration_config = artifact.get("calibration_config")
        if isinstance(calibration_config, dict):
            expected_config_fields = {
                "experiment_campaign": result.get("campaign"),
                "experiment_role": artifact.get("experiment_role"),
                "seed": result.get("training_seed"),
                "pseudo_val_seed": split_hash,
                "alignment_geometry": "log_map",
                "alignment_anchor": "text",
                "alignment_target_gradient": (
                    "symmetric"
                    if artifact.get("experiment_role")
                    == "alignment_mean_text_log_symmetric"
                    else "detached"
                ),
                "lambda_alignment_covariance": 0.0,
            }
            for field, expected in expected_config_fields.items():
                actual = calibration_config.get(field)
                if field == "pseudo_val_seed":
                    split_identity = result.get("pseudo_split_identity", {})
                    expected = split_identity.get(
                        "pseudo_validation_seed", split_identity.get("seed")
                    )
                if actual != expected:
                    mismatches.append(
                        _mismatch(
                            f"calibration_artifact.config.{field}",
                            expected,
                            actual,
                            f"{arm} calibration artifact config is stale",
                        )
                    )
        fixed_identity = calibration.get("fixed_batch_identity")
        if not isinstance(fixed_identity, dict) or not fixed_identity.get("sha256"):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.fixed_batch_identity",
                    "identity with sha256",
                    fixed_identity,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        elif (
            fixed_identity.get("count") != calibration.get("batches")
            or not isinstance(fixed_identity.get("batches"), list)
            or len(fixed_identity["batches"]) != calibration.get("batches")
            or fixed_identity.get("sha256")
            != _canonical_hash(fixed_identity["batches"])
            or fixed_identity.get("sampler_epoch_before") != 0
            or fixed_identity.get("worker_lifecycle_verified") is not True
        ):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.fixed_batch_identity",
                    {"count": calibration.get("batches"), "sha256": "hash(batches)"},
                    fixed_identity,
                    f"{arm} calibration evidence is insufficient",
                )
            )
        calibration_config = artifact.get("calibration_config")
        if not isinstance(calibration_config, dict) or artifact.get("config_hash") != _canonical_hash(calibration_config):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.config_hash",
                    "hash(calibration_config)",
                    artifact.get("config_hash"),
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if calibration.get("config_hash") != artifact.get("config_hash"):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.calibration.config_hash",
                    artifact.get("config_hash"),
                    calibration.get("config_hash"),
                    f"{arm} calibration evidence is insufficient",
                )
            )
        if isinstance(calibration_config, dict):
            ignored_config_keys = {
                CALIBRATION_CONFIG_KEY,
                "alignment_target_gradient",
                "calibration_only",
                "experiment_name",
                "experiment_role",
                "lambda_alignment_mean",
            }
            config_keys = (
                set(resolved) | set(calibration_config)
            ) - ignored_config_keys
            for field in sorted(config_keys):
                expected = resolved.get(field)
                actual = calibration_config.get(field)
                if not _same_config_value(expected, actual):
                    mismatches.append(
                        _mismatch(
                            f"calibration_artifact.config.{field}",
                            expected,
                            actual,
                            f"{arm} calibration artifact config is stale",
                        )
                    )

        base = artifact.get("base")
        detached = artifact.get("detached")
        symmetric = artifact.get("symmetric")
        if not all(isinstance(value, dict) for value in (base, detached, symmetric)):
            mismatches.append(
                _mismatch(
                    "calibration_artifact.evidence",
                    "base, detached, and symmetric diagnostics",
                    {"base": base, "detached": detached, "symmetric": symmetric},
                    f"{arm} calibration evidence is insufficient",
                )
            )
        else:
            batches_count = calibration.get("batches")
            if (
                not isinstance(batches_count, int)
                or isinstance(batches_count, bool)
                or batches_count <= 0
            ):
                mismatches.append(
                    _mismatch(
                        "calibration_artifact.batches",
                        "positive integer",
                        batches_count,
                        f"{arm} calibration evidence is insufficient",
                    )
                )
                batches_count = -1
            norm_fields = (
                (base, "sketch_gradient_norms", False, True),
                (base, "photo_gradient_norms", False, True),
                (detached, "sketch_gradient_norms", False, True),
                (detached, "photo_gradient_norms", False, True),
                (symmetric, "sketch_gradient_norms", False, True),
                (symmetric, "photo_gradient_norms", False, True),
                (detached, "weighted_sketch_ratios", True, True),
                (detached, "weighted_photo_ratios", True, True),
                (symmetric, "weighted_sketch_ratios", True, True),
                (symmetric, "weighted_photo_ratios", True, True),
                (detached, "sketch_cosines_with_base", True, False),
                (detached, "photo_cosines_with_base", True, False),
                (symmetric, "sketch_cosines_with_base", True, False),
                (symmetric, "photo_cosines_with_base", True, False),
                (detached, "unweighted_sketch_ratios", True, True),
            )
            for policy, field, allow_none, nonnegative in norm_fields:
                if not _series_is_finite(
                    policy.get(field),
                    batches_count,
                    allow_none=allow_none,
                    nonnegative=nonnegative,
                ):
                    mismatches.append(
                        _mismatch(
                            f"calibration_artifact.{field}",
                            f"finite series of length {batches_count}",
                            policy.get(field),
                            f"{arm} calibration evidence is insufficient",
                        )
                    )
            for policy in (detached, symmetric):
                for value_field, reason_field in (
                    ("weighted_sketch_ratios", "weighted_sketch_ratio_reasons"),
                    ("weighted_photo_ratios", "weighted_photo_ratio_reasons"),
                    ("sketch_cosines_with_base", "sketch_cosine_reasons"),
                    ("photo_cosines_with_base", "photo_cosine_reasons"),
                ):
                    if not _reasons_are_explicit(
                        policy.get(value_field), policy.get(reason_field), batches_count
                    ):
                        mismatches.append(
                            _mismatch(
                                f"calibration_artifact.{reason_field}",
                                f"reason for every null {value_field}",
                                policy.get(reason_field),
                                f"{arm} calibration evidence is insufficient",
                            )
                        )
            if not _reasons_are_explicit(
                detached.get("unweighted_sketch_ratios"),
                detached.get("unweighted_sketch_ratio_reasons"),
                batches_count,
            ):
                mismatches.append(
                    _mismatch(
                        "calibration_artifact.unweighted_sketch_ratio_reasons",
                        "reason for every null unweighted_sketch_ratios",
                        detached.get("unweighted_sketch_ratio_reasons"),
                        f"{arm} calibration evidence is insufficient",
                    )
                )
            base_sketch = base.get("sketch_gradient_norms")
            if _series_is_finite(base_sketch, batches_count, allow_none=False) and any(
                float(value) <= 1e-12 for value in base_sketch
            ):
                mismatches.append(
                    _mismatch(
                        "calibration_artifact.base.sketch_gradient_norms",
                        "strictly positive denominators",
                        base_sketch,
                        f"{arm} calibration evidence is insufficient",
                    )
                )
            detached_photo = detached.get("photo_gradient_norms")
            if _series_is_finite(detached_photo, batches_count, allow_none=False) and any(
                float(value) > 1e-12 for value in detached_photo
            ):
                mismatches.append(
                    _mismatch(
                        "calibration_artifact.detached.photo_gradient_norms",
                        "all zeros",
                        detached_photo,
                        f"{arm} calibration evidence is insufficient",
                    )
                )

            artifact_lambda = calibration.get("lambda_alignment_mean")
            if not _same_float(artifact_lambda, config_lambda):
                mismatches.append(
                    _mismatch(
                        "calibration_artifact.lambda_alignment_mean",
                        config_lambda,
                        artifact_lambda,
                        f"{arm} calibration artifact lambda does not match resolved lambda",
                    )
                )
            if _finite_number(artifact_lambda) and _finite_number(target) and batches_count > 0:
                base_sketch = base.get("sketch_gradient_norms")
                detached_sketch = detached.get("sketch_gradient_norms")
                symmetric_sketch = symmetric.get("sketch_gradient_norms")
                base_valid = _series_is_finite(
                    base_sketch, batches_count, allow_none=False
                )
                detached_valid = _series_is_finite(
                    detached_sketch, batches_count, allow_none=False
                )
                raw = (
                    [
                        None if value <= 1e-12 else float(numerator) / float(value)
                        for numerator, value in zip(
                            base_sketch, detached_sketch, strict=True
                        )
                    ]
                    if base_valid and detached_valid
                    else []
                )
                if not _series_matches(detached.get("unweighted_sketch_ratios"), raw):
                    mismatches.append(
                        _mismatch(
                            "calibration_artifact.unweighted_sketch_ratios",
                            raw,
                            detached.get("unweighted_sketch_ratios"),
                            f"{arm} calibration evidence is insufficient",
                        )
                    )
                if not raw or not all(_finite_number(value) for value in raw):
                    mismatches.append(
                        _mismatch(
                            "calibration_artifact.unweighted_sketch_ratios",
                            "finite ratio for every calibration batch",
                            raw,
                            f"{arm} calibration evidence is insufficient",
                        )
                    )
                if raw and all(_finite_number(value) for value in raw):
                    expected_lambda = float(target) * statistics.median(raw)
                    if not _same_float(artifact_lambda, expected_lambda):
                        mismatches.append(
                            _mismatch(
                                "calibration_artifact.lambda_alignment_mean",
                                expected_lambda,
                                artifact_lambda,
                                f"{arm} calibration artifact lambda does not match resolved lambda",
                            )
                        )
                for policy, norms in ((detached, detached_sketch), (symmetric, symmetric_sketch)):
                    if not base_valid or not _series_is_finite(
                        norms, batches_count, allow_none=False
                    ):
                        continue
                    expected_ratios = [
                        None
                        if float(base_value) <= 1e-12
                        else float(artifact_lambda)
                        * float(alignment_value)
                        / float(base_value)
                        for base_value, alignment_value in zip(
                            base_sketch, norms, strict=True
                        )
                    ]
                    if not _series_matches(
                        policy.get("weighted_sketch_ratios"), expected_ratios
                    ):
                        mismatches.append(
                            _mismatch(
                                "calibration_artifact.weighted_sketch_ratios",
                                expected_ratios,
                                policy.get("weighted_sketch_ratios"),
                                f"{arm} calibration evidence is insufficient",
                            )
                        )
                base_photo = base.get("photo_gradient_norms")
                base_photo_valid = _series_is_finite(
                    base_photo, batches_count, allow_none=False
                )
                if base_photo_valid:
                    for policy, norms in (
                        (detached, detached.get("photo_gradient_norms")),
                        (symmetric, symmetric.get("photo_gradient_norms")),
                    ):
                        if not _series_is_finite(norms, batches_count, allow_none=False):
                            continue
                        expected_ratios = [
                            None
                            if float(base_value) <= 1e-12
                            else float(artifact_lambda)
                            * float(alignment_value)
                            / float(base_value)
                            for base_value, alignment_value in zip(
                                base_photo, norms, strict=True
                            )
                        ]
                        if not _series_matches(
                            policy.get("weighted_photo_ratios"), expected_ratios
                        ):
                            mismatches.append(
                                _mismatch(
                                    "calibration_artifact.weighted_photo_ratios",
                                    expected_ratios,
                                    policy.get("weighted_photo_ratios"),
                                    f"{arm} calibration evidence is insufficient",
                                )
                            )

    return {
        "status": "VERIFIED" if not mismatches else "UNVERIFIED",
        "mismatches": mismatches,
        "artifact_path": None if artifact_path is None else str(artifact_path),
        "artifact_sha256": evidence.get("artifact_sha256"),
    }


def _recompute_data_identities(
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = result["resolved_config"]
    data = load_data_config(_resolve_run_path(result, resolved["data_config"]))
    names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches,
        photos,
        names,
        num_validation_classes=int(resolved["pseudo_val_num_classes"]),
        seed=int(resolved["pseudo_val_seed"]),
    )
    split_identity: dict[str, Any] = {
        "dataset": data.name,
        "seed": split.seed,
        "train_class_ids": list(split.train_class_ids),
        "validation_class_ids": list(split.validation_class_ids),
        "train_sketches": len(split.train_sketch_entries),
        "train_photos": len(split.train_photo_entries),
        "validation_sketches": len(split.validation_sketch_entries),
        "validation_photos": len(split.validation_photo_entries),
    }
    split_identity["sha256"] = _split_identity_hash(split_identity)
    manifest_identity = split_manifest_identity(
        split,
        dataset_name=data.name,
        dataset_root=data.root,
        manifest_paths={
            "train_sketch": data.train.sketch_manifest,
            "train_photo": data.train.photo_manifest,
            "train_class_map": data.train.class_map,
        },
    )
    return split_identity, manifest_identity


def _manifest_entry_errors(result: dict[str, Any]) -> list[str]:
    identity = result.get("manifest_entry_identity")
    if not isinstance(identity, dict):
        return ["manifest entry identity is missing"]
    identity_value = identity.get("manifest_path")
    resolved = result.get("resolved_config")
    resolved = resolved if isinstance(resolved, dict) else {}
    result_value = result.get("manifest_path")
    config_value = resolved.get("experiment_manifest_path")
    errors: list[str] = []
    if not identity_value:
        errors.append("manifest entry manifest_path is missing")
    if not result_value:
        errors.append("top-level manifest path is missing")
    if not config_value:
        errors.append("resolved manifest path is missing")
    if errors and not identity_value:
        return errors
    if not identity_value:
        return errors
    manifest_path = _resolve_run_path(
        result, identity_value, resolve=False, prefer_artifact=True
    )
    for declared_value in (result_value, config_value):
        if declared_value is None:
            continue
        declared_path = _resolve_run_path(
            result, declared_value, resolve=False, prefer_artifact=True
        )
        if manifest_path.absolute() != declared_path.absolute():
            errors.append("manifest entry path differs from configured manifest path")
    if not manifest_path.is_file():
        return errors + [f"manifest is missing: {manifest_path}"]
    if manifest_path.is_symlink() or manifest_path.lstat().st_nlink != 1:
        errors.append("corrected manifest path is symlinked or multiply linked")
    if manifest_path.name == "corrected_pilot_manifest.json":
        errors.append("historical corrected manifest is not mutable")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return errors + [f"manifest is unreadable: {error}"]
    if not isinstance(manifest, dict):
        return errors + ["corrected manifest is not an object"]
    if identity.get("manifest_sha256") != manifest_identity_sha256(manifest):
        errors.append("manifest SHA-256 mismatch")
    if manifest.get("manifest_marker") != CORRECTED_MANIFEST_MARKER:
        errors.append("corrected manifest marker is invalid")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
        or manifest.get("campaign") != result.get("campaign")
    ):
        errors.append("corrected manifest schema or campaign is invalid")
    resolved = result.get("resolved_config", {})
    if not isinstance(resolved, dict):
        resolved = {}
    for field, expected in (
        ("dataset", result.get("dataset")),
        ("data_config", resolved.get("data_config")),
        ("roles", list(CORRECTED_PILOT_ROLES)),
    ):
        if expected is not None and not _same_json_value(manifest.get(field), expected):
            errors.append(f"corrected manifest {field} mismatch")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("corrected manifest protocol is missing")
    else:
        for field, expected in (
            ("selection_metric", "full_pseudo_unseen_mAP"),
            ("official_unseen_used_for_selection", False),
            ("text_used_for_predictor", False),
            ("photo_used_for_predictor", False),
            ("train_only_alignment_targets", True),
        ):
            if not _same_json_value(protocol.get(field), expected):
                errors.append(f"corrected manifest protocol {field} mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        errors.append("corrected manifest entries are invalid")
        entries = []
    run_ids: set[str] = set()
    arm_ids: set[tuple[str, int, str]] = set()
    required_entry_fields = {
        "run_id",
        "status",
        "experiment_role",
        "campaign",
        "training_seed",
        "pseudo_validation_seed",
        "replicate_id",
        "config_hash",
        "source_hash",
        "training_horizon",
        "initialization_identity",
        "treatment",
        "dataset_identity",
    }
    for item in entries:
        if not isinstance(item, dict):
            errors.append("corrected manifest contains an invalid entry")
            continue
        if not required_entry_fields.issubset(item):
            errors.append("corrected manifest contains an incomplete entry")
            continue
        run_id = item["run_id"]
        role = item["experiment_role"]
        seed = item["training_seed"]
        config_hash = item["config_hash"]
        if not isinstance(run_id, str) or not run_id:
            errors.append("corrected manifest entry run_id is invalid")
        elif run_id in run_ids:
            errors.append("corrected manifest contains duplicate run_id")
        else:
            run_ids.add(run_id)
        if not isinstance(role, str) or role not in CORRECTED_PILOT_ROLES:
            errors.append("corrected manifest entry experiment_role is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int):
            errors.append("corrected manifest entry training_seed is invalid")
        if isinstance(item["pseudo_validation_seed"], bool) or not isinstance(
            item["pseudo_validation_seed"], int
        ):
            errors.append("corrected manifest entry pseudo_validation_seed is invalid")
        if not isinstance(config_hash, str) or not config_hash:
            errors.append("corrected manifest entry config_hash is invalid")
        if item["status"] != "REGISTERED":
            errors.append("corrected manifest entry status is invalid")
        if item["campaign"] != result.get("campaign"):
            errors.append("corrected manifest entry campaign mismatch")
        if not isinstance(item["replicate_id"], str) or not item["replicate_id"]:
            errors.append("corrected manifest entry replicate_id is invalid")
        if (
            isinstance(item["source_hash"], str) or item["source_hash"] is None
        ) is False:
            errors.append("corrected manifest entry source_hash is invalid")
        if (
            isinstance(item["training_horizon"], bool)
            or not isinstance(item["training_horizon"], int)
            or item["training_horizon"] < 0
        ):
            errors.append("corrected manifest entry training_horizon is invalid")
        initialization = item["initialization_identity"]
        if not isinstance(initialization, dict) or not isinstance(
            initialization.get("initial_model_state_hash"), str
        ) or not initialization["initial_model_state_hash"]:
            errors.append("corrected manifest entry initialization is invalid")
        treatment = item["treatment"]
        if not isinstance(treatment, dict):
            errors.append("corrected manifest entry treatment is invalid")
        elif isinstance(role, str) and role in CORRECTED_PILOT_ROLES and isinstance(
            seed, int
        ) and not isinstance(seed, bool) and isinstance(
            item["pseudo_validation_seed"], int
        ) and not isinstance(item["pseudo_validation_seed"], bool):
            expected_treatment = treatment_for_role(
                role, seed=seed, pseudo_val_seed=item["pseudo_validation_seed"]
            )
            if any(
                treatment.get(key) != value
                for key, value in expected_treatment.items()
                if not (key == "lambda_alignment_mean" and role != "alignment_control")
            ) or any(
                not _finite_number(treatment.get(key))
                for key in ("lambda_alignment_mean", "lambda_alignment_covariance")
            ):
                errors.append("corrected manifest entry treatment is invalid")
        dataset_identity = item["dataset_identity"]
        if not isinstance(dataset_identity, dict):
            errors.append("corrected manifest entry dataset identity is invalid")
        else:
            split_entry = dataset_identity.get("split_identity")
            if (
                dataset_identity.get("dataset") != result.get("dataset")
                or dataset_identity.get("data_config") != resolved.get("data_config")
                or not isinstance(split_entry, dict)
                or split_entry.get("sha256") != _split_identity_hash(split_entry)
            ):
                errors.append("corrected manifest entry dataset identity is invalid")
        if (
            isinstance(role, str)
            and role in CORRECTED_PILOT_ROLES
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(config_hash, str)
            and config_hash
        ):
            arm_id = (role, seed, config_hash)
            if arm_id in arm_ids:
                errors.append("corrected manifest contains duplicate role/seed/config entry")
            else:
                arm_ids.add(arm_id)
        if isinstance(role, str) and isinstance(seed, int) and isinstance(
            item["pseudo_validation_seed"], int
        ) and isinstance(item["replicate_id"], str) and isinstance(config_hash, str):
            expected_run_id = _canonical_hash(
                {
                    "campaign": item["campaign"],
                    "role": role,
                    "training_seed": seed,
                    "split_identity": item.get("dataset_identity", {}).get(
                        "split_identity"
                    )
                    if isinstance(item.get("dataset_identity"), dict)
                    else None,
                    "config_hash": config_hash,
                    "replicate_id": item["replicate_id"],
                }
            )
            if run_id != expected_run_id:
                errors.append("corrected manifest entry run_id is invalid")
    pointer = identity.get("entry_pointer")
    entry: Any = None
    parts = pointer.split("/") if isinstance(pointer, str) else []
    if (
        len(parts) != 3
        or parts[1] != "entries"
        or not parts[2].isdigit()
    ):
        errors.append("manifest entry pointer is invalid")
    else:
        index = int(parts[2])
        if index >= len(entries):
            errors.append("manifest entry pointer does not resolve")
        else:
            entry = entries[index]
    if not isinstance(entry, dict):
        errors.append("manifest entry is missing")
    else:
        if identity.get("entry_sha256") != _canonical_hash(entry):
            errors.append("manifest entry SHA-256 mismatch")
        expected = {
            "experiment_role": result.get("experiment_role"),
            "campaign": result.get("campaign"),
            "training_seed": result.get("training_seed"),
            "pseudo_validation_seed": result.get("pseudo_validation_seed"),
            "config_hash": _canonical_hash(resolved),
            "source_hash": result.get("source_snapshot_hash"),
            "training_horizon": resolved.get("max_steps"),
        }
        for field, expected_value in expected.items():
            if expected_value is not None and not _same_json_value(
                entry.get(field), expected_value
            ):
                errors.append(f"manifest entry {field} mismatch")
        initialization = entry.get("initialization_identity")
        if not isinstance(initialization, dict) or not _same_json_value(
            initialization.get("initial_model_state_hash"),
            result.get("initial_model_state_hash"),
        ):
            errors.append("manifest entry initialization identity mismatch")
        dataset_identity = entry.get("dataset_identity")
        if not isinstance(dataset_identity, dict):
            errors.append("manifest entry dataset identity is missing")
        else:
            for field, expected_value in (
                ("dataset", result.get("dataset")),
                ("data_config", resolved.get("data_config")),
                ("split_identity", result.get("pseudo_split_identity")),
            ):
                if expected_value is not None and not _same_json_value(
                    dataset_identity.get(field), expected_value
                ):
                    errors.append(f"manifest entry dataset identity {field} mismatch")
        entry_treatment = entry.get("treatment")
        try:
            expected_treatment = treatment_for_role(
                str(result.get("experiment_role")),
                seed=int(result.get("training_seed")),
                pseudo_val_seed=int(result.get("pseudo_validation_seed")),
            )
        except (TypeError, ValueError):
            expected_treatment = {}
        if not isinstance(entry_treatment, dict):
            errors.append("manifest entry treatment is missing")
        else:
            for field in expected_treatment:
                expected_value = resolved.get(field)
                if not _same_json_value(entry_treatment.get(field), expected_value):
                    errors.append(f"manifest entry treatment {field} mismatch")
    return errors


def _validate(result: dict[str, Any], requested_horizon: int) -> dict[str, Any]:
    artifact_path = Path(result["_artifact_path"])
    history = result.get("history", [])
    matching_mode = _matching_mode(result)
    corrected = matching_mode == "corrected_v2"
    errors: list[str] = []
    if result.get("_load_error") is not None:
        errors.append(f"run_result is unreadable: {result['_load_error']}")
    if not isinstance(history, list):
        errors.append("history is not a list")
        history = []
    if result.get("official_unseen_used_for_selection") is not False:
        errors.append("official unseen data was used for selection")
    protocol_value = result.get("protocol", {})
    protocol = protocol_value if isinstance(protocol_value, dict) else {}
    if not isinstance(protocol_value, dict):
        errors.append("protocol identity is missing")
    for key in (
        "validation_used_for_alignment",
        "test_used_for_alignment",
        "text_used_for_predictor",
        "photo_used_for_predictor",
    ):
        if protocol.get(key) is not False:
            errors.append(f"protocol violation: {key}")
    if corrected:
        for key, expected in (
            ("selection_metric", "full_pseudo_unseen_mAP"),
            ("official_unseen_used_for_selection", False),
            ("train_class_scope", "pseudo_train"),
            ("alignment_fit_scope", "pseudo_train_only"),
            ("ranking_positive_reduction", "mean_over_4_positive_photos"),
        ):
            if protocol.get(key) != expected:
                errors.append(f"corrected protocol mismatch: {key}")
    if corrected:
        inference = result.get("inference_contract")
        for field in corrected_pilot_inference_contract_mismatches(inference):
            if field == "<contract>":
                errors.append("corrected inference contract is missing")
            elif field.startswith("unexpected:"):
                errors.append(f"corrected inference contract {field}")
            else:
                errors.append(f"corrected inference contract mismatch: {field}")
    if not history:
        errors.append("run has no history")
    steps: list[int] = []
    valid_rows: list[dict[str, Any]] = []
    hash_checks: list[dict[str, Any]] = []
    for row in history:
        try:
            raw_step = row["training_global_step"]
            if corrected:
                if (
                    isinstance(raw_step, bool)
                    or not isinstance(raw_step, int)
                    or raw_step < 0
                ):
                    raise ValueError(
                        "training_global_step must be a non-negative integer"
                    )
                step = raw_step
            else:
                step = int(raw_step)
            checkpoint = _resolve_run_path(
                result, row["checkpoint"], prefer_artifact=True
            )
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
        if corrected:
            try:
                import torch

                checkpoint_payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )
            except Exception as error:  # noqa: BLE001 - artifact is external input
                errors.append(f"checkpoint payload is unreadable at step {step}: {error}")
                continue
            if not isinstance(checkpoint_payload, dict):
                errors.append(f"checkpoint payload is invalid at step {step}")
                continue
            raw_calibration = result.get("gradient_calibration")
            if result.get("experiment_role") == "alignment_control":
                checkpoint_calibration = None
            elif isinstance(raw_calibration, dict):
                fixed_batches: list[Any] | None = None
                try:
                    calibration_path = _resolve_run_path(
                        result, raw_calibration["artifact"]
                    )
                    calibration_payload = json.loads(calibration_path.read_text())
                    fixed_batches = calibration_payload["calibration"][
                        "fixed_batch_identity"
                    ]["batches"]
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
                replay_count = None if fixed_batches is None else min(step, len(fixed_batches))
                checkpoint_calibration = {
                    "artifact": raw_calibration.get("artifact"),
                    "artifact_sha256": raw_calibration.get("artifact_sha256"),
                    "fixed_batch_identity_sha256": raw_calibration.get(
                        "fixed_batch_identity_sha256"
                    ),
                    "first_batch_replay_verified": None
                    if replay_count is None
                    else replay_count >= 1,
                    "calibration_batch_replay_verified": None
                    if replay_count is None
                    else replay_count == len(fixed_batches),
                    "calibration_batch_replay_count": replay_count,
                    "calibration_batch_replay_expected_count": None
                    if fixed_batches is None
                    else len(fixed_batches),
                    "calibration_batch_replay_prefix_sha256": None
                    if fixed_batches is None
                    else _canonical_hash(fixed_batches[:replay_count]),
                    "worker_lifecycle_verified": raw_calibration.get(
                        "worker_lifecycle_verified"
                    ),
                }
            else:
                checkpoint_calibration = raw_calibration
            checkpoint_metadata = {
                "campaign": result.get("campaign"),
                "experiment_role": result.get("experiment_role"),
                "run_kind": result.get("run_kind"),
                "source_snapshot_hash": result.get("source_snapshot_hash"),
                "training_seed": result.get("training_seed"),
                "data_split_identity": result.get("pseudo_split_identity", result.get("split_identity")),
                "data_manifest_identity": result.get(
                    "manifest_identity", result.get("data_manifest_identity")
                ),
                "manifest_entry_identity": result.get("manifest_entry_identity"),
                "resolved_config": result.get("resolved_config"),
                "resolved_treatment": result.get("resolved_treatment"),
                "optimizer_groups": result.get("optimizer_groups"),
                "initial_model_state_hash": result.get("initial_model_state_hash"),
                "initial_text_bank_state_hash": result.get("initial_text_bank_state_hash"),
                "gradient_calibration_identity": checkpoint_calibration,
            }
            for field, expected in checkpoint_metadata.items():
                actual = checkpoint_payload.get(field)
                if expected is None:
                    if actual is not None:
                        errors.append(f"checkpoint {field} identity mismatch at step {step}")
                elif not _identity_present(expected) or not _identity_present(actual):
                    errors.append(f"checkpoint {field} identity is missing at step {step}")
                elif not _same_config_value(actual, expected):
                    errors.append(f"checkpoint {field} mismatch at step {step}")
            checkpoint_steps = [checkpoint_payload.get(field) for field in ("step", "training_global_step")]
            if (
                any(
                    isinstance(item, bool)
                    or type(item) is not int
                    or item < 0
                    for item in checkpoint_steps
                )
                or any(item != step for item in checkpoint_steps)
            ):
                errors.append(f"checkpoint step mismatch at step {step}")
                continue
            checkpoint_value = checkpoint_payload.get("full_pseudo_unseen_mAP")
            if not _finite_number(checkpoint_value):
                errors.append(f"checkpoint mAP is missing at step {step}")
                continue
            if not _same_float(checkpoint_value, value):
                errors.append(f"checkpoint mAP mismatch at step {step}")
                continue
        valid_rows.append({"step": step, "mAP": value, "checkpoint": str(checkpoint)})
    duplicate_steps = sorted({step for step in steps if steps.count(step) > 1})
    if duplicate_steps:
        errors.append(f"duplicate history steps: {duplicate_steps}")
    if steps != sorted(steps):
        errors.append("history steps are not chronological")
    resolved_value = result.get("resolved_config", {})
    resolved = resolved_value if isinstance(resolved_value, dict) else {}
    configured_horizon = resolved.get("max_steps")
    try:
        configured_horizon_value = int(configured_horizon)
    except (TypeError, ValueError):
        configured_horizon_value = None
    if configured_horizon_value != requested_horizon:
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
    if fixed_status != "VALID":
        errors.append("requested horizon checkpoint is incomplete or duplicated")
    if corrected and any(row["step"] > requested_horizon for row in valid_rows):
        errors.append("history contains post-horizon metrics")
    peak_rows = (
        [row for row in valid_rows if row["step"] <= requested_horizon]
        if corrected
        else valid_rows
    )
    peak_row = min(
        peak_rows,
        key=lambda row: (-row["mAP"], row["step"]),
        default=None,
    )
    peak_value = None if peak_row is None else peak_row["mAP"]
    peak_step = None if peak_row is None else peak_row["step"]
    initialization = _step0_metadata(result, strict_steps=corrected)
    if initialization.get("status") != "VERIFIED":
        errors.append(
            f"step-0 initialization is not verified: {initialization.get('status')}"
        )
    source_hash = result.get("source_snapshot_hash")
    if source_hash is None:
        provenance = result.get("provenance")
        source_snapshot = (
            provenance.get("source_snapshot")
            if isinstance(provenance, dict)
            else None
        )
        source_hash = (
            source_snapshot.get("sha256")
            if isinstance(source_snapshot, dict)
            else None
        )
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
    if corrected:
        if not isinstance(split, dict) or split.get("sha256") != _split_identity_hash(split):
            errors.append("pseudo split identity hash mismatch")
    treatment = result.get("resolved_treatment", {})
    config_hash = _canonical_hash(resolved)
    seed_value = result.get("training_seed", result.get("seed"))
    try:
        if corrected and (
            isinstance(seed_value, bool) or type(seed_value) is not int
        ):
            raise ValueError("training seed must be an integer")
        seed = None if seed_value is None else int(seed_value)
    except (TypeError, ValueError):
        seed = None
        errors.append("training seed is invalid")
    if corrected and seed is not None:
        pseudo_seed = resolved.get("pseudo_val_seed", result.get("pseudo_validation_seed"))
        try:
            if isinstance(pseudo_seed, bool) or type(pseudo_seed) is not int:
                raise ValueError
            expected_treatment = treatment_for_role(
                str(result.get("experiment_role")),
                seed=seed,
                pseudo_val_seed=pseudo_seed,
            )
        except (TypeError, ValueError):
            expected_treatment = None
            errors.append("corrected treatment identity is invalid")
        if expected_treatment is not None:
            for key, expected_value in expected_treatment.items():
                configured_value = resolved.get(key)
                if key != "lambda_alignment_mean" and configured_value != expected_value:
                    errors.append(f"corrected treatment config mismatch: {key}")
                if key == "lambda_alignment_mean" and str(result.get("experiment_role")) == "alignment_control" and configured_value != 0.0:
                    errors.append("corrected treatment config mismatch: lambda_alignment_mean")
                recorded_value = treatment.get(key) if isinstance(treatment, dict) else None
                if not _same_config_value(recorded_value, configured_value):
                    errors.append(f"resolved treatment mismatch: {key}")
    manifest_identity = result.get("manifest_identity")
    if manifest_identity is None:
        manifest_identity = result.get("data_manifest_identity")
    backbone_identity = initialization.get("backbone_identity")
    if backbone_identity is None:
        backbone_identity = result.get("backbone_identity")
    if corrected:
        checkpoint_metadata = (
            ("campaign", result.get("campaign"), initialization.get("campaign")),
            ("experiment_role", result.get("experiment_role"), initialization.get("experiment_role")),
            ("run_kind", result.get("run_kind"), initialization.get("run_kind")),
            ("initial_model_state_hash", result.get("initial_model_state_hash"), initialization.get("initial_model_state_hash")),
            ("initial_text_bank_state_hash", result.get("initial_text_bank_state_hash"), initialization.get("initial_text_bank_state_hash")),
            ("source_snapshot_hash", result.get("source_snapshot_hash"), initialization.get("source_snapshot_hash")),
            ("initialization_identity", result.get("initialization_identity"), {"model_state_hash": initialization.get("initial_model_state_hash"), "text_bank_state_hash": initialization.get("initial_text_bank_state_hash")}),
            ("training_seed", seed, initialization.get("training_seed")),
            ("pseudo_split_identity", split, initialization.get("split_identity")),
            ("manifest_identity", manifest_identity, initialization.get("manifest_identity")),
            ("manifest_entry_identity", result.get("manifest_entry_identity"), initialization.get("manifest_entry_identity")),
            ("resolved_config", resolved, initialization.get("resolved_config")),
            ("resolved_treatment", result.get("resolved_treatment"), initialization.get("resolved_treatment")),
            ("optimizer_groups", result.get("optimizer_groups"), initialization.get("optimizer_identity")),
        )
        for field, result_value, checkpoint_value in checkpoint_metadata:
            if not _identity_present(result_value) or not _identity_present(checkpoint_value):
                errors.append(f"checkpoint {field} identity is missing")
            elif not _same_config_value(result_value, checkpoint_value):
                errors.append(f"checkpoint {field} mismatch")
        errors.extend(_manifest_entry_errors(result))
        try:
            expected_split_identity, expected_manifest_identity = _recompute_data_identities(
                result
            )
        except Exception as error:  # noqa: BLE001 - artifact metadata is external input
            errors.append(f"configured data identities cannot be recomputed: {error}")
        else:
            if split != expected_split_identity:
                errors.append("pseudo split identity does not match configured manifests")
            if manifest_identity != expected_manifest_identity:
                errors.append("data manifest identity does not match configured manifests")
        result_calibration = result.get("gradient_calibration")
        checkpoint_calibration = initialization.get("gradient_calibration_identity")
        if result.get("experiment_role") == "alignment_control":
            if result_calibration is not None or checkpoint_calibration is not None:
                errors.append("control calibration identity is not absent")
        else:
            calibration_fields = (
                "artifact",
                "artifact_sha256",
                "fixed_batch_identity_sha256",
                "worker_lifecycle_verified",
            )
            if not isinstance(result_calibration, dict) or not isinstance(
                checkpoint_calibration, dict
            ):
                errors.append("checkpoint calibration identity is missing")
            else:
                for field in calibration_fields:
                    result_value = result_calibration.get(field)
                    checkpoint_value = checkpoint_calibration.get(field)
                    if not _identity_present(result_value) or not _identity_present(
                        checkpoint_value
                    ):
                        errors.append(f"checkpoint calibration {field} identity is missing")
                    elif not _same_config_value(result_value, checkpoint_value):
                        errors.append(f"checkpoint calibration {field} mismatch")
    calibration_validation = _calibration_validation(result)
    if corrected:
        if result.get("experiment_role") == "alignment_control":
            if calibration_validation.get("status") != "ABSENT":
                errors.append("control calibration evidence is unverified")
        elif calibration_validation.get("status") != "VERIFIED":
            errors.append("calibration evidence is unverified")
    if seed is None:
        errors.append("training seed is missing")
    run_status = "VALID"
    if any(
        "mismatch" in error
        or "run_result is unreadable" in error
        or "history is not a list" in error
        or "missing checkpoint" in error
        or (
            "checkpoint " in error
            and error != "requested horizon checkpoint is incomplete or duplicated"
        )
        or "manifest" in error
        or "calibration evidence is unverified" in error
        or "inference contract" in error
        or "not chronological" in error
        for error in errors
    ):
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
        "matching_mode": matching_mode,
        "experiment_role": result.get("experiment_role"),
        "run_kind": result.get("run_kind"),
        "training_seed": None if seed is None else int(seed),
        "split_identity": split,
        "split_identity_hash": split_hash,
        "initialization": initialization,
        "initialization_regime": initialization.get("initial_model_state_hash", "UNVERIFIED"),
        "text_bank_initialization_hash": initialization.get(
            "initial_text_bank_state_hash", "UNVERIFIED"
        ),
        "scheduler_identity": initialization.get("scheduler_identity"),
        "source_hash": source_hash,
        "config_hash": config_hash,
        "resolved_config": resolved,
        "resolved_treatment": treatment,
        "dataset": result.get("dataset", resolved.get("dataset")),
        "manifest_identity": manifest_identity,
        "manifest_entry_identity": result.get("manifest_entry_identity"),
        "backbone_identity": backbone_identity,
        "sampler_identity": result.get("matched_sampler"),
        "optimizer_identity": result.get("optimizer_groups"),
        "calibration_validation": calibration_validation,
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
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    excluded_keys: set[str] | frozenset[str] = frozenset(),
) -> dict[str, tuple[Any, Any]]:
    control_config = control.get("resolved_config", {})
    candidate_config = candidate.get("resolved_config", {})
    if not isinstance(control_config, dict):
        control_config = {}
    if not isinstance(candidate_config, dict):
        candidate_config = {}
    keys = (set(control_config) | set(candidate_config)) - set(excluded_keys)
    return {
        key: (control_config.get(key), candidate_config.get(key))
        for key in sorted(keys)
        if not _same_config_value(
            control_config.get(key), candidate_config.get(key)
        )
    }


def _difference_records(
    differences: dict[str, tuple[Any, Any]],
) -> list[dict[str, Any]]:
    return [
        {"field": key, "control": values[0], "candidate": values[1]}
        for key, values in sorted(differences.items())
    ]


def _provenance_complete(run: dict[str, Any]) -> bool:
    split = run.get("split_identity", {})
    protocol = run.get("evaluation_protocol", {})
    return bool(
        run.get("source_hash")
        and run.get("training_seed") is not None
        and run.get("training_horizon") is not None
        and isinstance(split, dict)
        and split.get("sha256")
        and run.get("initialization_regime") not in (None, "UNVERIFIED")
        and all(value is not None for value in protocol.values())
    )


def _identity_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, str)):
        return bool(value)
    return True


def _strict_identity_checks(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[list[str], dict[str, tuple[Any, Any]]]:
    fields = (
        "dataset",
        "manifest_identity",
        "backbone_identity",
        "text_bank_initialization_hash",
        "scheduler_identity",
        "sampler_identity",
        "optimizer_identity",
    )
    missing: list[str] = []
    different: dict[str, tuple[Any, Any]] = {}
    for field in fields:
        left_value = left.get(field)
        right_value = right.get(field)
        if not _identity_present(left_value) or not _identity_present(right_value):
            missing.append(field)
        elif _canonical_hash(left_value) != _canonical_hash(right_value):
            different[field] = (left_value, right_value)
    left_entry = left.get("manifest_entry_identity")
    right_entry = right.get("manifest_entry_identity")
    if not isinstance(left_entry, dict) or not isinstance(right_entry, dict):
        missing.append("manifest_entry_identity")
    elif not _identity_present(left_entry.get("manifest_sha256")) or not _identity_present(
        right_entry.get("manifest_sha256")
    ):
        missing.append("manifest_entry_identity.manifest_sha256")
    elif left_entry["manifest_sha256"] != right_entry["manifest_sha256"]:
        different["manifest_entry_identity.manifest_sha256"] = (
            left_entry["manifest_sha256"],
            right_entry["manifest_sha256"],
        )
    return missing, different


def _corrected_config_checks(
    control: dict[str, Any],
    candidate: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, tuple[Any, Any]], list[str], list[dict[str, Any]]]:
    excluded_keys = (
        {CALIBRATION_CONFIG_KEY}
        if spec.get("pair_id") in {"R-MD", "R-MS"}
        else frozenset()
    )
    differences = _config_differences(
        control, candidate, excluded_keys=excluded_keys
    )
    control_config = control.get("resolved_config", {})
    candidate_config = candidate.get("resolved_config", {})
    if not isinstance(control_config, dict):
        control_config = {}
    if not isinstance(candidate_config, dict):
        candidate_config = {}
    missing = [
        key
        for key in CORRECTED_PILOT_REQUIRED_CONFIG_KEYS
        if key not in control_config or key not in candidate_config
    ]
    for key in missing:
        differences.setdefault(
            key, (control_config.get(key, "UNVERIFIED"), candidate_config.get(key, "UNVERIFIED"))
        )
    allowed = set(spec.get("allowed_config_differences", ()))
    disallowed = sorted(set(differences) - allowed)
    mismatches = _difference_records(
        {key: differences[key] for key in disallowed}
    )
    for key in spec.get("equal_config_keys", ()):
        left_value = control_config.get(key)
        right_value = candidate_config.get(key)
        if not _same_config_value(left_value, right_value) and key not in disallowed:
            disallowed.append(key)
            mismatches.append(
                {"field": key, "control": left_value, "candidate": right_value}
            )
    disallowed.sort()
    role_mismatches: list[dict[str, Any]] = []
    pair_id = spec.get("pair_id")
    expected_roles = (
        ("alignment_control", candidate.get("experiment_role"))
        if pair_id in {"R-MD", "R-MS"}
        else ("alignment_mean_text_log", "alignment_mean_text_log_symmetric")
    )
    if control.get("experiment_role") != expected_roles[0]:
        role_mismatches.append(
            _mismatch(
                "experiment_role",
                expected_roles[0],
                control.get("experiment_role"),
                "corrected pilot left role is invalid",
            )
        )
    if candidate.get("experiment_role") != expected_roles[1]:
        role_mismatches.append(
            _mismatch(
                "experiment_role",
                expected_roles[1],
                candidate.get("experiment_role"),
                "corrected pilot right role is invalid",
            )
        )
    expected_config_roles = (
        ("alignment_control", expected_roles[1])
        if pair_id in {"R-MD", "R-MS"}
        else ("alignment_mean_text_log", "alignment_mean_text_log_symmetric")
    )
    for run, expected_role, side in zip(
        (control, candidate), expected_config_roles, ("control", "candidate"), strict=True
    ):
        config = run.get("resolved_config", {})
        if not isinstance(config, dict):
            config = {}
        for field, expected in (
            ("experiment_campaign", run.get("campaign")),
            ("experiment_role", expected_role),
            ("run_kind", run.get("run_kind")),
            ("seed", run.get("training_seed")),
            (
                "pseudo_val_seed",
                run.get("split_identity", {}).get(
                    "pseudo_validation_seed", run.get("split_identity", {}).get("seed")
                )
                if isinstance(run.get("split_identity"), dict)
                else None,
            ),
        ):
            if config.get(field) != expected:
                role_mismatches.append(
                    _mismatch(
                        field,
                        expected,
                        config.get(field),
                        f"{side} corrected pilot metadata is invalid",
                    )
                )
        if config.get("lambda_alignment_covariance") != 0:
            role_mismatches.append(
                _mismatch(
                    "lambda_alignment_covariance",
                    0.0,
                    config.get("lambda_alignment_covariance"),
                    f"{side} lambda_alignment_covariance must be zero",
                )
            )
        if config.get("alignment_geometry") != "log_map":
            role_mismatches.append(
                _mismatch(
                    "alignment_geometry",
                    "log_map",
                    config.get("alignment_geometry"),
                    f"{side} corrected pilot alignment geometry is invalid",
                )
            )
        if config.get("alignment_anchor") != "text":
            role_mismatches.append(
                _mismatch(
                    "alignment_anchor",
                    "text",
                    config.get("alignment_anchor"),
                    f"{side} corrected pilot alignment anchor is invalid",
                )
            )
    if pair_id in {"R-MD", "R-MS"} and control_config.get("lambda_alignment_mean") != 0:
        role_mismatches.append(
            _mismatch(
                "lambda_alignment_mean",
                0.0,
                control_config.get("lambda_alignment_mean"),
                "R lambda_alignment_mean must be zero",
            )
        )
    expected_gradients = (
        "detached",
        "symmetric"
        if candidate.get("experiment_role") == "alignment_mean_text_log_symmetric"
        else "detached",
    )
    for run, expected_gradient, side in zip(
        (control, candidate), expected_gradients, ("control", "candidate"), strict=True
    ):
        config = run.get("resolved_config", {})
        if isinstance(config, dict) and config.get("alignment_target_gradient") != expected_gradient:
            role_mismatches.append(
                _mismatch(
                    "alignment_target_gradient",
                    expected_gradient,
                    config.get("alignment_target_gradient"),
                    f"{side} alignment target-gradient policy is invalid",
                )
            )
    return differences, disallowed, mismatches + role_mismatches


def _calibration_pair_mismatches(
    run: dict[str, Any], side: str
) -> list[dict[str, Any]]:
    validation = run.get("calibration_validation", {})
    return [
        {**item, "side": side}
        for item in validation.get("mismatches", [])
        if isinstance(item, dict)
    ]


def _calibration_artifact_pair_mismatches(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    left_validation = left.get("calibration_validation", {})
    right_validation = right.get("calibration_validation", {})
    mismatches: list[dict[str, Any]] = []
    left_path = left_validation.get("artifact_path")
    right_path = right_validation.get("artifact_path")
    if not left_path or not right_path or Path(left_path).resolve() != Path(right_path).resolve():
        mismatches.append(
            _mismatch(
                "calibration_artifact.canonical_path",
                right_path,
                left_path,
                "MD/MS calibration artifacts are not the same canonical file",
            )
        )
    left_hash = left_validation.get("artifact_sha256")
    right_hash = right_validation.get("artifact_sha256")
    if not left_hash or not right_hash or left_hash != right_hash:
        mismatches.append(
            _mismatch(
                "calibration_artifact.artifact_sha256",
                right_hash,
                left_hash,
                "MD/MS calibration artifact SHA-256 values differ",
            )
        )
    return mismatches


def _pair(
    candidate: dict[str, Any], controls: list[dict[str, Any]]
) -> dict[str, Any]:
    role = candidate.get("experiment_role")
    pair_id = (
        "R-MD"
        if role == "alignment_mean_text_log"
        else "R-MS"
        if role == "alignment_mean_text_log_symmetric"
        else f"R-{role}"
    )
    same_identity = [
        control for control in controls if _matching_key(control) == _matching_key(candidate)
    ]
    if not same_identity:
        return {
            "pair_id": pair_id,
            "status": "UNMATCHED",
            "control_artifact": None,
            "control_seed": None,
            "paired_delta": None,
            "peak_delta": None,
            "reason": "no control with identical campaign/seed/split/init/horizon/protocol",
        }
    same_identity.sort(key=lambda run: run["artifact_path"])
    control = same_identity[0]
    corrected = candidate.get("matching_mode") == "corrected_v2"
    spec = CORRECTED_PILOT_PAIR_SPECS.get(
        (control.get("experiment_role"), candidate.get("experiment_role"))
    ) if corrected else None
    if corrected and spec is None:
        return {
            "pair_id": pair_id,
            "status": "UNMATCHED_CONFIG",
            "control_artifact": control["artifact_path"],
            "control_seed": control["training_seed"],
            "paired_delta": None,
            "peak_delta": None,
            "reason": "role is outside the corrected mean-only pilot specification",
            "config_differences": {},
            "disallowed_config_differences": ["experiment_role"],
            "config_mismatches": [],
            "calibration_mismatches": [],
        }
    if corrected:
        differences, disallowed, config_mismatches = _corrected_config_checks(
            control, candidate, spec
        )
    else:
        differences = _config_differences(
            control, candidate, excluded_keys={CALIBRATION_CONFIG_KEY}
        )
        allowed = HISTORICAL_ALIGNMENT_ALLOWED_CONFIG_KEYS | {
            "experiment_role",
            "experiment_name",
        }
        disallowed = sorted(set(differences) - allowed)
        config_mismatches = _difference_records(
            {key: differences[key] for key in disallowed}
        )
    common = {
        "pair_id": spec.get("pair_id", pair_id) if spec else pair_id,
        "control_artifact": control["artifact_path"],
        "control_seed": control["training_seed"],
        "config_differences": differences,
        "disallowed_config_differences": disallowed,
        "config_mismatches": config_mismatches,
        "source_hashes": [control["source_hash"], candidate["source_hash"]],
        "duplicate_control_count": len(same_identity),
    }
    if len(same_identity) > 1:
        return {
            **common,
            "status": "DUPLICATE_CONTROL",
            "paired_delta": None,
            "peak_delta": None,
            "reason": "ambiguous duplicate controls; delta suppressed",
            "calibration_mismatches": [],
        }
    if not _provenance_complete(control) or not _provenance_complete(candidate):
        return {
            **common,
            "status": "UNMATCHED_PROVENANCE",
            "paired_delta": None,
            "peak_delta": None,
            "reason": "source/split/initialization/protocol evidence is incomplete",
            "calibration_mismatches": [],
        }
    if control["source_hash"] != candidate["source_hash"]:
        return {
            **common,
            "status": "UNMATCHED_SOURCE",
            "paired_delta": None,
            "peak_delta": None,
            "reason": "source snapshot hashes differ",
            "calibration_mismatches": [],
        }
    if corrected:
        missing_identity, different_identity = _strict_identity_checks(control, candidate)
        if missing_identity:
            return {
                **common,
                "status": "UNMATCHED_PROVENANCE",
                "paired_delta": None,
                "peak_delta": None,
                "reason": "dataset/manifest/backbone/sampler/optimizer evidence is incomplete",
                "identity_mismatches": missing_identity,
                "calibration_mismatches": [],
            }
        if different_identity:
            return {
                **common,
                "status": "UNMATCHED_CONFIG",
                "paired_delta": None,
                "peak_delta": None,
                "reason": "mandatory dataset/model/sampler identities differ",
                "identity_mismatches": _difference_records(different_identity),
                "calibration_mismatches": [],
            }
    if disallowed or config_mismatches:
        return {
            **common,
            "status": "UNMATCHED_CONFIG",
            "paired_delta": None,
            "peak_delta": None,
            "reason": "config differs outside the pair specification",
            "calibration_mismatches": [],
        }
    calibration_mismatches = (
        _calibration_pair_mismatches(control, "control")
        + _calibration_pair_mismatches(candidate, "candidate")
        if corrected
        else []
    )
    if calibration_mismatches:
        return {
            **common,
            "status": "UNMATCHED_CALIBRATION",
            "paired_delta": None,
            "peak_delta": None,
            "reason": "calibration evidence is UNVERIFIED",
            "calibration_mismatches": calibration_mismatches,
        }
    if control["status"] == "ARTIFACT_INVALID" or candidate["status"] == "ARTIFACT_INVALID":
        return {
            **common,
            "status": "ARTIFACT_INVALID",
            "paired_delta": None,
            "peak_delta": None,
            "reason": "invalid run artifact; delta suppressed",
            "calibration_mismatches": [],
        }
    if control["status"] != "VALID" or candidate["status"] != "VALID":
        return {
            **common,
            "status": "INCOMPLETE",
            "paired_delta": None,
            "peak_delta": None,
            "reason": "run is incomplete; delta suppressed",
            "calibration_mismatches": [],
        }
    if control["fixed_step"]["mAP"] is None or candidate["fixed_step"]["mAP"] is None:
        status = "INCOMPLETE"
        delta = None
    else:
        status = "MATCHED"
        delta = candidate["fixed_step"]["mAP"] - control["fixed_step"]["mAP"]
    peak_delta = None
    if (
        status == "MATCHED"
        and control["peak"]["mAP"] is not None
        and candidate["peak"]["mAP"] is not None
    ):
        peak_delta = candidate["peak"]["mAP"] - control["peak"]["mAP"]
    return {
        **common,
        "status": status,
        "paired_delta": delta,
        "peak_delta": peak_delta,
        "calibration_mismatches": [],
    }


def _direct_corrected_pair(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    spec = CORRECTED_PILOT_PAIR_SPECS[(
        "alignment_mean_text_log",
        "alignment_mean_text_log_symmetric",
    )]
    differences, disallowed, config_mismatches = _corrected_config_checks(
        left, right, spec
    )
    common = {
        "pair_id": "MD-MS",
        "left_artifact": left["artifact_path"],
        "right_artifact": right["artifact_path"],
        "left_seed": left.get("training_seed"),
        "right_seed": right.get("training_seed"),
        "config_differences": differences,
        "disallowed_config_differences": disallowed,
        "config_mismatches": config_mismatches,
        "calibration_mismatches": _calibration_pair_mismatches(left, "left")
        + _calibration_pair_mismatches(right, "right")
        + _calibration_artifact_pair_mismatches(left, right),
    }
    if _matching_key(left) != _matching_key(right):
        return {
            **common,
            "status": "UNMATCHED",
            "reason": "MD/MS have different campaign/seed/split/init/horizon/protocol",
        }
    if not _provenance_complete(left) or not _provenance_complete(right):
        return {
            **common,
            "status": "UNMATCHED_PROVENANCE",
            "reason": "source/split/initialization/protocol evidence is incomplete",
        }
    if left["source_hash"] != right["source_hash"]:
        return {
            **common,
            "status": "UNMATCHED_SOURCE",
            "reason": "source snapshot hashes differ",
        }
    missing_identity, different_identity = _strict_identity_checks(left, right)
    if missing_identity:
        return {
            **common,
            "status": "UNMATCHED_PROVENANCE",
            "reason": "dataset/manifest/backbone/sampler/optimizer evidence is incomplete",
            "identity_mismatches": missing_identity,
        }
    if different_identity or disallowed or config_mismatches:
        return {
            **common,
            "status": "UNMATCHED_CONFIG",
            "reason": "MD/MS config or mandatory identity differs",
            "identity_mismatches": _difference_records(different_identity),
        }
    if common["calibration_mismatches"]:
        return {
            **common,
            "status": "UNMATCHED_CALIBRATION",
            "reason": "calibration evidence is UNVERIFIED",
        }
    if left["status"] != "VALID" or right["status"] != "VALID":
        return {
            **common,
            "status": "INCOMPLETE",
            "reason": "run is incomplete; comparison is unverified",
        }
    return {**common, "status": "MATCHED", "reason": "MD/MS specification verified"}



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
    campaign = runs[0]["campaign"] if runs else None
    corrected = bool(runs) and runs[0].get("matching_mode") == "corrected_v2"
    controls = [run for run in runs if run["experiment_role"] == "alignment_control"]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_role.setdefault(str(run["experiment_role"]), []).append(run)
    if corrected:
        for role in CORRECTED_PILOT_ROLES:
            by_role.setdefault(role, [])
    for values in by_role.values():
        values.sort(key=lambda run: run["artifact_path"])
    unexpected_roles = (
        sorted(set(by_role) - set(CORRECTED_PILOT_ROLES)) if corrected else []
    )
    missing_arms: list[dict[str, Any]] = []
    duplicate_arms: list[dict[str, Any]] = []
    if corrected:
        all_seeds = sorted(
            {
                run["training_seed"]
                for run in runs
                if run.get("training_seed") is not None
            }
        )
        for role in CORRECTED_PILOT_ROLES:
            role_seeds = {
                run["training_seed"]
                for run in by_role[role]
                if run.get("training_seed") is not None
            }
            for seed in all_seeds or [None]:
                if seed not in role_seeds:
                    missing_arms.append(
                        {
                            "pair_id": "R-MD"
                            if role == "alignment_mean_text_log"
                            else "R-MS"
                            if role == "alignment_mean_text_log_symmetric"
                            else "R",
                            "role": role,
                            "training_seed": seed,
                            "status": "UNMATCHED",
                            "reason": f"missing corrected pilot arm: {role}",
                        }
                    )
        for role in CORRECTED_PILOT_ROLES:
            by_seed: dict[int, list[str]] = {}
            for run in by_role[role]:
                seed = run.get("training_seed")
                if seed is not None:
                    by_seed.setdefault(int(seed), []).append(run["artifact_path"])
            for seed, artifacts in sorted(by_seed.items()):
                if len(artifacts) > 1:
                    duplicate_arms.append(
                        {
                            "role": role,
                            "training_seed": seed,
                            "artifacts": artifacts,
                            "status": "DUPLICATE_ARM",
                            "reason": f"multiple {role} runs for one training seed",
                        }
                    )
    pairs: dict[str, list[dict[str, Any]]] = {}
    candidate_roles = (
        CORRECTED_PILOT_ROLES[1:] if corrected else tuple(role for role in by_role if role != "alignment_control")
    )
    for role in candidate_roles:
        pairs[role] = []
        values = by_role.get(role, [])
        for run in values:
            pair = {
                "artifact_path": run["artifact_path"],
                "training_seed": run["training_seed"],
                **_pair(run, controls),
            }
            if corrected:
                duplicate_arm = [
                    other
                    for other in values
                    if other is not run
                    and other.get("training_seed") == run.get("training_seed")
                ]
                duplicate_control = [
                    control
                    for control in controls
                    if control.get("training_seed") == run.get("training_seed")
                ]
                if duplicate_arm:
                    pair.update(
                        {
                            "status": "DUPLICATE_ARM",
                            "paired_delta": None,
                            "peak_delta": None,
                            "reason": f"ambiguous duplicate {role} arm; delta suppressed",
                        }
                    )
                elif len(duplicate_control) > 1:
                    pair.update(
                        {
                            "status": "DUPLICATE_CONTROL",
                            "paired_delta": None,
                            "peak_delta": None,
                            "reason": "ambiguous duplicate control arm; delta suppressed",
                        }
                    )
            pairs[role].append(pair)
        if corrected and not values:
            matching_controls = controls or [None]
            for control in matching_controls:
                pair_id = "R-MD" if role == "alignment_mean_text_log" else "R-MS"
                pairs[role].append(
                    {
                        "artifact_path": None,
                        "training_seed": None if control is None else control["training_seed"],
                        "pair_id": pair_id,
                        "status": "UNMATCHED",
                        "control_artifact": None if control is None else control["artifact_path"],
                        "control_seed": None if control is None else control["training_seed"],
                        "paired_delta": None,
                        "peak_delta": None,
                        "reason": f"missing corrected pilot arm: {role}",
                        "config_differences": {},
                        "disallowed_config_differences": [],
                        "config_mismatches": [],
                        "calibration_mismatches": [],
                    }
                )

    pairwise_comparisons: list[dict[str, Any]] = []
    if corrected:
        md_by_seed: dict[int, list[dict[str, Any]]] = {}
        ms_by_seed: dict[int, list[dict[str, Any]]] = {}
        for run in by_role["alignment_mean_text_log"]:
            if run.get("training_seed") is not None:
                md_by_seed.setdefault(int(run["training_seed"]), []).append(run)
        for run in by_role["alignment_mean_text_log_symmetric"]:
            if run.get("training_seed") is not None:
                ms_by_seed.setdefault(int(run["training_seed"]), []).append(run)
        for seed in sorted(set(md_by_seed) | set(ms_by_seed)):
            left_values = md_by_seed.get(seed, [])
            right_values = ms_by_seed.get(seed, [])
            if len(left_values) != 1 or len(right_values) != 1:
                pairwise_comparisons.append(
                    {
                        "pair_id": "MD-MS",
                        "status": "DUPLICATE_ARM"
                        if len(left_values) > 1 or len(right_values) > 1
                        else "UNMATCHED",
                        "left_artifact": None
                        if not left_values
                        else left_values[0]["artifact_path"],
                        "right_artifact": None
                        if not right_values
                        else right_values[0]["artifact_path"],
                        "left_seed": seed,
                        "right_seed": seed,
                        "reason": "ambiguous duplicate or missing MD/MS arm",
                        "paired_delta": None,
                    }
                )
            else:
                left = left_values[0]
                right = right_values[0]
                matching_controls = [
                    control
                    for control in controls
                    if _matching_key(control) == _matching_key(left)
                ]
                if not matching_controls:
                    pairwise_comparisons.append(
                        {
                            "pair_id": "MD-MS",
                            "status": "UNMATCHED",
                            "left_artifact": left["artifact_path"],
                            "right_artifact": right["artifact_path"],
                            "left_seed": seed,
                            "right_seed": seed,
                            "reason": "missing control arm with identical pairing identity",
                            "paired_delta": None,
                        }
                    )
                elif len(matching_controls) > 1:
                    pairwise_comparisons.append(
                        {
                            "pair_id": "MD-MS",
                            "status": "DUPLICATE_CONTROL",
                            "left_artifact": left["artifact_path"],
                            "right_artifact": right["artifact_path"],
                            "left_seed": seed,
                            "right_seed": seed,
                            "reason": "ambiguous matching control arm",
                            "paired_delta": None,
                        }
                    )
                elif (
                    not _provenance_complete(matching_controls[0])
                    or matching_controls[0]["status"] != "VALID"
                ):
                    pairwise_comparisons.append(
                        {
                            "pair_id": "MD-MS",
                            "status": "UNMATCHED_PROVENANCE",
                            "left_artifact": left["artifact_path"],
                            "right_artifact": right["artifact_path"],
                            "left_seed": seed,
                            "right_seed": seed,
                            "reason": "matching control arm is unverified",
                            "paired_delta": None,
                        }
                    )
                elif any(
                    matching_controls[0]["source_hash"] != run["source_hash"]
                    for run in (left, right)
                ):
                    pairwise_comparisons.append(
                        {
                            "pair_id": "MD-MS",
                            "status": "UNMATCHED_SOURCE",
                            "left_artifact": left["artifact_path"],
                            "right_artifact": right["artifact_path"],
                            "left_seed": seed,
                            "right_seed": seed,
                            "reason": "matching R/MD/MS source snapshots differ",
                            "paired_delta": None,
                        }
                    )
                else:
                    pairwise_comparisons.append(_direct_corrected_pair(left, right))

    if corrected and unexpected_roles:
        for role in CORRECTED_PILOT_ROLES[1:]:
            for pair in pairs[role]:
                if pair.get("status") == "MATCHED":
                    pair.update(
                        {
                            "status": "UNMATCHED_CONFIG",
                            "paired_delta": None,
                            "peak_delta": None,
                            "reason": "unexpected role in corrected pilot campaign",
                        }
                    )
        for comparison in pairwise_comparisons:
            if comparison.get("status") == "MATCHED":
                comparison.update(
                    {
                        "status": "UNMATCHED_CONFIG",
                        "paired_delta": None,
                        "reason": "unexpected role in corrected pilot campaign",
                    }
                )

    # Missing/duplicate/invalid arms suppress only their own pair; complete
    # seeds remain useful aggregate evidence. Unexpected roles invalidate the
    # whole corrected campaign above.
    suppress_campaign_deltas = corrected and bool(unexpected_roles)
    summaries: list[dict[str, Any]] = []
    for role in sorted(by_role):
        values = by_role[role]
        deltas = [] if suppress_campaign_deltas else [
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
    if corrected and not controls:
        pairs["alignment_control"] = [
            {
                "artifact_path": None,
                "training_seed": None,
                "pair_id": "R",
                "status": "UNMATCHED",
                "control_artifact": None,
                "control_seed": None,
                "paired_delta": None,
                "peak_delta": None,
                "reason": "missing corrected pilot arm: alignment_control",
                "config_differences": {},
                "disallowed_config_differences": [],
                "config_mismatches": [],
                "calibration_mismatches": [],
            }
        ]
    pair_statuses = [
        pair.get("status")
        for role_pairs in pairs.values()
        for pair in role_pairs
    ] + [comparison.get("status") for comparison in pairwise_comparisons]
    campaign_status = (
        "UNMATCHED_CONFIG"
        if unexpected_roles
        else "VALID"
        if pair_statuses
        and not missing_arms
        and not duplicate_arms
        and all(status == "MATCHED" for status in pair_statuses)
        and all(run["status"] == "VALID" for run in runs)
        else "UNMATCHED"
    )
    return {
        "campaign": campaign,
        "matching_mode": runs[0].get("matching_mode", "historical") if runs else "historical",
        "requested_horizon": horizon,
        "run_count": len(runs),
        "runs": runs,
        "pairs": pairs,
        "pairwise_comparisons": pairwise_comparisons,
        "missing_arms": missing_arms,
        "duplicate_arms": duplicate_arms,
        "unexpected_roles": unexpected_roles,
        "campaign_status": campaign_status,
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
            f"## Campaign `{campaign['campaign']}` — requested horizon `{campaign['requested_horizon']}` "
            f"— status `{campaign.get('campaign_status', 'UNKNOWN')}`",
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
        for missing in campaign.get("missing_arms", []):
            lines.append(
                f"- `{missing['pair_id']}` seed {missing['training_seed']}: "
                f"UNMATCHED — {missing['reason']}"
            )
        for duplicate in campaign.get("duplicate_arms", []):
            lines.append(
                f"- `{duplicate['role']}` seed {duplicate['training_seed']}: "
                f"DUPLICATE_ARM — {duplicate['reason']}"
            )
        if campaign.get("unexpected_roles"):
            lines.append(
                "- UNMATCHED_CONFIG — unexpected corrected-pilot roles: "
                + ", ".join(campaign["unexpected_roles"])
            )
        for comparison in campaign.get("pairwise_comparisons", []):
            if comparison.get("status") != "MATCHED":
                lines.append(
                    f"- `{comparison.get('pair_id')}`: {comparison.get('status')} — "
                    f"{comparison.get('reason', '')}"
                )
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
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in validated:
        key = (str(run["campaign"]), str(run["matching_mode"]))
        grouped.setdefault(key, []).append(run)
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
            "duplicate_rule": "ambiguous duplicate R/MD/MS arms are reported and paired deltas are suppressed",
            "horizon_rule": "explicit --horizon; missing final step is INCOMPLETE",
            "comparison_specifications": {
                "corrected_mean_only_pilot": {
                    pair_id: {
                        "left": left,
                        "right": right,
                        "allowed_config_differences": sorted(
                            spec["allowed_config_differences"]
                        ),
                        "equal_config_keys": sorted(spec.get("equal_config_keys", ())),
                    }
                    for (left, right), spec in CORRECTED_PILOT_PAIR_SPECS.items()
                    for pair_id in [spec["pair_id"]]
                },
                "historical_alignment": {
                    "allowed_config_differences": sorted(
                        HISTORICAL_ALIGNMENT_ALLOWED_CONFIG_KEYS
                        | {"experiment_role", "experiment_name"}
                    ),
                    "calibration_artifact": "not part of historical matching",
                },
            },
            "calibration_artifact_policy": "corrected pilot references are verified, not ignored",
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
