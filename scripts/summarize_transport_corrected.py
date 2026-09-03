"""Integrity-first transport summary generated only from structured raw runs."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

try:
    from scripts.transport_artifact_utils import (
        ArtifactIntegrityError,
        ROOT,
        assert_matched_runs,
        assert_same_run_conditions,
        collect_runs,
        config_of,
        explicit_transport_enabled,
        factorial_effects,
        missing_result,
        point_map,
        repository_provenance,
        select_unique_role,
        validate_freeze_optimizer_artifact,
        validate_freeze_optimizer_role,
        write_new,
    )
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from transport_artifact_utils import (
        ArtifactIntegrityError,
        ROOT,
        assert_matched_runs,
        assert_same_run_conditions,
        collect_runs,
        config_of,
        explicit_transport_enabled,
        factorial_effects,
        missing_result,
        point_map,
        repository_provenance,
        select_unique_role,
        validate_freeze_optimizer_artifact,
        validate_freeze_optimizer_role,
        write_new,
    )

CAMPAIGN = "transport_corrected_2026-09-02_v2"
FACTORIAL_ROLES = {
    "A": "endpoint0_factorial_A",
    "B": "endpoint0_factorial_B",
    "C": "endpoint0_factorial_C",
    "D": "endpoint0_factorial_D",
}
P1_ROLES = {
    "A": "freeze_optimizer_A",
    "B": "freeze_optimizer_B",
    "C": "freeze_optimizer_C",
    "D": "freeze_optimizer_D",
}
P2_ROLES = {
    "S1": "two_stage_S1",
    "S2": "two_stage_S2",
    "S3": "two_stage_S3",
    "S4": "two_stage_S4",
}
FACTORIAL_PROBE_STEPS = (0, 15, 44, 73, 100, 500, 1000, 1800, 5400)
P2_POST_ORIGIN_PROBE_STEPS = (100, 150, 250, 500, 1800, 5400)


def _p2_probe_steps(origin_step: int) -> list[int]:
    return sorted({int(origin_step), *P2_POST_ORIGIN_PROBE_STEPS})
P2_DIRECTION_CONFIG = {
    "S1": ("none", 0.0),
    "S2": ("class_centroid", 1.0),
    "S3": ("moving", 1.0),
    "S4": ("fixed_reference", 1.0),
}


def _role(
    records: list[tuple[Path, dict[str, Any]]], role: str
) -> tuple[Path, dict[str, Any]] | None:
    try:
        return select_unique_role(records, role)
    except ArtifactIntegrityError as error:
        if str(error).startswith("missing raw run"):
            return None
        raise


def _number(point: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = point
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(value)
        else None
    )


def _representation_value(
    result: dict[str, Any], step: int, representation: str
) -> float | None:
    point = point_map(result).get(step)
    transport = explicit_transport_enabled(result)
    if transport is None:
        return None
    if representation == "q" or transport is False:
        return _number(point, "val", "mAP")
    return _number(point, "base_val", "mAP")


def _stability(result: dict[str, Any], representation: str = "q") -> dict[str, Any]:
    values = [
        (step, value)
        for step in sorted(point_map(result))
        if (value := _representation_value(result, step, representation)) is not None
    ]
    if not values:
        return missing_result("no raw pseudo-unseen mAP checkpoints")
    peak_step, peak = max(values, key=lambda item: item[1])
    latest_step, latest = values[-1]
    late = _representation_value(result, 5400, representation)
    late_step = 5400 if late is not None else latest_step
    if late is None:
        late = latest
    return {
        "status": "completed" if late_step == 5400 else "partial",
        "peak_mAP": peak,
        "peak_step": peak_step,
        "late_step": late_step,
        "late_mAP": late,
        "retention_ratio": None if late is None or peak == 0 else late / peak,
        "absolute_decay": None if late is None else peak - late,
    }


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _same_checkpoint(left: object, right: object) -> bool:
    def resolve(value: object) -> Path:
        path = Path(str(value))
        return (path if path.is_absolute() else ROOT / path).resolve()

    return resolve(left) == resolve(right)


def _raw_run(record: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
    path, result = record
    provenance = result.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("status") != "valid":
        validation = "legacy_provenance_incomplete"
    elif (
        isinstance(result.get("data_manifest_identity"), dict)
        and result["data_manifest_identity"].get("sha256")
    ):
        validation = "valid"
    else:
        validation = "legacy_split_provenance"
    return {
        "run_id": result.get("wandb", {}).get("run_id") or _relative(path),
        "artifact_path": _relative(path / "run_result.json"),
        "checkpoint_id": result.get("checkpoint"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "resolved_config": result.get("resolved_config") or config_of(result),
        "seed": config_of(result).get("seed"),
        "data_split_identity": result.get("data_split_identity")
        or result.get("pseudo_split"),
        "data_manifest_identity": result.get("data_manifest_identity"),
        "provenance": provenance,
        "validation_status": validation,
        "raw_metrics": result.get("probe_history", []),
        "gradient_diagnostics": result.get("gradient_conflicts", []),
        "official_test_diagnostic_only": True,
        "resume": result.get("resume"),
    }


def _split_provenance(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    identities = [result.get("data_manifest_identity") for _, result in records]
    if not identities or not any(identity is not None for identity in identities):
        return {
            "status": "legacy",
            "reason": "historical runs record class/count split identity but not manifest-entry identity",
        }
    if any(identity is None for identity in identities):
        return {
            "status": "mixed",
            "reason": "some runs record manifest-entry identity and some are legacy",
        }
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ArtifactIntegrityError("corrected runs have different manifest-entry split identities")
    return {"status": "completed", "identity": first}


def _causal(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    base = _role(records, "endpoint0_factorial_B")
    transport = _role(records, "endpoint0_factorial_D")
    if base is None or transport is None:
        return missing_result(
            "matched structured roles endpoint0_factorial_B/D have not both run"
        )
    assert_matched_runs(base[1], transport[1])
    rows = []
    for step in sorted(set(point_map(base[1])) & set(point_map(transport[1]))):
        z0_b = _representation_value(base[1], step, "z0")
        z0_t = _representation_value(transport[1], step, "z0")
        q_t = _representation_value(transport[1], step, "q")
        if None in (z0_b, z0_t, q_t):
            continue
        rows.append(
            {
                "step": step,
                "mAP_z0_B": z0_b,
                "mAP_z0_T": z0_t,
                "mAP_q_T": q_t,
                "encoder_training_effect": z0_t - z0_b,
                "inference_head_effect": q_t - z0_t,
                "total_transport_effect": q_t - z0_b,
            }
        )
    if not rows:
        return missing_result(
            "matched runs contain no common raw evaluation checkpoint"
        )
    return {
        "status": "completed",
        "runs": {"z0_B": _raw_run(base), "z0_T_q_T": _raw_run(transport)},
        "matched_validation": "passed",
        "pointwise": rows,
        "step_5400": next((row for row in rows if row["step"] == 5400), None),
        "definitions": {
            "encoder_training_effect": "mAP(z0_T)-mAP(z0_B)",
            "inference_head_effect": "mAP(q_T)-mAP(z0_T)",
            "total_transport_effect": "mAP(q_T)-mAP(z0_B)",
        },
    }


def _validate_factorial_cell(name: str, result: dict[str, Any]) -> None:
    expected = {
        "A": (False, False),
        "B": (False, True),
        "C": (True, False),
        "D": (True, True),
    }[name]
    config = config_of(result)
    observed = (config.get("transport_enabled"), config.get("use_text_cls"))
    if observed != expected or float(config.get("lambda_endpoint", 1.0)) != 0.0:
        raise ArtifactIntegrityError(
            f"factorial role {name} requires transport/text={expected}, endpoint=0; observed {observed}"
        )


def _factorial(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    selected = {name: _role(records, role) for name, role in FACTORIAL_ROLES.items()}
    missing = [name for name, record in selected.items() if record is None]
    if missing:
        return missing_result(
            f"structured endpoint=0 factorial roles not run: {', '.join(missing)}"
        )
    complete = {name: record for name, record in selected.items() if record is not None}
    for name, record in complete.items():
        _validate_factorial_cell(name, record[1])
    assert_same_run_conditions(complete["A"][1], complete["B"][1], label="factorial A/B")
    assert_same_run_conditions(complete["C"][1], complete["D"][1], label="factorial C/D")
    assert_matched_runs(complete["B"][1], complete["D"][1])
    common_steps = sorted(
        set.intersection(*(set(point_map(record[1])) for record in complete.values()))
    )
    if common_steps != list(FACTORIAL_PROBE_STEPS):
        raise ArtifactIntegrityError(
            f"endpoint=0 factorial probe schedule is {common_steps}, expected {list(FACTORIAL_PROBE_STEPS)}"
        )
    pointwise: dict[str, list[dict[str, Any]]] = {}
    for representation in ("z0", "q"):
        rows = []
        for step in common_steps:
            cells = {
                name: _representation_value(record[1], step, representation)
                for name, record in complete.items()
            }
            if any(value is None for value in cells.values()):
                continue
            numeric = {
                name: float(value) for name, value in cells.items() if value is not None
            }
            rows.append(
                {"step": step, "cells": numeric, "effects": factorial_effects(numeric)}
            )
        if [row["step"] for row in rows] != list(FACTORIAL_PROBE_STEPS):
            raise ArtifactIntegrityError(
                f"endpoint=0 factorial {representation} values are incomplete"
            )
        pointwise[representation] = rows
    peaks = {
        representation: {
            name: _stability(record[1], representation)
            for name, record in complete.items()
        }
        for representation in ("z0", "q")
    }
    peak_effects = {}
    for representation, rows in peaks.items():
        values = {name: row.get("peak_mAP") for name, row in rows.items()}
        peak_effects[representation] = (
            factorial_effects(values)
            if all(isinstance(value, (int, float)) for value in values.values())
            else missing_result("a cell has no peak")
        )
    return {
        "status": "completed",
        "runs": {name: _raw_run(record) for name, record in complete.items()},
        "pointwise": pointwise,
        "step_5400": {
            representation: next((row for row in rows if row["step"] == 5400), None)
            for representation, rows in pointwise.items()
        },
        "cell_wise_peak": {
            "label": "best-achievable/early-stopped comparison; not a pointwise causal effect",
            "cells": peaks,
            "effects": peak_effects,
        },
    }


def _freeze_factorial(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    source = _role(records, "freeze_optimizer_source")
    branches = {name: _role(records, role) for name, role in P1_ROLES.items()}
    missing = [name for name, record in branches.items() if record is None]
    if source is None or missing:
        return missing_result(
            "P1 source/branches not complete: "
            + ("source " if source is None else "")
            + ",".join(missing)
        )
    complete = {name: record for name, record in branches.items() if record is not None}
    source_checkpoint = source[1].get("checkpoint")
    rows = {}
    artifact_validation = {}
    for name, record in complete.items():
        validate_freeze_optimizer_role(config_of(record[1]))
        artifact_validation[name] = validate_freeze_optimizer_artifact(record[1], source[1])
        resume = record[1].get("resume", {})
        if not _same_checkpoint(resume.get("checkpoint"), source_checkpoint):
            raise ArtifactIntegrityError(
                "P1 branches do not resume the identical source checkpoint"
            )
        checkpoints = {}
        for step in (500, 1800, 5400):
            point = point_map(record[1]).get(step)
            checkpoints[str(step)] = {
                "mAP_z0": _number(point, "base_val", "mAP"),
                "mAP_q": _number(point, "val", "mAP"),
                "semantic_margin": _number(
                    point, "val_geometry", "semantic", "semantic_margin"
                ),
                "base_reference_cosine": _number(
                    point, "val_geometry", "reference", "base_reference_cosine"
                ),
                "query_reference_cosine": _number(
                    point, "val_geometry", "reference", "query_reference_cosine"
                ),
                "hidden_CKA": _number(
                    point, "val_geometry", "hidden_space_compatibility", "linear_cka"
                ),
            }
        rows[name] = {
            "semantics": validate_freeze_optimizer_role(config_of(record[1])),
            "checkpoints": checkpoints,
            "stability_z0": _stability(record[1], "z0"),
            "stability_q": _stability(record[1], "q"),
            "run": _raw_run(record),
        }
    all_validated = all(
        value.get("status") == "validated" for value in artifact_validation.values()
    )
    return {
        "status": "completed" if all_validated else "completed_legacy",
        "reason": None
        if all_validated
        else "historical P1 artifacts lack recorded checkpoint hashes or complete resume semantics",
        "source": _raw_run(source),
        "branches": rows,
        "artifact_validation": artifact_validation,
        "question": "Encoder updates or optimizer state causes late degradation",
    }


def _p2_pointwise(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step in sorted(point_map(result)):
        point = point_map(result)[step]
        target_angles = point.get("val_geometry", {}).get("transport", {}).get("target_angles", {})
        row = {
            "step": step,
            "mAP": _number(point, "val", "mAP"),
            "P@200": _number(point, "val", "precision_at_k", "200"),
            "mAP@200": _number(point, "val", "mAP_at_k", "200"),
            "mean_rho_degrees": _number(point, "val_geometry", "transport", "mean_rho_degrees"),
            "moving_target_alignment": _number(target_angles, "moving_target_alignment"),
            "fixed_target_alignment": _number(target_angles, "fixed_target_alignment"),
            "moving_mean_degrees": _number(target_angles, "moving", "mean_degrees"),
            "fixed_mean_degrees": _number(target_angles, "fixed", "mean_degrees"),
        }
        rows.append(row)
    return rows


def _validate_two_stage_run(
    name: str,
    result: dict[str, Any],
    *,
    stage1: dict[str, Any],
    selected: dict[str, Any],
    selection_manifest: str,
) -> None:
    config = config_of(result)
    expected_direction, expected_lambda = P2_DIRECTION_CONFIG[name]
    required = {
        "transport_enabled": True,
        "K": 1,
        "rho_strategy": "fixed",
        "fixed_rho_degrees": 15.0,
        "lambda_dir": expected_lambda,
        "lambda_dist": 0.0,
        "lambda_endpoint": 0.0,
        "direction_target": expected_direction,
        "text_loss_location": "q",
        "freeze_encoder_on_resume": True,
        "freeze_encoder_at_step": None,
        "reset_optimizer_on_resume": True,
        "optimizer_reset_scope": "all",
        "reset_transport_head_on_resume": True,
        "train_class_scope": "pseudo_train",
        "experiment_campaign": CAMPAIGN,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ArtifactIntegrityError(
                f"P2 {name} has invalid {key}: {config.get(key)!r}, expected {value!r}"
            )
    if result.get("provenance", {}).get("status") != "valid":
        raise ArtifactIntegrityError(f"P2 {name} provenance is incomplete")
    if result.get("data_split_identity") != stage1.get("data_split_identity"):
        raise ArtifactIntegrityError(f"P2 {name} split differs from Stage 1")
    if (
        stage1.get("data_manifest_identity") is not None
        and result.get("data_manifest_identity") != stage1.get("data_manifest_identity")
    ):
        raise ArtifactIntegrityError(f"P2 {name} manifest entries differ from Stage 1")
    resume = result.get("resume")
    continuation = result.get("continuation")
    if not isinstance(resume, dict) or not isinstance(continuation, dict):
        raise ArtifactIntegrityError(f"P2 {name} continuation metadata is missing")
    origin_step = int(selected["step"])
    if not (
        resume.get("starting_step") == origin_step
        and resume.get("freeze_applied_before_first_update") is True
        and resume.get("encoder_frozen_immediately") is True
        and resume.get("optimizer_state_reset") is True
        and resume.get("optimizer_state_restored") is False
        and resume.get("transport_head_reinitialized") is True
        and continuation.get("encoder_frozen_at_start") is True
    ):
        raise ArtifactIntegrityError(f"P2 {name} does not prove immediate freeze/reset semantics")
    if not _same_checkpoint(resume.get("checkpoint"), selected.get("checkpoint")):
        raise ArtifactIntegrityError(f"P2 {name} does not resume the selected Stage-1 checkpoint")
    embedded = result.get("stage1_selection")
    if not isinstance(embedded, dict):
        raise ArtifactIntegrityError(f"P2 {name} has no embedded Stage-1 selection")
    if (
        embedded.get("selected") != selected
        or embedded.get("manifest_path") != selection_manifest
        or embedded.get("official_unseen_used") is not False
        or embedded.get("selection_metric") != "pseudo_unseen_validation_mAP"
        or embedded.get("data_split_identity") != stage1.get("data_split_identity")
        or (
            stage1.get("data_manifest_identity") is not None
            and embedded.get("data_manifest_identity") != stage1.get("data_manifest_identity")
        )
    ):
        raise ArtifactIntegrityError(f"P2 {name} embeds an unsafe or different Stage-1 selection")
    steps = sorted(point_map(result))
    expected_steps = _p2_probe_steps(origin_step)
    if steps != expected_steps:
        raise ArtifactIntegrityError(
            f"P2 {name} probe schedule is {steps}, expected {expected_steps}"
        )


def _two_stage(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    stage1_record = _role(records, "two_stage_stage1")
    stage_records = {name: _role(records, role) for name, role in P2_ROLES.items()}
    if stage1_record is None or any(record is None for record in stage_records.values()):
        return missing_result(
            "P2 stage 1 and all S1-S4 structured runs are not complete; S0 is stage-1 z0"
        )
    stage1 = stage1_record[1]
    if stage1.get("provenance", {}).get("status") != "valid":
        raise ArtifactIntegrityError("P2 Stage 1 provenance is incomplete")
    stage1_config = config_of(stage1)
    if not (
        stage1_config.get("transport_enabled") is False
        and stage1_config.get("train_class_scope") == "pseudo_train"
        and stage1_config.get("experiment_campaign") == CAMPAIGN
        and stage1_config.get("steps") == 73
        and sorted(point_map(stage1).keys()) == [0, 44, 73]
    ):
        raise ArtifactIntegrityError("P2 Stage 1 is not the corrected 0/44/73 semantic origin")
    selection = None
    selected = None
    selection_manifest = None
    variants = {}
    for name, record in stage_records.items():
        assert record is not None
        embedded = record[1].get("stage1_selection")
        if not isinstance(embedded, dict) or not isinstance(embedded.get("selected"), dict):
            raise ArtifactIntegrityError(f"P2 {name} has no valid embedded Stage-1 selection")
        if selection is None:
            selection, selected, selection_manifest = embedded, embedded["selected"], embedded.get("manifest_path")
        _validate_two_stage_run(
            name,
            record[1],
            stage1=stage1,
            selected=selected,
            selection_manifest=selection_manifest,
        )
        if (
            embedded.get("manifest_sha256") != selection.get("manifest_sha256")
            or embedded.get("data_manifest_identity") != selection.get("data_manifest_identity")
        ):
            raise ArtifactIntegrityError("P2 variants use different selection manifests")
        if selection_manifest is None:
            raise ArtifactIntegrityError("P2 selection manifest path is missing")
        manifest_path = Path(selection_manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if not manifest_path.is_file() or embedded.get("manifest_sha256") != _sha256(manifest_path):
            raise ArtifactIntegrityError("P2 selection manifest hash/path is invalid")
        variants[name] = {
            "run": _raw_run(record),
            "stability": _stability(record[1]),
            "pointwise": _p2_pointwise(record[1]),
        }
    manifest_records = [stage1, *(record[1] for record in stage_records.values())]
    manifest_complete = all(record.get("data_manifest_identity") is not None for record in manifest_records)
    return {
        "status": "completed" if manifest_complete else "completed_legacy",
        "reason": None
        if manifest_complete
        else "historical P2 artifacts lack manifest-entry identity",
        "S0": _raw_run(stage1_record),
        "selected_origin": selection,
        "variants": variants,
    }


def _gradient_report(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    values = []
    for path, result in records:
        for item in result.get("gradient_conflicts", []):
            if not isinstance(item, dict) or "spaces" not in item:
                continue
            spaces = item["spaces"]
            if set(spaces) != {"query", "base", "parameter"}:
                raise ArtifactIntegrityError(
                    f"gradient spaces mixed or missing in {path}"
                )
            values.append({"run": _relative(path), **item})
    return (
        {"status": "completed", "raw_cosine_matrices": values}
        if values
        else missing_result("no corrected gradient artifacts")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_refit(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return missing_result("P3 projection-refit artifact has not run")
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"cannot read P3 artifact: {path}") from error
    if not isinstance(artifact, dict):
        raise ArtifactIntegrityError("P3 artifact must be a JSON object")
    if not (
        artifact.get("schema_version") == 2
        and artifact.get("probe") == "projection_refit_control"
        and artifact.get("official_unseen_used") is False
        and artifact.get("selection_metric") is None
        and artifact.get("fit_split") is None
        and artifact.get("seed") == 3407
        and isinstance(artifact.get("provenance"), dict)
        and artifact["provenance"].get("status") == "valid"
    ):
        raise ArtifactIntegrityError("P3 artifact has an unsafe or unsupported protocol")
    source_artifacts = artifact.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise ArtifactIntegrityError("P3 artifact does not preserve source artifacts")
    manifest_identity = artifact.get("data_manifest_identity")
    if manifest_identity is not None and (
        not isinstance(manifest_identity, dict) or not manifest_identity.get("sha256")
    ):
        raise ArtifactIntegrityError("P3 manifest-entry identity is invalid")
    for source in source_artifacts:
        if not isinstance(source, dict):
            raise ArtifactIntegrityError("P3 source artifact entry is not an object")
        source_path = Path(str(source.get("path")))
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not source_path.is_file() or source.get("sha256") != _sha256(source_path):
            raise ArtifactIntegrityError(f"P3 source artifact hash/path is invalid: {source_path}")
    rows = artifact.get("values")
    if not isinstance(rows, list) or not rows:
        raise ArtifactIntegrityError("P3 artifact has no checkpoint values")
    required_methods = {
        "frozen_original_W_CLIP",
        "orthogonal_adapter",
        "ridge_projection",
    }
    seen = set()
    row_manifest_identities = []
    for row in rows:
        if not isinstance(row, dict):
            raise ArtifactIntegrityError("P3 checkpoint row is not an object")
        role = row.get("experiment_role")
        step = row.get("checkpoint_step")
        if role not in P1_ROLES.values() and role != "freeze_optimizer_source":
            raise ArtifactIntegrityError(f"P3 row has an invalid role: {role!r}")
        identity = (role, step)
        if identity in seen:
            raise ArtifactIntegrityError(f"duplicate P3 checkpoint row: {identity}")
        seen.add(identity)
        if row.get("experiment_campaign") != CAMPAIGN:
            raise ArtifactIntegrityError("P3 row campaign differs from corrected P1 campaign")
        row_manifest_identity = row.get("data_manifest_identity")
        if row_manifest_identity is not None:
            if row_manifest_identity != manifest_identity and manifest_identity is not None:
                raise ArtifactIntegrityError("P3 row manifest-entry identity differs from artifact")
            row_manifest_identities.append(row_manifest_identity)
        if row.get("fit_split") != "pseudo_train only" or row.get("evaluation_split") != "pseudo-unseen only":
            raise ArtifactIntegrityError("P3 row does not prove train-only fitting and pseudo-unseen evaluation")
        methods = row.get("methods")
        if not isinstance(methods, dict) or not required_methods <= methods.keys():
            raise ArtifactIntegrityError(f"P3 row is missing required methods: {identity}")
        if role == "freeze_optimizer_source" and step != 73:
            raise ArtifactIntegrityError("P3 source must be the step-73 checkpoint")
        if role != "freeze_optimizer_source" and step not in {500, 1800, 5400}:
            raise ArtifactIntegrityError(f"P3 branch checkpoint step is invalid: {identity}")
        checkpoint = Path(str(row.get("checkpoint")))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file() or row.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ArtifactIntegrityError(f"P3 checkpoint hash/path is invalid: {checkpoint}")
        if any(
            not isinstance(methods[method], dict)
            or not isinstance(methods[method].get("mAP"), (int, float))
            or not math.isfinite(float(methods[method]["mAP"]))
            for method in required_methods
        ):
            raise ArtifactIntegrityError(f"P3 checkpoint metrics are invalid: {identity}")
    if (manifest_identity is None) != (not row_manifest_identities):
        raise ArtifactIntegrityError("P3 manifest-entry identity is only partially recorded")
    if manifest_identity is not None and len(row_manifest_identities) != len(rows):
        raise ArtifactIntegrityError("P3 rows do not all record manifest-entry identity")
    manifest_complete = manifest_identity is not None and len(row_manifest_identities) == len(rows)
    return {
        "status": "completed" if manifest_complete else "completed_legacy",
        "reason": None
        if manifest_complete
        else "historical P3 artifact lacks manifest-entry identity",
        "artifact_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "artifact_sha256": _sha256(path),
        "official_unseen_used": False,
        "data_manifest_identity": manifest_identity,
        "manifest_identity_status": (
            "completed"
            if manifest_identity is not None and len(row_manifest_identities) == len(rows)
            else "legacy"
        ),
        "selection_metric": None,
        "values": rows,
        "provenance": artifact.get("provenance"),
    }


def _plot(
    path: Path, title: str, series: dict[str, list[tuple[int, float]]], source: Any
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, values in series.items():
        if values:
            axis.plot(
                [x for x, _ in values], [y for _, y in values], marker="o", label=label
            )
    if any(series.values()):
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "NOT RUN / no valid raw artifact",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel("pseudo-unseen mAP")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    write_new(path, buffer.getvalue())
    write_new(
        path.with_suffix(".json"), json.dumps(source, indent=2, sort_keys=True) + "\n"
    )


def _is_completed(value: dict[str, Any]) -> bool:
    return value.get("status") in {"completed", "completed_legacy"}


def _make_plots(output: Path, report: dict[str, Any]) -> list[dict[str, str]]:
    causal = report["causal_decomposition"]
    causal_rows = causal.get("pointwise", []) if _is_completed(causal) else []
    factorial = report["endpoint0_factorial"]
    factorial_rows = (
        factorial.get("pointwise", {}).get("q", [])
        if _is_completed(factorial)
        else []
    )
    freeze = report["freeze_optimizer_factorial"]
    p1_series = {}
    if _is_completed(freeze):
        p1_series = {
            name: [
                (int(step), row["mAP_q"])
                for step, row in value["checkpoints"].items()
                if row["mAP_q"] is not None
            ]
            for name, value in freeze["branches"].items()
        }
    two_stage = report["two_stage_transport"]
    p2_series = {}
    if _is_completed(two_stage):
        for name, value in two_stage["variants"].items():
            p2_series[name] = [
                (int(point["step"]), float(point["mAP"]))
                for point in value["pointwise"]
                if isinstance(point.get("mAP"), (int, float))
            ]
    projection = report["projection_refit_control"]
    p3_series = {}
    if _is_completed(projection):
        for row in projection["values"]:
            for method in ("frozen_original_W_CLIP", "orthogonal_adapter", "ridge_projection"):
                value = row["methods"][method].get("mAP")
                if isinstance(value, (int, float)):
                    label = f"{row['experiment_role']}/{method}"
                    p3_series.setdefault(label, []).append((int(row["checkpoint_step"]), float(value)))
    suffix = f"{report['report_date']}_report"
    specifications = [
        (
            f"corrected_causal_decomposition_{suffix}.png",
            "Corrected causal decomposition",
            {
                key: [(row["step"], row[key]) for row in causal_rows]
                for key in ("mAP_z0_B", "mAP_z0_T", "mAP_q_T")
            },
            causal,
        ),
        (
            f"corrected_endpoint0_factorial_{suffix}.png",
            "Corrected endpoint=0 factorial",
            {
                name: [(row["step"], row["cells"][name]) for row in factorial_rows]
                for name in FACTORIAL_ROLES
            },
            factorial,
        ),
        (
            f"freeze_optimizer_factorial_{suffix}.png",
            "Freeze × optimizer factorial",
            p1_series,
            freeze,
        ),
        (
            f"two_stage_transport_{suffix}.png",
            "Two-stage frozen-origin transport",
            p2_series,
            two_stage,
        ),
        (
            f"projection_refit_control_{suffix}.png",
            "Projection refit control",
            p3_series,
            projection,
        ),
        (
            f"frozen_origin_direction_ablation_{suffix}.png",
            "Frozen-origin direction ablation",
            {},
            report["frozen_origin_direction_ablation"],
        ),
        (
            f"stability_retention_corrected_{suffix}.png",
            "Corrected stability",
            p1_series or p2_series,
            report["stability"],
        ),
    ]
    artifacts = []
    for filename, title, series, source in specifications:
        path = output / filename
        _plot(path, title, series, source)
        artifacts.append(
            {
                "plot": str(path.relative_to(ROOT)),
                "sidecar": str(path.with_suffix(".json").relative_to(ROOT)),
            }
        )
    return artifacts


def _markdown(report: dict[str, Any]) -> str:
    def status(name: str) -> str:
        value = report[name]
        return f"{value.get('status', 'unknown').upper()}: {value.get('reason', 'raw artifact validated')}"

    lines = [
        f"# Corrected SPICA transport summary ({report['report_date']})",
        "",
        "All selection uses pseudo-unseen validation and structured `experiment_role`; official unseen is excluded.",
        "",
        "## Integrity status",
        f"- Causal decomposition — {status('causal_decomposition')}",
        f"- Endpoint=0 factorial — {status('endpoint0_factorial')}",
        f"- P1 freeze × optimizer — {status('freeze_optimizer_factorial')}",
        f"- P2 two-stage transport — {status('two_stage_transport')}",
        f"- P3 projection refit — {status('projection_refit_control')}",
        f"- P4 direction ablation — {status('frozen_origin_direction_ablation')}",
        f"- P5 statistical confirmation — {status('statistical_confirmation')}",
        f"- P6 deterministic K — {status('deterministic_K')}",
        f"- Split manifest-entry provenance — {report['split_provenance'].get('status', 'unknown').upper()}: {report['split_provenance'].get('reason', 'validated')}",
        "- Mo-vMF — DEFER",
        "",
        "## Scientific result",
    ]
    if report["freeze_optimizer_factorial"].get("status") in {"completed", "completed_legacy"}:
        freeze = report["freeze_optimizer_factorial"]
        lines.append("P1 raw branch trajectories are available in the JSON and plot sidecar; interpretation uses matched late metrics.")
        lines += [
            "",
            "### P1 step-5400 matched values",
            "| branch | z0 mAP | q mAP |",
            "|---|---:|---:|",
        ]
        for name, value in freeze["branches"].items():
            late = value["checkpoints"]["5400"]
            lines.append(f"| {name} | {late['mAP_z0']:.6f} | {late['mAP_q']:.6f} |")
        late = {
            name: value["checkpoints"]["5400"]["mAP_z0"]
            for name, value in freeze["branches"].items()
        }
        lines.append(
            f"At step 5400, trainable A/C trail matched frozen B/D by {late['B'] - late['A']:.4f}/{late['D'] - late['C']:.4f} z0 mAP; reset changes A→C by {late['C'] - late['A']:+.4f} and B→D by {late['D'] - late['B']:+.4f}."
        )
        if freeze.get("status") == "completed_legacy":
            lines.append("P1 measurements are retained as legacy evidence: the historical artifacts do not record checkpoint hashes and complete continuation semantics required by the current validator.")
    else:
        lines.append("The requested causal mechanisms are not yet known because no complete corrected raw factorial is available.")
    two_stage = report["two_stage_transport"]
    if _is_completed(two_stage):
        selected = two_stage["selected_origin"]["selected"]
        lines += [
            "",
            f"P2 selected Stage-1 origin: step {selected['step']} (pseudo-unseen mAP {selected['mAP']:.6f}); all S1-S4 resume from that hashed checkpoint, freeze before the first update, reset optimizer state, and reinitialize the head.",
            "",
            "### P2 step-5400 direction ablation",
            "| variant | direction target | mAP | P@200 | mAP@200 |",
            "|---|---|---:|---:|---:|",
        ]
        for name, value in two_stage["variants"].items():
            point = next(point for point in value["pointwise"] if point["step"] == 5400)
            target = P2_DIRECTION_CONFIG[name][0]
            lines.append(f"| {name} | {target} | {point['mAP']:.6f} | {point['P@200']:.6f} | {point['mAP@200']:.6f} |")
        late_points = {
            name: next(point for point in value["pointwise"] if point["step"] == 5400)
            for name, value in two_stage["variants"].items()
        }
        best_name = max(late_points, key=lambda name: late_points[name]["mAP"])
        lines.append(
            f"The best step-5400 P2 variant is {best_name}; direction-target differences are descriptive and remain pseudo-unseen-only."
        )
    projection = report["projection_refit_control"]
    if _is_completed(projection):
        lines += [
            "",
            "### P3 projection-refit pseudo-unseen mAP",
            "| role | step | frozen W | orthogonal | ridge |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in projection["values"]:
            if row["checkpoint_step"] not in {73, 5400}:
                continue
            methods = row["methods"]
            lines.append(
                f"| {row['experiment_role']} | {row['checkpoint_step']} | {methods['frozen_original_W_CLIP']['mAP']:.6f} | {methods['orthogonal_adapter']['mAP']:.6f} | {methods['ridge_projection']['mAP']:.6f} |"
            )
        lines.append("P3 refit metrics are descriptive controls, not a selected model; all fits use pseudo-train rows and evaluation uses pseudo-unseen rows only.")
        lines.append("Matched-control recovery fractions, absolute mAP, and alignment/rank diagnostics are recorded per method; neither refit is a selected model or evidence of semantic recovery by itself.")
    lines += [
        "",
        "Cell-wise peaks, when present, are labeled best-achievable/early-stopped comparisons and are not pointwise causal effects.",
        "Official-unseen/test metrics are retained only as diagnostics and never select a checkpoint or variant.",
        "Missing values are `null` with an explicit status/reason; no historical headline was carried forward.",
        "Mo-vMF and K>1 remain deferred until this deterministic K=1 baseline is stable.",
        "",
        "## Raw artifacts",
    ]
    for role, raw in report["run_index"].items():
        lines.append(
            f"- `{role}`: `{raw['artifact_path']}` ({raw['validation_status']})"
        )
    return "\n".join(lines) + "\n"


def build_report(
    run_root: Path,
    report_date: str,
    projection_refit_path: Path | None = None,
) -> dict[str, Any]:
    records = collect_runs(run_root)
    role_records = [
        record
        for record in records
        if config_of(record[1]).get("experiment_role")
        and config_of(record[1]).get("experiment_campaign") == CAMPAIGN
    ]
    roles = [str(config_of(result)["experiment_role"]) for _, result in role_records]
    duplicates = sorted({role for role in roles if roles.count(role) > 1})
    if duplicates:
        raise ArtifactIntegrityError(
            f"ambiguous duplicate experiment roles: {duplicates}"
        )
    run_index = {
        str(config_of(result)["experiment_role"]): _raw_run((path, result))
        for path, result in role_records
    }
    freeze = _freeze_factorial(role_records)
    two_stage = _two_stage(role_records)
    if projection_refit_path is None:
        projection_refit_path = ROOT / "outputs" / f"projection_refit_control_{report_date}_corrected_final.json"
    projection_refit = _projection_refit(projection_refit_path)
    stability_rows = {
        role: _stability(result)
        for path, result in role_records
        if (role := config_of(result).get("experiment_role"))
    }
    return {
        "schema_version": 1,
        "report_date": report_date,
        "repository": repository_provenance(),
        "selection_protocol": {
            "selector": f"exact structured experiment_role within campaign {CAMPAIGN}",
            "official_unseen_selection": False,
            "duplicate_roles": "fatal",
            "missing_artifacts": "null/status=not_run",
        },
        "run_index": run_index,
        "split_provenance": _split_provenance(role_records),
        "causal_decomposition": _causal(role_records),
        "endpoint0_factorial": _factorial(role_records),
        "freeze_optimizer_factorial": freeze,
        "two_stage_transport": two_stage,
        "projection_refit_control": projection_refit,
        "frozen_origin_direction_ablation": missing_result(
            "P4 dedicated frozen-origin direction runs have not run; P2 S1-S4 are reported separately"
        ),
        "statistical_confirmation": missing_result(
            "P5 three-seed confirmation has not run"
        ),
        "deterministic_K": missing_result("P6 is deferred until P1-P5 finalists"),
        "gradient_conflict": _gradient_report(role_records),
        "stability": {"status": "completed", "runs": stability_rows}
        if stability_rows
        else missing_result("no corrected runs"),
    }


def _commands() -> dict[str, Any]:
    train = "PYTHONPATH=src python -m spica.train_transport"
    return {
        "P1_source": f"{train} +experiments=transport_corrected_p1_source73",
        "P1_forks": [
            f"{train} +experiments=transport_corrected_p1_{suffix} resume_checkpoint_path=$SOURCE73"
            for suffix in (
                "A_trainable_restored",
                "B_frozen_restored",
                "C_trainable_reset",
                "D_frozen_reset",
            )
        ],
        "P2_stage1": f"{train} +experiments=transport_corrected_p2_stage1",
        "P2_stage2": [
            f"{train} +experiments=transport_corrected_p2_{suffix} resume_checkpoint_path=$STAGE1 stage1_selection_manifest_path=$STAGE1_MANIFEST"
            for suffix in (
                "S1_no_direction",
                "S2_class_centroid",
                "S3_moving",
                "S4_fixed_reference",
            )
        ],
        "summary": "PYTHONPATH=src python scripts/summarize_transport_corrected.py --date YYYY-MM-DD",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "outputs" / "experiments"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--projection-refit", type=Path, default=None)
    args = parser.parse_args()
    projection_refit = args.projection_refit
    if projection_refit is not None and not projection_refit.is_absolute():
        projection_refit = ROOT / projection_refit
    report = build_report(args.run_root, args.date, projection_refit)
    output = args.output_dir.resolve()
    plots = _make_plots(output, report)
    report["plots"] = plots
    manifest = {
        "schema_version": 1,
        "report_date": args.date,
        "runs": report["run_index"],
        "commands": _commands(),
        "probe_status": {
            key: report[key]
            for key in (
                "freeze_optimizer_factorial",
                "two_stage_transport",
                "projection_refit_control",
                "frozen_origin_direction_ablation",
                "statistical_confirmation",
                "deterministic_K",
            )
        },
    }
    json_path = output / f"research_summary_transport_corrected_{args.date}.json"
    md_path = output / f"research_summary_transport_corrected_{args.date}.md"
    manifest_path = output / f"experiment_manifest_transport_corrected_{args.date}.json"
    write_new(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_new(md_path, _markdown(report))
    write_new(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(md_path)
    print(json_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
