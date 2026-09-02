"""Integrity-first transport summary generated only from structured raw runs."""

from __future__ import annotations

import argparse
from datetime import date
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
        collect_runs,
        config_of,
        factorial_effects,
        missing_result,
        point_map,
        repository_provenance,
        select_unique_role,
        validate_freeze_optimizer_role,
        write_new,
    )
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from transport_artifact_utils import (
        ArtifactIntegrityError,
        ROOT,
        assert_matched_runs,
        collect_runs,
        config_of,
        factorial_effects,
        missing_result,
        point_map,
        repository_provenance,
        select_unique_role,
        validate_freeze_optimizer_role,
        write_new,
    )

STEPS = (0, 15, 44, 73, 100, 500, 1000, 1800, 5400)
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
    transport = config_of(result).get("transport_enabled") is True
    if representation == "q" or not transport:
        return _number(point, "val", "mAP")
    return _number(point, "base_val", "mAP")


def _stability(result: dict[str, Any], representation: str = "q") -> dict[str, Any]:
    values = [
        (step, value)
        for step in STEPS
        if (value := _representation_value(result, step, representation)) is not None
    ]
    if not values:
        return missing_result("no raw pseudo-unseen mAP checkpoints")
    peak_step, peak = max(values, key=lambda item: item[1])
    late = _representation_value(result, 5400, representation)
    return {
        "status": "completed" if late is not None else "partial",
        "peak_mAP": peak,
        "peak_step": peak_step,
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
    validation = (
        "valid"
        if isinstance(provenance, dict) and provenance.get("status") == "valid"
        else "legacy_provenance_incomplete"
    )
    return {
        "run_id": result.get("wandb", {}).get("run_id") or _relative(path),
        "artifact_path": _relative(path / "run_result.json"),
        "checkpoint_id": result.get("checkpoint"),
        "resolved_config": result.get("resolved_config") or config_of(result),
        "seed": config_of(result).get("seed"),
        "data_split_identity": result.get("data_split_identity")
        or result.get("pseudo_split"),
        "provenance": provenance,
        "validation_status": validation,
        "raw_metrics": result.get("probe_history", []),
        "gradient_diagnostics": result.get("gradient_conflicts", []),
        "resume": result.get("resume"),
    }


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
    assert_matched_runs(complete["B"][1], complete["D"][1])
    common_steps = sorted(
        set.intersection(*(set(point_map(record[1])) for record in complete.values()))
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
    for name, record in complete.items():
        validate_freeze_optimizer_role(config_of(record[1]))
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
    return {
        "status": "completed",
        "source": _raw_run(source),
        "branches": rows,
        "question": "Encoder updates or optimizer state causes late degradation",
    }


def _two_stage(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    stage1 = _role(records, "two_stage_stage1")
    stages = {name: _role(records, role) for name, role in P2_ROLES.items()}
    if stage1 is None or any(record is None for record in stages.values()):
        return missing_result(
            "P2 stage 1 and all S1-S4 structured runs are not complete; S0 is stage-1 z0"
        )
    return {
        "status": "completed",
        "S0": _raw_run(stage1),
        "variants": {
            name: {"run": _raw_run(record), "stability": _stability(record[1])}
            for name, record in stages.items()
            if record is not None
        },
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


def _make_plots(output: Path, report: dict[str, Any]) -> list[dict[str, str]]:
    causal = report["causal_decomposition"]
    causal_rows = (
        causal.get("pointwise", []) if causal.get("status") == "completed" else []
    )
    factorial = report["endpoint0_factorial"]
    factorial_rows = (
        factorial.get("pointwise", {}).get("q", [])
        if factorial.get("status") == "completed"
        else []
    )
    freeze = report["freeze_optimizer_factorial"]
    p1_series = {}
    if freeze.get("status") == "completed":
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
    if two_stage.get("status") == "completed":
        for name, value in two_stage["variants"].items():
            p2_series[name] = [
                (int(point["step"]), float(point["val"]["mAP"]))
                for point in value["run"]["raw_metrics"]
                if isinstance(point.get("val", {}).get("mAP"), (int, float))
            ]
    specifications = [
        (
            "corrected_causal_decomposition.png",
            "Corrected causal decomposition",
            {
                key: [(row["step"], row[key]) for row in causal_rows]
                for key in ("mAP_z0_B", "mAP_z0_T", "mAP_q_T")
            },
            causal,
        ),
        (
            "corrected_endpoint0_factorial.png",
            "Corrected endpoint=0 factorial",
            {
                name: [(row["step"], row["cells"][name]) for row in factorial_rows]
                for name in FACTORIAL_ROLES
            },
            factorial,
        ),
        (
            "freeze_optimizer_factorial.png",
            "Freeze × optimizer factorial",
            p1_series,
            freeze,
        ),
        (
            "two_stage_transport.png",
            "Two-stage frozen-origin transport",
            p2_series,
            two_stage,
        ),
        (
            "projection_refit_control.png",
            "Projection refit control",
            {},
            report["projection_refit_control"],
        ),
        (
            "frozen_origin_direction_ablation.png",
            "Frozen-origin direction ablation",
            p2_series,
            two_stage,
        ),
        (
            "stability_retention_corrected.png",
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
        "- Mo-vMF — DEFER",
        "",
        "## Scientific result",
    ]
    if report["freeze_optimizer_factorial"].get("status") == "completed":
        lines.append(
            "P1 raw branch trajectories are available in the JSON and plot sidecar; interpretation must use their matched late metrics."
        )
    else:
        lines.append(
            "The requested causal mechanisms are not yet known because no complete corrected raw factorial is available."
        )
    lines += [
        "",
        "Cell-wise peaks, when present, are labeled best-achievable/early-stopped comparisons and are not pointwise causal effects.",
        "Missing values are `null` with an explicit status/reason; no historical headline was carried forward.",
        "",
        "## Raw artifacts",
    ]
    for role, raw in report["run_index"].items():
        lines.append(
            f"- `{role}`: `{raw['artifact_path']}` ({raw['validation_status']})"
        )
    return "\n".join(lines) + "\n"


def build_report(run_root: Path, report_date: str) -> dict[str, Any]:
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
        "causal_decomposition": _causal(role_records),
        "endpoint0_factorial": _factorial(role_records),
        "freeze_optimizer_factorial": freeze,
        "two_stage_transport": two_stage,
        "projection_refit_control": missing_result(
            "P3 adapter/projection refit has not run"
        ),
        "frozen_origin_direction_ablation": two_stage
        if two_stage.get("status") == "completed"
        else missing_result("P4 frozen-origin direction runs have not completed"),
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
            f"{train} +experiments=transport_corrected_p2_{suffix} resume_checkpoint_path=$STAGE1"
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
    args = parser.parse_args()
    report = build_report(args.run_root, args.date)
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
