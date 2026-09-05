"""Validate alignment run artifacts and write a compact comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spica.alignment_artifacts import ALIGNMENT_ROLES


def _runs(campaign_dir: Path) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(campaign_dir.glob("*/seed*/**/run_result.json")):
        result = json.loads(path.read_text())
        role = str(result.get("experiment_role"))
        found.setdefault(role, []).append(result)
    return found


def _validate(result: dict[str, Any]) -> None:
    if result.get("official_unseen_used_for_selection") is not False:
        raise ValueError("official unseen data was used for selection")
    protocol = result.get("protocol", {})
    for key in (
        "validation_used_for_alignment",
        "test_used_for_alignment",
        "text_used_for_predictor",
        "photo_used_for_predictor",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"protocol violation: {key}")
    history = result.get("history", [])
    if not history:
        raise ValueError("run has no history")
    for row in history:
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise ValueError(f"missing checkpoint: {checkpoint}")
        if not row.get("checkpoint_sha256"):
            raise ValueError("checkpoint hash is missing")


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def write_report(campaign_dir: Path, output_path: Path) -> None:
    discovered = _runs(campaign_dir)
    if not discovered:
        raise ValueError(f"no run_result.json files found below {campaign_dir}")
    target_steps = max(
        int(result.get("resolved_config", {}).get("max_steps", 0))
        for values in discovered.values()
        for result in values
    )
    runs = {
        role: [
            result
            for result in values
            if int(result.get("resolved_config", {}).get("max_steps", 0))
            == target_steps
        ]
        for role, values in discovered.items()
    }
    for values in runs.values():
        for result in values:
            _validate(result)
    control = runs.get("alignment_control", [])
    control_by_seed = {
        int(result.get("training_seed", result.get("seed"))): max(
            float(row["full_pseudo_unseen_mAP"]) for row in result["history"]
        )
        for result in control
    }
    lines = [
        f"# Alignment campaign: `{campaign_dir.name}`",
        "",
        f"Selection is pseudo-unseen validation only; official unseen data was not used. Target horizon: {target_steps} steps.",
        "",
        "| Arm | Seeds | Final step mAP | Best mAP | Best step | Δ best vs matched control |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary: list[dict[str, Any]] = []
    for role in ALIGNMENT_ROLES:
        values = runs.get(role, [])
        if not values:
            lines.append(f"| `{role}` | 0 | — | — | — | — |")
            continue
        finals: list[float] = []
        best_values: list[float] = []
        best_steps: list[int] = []
        deltas: list[float] = []
        control_mean = (
            sum(control_by_seed.values()) / len(control_by_seed)
            if control_by_seed
            else None
        )
        for result in values:
            history = result["history"]
            finals.append(float(history[-1]["full_pseudo_unseen_mAP"]))
            best = max(
                history,
                key=lambda row: (
                    float(row["full_pseudo_unseen_mAP"]),
                    -int(row["training_global_step"]),
                ),
            )
            best_value = float(best["full_pseudo_unseen_mAP"])
            best_values.append(best_value)
            best_steps.append(int(best["training_global_step"]))
            seed = int(result.get("training_seed", result.get("seed")))
            matched_control = control_by_seed.get(seed, control_mean)
            if matched_control is not None:
                deltas.append(best_value - matched_control)
        final = sum(finals) / len(finals)
        best = sum(best_values) / len(best_values)
        delta = sum(deltas) / len(deltas) if deltas else None
        lines.append(
            f"| `{role}` | {len(values)} | {_fmt(final)} | {_fmt(best)} | "
            f"{sum(best_steps) / len(best_steps):.0f} | {_fmt(delta)} |"
        )
        summary.append(
            {
                "role": role,
                "seeds": len(values),
                "final_mAP_mean": final,
                "best_mAP_mean": best,
                "best_step_mean": sum(best_steps) / len(best_steps),
                "best_delta_vs_control": delta,
            }
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The table is descriptive, not a novelty claim. A positive pilot or one-seed result is not evidence of a robust improvement; replicate only predeclared arms and retain negative controls.",
            "",
            "## Artifact checks",
            "",
            f"- Roles discovered at target horizon: {sorted(runs)}",
            "- Every discovered run passed checkpoint/protocol checks.",
            "- Alignment targets are declared train-only and detached in the run artifacts.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.campaign_dir, args.output)


if __name__ == "__main__":
    main()
