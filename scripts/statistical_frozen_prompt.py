"""Paired pseudo-unseen bootstrap for the validated frozen-prompt campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

try:
    from scripts.summarize_frozen_prompt import load_runs
except ModuleNotFoundError:
    from summarize_frozen_prompt import load_runs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/frozen_prompt_v2_statistical_2026-09-04.json"
COMPARISONS = {
    "FP1_vs_FP0": ("frozen_prompt_v2_FP1", 5400, "frozen_prompt_v2_FP0", 0),
    "FP1S_vs_FP1": ("frozen_prompt_v2_FP1S", 5400, "frozen_prompt_v2_FP1", 5400),
    "FP2_vs_FP1": ("frozen_prompt_v2_FP2", 5400, "frozen_prompt_v2_FP1", 5400),
    "FP3_vs_FP2": ("frozen_prompt_v2_FP3", 5400, "frozen_prompt_v2_FP2", 5400),
    "FP3_vs_FP5": ("frozen_prompt_v2_FP3", 5400, "frozen_prompt_v2_FP5", 5400),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(run: dict, step: int) -> dict:
    if run["experiment_role"] == "frozen_prompt_v2_FP5" and step == 5400:
        return next(
            item
            for item in run["frozen_hold_evaluation"]
            if item["comparison_horizon"] == 5400
        )
    if run["experiment_role"] == "frozen_prompt_v2_FP0" and step == 5400:
        step = 0
    return next(row for row in run["history"] if row["training_global_step"] == step)


def _bootstrap(left: np.ndarray, right: np.ndarray, draws: int, seed: int) -> dict:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired AP vectors must be one-dimensional and equal-sized")
    difference = left - right
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.integers(0, difference.size, size=difference.size)
        estimates[index] = difference[sample].mean()
    return {
        "num_queries": int(difference.size),
        "observed_delta_mAP": float(difference.mean()),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "ci_95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "p_two_sided": float(
            2.0 * min(np.mean(estimates <= 0), np.mean(estimates >= 0))
        ),
    }


def run(
    paths: list[Path], output: Path, *, draws: int = 2000, seed: int = 20260904
) -> dict:
    runs = load_runs(paths)
    source_by_role = {
        json.loads(path.read_text())["experiment_role"]: path for path in paths
    }
    results = {}
    for name, (left_role, left_step, right_role, right_step) in COMPARISONS.items():
        left_row = _row(runs[left_role], left_step)
        right_row = _row(runs[right_role], right_step)
        left = np.asarray(
            left_row["val"]["average_precision_per_query"], dtype=np.float64
        )
        right = np.asarray(
            right_row["val"]["average_precision_per_query"], dtype=np.float64
        )
        results[name] = {
            "left": {"role": left_role, "training_global_step": left_step},
            "right": {"role": right_role, "training_global_step": right_step},
            **_bootstrap(left, right, draws, seed + len(results)),
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "schema_version": 1,
        "campaign": "frozen_prompt_probe_v2_2026-09-04",
        "experiment_kind": "statistical",
        "status": "completed",
        "selection_metric": "full_pseudo_unseen_mAP",
        "official_unseen_used_for_selection": False,
        "selection_policy": "all checkpoints and contrasts are fixed from pseudo-unseen validation; official unseen is not read",
        "method": "paired bootstrap over per-query pseudo-unseen average precision",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "source_runs": {
            role: {"path": str(path), "sha256": _sha256(path)}
            for role, path in sorted(source_by_role.items())
        },
        "experiment_code_commit": commit,
        "working_tree_state": "dirty"
        if subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        else "clean",
        "comparisons": results,
        "interpretation": "These are paired query-level uncertainty intervals, not independent-seed confirmation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite statistical artifact: {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.run_results, args.output)


if __name__ == "__main__":
    main()
