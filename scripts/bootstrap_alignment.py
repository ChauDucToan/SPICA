"""Paired bootstrap for raw pseudo-unseen per-query AP vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _row(result: dict[str, Any], horizon: int) -> dict[str, Any]:
    rows = [
        row
        for row in result.get("history", [])
        if int(row.get("training_global_step", -1)) == horizon
    ]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one raw row at step {horizon}, got {len(rows)}")
    value = rows[0].get("val", {})
    if not isinstance(value, dict) or not value.get("average_precision_per_query"):
        raise ValueError("aligned per-query AP is missing")
    if not value.get("query_identity"):
        raise ValueError("query identity is missing")
    return rows[0]


def _bootstrap(
    delta: np.ndarray, *, repetitions: int, seed: int, chunk_size: int = 256
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        means.append(delta[indices].mean(axis=1))
    samples = np.concatenate(means)
    return {
        "n_queries": int(delta.size),
        "repetitions": repetitions,
        "bootstrap_seed": seed,
        "mean_delta": float(delta.mean()),
        "sample_std_query_delta": float(delta.std(ddof=1)) if delta.size > 1 else None,
        "ci_95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    candidate_path: Path,
    control_path: Path,
    *,
    horizon: int,
    output: Path,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text())
    control = json.loads(control_path.read_text())
    candidate_row = _row(candidate, horizon)
    control_row = _row(control, horizon)
    candidate_val = candidate_row["val"]
    control_val = control_row["val"]
    if candidate_val["query_identity"] != control_val["query_identity"]:
        raise ValueError("query identities are not exactly aligned")
    candidate_ap = np.asarray(candidate_val["average_precision_per_query"], dtype=np.float64)
    control_ap = np.asarray(control_val["average_precision_per_query"], dtype=np.float64)
    if candidate_ap.shape != control_ap.shape:
        raise ValueError("per-query AP vectors have different shapes")
    if not np.isfinite(candidate_ap).all() or not np.isfinite(control_ap).all():
        raise ValueError("per-query AP contains non-finite values")
    result = {
        "schema_version": 1,
        "candidate": str(candidate_path.resolve()),
        "control": str(control_path.resolve()),
        "candidate_checkpoint": candidate_row.get("checkpoint"),
        "control_checkpoint": control_row.get("checkpoint"),
        "candidate_checkpoint_sha256": candidate_row.get("checkpoint_sha256"),
        "control_checkpoint_sha256": control_row.get("checkpoint_sha256"),
        "horizon": horizon,
        "metric": "pseudo_unseen_per_query_average_precision",
        "query_identity_sha256": candidate_val["query_identity"].get("sha256"),
        "bootstrap": _bootstrap(
            candidate_ap - control_ap,
            repetitions=repetitions,
            seed=seed,
        ),
        "raw_mAP_delta": float(candidate_ap.mean() - control_ap.mean()),
        "official_unseen_used": False,
        "artifact_hashes": {
            "candidate_run_result_sha256": _sha256(candidate_path),
            "control_run_result_sha256": _sha256(control_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["bootstrap"], indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    run(
        args.candidate,
        args.control,
        horizon=args.horizon,
        output=args.output,
        repetitions=args.repetitions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
