"""Select v2 prompt checkpoints only after strict artifact validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.summarize_frozen_prompt import load_runs
except ModuleNotFoundError:  # direct ``python scripts/select_frozen_prompt.py``
    from summarize_frozen_prompt import load_runs
from spica.frozen_prompt_artifacts import CAMPAIGN, ROLES


def select(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("at least one run_result.json is required")
    raw = [json.loads(path.read_text()) for path in paths]
    raw_roles = [item.get("experiment_role") for item in raw]
    if len(raw_roles) != len(set(raw_roles)):
        duplicates = sorted({role for role in raw_roles if raw_roles.count(role) > 1})
        raise ValueError(f"duplicate experiment role: {duplicates}")
    missing = sorted(set(ROLES) - set(raw_roles))
    if missing:
        raise ValueError(f"missing required experiment roles: {missing}")
    runs = load_runs(paths)
    candidates = []
    for role in ROLES:
        run = runs[role]
        peak = max(
            run["history"],
            key=lambda row: (
                float(row["full_pseudo_unseen_mAP"]),
                -int(row["training_global_step"]),
            ),
        )
        candidates.append(
            {
                "role": role,
                "training_global_step": int(peak["training_global_step"]),
                "full_pseudo_unseen_mAP": float(peak["full_pseudo_unseen_mAP"]),
                "checkpoint": peak["checkpoint"],
                "checkpoint_sha256": peak["checkpoint_sha256"],
            }
        )
    prompted = [
        item
        for item in candidates
        if item["role"]
        in {
            "frozen_prompt_v2_FP1",
            "frozen_prompt_v2_FP1S",
            "frozen_prompt_v2_FP2",
            "frozen_prompt_v2_FP3",
            "frozen_prompt_v2_FP_LN",
        }
    ]
    selected = max(
        prompted, key=lambda item: (item["full_pseudo_unseen_mAP"], item["role"])
    )
    result = {
        "schema_version": 2,
        "campaign": CAMPAIGN,
        "selection_metric": "full_pseudo_unseen_mAP",
        "official_unseen_used_for_selection": False,
        "candidates": candidates,
        "selected_prompt": selected,
        "matched_fp5": next(
            item for item in candidates if item["role"] == "frozen_prompt_v2_FP5"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing selection artifact: {output}"
        )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    select(args.run_results, args.output)


if __name__ == "__main__":
    main()
