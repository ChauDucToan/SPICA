"""Select frozen-prompt checkpoints using pseudo-unseen mAP only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROLES = {
    "frozen_prompt_FP0", "frozen_prompt_FP1", "frozen_prompt_FP2",
    "frozen_prompt_FP3", "frozen_prompt_FP4", "frozen_prompt_FP5",
    "frozen_prompt_FP_LN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("at least one run_result.json is required")
    runs = [json.loads(path.read_text()) for path in paths]
    by_role: dict[str, dict] = {}
    for run in runs:
        role = run.get("experiment_role")
        if role not in ROLES:
            raise ValueError(f"invalid or missing experiment role: {role!r}")
        if role in by_role:
            raise ValueError(f"duplicate experiment role: {role}")
        if run.get("official_unseen_used_for_selection") is not False:
            raise ValueError(f"official unseen is not permitted for {role}")
        if not run.get("history"):
            raise ValueError(f"run has no pointwise history: {role}")
        by_role[role] = run
    missing = ROLES - by_role.keys()
    if missing:
        raise ValueError(f"missing required experiment roles: {sorted(missing)}")
    identities = {(json.dumps(run.get("pseudo_split"), sort_keys=True), json.dumps(run.get("manifest_identity"), sort_keys=True)) for run in by_role.values()}
    if len(identities) != 1:
        raise ValueError("runs do not share one split/manifest identity")
    candidates = []
    for role, run in sorted(by_role.items()):
        peak = max(run["history"], key=lambda row: (float(row["val"]["full_mAP"]), -int(row["step"])))
        candidates.append({"role": role, "step": int(peak["step"]), "full_mAP": float(peak["val"]["full_mAP"]), "checkpoint": peak["checkpoint"], "checkpoint_sha256": _sha256(Path(peak["checkpoint"]))})
    selected = max(candidates, key=lambda row: (row["full_mAP"], row["role"] == "frozen_prompt_FP0"))
    result = {"schema_version": 1, "campaign": "frozen_prompt_probe_2026-09-03", "selection_metric": "full_pseudo_unseen_mAP", "official_unseen_used_for_selection": False, "candidates": candidates, "selected": selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    select(args.run_results, args.output)


if __name__ == "__main__":
    main()
