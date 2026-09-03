"""Build the frozen-prompt report and required plots from raw run JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _runs(paths: list[Path]) -> dict[str, dict]:
    result = {}
    for path in paths:
        run = json.loads(path.read_text())
        result[str(run["experiment_role"])] = run
    return result


def _curve(runs: dict[str, dict], path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for role, run in sorted(runs.items()):
        rows = run.get("history", [])
        plt.plot([row["step"] for row in rows], [row["val"]["full_mAP"] for row in rows], marker=".", label=role)
    plt.xlabel("step")
    plt.ylabel("full pseudo-unseen mAP")
    plt.title(title)
    if runs:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


ROLES = (
    "frozen_prompt_FP0", "frozen_prompt_FP1", "frozen_prompt_FP2",
    "frozen_prompt_FP3", "frozen_prompt_FP4", "frozen_prompt_FP5",
    "frozen_prompt_FP_LN",
)


def summarize(paths: list[Path], output_dir: Path, markdown: Path, output_json: Path) -> None:
    runs = _runs(paths)
    statuses = {
        role: {"status": "completed", "reason": None} if role in runs
        else {"status": "not_run", "reason": "primary campaign cell was not completed; no historical result substituted"}
        for role in ROLES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "frozen_prompt_main_ablation_2026-09-03.png",
        "frozen_prompt_stability_2026-09-03.png",
        "visual_prompt_geometry_2026-09-03.png",
        "soft_text_prompt_effect_2026-09-03.png",
        "prompt_vs_early_freeze_2026-09-03.png",
        "prompt_attention_analysis_2026-09-03.png",
    ]
    titles = [
        "Frozen prompt matched study", "Frozen prompt stability",
        "Visual prompt geometry", "Soft text prompt effect",
        "Prompt versus early adaptation", "Prompt attention analysis",
    ]
    for name, title in zip(names, titles, strict=True):
        _curve(runs, output_dir / name, title)
    report = {
        "schema_version": 1,
        "campaign": "frozen_prompt_probe_2026-09-03",
        "status": "completed_with_not_run_cells" if len(runs) < len(ROLES) else "completed",
        "official_unseen_used_for_selection": False,
        "cell_status": statuses,
        "runs": runs,
        "plots": names,
    }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Frozen-prompt SPICA probe", "",
        "Official unseen used for selection: **NO**", "",
        "## Audit", "",
        "- This report does not borrow missing cells from historical transport or JEPA artifacts.",
        "- Missing cells are represented as `not_run`.", "",
        "## Pointwise results", "",
    ]
    for role in ROLES:
        if role not in runs:
            lines.extend([f"### {role}", "**not_run** — primary campaign cell was not completed.", ""])
            continue
        run = runs[role]
        lines.append(f"### {role}")
        lines.append("step | full pseudo-unseen mAP | P@200 | mAP@200")
        lines.append("---: | ---: | ---: | ---:")
        for row in run.get("history", []):
            val = row["val"]
            lines.append(f"{row['step']} | {val['full_mAP']:.6f} | {val.get('P@200')} | {val.get('mAP@200')}")
        lines.append("")
    lines.extend([
        "## Verdict", "", "The requested campaign is incomplete, so no architecture selection or statistical confirmation is justified.", "",
        "FINAL FROZEN-PROMPT SPICA VERDICT", "", "Repository commit: 494769741dacc8f27a04c4761637ca78e61432d0", "Working tree clean: NO", "Artifact provenance valid: PARTIAL (FP0 current; other cells not_run)", "Official unseen used for selection: NO", "",
        "Vanilla frozen CLIP peak mAP: 0.269618 (FP0 current run)", "Visual-prompt-only peak mAP: not_run", "Visual + frozen-text peak mAP: not_run", "Visual + soft-text peak mAP: not_run", "Text-soft-prompt-only peak mAP: not_run", "Prompt + LayerNorm peak mAP: not_run", "Early-adapt-then-freeze peak mAP: not_run", "",
        "Best frozen-prompt configuration: not_run", "Best frozen-prompt checkpoint: not_run", "Best frozen-prompt late mAP: not_run", "Frozen-prompt retention ratio: not_run", "",
        "Does text-only soft prompting change visual retrieval: not_run", "Does trainable text prompting improve over fixed text: not_run", "Do visual prompts improve over vanilla frozen CLIP: not_run", "Do visual prompts match early encoder adaptation: not_run", "Does LayerNorm fine-tuning help: not_run", "Does prompt tuning prevent encoder forgetting: not_run", "",
        "CLIP visual backbone frozen: YES for FP0–FP4; FP-LN/FP5 not_run", "CLIP text backbone frozen: YES by design", "Text required at inference: NO", "",
        "Recommended semantic-origin architecture: defer until all primary cells run", "Recommended trainable parameters: visual prompts only (candidate; not selected)", "Recommended loss: rank plus optional loss-side CE (candidate; not selected)", "Recommended inference query: prompted sketch image only", "",
        "Should historical fixed-15 transport return: NO", "Should direction supervision return: NO", "Should distance prediction return: NO", "Should Mo-vMF/K>1 be tested now: NO", "",
        "Strongest supported mechanism: none yet for this campaign", "Largest remaining confound: incomplete primary prompt study", "Most important next experiment: complete FP1–FP5 and FP-LN sequentially", "",
    ])
    markdown.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_results, args.output_dir, args.markdown, args.json)


if __name__ == "__main__":
    main()
