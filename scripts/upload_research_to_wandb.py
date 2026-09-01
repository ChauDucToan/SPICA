"""Upload SPICA research experiments as individually grouped W&B runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import wandb


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def _numeric_metrics(experiment: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("mAP", "mAP@200", "P@200"):
        value = experiment.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _experiment_files(root: Path, metrics_path: Path) -> list[Path]:
    # Include only the experiment's logs/configs/metrics. Checkpoints are
    # intentionally excluded because they are large and were not part of the
    # previous research-log upload.
    experiment_root = metrics_path.parent
    paths = [
        path
        for path in experiment_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".log", ".yaml", ".md"}
    ]
    return sorted(set(paths))


def upload(summary_path: Path, *, project: str, group: str, entity: str | None) -> None:
    root = summary_path.parents[1]
    summary = json.loads(summary_path.read_text())
    experiments = summary["experiments"]
    overview_rows: list[list[Any]] = []

    for index, experiment in enumerate(experiments):
        metrics_path = root / experiment["artifact"]
        if not metrics_path.is_file():
            print(f"Skipping missing metrics artifact: {metrics_path}")
            continue

        run_name = f"{index:02d}-{_safe_name(experiment['model'])}-{_safe_name(experiment['scoring'])}"
        run_config = {
            "source_commit": summary["repository"]["commit"],
            "branch": summary["repository"]["branch"],
            "dataset": summary["repository"]["dataset"],
            "seed": summary["repository"]["seed"],
            "diagnostic_test_evaluation": True,
            "model": experiment["model"],
            "K": experiment["K"],
            "M": experiment["M"],
            "kappa": experiment["kappa"],
            "scoring": experiment["scoring"],
            "params": experiment["params"],
            "steps_epochs": experiment["steps"],
        }
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            job_type="evaluation" if "eval" in metrics_path.parts else "research",
            tags=[
                "spica",
                "research",
                "diagnostic-test",
                f"K{experiment['K']}",
                f"M{experiment['M']}",
                _safe_name(experiment["scoring"]),
            ],
            config=run_config,
        )
        run.log({f"retrieval/{key}": value for key, value in _numeric_metrics(experiment).items()})
        run.summary.update({
            "model": experiment["model"],
            "K": experiment["K"],
            "M": experiment["M"],
            "scoring": experiment["scoring"],
            "mAP": experiment.get("mAP"),
            "mAP@200": experiment.get("mAP@200"),
            "P@200": experiment.get("P@200"),
            "diagnostic_test_evaluation": True,
        })

        artifact = wandb.Artifact(
            name=f"spica-{index:02d}-{_safe_name(experiment['model'])}-{_safe_name(experiment['scoring'])}",
            type="experiment-logs",
            description=f"Logs, metrics, and Hydra configuration for {experiment['model']} ({experiment['scoring']}).",
            metadata={"group": group, "source_commit": summary["repository"]["commit"], "checkpoints_included": False},
        )
        for path in _experiment_files(root, metrics_path):
            artifact.add_file(str(path), name=path.relative_to(root).as_posix())
        run.log_artifact(artifact, aliases=["latest", "research"])
        run.finish()
        print(f"{run_name}: {run.url}")

        overview_rows.append([
            experiment["model"], experiment["K"], experiment["M"], experiment["kappa"],
            experiment["scoring"], experiment["params"], experiment["steps"],
            experiment.get("mAP"), experiment.get("mAP@200"), experiment.get("P@200"),
        ])

    overview = wandb.init(
        project=project,
        entity=entity,
        name="overview",
        group=group,
        job_type="research-summary",
        tags=["spica", "research", "overview", "diagnostic-test"],
        config={
            "source_commit": summary["repository"]["commit"],
            "dataset": summary["repository"]["dataset"],
            "seed": summary["repository"]["seed"],
            "diagnostic_test_evaluation": True,
        },
    )
    overview.log({
        "research/experiment_table": wandb.Table(
            columns=["model", "K", "M", "kappa", "scoring", "params", "steps_epochs", "mAP", "mAP@200", "P@200"],
            # Keep mixed metadata columns homogeneous for W&B's inferred schema.
            data=[
                [
                    *map(str, row[:7]),
                    row[7],
                    row[8] if isinstance(row[8], (int, float)) else None,
                    row[9],
                ]
                for row in overview_rows
            ],
        )
    })
    overview.summary["group"] = group
    overview.summary["experiment_count"] = len(overview_rows)
    summary_artifact = wandb.Artifact(
        name=f"spica-{_safe_name(group)}-summary",
        type="research-summary",
        metadata={"group": group, "source_commit": summary["repository"]["commit"]},
    )
    summary_artifact.add_file(str(summary_path), name="outputs/research_summary.json")
    summary_artifact.add_file(str(summary_path.with_suffix(".md")), name="outputs/research_summary.md")
    overview.log_artifact(summary_artifact, aliases=["latest", "research"])
    overview.finish()
    print(f"overview: {overview.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("outputs/research_summary.json"))
    parser.add_argument("--project", default="spica")
    parser.add_argument("--group", default="spica-research-2026-09-01")
    parser.add_argument("--entity", default=None)
    arguments = parser.parse_args()
    upload(arguments.summary, project=arguments.project, group=arguments.group, entity=arguments.entity)
