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
    for key in ("mAP", "P@200"):
        value = experiment.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    denominator = experiment.get("map_at_k_denominator")
    explicit_key = (
        f"mAP@200_{denominator}" if isinstance(denominator, str) else None
    )
    for key in (explicit_key, "mAP@200_prefix_positive", "mAP@200_all_relevant"):
        if key is None:
            continue
        value = experiment.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _prefix_positive_map_at_200(experiment: dict[str, Any]) -> float | None:
    value = experiment.get("mAP@200_prefix_positive")
    if isinstance(value, (int, float)):
        return float(value)
    if experiment.get("map_at_k_denominator") == "prefix_positive":
        value = experiment.get("mAP@200")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _normalized_experiments(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept both the legacy summary schema and the current run ledger."""
    if "experiments" in summary:
        experiments = [dict(experiment) for experiment in summary["experiments"]]
        metric_definition = summary.get("metric_definition", {})
        historical_definition = str(metric_definition.get("mAP@200", ""))
        denominator = (
            "prefix_positive"
            if "positives in the top-200 prefix" in historical_definition
            else None
        )
        for experiment in experiments:
            if denominator is not None and "mAP@200" in experiment:
                experiment["mAP@200_prefix_positive"] = experiment.pop("mAP@200")
                experiment["map_at_k_denominator"] = denominator
        return experiments

    if "existing_results" in summary:
        experiments = []
        for result in summary["existing_results"]:
            experiment = dict(result)
            experiment["mAP"] = experiment.get("map")
            experiment["P@200"] = experiment.get("p_at_200")
            if "map_at_200_historical" in experiment:
                experiment["mAP@200_prefix_positive"] = experiment.pop(
                    "map_at_200_historical"
                )
                experiment["map_at_k_denominator"] = "prefix_positive"
            experiment.setdefault("scoring", "cosine")
            experiment.setdefault("artifact", None)
            experiments.append(experiment)
        return experiments

    experiments: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        primary = run.get("primary_metrics", {})
        config = run.get("config", {})
        experiments.append(
            {
                "artifact": run["analysis_artifact"],
                "model": run["model"],
                "K": config.get(
                    "num_components",
                    1 if str(run.get("model", "")).startswith("K=1") else 3,
                ),
                "M": config.get("num_positive_photos", 3),
                "kappa": run.get("kappa"),
                "scoring": "gate_barycenter",
                "params": run.get("trainable_parameters"),
                "steps": run.get("steps"),
                "mAP": primary.get("mAP"),
                "mAP@200_prefix_positive": primary.get("mAP@200_prefix_positive"),
                "P@200": primary.get("P@200"),
                "map_at_k_denominator": "prefix_positive",
            }
        )
    return experiments


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
    experiments = _normalized_experiments(summary)
    overview_rows: list[list[Any]] = []
    repository = summary.get("repository", {})
    protocol = summary.get("protocol", {})
    dataset = repository.get("dataset", protocol.get("dataset"))
    seed = repository.get("seed")

    for index, experiment in enumerate(experiments):
        artifact_path = experiment.get("artifact")
        if not isinstance(artifact_path, str):
            print(f"Skipping {experiment['model']}: no metrics artifact recorded")
            continue
        metrics_path = root / artifact_path
        if not metrics_path.is_file():
            print(f"Skipping missing metrics artifact: {metrics_path}")
            continue

        run_name = f"{index:02d}-{_safe_name(experiment['model'])}-{_safe_name(experiment['scoring'])}"
        run_config = {
            "source_commit": summary["repository"]["commit"],
            "branch": summary["repository"]["branch"],
            "dataset": dataset,
            "seed": seed,
            "diagnostic_test_evaluation": True,
            "map_at_k_denominator": experiment.get(
                "map_at_k_denominator", "prefix_positive"
            ),
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
            "mAP@200_prefix_positive": _prefix_positive_map_at_200(
                experiment
            ),
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
            experiment.get("mAP"),
            _prefix_positive_map_at_200(experiment),
            experiment.get("P@200"),
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
            "dataset": dataset,
            "seed": seed,
            "diagnostic_test_evaluation": True,
        },
    )
    overview.log({
        "research/experiment_table": wandb.Table(
            columns=["model", "K", "M", "kappa", "scoring", "params", "steps_epochs", "mAP", "mAP@200_prefix_positive", "P@200"],
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
