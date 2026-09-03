"""Upload completed JEPA training histories and probes to W&B.

This complements the live logger in ``spica.train_jepa``: it backfills the
runs that were intentionally executed with W&B disabled while preserving each
run's local, timestamped artifacts. Checkpoints are not uploaded by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import wandb

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def latest_summary() -> Path:
    paths = sorted(OUTPUTS.glob("research_summary_jepa_*.json"))
    if not paths:
        raise FileNotFoundError("No versioned JEPA research summary exists")
    return paths[-1]


def training_runs() -> list[Path]:
    names = (
        "jepa_v2_",
        "jepa_selected_j4_",
        "jepa_no_text_j3_",
        "jepa_soft_prompt_",
    )
    runs: list[Path] = []
    for directory in sorted((OUTPUTS / "experiments").iterdir()):
        if not directory.is_dir() or not directory.name.startswith(names):
            continue
        runs.extend(path.parent for path in directory.glob("*/run_result.json"))
    return sorted(runs)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    precision = metrics.get("precision_at_k", {})
    values = {
        "mAP": metrics.get("mAP"),
        "P@200": precision.get("200"),
    }
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def probe_metrics(probe: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for prefix, metric_name in (
        ("validation", "val"),
        ("diagnostic", "diagnostic_test"),
    ):
        metrics = probe.get(metric_name)
        if isinstance(metrics, dict):
            for key, value in compact_metrics(metrics).items():
                values[f"{prefix}/{key}"] = value
    geometry = probe.get("diagnostic_test_geometry", {})
    if isinstance(geometry, dict):
        q = geometry.get("q", {})
        semantic = geometry.get("semantic", {})
        targets = geometry.get("photo_targets", {})
        for source, destination in (
            (q, "geometry/q"),
            (semantic, "geometry/semantic"),
            (targets, "geometry/photo_targets"),
        ):
            if isinstance(source, dict):
                for key in (
                    "effective_rank",
                    "mean_feature_variance",
                    "minimum_feature_variance",
                    "near_zero_variance_fraction",
                    "covariance_offdiag",
                    "mean_pairwise_cosine",
                    "global_anisotropy",
                    "predicted_target_cosine",
                    "semantic_margin",
                    "individual_positive_cosine",
                    "positive_centroid_cosine",
                    "negative_gallery_cosine",
                    "positive_negative_margin",
                ):
                    value = source.get(key)
                    if isinstance(value, (int, float)):
                        values[f"{destination}/{key}"] = float(value)
    return values


def event_history(run_result: dict[str, Any]) -> dict[int, dict[str, float]]:
    events: dict[int, dict[str, float]] = {}
    for entry in run_result.get("training_history", []):
        step = int(entry["step"])
        event = events.setdefault(step, {})
        for key, value in entry.items():
            if key not in {"step", "equivalent_epochs"} and isinstance(
                value, (int, float)
            ):
                event[f"train/{key}"] = float(value)
        event["train/equivalent_epochs"] = float(entry.get("equivalent_epochs", 0.0))
    for probe in run_result.get("probe_history", []):
        step = int(probe["step"])
        event = events.setdefault(step, {})
        event.update(probe_metrics(probe))
    return events


def artifact_files(run_dir: Path) -> list[Path]:
    names = {
        "run_result.json",
        "training_history.json",
        "train_jepa.log",
        "seen_text_bank.pt",
    }
    paths = [
        path for path in run_dir.iterdir() if path.is_file() and path.name in names
    ]
    paths.extend(sorted(run_dir.glob("probe_step*.json")))
    paths.extend(sorted((run_dir / ".hydra").glob("*.yaml")))
    return sorted(set(path for path in paths if path.is_file()))


def upload_run(
    run_dir: Path,
    *,
    project: str,
    entity: str | None,
    group: str,
    source_commit: str,
) -> dict[str, Any]:
    run_result = json.loads((run_dir / "run_result.json").read_text())
    config = dict(run_result.get("config", {}))
    run_name = f"{run_dir.parent.name}-{run_dir.name}"
    run = wandb.init(
        project=project,
        entity=entity,
        name=safe_name(run_name),
        group=group,
        job_type="jepa-training-history",
        mode="online",
        tags=[
            "spica",
            "cross-modal-jepa",
            f"encoder-{config.get('encoder_mode', 'unknown')}",
            f"m-{config.get('M', 'unknown')}",
            f"scope-{config.get('train_class_scope', 'unknown')}",
        ],
        config={
            **config,
            "source_commit": source_commit,
            "source_run_dir": str(run_dir.relative_to(ROOT)),
            "backfilled_from_local_artifact": True,
            "checkpoints_uploaded": False,
        },
    )
    events = event_history(run_result)
    for step in sorted(events):
        run.log(events[step], step=step)

    train_history = run_result.get("training_history", [])
    probes = run_result.get("probe_history", [])
    summary: dict[str, Any] = {
        "source_run_dir": str(run_dir.relative_to(ROOT)),
        "final_step": int(run_result.get("step", 0)),
        "training_points": len(train_history),
        "probe_points": len(probes),
        "text_enters_predictor": False,
    }
    validation_maps = [
        float(probe["val"]["mAP"])
        for probe in probes
        if isinstance(probe.get("val"), dict)
        and isinstance(probe["val"].get("mAP"), (int, float))
    ]
    if validation_maps:
        summary["best_validation_mAP"] = max(validation_maps)
        summary["last_validation_mAP"] = validation_maps[-1]
    if events:
        last_event = events[max(events)]
        for key, value in last_event.items():
            if key.startswith("diagnostic/") or key.startswith("train/"):
                summary[f"last/{key}"] = value
    run.summary.update(summary)

    paths = artifact_files(run_dir)
    if paths:
        artifact = wandb.Artifact(
            name=f"{safe_name(group)}-{safe_name(run_name)}",
            type="jepa-training-record",
            description="Timestamped JEPA training history, probes, and Hydra configuration; checkpoints excluded.",
            metadata={
                "source_commit": source_commit,
                "source_run_dir": str(run_dir.relative_to(ROOT)),
                "checkpoints_uploaded": False,
            },
        )
        for path in paths:
            artifact.add_file(str(path), name=str(path.relative_to(run_dir)))
        run.log_artifact(artifact, aliases=["latest", "jepa"])
    url = run.url
    run.finish()
    print(f"uploaded {run_name}: {url}")
    return {"name": run_name, "url": url, **summary}


def upload_overview(
    summary_path: Path,
    rows: list[dict[str, Any]],
    *,
    project: str,
    entity: str | None,
    group: str,
) -> None:
    summary = json.loads(summary_path.read_text())
    run = wandb.init(
        project=project,
        entity=entity,
        name="jepa-overview",
        group=group,
        job_type="jepa-research-summary",
        mode="online",
        tags=["spica", "cross-modal-jepa", "overview"],
        config={
            "source_commit": summary["repository"]["audited_commit"],
            "dataset": summary["protocol"]["dataset"],
            "pseudo_split_seed": summary["protocol"]["pseudo_split_seed"],
            "external_benchmark_verified": summary["protocol"][
                "official_external_benchmark_verified"
            ],
        },
    )
    columns = [
        "name",
        "final_step",
        "best_validation_mAP",
        "last_validation_mAP",
        "source_run_dir",
    ]
    run.log(
        {
            "research/training_runs": wandb.Table(
                columns=columns,
                data=[[row.get(column) for column in columns] for row in rows],
            )
        }
    )
    run.summary.update(
        {
            "training_run_count": len(rows),
            "source_summary": str(summary_path.relative_to(ROOT)),
        }
    )

    artifact = wandb.Artifact(
        name=f"{safe_name(group)}-summary",
        type="jepa-research-summary",
        metadata={"source_commit": summary["repository"]["audited_commit"]},
    )
    artifact.add_file(str(summary_path), name=str(summary_path.relative_to(ROOT)))
    markdown = summary_path.with_suffix(".md")
    if markdown.is_file():
        artifact.add_file(str(markdown), name=str(markdown.relative_to(ROOT)))
    plot_dir = (
        OUTPUTS
        / f"research_jepa_{summary_path.stem.removeprefix('research_summary_jepa_')}"
    )
    for plot in sorted(plot_dir.glob("*.png")):
        artifact.add_file(str(plot), name=str(plot.relative_to(ROOT)))
    run.log_artifact(artifact, aliases=["latest", "jepa", "research"])
    print(f"uploaded overview: {run.url}")
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--project", default="spica")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default="spica-jepa-2026-09-02")
    args = parser.parse_args()
    summary_path = (args.summary or latest_summary()).resolve()
    summary = json.loads(summary_path.read_text())
    run_dirs = training_runs()
    if not run_dirs:
        raise RuntimeError("No completed JEPA training runs found")
    rows = [
        upload_run(
            run_dir,
            project=args.project,
            entity=args.entity,
            group=args.group,
            source_commit=summary["repository"]["audited_commit"],
        )
        for run_dir in run_dirs
    ]
    upload_overview(
        summary_path,
        rows,
        project=args.project,
        entity=args.entity,
        group=args.group,
    )
    print(
        f"Uploaded {len(rows)} JEPA training histories to W&B project {args.project!r}."
    )


if __name__ == "__main__":
    main()
