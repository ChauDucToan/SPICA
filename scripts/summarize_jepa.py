"""Build the versioned JEPA research summary and compact diagnostic plots."""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DATE = os.environ.get("SPICA_REPORT_DATE", date.today().isoformat())
SUMMARY_JSON = OUTPUTS / f"research_summary_jepa_{DATE}.json"
SUMMARY_MD = OUTPUTS / f"research_summary_jepa_{DATE}.md"
ARTIFACT_DIR = OUTPUTS / f"research_jepa_{DATE}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def one_run(name: str) -> tuple[Path, dict[str, Any]]:
    paths = sorted((OUTPUTS / "experiments" / name).glob("*/run_result.json"))
    if not paths:
        raise FileNotFoundError(f"No run_result.json found for {name}")
    path = paths[-1]
    return path.parent, json.loads(path.read_text())


def one_metrics(name: str) -> tuple[Path, dict[str, Any]]:
    paths = sorted((OUTPUTS / "experiments" / name).glob("*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No metrics.json found for {name}")
    path = paths[-1]
    return path, json.loads(path.read_text())


def one_pseudo(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = run_dir / "pseudo_validation.json"
    if not path.is_file():
        raise FileNotFoundError(f"No pseudo-validation result at {path}")
    return path, json.loads(path.read_text())


def metric_block(value: dict[str, Any]) -> dict[str, Any]:
    precision = value.get("precision_at_k", {})
    return {
        "mAP": float(value["mAP"]),
        "P@1": float(precision.get("1", float("nan"))),
        "P@10": float(precision.get("10", float("nan"))),
        "P@200": float(precision["200"]),
        "mAP@200": float(value.get("mAP_at_k", {}).get("200", float("nan"))),
        "num_queries": int(value["num_queries"]),
        "num_gallery_items": int(value["num_gallery_items"]),
    }


def geometry_block(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "effective_rank",
        "mean_feature_variance",
        "minimum_feature_variance",
        "near_zero_variance_fraction",
        "covariance_offdiag",
        "mean_pairwise_cosine",
        "global_anisotropy",
        "mean_norm",
    )
    return {key: float(value[key]) for key in keys if key in value}


def compact_geometry(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: geometry_block(value[key]) for key in ("h", "u", "q") if key in value
    }
    for key in ("semantic", "photo_targets"):
        if key in value:
            result[key] = {
                subkey: float(subvalue)
                if isinstance(subvalue, (int, float))
                else subvalue
                for subkey, subvalue in value[key].items()
                if subkey not in {"num_classes"}
            }
    return result


def probe_block(probe: dict[str, Any]) -> dict[str, Any]:
    geometry = probe.get("diagnostic_test_geometry", {})
    semantic = geometry.get("semantic", {})
    q = geometry.get("q", {})
    return {
        "step": int(probe["step"]),
        "equivalent_epochs": float(probe.get("equivalent_epochs", float("nan"))),
        "pseudo_validation": metric_block(probe["val"]),
        "diagnostic_test": metric_block(probe["diagnostic_test"]),
        "q_effective_rank": float(q.get("effective_rank", float("nan"))),
        "q_mean_pairwise_cosine": float(q.get("mean_pairwise_cosine", float("nan"))),
        "q_global_anisotropy": float(q.get("global_anisotropy", float("nan"))),
        "predicted_target_cosine": float(
            semantic.get("predicted_target_cosine", float("nan"))
        ),
        "semantic_margin": float(semantic.get("semantic_margin", float("nan"))),
        "protocol": probe.get("protocol", {}),
        "artifact": rel(Path(str(probe["checkpoint"])))
        if "checkpoint" in probe
        else None,
    }


def load_ablation() -> dict[str, Any]:
    definitions = {
        "J0": ("jepa_v2_j0_frozen_m1", "jepa_v2_eval_j0", "frozen encoder, M=1"),
        "J1": ("jepa_v2_j1_partial_m1", "jepa_v2_eval_j1", "partial encoder, M=1"),
        "J2": ("jepa_v2_j2_full_m1", "jepa_v2_eval_j2", "full encoder, M=1"),
        "J3": ("jepa_v2_j3_partial_m3", "jepa_v2_eval_j3", "partial encoder, M=3"),
        "J4": (
            "jepa_v2_j4_partial_m3_text",
            "jepa_v2_eval_j4",
            "partial encoder, M=3 + frozen text classification",
        ),
        "J5": (
            "jepa_v2_j5_partial_m3_vicreg",
            "jepa_v2_eval_j5",
            "J4 objective + VICReg",
        ),
        "J6": (
            "jepa_v2_j6_partial_m3_sigreg",
            "jepa_v2_eval_j6",
            "J4 objective + SIGReg",
        ),
    }
    result: dict[str, Any] = {}
    for ablation, (run_name, eval_name, description) in definitions.items():
        run_dir, run = one_run(run_name)
        pseudo_path, pseudo = one_pseudo(run_dir)
        eval_path, evaluation = one_metrics(eval_name)
        result[ablation] = {
            "description": description,
            "config": run["config"],
            "pseudo_validation": metric_block(pseudo["metrics"]),
            "diagnostic_test": metric_block(evaluation["metrics"]),
            "geometry": compact_geometry(evaluation["feature_geometry"]),
            "artifacts": {
                "run_result": rel(run_dir / "run_result.json"),
                "pseudo_validation": rel(pseudo_path),
                "diagnostic_evaluation": rel(eval_path),
                "checkpoint": rel(Path(str(run["checkpoint"]))),
            },
        }
    return result


def load_long_run(name: str) -> dict[str, Any]:
    run_dir, run = one_run(name)
    return {
        "config": run["config"],
        "probes": [probe_block(probe) for probe in run["probe_history"]],
        "artifacts": {
            "run_result": rel(run_dir / "run_result.json"),
            "training_history": rel(run_dir / "training_history.json"),
            "probe_files": [
                rel(run_dir / f"probe_step{int(probe['step'])}.json")
                for probe in run["probe_history"]
            ],
        },
    }


def seed_result(run_name: str, eval_name: str | None = None) -> dict[str, Any]:
    run_dir, run = one_run(run_name)
    pseudo_path, pseudo = one_pseudo(run_dir)
    result: dict[str, Any] = {
        "seed": int(run["config"]["seed"]),
        "pseudo_validation": metric_block(pseudo["metrics"]),
        "artifacts": {
            "run_result": rel(run_dir / "run_result.json"),
            "pseudo_validation": rel(pseudo_path),
            "checkpoint": rel(Path(str(run["checkpoint"]))),
        },
    }
    if eval_name is not None:
        eval_path, evaluation = one_metrics(eval_name)
        result["diagnostic_test"] = metric_block(evaluation["metrics"])
        result["artifacts"]["diagnostic_evaluation"] = rel(eval_path)
    return result


def aggregate_seed(results: list[dict[str, Any]], field: str) -> dict[str, float]:
    maps = [float(result[field]["mAP"]) for result in results]
    p200 = [float(result[field]["P@200"]) for result in results]
    return {
        "mAP_mean": mean(maps),
        "mAP_population_std": pstdev(maps),
        "P@200_mean": mean(p200),
        "P@200_population_std": pstdev(p200),
    }


def build_summary() -> dict[str, Any]:
    ablations = load_ablation()
    seed_results = [
        seed_result("jepa_v2_j4_partial_m3_text", "jepa_v2_eval_j4"),
        seed_result("jepa_selected_j4_seed123", "jepa_eval_seed123"),
        seed_result("jepa_selected_j4_seed3407", "jepa_eval_seed3407"),
    ]
    soft_dir, soft_run = one_run("jepa_soft_prompt_j4_100")
    soft_probe = soft_run["probe_history"][0]

    selected_dir, selected_run = one_run("jepa_selected_j4_all_seen_long")
    selected_eval_path, selected_eval = one_metrics(
        "jepa_selected_j4_all_seen_step100_eval"
    )
    selected_probes = {
        int(probe["step"]): probe for probe in selected_run["probe_history"]
    }
    selected_retrain = {
        "selection_rule": "best valid pseudo-unseen mAP from the 100-step J4 ablation; retrain on all 104 seen classes",
        "selected_step": 100,
        "selection_artifact": ablations["J4"]["artifacts"]["pseudo_validation"],
        "step_100_diagnostic": metric_block(selected_eval["metrics"]),
        "step_5400_diagnostic": metric_block(selected_probes[5400]["diagnostic_test"]),
        "step_100_geometry": compact_geometry(selected_eval["feature_geometry"]),
        "step_5400_geometry": compact_geometry(
            selected_probes[5400]["diagnostic_test_geometry"]
        ),
        "validation_is_diagnostic_after_retraining": True,
        "artifacts": {
            "run_result": rel(selected_dir / "run_result.json"),
            "checkpoint_step_100": rel(
                selected_dir / "checkpoints" / "jepa_step100.pt"
            ),
            "checkpoint_final_step_5400": rel(Path(str(selected_run["checkpoint"]))),
            "step_100_evaluation": rel(selected_eval_path),
        },
    }

    summary: dict[str, Any] = {
        "report_version": 1,
        "date": DATE,
        "repository": {
            "audited_commit": "1a6ec5a38ef41b534b17f641a8d5534d1b52cfe2",
            "working_tree_contains_jepa_changes": True,
            "branch_at_audit": "experiments",
        },
        "protocol": {
            "dataset": "local Sketchy zeroshot0 / sketchy_104_21",
            "train_classes": 104,
            "official_unseen_classes": 21,
            "official_queries": 12694,
            "official_gallery": 12553,
            "pseudo_split": "84 train / 20 validation classes",
            "pseudo_split_seed": 3407,
            "metric": "full-gallery category mAP and P@200",
            "map_at_k_denominator": "prefix_positive",
            "official_external_benchmark_verified": False,
            "official_test_during_training": "diagnostic only; never used for model selection",
            "photo_gallery_reencoded_at_inference": False,
        },
        "architecture": {
            "model_family": "cross_modal_jepa",
            "sketch_path": [
                "raw sketch image",
                "CLIP-initialized trainable visual tower",
                "residual predictor",
                "L2-normalized 512-D predicted photo-semantic query",
            ],
            "photo_target": "frozen OpenCLIP ViT-B-32-quickgelu/OpenAI normalized multi-photo centroid",
            "target_stop_gradient": True,
            "inference_inputs": ["raw_sketch_image"],
            "text_at_inference": False,
            "photos_at_inference": False,
            "oracle_class_at_inference": False,
            "movmf_in_initial_jepa": False,
            "parameter_counts_J4": ablations["J4"]["config"],
        },
        "ablations_J0_J6": ablations,
        "multi_seed_J4": {
            "seeds": seed_results,
            "pseudo_validation_aggregate": aggregate_seed(
                seed_results, "pseudo_validation"
            ),
            "diagnostic_test_aggregate": aggregate_seed(
                seed_results, "diagnostic_test"
            ),
            "note": "These are independent 100-step pseudo-train runs with the same fixed 20-class pseudo-validation split.",
        },
        "long_training": {
            "selected_J4_pseudo_train": load_long_run("jepa_selected_j4_pseudo_long"),
            "no_text_J3_pseudo_train": load_long_run("jepa_no_text_j3_pseudo_long"),
            "selected_J4_all_seen_retrain": load_long_run(
                "jepa_selected_j4_all_seen_long"
            ),
            "no_text_J3_all_seen_diagnostic": load_long_run(
                "jepa_no_text_j3_all_seen_long"
            ),
            "interpretation": "Long-run validation is valid only for the pseudo_train runs; all_seen validation is labeled diagnostic because all 104 classes were fitted.",
        },
        "selected_retrain": selected_retrain,
        "soft_prompt_phase": {
            "config": soft_run["config"],
            "step_100_pseudo_validation": metric_block(soft_probe["val"]),
            "step_100_diagnostic_test": metric_block(soft_probe["diagnostic_test"]),
            "geometry": compact_geometry(soft_probe["diagnostic_test_geometry"]),
            "artifacts": {
                "run_result": rel(soft_dir / "run_result.json"),
                "checkpoint": rel(Path(str(soft_run["checkpoint"]))),
                "text_bank": rel(soft_dir / "seen_text_bank.pt"),
            },
            "note": "The CLIP text tower is frozen; only four context vectors receive the classification-loss gradient. The bank is not part of the predictor or inference path.",
        },
        "geometry_reference": {
            "raw_frozen_clip_sketch": {
                "effective_rank": 28.158601760864258,
                "mean_feature_variance": 0.00037781629362143576,
                "mean_pairwise_cosine": 0.8069459795951843,
                "global_anisotropy": 0.12153153866529465,
                "source": "outputs/sketchy_104_21/clip_openai_quickgelu/sketches.pt",
            },
            "selected_J4_step_100": compact_geometry(selected_eval["feature_geometry"]),
            "selected_J4_step_5400": compact_geometry(
                selected_probes[5400]["diagnostic_test_geometry"]
            ),
        },
        "historical_baselines": {
            "matched_stageE_no_vmf_control": {
                "mAP": 0.5003783569947221,
                "P@200": 0.5842815383570893,
                "artifact": "outputs/experiments/stageE_mechanism_2026-09-01/no_vmf_100/analysis.json",
                "note": "Best prior verified internal matched control; preserved and not overwritten.",
            },
            "raw_CLIP": {
                "full_mAP": 0.247028,
                "P@200": 0.336575,
                "source": "outputs/research_summary.json",
            },
        },
        "research_verdict": {
            "headline": "Conditional success: the sketch-only JEPA is viable and slightly exceeds the prior matched no-vMF control at its validated early checkpoint, but it is not long-training stable.",
            "what_is_supported": [
                "The predictor output is directly retrievable against the frozen photo gallery using raw sketches only at inference.",
                "Partial CLIP visual adaptation is the best encoder choice in the 100-step ablation; full adaptation is not better.",
                "Frozen text classification is a strong loss-side semantic shield in short training, and a four-vector soft prompt improves the single-seed short run further.",
                "No-vMF JEPA implementation and ablations do not reintroduce Mo-vMF.",
            ],
            "what_is_not_supported": [
                "M=3 alone does not beat M=1 in the no-text comparison.",
                "VICReg and SIGReg do not provide a decisive retrieval improvement over J4 at 100 steps.",
                "Frozen text classification does not preserve absolute zero-shot performance during longer training: valid pseudo-unseen J4 mAP falls from 0.496992 at step 100 to 0.363167 at step 5400.",
                "The external benchmark mAP@200 protocol remains unverified, so these are internal reproducible results rather than a claim of official benchmark parity.",
            ],
            "numbers": {
                "J4_pseudo_mAP_step100": 0.4969915484508847,
                "J4_pseudo_mAP_step5400": 0.363167,
                "J3_no_text_pseudo_mAP_step100": 0.440555,
                "J3_no_text_pseudo_mAP_step5400": 0.336334,
                "J4_retrained_all_seen_mAP_step100": 0.5102346400112266,
                "J4_retrained_all_seen_mAP_step5400": 0.3906093074529876,
                "prior_matched_no_vmf_mAP": 0.5003783569947221,
                "soft_prompt_pseudo_mAP_step100": 0.506279,
            },
            "recommendation": "Keep the text-free JEPA code path and use early stopping selected only on pseudo-unseen classes. Treat soft prompting as promising but provisional. Do not claim long-training stability; next work should tune schedules/regularization against validated curves, add independent datasets, and verify the external evaluator before scaling the claim.",
        },
    }
    return summary


def write_markdown(summary: dict[str, Any]) -> None:
    a = summary["ablations_J0_J6"]
    lines = [
        f"# SPICA cross-modal JEPA research summary ({DATE})",
        "",
        f"Audited at `{summary['repository']['audited_commit']}`; outputs are timestamped under `outputs/experiments/`.",
        "",
        "## Verdict",
        f"**{summary['research_verdict']['headline']}**",
        "",
        summary["research_verdict"]["recommendation"],
        "",
        "## Protocol",
        "- Local Sketchy `zeroshot0`: 104 seen / 21 unseen classes; 12,694 queries and 12,553 gallery photos.",
        "- Pseudo-zero-shot split: 84 train / 20 validation classes, seed 3407.",
        "- Primary metric: full-gallery category mAP with prefix-positive mAP@K denominator; external benchmark compatibility is unverified.",
        "- Official test results during fitting are diagnostic only. Inference consumes raw sketch images only.",
        "",
        "## J0–J6 ablation (100 steps, seed 42)",
        "| ID | Configuration | pseudo mAP | pseudo P@200 | diagnostic mAP | diagnostic P@200 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in ("J0", "J1", "J2", "J3", "J4", "J5", "J6"):
        value = a[key]
        p = value["pseudo_validation"]
        t = value["diagnostic_test"]
        lines.append(
            f"| {key} | {value['description']} | {p['mAP']:.4f} | {p['P@200']:.4f} | {t['mAP']:.4f} | {t['P@200']:.4f} |"
        )
    lines += [
        "",
        "J4 is selected because it has the best valid pseudo-validation mAP among J0–J6 (J6 is statistically tied but slightly lower).",
        "",
        "## Multi-seed J4",
        "| seed | pseudo mAP | pseudo P@200 | diagnostic mAP | diagnostic P@200 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for value in summary["multi_seed_J4"]["seeds"]:
        lines.append(
            f"| {value['seed']} | {value['pseudo_validation']['mAP']:.4f} | {value['pseudo_validation']['P@200']:.4f} | {value['diagnostic_test']['mAP']:.4f} | {value['diagnostic_test']['P@200']:.4f} |"
        )
    aggregate = summary["multi_seed_J4"]["pseudo_validation_aggregate"]
    lines.append(
        f"Aggregate pseudo mAP: **{aggregate['mAP_mean']:.4f} ± {aggregate['mAP_population_std']:.4f}** (population SD)."
    )
    lines += [
        "",
        "## Long-training stability",
        "| model / split | step 100 | step 500 | step 1000 | step 1800 | step 5400 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, run_key in (
        ("J4 + frozen text / pseudo validation", "selected_J4_pseudo_train"),
        ("J3 no text / pseudo validation", "no_text_J3_pseudo_train"),
    ):
        probes = summary["long_training"][run_key]["probes"]
        by_step = {probe["step"]: probe["pseudo_validation"]["mAP"] for probe in probes}
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{by_step[step]:.4f}" for step in (100, 500, 1000, 1800, 5400)
            )
            + " |"
        )
    lines += [
        "",
        "Text classification improves the absolute curve at every listed pseudo-validation checkpoint and retains a late advantage, but it does **not** preserve the step-100 score. Both models lose zero-shot retrieval quality with longer training.",
        "",
        "## Geometry and target semantics",
        "- The selected all-seen retrain at step 100 has query effective rank 11.93/512, mean pairwise query cosine 0.627, and global anisotropy 0.234; raw frozen CLIP sketch features have rank 28.16 and pairwise cosine 0.807.",
        "- By step 5400, J4 query rank rises to 15.15 while predicted-to-photo-centroid cosine falls from 0.834 to 0.767 and semantic margin falls from 0.195 to 0.172. Retrieval degradation is therefore semantic drift/overfitting, not a single monotonic scalar-collapse signature.",
        "- At selected step 100, mean cosine to individual positive photos is 0.707 versus 0.834 to the normalized positive centroid and 0.562 to negative-gallery photos; the centroid is a materially cleaner target than any one photo.",
        "",
        "## Soft-prompt phase",
        "A four-vector trainable context with a frozen CLIP text tower reaches pseudo mAP **0.5063** and diagnostic mAP **0.4888** at step 100, versus fixed-prompt J4 pseudo mAP **0.4970** and diagnostic mAP **0.4816**. This is promising but single-seed/short-run evidence only.",
        "",
        "## Artifacts",
        f"- Machine-readable report: `{rel(SUMMARY_JSON)}`",
        f"- Curves and compact plots: `{rel(ARTIFACT_DIR)}/`",
        "- Every ablation, probe, checkpoint, config, and metric path is recorded in the JSON report.",
        "",
        "## Required research conclusions",
    ]
    for heading, items in (
        ("Supported", summary["research_verdict"]["what_is_supported"]),
        ("Not supported", summary["research_verdict"]["what_is_not_supported"]),
    ):
        lines.append(f"### {heading}")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    SUMMARY_MD.write_text("\n".join(lines).rstrip() + "\n")


def write_plots(summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, key, color in (
        ("J4 + text", "selected_J4_pseudo_train", "tab:blue"),
        ("J3 no text", "no_text_J3_pseudo_train", "tab:orange"),
    ):
        probes = summary["long_training"][key]["probes"]
        steps = [probe["step"] for probe in probes]
        axes[0].plot(
            steps,
            [probe["pseudo_validation"]["mAP"] for probe in probes],
            "o-",
            label=label,
            color=color,
        )
        axes[1].plot(
            steps,
            [probe["q_effective_rank"] for probe in probes],
            "o-",
            label=label,
            color=color,
        )
        axes[2].plot(
            steps,
            [probe["semantic_margin"] for probe in probes],
            "o-",
            label=label,
            color=color,
        )
    for axis, title, ylabel in (
        (axes[0], "Valid pseudo-unseen retrieval", "mAP"),
        (axes[1], "Query geometry", "effective rank"),
        (axes[2], "Semantic separation", "centroid margin"),
    ):
        axis.set_xscale("symlog", linthresh=100)
        axis.set_xlabel("training step")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "jepa_validation_curves.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = list(summary["ablations_J0_J6"])
    maps = [
        summary["ablations_J0_J6"][name]["pseudo_validation"]["mAP"] for name in names
    ]
    ranks = [
        summary["ablations_J0_J6"][name]["geometry"]["q"].get(
            "effective_rank", float("nan")
        )
        for name in names
    ]
    axes[0].bar(names, maps, color="slateblue")
    axes[0].set_title("Ablation pseudo-validation mAP")
    axes[0].set_ylabel("mAP")
    axes[1].bar(names, ranks, color="seagreen")
    axes[1].set_title("Ablation query effective rank")
    axes[1].set_ylabel("rank")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "jepa_ablation_geometry.png", dpi=160)
    plt.close(figure)


def main() -> None:
    summary = build_summary()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary)
    write_plots(summary)
    print(f"wrote {SUMMARY_JSON}")
    print(f"wrote {SUMMARY_MD}")
    print(f"wrote {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
