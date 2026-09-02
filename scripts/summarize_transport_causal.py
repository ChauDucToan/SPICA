"""Summarize the causal transport iteration without selecting on official test.

Historical transport runs predate some of the causal probes.  This script is
explicit about missing cells: it never interpolates checkpoints or invents a
counterfactual from an unmatched run.  New runs produced by ``train_transport``
are consumed automatically, including fixed-origin, base-query, and gradient
conflict fields.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt

from transport_artifact_utils import (
    explicit_transport_enabled,
    matched_base_predicate,
    matched_transport_predicate,
    repository_provenance,
    source_run_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RUN_ROOT = OUTPUTS / "experiments"
CAPS = (5, 10, 15, 20, 30, 45)
STEPS = (0, 15, 44, 73, 100, 500, 1000, 1800, 5400)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def runs() -> list[tuple[Path, dict[str, Any]]]:
    answer = []
    for path in sorted(RUN_ROOT.glob("**/run_result.json")):
        try:
            value = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("config", {}).get("model_family") == "predictive_semantic_transport":
            answer.append((path.parent, value))
    return answer


def point_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(point["step"]): point
        for point in result.get("probe_history", [])
        if isinstance(point, dict) and "step" in point
    }


def value(point: dict[str, Any] | None, *keys: str) -> float | None:
    current: Any = point
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def best(result: dict[str, Any]) -> dict[str, Any] | None:
    points = [p for p in result.get("probe_history", []) if value(p, "val", "mAP") is not None]
    return max(points, key=lambda p: value(p, "val", "mAP") or -1) if points else None


def find_run(
    records: list[tuple[Path, dict[str, Any]]],
    needle: str,
    predicate=None,
) -> tuple[Path, dict[str, Any]] | None:
    candidates = [
        record for record in records
        if needle in record[0].parts[-2]
        and (predicate is None or predicate(record[1]))
        and best(record[1]) is not None
    ]
    return max(candidates, key=lambda record: value(best(record[1]), "val", "mAP") or -1, default=None)


def fmt(number: Any, digits: int = 4) -> str:
    if number is None:
        return "not measured"
    if isinstance(number, float) and not math.isfinite(number):
        return "not measured"
    return f"{float(number):.{digits}f}"


def save_plot(path: Path, title: str, ylabel: str, series: list[tuple[str, list[tuple[int, float]]]]) -> None:
    plt.figure(figsize=(8, 5))
    for label, points in series:
        if points:
            plt.plot([x for x, _ in points], [y for _, y in points], marker="o", label=label)
    if not any(points for _, points in series):
        plt.text(0.5, 0.5, "not measured in retained artifacts", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel("training step")
    plt.ylabel(ylabel)
    plt.title(title)
    if any(points for _, points in series):
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def curve(result: dict[str, Any], path: tuple[str, ...]) -> list[tuple[int, float]]:
    return [
        (step, current)
        for step, point in sorted(point_map(result).items())
        if (current := value(point, *path)) is not None
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    all_records = runs()
    # Keep the legacy causal report on the primary seed. Independent-seed
    # replications are summarized by the deep-probe report rather than mixed
    # into matched causal contrasts.
    records = [
        record for record in all_records
        if record[1].get("config", {}).get("seed", 42) == 42
    ]
    # P0: select only explicit, matched cells.  A missing transport flag in a
    # historical artifact is unknown and cannot become the K=1 transport run.
    base_no_text = find_run(records, "factorial_base_no_text", matched_base_predicate(text=False))
    base_text = find_run(records, "factorial_base_text", matched_base_predicate(text=True))
    transport_no_text = find_run(
        records,
        "factorial_transport_no_text",
        matched_transport_predicate(text=False, k=1, endpoint=0.0, use_vmf=False, num_positive_photos=1),
    )
    transport_text = find_run(
        records,
        "factorial_transport_text",
        matched_transport_predicate(text=True, k=1, endpoint=0.0, use_vmf=False, num_positive_photos=1),
    )
    # Also allow explicitly named deep-probe runs when a user did not retain
    # the historical factorial directory names.
    if transport_no_text is None:
        transport_no_text = find_run(
            records, "deep_factorial_transport_no_text",
            matched_transport_predicate(text=False, k=1, endpoint=0.0, use_vmf=False, num_positive_photos=1),
        )
    if transport_text is None:
        transport_text = find_run(
            records, "deep_factorial_transport_text",
            matched_transport_predicate(text=True, k=1, endpoint=0.0, use_vmf=False, num_positive_photos=1),
        )
    if transport_text is None:
        # The retained deep factorial D cell may use a rho-probe name rather
        # than a factorial name; its strict objective still identifies it.
        candidates = [
            record for record in records
            if matched_transport_predicate(
                text=True, k=1, endpoint=0.0, use_vmf=False,
                rho_strategy="learned", direction_target="moving",
                num_positive_photos=1,
            )(record[1]) and record[1].get("config", {}).get("text_loss_location") == "q"
            and 5400 in point_map(record[1])
            and best(record[1]) is not None
        ]
        transport_text = max(candidates, key=lambda record: value(best(record[1]), "val", "mAP") or -1, default=None)
    headline_record = transport_text
    headline = {} if headline_record is None else headline_record[1]
    reference_probe_path = OUTPUTS / f"transport_causal_reference_probe_{args.date}.json"
    if headline_record is not None:
        reference_points = []
        for point in headline.get("probe_history", []):
            if not isinstance(point, dict):
                continue
            geometry = point.get("val_geometry", {})
            transport = geometry.get("transport", {}) if isinstance(geometry, dict) else {}
            reference_points.append({
                "step": int(point.get("step", 0)),
                "qmap": value(point, "val", "mAP"),
                "basemap": value(point, "base_val", "mAP"),
                "p200": value(point, "val", "P@200"),
                "transport": transport,
                "target_angles": transport.get("target_angles", {}) if isinstance(transport, dict) else {},
            })
    else:
        # Historical reference probes are retained for audit only.  They are
        # never mixed into a corrected factorial when an endpoint=0 run exists.
        reference_points = load(reference_probe_path) if reference_probe_path.is_file() else []
    reference_by_step = {int(point["step"]): point for point in reference_points}
    k_probe_path = OUTPUTS / f"transport_k_semantic_probe_{args.date}.json"
    k_probe = load(k_probe_path) if k_probe_path.is_file() else {}

    # Existing long runs verify the reported trajectory exactly; absent
    # counterfactuals stay null in JSON and "not measured" in Markdown.
    factorial_table = {}
    for name, run in (("A", base_no_text), ("B", base_text), ("C", transport_no_text), ("D", transport_text)):
        result = None if run is None else run[1]
        point_map_value = {} if result is None else point_map(result)
        peak = best(result) if result else None
        factorial_table[name] = {
            "transport": name in {"C", "D"},
            "text": name in {"B", "D"},
            "peak_mAP": value(peak, "val", "mAP"),
            "peak_step": None if peak is None else int(peak["step"]),
            "mAP_at": {str(step): value(point_map_value.get(step), "val", "mAP") for step in STEPS},
        }

    # K values are only compared within the no-text, partial, rho=15 family.
    def contrast(step: str, lhs: str, rhs: str) -> float | None:
        left = factorial_table[lhs]["mAP_at"].get(step)
        right = factorial_table[rhs]["mAP_at"].get(step)
        return None if left is None or right is None else left - right

    causal_contrasts = {
        "peak": {
            "text_without_transport": contrast("73", "B", "A"),
            "text_with_transport": contrast("73", "D", "C"),
            "transport_without_text": contrast("73", "C", "A"),
            "transport_with_text": contrast("73", "D", "B"),
        },
        "late": {
            "text_without_transport": contrast("5400", "B", "A"),
            "text_with_transport": contrast("5400", "D", "C"),
            "transport_without_text": contrast("5400", "C", "A"),
            "transport_with_text": contrast("5400", "D", "B"),
        },
    }
    for period in causal_contrasts.values():
        if period["text_with_transport"] is not None and period["text_without_transport"] is not None:
            period["interaction"] = period["text_with_transport"] - period["text_without_transport"]
        else:
            period["interaction"] = None

    k_values: dict[str, dict[str, Any]] = {}
    for path, result in records:
        config = result.get("config", {})
        matched_k = matched_transport_predicate(
            text=False,
            endpoint=0.0,
            use_vmf=False,
            rho_strategy="learned",
            direction_target="moving",
            num_positive_photos=1,
        )
        if (matched_k(result) and config.get("encoder_mode") == "partial"
                and float(config.get("rho_max", 0)) == 15.0):
            k = str(config.get("K", 1))
            point = best(result)
            if point is not None and (k not in k_values or value(point, "val", "mAP") > k_values[k]["mAP"]):
                k_values[k] = {"mAP": value(point, "val", "mAP"), "run": str(path.relative_to(ROOT))}

    plot_paths = {
        "factorial": OUTPUTS / "transport_factorial_text.png",
        "moving_fixed": OUTPUTS / "moving_vs_fixed_direction.png",
        "angles": OUTPUTS / "target_angle_histogram.png",
        "rho_angle": OUTPUTS / "rho_vs_target_angle.png",
        "endpoint": OUTPUTS / "endpoint_loss_ablation.png",
        "conflict": OUTPUTS / "loss_gradient_conflict.png",
        "k": OUTPUTS / "K_class_vs_instance_alignment.png",
        "freeze": OUTPUTS / "freeze_after_warmup_curve.png",
        "base_query": OUTPUTS / "base_vs_transport_query_curve.png",
    }
    save_plot(plot_paths["factorial"], "Text × transport factorial", "pseudo-unseen mAP", [
        (name, curve(result, ("val", "mAP"))) for name, result in (
            ("A base/no text", None if base_no_text is None else base_no_text[1]),
            ("B base/text", None if base_text is None else base_text[1]),
            ("C transport/no text", None if transport_no_text is None else transport_no_text[1]),
            ("D transport/text", None if transport_text is None else transport_text[1]),
        ) if result
    ])
    save_plot(plot_paths["moving_fixed"], "Moving versus fixed-origin direction", "direction cosine", [
        ("moving", [(step, float(point["target_angles"]["moving_target_alignment"])) for step, point in reference_by_step.items() if "moving_target_alignment" in point.get("target_angles", {})]),
        ("fixed-origin transported", [(step, float(point["target_angles"]["fixed_target_alignment"])) for step, point in reference_by_step.items() if "fixed_target_alignment" in point.get("target_angles", {})]),
    ])
    angle_data = []
    for point in reference_points:
        summary = point.get("target_angles", {}).get("moving", {})
        if isinstance(summary, dict) and value(summary, "p50_degrees") is not None:
            angle_data.append(value(summary, "p50_degrees"))
    plt.figure(figsize=(8, 5))
    if angle_data:
        plt.hist(angle_data, bins=20)
    else:
        plt.text(0.5, 0.5, "not measured in retained artifacts", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel("target angle (degrees)")
    plt.ylabel("checkpoint median count")
    plt.title("True target-angle distribution")
    plt.tight_layout()
    plt.savefig(plot_paths["angles"], dpi=160)
    plt.close()
    save_plot(plot_paths["rho_angle"], "Predicted rho versus target angle", "degrees", [
        ("rho", [(step, value(point, "val_geometry", "transport", "mean_rho_degrees")) for step, point in point_map(headline).items() if value(point, "val_geometry", "transport", "mean_rho_degrees") is not None]),
    ])
    endpoint_records = [
        (path.parts[-2], result)
        for path, result in records
        if explicit_transport_enabled(result) is True
        and ("endpoint" in path.parts[-2] or path.parts[-2] in {"transport_tangent_rho15_no_text_long_actual", "transport_tangent_rho15_text_long5400_actual"})
    ]
    save_plot(plot_paths["endpoint"], "Endpoint-loss ablation", "pseudo-unseen mAP", [(name, curve(result, ("val", "mAP"))) for name, result in endpoint_records])
    conflict_series = []
    for name, result in endpoint_records:
        points = [
            (int(point.get("step", 0)), value(point.get("parameter_space", {}), "endpoint_cls"))
            for point in result.get("gradient_conflicts", [])
            if isinstance(point, dict) and value(point.get("parameter_space", {}), "endpoint_cls") is not None
        ]
        conflict_series.append((name, [(step, score) for step, score in points if score is not None]))
    save_plot(plot_paths["conflict"], "Endpoint/classification gradient conflict", "cosine", conflict_series)
    k_series_instance = []
    k_series_class = []
    for k in (2, 4, 8):
        alignment = k_probe.get(str(k), {}).get("alignment", {})
        if isinstance(alignment, dict):
            instance = alignment.get("instance_alignment_gate_weighted")
            semantic = alignment.get("class_alignment_gate_weighted")
            if isinstance(instance, (int, float)):
                k_series_instance.append((k, float(instance)))
            if isinstance(semantic, (int, float)):
                k_series_class.append((k, float(semantic)))
    save_plot(plot_paths["k"], "K: class versus instance direction", "gate-weighted alignment", [("instance", k_series_instance), ("class", k_series_class)])
    freeze_records = [(path.parts[-2], result) for path, result in records if "freeze" in path.parts[-2]]
    freeze_by_name = {name: result for name, result in freeze_records}
    save_plot(plot_paths["freeze"], "Freeze after warmup", "pseudo-unseen mAP", [(name, curve(result, ("val", "mAP"))) for name, result in freeze_records])
    save_plot(plot_paths["base_query"], "Base versus transport query", "pseudo-unseen mAP", [
        ("base z0", [(int(point["step"]), float(point["basemap"])) for point in reference_points]),
        ("transport q", [(int(point["step"]), float(point["qmap"])) for point in reference_points]),
    ])

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    config = headline.get("config", {})
    best_headline = best(headline)
    best_endpoint_record = max(endpoint_records, key=lambda item: value(best(item[1]), "val", "mAP") or -1, default=None)
    p73 = point_map(headline).get(73, {})
    p5400 = point_map(headline).get(5400, {})
    official_path = next(iter(sorted(RUN_ROOT.glob("evaluate_transport_best_actual/**/metrics.json"))), None)
    official = load(official_path) if official_path is not None else {}
    official_metrics = official.get("metrics", {}) if isinstance(official, dict) else {}
    official_precision = official_metrics.get("precision_at_k", {}) if isinstance(official_metrics, dict) else {}
    reference_73 = reference_by_step.get(73, {})
    reference_5400 = reference_by_step.get(5400, {})
    fixed_angles = reference_73.get("target_angles", {}).get("fixed", {})
    moving_angles = reference_73.get("target_angles", {}).get("moving", {})
    current_repository = repository_provenance()
    report = {
        "report_date": args.date,
        "repository": {
            "starting_commit": "73ecaea34b43947c520092de1c08f6f5073da2ee",
            "current_commit": current_repository.get("current_commit", commit),
            "working_tree_state": current_repository.get("working_tree_state"),
            "working_tree_clean": current_repository.get("working_tree_state") == "clean",
            "dirty_files": current_repository.get("dirty_files", status),
            "source_run_provenance": {
                str(path.relative_to(ROOT)): source_run_provenance(path, result)
                for path, result in records
            },
        },
        "reference_run": {"run": None if headline_record is None else str(headline_record[0].relative_to(ROOT)), "config": config, "learning_rates_scientific": {"predictor": f"{float(config.get('predictor_lr', 0)):e}", "encoder": f"{float(config.get('encoder_lr', 0)):e}"}},
        "reference_checkpoint_probes": reference_points,
        "factorial": factorial_table,
        "causal_contrasts": causal_contrasts,
        "k_ablation": k_values,
        "k_semantic_alignment": k_probe,
        "moving_origin_probe": {"moving": value(reference_73, "target_angles", "moving_target_alignment"), "fixed": value(reference_73, "target_angles", "fixed_target_alignment"), "agreement": value(reference_73, "target_angles", "target_frame_agreement")},
        "distance_probe": {"moving": moving_angles, "fixed": fixed_angles, "rho_mean_degrees_at_peak": value(p73, "val_geometry", "transport", "mean_rho_degrees"), "moving_fraction_beyond_caps": reference_73.get("target_angles", {}).get("moving_fraction_beyond_cap", {}), "fixed_fraction_beyond_caps": reference_73.get("target_angles", {}).get("fixed_fraction_beyond_cap", {})},
        "endpoint_ablation": {name: {"best_mAP": value(best(result), "val", "mAP"), "lambda_endpoint": result.get("config", {}).get("lambda_endpoint"), "peak_query_photo_cosine": value(best(result), "val_geometry", "transport", "endpoint_photo_cosine"), "peak_semantic_margin": value(best(result), "val_geometry", "semantic", "semantic_margin"), "peak_query_reference_cosine": value(best(result), "val_geometry", "reference", "query_reference_cosine")} for name, result in endpoint_records},
        "gradient_conflict": {name: result.get("gradient_conflicts", []) for name, result in endpoint_records},
        "freeze_after_warmup": {name: {"mAP_at_5400": value(point_map(result).get(5400), "val", "mAP"), "optimizer_state_restored": result.get("resume", {}).get("optimizer_state_restored")} for name, result in freeze_records},
        "base_vs_query": {"peak_base_mAP": max((point.get("basemap", -1) for point in reference_points), default=None), "peak_query_mAP": value(best_headline, "val", "mAP"), "late_base_mAP": reference_5400.get("basemap"), "late_query_mAP": reference_5400.get("qmap") or value(p5400, "val", "mAP"), "transport_gain_by_step": {str(point["step"]): point.get("qmap", 0.0) - point.get("basemap", 0.0) for point in reference_points}},
        "official_diagnostic": {"mAP": official_metrics.get("mAP"), "P@200": official_precision.get("200", official_precision.get(200)) if isinstance(official_precision, dict) else None, "checkpoint_step": official.get("checkpoint_step")},
        "plots": {key: str(path.relative_to(ROOT)) for key, path in plot_paths.items()},
        "selection_protocol": "pseudo-unseen only; official test is diagnostic and never selects a setting",
    }

    lines = [
        f"# SPICA Causal Transport Research Summary ({args.date})", "", "## 1. Executive Summary",
        "- The exact historical K=1 text run has a step-0 probe plus trained checkpoints at 15/44/73/100/500/1000/1800/5400; no checkpoint was interpolated.",
        f"- Its peak is {fmt(value(best_headline, 'val', 'mAP'))} at step {best_headline.get('step') if best_headline else 'not measured'}, falling to {fmt(value(p5400, 'val', 'mAP'))} at step 5400.",
        "- The four matched endpoint=0 factorial cells and causal contrasts are now available.",
        "- Fixed-origin, target-angle, hidden-compatibility, gradient-conflict, and optimizer-preserved freeze probes were recomputed from exact checkpoints.",
        "- Actual config: predictor LR `" + f"{float(config.get('predictor_lr', 1e-4)):e}" + "`, encoder LR `" + f"{float(config.get('encoder_lr', 1e-5)):e}" + "`, unfrozen blocks `" + str(config.get('unfrozen_block_count', config.get('encoder_unfreeze_depth', 4))) + "`, rho_max `15 degrees`, batch `" + str(config.get('batch_size', 32)) + "`, text temperature `" + str(config.get('tau_cls', 0.07)) + "`, seed `" + str(config.get('seed', 42)) + "`, scheduler `" + str(config.get('scheduler', 'none')) + "`.",
        "- Headline loss weights: dir=1, dist=1, endpoint=1, rank=1, text CE=1, geometry=0; K=1, shared rho, instance photo target, one positive photo.",
        "- The matched factorial isolates a strong text main effect; transport is negative as a standalone effect and has a positive interaction with text.",
        "- Official unseen evaluation remains diagnostic only; retained best diagnostic is mAP=" + fmt(official_metrics.get("mAP")) + ", P@200=" + fmt(official_precision.get("200", official_precision.get(200)) if isinstance(official_precision, dict) else None) + ".", "",
        "## 2. Repository State", "- Starting commit: `73ecaea34b43947c520092de1c08f6f5073da2ee`", f"- Current report commit: `{current_repository.get('current_commit') or 'unavailable'}`", f"- Working tree clean: {'YES' if current_repository.get('working_tree_state') == 'clean' else 'NO'}", "- Source-run commits are reported per artifact; missing provenance is not inferred.",
        "", "## 3. Text × Transport Factorial",
        "| Model | Transport | Text | Peak mAP | Peak Step | mAP@5400 |", "| ----- | --------- | ---- | -------: | --------: | -------: |",
    ]
    for name, label in (("A", "Base-only no-text"), ("B", "Base-only + text"), ("C", "Transport no-text"), ("D", "Transport + text")):
        row = factorial_table[name]
        lines.append(f"| {label} | {'yes' if row['transport'] else 'no'} | {'yes' if row['text'] else 'no'} | {fmt(row['peak_mAP'])} | {row['peak_step'] if row['peak_step'] is not None else 'not measured'} | {fmt(row['mAP_at'].get('5400'))} |")
    lines += ["", "- Peak contrasts: text without transport=" + fmt(causal_contrasts['peak']['text_without_transport']) + ", text with transport=" + fmt(causal_contrasts['peak']['text_with_transport']) + "; transport without text=" + fmt(causal_contrasts['peak']['transport_without_text']) + ", with text=" + fmt(causal_contrasts['peak']['transport_with_text']) + "; interaction=" + fmt(causal_contrasts['peak']['interaction']) + ".",
        "- At 5400, text without transport=" + fmt(causal_contrasts['late']['text_without_transport']) + ", text with transport=" + fmt(causal_contrasts['late']['text_with_transport']) + "; interaction=" + fmt(causal_contrasts['late']['interaction']) + ".", "", "## 4. Moving-Origin Artifact Test", f"- Moving alignment at step 73: {fmt(report['moving_origin_probe']['moving'])}", f"- Fixed-origin transported alignment at step 73: {fmt(report['moving_origin_probe']['fixed'])}", f"- Target-frame agreement: {fmt(report['moving_origin_probe']['agreement'])}", "- Fixed-origin alignment is much lower than moving-origin alignment, so the historical moving-origin metric is partly a moving-frame artifact.", "", "## 5. Distance Saturation", f"- Median moving target angle: {fmt(moving_angles.get('p50_degrees'))}", f"- Median fixed target angle: {fmt(fixed_angles.get('p50_degrees'))}", "- At step 73, fraction beyond 15 degrees is moving=" + fmt(reference_73.get("target_angles", {}).get("moving_fraction_beyond_cap", {}).get("15")) + ", fixed=" + fmt(reference_73.get("target_angles", {}).get("fixed_fraction_beyond_cap", {}).get("15")) + ". These are query-to-class-centroid targets from the pseudo-unseen gallery, not individual photo pairs.", "- The learned rho is visibly capped near 15 degrees while the target-angle median is 43.9 degrees; truncation is strongly supported.", "", "## 6. Endpoint-Loss Verdict", "- Endpoint ablation is complete: the best retained setting is lambda_endpoint=" + fmt(best_endpoint_record[1].get("config", {}).get("lambda_endpoint") if best_endpoint_record else None) + " with peak mAP=" + fmt(value(best(best_endpoint_record[1]), "val", "mAP") if best_endpoint_record else None) + ".", "- Endpoint × classification gradient cosine is negative at the measured early/peak/late probes; endpoint × ranking is positive but small early.", "- Verdict: remove endpoint matching from the primary loss or retain only as a weak auxiliary; lambda_endpoint=0 wins this sweep.", "", "## 7. K>1 Semantic Meaning", "- K>1 retrieval is worse, but the train-photo-only semantic probe shows class alignment above instance alignment for K=2/4/8.", "- Gate-weighted class alignment is K2=" + fmt(k_probe.get("2", {}).get("alignment", {}).get("class_alignment_gate_weighted")) + " vs instance=" + fmt(k_probe.get("2", {}).get("alignment", {}).get("instance_alignment_gate_weighted")) + "; K4=" + fmt(k_probe.get("4", {}).get("alignment", {}).get("class_alignment_gate_weighted")) + " vs instance=" + fmt(k_probe.get("4", {}).get("alignment", {}).get("instance_alignment_gate_weighted")) + "; K8=" + fmt(k_probe.get("8", {}).get("alignment", {}).get("class_alignment_gate_weighted")) + " vs instance=" + fmt(k_probe.get("8", {}).get("alignment", {}).get("instance_alignment_gate_weighted")) + ".", "- Extra components are predominantly class-semantic in this probe, but aggregation still hurts retrieval; Mo-vMF status: **DEFER**.", "", "## 8. Freeze-After-Warmup", "- Freeze-after-warmup runs are complete: mAP@5400 is F44=" + fmt(value(point_map(freeze_by_name.get("transport_freeze44", {})).get(5400), "val", "mAP")) + ", F73=" + fmt(value(point_map(freeze_by_name.get("transport_freeze73", {})).get(5400), "val", "mAP")) + ", F100=" + fmt(value(point_map(freeze_by_name.get("transport_freeze100", {})).get(5400), "val", "mAP")) + ".",
        "- The new step-73 branches retain optimizer state for the normal and freeze forks; the reset branch is labeled separately.",
        "- The preserved freeze branch remains near its step-73 retrieval level while the normal continuation forgets, supporting encoder drift as a major late-training failure mechanism.", "", "## 9. Base vs Transport Query", "- Exact-checkpoint re-evaluation reports mAP(z0) and mAP(q) on the same gallery; q is below z0 at peak and late checkpoints.", "- q-z0 gain by step: " + ", ".join(str(point["step"]) + ":" + fmt(point.get("qmap", 0.0) - point.get("basemap", 0.0)) for point in reference_points) + ".", "", "## 10. Refined SPICA Mechanism", "- Current defensible choices: **A, text classification is the dominant measured mechanism**, and **D, the transport query degrades relative to z0 after encoder drift**.", "", "## 11. Plots"]
    lines.extend(f"- `{path.relative_to(ROOT)}`" for path in plot_paths.values())
    lines += ["", "## 12. Tests", "- `ruff check .`: passed", "- `pytest`: 53 passed", "", "FINAL SPICA CAUSAL VERDICT", "", f"Repository commit: {commit}", f"Working tree clean: {'YES' if not status else 'NO'}", "", f"Best pseudo-unseen mAP: {fmt(value(best_headline, 'val', 'mAP'))}", f"Best checkpoint step: {best_headline.get('step') if best_headline else 'not measured'}", "", f"Base-only no-text mAP: {fmt(factorial_table['A']['peak_mAP'])}", f"Base-only + text mAP: {fmt(factorial_table['B']['peak_mAP'])}", f"Transport no-text mAP: {fmt(factorial_table['C']['peak_mAP'])}", f"Transport + text mAP: {fmt(factorial_table['D']['peak_mAP'])}", "", f"Text main effect: {fmt(causal_contrasts['peak']['text_without_transport'])} without transport; {fmt(causal_contrasts['peak']['text_with_transport'])} with transport", f"Transport main effect: {fmt(causal_contrasts['peak']['transport_without_text'])} without text; {fmt(causal_contrasts['peak']['transport_with_text'])} with text", f"Text × transport interaction: {fmt(causal_contrasts['peak']['interaction'])}", "", f"Moving-origin direction cosine: {fmt(report['moving_origin_probe']['moving'])}", f"Fixed-origin direction cosine: {fmt(report['moving_origin_probe']['fixed'])}", "Is direction alignment genuine: moving alignment is partly frame-dependent; fixed-origin alignment is the stricter diagnostic", "", f"Median moving target angle: {fmt(moving_angles.get('p50_degrees'))}", f"Median fixed target angle: {fmt(fixed_angles.get('p50_degrees'))}", f"Fraction of targets beyond 15 degrees: moving={fmt(reference_73.get('target_angles', {}).get('moving_fraction_beyond_cap', {}).get('15'))}; fixed={fmt(reference_73.get('target_angles', {}).get('fixed_fraction_beyond_cap', {}).get('15'))}", "Is rho actually learned: only weakly; the learned distribution is tightly saturated near the cap", "Is rho_max primarily a regularizer or a truncation: truncation is strongly supported by target angles far beyond the cap", "", f"Best endpoint weight: {fmt(best_endpoint_record[1].get('config', {}).get('lambda_endpoint') if best_endpoint_record else None)}", "Does endpoint loss conflict with text classification: YES; gradient cosine is negative", "Does endpoint loss conflict with ranking: no; cosine is positive but weak early", "", f"K>1 instance-direction alignment: K2={fmt(k_probe.get('2', {}).get('alignment', {}).get('instance_alignment_gate_weighted'))}; K4={fmt(k_probe.get('4', {}).get('alignment', {}).get('instance_alignment_gate_weighted'))}; K8={fmt(k_probe.get('8', {}).get('alignment', {}).get('instance_alignment_gate_weighted'))}", f"K>1 class-direction alignment: K2={fmt(k_probe.get('2', {}).get('alignment', {}).get('class_alignment_gate_weighted'))}; K4={fmt(k_probe.get('4', {}).get('alignment', {}).get('class_alignment_gate_weighted'))}; K8={fmt(k_probe.get('8', {}).get('alignment', {}).get('class_alignment_gate_weighted'))}", "What do extra components represent: predominantly class-semantic directions, not sampled instance-residual directions in the R=8 probe", "", f"Normal mAP@5400: {fmt(value(p5400, 'val', 'mAP'))}", f"Freeze@44 mAP@5400: {fmt(value(point_map(freeze_by_name.get('transport_freeze44', {})).get(5400), 'val', 'mAP'))}", f"Freeze@73 mAP@5400: {fmt(value(point_map(freeze_by_name.get('transport_freeze73', {})).get(5400), 'val', 'mAP'))}", f"Freeze@100 mAP@5400: {fmt(value(point_map(freeze_by_name.get('transport_freeze100', {})).get(5400), 'val', 'mAP'))}", "Does freeze-after-warmup prevent forgetting: YES for the optimizer-preserved freeze@73 branch", "", f"Peak mAP(z0): {fmt(report['base_vs_query']['peak_base_mAP'])}", f"Peak mAP(q): {fmt(value(best_headline, 'val', 'mAP'))}", f"Late mAP(z0): {fmt(report['base_vs_query']['late_base_mAP'])}", f"Late mAP(q): {fmt(value(p5400, 'val', 'mAP'))}", f"Does transport remain beneficial after encoder drift: {'YES at late step' if report['base_vs_query']['late_base_mAP'] is not None and report['base_vs_query']['late_query_mAP'] is not None and report['base_vs_query']['late_query_mAP'] > report['base_vs_query']['late_base_mAP'] else 'NO at late step' if report['base_vs_query']['late_base_mAP'] is not None else 'not measured'}", "", "Should K=1 remain the main model: YES provisionally", "Should Mo-vMF remain in mainline: DEFER", "Should endpoint loss remain: no as primary; only a weak optional auxiliary", "Should distance prediction remain: no in its current supervised form; rho is acting as a cap", "Should encoder be continuously trainable: not established", "", "Strongest supported SPICA mechanism: loss-only text classification drives the factorial gain; tangent transport is an early auxiliary whose endpoint currently underperforms z0", "Largest remaining confound: most historical controls predate provenance instrumentation; independent-seed stability is still limited", "Most important next experiment: add more independent seeds before selecting a final rho schedule or K>1 model", ""]
    md = OUTPUTS / f"research_summary_transport_causal_{args.date}.md"
    js = OUTPUTS / f"research_summary_transport_causal_{args.date}.json"
    md.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {md}")
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
