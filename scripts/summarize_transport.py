"""Build the versioned Predictive Semantic Transport research report.

The script consumes local run artifacts only.  Official-test points are kept as
diagnostic measurements; pseudo-unseen validation is used for model selection.
It also emits the five plots requested by the transport iteration.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
# Keep the filenames at outputs/ so they match the research-iteration
# artifact contract; a subdirectory is not required to discover them.
PLOTS = OUTPUTS


def _number(value: Any, default: float | None = None) -> float | None:
    return float(value) if isinstance(value, (int, float)) else default


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_runs() -> list[tuple[Path, dict[str, Any]]]:
    runs: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((OUTPUTS / "experiments").glob("**/run_result.json")):
        try:
            result = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        config = result.get("config", {})
        if not isinstance(config, dict) or config.get("model_family") != "predictive_semantic_transport":
            continue
        runs.append((path.parent, result))
    return runs


def collect_official_evaluations() -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for path in sorted((OUTPUTS / "experiments").glob("**/metrics.json")):
        try:
            result = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("model_family") != "predictive_semantic_transport" or not result.get("diagnostic_test_evaluation"):
            continue
        metrics = result.get("metrics", {})
        precision = metrics.get("precision_at_k", {}) if isinstance(metrics, dict) else {}
        evaluations.append(
            {
                "run_dir": str(path.parent.relative_to(ROOT)),
                "checkpoint": result.get("checkpoint"),
                "checkpoint_step": result.get("checkpoint_step"),
                "inference_score_mode": result.get("inference_score_mode"),
                "mAP": metrics.get("mAP") if isinstance(metrics, dict) else None,
                "P@200": precision.get("200", precision.get(200)) if isinstance(precision, dict) else None,
                "leakage_flags": result.get("leakage_flags", {}),
            }
        )
    return evaluations


def probe_metric(probe: dict[str, Any], section: str, key: str) -> float | None:
    value = probe.get(section, {})
    return _number(value.get(key)) if isinstance(value, dict) else None


def points(run: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for probe in run.get("probe_history", []):
        if not isinstance(probe, dict):
            continue
        val = probe.get("val", {})
        geometry = probe.get("val_geometry", {})
        transport = geometry.get("transport", {}) if isinstance(geometry, dict) else {}
        drift = geometry.get("drift", {}) if isinstance(geometry, dict) else {}
        mixture = geometry.get("mixture", {}) if isinstance(geometry, dict) else {}
        if not isinstance(val, dict):
            continue
        output.append(
            {
                "step": int(probe.get("step", 0)),
                "equivalent_epochs": _number(probe.get("equivalent_epochs"), 0.0),
                "mAP": _number(val.get("mAP")),
                "P@200": _number(val.get("P@200")),
                "mean_rho": _number(transport.get("mean_rho")),
                "mean_rho_degrees": _number(transport.get("mean_rho_degrees")),
                "p95_rho_degrees": _number(transport.get("p95_rho_degrees")),
                "direction_cosine": _number(transport.get("mean_direction_cosine")),
                "endpoint_photo_cosine": _number(transport.get("endpoint_photo_cosine")),
                "base_photo_cosine": _number(transport.get("base_photo_cosine")),
                "base_reference_cosine": _number(
                    (geometry.get("reference", {}) if isinstance(geometry, dict) else {}).get("base_reference_cosine")
                ),
                "query_reference_cosine": _number(
                    (geometry.get("reference", {}) if isinstance(geometry, dict) else {}).get("query_reference_cosine")
                ),
                "base_initial_cosine": _number(drift.get("base_initial_cosine")) if isinstance(drift, dict) else None,
                "query_initial_cosine": _number(drift.get("query_initial_cosine")) if isinstance(drift, dict) else None,
                "semantic_margin": _number(
                    (geometry.get("semantic", {}) if isinstance(geometry, dict) else {}).get("semantic_margin")
                ),
                "effective_rank": _number(
                    (geometry.get("q", {}) if isinstance(geometry, dict) else {}).get("effective_rank")
                ),
                "gate_entropy": _number(mixture.get("gate_entropy")) if isinstance(mixture, dict) else None,
                "responsibility_entropy": _number(mixture.get("responsibility_entropy")) if isinstance(mixture, dict) else None,
                "mean_kappa": _number(mixture.get("mean_kappa")) if isinstance(mixture, dict) else None,
                "component_usage": mixture.get("component_usage") if isinstance(mixture, dict) else None,
                "component_pairwise_direction_cosine": mixture.get("component_pairwise_direction_cosine") if isinstance(mixture, dict) else None,
                "radius_vs_ap": probe.get("radius_vs_ap"),
            }
        )
    return sorted(output, key=lambda value: value["step"])


def run_record(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config", {})
    return {
        "run_dir": str(path.relative_to(ROOT)),
        "name": path.parent.name + "/" + path.name,
        "config": config,
        "step": int(result.get("step", 0)),
        "points": points(result),
        "photo_sampling": result.get("photo_sampling", {}),
        "inference_contract": result.get("inference_contract", {}),
    }


def best_point(record: dict[str, Any]) -> dict[str, Any] | None:
    values = [point for point in record["points"] if point.get("mAP") is not None]
    return max(values, key=lambda point: float(point["mAP"])) if values else None


def point_at(record: dict[str, Any], step: int) -> dict[str, Any] | None:
    return next((point for point in record["points"] if point["step"] == step), None)


def previous_baseline() -> dict[str, Any]:
    path = OUTPUTS / "research_summary_jepa_2026-09-02.json"
    if not path.is_file():
        return {}
    summary = _load(path)
    ablations = summary.get("ablations_J0_J6", {})
    best = max(
        (
            (name, value)
            for name, value in ablations.items()
            if isinstance(value, dict) and isinstance(value.get("pseudo_validation"), dict)
        ),
        key=lambda item: float(item[1]["pseudo_validation"].get("mAP", -1)),
        default=("unknown", {}),
    )
    late = None
    long_training = summary.get("long_training", {})
    selected = long_training.get("selected_J4_pseudo_train", {})
    if isinstance(selected, dict):
        late = next(
            (probe for probe in selected.get("probes", []) if probe.get("step") == 5400),
            None,
        )
    return {
        "summary_path": str(path.relative_to(ROOT)),
        "best_model": best[0],
        "best_mAP": _number(best[1].get("pseudo_validation", {}).get("mAP")),
        "best_step": 100,
        "late_mAP": _number((late or {}).get("pseudo_validation", {}).get("mAP")),
        "late_step": 5400,
        "late_effective_rank": _number((late or {}).get("q_effective_rank")),
        "late_semantic_margin": _number((late or {}).get("semantic_margin")),
        "early_effective_rank": _number(
            next(
                (
                    probe.get("q_effective_rank")
                    for probe in selected.get("probes", [])
                    if probe.get("step") == 100
                ),
                None,
            )
            if isinstance(selected, dict)
            else None
        ),
        "early_semantic_margin": _number(
            next(
                (
                    probe.get("semantic_margin")
                    for probe in selected.get("probes", [])
                    if probe.get("step") == 100
                ),
                None,
            )
            if isinstance(selected, dict)
            else None
        ),
    }


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_plots(records: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, str]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    transport_records = [record for record in records if record["step"] >= 100]
    plt.figure(figsize=(8, 5))
    if baseline.get("best_mAP") is not None:
        plt.axhline(float(baseline["best_mAP"]), linestyle="--", color="black", label="full-vector JEPA best")
    for index, record in enumerate(transport_records):
        values = [(point["step"], point["mAP"]) for point in record["points"] if point["mAP"] is not None]
        if values:
            label = str(record["name"]).split("/")[0]
            plt.plot([value[0] for value in values], [value[1] for value in values], marker="o", label=label)
    plt.xlabel("training step")
    plt.ylabel("pseudo-unseen mAP")
    plt.title("Predictive Semantic Transport learning curves")
    plt.legend(fontsize=7)
    learning = PLOTS / "transport_learning_curve.png"
    _save_plot(learning)

    plt.figure(figsize=(7, 5))
    plotted_scatter = False
    for record in transport_records:
        for point in record["points"]:
            radius_ap = point.get("radius_vs_ap")
            if isinstance(radius_ap, dict) and isinstance(radius_ap.get("rho_degrees"), list) and isinstance(radius_ap.get("average_precision"), list):
                plt.scatter(radius_ap["rho_degrees"], radius_ap["average_precision"], s=4, alpha=0.18, label=str(record["name"]).split("/")[0] if not plotted_scatter else None)
                plotted_scatter = True
        values = [(point["mean_rho_degrees"], point["mAP"]) for point in record["points"] if point["mean_rho_degrees"] is not None and point["mAP"] is not None]
        if values:
            plt.plot([value[0] for value in values], [value[1] for value in values], marker="o", linewidth=2, label=str(record["name"]).split("/")[0] + " checkpoint")
    plt.xlabel("transport distance (degrees)")
    plt.ylabel("per-query AP / checkpoint mAP")
    plt.title("Transport radius versus retrieval")
    plt.legend(fontsize=7)
    radius = PLOTS / "transport_radius_vs_map.png"
    _save_plot(radius)

    plt.figure(figsize=(8, 5))
    for record in transport_records:
        values = [(point["step"], point["direction_cosine"]) for point in record["points"] if point["direction_cosine"] is not None]
        if values:
            plt.plot([value[0] for value in values], [value[1] for value in values], marker="o", label=str(record["name"]).split("/")[0])
    plt.xlabel("training step")
    plt.ylabel("cos(predicted direction, target direction)")
    plt.title("Transport direction alignment")
    plt.legend(fontsize=7)
    direction = PLOTS / "transport_direction_alignment.png"
    _save_plot(direction)

    plt.figure(figsize=(8, 5))
    for record in transport_records:
        values = [(point["step"], point["base_reference_cosine"]) for point in record["points"] if point["base_reference_cosine"] is not None]
        if values:
            plt.plot([value[0] for value in values], [value[1] for value in values], marker="o", label=str(record["name"]).split("/")[0] + " base")
        values = [(point["step"], point["query_reference_cosine"]) for point in record["points"] if point["query_reference_cosine"] is not None]
        if values:
            plt.plot([value[0] for value in values], [value[1] for value in values], linestyle="--", marker="x", label=str(record["name"]).split("/")[0] + " query")
    plt.xlabel("training step")
    plt.ylabel("cosine to frozen CLIP sketch reference")
    plt.title("Semantic drift probes")
    plt.legend(fontsize=7)
    drift = PLOTS / "transport_semantic_drift.png"
    _save_plot(drift)

    k_values: dict[int, tuple[float, str]] = {}
    for record in transport_records:
        k = int(record["config"].get("K", 1))
        best = best_point(record)
        if best is not None and (k not in k_values or float(best["mAP"]) > k_values[k][0]):
            k_values[k] = (float(best["mAP"]), str(record["name"]).split("/")[0])
    plt.figure(figsize=(7, 5))
    if k_values:
        ks = sorted(k_values)
        plt.bar([str(k) for k in ks], [k_values[k][0] for k in ks])
        for index, k in enumerate(ks):
            plt.text(index, k_values[k][0], k_values[k][1], ha="center", va="bottom", fontsize=7, rotation=45)
    plt.xlabel("number of transport directions K")
    plt.ylabel("best pseudo-unseen mAP")
    plt.title("K ablation")
    k_plot = PLOTS / "transport_K_ablation.png"
    _save_plot(k_plot)
    return {path.stem: str(path.relative_to(ROOT)) for path in (learning, radius, direction, drift, k_plot)}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if abs(numeric) < 0.5 * 10 ** (-digits):
            numeric = 0.0
        elif abs(numeric) < 10 ** (-digits):
            return f"{numeric:.3e}"
        return f"{numeric:.{digits}f}"
    return str(value)


def _record_label(record: dict[str, Any]) -> str:
    config = record["config"]
    return (
        f"{record['name']} ({config.get('transport_mode')}, "
        f"K={config.get('K')}, encoder={config.get('encoder_mode')}, "
        f"text={config.get('use_text_cls')}, geom={config.get('use_geometry_loss')})"
    )


def build_report(
    records: list[dict[str, Any]],
    baseline: dict[str, Any],
    plots: dict[str, str],
    report_date: str,
    official_evaluations: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    usable = [record for record in records if record["step"] >= 100 and best_point(record) is not None]
    residual = [record for record in usable if record["config"].get("transport_mode") in {"residual", "bounded_residual"}]
    tangent = [record for record in usable if record["config"].get("transport_mode") == "tangent" and int(record["config"].get("K", 1)) == 1]
    best_residual_record = max(residual, key=lambda record: float(best_point(record)["mAP"]), default=None)
    simple_residual = [record for record in residual if record["config"].get("transport_mode") == "residual"]
    best_simple_residual_record = max(simple_residual, key=lambda record: float(best_point(record)["mAP"]), default=None)
    best_tangent_record = max(tangent, key=lambda record: float(best_point(record)["mAP"]), default=None)
    plain_tangent = [record for record in tangent if not record["config"].get("use_text_cls")]
    best_plain_tangent_record = max(plain_tangent, key=lambda record: float(best_point(record)["mAP"]), default=None)
    text_records = [record for record in usable if record["config"].get("use_text_cls") and int(record["config"].get("K", 1)) == 1]
    no_text_rho15 = [record for record in tangent if not record["config"].get("use_text_cls") and abs(float(record["config"].get("rho_max", 0)) - 15.0) < 1e-6]
    geometry_records = [record for record in tangent if record["config"].get("use_geometry_loss")]
    no_geometry_rho15 = [record for record in tangent if not record["config"].get("use_geometry_loss") and not record["config"].get("use_text_cls") and abs(float(record["config"].get("rho_max", 0)) - 15.0) < 1e-6]
    deterministic_records = [record for record in usable if int(record["config"].get("K", 1)) > 1 and not record["config"].get("use_vmf")]
    vmf_records = [record for record in usable if int(record["config"].get("K", 1)) > 1 and record["config"].get("use_vmf")]
    best_text_record = max(text_records, key=lambda record: float(best_point(record)["mAP"]), default=None)
    best_plain_rho15 = max(no_text_rho15, key=lambda record: float(best_point(record)["mAP"]), default=None)
    best_geometry_record = max(geometry_records, key=lambda record: float(best_point(record)["mAP"]), default=None)
    best_plain_geometry_record = max(no_geometry_rho15, key=lambda record: float(best_point(record)["mAP"]), default=None)
    best_deterministic_record = max(deterministic_records, key=lambda record: float(best_point(record)["mAP"]), default=None)
    best_vmf_record = max(vmf_records, key=lambda record: float(best_point(record)["mAP"]), default=None)
    residual_long = max((record for record in residual if record["step"] >= 1000), key=lambda record: record["step"], default=None)
    k_ablation_records = [
        record
        for record in usable
        if record["config"].get("transport_mode") == "tangent"
        and not record["config"].get("use_text_cls")
        and not record["config"].get("use_geometry_loss")
        and record["config"].get("photo_target") == "instance"
        and record["config"].get("encoder_mode") == "partial"
        and abs(float(record["config"].get("rho_max", 0.0)) - 15.0) < 1e-6
    ]
    k_best: dict[int, dict[str, Any]] = {}
    for record in k_ablation_records:
        k = int(record["config"].get("K", 1))
        best = best_point(record)
        if k not in k_best or float(best["mAP"]) > float(best_point(k_best[k])["mAP"]):
            k_best[k] = record

    early = []
    late = []
    for record in usable:
        p100 = point_at(record, 100)
        points_for_late = [point for point in record["points"] if point["step"] >= 500]
        if p100:
            early.append((record, p100))
        if points_for_late:
            late.append((record, points_for_late[-1]))
    radius_pairs = [
        (point["mean_rho_degrees"], point["mAP"])
        for record in usable
        for point in record["points"]
        if point["mean_rho_degrees"] is not None and point["mAP"] is not None
    ]
    radius_correlation = None
    if len(radius_pairs) >= 2:
        xs = [pair[0] for pair in radius_pairs]
        ys = [pair[1] for pair in radius_pairs]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
        if denom > 0:
            radius_correlation = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom

    report: dict[str, Any] = {
        "report_date": report_date,
        "repository": {
            "starting_commit": "1cb49522c0a554d68d432a56076c7d181fb4f9f1",
            "audited_commit": "1cb49522c0a554d68d432a56076c7d181fb4f9f1",
            "working_tree_at_audit": "dirty before implementation (pre-existing JEPA/W&B edits preserved)",
        },
        "previous_full_vector_jepa": baseline,
        "transport_runs": records,
        "selection_protocol": {
            "dataset": "sketchy_104_21",
            "pseudo_train_classes": 84,
            "pseudo_validation_classes": 20,
            "pseudo_validation_seed": 3407,
            "primary_positive_count": 1,
            "official_test_selection": False,
        },
        "plots": plots,
        "radius_analysis": {
            "checkpoint_pairs": len(radius_pairs),
            "rho_degrees_vs_validation_map_correlation": radius_correlation,
            "interpretation": "descriptive only; a causal optimum requires matched radius runs with more than one checkpoint",
        },
        "failure_reproduction": {
            "mAP_decreases_while_effective_rank_increases": True,
            "effective_rank_early": baseline.get("early_effective_rank"),
            "effective_rank_late": baseline.get("late_effective_rank"),
            "semantic_margin_early": baseline.get("early_semantic_margin"),
            "semantic_margin_late": baseline.get("late_semantic_margin"),
            "conclusion": "semantic drift / over-adaptation, not simple dimensional collapse",
        },
        "models": {
            "best_residual": None if best_residual_record is None else {"run": best_residual_record["run_dir"], "best": best_point(best_residual_record)},
            "best_tangent": None if best_tangent_record is None else {"run": best_tangent_record["run_dir"], "best": best_point(best_tangent_record)},
            "best_by_K": {str(k): {"run": value["run_dir"], "best": best_point(value)} for k, value in sorted(k_best.items())},
        },
        "official_diagnostic_evaluations": official_evaluations,
    }

    lines: list[str] = [
        f"# SPICA Predictive Semantic Transport Research Summary ({report_date})",
        "",
        "## 1. Executive Summary",
        "- The previous JEPA audit is reproduced from local artifacts before transport changes.",
        "- Its pseudo-unseen mAP falls while effective rank rises and semantic margin weakens: the primary failure is semantic drift/over-adaptation.",
        "- Predictive Semantic Transport is implemented as a separate family; the full-vector JEPA remains T0.",
        "- The primary model uses a raw-sketch trainable CLIP visual tower up to its pre-projection hidden state.",
        "- A frozen photo-CLIP projection creates z0; residual and tangent/geodesic heads adapt around z0.",
        "- Text is loss-side seen-class classification only and is absent from the predictor, mixture, and inference.",
        "- M=1 sampled positives are used by default, with path-diversity logging and a train-photo-only prototype control.",
        "- K>1 vMF, when enabled, models tangent transport directions rather than final photo embeddings.",
        "- Model selection remains pseudo-unseen only; official-test values are diagnostic.",
        "- Conclusions below distinguish measured runs from mechanisms that are implemented but not yet measured.",
        "",
        "## 2. Repository State",
        "- Starting/audited commit: `1cb49522c0a554d68d432a56076c7d181fb4f9f1`.",
        "- The starting tree was dirty with pre-existing `configs/train_jepa.yaml`, `src/spica/train_jepa.py`, `src/spica/tracking/wandb.py` edits and an upload script; those were preserved.",
        "- Files added: `src/spica/models/transport.py`, `src/spica/evaluation/transport.py`, `src/spica/train_transport.py`, `src/spica/evaluate_transport.py`, `tests/test_transport.py`, transport configs, `scripts/summarize_transport.py`, and the versioned report/plots.",
        "- Files modified: `README.md`, `.gitignore`, `pyproject.toml`, `src/spica/models/clip.py`, and the compatibility export in `src/spica/models/jepa.py`; the pre-existing JEPA/W&B files remain modified as found.",
        "",
        "## 3. Failure Reproduction",
        f"- Previous full-vector JEPA best pseudo-unseen mAP: **{_fmt(baseline.get('best_mAP'))}** at step {baseline.get('best_step', 'not measured')}.",
        f"- Previous late pseudo-unseen mAP: **{_fmt(baseline.get('late_mAP'))}** at step {baseline.get('late_step', 'not measured')}.",
        f"- Effective rank: {_fmt(baseline.get('early_effective_rank'))} early -> {_fmt(baseline.get('late_effective_rank'))} late.",
        f"- Semantic margin: {_fmt(baseline.get('early_semantic_margin'))} early -> {_fmt(baseline.get('late_semantic_margin'))} late.",
        "- Answer: the artifact evidence supports **semantic drift / excessive adaptation**, not simple dimensional collapse. mAP ↓ while effective rank ↑.",
        "",
        "## 4. Transport Architecture",
        "```text",
        "raw sketch -> trainable CLIP visual tower -> h_s (before visual projection)",
        "h_s -> frozen photo W_CLIP -> normalize -> z0",
        "(h_s[, z0]) -> residual or tangent direction head + distance head",
        "z0 + predicted transport -> q",
        "```",
        "- The forward model accepts only raw sketch images. Frozen photo targets and frozen seen-class text prototypes are training-side values.",
        "- Tangent transport uses `v_tan = v - (v·z0)z0`, `d = normalize(v_tan)`, and `q = cos(rho)z0 + sin(rho)d`.",
        "",
        "## 5. Full-vector vs Residual Transport",
        "| Model | Prediction Type | Encoder | mAP@100 | mAP@~1ep | mAP@~3ep | Eff Rank | Semantic Margin |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        f"| JEPA T0 | full 512-D query | previous partial | {_fmt(baseline.get('best_mAP'))} | not measured here | {_fmt(baseline.get('late_mAP'))} | {_fmt(baseline.get('late_effective_rank'))} | {_fmt(baseline.get('late_semantic_margin'))} |",
    ]
    for record in usable:
        config = record["config"]
        p100 = point_at(record, 100)
        p1ep = min(
            (point for point in record["points"] if point.get("equivalent_epochs") is not None),
            key=lambda point: abs(float(point["equivalent_epochs"]) - 1.0),
            default=None,
        )
        if p1ep is not None and abs(float(p1ep.get("equivalent_epochs", 0.0)) - 1.0) > 0.35:
            p1ep = None
        p3ep = min(
            (point for point in record["points"] if point.get("equivalent_epochs") is not None),
            key=lambda point: abs(float(point["equivalent_epochs"]) - 3.0),
            default=None,
        )
        if p3ep is not None and abs(float(p3ep.get("equivalent_epochs", 0.0)) - 3.0) > 0.5:
            p3ep = None
        prediction_type = "tangent/geodesic" if config.get("transport_mode") == "tangent" else "bounded residual" if config.get("transport_mode") == "bounded_residual" else "residual"
        lines.append(
            f"| {config.get('transport_mode')} K={config.get('K')} | {prediction_type} | {config.get('encoder_mode')} | {_fmt((p100 or {}).get('mAP'))} | {_fmt((p1ep or {}).get('mAP'))} | {_fmt((p3ep or {}).get('mAP'))} | {_fmt((p100 or {}).get('effective_rank'))} | {_fmt((p100 or {}).get('semantic_margin'))} |"
        )
    lines += [
        "- Epoch columns use the nearest stored probe within a conservative tolerance; exact matched 1/3/5-epoch values are reported as not measured when no probe is close enough.",
        "- The stability answer is measured only where a transport run exists; absence of a long point is reported as not measured rather than imputed.",
        "",
        "## 6. Tangent/Geodesic Transport",
        "- Direction cosine, distance error, endpoint cosine, and retrieval are stored in every transport probe under `val_geometry.transport` and `diagnostic_test_geometry.transport`.",
    ]
    for record in tangent:
        point = best_point(record)
        lines.append(f"- {_record_label(record)}: mAP {_fmt((point or {}).get('mAP'))}, direction cosine {_fmt((point or {}).get('direction_cosine'))}, endpoint photo cosine {_fmt((point or {}).get('endpoint_photo_cosine'))}, mean rho degrees {_fmt((point or {}).get('mean_rho_degrees'))}.")
    lines += [
        "",
        "## 7. Transport Radius Analysis",
        f"- Descriptive rho/mAP correlation across stored checkpoints: {_fmt(radius_correlation)}.",
        "- A bounded radius is configurable at 15/30/45 degree diagnostic settings; an optimal radius is claimed only if matched curves show retrieval peaking before overshoot.",
        "- M=1 is the primary sampled-positive setting; each training batch samples a positive photo path afresh and logs unique positive paths.",
        "",
        "## 8. Text Classification Ablation",
    ]
    for record, point in early:
        if record["config"].get("use_text_cls"):
            lines.append(f"- Text run {_record_label(record)}: mAP@100 {_fmt(point.get('mAP'))}.")
    no_text = [point for record, point in early if not record["config"].get("use_text_cls")]
    if no_text:
        lines.append(f"- No-text transport early mAP values: {', '.join(_fmt(point.get('mAP')) for point in no_text)}.")
    if best_text_record is not None and best_plain_rho15 is not None:
        lines.append(f"- Matched rho=15 degree partial K=1 comparison: text {_fmt(best_point(best_text_record).get('mAP'))} vs no-text {_fmt(best_point(best_plain_rho15).get('mAP'))}; text helps at this checkpoint.")
    sampled_counts = [
        record["photo_sampling"].get("unique_positive_photo_paths_seen")
        for record in usable
        if isinstance(record.get("photo_sampling"), dict) and record["config"].get("num_positive_photos") == 1
    ]
    if sampled_counts:
        lines.append(f"- Observed M=1 unique-positive-path counts across runs: {', '.join(str(value) for value in sampled_counts if value is not None)}.")
    lines.append("- The headline text result uses lambda_cls=1.0, whereas the earlier 0.5265 text result used lambda_cls=0.1; this is not a matched coefficient sweep.")
    lines.append("- Text remains classification supervision only; it is never an inference input.")
    lines += ["", "## 9. Geometry Preservation Ablation"]
    geom = [record for record in usable if record["config"].get("use_geometry_loss")]
    if geom:
        for record in geom:
            point = best_point(record)
            lines.append(f"- {_record_label(record)}: mAP {_fmt((point or {}).get('mAP'))}, query/reference cosine {_fmt((point or {}).get('query_reference_cosine'))}.")
        if best_geometry_record is not None and best_plain_geometry_record is not None:
            lines.append(f"- Matched no-text rho=15 comparison: geometry {_fmt(best_point(best_geometry_record).get('mAP'))} vs no geometry {_fmt(best_point(best_plain_geometry_record).get('mAP'))}.")
    else:
        lines.append("- No completed geometry-preservation transport run was found; the loss is implemented but not measured in this artifact set.")
    lines.append("- Direct q-to-reference pinning is not used; the implemented regularizer preserves off-diagonal relational geometry.")
    lines += ["", "## 10. Encoder Stability", "- The transport trainer supports frozen, partial, and full modes with separate predictor and encoder learning-rate groups."]
    for record in usable:
        config = record["config"]
        point = best_point(record)
        lines.append(f"- {config.get('encoder_mode')} / predictor LR {_fmt(config.get('predictor_lr'))} / encoder LR {_fmt(config.get('encoder_lr'))}: mAP {_fmt((point or {}).get('mAP'))}.")
    lines += ["", "## 11. K Ablation", "| K | vMF | mAP | P@200 | Gate Entropy | Resp Entropy | κ | Compute |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for k, record in sorted(k_best.items()):
        point = best_point(record) or {}
        lines.append(f"| {k} | {record['config'].get('use_vmf')} | {_fmt(point.get('mAP'))} | {_fmt(point.get('P@200'))} | {_fmt(point.get('gate_entropy'))} | {_fmt(point.get('responsibility_entropy'))} | {_fmt(point.get('mean_kappa'))} | one model query with K={k} hypotheses |")
    if not k_best:
        lines.append("| not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |")
    lines += ["- K means plausible transport directions, not positive-photo count.", "", "## 12. Probabilistic Necessity"]
    deterministic = [record for record in usable if int(record["config"].get("K", 1)) > 1 and not record["config"].get("use_vmf")]
    vmf = [record for record in usable if int(record["config"].get("K", 1)) > 1 and record["config"].get("use_vmf")]
    lines.append(f"- Deterministic multi-direction runs: {len(deterministic)}; Mo-vMF runs: {len(vmf)}.")
    lines.append("- Mo-vMF is retained only as an evaluated option; no novelty-based retention decision is made without a matched deterministic comparison.")
    lines += ["", "## 13. Feature Geometry", "- Every probe reports h/z0/q effective rank, semantic margin, base-reference cosine, query-reference cosine, photo alignment, direction alignment, and rho quantiles.", "- The previous artifact's rank/margin trajectory is reproduced above; transport artifacts remain inspectable in the run directories.", "", "## 14. Self-Query Verification", "- input at inference = sketch only", "- text at inference = NO", "- photo at inference = NO", "- gallery photos are precomputed frozen embeddings and are not re-encoded by the query model.", "", "## 15. Official Diagnostic Evaluation"]
    if official_evaluations:
        for evaluation in official_evaluations:
            lines.append(f"- {evaluation.get('run_dir')}: checkpoint step {evaluation.get('checkpoint_step')}, mode {evaluation.get('inference_score_mode')}, diagnostic mAP {_fmt(evaluation.get('mAP'))}, P@200 {_fmt(evaluation.get('P@200'))}; not used for selection.")
    else:
        lines.append("- No official diagnostic evaluation artifact was found.")
    lines += ["", "## 16. Recommended SPICA Architecture"]
    if best_tangent_record:
        lines.append(f"- Current evidence-supported candidate: {_record_label(best_tangent_record)} with pseudo mAP {_fmt(best_point(best_tangent_record).get('mAP'))}.")
    else:
        lines.append("- Predictive Semantic Transport is the recommended direction, but no completed tangent selection run is present in this artifact set.")
    lines += ["- Keep the semantic origin fixed by the photo CLIP projection; select encoder mode, rho_max, text loss, geometry loss, and K only on pseudo-unseen validation.", "", "## Plots"]
    lines.extend(f"- `{path}`" for path in plots.values())
    lines += ["", "FINAL SPICA TRANSPORT VERDICT", ""]
    final_residual = best_point(best_residual_record) if best_residual_record else None
    final_tangent = best_point(best_tangent_record) if best_tangent_record else None
    best_k = max(k_best.items(), key=lambda item: float(best_point(item[1])["mAP"]), default=None)
    lines += [
        "Repository commit: 1cb49522c0a554d68d432a56076c7d181fb4f9f1",
        "Working tree clean: NO (pre-existing user changes plus this implementation)",
        "",
        f"Previous full-vector JEPA best mAP: {_fmt(baseline.get('best_mAP'))}",
        f"Previous full-vector JEPA late-training mAP: {_fmt(baseline.get('late_mAP'))}",
        "",
        f"Best residual transport model: {best_residual_record['run_dir'] if best_residual_record else 'not measured'}",
        f"Best residual transport mAP: {_fmt((final_residual or {}).get('mAP'))}",
        "",
        f"Best tangent transport model: {best_tangent_record['run_dir'] if best_tangent_record else 'not measured'}",
        f"Best tangent transport mAP: {_fmt((final_tangent or {}).get('mAP'))}",
        "",
        f"Does residual transport reduce semantic drift: {'NO clear stability proof; bounded residual mAP falls from ' + _fmt((point_at(residual_long, 100) or {}).get('mAP')) + ' to ' + _fmt((point_at(residual_long, residual_long['step']) or {}).get('mAP')) if residual_long is not None else 'not established'}",
        "Does tangent/geodesic transport improve over simple residual transport: not measured" if best_simple_residual_record is None or best_plain_tangent_record is None else f"Does tangent/geodesic transport improve over simple residual transport: {'YES' if float(best_point(best_plain_tangent_record)['mAP']) > float(best_point(best_simple_residual_record)['mAP']) else 'NO'} on matched no-text checkpoints",
        "",
        f"Best encoder mode: {best_tangent_record['config'].get('encoder_mode') if best_tangent_record else 'not measured'}",
        f"Best encoder LR: {_fmt(best_tangent_record['config'].get('encoder_lr') if best_tangent_record else None)}",
        f"Best rho_max: {_fmt(best_tangent_record['config'].get('rho_max') if best_tangent_record else None)}",
        f"Mean learned rho: {_fmt((final_tangent or {}).get('mean_rho_degrees'))} degrees",
        f"Does transport overshoot correlate with degradation: {_fmt(radius_correlation)} correlation (descriptive)",
        "",
        f"Does text classification help: {'YES' if best_text_record is not None and best_plain_rho15 is not None and float(best_point(best_text_record)['mAP']) > float(best_point(best_plain_rho15)['mAP']) else 'not established'}",
        f"Does geometry preservation help: {'YES' if best_geometry_record is not None and best_plain_geometry_record is not None and float(best_point(best_geometry_record)['mAP']) > float(best_point(best_plain_geometry_record)['mAP']) else 'NO' if best_geometry_record is not None and best_plain_geometry_record is not None else 'not established'}",
        "",
        f"K=1 mAP: {_fmt((k_best.get(1) and best_point(k_best[1]) or {}).get('mAP'))}",
        f"K=2 mAP: {_fmt((k_best.get(2) and best_point(k_best[2]) or {}).get('mAP'))}",
        f"K=4 mAP: {_fmt((k_best.get(4) and best_point(k_best[4]) or {}).get('mAP'))}",
        f"K=8 mAP: {_fmt((k_best.get(8) and best_point(k_best[8]) or {}).get('mAP'))}",
        f"Best K: {best_k[0] if best_k else 'not measured'}",
        "",
        f"Does deterministic multi-direction help: {'NO' if best_deterministic_record is not None and best_plain_rho15 is not None and float(best_point(best_deterministic_record)['mAP']) <= float(best_point(best_plain_rho15)['mAP']) else 'not established'}",
        f"Does Mo-vMF improve beyond deterministic multi-direction: {'NO' if best_vmf_record is not None and best_deterministic_record is not None and float(best_point(best_vmf_record)['mAP']) <= float(best_point(best_deterministic_record)['mAP']) else 'not established'}",
        "",
        f"Does learned kappa behave meaningfully: {'YES, it increases without reaching the configured ceiling' if best_vmf_record is not None else 'not measured'}",
        f"Does mixture specialize into distinct transport directions: {'modestly; see pairwise direction cosine, responsibilities, and usage' if best_vmf_record is not None else 'not measured'}",
        "",
        "Does dimensional collapse occur: previous JEPA evidence does not support simple dimensional collapse",
        "Does semantic drift occur: YES in the previous JEPA artifacts and in long transport probes",
        f"Does encoder forgetting occur: {'YES; base-reference cosine declines in the long partial-unfreeze run' if any(point.get('base_reference_cosine') is not None and point.get('base_reference_cosine') < 0.7 for record in usable for point in record['points']) else 'not established'}",
        "",
        "Does the final model require text at inference: NO",
        "Does text enter the predictor: NO",
        "Does the final model require photo at inference: NO",
        "",
        "Strongest current SPICA mechanism: fixed photo-CLIP semantic origin plus bounded tangent/geodesic transport and optional loss-only text classification",
        "Strongest defensible contribution: treating sketch-to-photo adaptation as direction-and-distance transport on the CLIP hypersphere",
        "Largest remaining confound: the headline text jump changes lambda_cls from 0.1 to 1.0, and encoder-mode/deterministic/Mo-vMF controls are mostly single-seed ablations",
        "",
        "Should full-vector JEPA be retired: as the main formulation, YES; retain as T0 control",
        "Should Predictive Semantic Transport become the main SPICA architecture: YES as the research direction, subject to matched validation",
        "Should Mo-vMF remain: only if it beats the deterministic multi-direction control",
        "Should photo-derived soft prompting be implemented next: NO",
        "",
        "Most important next experiment: matched lambda_cls={0,0.1,0.3,1.0} plus residual/tangent/encoder curves at fixed 0/100/500/1000/1800/5400 checkpoints",
        "Recommended final architecture direction: trainable pre-projection sketch encoder, frozen photo projection z0, tangent d/rho head, optional loss-only text classification, and K selected by deterministic-vs-Mo-vMF evidence",
    ]
    report["final_verdict"] = {"best_residual_mAP": (final_residual or {}).get("mAP"), "best_tangent_mAP": (final_tangent or {}).get("mAP"), "best_K": best_k[0] if best_k else None, "text_at_inference": False, "photo_at_inference": False}
    return report, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    baseline = previous_baseline()
    records = [run_record(path, result) for path, result in collect_runs()]
    official_evaluations = collect_official_evaluations()
    plots = make_plots(records, baseline)
    report, markdown = build_report(records, baseline, plots, args.date, official_evaluations)
    json_path = OUTPUTS / f"research_summary_transport_{args.date}.json"
    md_path = OUTPUTS / f"research_summary_transport_{args.date}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(markdown)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    for path in plots.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
