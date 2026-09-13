"""CPU-only official dataset geometry and category-count statistics.

This module is deliberately inert until its CLI is called.  It reads the three
validated official protocols, never trains or contacts W&B, and refuses to
reuse an output directory.  A failed decode leaves a partial CSV and a
FAILED_PARTIAL receipt; only a complete scan writes the authoritative thesis
figures/tables.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

# Keep BLAS/OpenCV from multiplying threads inside the process pool.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
THESIS_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 224
PIXELS = IMAGE_SIZE * IMAGE_SIZE
INK_THRESHOLD = 0.9
INK_SUM_THRESHOLD = INK_THRESHOLD * 3 * 255  # 688.5; no integer pixel sum equals it.
CONFIGS = {
    "sketchy_104_21": "configs/data/sketchy_104_21.yaml",
    "tuberlin_220_30": "configs/data/tuberlin_220_30.yaml",
    "quickdraw_80_30": "configs/data/quickdraw_80_30.yaml",
}
TASK = tuple[str, str, str, int, str, str]

# Set by each pool worker.  Keeping the transform construction in the worker
# avoids pickling torchvision transform objects.
_GEOMETRY: list[Any] | None = None
_FULL_TRANSFORM: Any | None = None
_THRESHOLD_VERIFIED = False


def _configure_cv2() -> None:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)


def _worker_init() -> None:
    global _GEOMETRY, _FULL_TRANSFORM, _THRESHOLD_VERIFIED
    _configure_cv2()
    from spica.data.transforms import build_clip_eval_transform

    full = build_clip_eval_transform(IMAGE_SIZE)
    _GEOMETRY = list(full.transforms[:2])
    _FULL_TRANSFORM = full
    _THRESHOLD_VERIFIED = False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _threshold_mask(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB {IMAGE_SIZE}x{IMAGE_SIZE}, got {rgb.shape} {rgb.dtype}")
    return rgb.astype(np.uint16).sum(axis=2) < INK_SUM_THRESHOLD


def _verify_against_normalized_transform(rgb_image: Image.Image, rgb: np.ndarray) -> None:
    """Check the integer threshold against the actual production tensor path."""
    if _FULL_TRANSFORM is None:
        raise RuntimeError("worker transform was not initialized")
    from spica.data.transforms import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD

    normalized = np.asarray(_FULL_TRANSFORM(rgb_image), dtype=np.float64)
    mean = np.asarray(CLIP_IMAGE_MEAN, dtype=np.float64)[:, None, None]
    std = np.asarray(CLIP_IMAGE_STD, dtype=np.float64)[:, None, None]
    denormalized = normalized * std + mean
    actual_mask = denormalized.mean(axis=0) < INK_THRESHOLD
    if not np.array_equal(_threshold_mask(rgb), actual_mask):
        raise AssertionError("uint8 geometry threshold differs from denormalized production transform")


def _ridge_width(mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    # Zero padding makes an all-black/fully foreground raster finite at its
    # boundary instead of allowing an unbounded distance interpretation.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    distances = cv2.distanceTransform(padded * 255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    kernel = np.ones((3, 3), dtype=np.uint8)
    ridge = mask & (distances >= cv2.dilate(distances, kernel))
    values = distances[ridge]
    return float(np.mean(values * 2.0)) if values.size else math.nan


def _image_metrics(rgb: np.ndarray) -> tuple[int, float, float]:
    mask = _threshold_mask(rgb)
    ink_pixels = int(mask.sum())
    return ink_pixels, float(ink_pixels * 100.0 / PIXELS), _ridge_width(mask)


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _process_one(task: TASK) -> dict[str, Any]:
    global _THRESHOLD_VERIFIED
    if _GEOMETRY is None or _FULL_TRANSFORM is None:
        _worker_init()
    dataset, split, modality, label, class_name, raw_path = task
    path = Path(raw_path)
    raw = path.read_bytes()
    raw_sha256 = _sha256_bytes(raw)
    try:
        with Image.open(BytesIO(raw)) as decoded:
            original_width, original_height = decoded.size
            decoded.load()
            rgb_image = decoded.convert("RGB")
        resized = _GEOMETRY[1](_GEOMETRY[0](rgb_image))  # type: ignore[index]
        rgb = np.array(resized, dtype=np.uint8, copy=True)
        if not _THRESHOLD_VERIFIED:
            _verify_against_normalized_transform(rgb_image, rgb)
            _THRESHOLD_VERIFIED = True
        ink_pixels, ink_fraction_percent, width = _image_metrics(rgb)
    except Exception as error:
        raise RuntimeError(f"decode/measure failed for {path}: {error}") from error
    return {
        "dataset": dataset,
        "split": split,
        "modality": modality,
        "label": label,
        "class_name": class_name,
        "path": _relative_path(path),
        "original_width": original_width,
        "original_height": original_height,
        "ink_pixels": ink_pixels,
        "ink_fraction_percent": ink_fraction_percent,
        "raster_line_width_estimate_px": width,
        "raw_file_sha256": raw_sha256,
        "resized_rgb_sha256": _sha256_bytes(rgb.tobytes(order="C")),
    }


def _process_chunk(chunk: tuple[TASK, ...]) -> list[dict[str, Any]]:
    return [_process_one(task) for task in chunk]


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _class_count_rows(protocols: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, protocol in protocols.items():
        for split_name in ("train", "test"):
            split = getattr(protocol, split_name)
            for modality in ("sketch", "photo"):
                entries = getattr(split, f"{modality}_entries")
                counts = Counter(entry.label for entry in entries)
                if set(counts) != set(split.class_names):
                    raise ValueError(f"{dataset} {split_name} {modality} class coverage mismatch")
                for label in sorted(split.class_names):
                    rows.append({
                        "dataset": dataset,
                        "split": split_name,
                        "modality": modality,
                        "label": label,
                        "class_name": split.class_names[label],
                        "count": counts[label],
                    })
    return rows


def _selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in CONFIGS:
        for split in ("train", "test"):
            for modality in ("sketch", "photo"):
                group = [row for row in rows if (row["dataset"], row["split"], row["modality"]) == (dataset, split, modality)]
                largest = sorted(group, key=lambda row: (-row["count"], row["class_name"]))[:10]
                largest_ids = {row["label"] for row in largest}
                smallest = [row for row in sorted(group, key=lambda row: (row["count"], row["class_name"])) if row["label"] not in largest_ids][:5]
                by_label: dict[int, dict[str, Any]] = {}
                for row in largest:
                    by_label[row["label"]] = {**row, "selection": "largest"}
                for row in smallest:
                    if row["label"] in by_label:
                        by_label[row["label"]]["selection"] = "largest+smallest"
                    else:
                        by_label[row["label"]] = {**row, "selection": "smallest"}
                ordered = sorted(by_label.values(), key=lambda row: (-row["count"], row["class_name"]))
                for rank, row in enumerate(ordered, start=1):
                    selected.append({**row, "rank": rank})
    return selected


def _write_count_figures(selected: list[dict[str, Any]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for dataset in CONFIGS:
        for split in ("train", "test"):
            for modality in ("sketch", "photo"):
                rows = [row for row in selected if row["dataset"] == dataset and row["split"] == split and row["modality"] == modality]
                rows.sort(key=lambda row: (-row["count"], row["class_name"]))
                fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(rows)), 5.0))
                bars = ax.bar(np.arange(len(rows)), [row["count"] for row in rows], color=["#4477aa" if row["selection"] == "largest" else "#ee7733" for row in rows])
                ax.set_yscale("log")
                ax.set_ylim(1, max(row["count"] for row in rows) * 3)
                ax.bar_label(bars, labels=[str(row["count"]) for row in rows], padding=3, fontsize=8)
                ax.set_xticks(np.arange(len(rows)), [row["class_name"] for row in rows], rotation=65, ha="right")
                ax.set_ylabel("Images (count, logarithmic y-axis)")
                ax.set_title(f"{dataset} — {split} — {modality} category counts")
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                fig.savefig(destination / f"{dataset}_{split}_{modality}_counts.png", dpi=180)
                fig.savefig(destination / f"{dataset}_{split}_{modality}_counts.svg")
                plt.close(fig)


def _summary_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "population_std": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "population_std": float(np.std(array, ddof=0)),
    }


def _summarize(metric_values: dict[tuple[str, str, str], dict[str, list[float]]], counts: dict[tuple[str, str, str], dict[str, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(metric_values):
        dataset, split, modality = key
        values = metric_values[key]
        result["|".join(key)] = {
            "dataset": dataset,
            "split": split,
            "modality": modality,
            "image_count": counts[key]["image_count"],
            "blank_count": counts[key]["blank_count"],
            "metrics": {name: _summary_stats(values[name]) for name in ("ink_pixels", "ink_fraction_percent", "raster_line_width_estimate_px")},
        }
    return result


def _markdown_tables(summary: dict[str, Any]) -> str:
    blocks: list[str] = ["# Thống kê sketch ở224×224", "", "Ba bảng tương ứng ba dataset; mỗi thống kê tính giữa các ảnh, không phải trung bình category. SD là độ lệch chuẩn tổng thể (ddof=0). Ảnh trắng được tính cho mật độ, chỉ loại khỏi thống kê độ dày.", "", "**Độ dày là proxy đường kính EDT tại các điểm cực đại cục bộ, không phải bề rộng nét chính xác. Ví dụ nét raster1px cho proxy2px; junction và lưới pixel gây sai lệch.**", ""]
    specs = (("Số pixel mực (px)", "ink_pixels"), ("Mật độ mực (%)", "ink_fraction_percent"), ("Độ dày proxy EDT (px)", "raster_line_width_estimate_px"))
    for dataset in CONFIGS:
        blocks += [f"## {dataset}", "", "| Split | Đại lượng | Số ảnh hợp lệ | Trung bình | Trung vị | Độ lệch chuẩn |", "|---|---|---:|---:|---:|---:|"]
        for split in ("train", "test"):
            record = summary[f"{dataset}|{split}|sketch"]
            for title, metric in specs:
                stats = record["metrics"][metric]
                values = ["—" if stats[k] is None else f"{stats[k]:.6f}" for k in ("mean", "median", "population_std")]
                blocks.append(f"| {split} | {title} | {stats['count']} | " + " | ".join(values) + " |")
        blocks.append("")
        blocks.append("Ảnh trắng: " + "; ".join(f"{s}={summary[f'{dataset}|{s}|sketch']['blank_count']}" for s in ("train", "test")))
        blocks.append("")
    return "\n".join(blocks)


def _write_authoritative_report(summary: dict[str, Any], selected: list[dict[str, Any]], destination: Path, raw_output: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite tracked report directory: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        tables = stage / "tables"
        figures = stage / "figures"
        tables.mkdir()
        (tables / "dataset_statistics.md").write_text(_markdown_tables(summary) + "\n", encoding="utf-8")
        for filename in ("class_counts_all.csv", "class_counts_selected.csv", "summary.json", "summary.csv"):
            shutil.copy2(raw_output / filename, tables / filename)
        _write_count_figures(selected, figures)
        lines = [
            "# Official dataset statistics",
            "",
            "Generated by `docs/thesis/dataset_stats.py` after a complete CPU scan.",
            "",
            "- Summary tables: [tables/dataset_statistics.md](tables/dataset_statistics.md)",
            "- Twelve count charts: `figures/` (dataset × train/test × sketch/photo).",
            f"- Raw per-image rows, hashes and receipt: `{raw_output.relative_to(ROOT)}`.",
            "- CSVs and summary JSON: `tables/`; all12 charts have PNG and SVG versions.",
            "- Blue: top10; orange: bottom5; equal-count ties use class name. Bottom5 excludes already selected top10, giving15 unique categories.",
            "- Method: [../dataset_statistics_method.md](../dataset_statistics_method.md)",
            "",
            "This report contains no training, model, W&B, or network results.",
        ]
        (stage / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _self_check() -> None:
    _configure_cv2()
    _worker_init()
    white = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    white[0, 0] = 0
    image = Image.fromarray(white, mode="RGB")
    _verify_against_normalized_transform(image, white)

    thin = np.zeros((32, 32), dtype=bool)
    thin[15, 4:28] = True
    three = np.zeros((32, 32), dtype=bool)
    three[14:17, 4:28] = True
    assert _ridge_width(thin) == 2.0, _ridge_width(thin)
    assert _ridge_width(three) == 4.0, _ridge_width(three)
    assert math.isnan(_ridge_width(np.zeros((32, 32), dtype=bool)))
    assert _summary_stats([])["count"] == 0
    assert math.isfinite(_ridge_width(np.ones((32, 32), dtype=bool)))
    circle = np.zeros((32, 32), dtype=np.uint8)
    cv2.circle(circle, (16, 16), 8, 1, -1)
    assert math.isfinite(_ridge_width(circle.astype(bool)))
    rows = [{"dataset": d, "split": s, "modality": m, "label": i, "class_name": f"class{i:02d}", "count": 70}
            for d in CONFIGS for s in ("train", "test") for m in ("sketch", "photo") for i in range(21)]
    selected = _selected_rows(rows)
    assert len(selected) == 12 * 15
    assert len({r["label"] for r in selected[:15]}) == 15
    print("PASS: threshold denormalization and EDT-ridge synthetic checks")


def _guard_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    outputs = (ROOT / "outputs").resolve()
    try:
        path.relative_to(outputs)
    except ValueError as error:
        raise ValueError(f"--output must be a fresh directory under {outputs}") from error
    if path == outputs or path.exists():
        raise FileExistsError(f"--output must be fresh and must not overwrite: {path}")
    path.mkdir(parents=True)
    return path


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    (path / "receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(output: Path, workers: int, chunk_size: int) -> None:
    output = _guard_output(output)
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "script_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "workers": workers,
        "chunk_size": chunk_size,
        "transform": "spica.data.transforms.build_clip_eval_transform(224).transforms[:2]",
        "threshold": {"definition": "mean RGB / 255 < 0.9", "integer_sum_cutoff": INK_SUM_THRESHOLD},
        "datasets": {},
    }
    _write_receipt(output, receipt)
    per_image_fields = ["dataset", "split", "modality", "label", "class_name", "path", "original_width", "original_height", "ink_pixels", "ink_fraction_percent", "raster_line_width_estimate_px", "raw_file_sha256", "resized_rgb_sha256"]
    per_image_path = output / "per_image_statistics.csv"
    try:
        _configure_cv2()
        from spica.data.coupled_benchmark import load_benchmark_protocol

        protocols = {name: load_benchmark_protocol(config) for name, config in CONFIGS.items()}
        class_rows = _class_count_rows(protocols)
        selected = _selected_rows(class_rows)
        _write_csv(output / "class_counts_all.csv", ["dataset", "split", "modality", "label", "class_name", "count"], class_rows)
        _write_csv(output / "class_counts_selected.csv", ["dataset", "split", "modality", "rank", "label", "class_name", "count", "selection"], selected)
        tasks: list[TASK] = []
        expected_by_group: dict[tuple[str, str, str], int] = {}
        for dataset, protocol in protocols.items():
            receipt["datasets"][dataset] = {
                "config": str(protocol.config_path.relative_to(ROOT)),
                "config_sha256": protocol.config_sha256,
                "protocol_identity_sha256": protocol.identity["sha256"],
                "counts": {},
            }
            for split_name in ("train", "test"):
                split = getattr(protocol, split_name)
                for modality in ("sketch", "photo"):
                    entries = getattr(split, f"{modality}_entries")
                    key = (dataset, split_name, modality)
                    expected_by_group[key] = len(entries)
                    receipt["datasets"][dataset]["counts"]["|".join((split_name, modality))] = len(entries)
                    if modality == "sketch":
                        tasks.extend((dataset, split_name, modality, entry.label, split.class_names[entry.label], str(entry.path)) for entry in entries)
                    else:
                        del expected_by_group[key]  # Photos are counted, never measured as line drawings.
        receipt["expected_image_rows"] = len(tasks)
        receipt["protocol_scope"] = "all official train and test manifest entries; manifest metadata/path validation delegated to load_benchmark_protocol"
        _write_receipt(output, receipt)

        metric_values = {
            key: {"ink_pixels": [], "ink_fraction_percent": [], "raster_line_width_estimate_px": []}
            for key in expected_by_group
        }
        group_counts = {key: {"image_count": 0, "blank_count": 0} for key in expected_by_group}
        written = 0
        with per_image_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_image_fields)
            writer.writeheader()
            chunks = [tuple(tasks[start:start + chunk_size]) for start in range(0, len(tasks), chunk_size)]
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
                for chunk_rows in pool.map(_process_chunk, chunks, chunksize=1):
                    for row in chunk_rows:
                        writer.writerow(row)
                        key = (row["dataset"], row["split"], row["modality"])
                        group_counts[key]["image_count"] += 1
                        group_counts[key]["blank_count"] += int(row["ink_pixels"] == 0)
                        metric_values[key]["ink_pixels"].append(float(row["ink_pixels"]))
                        metric_values[key]["ink_fraction_percent"].append(float(row["ink_fraction_percent"]))
                        width = float(row["raster_line_width_estimate_px"])
                        if math.isfinite(width):
                            metric_values[key]["raster_line_width_estimate_px"].append(width)
                        written += 1
                    handle.flush()
                    if written % (chunk_size * 20) == 0:
                        print(f"Processed {written}/{len(tasks)} sketches", flush=True)
        if written != len(tasks) or any(group_counts[key]["image_count"] != count for key, count in expected_by_group.items()):
            raise RuntimeError(f"row count mismatch: wrote {written}, expected {len(tasks)}")
        summary = _summarize(metric_values, group_counts)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        flattened = [{"dataset": r["dataset"], "split": r["split"], "image_count": r["image_count"], "blank_count": r["blank_count"], "metric": metric, **stats}
                     for r in summary.values() for metric, stats in r["metrics"].items()]
        _write_csv(output / "summary.csv", list(flattened[0]), flattened)
        _write_authoritative_report(summary, selected, THESIS_ROOT / "dataset_statistics", output)
        with per_image_path.open("rb") as handle:
            receipt["per_image_csv_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        receipt["software"] = {"opencv": cv2.__version__, "numpy": np.__version__}
        receipt["geometry_source_sha256"] = _sha256_bytes((ROOT / "src/spica/data/transforms.py").read_bytes())
        receipt["image_decode_scope"] = "sketch only; photo counts from manifests"
        receipt.update({"status": "PASS_FULL", "written_image_rows": written, "blank_counts": {"|".join(key): value["blank_count"] for key, value in group_counts.items()}})
        _write_receipt(output, receipt)
    except Exception as error:
        receipt.update({"status": "FAILED_PARTIAL", "error": f"{type(error).__name__}: {error}", "partial_csv": per_image_path.exists()})
        _write_receipt(output, receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="run only cheap synthetic checks; no output directory is created")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", type=Path, help="fresh ignored output directory under outputs/")
    args = parser.parse_args()
    if args.self_check:
        if args.workers != 8 or args.chunk_size != 512 or args.output is not None:
            raise SystemExit("--self-check cannot be combined with scan options")
        _self_check()
        return
    if args.output is None:
        parser.error("--output is required for a production scan")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    run(args.output, args.workers, args.chunk_size)


if __name__ == "__main__":
    main()
