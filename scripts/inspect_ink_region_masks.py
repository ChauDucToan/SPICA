"""Create a small, deterministic visual and JSON review of ink masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from spica.config.data import load_data_config
from spica.data.masking import apply_ink_mask, mask_seed
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split
from spica.data.transforms import build_clip_eval_transform

ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _choose(entries: tuple, count: int) -> tuple:
    by_label: dict[int, list] = {}
    for entry in entries:
        by_label.setdefault(entry.label, []).append(entry)
    ordered = [entry for label in sorted(by_label) for entry in sorted(by_label[label], key=lambda x: str(x.path))]
    if len(ordered) <= count:
        return tuple(ordered)
    return tuple(ordered[index * len(ordered) // count] for index in range(count))


def _preview(tensor, size: tuple[int, int]) -> Image.Image:
    mean = tensor.new_tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = tensor.new_tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    rgb = ((tensor * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0)
    return Image.fromarray(rgb.cpu().numpy()).resize(size, Image.Resampling.NEAREST)


def inspect(config_path: Path, output: Path, *, count: int, seed: int) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    config = load_data_config(config_path)
    sketches = read_manifest(config.train.sketch_manifest, config.root)
    photos = read_manifest(config.train.photo_manifest, config.root)
    classes = read_class_map(config.train.class_map)
    split = make_classwise_retrieval_split(sketches, photos, classes, seed=3407)
    entries = _choose(split.train_sketch_entries, count)
    val_entries = _choose(split.validation_sketch_entries, min(max(1, count // 5), len(split.validation_sketch_entries)))
    transform = build_clip_eval_transform()
    per_query: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    sheets: dict[float, list[Image.Image]] = {fraction: [] for fraction in FRACTIONS}

    for entry in (*entries, *val_entries):
        try:
            with Image.open(entry.path) as source:
                clean = transform(source.convert("RGB"))
            key = entry.path.resolve().relative_to(config.root.resolve()).as_posix()
            query = {"path": key, "label": entry.label, "split": "pseudo_train" if entry in entries else "pseudo_val", "seed": mask_seed(seed, key)}
            for fraction in FRACTIONS:
                masked, metadata = apply_ink_mask(clean, fraction=fraction, seed=int(query["seed"]))
                query[str(fraction)] = {key: metadata[key] for key in ("requested_fraction", "realized_fraction", "ink_pixels_before", "ink_pixels_erased", "ink_pixels_after", "bbox", "status")}
                sheets[fraction].append(_preview(masked, (160, 160)))
            per_query.append(query)
        except (OSError, ValueError, RuntimeError) as error:
            failures.append({"path": str(entry.path), "error": str(error)})

    columns = 10
    for fraction, previews in sheets.items():
        rows = (len(previews) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * 160, max(1, rows) * 160), "white")
        for index, preview in enumerate(previews):
            sheet.paste(preview, ((index % columns) * 160, (index // columns) * 160))
        sheet.save(output / ("ink_masks_clean.png" if fraction == 0 else f"ink_masks_{fraction:g}.png"))

    distribution: dict[str, dict[str, object]] = {}
    for fraction in FRACTIONS:
        records = [query[str(fraction)] for query in per_query]
        realized = [float(record["realized_fraction"]) for record in records]
        distribution[str(fraction)] = {
            "count": len(records),
            "realized_min": min(realized, default=0.0),
            "realized_mean": sum(realized) / len(realized) if realized else 0.0,
            "realized_max": max(realized, default=0.0),
            "status_counts": {status: sum(record["status"] == status for record in records) for status in ("ok", "blank_input", "zero_fraction", "target_unreachable")},
            "blank_count": sum(record["status"] == "blank_input" for record in records),
            "noop_count": sum(float(record["ink_pixels_erased"]) == 0 for record in records),
            "fullzero_prevented_count": sum(int(record["ink_pixels_before"]) > 0 and int(record["ink_pixels_after"]) >= 1 for record in records),
        }

    payload = {
        "policy_version": "ink_centered_square_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "dataset_root": str(config.root),
        "source_reference": "/home/oslamelon/Downloads/Research/ZS BIR dataset (read-only)",
        "manifests": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in (("train_sketch", config.train.sketch_manifest), ("train_photo", config.train.photo_manifest), ("class_map", config.train.class_map))},
        "pseudo_split": {"seed": 3407, "train_count": len(split.train_sketch_entries), "validation_count": len(split.validation_sketch_entries)},
        "selection": {"requested_count": count, "train_selected": len(entries), "validation_selected": len(val_entries), "mask_base_seed": seed, "fractions": list(FRACTIONS)},
        "sample_count": len(per_query),
        "failure_count": len(failures),
        "failures": failures,
        "distribution": distribution,
        "queries": per_query,
    }
    (output / "mask_review.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/sketchy_100_25.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    output = args.output or ROOT / f"outputs/masking_preparation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    payload = inspect(args.config, output, count=args.count, seed=args.seed)
    print(json.dumps({key: payload[key] for key in ("sample_count", "failure_count", "selection")}, indent=2, sort_keys=True))
    print(f"previews: {output}")


if __name__ == "__main__":
    main()
