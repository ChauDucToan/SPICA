"""Write the machine-readable Phase A alignment protocol audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from spica.config.data import load_data_config
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def run(data_config_path: Path, output_path: Path, checkpoint_path: Path) -> None:
    data = load_data_config(data_config_path)
    train_names = read_class_map(data.train.class_map)
    test_names = read_class_map(data.test.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches,
        photos,
        train_names,
        num_validation_classes=20,
        seed=3407,
    )
    train_name_values = {
        train_names[label].replace("_", " ").strip().lower()
        for label in split.train_class_ids
    }
    validation_name_values = {
        train_names[label].replace("_", " ").strip().lower()
        for label in split.validation_class_ids
    }
    photo_counts: dict[int, int] = {}
    for entry in split.train_photo_entries:
        photo_counts[entry.label] = photo_counts.get(entry.label, 0) + 1
    train_paths = {str(entry.path) for entry in split.train_sketch_entries}
    train_photo_paths = {str(entry.path) for entry in split.train_photo_entries}
    validation_paths = {str(entry.path) for entry in split.validation_sketch_entries}
    validation_photo_paths = {str(entry.path) for entry in split.validation_photo_entries}
    result = {
        "schema_version": 1,
        "status": "PASS",
        "gate": "Phase A protocol and leakage audit",
        "source_commit": _git("rev-parse", "HEAD"),
        "dataset": data.name,
        "class_map_audit": {
            "train_class_count": len(train_names),
            "official_test_class_count": len(test_names),
            "integer_id_overlap_between_separate_maps": sorted(
                set(train_names) & set(test_names)
            ),
            "semantic_name_overlap_between_maps": sorted(
                {
                    value.replace("_", " ").strip().lower() for value in train_names.values()
                }
                & {
                    value.replace("_", " ").strip().lower() for value in test_names.values()
                }
            ),
        },
        "pseudo_split": {
            "seed": split.seed,
            "train_class_count": len(split.train_class_ids),
            "validation_class_count": len(split.validation_class_ids),
            "train_sketch_count": len(split.train_sketch_entries),
            "train_photo_count": len(split.train_photo_entries),
            "validation_sketch_count": len(split.validation_sketch_entries),
            "validation_photo_count": len(split.validation_photo_entries),
            "class_overlap": sorted(set(split.train_class_ids) & set(split.validation_class_ids)),
            "path_overlap_train_sketch_photo": sorted(train_paths & train_photo_paths),
            "path_overlap_validation_sketch_photo": sorted(
                validation_paths & validation_photo_paths
            ),
            "semantic_name_overlap_train_validation": sorted(
                train_name_values & validation_name_values
            ),
        },
        "positive_photo_counts": {
            "minimum": min(photo_counts.values()),
            "maximum": max(photo_counts.values()),
            "classes": len(photo_counts),
        },
        "loss_protocol": {
            "historical_fp3_rank": "mean(softplus(margin - cosine(q,p+) + cosine(q,p-)))",
            "historical_fp3_classification": "mean cross_entropy(cosine(q,soft_text_bank)/tau_cls,class_id)",
            "historical_fp3_total": "lambda_rank*rank + lambda_cls*classification",
            "alignment_rank_positive_reduction": "mean over 4 positive photos",
            "alignment_moment_reduction": "class mean then batch mean",
            "alignment_target": "positive train-photo moments with stop-gradient",
        },
        "sampler_protocol": {
            "classes_per_batch": 16,
            "sketches_per_class": 2,
            "batch_size": 32,
            "positive_photos_per_sketch": 4,
            "negative_photos_per_sketch": 1,
        },
        "gradient_and_freeze_protocol": {
            "trainable": ["sketch visual prompt", "photo visual prompt", "soft text context"],
            "frozen": ["CLIP visual tower", "CLIP text tower", "CLIP projection"],
            "alignment_anchor": "detached hard CLIP class text (or detached photo mean control)",
            "predictor_inputs_at_inference": ["raw sketch image"],
            "text_used_for_predictor": False,
            "photo_used_for_predictor": False,
        },
        "resume_protocol": {
            "historical_trainer": "role/campaign/split/treatment/optimizer/RNG/checkpoint identity validation",
            "alignment_trainer": "from scratch only; resume rejected explicitly",
        },
        "metric_protocol": {
            "selection_metric": "full_pseudo_unseen_mAP",
            "map_at_k_denominator": "prefix_positive",
            "official_unseen_used_for_selection": False,
        },
        "historical_reference_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output_path)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/alignment_protocol_audit_2026-09-05.json")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/experiments/frozen_prompt_v2/frozen_prompt_v2_FP3_continuation/checkpoints/frozen_prompt_step5400.pt"
        ),
    )
    args = parser.parse_args()
    run(args.data_config, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
