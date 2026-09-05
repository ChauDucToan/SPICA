"""Measure class-conditional CLIP-sphere geometry without fitting anything."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split
from spica.evaluation.frozen_prompt import encode_prompted_loader
from spica.evaluation.text_bank import encode_class_text_bank
from spica.models.alignment import class_conditional_geometry_diagnostics
from spica.models.clip import load_frozen_clip
from spica.models.frozen_prompt import FrozenPromptModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(entries: Any, max_per_class: int) -> tuple[Any, ...]:
    grouped: dict[int, list[Any]] = {}
    for entry in sorted(entries, key=lambda value: (int(value.label), str(value.path))):
        grouped.setdefault(int(entry.label), []).append(entry)
    return tuple(
        entry
        for label in sorted(grouped)
        for entry in grouped[label][:max_per_class]
    )


def _encode(entries: Any, transform: Any, model: Any, batch_size: int) -> Any:
    loader = DataLoader(
        RetrievalEvalDataset(entries, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return encode_prompted_loader(model, loader)


def run(
    checkpoint_path: Path,
    data_config_path: Path,
    output_path: Path,
    device: str,
    max_per_class: int,
    batch_size: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")
    resolved = checkpoint.get("resolved_config", {})
    device_value = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else device
    )
    clip = load_frozen_clip(
        model_name=str(resolved.get("model_name", "ViT-B-32-quickgelu")),
        pretrained=resolved.get("pretrained", "openai"),
        device=device_value,
    )
    model = FrozenPromptModel(
        clip.encoder.model.visual,
        prompt_length=int(resolved.get("visual_prompt_length", 3)),
        train_visual_layernorm=False,
        train_sketch_prompt=bool(resolved.get("train_sketch_prompt", True)),
        train_photo_prompt=bool(resolved.get("train_photo_prompt", True)),
    ).to(device_value)
    model.load_state_dict(checkpoint.get("model_state_dict", {}), strict=False)
    model.eval()

    data = load_data_config(data_config_path)
    names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches,
        photos,
        names,
        num_validation_classes=int(resolved.get("pseudo_val_num_classes", 20)),
        seed=int(resolved.get("pseudo_val_seed", 3407)),
    )

    train_sketches = _select(split.train_sketch_entries, max_per_class)
    train_photos = _select(split.train_photo_entries, max_per_class)
    val_sketches = _select(split.validation_sketch_entries, max_per_class)
    val_photos = _select(split.validation_photo_entries, max_per_class)
    train_sketch_values = _encode(train_sketches, clip.transform, model, batch_size)
    train_photo_values = _encode(train_photos, clip.transform, model, batch_size)
    val_sketch_values = _encode(val_sketches, clip.transform, model, batch_size)
    val_photo_values = _encode(val_photos, clip.transform, model, batch_size)
    train_names = {label: names[label] for label in split.train_class_ids}
    val_names = {label: names[label] for label in split.validation_class_ids}
    train_text = encode_class_text_bank(
        clip.encoder, clip.tokenizer, train_names
    )
    val_text = encode_class_text_bank(clip.encoder, clip.tokenizer, val_names)

    def diagnostic(sketches: Any, photos: Any, text: Any) -> dict[str, Any]:
        return class_conditional_geometry_diagnostics(
            sketches.embeddings,
            photos.embeddings,
            sketches.labels,
            photo_labels=photos.labels,
            text_embeddings=text.embeddings,
            text_labels=text.labels,
        )

    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "experiment_role": checkpoint.get("experiment_role"),
        "source_snapshot_hash": checkpoint.get("source_snapshot_hash"),
        "device": str(device_value),
        "max_per_class": max_per_class,
        "train_only_fitting": True,
        "validation_used_for_fitting": False,
        "official_unseen_used_for_selection": False,
        "pseudo_split": {
            "seed": split.seed,
            "train_class_ids": list(split.train_class_ids),
            "validation_class_ids": list(split.validation_class_ids),
            "train_class_count": len(split.train_class_ids),
            "validation_class_count": len(split.validation_class_ids),
        },
        "semantic_class_name_overlap": sorted(
            {
                names[label].replace("_", " ").strip().lower()
                for label in split.train_class_ids
            }
            & {
                names[label].replace("_", " ").strip().lower()
                for label in split.validation_class_ids
            }
        ),
        "train": {
            "sketch_count": len(train_sketches),
            "photo_count": len(train_photos),
            "geometry": diagnostic(train_sketch_values, train_photo_values, train_text),
        },
        "pseudo_unseen": {
            "sketch_count": len(val_sketches),
            "photo_count": len(val_photos),
            "geometry": diagnostic(val_sketch_values, val_photo_values, val_text),
        },
        "protocol": {
            "text_used_for_predictor": False,
            "photo_used_for_predictor": False,
            "text_used_only_as_diagnostic_anchor": True,
            "selection_metric": "full_pseudo_unseen_mAP",
            "official_unseen_used_for_selection": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["pseudo_unseen"]["geometry"], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/alignment_geometry_baseline_2026-09-05.json")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-per-class", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.max_per_class < 2:
        raise ValueError("--max-per-class must be at least 2")
    run(
        args.checkpoint,
        args.data_config,
        args.output,
        args.device,
        args.max_per_class,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
