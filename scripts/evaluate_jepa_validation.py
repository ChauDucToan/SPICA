"""Evaluate one JEPA checkpoint on its pseudo-unseen class split.

This script intentionally never reads the official unseen class names or test
labels for model selection.  The held-out classes and their entries come from
the checkpoint's fixed training-split metadata.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.data.manifest import read_manifest
from spica.evaluation.embeddings import encode_retrieval_loader
from spica.evaluation.jepa import (
    encode_jepa_loader,
    evaluate_jepa_features,
    feature_probe_dict,
)
from spica.models.clip import load_frozen_clip
from spica.evaluate_jepa import _load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint, transform = _load_model(args.checkpoint, device=device)
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a dictionary")
    raw_ids = metadata.get("pseudo_validation_class_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("Checkpoint does not contain pseudo_validation_class_ids")
    validation_ids = {int(class_id) for class_id in raw_ids}
    data = load_data_config(args.data_config)
    all_sketches = read_manifest(data.train.sketch_manifest, data.root)
    all_photos = read_manifest(data.train.photo_manifest, data.root)
    sketches = tuple(entry for entry in all_sketches if entry.label in validation_ids)
    photos = tuple(entry for entry in all_photos if entry.label in validation_ids)
    if not sketches or not photos:
        raise ValueError("Pseudo-validation split has no sketches or photos")

    photo_clip = load_frozen_clip(
        model_name=str(metadata["model_name"]),
        pretrained=None
        if metadata["pretrained"] is None
        else str(metadata["pretrained"]),
        device=device,
    )
    gallery_loader = DataLoader(
        RetrievalEvalDataset(photos, photo_clip.transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    query_loader = DataLoader(
        RetrievalEvalDataset(sketches, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    gallery = encode_retrieval_loader(photo_clip.encoder, gallery_loader)
    features = encode_jepa_loader(model, query_loader, device=device)
    evaluation = evaluate_jepa_features(
        features,
        gallery,
        precision_at_k=(1, 5, 10, 100, 200),
        map_at_k=(200,),
        query_chunk_size=args.query_chunk_size,
        device=device,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "pseudo_validation_class_ids": sorted(validation_ids),
        "metrics": {
            "mAP": evaluation.metrics.mean_average_precision,
            "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
            "precision_at_k": evaluation.metrics.precision_at_k,
            "num_queries": evaluation.metrics.num_queries,
            "num_gallery_items": evaluation.metrics.num_gallery_items,
        },
        "feature_geometry": feature_probe_dict(features, gallery),
        "protocol": {
            "class_level_zero_shot": True,
            "official_test_classes_used": False,
            "text_used": False,
            "photo_gallery_reencoded_with_frozen_clip": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"pseudo-unseen mAP: {evaluation.metrics.mean_average_precision:.6f}")
    print(f"pseudo-unseen P@200: {evaluation.metrics.precision_at_k[200]:.6f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
