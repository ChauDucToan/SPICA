"""Run the train-photo-only multi-photo transport-direction probe."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split
from spica.evaluation.embeddings import encode_retrieval_loader
from spica.evaluation.transport import encode_transport_loader, multi_photo_component_alignment
from spica.evaluate_transport import _load_model, _resolve_device
from spica.models.clip import load_frozen_clip

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "1": "outputs/experiments/deep_rho_learned/2026-09-02_13-41-26/checkpoints/transport_step73.pt",
    "2": "outputs/experiments/deep_deterministic_k2/2026-09-02_14-57-35/checkpoints/transport_step73.pt",
    "4": "outputs/experiments/deep_deterministic_k4/2026-09-02_15-09-50/checkpoints/transport_step73.pt",
    "8": "outputs/experiments/deep_deterministic_k8/2026-09-02_15-21-47/checkpoints/transport_step73.pt",
}


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-09-02")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--photos-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-config", default="configs/data/sketchy_104_21.yaml")
    args = parser.parse_args()
    device = _resolve_device(args.device)
    data = load_data_config(project_path(args.data_config))
    class_names = read_class_map(data.train.class_map)
    train_sketches = read_manifest(data.train.sketch_manifest, data.root)
    train_photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        train_sketches,
        train_photos,
        class_names,
        num_validation_classes=20,
        seed=3407,
    )

    photo_clip = load_frozen_clip(
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        device=device,
    )
    photo_loader = DataLoader(
        RetrievalEvalDataset(train_photos, photo_clip.transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print("Encoding train-only photo embeddings...")
    train_photo_set = encode_retrieval_loader(photo_clip.encoder, photo_loader)
    results: dict[str, object] = {}
    for k, configured_checkpoint in DEFAULT_RUNS.items():
        checkpoint = project_path(configured_checkpoint)
        if not checkpoint.is_file():
            results[k] = {"run": str(checkpoint.relative_to(ROOT)), "alignment": None, "error": "checkpoint not found"}
            continue
        print(f"Encoding train sketches for K={k}...")
        model, payload, sketch_transform = _load_model(checkpoint, device=device)
        sketch_loader = DataLoader(
            RetrievalEvalDataset(split.train_sketch_entries, sketch_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        features = encode_transport_loader(model, sketch_loader, device=device)
        alignment = multi_photo_component_alignment(
            features,
            train_photo_set.embeddings,
            train_photo_set.labels,
            photos_per_class=args.photos_per_class,
            seed=3407,
        )
        results[k] = {
            "run": str(checkpoint.parent.parent.relative_to(ROOT)),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_step": int(payload["step"]),
            "alignment": alignment,
        }
        del features, sketch_loader, model, payload, sketch_transform
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = ROOT / "outputs" / f"transport_multi_photo_probe_{args.date}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "probe": "train sketches versus train-photo-only class prototypes and sampled instance residuals",
                "photos_per_class": args.photos_per_class,
                "seed": 3407,
                "split": {
                    "seed": split.seed,
                    "train_class_ids": list(split.train_class_ids),
                    "validation_class_ids": list(split.validation_class_ids),
                    "train_sketches": len(split.train_sketch_entries),
                    "train_photos": len(split.train_photo_entries),
                },
                "values": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
