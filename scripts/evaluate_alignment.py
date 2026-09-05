"""Evaluate an alignment checkpoint on official unseen data diagnostically only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.data.manifest import read_manifest
from spica.evaluation.frozen_prompt import encode_prompted_loader, evaluate_prompted
from spica.models.clip import load_frozen_clip
from spica.models.frozen_prompt import FrozenPromptModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(checkpoint_path: Path, data_config_path: Path, output_path: Path, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "frozen_prompt_alignment":
        raise ValueError("checkpoint is not an alignment checkpoint")
    if checkpoint.get("campaign") != "objective_alignment_2026-09-05":
        raise ValueError("only the primary alignment campaign is evaluated here")
    resolved = checkpoint["resolved_config"]
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else device
    )
    clip = load_frozen_clip(
        model_name=str(resolved["model_name"]),
        pretrained=resolved.get("pretrained"),
        device=selected_device,
    )
    model = FrozenPromptModel(
        clip.encoder.model.visual,
        prompt_length=int(resolved["visual_prompt_length"]),
        train_visual_layernorm=False,
        train_sketch_prompt=True,
        train_photo_prompt=True,
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    data = load_data_config(data_config_path)
    sketch_loader = DataLoader(
        RetrievalEvalDataset(
            read_manifest(data.test.sketch_manifest, data.root), clip.transform
        ),
        batch_size=int(resolved["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    photo_loader = DataLoader(
        RetrievalEvalDataset(
            read_manifest(data.test.photo_manifest, data.root), clip.transform
        ),
        batch_size=int(resolved["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    queries = encode_prompted_loader(model, sketch_loader)
    gallery = encode_prompted_loader(model, photo_loader, photo=True)
    evaluation = evaluate_prompted(
        queries,
        gallery,
        query_chunk_size=int(resolved["query_chunk_size"]),
        device=selected_device,
    )
    result = {
        "schema_version": 1,
        "campaign": checkpoint["campaign"],
        "experiment_role": checkpoint["experiment_role"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_global_step": checkpoint["training_global_step"],
        "official_unseen_diagnostic_only": True,
        "official_unseen_used_for_selection": False,
        "text_used_for_inference": False,
        "photo_used_for_predictor": False,
        "metrics": {
            "full_mAP": evaluation.metrics.mean_average_precision,
            "P@200": evaluation.metrics.precision_at_k.get(200),
            "mAP@200": evaluation.metrics.mean_average_precision_at_k.get(200),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["metrics"], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.checkpoint, args.data_config, args.output, args.device)


if __name__ == "__main__":
    main()
