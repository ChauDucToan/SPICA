"""Diagnostic official-test evaluator for a selected frozen-prompt checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from .config.data import load_data_config
from .data.datasets import RetrievalEvalDataset
from .data.manifest import read_manifest
from .evaluation.frozen_prompt import encode_prompted_loader, evaluate_prompted
from .models.clip import load_frozen_clip, load_trainable_sketch_hidden_encoder
from .models.frozen_prompt import FrozenPromptModel
from .frozen_prompt_artifacts import CAMPAIGN
from .train_frozen_prompt import _EarlyAdaptModel, _FrozenEncoderAdapter, _path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(
    checkpoint: dict[str, Any], args: DictConfig, device: torch.device
) -> tuple[Any, Any]:
    role = str(checkpoint["experiment_role"])
    if role == "frozen_prompt_v2_FP5":
        bundle = load_trainable_sketch_hidden_encoder(
            model_name=str(args.model_name),
            pretrained=args.pretrained,
            device=device,
            mode="partial",
            unfreeze_depth=4,
            train_ln_post=False,
        )
        model = _EarlyAdaptModel(bundle)
        transform = bundle.transform
    else:
        clip = load_frozen_clip(
            model_name=str(args.model_name),
            pretrained=args.pretrained,
            device=device,
        )
        model = FrozenPromptModel(
            clip.encoder.model.visual,
            prompt_length=int(checkpoint["resolved_config"]["visual_prompt_length"]),
            train_visual_layernorm=role == "frozen_prompt_v2_FP_LN",
            train_sketch_prompt=bool(
                checkpoint["resolved_config"].get("train_sketch_prompt", True)
            ),
            train_photo_prompt=bool(
                checkpoint["resolved_config"].get("train_photo_prompt", True)
            ),
        ).to(device)
        transform = clip.transform
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, transform


def run(args: DictConfig) -> None:
    checkpoint_path = _path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("campaign") != CAMPAIGN
        or checkpoint.get("format_version") != 2
        or checkpoint.get("run_kind") != "primary"
    ):
        raise ValueError("checkpoint is not a frozen-prompt v2 primary checkpoint")
    device = _device(str(args.device))
    data = load_data_config(_path(args.data_config))
    model, transform = _load_model(checkpoint, args, device)
    sketches = read_manifest(data.test.sketch_manifest, data.root)
    photos = read_manifest(data.test.photo_manifest, data.root)
    loader_options = {
        "batch_size": int(args.eval_batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
    }
    sketch_loader = DataLoader(
        RetrievalEvalDataset(sketches, transform), **loader_options
    )
    photo_loader = DataLoader(RetrievalEvalDataset(photos, transform), **loader_options)
    queries = encode_prompted_loader(model, sketch_loader)
    gallery_model = (
        _FrozenEncoderAdapter(model.encoder)
        if str(checkpoint["experiment_role"]) == "frozen_prompt_v2_FP5"
        else model
    )
    gallery = encode_prompted_loader(gallery_model, photo_loader, photo=True)
    evaluation = evaluate_prompted(
        queries, gallery, query_chunk_size=int(args.query_chunk_size), device=device
    )
    result = {
        "campaign": CAMPAIGN,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "experiment_role": checkpoint["experiment_role"],
        "official_unseen_diagnostic_only": True,
        "official_unseen_used_for_selection": False,
        "text_used_for_inference": False,
        "metrics": {
            "full_mAP": evaluation.metrics.mean_average_precision,
            "P@200": evaluation.metrics.precision_at_k.get(200),
            "mAP@200": evaluation.metrics.mean_average_precision_at_k.get(200),
        },
    }
    output = _path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="evaluate_frozen_prompt",
)
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
