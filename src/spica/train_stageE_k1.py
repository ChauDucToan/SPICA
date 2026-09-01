"""K=1 semantic-core control for the Stage-E shielding ablation."""

import json
import math
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.loaders import build_multi_positive_retrieval_train_loader
from .models.clip import load_frozen_clip
from .models.retrieval import (
    DeterministicPhotoPredictor,
    deterministic_single_direction_multi_positive_retrieval_loss,
)
from .train_deterministic import (
    HYDRA_CONFIG_DIR,
    _check_module_parameters_finite,
    _resolve_checkpoint_path,
    _resolve_data_config_path,
    _resolve_device,
    _seed_everything,
    _validate_training_options,
)
from .training_utils import encode_multi_positive_images

OBJECTIVE_NAME = "deterministic_k1_stageE_semantic_core"


def _validate_options(args: DictConfig) -> None:
    if int(args.num_positive_photos) != 3:
        raise ValueError("The K=1 semantic-core control requires M=3")
    for name in (
        "consistency_temperature",
        "dominant_sketch_anchor_weight",
        "dominant_photo_anchor_weight",
        "semantic_consistency_weight",
    ):
        value = float(args[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(args.consistency_temperature) <= 0:
        raise ValueError("consistency_temperature must be positive")


def _save_checkpoint(
    path: Path,
    predictor: DeterministicPhotoPredictor,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_name: str,
    args: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "deterministic_photo_predictor",
            "step": step,
            "model_config": {
                "embedding_dim": predictor.embedding_dim,
                "hidden_dim": predictor.hidden_dim,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in predictor.parameters()
                ),
            },
            "model_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "dataset": data_name,
                "split": "train",
                "model_name": str(args.model_name),
                "pretrained": None
                if args.pretrained is None
                else str(args.pretrained),
                "frozen_clip": True,
                "num_components": 1,
                "positives_per_anchor_per_step": int(args.num_positive_photos),
                "objective": OBJECTIVE_NAME,
                "ablation_stage": "E_k1_semantic_core",
                "vMF": False,
                "margin": float(args.margin),
                "dominant_sketch_anchor_weight": float(
                    args.dominant_sketch_anchor_weight
                ),
                "dominant_photo_anchor_weight": float(
                    args.dominant_photo_anchor_weight
                ),
                "semantic_consistency_weight": float(
                    args.semantic_consistency_weight
                ),
                "consistency_temperature": float(args.consistency_temperature),
                "warmup_steps": int(args.warmup_steps),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "seed": int(args.seed),
                "max_steps": int(args.max_steps),
                "data_config": str(args.data_config),
                "map_at_k_denominator": "prefix_positive",
            },
        },
        path,
    )


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_stageE_k1",
)
def main(args: DictConfig) -> None:
    _validate_options(args)
    _validate_training_options(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        max_steps=int(args.max_steps),
        log_every=int(args.log_every),
    )
    seed = int(args.seed)
    _seed_everything(seed)
    device = _resolve_device(str(args.device))
    data = load_data_config(_resolve_data_config_path(str(args.data_config)))
    pretrained = None if args.pretrained is None else str(args.pretrained)
    clip = load_frozen_clip(
        model_name=str(args.model_name),
        pretrained=pretrained,
        device=device,
    )
    loader = build_multi_positive_retrieval_train_loader(
        data,
        clip.transform,
        clip.transform,
        num_positive_photos=3,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=bool(args.drop_last),
    )
    if len(loader) == 0:
        raise ValueError("K=1 training loader has no batches")
    predictor = DeterministicPhotoPredictor(
        int(args.embedding_dim), int(args.hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    _seed_everything(seed)
    non_blocking = device.type == "cuda" and bool(args.pin_memory)
    step = 0
    history = []
    predictor.train()
    while step < int(args.max_steps):
        for batch in loader:
            if torch.eq(batch["label"], batch["negative_label"]).any().item():
                raise RuntimeError("Training batch contains a same-class negative")
            sketch, positives, negative = encode_multi_positive_images(
                clip.encoder,
                batch["sketch"].to(device, non_blocking=non_blocking),
                batch["positive_photos"].to(device, non_blocking=non_blocking),
                batch["negative_photo"].to(device, non_blocking=non_blocking),
            )
            optimizer.zero_grad(set_to_none=True)
            predicted = predictor(sketch)
            ranking = deterministic_single_direction_multi_positive_retrieval_loss(
                predicted,
                positives,
                negative,
                margin=float(args.margin),
            )
            predicted_sketch_anchor = (
                1.0 - (predicted * F.normalize(sketch, dim=-1)).sum(dim=-1)
            ).mean()
            centroid = F.normalize(positives.mean(dim=1), dim=-1)
            predicted_photo_anchor = (
                1.0 - (predicted * centroid).sum(dim=-1)
            ).mean()
            consistency_logits = (
                predicted @ F.normalize(sketch, dim=-1).T
                / float(args.consistency_temperature)
            )
            labels = torch.arange(sketch.shape[0], device=device)
            semantic_consistency = 0.5 * (
                F.cross_entropy(consistency_logits, labels)
                + F.cross_entropy(consistency_logits.T, labels)
            )
            loss = (
                ranking
                + float(args.dominant_sketch_anchor_weight) * predicted_sketch_anchor
                + float(args.dominant_photo_anchor_weight) * predicted_photo_anchor
                + float(args.semantic_consistency_weight) * semantic_consistency
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("K=1 Stage-E loss is not finite")
            loss.backward()
            for name, parameter in predictor.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError(f"Invalid predictor gradient: {name}")
            optimizer.step()
            _check_module_parameters_finite(predictor)
            step += 1
            if step % int(args.log_every) == 0 or step == int(args.max_steps):
                values = {
                    "step": step,
                    "loss": loss.item(),
                    "ranking": ranking.item(),
                    "dominant_sketch_anchor": predicted_sketch_anchor.item(),
                    "dominant_photo_anchor": predicted_photo_anchor.item(),
                    "semantic_consistency": semantic_consistency.item(),
                }
                history.append(values)
                print(
                    f"step={step:04d} loss={values['loss']:.6f} "
                    f"rank={values['ranking']:.6f}"
                )
            if step >= int(args.max_steps):
                break

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    _save_checkpoint(checkpoint_path, predictor, optimizer, step, data.name, args)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
