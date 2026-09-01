"""Matched Stage-E architecture control with all vMF machinery removed."""

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.loaders import build_multi_positive_retrieval_train_loader
from .models.clip import load_frozen_clip
from .models.retrieval import (
    DeterministicK3PhotoPredictor,
    DeterministicK3Prediction,
    deterministic_dominant_satellite_regularization,
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
from .train_movmf_ablation import _encode_multi_positive_images

OBJECTIVE_NAME = "deterministic_k3_stageE_no_vmf"


def _scheduled_prediction(raw, *, step: int, warmup_steps: int, dominant_weight: float, temperature: float):
    if step < warmup_steps:
        weights = raw.gate_logits.new_tensor((dominant_weight, (1 - dominant_weight) / 2, (1 - dominant_weight) / 2))
        logits = weights.log()[None].expand_as(raw.gate_logits)
    else:
        logits = raw.gate_logits / temperature
    return DeterministicK3Prediction(raw.directions, logits)


def _save_checkpoint(path, predictor, optimizer, step, data_name, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "model_type": "deterministic_k3_photo_predictor",
        "step": step,
        "model_config": {
            "embedding_dim": predictor.embedding_dim,
            "hidden_dim": predictor.hidden_dim,
            "num_components": 3,
            "trainable_parameters": sum(p.numel() for p in predictor.parameters()),
        },
        "model_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "dataset": data_name, "split": "train", "model_name": str(args.model_name),
            "pretrained": None if args.pretrained is None else str(args.pretrained),
            "frozen_clip": True, "num_components": 3, "positives_per_anchor_per_step": 3,
            "objective": OBJECTIVE_NAME, "ablation_stage": "E_no_vmf",
            "margin": float(args.margin), "ranking_weight": 1.0,
            "gate_prior_weight": float(args.gate_prior_weight),
            "dominant_sketch_anchor_weight": float(args.dominant_sketch_anchor_weight),
            "dominant_photo_anchor_weight": float(args.dominant_photo_anchor_weight),
            "semantic_consistency_weight": float(args.semantic_consistency_weight),
            "satellite_coverage_weight": float(args.satellite_coverage_weight),
            "spread_matching_weight": float(args.spread_matching_weight),
            "target_dominant_weight": float(args.target_dominant_weight),
            "consistency_temperature": float(args.consistency_temperature),
            "warmup_steps": int(args.warmup_steps), "gate_temperature": float(args.gate_temperature),
            "batch_size": int(args.batch_size), "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay), "seed": int(args.seed),
            "max_steps": int(args.max_steps), "data_config": str(args.data_config),
        },
    }, path)


@hydra.main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="train_stageE_no_vmf")
def main(args: DictConfig) -> None:
    if int(args.num_components) != 3 or int(args.num_positive_photos) != 3:
        raise ValueError("stageE_no_vmf requires K=3 and M=3")
    _validate_training_options(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        max_steps=int(args.max_steps),
        log_every=int(args.log_every),
    )
    _seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))
    data = load_data_config(_resolve_data_config_path(str(args.data_config)))
    clip = load_frozen_clip(model_name=str(args.model_name), pretrained=None if args.pretrained is None else str(args.pretrained), device=device)
    loader = build_multi_positive_retrieval_train_loader(data, clip.transform, num_positive_photos=3,
        batch_size=int(args.batch_size), num_workers=int(args.num_workers), pin_memory=bool(args.pin_memory), drop_last=bool(args.drop_last))
    predictor = DeterministicK3PhotoPredictor(int(args.embedding_dim), int(args.hidden_dim), initial_dominant_weight=float(args.target_dominant_weight)).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    _seed_everything(int(args.seed))
    step = 0
    predictor.train()
    while step < int(args.max_steps):
        for batch in loader:
            if torch.eq(batch["label"], batch["negative_label"]).any().item():
                raise RuntimeError("Training batch contains a same-class negative")
            sketch, positives, negative = _encode_multi_positive_images(
                clip.encoder,
                batch["sketch"].to(device),
                batch["positive_photos"].to(device),
                batch["negative_photo"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = _scheduled_prediction(
                predictor(sketch),
                step=step,
                warmup_steps=int(args.warmup_steps),
                dominant_weight=float(args.target_dominant_weight),
                temperature=float(args.gate_temperature),
            )
            directions = F.normalize(prediction.directions, dim=-1)
            center = F.normalize(
                (prediction.gate_logits.softmax(-1)[..., None] * directions).sum(1),
                dim=-1,
            )
            pos = torch.einsum("bd,bmd->bm", center, positives)
            neg = (center * negative).sum(-1)
            loss = F.softplus(float(args.margin) - pos + neg[:, None]).mean()
            reg = deterministic_dominant_satellite_regularization(
                prediction,
                sketch,
                positives,
                target_dominant_weight=float(args.target_dominant_weight),
                consistency_temperature=float(args.consistency_temperature),
            )
            loss = (
                loss
                + float(args.gate_prior_weight) * reg.gate_prior
                + float(args.dominant_sketch_anchor_weight)
                * reg.dominant_sketch_anchor
                + float(args.dominant_photo_anchor_weight)
                * reg.dominant_photo_anchor
                + float(args.semantic_consistency_weight) * reg.semantic_consistency
                + float(args.satellite_coverage_weight) * reg.satellite_coverage
                + float(args.spread_matching_weight) * reg.spread_matching
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Stage-E no-vMF loss is not finite")
            loss.backward()
            for name, p in predictor.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all().item():
                    raise FloatingPointError(f"Invalid predictor gradient: {name}")
            optimizer.step()
            _check_module_parameters_finite(predictor)
            step += 1
            if step % int(args.log_every) == 0 or step == int(args.max_steps):
                gate_entropy = -(
                    prediction.gate_logits.softmax(-1)
                    * prediction.gate_logits.log_softmax(-1)
                ).sum(-1).mean()
                print(
                    f"step={step:04d} loss={loss.item():.6f} "
                    f"gate_entropy={gate_entropy.item():.4f}"
                )
            if step >= int(args.max_steps):
                break
    path = _resolve_checkpoint_path(args.checkpoint_path)
    _save_checkpoint(path, predictor, optimizer, step, data.name, args)
    print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    main()
