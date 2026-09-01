from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.loaders import build_multi_positive_retrieval_train_loader
from .models.clip import load_frozen_clip
from .models.retrieval import (
    DeterministicK3PhotoPredictor,
    deterministic_gate_weighted_barycenter,
    deterministic_k3_multi_positive_retrieval_loss,
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

OBJECTIVE_NAME = "deterministic_k3_multi_positive_gate_barycenter_ranking"


def _save_checkpoint(
    path: Path, predictor, optimizer, step: int, data_name: str, args: DictConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "deterministic_k3_photo_predictor",
            "step": step,
            "model_config": {
                "embedding_dim": predictor.embedding_dim,
                "hidden_dim": predictor.hidden_dim,
                "num_components": 3,
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
                "pretrained": None if args.pretrained is None else str(args.pretrained),
                "frozen_clip": True,
                "num_components": 3,
                "positives_per_anchor_per_step": int(args.num_positive_photos),
                "objective": OBJECTIVE_NAME,
                "margin": float(args.margin),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "num_workers": int(args.num_workers),
                "drop_last": bool(args.drop_last),
                "data_config": str(args.data_config),
                "gate_prior_weight": float(args.gate_prior_weight),
                "anchor_weight": float(args.anchor_weight),
                "diversity_weight": float(args.diversity_weight),
                "seed": int(args.seed),
                "max_steps": int(args.max_steps),
                "map_at_k_denominator": "prefix_positive",
            },
        },
        path,
    )


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_deterministic_k3",
)
def main(args: DictConfig) -> None:
    if int(args.num_positive_photos) != 3:
        raise ValueError("The deterministic K=3 control requires num_positive_photos=3")
    _validate_training_options(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        max_steps=int(args.max_steps),
        log_every=int(args.log_every),
    )
    _seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))
    data = load_data_config(_resolve_data_config_path(str(args.data_config)))
    clip = load_frozen_clip(
        model_name=str(args.model_name),
        pretrained=None if args.pretrained is None else str(args.pretrained),
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
        raise ValueError(
            "Training loader has no batches; reduce batch_size or disable drop_last"
        )
    predictor = DeterministicK3PhotoPredictor(
        int(args.embedding_dim), int(args.hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    non_blocking = device.type == "cuda" and bool(args.pin_memory)
    step = 0
    predictor.train()
    while step < int(args.max_steps):
        for batch in loader:
            if torch.eq(batch["label"], batch["negative_label"]).any().item():
                raise RuntimeError("Training batch contains a same-class negative")
            sketch, positives, negative = _encode_multi_positive_images(
                clip.encoder,
                batch["sketch"].to(device, non_blocking=non_blocking),
                batch["positive_photos"].to(device, non_blocking=non_blocking),
                batch["negative_photo"].to(device, non_blocking=non_blocking),
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = predictor(sketch)
            loss = deterministic_k3_multi_positive_retrieval_loss(
                prediction,
                positives,
                negative,
                margin=float(args.margin),
                gate_prior_weight=float(args.gate_prior_weight),
                anchor_weight=float(args.anchor_weight),
                diversity_weight=float(args.diversity_weight),
                sketch_embeddings=sketch,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Deterministic K3 loss is not finite")
            loss.backward()
            for name, parameter in predictor.named_parameters():
                if (
                    parameter.grad is None
                    or not torch.isfinite(parameter.grad).all().item()
                ):
                    raise FloatingPointError(f"Invalid predictor gradient: {name}")
            optimizer.step()
            _check_module_parameters_finite(predictor)
            step += 1
            if step % int(args.log_every) == 0 or step == int(args.max_steps):
                barycenter = deterministic_gate_weighted_barycenter(
                    prediction.directions, prediction.gate_logits
                )
                print(
                    f"step={step:04d} loss={loss.item():.6f} gate_entropy={(-(prediction.gate_logits.softmax(-1) * prediction.gate_logits.log_softmax(-1)).sum(-1).mean()).item():.4f} barycenter_norm={barycenter.norm(-1).mean().item():.4f}"
                )
            if step >= int(args.max_steps):
                break
    path = _resolve_checkpoint_path(args.checkpoint_path)
    _save_checkpoint(path, predictor, optimizer, step, data.name, args)
    print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    main()
