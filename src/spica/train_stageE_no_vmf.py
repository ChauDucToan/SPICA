"""Matched deterministic Stage-E controls without vMF machinery.

The default ``routing_mode=none`` is the no-vMF/no-explicit-routing arm.  The
angular-routing arm uses the same trainer with ``routing_mode=angular`` and adds
only the categorical angular assignment objective.
"""

import json
import math
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.loaders import build_multi_positive_retrieval_train_loader
from .models.clip import load_frozen_clip
from .models.retrieval import (
    DeterministicK3PhotoPredictor,
    DeterministicK3Prediction,
    deterministic_angular_positive_assignment_loss,
    deterministic_dominant_satellite_regularization,
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
from .training_utils import encode_multi_positive_images

OBJECTIVE_NAME = "deterministic_k3_stageE_no_vmf"
ANGULAR_OBJECTIVE_NAME = "deterministic_k3_stageE_angular_routing"


def _scheduled_prediction(
    raw_prediction: DeterministicK3Prediction,
    *,
    step: int,
    warmup_steps: int,
    dominant_weight: float,
    temperature: float,
) -> tuple[DeterministicK3Prediction, bool]:
    if step < warmup_steps:
        satellite_weight = (1.0 - dominant_weight) / 2.0
        weights = raw_prediction.gate_logits.new_tensor(
            (dominant_weight, satellite_weight, satellite_weight)
        )
        logits = weights.log()[None].expand_as(raw_prediction.gate_logits)
        return DeterministicK3Prediction(raw_prediction.directions, logits), True
    return (
        DeterministicK3Prediction(
            raw_prediction.directions,
            raw_prediction.gate_logits / temperature,
        ),
        False,
    )


def _check_predictor_gradients(
    predictor: DeterministicK3PhotoPredictor,
    *,
    in_warmup: bool,
) -> None:
    for name, parameter in predictor.named_parameters():
        # Gates are intentionally fixed to the dominant prior during warmup.
        if parameter.grad is None:
            if in_warmup and name.startswith("gate_head"):
                continue
            raise RuntimeError(
                f"Predictor parameter did not receive a gradient: {name}"
            )
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"Predictor gradient is not finite: {name}")


def _save_checkpoint(
    path: Path,
    predictor: DeterministicK3PhotoPredictor,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_name: str,
    args: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    routing_mode = str(args.routing_mode)
    objective = ANGULAR_OBJECTIVE_NAME if routing_mode == "angular" else OBJECTIVE_NAME
    torch.save(
        {
            "format_version": 1,
            "model_type": "deterministic_k3_photo_predictor",
            "step": step,
            "model_config": {
                "embedding_dim": predictor.embedding_dim,
                "hidden_dim": predictor.hidden_dim,
                "num_components": predictor.num_components,
                "initial_dominant_weight": predictor.initial_dominant_weight,
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
                "num_components": predictor.num_components,
                "initialization_scheme": "shared_direction_gate_order_v2",
                "positives_per_anchor_per_step": int(args.num_positive_photos),
                "objective": objective,
                "ablation_stage": "E_angular_routing"
                if routing_mode == "angular"
                else "E_no_vmf",
                "vMF": False,
                "routing_mode": routing_mode,
                "assignment_temperature": float(args.assignment_temperature),
                "angular_assignment_weight": float(args.angular_assignment_weight),
                "margin": float(args.margin),
                "ranking_weight": 1.0,
                "gate_prior_weight": float(args.gate_prior_weight),
                "dominant_sketch_anchor_weight": float(
                    args.dominant_sketch_anchor_weight
                ),
                "dominant_photo_anchor_weight": float(
                    args.dominant_photo_anchor_weight
                ),
                "semantic_consistency_weight": float(args.semantic_consistency_weight),
                "satellite_coverage_weight": float(args.satellite_coverage_weight),
                "spread_matching_weight": float(args.spread_matching_weight),
                "target_dominant_weight": float(args.target_dominant_weight),
                "consistency_temperature": float(args.consistency_temperature),
                "warmup_steps": int(args.warmup_steps),
                "gate_temperature": float(args.gate_temperature),
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


def _validate_options(args: DictConfig) -> str:
    if int(args.num_components) != 3 or int(args.num_positive_photos) != 3:
        raise ValueError("Stage-E deterministic controls require K=3 and M=3")
    routing_mode = str(args.routing_mode)
    if routing_mode not in {"none", "angular"}:
        raise ValueError("routing_mode must be 'none' or 'angular'")
    assignment_temperature = float(args.assignment_temperature)
    if not math.isfinite(assignment_temperature) or assignment_temperature <= 0:
        raise ValueError("assignment_temperature must be finite and positive")
    angular_weight = float(args.angular_assignment_weight)
    if not math.isfinite(angular_weight) or angular_weight < 0:
        raise ValueError("angular_assignment_weight must be finite and non-negative")
    if routing_mode == "none" and angular_weight != 0:
        raise ValueError("routing_mode=none requires angular_assignment_weight=0")
    if routing_mode == "angular" and angular_weight <= 0:
        raise ValueError("routing_mode=angular requires angular_assignment_weight>0")
    gate_temperature = float(args.gate_temperature)
    if not math.isfinite(gate_temperature) or gate_temperature < 1:
        raise ValueError("gate_temperature must be finite and at least one")
    for name in (
        "gate_prior_weight",
        "dominant_sketch_anchor_weight",
        "dominant_photo_anchor_weight",
        "semantic_consistency_weight",
        "satellite_coverage_weight",
        "spread_matching_weight",
        "target_dominant_weight",
        "consistency_temperature",
    ):
        value = float(args[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    target = float(args.target_dominant_weight)
    if not 1 / 3 < target < 1:
        raise ValueError("target_dominant_weight must be between 1/3 and 1")
    if float(args.consistency_temperature) <= 0:
        raise ValueError("consistency_temperature must be positive")
    return routing_mode


def run(args: DictConfig) -> None:
    routing_mode = _validate_options(args)
    seed = int(args.seed)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    max_steps = int(args.max_steps)
    log_every = int(args.log_every)
    _validate_training_options(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        log_every=log_every,
    )
    _seed_everything(seed)

    device = _resolve_device(str(args.device))
    data = load_data_config(_resolve_data_config_path(str(args.data_config)))
    pretrained = None if args.pretrained is None else str(args.pretrained)
    print(f"Loading {args.model_name} ({pretrained}) on {device}...")
    clip = load_frozen_clip(
        model_name=str(args.model_name),
        pretrained=pretrained,
        device=device,
    )
    loader = build_multi_positive_retrieval_train_loader(
        data,
        clip.transform,
        clip.transform,
        num_positive_photos=int(args.num_positive_photos),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=bool(args.drop_last),
    )
    if len(loader) == 0:
        raise ValueError("Training loader has no batches")

    predictor = DeterministicK3PhotoPredictor(
        int(args.embedding_dim),
        int(args.hidden_dim),
        initial_dominant_weight=float(args.target_dominant_weight),
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    non_blocking = device.type == "cuda" and bool(args.pin_memory)
    print(
        f"Training Stage-E {routing_mode}: K=3, M=3, steps={max_steps}, "
        f"warmup={args.warmup_steps}, params="
        f"{sum(parameter.numel() for parameter in predictor.parameters())}"
    )

    # Reseed after model/optimizer construction so all deterministic controls
    # consume the same loader shuffle and per-sample random choices.
    _seed_everything(seed)
    step = 0
    history: list[dict[str, float | int]] = []
    predictor.train()
    while step < max_steps:
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
            raw_prediction = predictor(sketch)
            prediction, in_warmup = _scheduled_prediction(
                raw_prediction,
                step=step,
                warmup_steps=int(args.warmup_steps),
                dominant_weight=float(args.target_dominant_weight),
                temperature=float(args.gate_temperature),
            )
            ranking = deterministic_k3_multi_positive_retrieval_loss(
                prediction,
                positives,
                negative,
                margin=float(args.margin),
            )
            regularization = deterministic_dominant_satellite_regularization(
                prediction,
                sketch,
                positives,
                target_dominant_weight=float(args.target_dominant_weight),
                consistency_temperature=float(args.consistency_temperature),
            )
            combined_loss = (
                ranking
                + float(args.gate_prior_weight) * regularization.gate_prior
                + float(args.dominant_sketch_anchor_weight)
                * regularization.dominant_sketch_anchor
                + float(args.dominant_photo_anchor_weight)
                * regularization.dominant_photo_anchor
                + float(args.semantic_consistency_weight)
                * regularization.semantic_consistency
                + float(args.satellite_coverage_weight)
                * regularization.satellite_coverage
                + float(args.spread_matching_weight) * regularization.spread_matching
            )
            angular = None
            if routing_mode == "angular":
                angular = deterministic_angular_positive_assignment_loss(
                    prediction,
                    positives,
                    negative,
                    margin=float(args.margin),
                    assignment_temperature=float(args.assignment_temperature),
                )
                combined_loss = (
                    combined_loss
                    + float(args.angular_assignment_weight) * angular.total
                )
            if not torch.isfinite(combined_loss).item():
                raise FloatingPointError("Stage-E deterministic loss is not finite")
            combined_loss.backward()
            _check_predictor_gradients(predictor, in_warmup=in_warmup)
            optimizer.step()
            _check_module_parameters_finite(predictor)

            with torch.no_grad():
                gates = prediction.gate_logits.softmax(dim=-1)
                gate_entropy = (
                    -(gates * gates.clamp_min(1e-12).log()).sum(dim=-1).mean()
                )
                values: dict[str, float | int] = {
                    "step": step + 1,
                    "loss": combined_loss.item(),
                    "ranking": ranking.item(),
                    "gate_entropy": gate_entropy.item(),
                    "gate_0": gates[:, 0].mean().item(),
                    "gate_1": gates[:, 1].mean().item(),
                    "gate_2": gates[:, 2].mean().item(),
                    "dominant_sketch_anchor": regularization.dominant_sketch_anchor.item(),
                    "dominant_photo_anchor": regularization.dominant_photo_anchor.item(),
                    "semantic_consistency": regularization.semantic_consistency.item(),
                    "satellite_coverage": regularization.satellite_coverage.item(),
                    "spread_matching": regularization.spread_matching.item(),
                }
                if angular is not None:
                    values.update(
                        {
                            "angular_assignment": angular.total.item(),
                            "angular_assignment_entropy": angular.assignment_entropy.item(),
                            "angular_max_responsibility": angular.assignment_responsibilities.max(
                                dim=-1
                            )
                            .values.mean()
                            .item(),
                        }
                    )
            step += 1
            if step % log_every == 0 or step == max_steps:
                history.append(values)
                suffix = ""
                if angular is not None:
                    suffix = (
                        f" angular={values['angular_assignment']:.5f}"
                        f" route_H={values['angular_assignment_entropy']:.4f}"
                    )
                print(
                    f"step={step:04d} loss={values['loss']:.6f} "
                    f"rank={values['ranking']:.6f} "
                    f"gate_H={values['gate_entropy']:.4f} "
                    f"gate={[values['gate_0'], values['gate_1'], values['gate_2']]}"
                    f"{suffix}"
                )
            if step >= max_steps:
                break

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    _save_checkpoint(checkpoint_path, predictor, optimizer, step, data.name, args)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Stage-E deterministic {routing_mode} run completed at step {step}.")


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_stageE_no_vmf",
)
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
