import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.loaders import build_multi_positive_retrieval_train_loader
from .models.clip import FrozenClipEncoder, load_frozen_clip
from .models.retrieval import MoVmfPhotoPredictor, MoVmfPrediction
from .models.vmf import (
    LOG_NORMALIZER_VERSION,
    SCORE_NORMALIZATION_VERSION,
    MoVmfMultiPositiveLoss,
    dominant_satellite_regularization,
    mo_vmf_multi_positive_retrieval_loss,
)
from .training_utils import encode_multi_positive_images
from .train_deterministic import (
    HYDRA_CONFIG_DIR,
    _check_module_parameters_finite,
    _resolve_checkpoint_path,
    _resolve_data_config_path,
    _resolve_device,
    _seed_everything,
    _validate_training_options,
)

OBJECTIVE_NAME = "multi_positive_mixture_vmf_anticollapse"


def _scheduled_prediction(
    raw_prediction: MoVmfPrediction,
    *,
    step: int,
    warmup_steps: int,
    warmup_concentration: float,
    gate_temperature_start: float,
    gate_temperature_anneal_steps: int,
    warmup_dominant_weight: float | None,
) -> tuple[MoVmfPrediction, float, bool]:
    in_warmup = step < warmup_steps
    if in_warmup:
        if warmup_dominant_weight is None:
            warmup_logits = torch.zeros_like(raw_prediction.mixture_logits)
        else:
            num_components = raw_prediction.mixture_logits.shape[1]
            satellite_weight = (1.0 - warmup_dominant_weight) / (num_components - 1)
            warmup_weights = raw_prediction.mixture_logits.new_full(
                (num_components,),
                satellite_weight,
            )
            warmup_weights[0] = warmup_dominant_weight
            warmup_logits = warmup_weights.log()[None].expand_as(
                raw_prediction.mixture_logits
            )
        return (
            MoVmfPrediction(
                mean_directions=raw_prediction.mean_directions,
                concentrations=torch.full_like(
                    raw_prediction.concentrations,
                    warmup_concentration,
                ),
                mixture_logits=warmup_logits,
            ),
            math.inf,
            True,
        )

    anneal_progress = min(
        (step - warmup_steps) / gate_temperature_anneal_steps,
        1.0,
    )
    gate_temperature = gate_temperature_start + anneal_progress * (
        1.0 - gate_temperature_start
    )
    return (
        MoVmfPrediction(
            mean_directions=raw_prediction.mean_directions,
            concentrations=raw_prediction.concentrations,
            mixture_logits=raw_prediction.mixture_logits / gate_temperature,
        ),
        gate_temperature,
        False,
    )


def _save_checkpoint(
    output_path: Path,
    *,
    predictor: MoVmfPhotoPredictor,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset_name: str,
    model_name: str,
    pretrained: str | None,
    args: DictConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "mo_vmf_photo_predictor",
            "step": step,
            "model_config": {
                "embedding_dim": predictor.embedding_dim,
                "hidden_dim": predictor.hidden_dim,
                "num_components": predictor.num_components,
                "min_concentration": predictor.min_concentration,
                "max_concentration": predictor.max_concentration,
                "initial_concentration": predictor.initial_concentration,
                "component_init_std": predictor.component_init_std,
                "initial_dominant_weight": predictor.initial_dominant_weight,
                "concentration_mode": predictor.concentration_mode,
                "fixed_concentration": predictor.fixed_concentration,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in predictor.parameters()
                ),
                "initialization_scheme": "shared_direction_gate_order_v2",
            },
            "model_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "dataset": dataset_name,
                "split": "train",
                "model_name": model_name,
                "pretrained": pretrained,
                "frozen_clip": True,
                "num_components": predictor.num_components,
                "initialization_scheme": "shared_direction_gate_order_v2",
                "objective": OBJECTIVE_NAME,
                "ablation_stage": str(args.ablation_stage),
                "log_normalizer": LOG_NORMALIZER_VERSION,
                "score_normalization": SCORE_NORMALIZATION_VERSION,
                "margin": float(args.margin),
                "nll_weight": float(args.nll_weight),
                "ranking_weight": float(args.ranking_weight),
                "balance_weight": float(args.balance_weight),
                "sharpness_weight": float(args.sharpness_weight),
                "diversity_weight": float(args.diversity_weight),
                "assignment_weight": float(args.assignment_weight),
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
                "satellite_concentration_floor_weight": float(
                    args.satellite_concentration_floor_weight
                ),
                "target_dominant_weight": float(args.target_dominant_weight),
                "consistency_temperature": float(args.consistency_temperature),
                "satellite_concentration_floor": float(
                    args.satellite_concentration_floor
                ),
                "diversity_cosine_threshold": float(args.diversity_cosine_threshold),
                "ranking_score_transform": str(args.ranking_score_transform),
                "positives_per_anchor_per_step": int(args.num_positive_photos),
                "warmup_steps": int(args.warmup_steps),
                "gate_temperature_start": float(args.gate_temperature_start),
                "gate_temperature_anneal_steps": int(
                    args.gate_temperature_anneal_steps
                ),
                "seed": int(args.seed),
            },
        },
        output_path,
    )


def _check_runtime_invariants(
    *,
    losses: MoVmfMultiPositiveLoss,
    prediction: MoVmfPrediction,
    predictor: MoVmfPhotoPredictor,
    encoder: FrozenClipEncoder,
) -> None:
    scalar_losses = {
        "total": losses.total,
        "positive NLL": losses.positive_nll,
        "density ranking": losses.density_ranking,
        "posterior balance": losses.posterior_balance,
        "posterior sharpness": losses.posterior_sharpness,
        "balanced assignment": losses.balanced_assignment,
        "direction diversity": losses.direction_diversity,
    }
    for name, value in scalar_losses.items():
        if not torch.isfinite(value).item():
            raise FloatingPointError(f"{name} is not finite: {value.item()}")

    for name, values in (
        ("directions", prediction.mean_directions),
        ("concentrations", prediction.concentrations),
        ("mixture logits", prediction.mixture_logits),
        ("responsibilities", losses.posterior_responsibilities),
        ("positive scores", losses.normalized_positive_scores),
        ("negative score", losses.normalized_negative_score),
        ("ranking positive scores", losses.ranking_positive_scores),
        ("ranking negative score", losses.ranking_negative_score),
    ):
        if not torch.isfinite(values).all().item():
            raise FloatingPointError(f"{name} contain non-finite values")

    norms = prediction.mean_directions.norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Mo-vMF directions are not unit normalized")
    if (
        torch.any(prediction.concentrations < predictor.min_concentration).item()
        or torch.any(prediction.concentrations > predictor.max_concentration).item()
    ):
        raise RuntimeError("Mo-vMF concentration escaped its configured bounds")
    if not torch.allclose(
        losses.posterior_responsibilities.sum(dim=-1),
        torch.ones_like(losses.posterior_responsibilities[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Posterior responsibilities do not sum to one")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("Frozen CLIP encoder unexpectedly received gradients")


def _check_predictor_gradients(
    predictor: MoVmfPhotoPredictor,
    *,
    in_warmup: bool,
) -> None:
    for name, parameter in predictor.named_parameters():
        if parameter.grad is None:
            if in_warmup and name.startswith(("concentration_head", "mixture_head")):
                continue
            raise RuntimeError(
                f"Predictor parameter did not receive a gradient: {name}"
            )
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"Predictor gradient is not finite: {name}")


def _component_statistics(
    prediction: MoVmfPrediction,
    losses: MoVmfMultiPositiveLoss,
) -> dict[str, float]:
    probabilities = losses.mixture_probabilities
    prior_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    posterior = losses.posterior_responsibilities
    posterior_entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum(dim=-1)
    query_mean = losses.mean_query_responsibilities
    query_usage_entropy = -(query_mean * query_mean.clamp_min(1e-12).log()).sum(dim=-1)

    num_components = prediction.mean_directions.shape[1]
    if num_components > 1:
        component_pairs = torch.triu_indices(
            num_components,
            num_components,
            offset=1,
            device=prediction.mean_directions.device,
        )
        component_cosine = (
            (
                prediction.mean_directions[:, component_pairs[0], :]
                * prediction.mean_directions[:, component_pairs[1], :]
            )
            .sum(dim=-1)
            .mean()
            .item()
        )
    else:
        component_cosine = 1.0
    return {
        "prior_effective_components": prior_entropy.exp().mean().item(),
        "posterior_effective_components": posterior_entropy.exp().mean().item(),
        "query_usage_effective_components": (query_usage_entropy.exp().mean().item()),
        "posterior_max_responsibility": posterior.max(dim=-1).values.mean().item(),
        "component_cosine": component_cosine,
    }


def _validate_ablation_options(args: DictConfig) -> None:
    if str(args.ablation_stage) not in {"A", "B", "C", "D", "E"}:
        raise ValueError("ablation_stage must be A, B, C, D, or E")
    if int(args.num_positive_photos) <= 0:
        raise ValueError("num_positive_photos must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("warmup_steps must be non-negative")
    if int(args.gate_temperature_anneal_steps) <= 0:
        raise ValueError("gate_temperature_anneal_steps must be positive")
    gate_temperature_start = float(args.gate_temperature_start)
    if not math.isfinite(gate_temperature_start) or gate_temperature_start < 1:
        raise ValueError("gate_temperature_start must be finite and at least one")
    for name in (
        "gate_prior_weight",
        "dominant_sketch_anchor_weight",
        "dominant_photo_anchor_weight",
        "semantic_consistency_weight",
        "satellite_coverage_weight",
        "spread_matching_weight",
        "satellite_concentration_floor_weight",
    ):
        value = float(args[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if str(args.ablation_stage) in {"D", "E"}:
        if int(args.num_components) < 2:
            raise ValueError("Dominant-satellite stage requires K >= 2")
        if args.initial_dominant_weight is None:
            raise ValueError(
                "Dominant-satellite stage requires initial_dominant_weight"
            )


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_movmf_ablation",
)
def main(args: DictConfig) -> None:
    _validate_ablation_options(args)
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
    data_config = load_data_config(_resolve_data_config_path(str(args.data_config)))
    pretrained = None if args.pretrained is None else str(args.pretrained)
    print(f"Loading {args.model_name} ({pretrained}) on {device}...")
    clip_bundle = load_frozen_clip(
        model_name=str(args.model_name),
        pretrained=pretrained,
        device=device,
    )
    encoder = clip_bundle.encoder
    train_loader = build_multi_positive_retrieval_train_loader(
        data_config,
        clip_bundle.transform,
        clip_bundle.transform,
        num_positive_photos=int(args.num_positive_photos),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=bool(args.drop_last),
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader has no batches")

    predictor = MoVmfPhotoPredictor(
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        num_components=int(args.num_components),
        min_concentration=float(args.min_concentration),
        max_concentration=float(args.max_concentration),
        initial_concentration=float(args.initial_concentration),
        component_init_std=float(args.component_init_std),
        initial_dominant_weight=(
            None
            if args.initial_dominant_weight is None
            else float(args.initial_dominant_weight)
        ),
        concentration_mode=str(args.concentration_mode),
        fixed_concentration=(
            None
            if args.fixed_concentration is None
            else float(args.fixed_concentration)
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    non_blocking = device.type == "cuda" and bool(args.pin_memory)
    print(
        f"Training ablation {args.ablation_stage}: K={predictor.num_components}, "
        f"M={args.num_positive_photos}, steps={max_steps}, "
        f"warmup={args.warmup_steps}"
    )
    print(
        f"weights: balance={args.balance_weight}, "
        f"sharpness={args.sharpness_weight}, assignment={args.assignment_weight}, "
        f"diversity={args.diversity_weight}, "
        f"ranking_transform={args.ranking_score_transform}"
    )
    if str(args.ablation_stage) in {"D", "E"}:
        print(
            "dominant-satellite weights: "
            f"gate={args.gate_prior_weight}, "
            f"sketch={args.dominant_sketch_anchor_weight}, "
            f"photo={args.dominant_photo_anchor_weight}, "
            f"semantic={args.semantic_consistency_weight}, "
            f"coverage={args.satellite_coverage_weight}, "
            f"spread={args.spread_matching_weight}"
        )

    tracked_names = (
        "total",
        "nll",
        "ranking",
        "balance",
        "sharpness",
        "assignment",
        "diversity",
        "score_gap",
        "kappa",
        "prior_effective_components",
        "posterior_effective_components",
        "query_usage_effective_components",
        "posterior_max_responsibility",
        "component_cosine",
        "gate_prior",
        "dominant_sketch_anchor",
        "dominant_photo_anchor",
        "semantic_consistency",
        "satellite_coverage",
        "spread_matching",
        "satellite_concentration_floor",
    )
    window_sums = dict.fromkeys(tracked_names, 0.0)
    window_kappa_min = math.inf
    window_kappa_max = -math.inf
    window_count = 0
    step = 0
    predictor.train()

    # Keep model initialization random but make all ablations consume exactly
    # the same anchor order and worker-level positive/negative samples.
    _seed_everything(seed)
    while step < max_steps:
        for batch in train_loader:
            if torch.eq(batch["label"], batch["negative_label"]).any().item():
                raise RuntimeError("Training batch contains a same-class negative")
            sketch_images = batch["sketch"].to(device, non_blocking=non_blocking)
            positive_images = batch["positive_photos"].to(
                device,
                non_blocking=non_blocking,
            )
            negative_images = batch["negative_photo"].to(
                device,
                non_blocking=non_blocking,
            )
            sketch_embeddings, positive_embeddings, negative_embeddings = (
                encode_multi_positive_images(
                    encoder,
                    sketch_images,
                    positive_images,
                    negative_images,
                )
            )

            optimizer.zero_grad(set_to_none=True)
            raw_prediction = predictor(sketch_embeddings)
            prediction, gate_temperature, in_warmup = _scheduled_prediction(
                raw_prediction,
                step=step,
                warmup_steps=int(args.warmup_steps),
                warmup_concentration=(
                    predictor.fixed_concentration
                    if predictor.concentration_mode == "fixed"
                    else predictor.initial_concentration
                ),
                gate_temperature_start=float(args.gate_temperature_start),
                gate_temperature_anneal_steps=int(args.gate_temperature_anneal_steps),
                warmup_dominant_weight=(
                    None
                    if args.initial_dominant_weight is None
                    else float(args.initial_dominant_weight)
                ),
            )
            losses = mo_vmf_multi_positive_retrieval_loss(
                prediction,
                positive_embeddings,
                negative_embeddings,
                margin=float(args.margin),
                nll_weight=float(args.nll_weight),
                ranking_weight=float(args.ranking_weight),
                balance_weight=float(args.balance_weight),
                sharpness_weight=float(args.sharpness_weight),
                diversity_weight=float(args.diversity_weight),
                assignment_weight=float(args.assignment_weight),
                diversity_cosine_threshold=float(args.diversity_cosine_threshold),
                ranking_score_transform=str(args.ranking_score_transform),
            )
            _check_runtime_invariants(
                losses=losses,
                prediction=prediction,
                predictor=predictor,
                encoder=encoder,
            )
            regularization_values = {
                "gate_prior": losses.total.new_zeros(()),
                "dominant_sketch_anchor": losses.total.new_zeros(()),
                "dominant_photo_anchor": losses.total.new_zeros(()),
                "semantic_consistency": losses.total.new_zeros(()),
                "satellite_coverage": losses.total.new_zeros(()),
                "spread_matching": losses.total.new_zeros(()),
                "satellite_concentration_floor": losses.total.new_zeros(()),
            }
            combined_loss = losses.total
            if str(args.ablation_stage) in {"D", "E"}:
                regularization = dominant_satellite_regularization(
                    prediction,
                    sketch_embeddings,
                    positive_embeddings,
                    target_dominant_weight=float(args.target_dominant_weight),
                    consistency_temperature=float(args.consistency_temperature),
                    satellite_concentration_floor=float(
                        args.satellite_concentration_floor
                    ),
                )
                regularization_values = {
                    "gate_prior": regularization.gate_prior,
                    "dominant_sketch_anchor": (regularization.dominant_sketch_anchor),
                    "dominant_photo_anchor": regularization.dominant_photo_anchor,
                    "semantic_consistency": regularization.semantic_consistency,
                    "satellite_coverage": regularization.satellite_coverage,
                    "spread_matching": regularization.spread_matching,
                    "satellite_concentration_floor": (
                        regularization.satellite_concentration_floor
                    ),
                }
                combined_loss = (
                    combined_loss
                    + float(args.gate_prior_weight) * regularization.gate_prior
                    + float(args.dominant_sketch_anchor_weight)
                    * regularization.dominant_sketch_anchor
                    + float(args.dominant_photo_anchor_weight)
                    * regularization.dominant_photo_anchor
                    + float(args.semantic_consistency_weight)
                    * regularization.semantic_consistency
                    + float(args.satellite_coverage_weight)
                    * regularization.satellite_coverage
                    + float(args.spread_matching_weight)
                    * regularization.spread_matching
                    + float(args.satellite_concentration_floor_weight)
                    * regularization.satellite_concentration_floor
                )
            if not torch.isfinite(combined_loss).item():
                raise FloatingPointError("Combined training loss is not finite")
            for name, value in regularization_values.items():
                if not torch.isfinite(value).item():
                    raise FloatingPointError(
                        f"Dominant-satellite loss is not finite: {name}"
                    )
            combined_loss.backward()
            _check_predictor_gradients(predictor, in_warmup=in_warmup)
            _check_runtime_invariants(
                losses=losses,
                prediction=prediction,
                predictor=predictor,
                encoder=encoder,
            )
            optimizer.step()
            _check_module_parameters_finite(predictor)

            with torch.no_grad():
                stats = _component_statistics(prediction, losses)
                concentrations = prediction.concentrations
            step += 1
            values = {
                "total": combined_loss.item(),
                "nll": losses.positive_nll.item(),
                "ranking": losses.density_ranking.item(),
                "balance": losses.posterior_balance.item(),
                "sharpness": losses.posterior_sharpness.item(),
                "assignment": losses.balanced_assignment.item(),
                "diversity": losses.direction_diversity.item(),
                "score_gap": (
                    losses.ranking_positive_scores
                    - losses.ranking_negative_score[:, None]
                )
                .mean()
                .item(),
                "kappa": concentrations.mean().item(),
                **stats,
                **{name: value.item() for name, value in regularization_values.items()},
            }
            for name, value in values.items():
                window_sums[name] += value
            window_kappa_min = min(window_kappa_min, concentrations.min().item())
            window_kappa_max = max(window_kappa_max, concentrations.max().item())
            window_count += 1

            if step % log_every == 0 or step == max_steps:
                means = {
                    name: value / window_count for name, value in window_sums.items()
                }
                temperature_text = (
                    "warmup_fixed_prior" if in_warmup else f"{gate_temperature:.2f}"
                )
                print(
                    f"step={step:04d} loss={means['total']:.5f} "
                    f"nll={means['nll']:.3f} rank={means['ranking']:.4f} "
                    f"bal={means['balance']:.4f} sharp={means['sharpness']:.4f} "
                    f"assign={means['assignment']:.4f} "
                    f"div={means['diversity']:.4f} "
                    f"gap={means['score_gap']:.3f} "
                    f"kappa={means['kappa']:.1f} "
                    f"range=[{window_kappa_min:.1f},{window_kappa_max:.1f}] "
                    f"effK_prior={means['prior_effective_components']:.2f} "
                    f"effK_post={means['posterior_effective_components']:.2f} "
                    f"effK_query={means['query_usage_effective_components']:.2f} "
                    f"max_post={means['posterior_max_responsibility']:.3f} "
                    f"mu_cos={means['component_cosine']:.3f} "
                    f"gate_T={temperature_text}"
                )
                if str(args.ablation_stage) in {"D", "E"}:
                    print(
                        "  dominant-satellite: "
                        f"gate={means['gate_prior']:.4f} "
                        f"sketch={means['dominant_sketch_anchor']:.4f} "
                        f"photo={means['dominant_photo_anchor']:.4f} "
                        f"semantic={means['semantic_consistency']:.4f} "
                        f"coverage={means['satellite_coverage']:.4f} "
                        f"spread={means['spread_matching']:.4f} "
                        f"kappa_floor="
                        f"{means['satellite_concentration_floor']:.4f}"
                    )
                for name in window_sums:
                    window_sums[name] = 0.0
                window_kappa_min = math.inf
                window_kappa_max = -math.inf
                window_count = 0
            if step >= max_steps:
                break

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    _save_checkpoint(
        checkpoint_path,
        predictor=predictor,
        optimizer=optimizer,
        step=step,
        dataset_name=data_config.name,
        model_name=str(args.model_name),
        pretrained=pretrained,
        args=args,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Ablation {args.ablation_stage} completed at step {step}.")


if __name__ == "__main__":
    main()
