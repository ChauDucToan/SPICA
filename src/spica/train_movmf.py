import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.loaders import build_retrieval_train_loader
from .models.clip import FrozenClipEncoder, load_frozen_clip
from .models.retrieval import MoVmfPhotoPredictor, MoVmfPrediction
from .models.vmf import (
    LOG_NORMALIZER_VERSION,
    SCORE_NORMALIZATION_VERSION,
    MoVmfLoss,
    mo_vmf_retrieval_loss,
)
from .train_deterministic import (
    HYDRA_CONFIG_DIR,
    _check_module_parameters_finite,
    _encode_triplet_images,
    _resolve_checkpoint_path,
    _resolve_data_config_path,
    _resolve_device,
    _seed_everything,
    _validate_training_options,
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
    margin: float,
    nll_weight: float,
    ranking_weight: float,
    seed: int,
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
                "objective": (
                    "positive_mixture_vmf_nll_plus_normalized_density_ranking"
                ),
                "log_normalizer": LOG_NORMALIZER_VERSION,
                "score_normalization": SCORE_NORMALIZATION_VERSION,
                "margin": margin,
                "nll_weight": nll_weight,
                "ranking_weight": ranking_weight,
                "positives_per_anchor_per_step": 1,
                "seed": seed,
            },
        },
        output_path,
    )


def _check_runtime_invariants(
    *,
    losses: MoVmfLoss,
    prediction: MoVmfPrediction,
    predictor: MoVmfPhotoPredictor,
    encoder: FrozenClipEncoder,
) -> None:
    for name, value in (
        ("total loss", losses.total),
        ("positive NLL", losses.positive_nll),
        ("density ranking loss", losses.density_ranking),
    ):
        if not torch.isfinite(value).item():
            raise FloatingPointError(f"{name} is not finite: {value.item()}")

    for name, values in (
        ("normalized positive scores", losses.normalized_positive_score),
        ("normalized negative scores", losses.normalized_negative_score),
        ("posterior responsibilities", losses.posterior_responsibilities),
        ("mixture probabilities", losses.mixture_probabilities),
        ("effective concentrations", losses.effective_concentration),
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
        max_error = (norms - 1).abs().max().item()
        raise RuntimeError(
            "Mo-vMF mean directions must have unit norm; "
            f"maximum norm error was {max_error:.3e}"
        )

    concentrations = prediction.concentrations
    if not torch.isfinite(concentrations).all().item():
        raise FloatingPointError("Mo-vMF concentrations contain non-finite values")
    if (
        torch.any(concentrations < predictor.min_concentration).item()
        or torch.any(concentrations > predictor.max_concentration).item()
    ):
        raise RuntimeError("Mo-vMF concentration escaped its configured bounds")
    if not torch.isfinite(prediction.mixture_logits).all().item():
        raise FloatingPointError("Mixture logits contain non-finite values")

    probability_sums = losses.mixture_probabilities.sum(dim=-1)
    posterior_sums = losses.posterior_responsibilities.sum(dim=-1)
    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Mixture probabilities do not sum to one")
    if not torch.allclose(
        posterior_sums,
        torch.ones_like(posterior_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Posterior responsibilities do not sum to one")

    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("Frozen CLIP encoder unexpectedly received gradients")


def _check_predictor_gradients(predictor: MoVmfPhotoPredictor) -> None:
    for name, parameter in predictor.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(
                f"Predictor parameter did not receive a gradient: {name}"
            )
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"Predictor gradient is not finite: {name}")


def _component_statistics(
    prediction: MoVmfPrediction,
    losses: MoVmfLoss,
) -> dict[str, float]:
    probabilities = losses.mixture_probabilities
    posterior = losses.posterior_responsibilities
    probability_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
        dim=-1
    )
    posterior_entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum(dim=-1)

    num_components = prediction.mean_directions.shape[1]
    upper_triangle = torch.triu(
        torch.ones(
            num_components,
            num_components,
            dtype=torch.bool,
            device=prediction.mean_directions.device,
        ),
        diagonal=1,
    )
    component_cosines = torch.einsum(
        "bkd,bjd->bkj",
        prediction.mean_directions,
        prediction.mean_directions,
    )[:, upper_triangle]

    return {
        "prior_effective_components": probability_entropy.exp().mean().item(),
        "posterior_effective_components": posterior_entropy.exp().mean().item(),
        "prior_max_probability": probabilities.max(dim=-1).values.mean().item(),
        "posterior_max_responsibility": posterior.max(dim=-1).values.mean().item(),
        "component_cosine": component_cosines.mean().item(),
    }


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_movmf",
)
def main(args: DictConfig) -> None:
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
    margin = float(args.margin)
    nll_weight = float(args.nll_weight)
    ranking_weight = float(args.ranking_weight)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    if not math.isfinite(nll_weight) or nll_weight <= 0:
        raise ValueError(f"nll_weight must be finite and positive, got {nll_weight}")
    if not math.isfinite(ranking_weight) or ranking_weight < 0:
        raise ValueError(
            f"ranking_weight must be finite and non-negative, got {ranking_weight}"
        )

    print(f"Loading {args.model_name} ({pretrained}) on {device}...")
    clip_bundle = load_frozen_clip(
        model_name=str(args.model_name),
        pretrained=pretrained,
        device=device,
    )
    encoder = clip_bundle.encoder
    train_loader = build_retrieval_train_loader(
        data_config,
        clip_bundle.transform,
        clip_bundle.transform,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=bool(args.drop_last),
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training loader has no batches; reduce batch_size or disable drop_last"
        )

    predictor = MoVmfPhotoPredictor(
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        num_components=int(args.num_components),
        min_concentration=float(args.min_concentration),
        max_concentration=float(args.max_concentration),
        initial_concentration=float(args.initial_concentration),
        component_init_std=float(args.component_init_std),
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    non_blocking = device.type == "cuda" and bool(args.pin_memory)

    print(
        f"Training K={predictor.num_components} Mo-vMF on "
        f"{len(train_loader.dataset)} sketch anchors for {max_steps} steps "
        f"(batch_size={args.batch_size})..."
    )
    print(
        f"Initial kappa={predictor.initial_concentration:g}, "
        f"bounds=({predictor.min_concentration:g}, "
        f"{predictor.max_concentration:g}), "
        f"normalizer={LOG_NORMALIZER_VERSION}, "
        f"score_normalization={SCORE_NORMALIZATION_VERSION}"
    )

    step = 0
    window_sums = {
        "total": 0.0,
        "nll": 0.0,
        "ranking": 0.0,
        "score_gap": 0.0,
        "kappa": 0.0,
        "prior_effective_components": 0.0,
        "posterior_effective_components": 0.0,
        "prior_max_probability": 0.0,
        "posterior_max_responsibility": 0.0,
        "component_cosine": 0.0,
    }
    window_kappa_min = math.inf
    window_kappa_max = -math.inf
    window_count = 0
    predictor.train()

    # Model heads consume a K-dependent number of random draws during
    # initialization. Reset before creating the DataLoader iterator so every K
    # sees the same shuffled anchors and worker-level positive/negative samples.
    _seed_everything(seed)

    while step < max_steps:
        for batch in train_loader:
            if torch.eq(batch["label"], batch["negative_label"]).any().item():
                raise RuntimeError("Training batch contains a same-class negative")

            sketch_images = batch["sketch"].to(
                device=device,
                non_blocking=non_blocking,
            )
            positive_images = batch["positive_photo"].to(
                device=device,
                non_blocking=non_blocking,
            )
            negative_images = batch["negative_photo"].to(
                device=device,
                non_blocking=non_blocking,
            )
            sketch_embeddings, positive_embeddings, negative_embeddings = (
                _encode_triplet_images(
                    encoder,
                    sketch_images,
                    positive_images,
                    negative_images,
                )
            )

            optimizer.zero_grad(set_to_none=True)
            prediction = predictor(sketch_embeddings)
            losses = mo_vmf_retrieval_loss(
                prediction,
                positive_embeddings,
                negative_embeddings,
                margin=margin,
                nll_weight=nll_weight,
                ranking_weight=ranking_weight,
            )
            _check_runtime_invariants(
                losses=losses,
                prediction=prediction,
                predictor=predictor,
                encoder=encoder,
            )

            losses.total.backward()
            _check_predictor_gradients(predictor)
            _check_runtime_invariants(
                losses=losses,
                prediction=prediction,
                predictor=predictor,
                encoder=encoder,
            )
            optimizer.step()
            _check_module_parameters_finite(predictor)

            with torch.no_grad():
                component_stats = _component_statistics(prediction, losses)
                concentrations = prediction.concentrations

            step += 1
            window_sums["total"] += losses.total.item()
            window_sums["nll"] += losses.positive_nll.item()
            window_sums["ranking"] += losses.density_ranking.item()
            window_sums["score_gap"] += (
                (losses.normalized_positive_score - losses.normalized_negative_score)
                .mean()
                .item()
            )
            window_sums["kappa"] += concentrations.mean().item()
            for name, value in component_stats.items():
                window_sums[name] += value
            window_kappa_min = min(window_kappa_min, concentrations.min().item())
            window_kappa_max = max(window_kappa_max, concentrations.max().item())
            window_count += 1

            if step % log_every == 0 or step == max_steps:
                means = {
                    name: value / window_count for name, value in window_sums.items()
                }
                print(
                    f"step={step:04d} "
                    f"loss={means['total']:.6f} "
                    f"nll={means['nll']:.6f} "
                    f"rank={means['ranking']:.6f} "
                    f"score_gap={means['score_gap']:.6f} "
                    f"kappa_mean={means['kappa']:.3f} "
                    f"kappa_range=[{window_kappa_min:.3f},"
                    f"{window_kappa_max:.3f}] "
                    f"effK_prior={means['prior_effective_components']:.3f} "
                    f"effK_post={means['posterior_effective_components']:.3f} "
                    f"max_post={means['posterior_max_responsibility']:.3f} "
                    f"mu_cos={means['component_cosine']:.4f}"
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
        margin=margin,
        nll_weight=nll_weight,
        ranking_weight=ranking_weight,
        seed=seed,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"K={predictor.num_components} Mo-vMF training run completed at step {step}.")


if __name__ == "__main__":
    main()
