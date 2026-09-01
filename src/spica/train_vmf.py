import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.loaders import build_retrieval_train_loader
from .models.clip import FrozenClipEncoder, load_frozen_clip
from .models.retrieval import K1VmfPhotoPredictor, VmfPrediction
from .models.vmf import K1VmfLoss, LOG_NORMALIZER_VERSION, k1_vmf_retrieval_loss
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
    predictor: K1VmfPhotoPredictor,
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
            "model_type": "k1_vmf_photo_predictor",
            "step": step,
            "model_config": {
                "embedding_dim": predictor.embedding_dim,
                "hidden_dim": predictor.hidden_dim,
                "min_concentration": predictor.min_concentration,
                "max_concentration": predictor.max_concentration,
                "initial_concentration": predictor.initial_concentration,
            },
            "model_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "dataset": dataset_name,
                "split": "train",
                "model_name": model_name,
                "pretrained": pretrained,
                "frozen_clip": True,
                "num_components": 1,
                "objective": "positive_vmf_nll_plus_cosine_ranking",
                "log_normalizer": LOG_NORMALIZER_VERSION,
                "margin": margin,
                "nll_weight": nll_weight,
                "ranking_weight": ranking_weight,
                "seed": seed,
            },
        },
        output_path,
    )


def _check_runtime_invariants(
    *,
    losses: K1VmfLoss,
    prediction: VmfPrediction,
    predictor: K1VmfPhotoPredictor,
    encoder: FrozenClipEncoder,
) -> None:
    for name, value in (
        ("total loss", losses.total),
        ("positive NLL", losses.positive_nll),
        ("cosine ranking loss", losses.cosine_ranking),
    ):
        if not torch.isfinite(value).item():
            raise FloatingPointError(f"{name} is not finite: {value.item()}")

    norms = prediction.mean_direction.norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    ):
        max_error = (norms - 1).abs().max().item()
        raise RuntimeError(
            "vMF mean directions must have unit norm; "
            f"maximum norm error was {max_error:.3e}"
        )

    concentration = prediction.concentration
    if not torch.isfinite(concentration).all().item():
        raise FloatingPointError("vMF concentration contains non-finite values")
    if (
        torch.any(concentration < predictor.min_concentration).item()
        or torch.any(concentration > predictor.max_concentration).item()
    ):
        raise RuntimeError("vMF concentration escaped its configured bounds")

    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("Frozen CLIP encoder unexpectedly received gradients")


def _check_concentration_gradient(predictor: K1VmfPhotoPredictor) -> None:
    output_layer = predictor.concentration_head[-1]
    if not isinstance(output_layer, torch.nn.Linear):
        raise TypeError("The concentration head must end with a Linear layer")
    gradient = output_layer.weight.grad
    if gradient is None:
        raise RuntimeError("The concentration head did not receive a gradient")
    if not torch.isfinite(gradient).all().item():
        raise FloatingPointError("The concentration gradient is not finite")


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_vmf",
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

    predictor = K1VmfPhotoPredictor(
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        min_concentration=float(args.min_concentration),
        max_concentration=float(args.max_concentration),
        initial_concentration=float(args.initial_concentration),
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    non_blocking = device.type == "cuda" and bool(args.pin_memory)

    print(
        f"Training K=1 vMF on {len(train_loader.dataset)} sketch anchors for "
        f"{max_steps} steps (batch_size={args.batch_size})..."
    )
    print(
        f"Initial kappa={predictor.initial_concentration:g}, "
        f"bounds=({predictor.min_concentration:g}, "
        f"{predictor.max_concentration:g}), "
        f"normalizer={LOG_NORMALIZER_VERSION}"
    )

    step = 0
    window_total = 0.0
    window_nll = 0.0
    window_ranking = 0.0
    window_positive = 0.0
    window_negative = 0.0
    window_likelihood_gap = 0.0
    window_kappa = 0.0
    window_kappa_min = math.inf
    window_kappa_max = -math.inf
    window_count = 0
    predictor.train()

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
            losses = k1_vmf_retrieval_loss(
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
            _check_concentration_gradient(predictor)
            _check_runtime_invariants(
                losses=losses,
                prediction=prediction,
                predictor=predictor,
                encoder=encoder,
            )
            optimizer.step()
            _check_module_parameters_finite(predictor)

            with torch.no_grad():
                likelihood_gap = prediction.concentration * (
                    losses.positive_cosine - losses.negative_cosine
                )
                concentration = prediction.concentration

            step += 1
            window_total += losses.total.item()
            window_nll += losses.positive_nll.item()
            window_ranking += losses.cosine_ranking.item()
            window_positive += losses.positive_cosine.mean().item()
            window_negative += losses.negative_cosine.mean().item()
            window_likelihood_gap += likelihood_gap.mean().item()
            window_kappa += concentration.mean().item()
            window_kappa_min = min(window_kappa_min, concentration.min().item())
            window_kappa_max = max(window_kappa_max, concentration.max().item())
            window_count += 1

            if step % log_every == 0 or step == max_steps:
                mean_positive = window_positive / window_count
                mean_negative = window_negative / window_count
                print(
                    f"step={step:04d} "
                    f"loss={window_total / window_count:.6f} "
                    f"nll={window_nll / window_count:.6f} "
                    f"rank={window_ranking / window_count:.6f} "
                    f"cos_gap={mean_positive - mean_negative:.6f} "
                    f"ll_gap={window_likelihood_gap / window_count:.6f} "
                    f"kappa_mean={window_kappa / window_count:.3f} "
                    f"kappa_range=[{window_kappa_min:.3f},"
                    f"{window_kappa_max:.3f}]"
                )
                window_total = 0.0
                window_nll = 0.0
                window_ranking = 0.0
                window_positive = 0.0
                window_negative = 0.0
                window_likelihood_gap = 0.0
                window_kappa = 0.0
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
    print(f"K=1 vMF training smoke run completed at step {step}.")


if __name__ == "__main__":
    main()
