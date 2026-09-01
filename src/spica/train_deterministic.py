import math
import random
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor, nn

from .config.data import load_data_config
from .data.loaders import build_retrieval_train_loader
from .models.clip import FrozenClipEncoder, load_frozen_clip
from .models.retrieval import DeterministicPhotoPredictor, pairwise_ranking_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_data_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute() or path.is_file():
        return path
    return PROJECT_ROOT / path


def _resolve_checkpoint_path(configured_path: object) -> Path:
    if configured_path is None:
        return Path(HydraConfig.get().runtime.output_dir) / "predictor.pt"

    path = Path(str(configured_path)).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _validate_training_options(
    *,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    log_every: int,
) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be finite and positive, got {learning_rate}"
        )
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(
            f"weight_decay must be finite and non-negative, got {weight_decay}"
        )
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}")


def _encode_triplet_images(
    encoder: FrozenClipEncoder,
    sketch_images: Tensor,
    positive_images: Tensor,
    negative_images: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if not (sketch_images.shape == positive_images.shape == negative_images.shape):
        raise RuntimeError(
            "Sketch, positive, and negative image batches must have equal shapes, "
            f"got {tuple(sketch_images.shape)}, {tuple(positive_images.shape)}, "
            f"and {tuple(negative_images.shape)}"
        )

    batch_size = sketch_images.shape[0]
    all_images = torch.cat(
        (sketch_images, positive_images, negative_images),
        dim=0,
    )
    with torch.no_grad():
        all_embeddings = encoder(all_images)

    chunks = all_embeddings.split(batch_size, dim=0)
    if len(chunks) != 3:
        raise RuntimeError(
            "The frozen encoder did not return three embedding batches, "
            f"got {len(chunks)}"
        )
    return chunks


def _save_checkpoint(
    output_path: Path,
    *,
    predictor: DeterministicPhotoPredictor,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset_name: str,
    model_name: str,
    pretrained: str | None,
    margin: float,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "deterministic_photo_predictor",
            "step": step,
            "model_config": {
                "embedding_dim": predictor.embedding_dim,
                "hidden_dim": predictor.hidden_dim,
            },
            "model_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "dataset": dataset_name,
                "split": "train",
                "model_name": model_name,
                "pretrained": pretrained,
                "margin": margin,
                "seed": seed,
            },
        },
        output_path,
    )


def _check_module_parameters_finite(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"Model parameter is not finite: {name}")


def _check_runtime_invariants(
    *,
    loss: Tensor,
    predicted_embeddings: Tensor,
    encoder: FrozenClipEncoder,
) -> None:
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"Training loss is not finite: {loss.item()}")

    norms = predicted_embeddings.norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    ):
        max_error = (norms - 1).abs().max().item()
        raise RuntimeError(
            "Predicted embeddings must have unit norm; "
            f"maximum norm error was {max_error:.3e}"
        )

    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("Frozen CLIP encoder unexpectedly received gradients")


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_deterministic",
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

    predictor = DeterministicPhotoPredictor(
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    margin = float(args.margin)
    non_blocking = device.type == "cuda" and bool(args.pin_memory)

    print(
        f"Training on {len(train_loader.dataset)} sketch anchors for "
        f"{max_steps} steps (batch_size={args.batch_size})..."
    )

    step = 0
    window_loss = 0.0
    window_positive = 0.0
    window_negative = 0.0
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
            predicted_embeddings = predictor(sketch_embeddings)
            loss = pairwise_ranking_loss(
                predicted_embeddings,
                positive_embeddings,
                negative_embeddings,
                margin=margin,
            )
            _check_runtime_invariants(
                loss=loss,
                predicted_embeddings=predicted_embeddings,
                encoder=encoder,
            )

            loss.backward()
            _check_runtime_invariants(
                loss=loss,
                predicted_embeddings=predicted_embeddings,
                encoder=encoder,
            )
            optimizer.step()
            _check_module_parameters_finite(predictor)

            with torch.no_grad():
                positive_scores = (predicted_embeddings * positive_embeddings).sum(
                    dim=-1
                )
                negative_scores = (predicted_embeddings * negative_embeddings).sum(
                    dim=-1
                )

            step += 1
            window_loss += loss.item()
            window_positive += positive_scores.mean().item()
            window_negative += negative_scores.mean().item()
            window_count += 1

            if step % log_every == 0 or step == max_steps:
                mean_loss = window_loss / window_count
                mean_positive = window_positive / window_count
                mean_negative = window_negative / window_count
                print(
                    f"step={step:04d} "
                    f"loss={mean_loss:.6f} "
                    f"positive_score={mean_positive:.6f} "
                    f"negative_score={mean_negative:.6f} "
                    f"score_gap={mean_positive - mean_negative:.6f}"
                )
                window_loss = 0.0
                window_positive = 0.0
                window_negative = 0.0
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
        seed=seed,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Deterministic training smoke run completed at step {step}.")


if __name__ == "__main__":
    main()
