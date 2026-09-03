"""Train the Predictive Semantic Transport family.

This script deliberately keeps the old full-vector JEPA trainer untouched as
T0.  Its models consume raw sketches only, while frozen photo embeddings and a
loss-only frozen text bank live in this training/evaluation harness.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .config.data import load_data_config
from .data.datasets import MultiPositiveRetrievalTrainDataset, RetrievalEvalDataset
from .data.manifest import read_class_map, read_manifest
from .data.splits import (
    ClasswiseRetrievalSplit,
    make_classwise_retrieval_split,
    split_manifest_identity,
)
from .evaluation.embeddings import EncodedRetrievalSet, encode_retrieval_loader, load_encoded_retrieval_set
from .evaluation.transport import (
    TransportFeatureSet,
    encode_transport_loader,
    evaluate_base_queries,
    evaluate_transport_features,
    hidden_space_compatibility,
    training_angle_summary,
    transport_probe_dict,
    transport_query_correlations,
)
from .evaluation.text_bank import EncodedTextBank, encode_class_text_bank
from .models.clip import (
    FrozenClipEncoder,
    frozen_visual_projection,
    load_frozen_clip,
    load_trainable_sketch_hidden_encoder,
    visual_pre_projection,
)
from .models.transport import (
    SemanticTransportPrediction,
    SpicaPredictiveTransport,
    deterministic_direction_mixture_loss,
    directional_mixture_loss,
    fixed_origin_transport_target,
    photo_transport_target,
    transport_direction_loss,
    transport_distance_loss,
    transport_endpoint_loss,
    transport_geometry_loss,
    transport_ranking_loss,
)
from .models.jepa import (
    classification_accuracy,
    jepa_text_classification_loss,
    photo_semantic_target,
)
from .provenance import (
    capture_provenance,
    capture_rng_state,
    hash_json,
    restore_rng_state,
)
from .tracking.wandb import WandbExperiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")
OBJECTIVE_NAME = "predictive_semantic_transport"
PSEUDO_SPLIT_SEED = 3407


def _git_provenance(
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture exact source/environment provenance, including dirty content."""
    return capture_provenance(
        PROJECT_ROOT,
        resolved_config=resolved_config,
        command=[sys.executable, "-m", "spica.train_transport", *sys.argv[1:]],
    )


def _angle_values(
    origin: Tensor,
    positive_embeddings: Tensor,
    *,
    max_items: int,
) -> Tensor:
    """Return acos(origin·photo) for actual sampled training pairs."""
    if positive_embeddings.ndim != 3:
        raise ValueError("positive_embeddings must have shape [B,M,D]")
    origin = F.normalize(origin.detach(), dim=-1)
    photos = F.normalize(positive_embeddings.detach(), dim=-1)
    values = (origin[:, None, :] * photos).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angles = torch.acos(values).reshape(-1).cpu()
    if angles.numel() > max_items:
        angles = angles[:max_items]
    return angles


def _gradient_wrt(loss: Tensor, variable: Tensor) -> Tensor:
    """Differentiate a diagnostic loss with respect to a representation tensor."""
    if not loss.requires_grad or not variable.requires_grad:
        return torch.zeros_like(variable)
    gradient = torch.autograd.grad(
        loss,
        variable,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(variable) if gradient is None else gradient


def _gradient_pair_cosine(first: Tensor, second: Tensor) -> float | None:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    first_norm = first_flat.norm()
    second_norm = second_flat.norm()
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return None
    return float((first_flat * second_flat).sum().div(first_norm * second_norm).item())


def _cosine_matrix(vectors: dict[str, Tensor]) -> list[list[float | None]]:
    names = list(vectors)
    return [
        [_gradient_pair_cosine(vectors[left], vectors[right]) for right in names]
        for left in names
    ]


def _representation_gradient_conflicts(
    losses: dict[str, Tensor],
    prediction: SemanticTransportPrediction,
) -> dict[str, dict[str, object]]:
    """Keep query and base representation gradients in explicit namespaces."""
    result: dict[str, dict[str, object]] = {}
    for key, name, representation in (
        ("query", "dL/dq", prediction.q),
        ("base", "dL/dz0", prediction.z0),
    ):
        gradients = {
            loss_name: _gradient_wrt(loss, representation)
            for loss_name, loss in losses.items()
        }
        result[key] = {
            "space": name,
            "loss_names": list(losses),
            "cosine_matrix": _cosine_matrix(gradients),
        }
    return result


def _validate_device_optional(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise ValueError("diagnostic limits must be numeric")


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_project_path(configured_path: object) -> Path:
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _resolve_checkpoint_path(configured_path: object) -> Path:
    if configured_path is None:
        return Path(HydraConfig.get().runtime.output_dir) / "transport_final.pt"
    return _resolve_project_path(configured_path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)


def _validate_options(args: DictConfig) -> None:
    mode = str(args.transport_mode)
    if mode not in {"residual", "bounded_residual", "tangent"}:
        raise ValueError("transport_mode must be residual, bounded_residual, or tangent")
    if not isinstance(args.transport_enabled, bool):
        raise ValueError("transport_enabled must be boolean")
    encoder_mode = str(args.encoder_mode)
    if encoder_mode not in {"frozen", "partial", "full"}:
        raise ValueError("encoder_mode must be frozen, partial, or full")
    depth = int(args.encoder_unfreeze_depth)
    if encoder_mode == "partial" and depth <= 0:
        raise ValueError("partial encoder mode requires encoder_unfreeze_depth > 0")
    if encoder_mode != "partial" and depth != 0:
        raise ValueError("encoder_unfreeze_depth must be zero outside partial mode")
    k = int(args.K)
    if k <= 0:
        raise ValueError("K must be positive")
    if mode != "tangent" and k != 1:
        raise ValueError("residual transport currently supports K=1 only")
    if bool(args.use_vmf) and mode != "tangent":
        raise ValueError("use_vmf is only valid for tangent transport")
    if bool(args.use_vmf) and k < 2:
        raise ValueError("Mo-vMF transport requires K > 1; use the deterministic K=1 control first")
    if float(args.lambda_vmf) > 0 and not bool(args.use_vmf):
        raise ValueError("lambda_vmf > 0 requires use_vmf=true")
    if bool(args.use_vmf) and float(args.lambda_vmf) <= 0:
        raise ValueError("use_vmf=true requires lambda_vmf > 0")
    if str(args.rho_mode) not in {"shared", "component"}:
        raise ValueError("rho_mode must be shared or component")
    if str(args.rho_strategy) not in {"learned", "zero", "fixed", "linear_warmup", "cosine_warmup"}:
        raise ValueError("rho_strategy is not supported")
    if int(args.rho_warmup_steps) <= 0:
        raise ValueError("rho_warmup_steps must be positive")
    if not 0 < float(args.fixed_rho_degrees) <= float(args.rho_max_degrees):
        raise ValueError("fixed_rho_degrees must be in (0, rho_max_degrees]")
    if str(args.photo_target) not in {"instance", "class_prototype"}:
        raise ValueError("photo_target must be instance or class_prototype")
    if str(args.loss_profile) not in {"endpoint_rank", "transport"}:
        raise ValueError("loss_profile must be endpoint_rank or transport")
    for name in (
        "predictor_learning_rate",
        "encoder_learning_rate",
        "weight_decay",
        "alpha",
        "alpha_max",
        "initial_alpha",
        "rho_max_degrees",
        "initial_rho_degrees",
        "min_kappa",
        "max_kappa",
        "initial_kappa",
        "margin",
        "lambda_dir",
        "lambda_dist",
        "lambda_endpoint",
        "lambda_rank",
        "lambda_cls",
        "lambda_vmf",
        "lambda_geom",
        "tau_cls",
        "score_temperature",
        "assignment_temperature",
    ):
        value = float(args[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(args.predictor_learning_rate) <= 0 or float(args.encoder_learning_rate) <= 0:
        raise ValueError("predictor and encoder learning rates must be positive")
    if int(args.num_positive_photos) <= 0:
        raise ValueError("num_positive_photos must be positive")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(args.max_steps) <= 0 or int(args.log_every) <= 0:
        raise ValueError("max_steps and log_every must be positive")
    if not isinstance(args.freeze_encoder_on_resume, bool):
        raise ValueError("freeze_encoder_on_resume must be boolean")
    if args.freeze_encoder_on_resume and args.resume_checkpoint_path is None:
        raise ValueError("freeze_encoder_on_resume requires a resume checkpoint")
    if args.freeze_encoder_at_step is not None:
        freeze_step = int(args.freeze_encoder_at_step)
        if freeze_step < 0 or freeze_step > int(args.max_steps):
            raise ValueError("freeze_encoder_at_step must be between 0 and max_steps")
        if bool(args.freeze_encoder_on_resume):
            raise ValueError("choose freeze_encoder_on_resume or freeze_encoder_at_step, not both")
    offsets = args.probe_step_offsets
    if offsets is not None:
        for offset in offsets:
            if int(offset) != float(offset) or int(offset) < 0:
                raise ValueError("probe_step_offsets must contain non-negative integers")
    if any(int(step) < 0 or int(step) > int(args.max_steps) for step in args.gradient_conflict_steps):
        raise ValueError("gradient conflict steps must be within training range")
    if int(args.query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    if float(args.rho_max_degrees) <= 0 or float(args.rho_max_degrees) > 180:
        raise ValueError("rho_max_degrees must be in (0, 180]")
    if float(args.initial_rho_degrees) >= float(args.rho_max_degrees):
        raise ValueError("initial_rho_degrees must be smaller than rho_max_degrees")
    if float(args.initial_alpha) >= float(args.alpha_max) and str(args.transport_mode) == "bounded_residual":
        raise ValueError("initial_alpha must be smaller than alpha_max")
    if float(args.max_kappa) <= float(args.min_kappa):
        raise ValueError("max_kappa must exceed min_kappa")
    if not float(args.min_kappa) < float(args.initial_kappa) < float(args.max_kappa):
        raise ValueError("initial_kappa must be strictly within kappa bounds")
    if float(args.lambda_dir) == 0 and float(args.lambda_dist) == 0 and float(args.lambda_endpoint) == 0 and float(args.lambda_rank) == 0 and float(args.lambda_cls) == 0 and float(args.lambda_vmf) == 0 and float(args.lambda_geom) == 0:
        raise ValueError("at least one transport loss weight must be positive")
    if not isinstance(args.use_geometry_loss, bool):
        raise ValueError("use_geometry_loss must be boolean")
    if float(args.lambda_geom) > 0 and not bool(args.use_geometry_loss):
        raise ValueError("lambda_geom > 0 requires use_geometry_loss=true")
    if bool(args.use_geometry_loss) and float(args.lambda_geom) <= 0:
        raise ValueError("use_geometry_loss requires lambda_geom > 0")
    if int(args.pseudo_val_num_classes) <= 0 or int(args.pseudo_val_seed) < 0:
        raise ValueError("pseudo validation options are invalid")
    if str(args.train_class_scope) not in {"pseudo_train", "all_seen"}:
        raise ValueError("train_class_scope must be pseudo_train or all_seen")
    if not str(args.prompt_template).count("{}") == 1:
        raise ValueError("prompt_template must contain exactly one '{}' placeholder")
    if str(args.text_loss_location) not in {"q", "z0", "both", "none"}:
        raise ValueError("text_loss_location must be q, z0, both, or none")
    if str(args.direction_target) not in {"moving", "fixed_reference", "class_centroid", "none"}:
        raise ValueError("direction_target must be moving, fixed_reference, class_centroid, or none")
    if int(args.training_angle_diagnostic_max) <= 0:
        raise ValueError("training_angle_diagnostic_max must be positive")
    if not isinstance(args.reset_optimizer_on_resume, bool):
        raise ValueError("reset_optimizer_on_resume must be boolean")
    if str(args.optimizer_reset_scope) not in {"none", "all", "encoder", "head"}:
        raise ValueError("optimizer_reset_scope must be none, all, encoder, or head")
    expected_scope = "all" if bool(args.reset_optimizer_on_resume) else "none"
    if str(args.optimizer_reset_scope) not in {expected_scope, "encoder", "head"}:
        raise ValueError("reset_optimizer_on_resume and optimizer_reset_scope disagree")
    if not isinstance(args.reset_transport_head_on_resume, bool):
        raise ValueError("reset_transport_head_on_resume must be boolean")
    role = None if args.experiment_role is None else str(args.experiment_role)
    branches = {
        "freeze_optimizer_A": (False, False),
        "freeze_optimizer_B": (True, False),
        "freeze_optimizer_C": (False, True),
        "freeze_optimizer_D": (True, True),
    }
    if role == "optimizer_reset_only" and (
        args.freeze_encoder_at_step is not None or bool(args.freeze_encoder_on_resume)
    ):
        raise ValueError("optimizer_reset_only cannot freeze the encoder")
    if role in branches:
        expected_freeze, expected_reset = branches[role]
        observed_freeze = (
            args.freeze_encoder_at_step is not None
            and int(args.freeze_encoder_at_step) == 73
        )
        if (
            observed_freeze != expected_freeze
            or bool(args.reset_optimizer_on_resume) != expected_reset
            or bool(args.freeze_encoder_on_resume)
        ):
            raise ValueError(
                f"{role} requires freeze={expected_freeze}, reset={expected_reset}"
            )
        if args.resume_checkpoint_path is None:
            raise ValueError(f"{role} requires a common step-73 resume checkpoint")
    corrected_roles = set(branches) | {
        "freeze_optimizer_source",
        "two_stage_stage1",
        "two_stage_S1",
        "two_stage_S2",
        "two_stage_S3",
        "two_stage_S4",
    }
    if role in corrected_roles and str(args.experiment_campaign) != "transport_corrected_2026-09-02_v2":
        raise ValueError("corrected transport roles require the corrected campaign")
    if role == "freeze_optimizer_source" and (
        int(args.max_steps) != 73
        or not bool(args.save_optimizer)
        or args.resume_checkpoint_path is not None
    ):
        raise ValueError(
            "freeze_optimizer_source must be a fresh optimizer-bearing step-73 run"
        )
    if role == "two_stage_stage1" and not (
        args.transport_enabled is False
        and str(args.text_loss_location) == "z0"
        and float(args.lambda_endpoint) == 0.0
        and float(args.lambda_dist) == 0.0
        and int(args.max_steps) in {44, 73}
        and str(args.train_class_scope) == "pseudo_train"
    ):
        raise ValueError("two_stage_stage1 must train rank(z0)+CE(z0) on pseudo-train")
    two_stage_roles = {
        "two_stage_S1": ("none", 0.0),
        "two_stage_S2": ("class_centroid", 1.0),
        "two_stage_S3": ("moving", 1.0),
        "two_stage_S4": ("fixed_reference", 1.0),
    }
    if role in two_stage_roles:
        expected_direction, expected_lambda = two_stage_roles[role]
        if not (
            bool(args.transport_enabled)
            and int(args.K) == 1
            and not bool(args.use_vmf)
            and str(args.rho_strategy) == "fixed"
            and float(args.fixed_rho_degrees) == 15.0
            and float(args.lambda_endpoint) == 0.0
            and float(args.lambda_dist) == 0.0
            and str(args.direction_target) == expected_direction
            and float(args.lambda_dir) == expected_lambda
            and str(args.text_loss_location) == "q"
            and bool(args.freeze_encoder_on_resume)
            and args.freeze_encoder_at_step is None
            and bool(args.reset_optimizer_on_resume)
            and str(args.optimizer_reset_scope) == "all"
            and bool(args.reset_transport_head_on_resume)
            and args.resume_checkpoint_path is not None
            and args.stage1_selection_manifest_path is not None
            and args.probe_step_offsets is not None
            and str(args.train_class_scope) == "pseudo_train"
        ):
            raise ValueError(
                f"{role} must use its exact direction target, immediate freeze, fresh K=1 fixed-15 head, and selected Stage-1 origin"
            )
    if str(args.wandb_mode) not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")


def _build_split(
    data_config: Any,
    *,
    num_validation_classes: int,
    seed: int,
) -> tuple[ClasswiseRetrievalSplit, dict[int, str]]:
    class_names = read_class_map(data_config.train.class_map)
    sketches = read_manifest(data_config.train.sketch_manifest, data_config.root)
    photos = read_manifest(data_config.train.photo_manifest, data_config.root)
    split = make_classwise_retrieval_split(
        sketches,
        photos,
        class_names,
        num_validation_classes=num_validation_classes,
        seed=seed,
    )
    return split, class_names


def _build_train_loader(
    sketches,
    photos,
    transform,
    args: DictConfig,
    *,
    seed: int,
) -> DataLoader:
    dataset = MultiPositiveRetrievalTrainDataset(
        sketch_entries=sketches,
        photo_entries=photos,
        sketch_transform=transform,
        photo_transform=transform,
        num_positive_photos=int(args.num_positive_photos),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=bool(args.drop_last),
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=int(args.num_workers) > 0,
    )


def _build_eval_loader(entries, transform, args: DictConfig) -> DataLoader:
    return DataLoader(
        RetrievalEvalDataset(entries=entries, transform=transform),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
    )


def _encode_photo_targets(
    encoder: FrozenClipEncoder,
    positive_images: Tensor,
    negative_images: Tensor,
) -> tuple[Tensor, Tensor]:
    if positive_images.ndim != 5 or negative_images.ndim != 4:
        raise ValueError("invalid positive/negative photo image dimensions")
    batch_size, num_positives = positive_images.shape[:2]
    if negative_images.shape[0] != batch_size or positive_images.shape[2:] != negative_images.shape[1:]:
        raise ValueError("positive and negative image shapes must match")
    flattened = positive_images.flatten(0, 1)
    with torch.no_grad():
        embeddings = encoder(torch.cat((flattened, negative_images), dim=0))
    positives, negative = embeddings.split((batch_size * num_positives, batch_size), dim=0)
    return positives.reshape(batch_size, num_positives, -1), negative


def _encode_reference(encoder: FrozenClipEncoder, sketch_images: Tensor) -> Tensor:
    with torch.no_grad():
        return encoder(sketch_images).detach()


def _build_class_prototypes(
    encoder: FrozenClipEncoder,
    entries,
    transform,
    args: DictConfig,
) -> tuple[Tensor, Tensor]:
    """Precompute train-photo-only class centroids for the optional control."""
    loader = _build_eval_loader(entries, transform, args)
    encoded = encode_retrieval_loader(encoder, loader)
    labels = torch.unique(encoded.labels, sorted=True)
    prototypes = torch.stack(
        [
            F.normalize(encoded.embeddings[encoded.labels == label].mean(dim=0), dim=-1)
            for label in labels
        ]
    )
    return labels, prototypes


def _target_for_labels(
    positive_embeddings: Tensor,
    labels: Tensor,
    *,
    photo_target: str,
    prototype_labels: Tensor | None,
    prototypes: Tensor | None,
) -> Tensor:
    if photo_target == "instance":
        return photo_semantic_target(positive_embeddings)
    if prototype_labels is None or prototypes is None:
        raise ValueError("class prototype target requested without prototypes")
    positions = torch.searchsorted(prototype_labels.to(labels.device), labels)
    if torch.any(positions >= prototype_labels.shape[0]).item():
        raise ValueError("a training label is missing from the photo prototypes")
    observed = prototype_labels.to(labels.device)[positions]
    if not torch.equal(observed, labels):
        raise ValueError("a training label is missing from the photo prototypes")
    return prototypes.to(device=labels.device, dtype=positive_embeddings.dtype)[positions].detach()


def _parameter_counts(model: SpicaPredictiveTransport, photo_encoder: FrozenClipEncoder) -> dict[str, int]:
    return {
        "total_parameters": model.total_parameter_count,
        "trainable_parameters": model.trainable_parameter_count,
        "frozen_parameters": model.total_parameter_count - model.trainable_parameter_count,
        "transport_parameters": model.transport_parameter_count,
        "sketch_encoder_trainable_parameters": model.sketch_encoder_trainable_parameter_count,
        "frozen_photo_encoder_parameters": sum(p.numel() for p in photo_encoder.parameters()),
    }


def _capture_initial_parameters(model: SpicaPredictiveTransport) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _parameter_drift(model: SpicaPredictiveTransport, initial: dict[str, Tensor]) -> dict[str, float]:
    numerator: dict[str, float] = {}
    denominator: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name not in initial:
            continue
        value = parameter.detach().float().cpu()
        before = initial[name].float()
        block = name
        marker = "sketch_context_encoder.visual.transformer.resblocks."
        if marker in name:
            tail = name.split(marker, 1)[1]
            block = "visual_block_" + tail.split(".", 1)[0]
        numerator[block] = numerator.get(block, 0.0) + (value - before).norm().item() ** 2
        denominator[block] = denominator.get(block, 0.0) + before.norm().item() ** 2
    return {
        block: math.sqrt(value) / max(math.sqrt(denominator[block]), 1e-12)
        for block, value in numerator.items()
    }


def _cpu_state_dict(model: SpicaPredictiveTransport) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _gradient_vector(loss: Tensor, parameters: list[torch.nn.Parameter]) -> Tensor:
    """Flatten a loss gradient, using zeros for parameters it does not touch."""
    if not loss.requires_grad:
        return loss.new_zeros(sum(parameter.numel() for parameter in parameters))
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    values = [
        gradient.reshape(-1) if gradient is not None else parameter.new_zeros(parameter.numel())
        for gradient, parameter in zip(gradients, parameters)
    ]
    return torch.cat(values) if values else loss.new_zeros(1)


def _parameter_gradient_space(
    losses: dict[str, Tensor], parameters: list[torch.nn.Parameter]
) -> dict[str, object]:
    vectors = {
        name: _gradient_vector(loss, parameters) for name, loss in losses.items()
    }
    return {
        "space": "dL/dtheta",
        "loss_names": list(losses),
        "cosine_matrix": _cosine_matrix(vectors),
    }


def _restore_training_state(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    restore_optimizer: bool = True,
    restore_rng: bool = True,
    generator: torch.Generator | None = None,
) -> int:
    """Restore all saved continuation state or fail instead of partial-resuming."""
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if restore_optimizer:
        state = payload.get("optimizer_state_dict")
        if state is None:
            raise ValueError(
                "resume requested optimizer restoration but checkpoint has no optimizer state"
            )
        optimizer.load_state_dict(state)
    if scheduler is not None:
        state = payload.get("scheduler_state_dict")
        if state is None:
            raise ValueError(
                "resume requested scheduler restoration but checkpoint has no scheduler state"
            )
        scheduler.load_state_dict(state)
    if restore_rng:
        state = payload.get("rng_state")
        if not isinstance(state, dict):
            raise ValueError(
                "resume requested RNG restoration but checkpoint has no RNG state"
            )
        restore_rng_state(state, generator)
    return int(payload["step"])


def _freeze_encoder(model: SpicaPredictiveTransport) -> None:
    """Freeze gradients and state updates for the sketch encoder."""
    model.sketch_context_encoder.requires_grad_(False)
    model.sketch_context_encoder.eval()
    for parameter in model.sketch_context_encoder.parameters():
        parameter.grad = None


def _apply_resume_controls(
    model: SpicaPredictiveTransport,
    optimizer: torch.optim.Optimizer,
    *,
    resume_step: int,
    freeze_encoder_on_resume: bool,
    freeze_encoder_at_step: int | None,
) -> tuple[torch.optim.Optimizer, int | None, bool]:
    """Apply continuation freeze semantics before the first new update."""
    if freeze_encoder_on_resume and freeze_encoder_at_step is not None:
        raise ValueError("resume freeze and scheduled freeze cannot both be enabled")
    if freeze_encoder_on_resume:
        _freeze_encoder(model)
        return _rebuild_optimizer_without_encoder(optimizer, model), resume_step, True
    if freeze_encoder_at_step is not None and resume_step >= freeze_encoder_at_step:
        _freeze_encoder(model)
        return _rebuild_optimizer_without_encoder(optimizer, model), freeze_encoder_at_step, True
    return optimizer, freeze_encoder_at_step, False


def _effective_probe_steps(
    *,
    resume_step: int,
    absolute_steps: object,
    relative_offsets: object,
    max_steps: int,
    run_probes: bool,
) -> list[int]:
    """Resolve global probe steps while retaining resume-relative offsets."""
    values = {int(step) for step in absolute_steps} if run_probes else set()
    if relative_offsets is not None and run_probes:
        values.update(resume_step + int(offset) for offset in relative_offsets)
    if run_probes:
        values.add(int(max_steps))
        if resume_step == 0:
            values.add(0)
    invalid = sorted(step for step in values if step < resume_step or step > max_steps)
    if invalid:
        raise ValueError(
            f"effective probe steps must be between resume step {resume_step} and max_steps {max_steps}: {invalid}"
        )
    return sorted(values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stage1_selection(
    manifest_path: Path,
    *,
    resume_path: Path,
    payload: dict[str, Any],
    data_split_identity: dict[str, Any],
    expected_campaign: str,
    expected_manifest_identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate the immutable pseudo-unseen Stage-1 selection contract."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Stage-1 selection manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Stage-1 selection manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Stage-1 selection manifest schema")
    if manifest.get("official_unseen_used") is not False:
        raise ValueError("Stage-1 selection manifest permits official-unseen selection")
    if manifest.get("selection_metric") != "pseudo_unseen_validation_mAP":
        raise ValueError("Stage-1 selection must use pseudo-unseen validation mAP")
    if manifest.get("source_campaign") != expected_campaign:
        raise ValueError("Stage-1 selection campaign differs from this run")
    manifest_identity = manifest.get("data_manifest_identity")
    if manifest_identity is not None and manifest_identity != expected_manifest_identity:
        raise ValueError("Stage-1 selection manifest entries differ from this run")
    if manifest.get("source_role") != "two_stage_stage1":
        raise ValueError("Stage-1 selection source role is invalid")
    selected = manifest.get("selected")
    candidates = manifest.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        raise ValueError("Stage-1 selection manifest is missing selected/candidates")
    if int(selected.get("step", -1)) not in {44, 73}:
        raise ValueError("Stage-1 selection must select checkpoint step 44 or 73")
    if manifest.get("data_split_identity") != data_split_identity:
        raise ValueError("Stage-1 selection split identity differs from this run")
    selected_path = _resolve_project_path(selected.get("checkpoint"))
    if selected_path.resolve() != resume_path.resolve():
        raise ValueError("resume checkpoint is not the selected Stage-1 checkpoint")
    if not selected_path.is_file():
        raise FileNotFoundError(f"selected Stage-1 checkpoint not found: {selected_path}")
    if selected.get("checkpoint_sha256") != _sha256_file(selected_path):
        raise ValueError("selected Stage-1 checkpoint hash differs from manifest")
    if int(payload.get("step", -1)) != int(selected["step"]):
        raise ValueError("selected Stage-1 checkpoint step differs from payload")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("experiment_role") != "two_stage_stage1":
        raise ValueError("resume checkpoint is not a Stage-1 semantic-origin checkpoint")
    if metadata.get("experiment_campaign") != manifest.get("source_campaign"):
        raise ValueError("Stage-1 source campaign differs from selection manifest")
    if metadata.get("transport_enabled") is not False:
        raise ValueError("Stage-1 selection source must have transport disabled")
    if metadata.get("data_split_identity") != data_split_identity:
        raise ValueError("Stage-1 checkpoint split identity differs from this run")
    checkpoint_manifest_identity = payload.get("data_manifest_identity")
    if checkpoint_manifest_identity is not None and checkpoint_manifest_identity != expected_manifest_identity:
        raise ValueError("Stage-1 checkpoint entries differ from this run")
    if len(candidates) != 2 or any(not isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("Stage-1 selection must contain exactly two candidate objects")
    candidate_steps = sorted(int(candidate.get("step", -1)) for candidate in candidates)
    if candidate_steps != [44, 73]:
        raise ValueError("Stage-1 selection must contain exactly candidates 44 and 73")
    selected_candidate = next(candidate for candidate in candidates if int(candidate["step"]) == int(selected["step"]))
    if any(selected.get(key) != selected_candidate.get(key) for key in ("checkpoint", "checkpoint_sha256", "mAP")):
        raise ValueError("selected Stage-1 candidate does not match its candidate record")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "selection_metric": manifest["selection_metric"],
        "source_role": manifest["source_role"],
        "source_campaign": manifest.get("source_campaign"),
        "source_run_result": manifest.get("source_run_result"),
        "candidates": candidates,
        "selected": selected,
        "data_split_identity": data_split_identity,
        "data_manifest_identity": manifest_identity,
        "manifest_identity_verified": manifest_identity is not None and checkpoint_manifest_identity is not None,
        "official_unseen_used": False,
        "provenance": manifest.get("provenance"),
    }


def _rebuild_optimizer_without_encoder(
    optimizer: torch.optim.Optimizer,
    model: SpicaPredictiveTransport,
) -> torch.optim.Optimizer:
    """Drop the encoder group while retaining Adam state for transport params."""
    transport_parameters = [
        parameter for parameter in model.transport_head.parameters() if parameter.requires_grad
    ]
    old_group = optimizer.param_groups[0]
    options = {
        key: old_group[key]
        for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused")
        if key in old_group
    }
    rebuilt = type(optimizer)([{"params": transport_parameters, **options}])
    for parameter in transport_parameters:
        if parameter in optimizer.state:
            rebuilt.state[parameter] = optimizer.state[parameter]
    return rebuilt


def _save_checkpoint(
    path: Path,
    *,
    model: SpicaPredictiveTransport,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_name: str,
    args: DictConfig,
    split: ClasswiseRetrievalSplit,
    parameter_counts: dict[str, int],
    provenance: dict[str, Any],
    resolved_config: dict[str, Any],
    data_split_identity: dict[str, Any],
    data_manifest_identity: dict[str, Any],
    generator: torch.Generator | None,
    include_optimizer: bool = False,
    continuation_metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rho_max = math.radians(float(args.rho_max_degrees))
    payload = {
        "format_version": 1,
        "model_type": "predictive_semantic_transport",
        "step": int(step),
        "model_config": {
            "hidden_dim": model.hidden_dim,
            "embedding_dim": model.embedding_dim,
            "predictor_hidden_dim": int(args.predictor_hidden_dim),
            "transport_mode": str(args.transport_mode),
            "transport_enabled": bool(args.transport_enabled),
            "K": int(args.K),
            "use_z0": bool(args.use_z0),
            "alpha": float(args.alpha),
            "alpha_max": float(args.alpha_max),
            "initial_alpha": float(args.initial_alpha),
            "rho_max": rho_max,
            "initial_rho": math.radians(float(args.initial_rho_degrees)),
            "rho_mode": str(args.rho_mode),
            "rho_strategy": str(args.rho_strategy),
            "fixed_rho": math.radians(float(args.fixed_rho_degrees)),
            "rho_warmup_steps": int(args.rho_warmup_steps),
            "use_vmf": bool(args.use_vmf),
            "min_kappa": float(args.min_kappa),
            "max_kappa": float(args.max_kappa),
            "initial_kappa": float(args.initial_kappa),
            "encoder_mode": str(args.encoder_mode),
            "encoder_unfreeze_depth": int(args.encoder_unfreeze_depth),
            "encoder_model_name": str(args.model_name),
            "encoder_pretrained": None
            if args.pretrained is None
            else str(args.pretrained),
            "parameter_counts": parameter_counts,
        },
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if include_optimizer else None,
        "scheduler_state_dict": None,
        "rng_state": capture_rng_state(generator),
        "global_step": int(step),
        "resolved_config": resolved_config,
        "data_split_identity": data_split_identity,
        "data_manifest_identity": data_manifest_identity,
        "continuation": continuation_metadata or {},
        "metadata": {
            "dataset": data_name,
            "split": "train",
            "model_name": str(args.model_name),
            "pretrained": None if args.pretrained is None else str(args.pretrained),
            "objective": OBJECTIVE_NAME,
            "model_family": "predictive_semantic_transport",
            "frozen_photo_projection": True,
            "photo_target_stop_gradient": True,
            "sketch_encoder_hidden_before_projection": True,
            "sketch_encoder_initialized_from_clip": True,
            "text_enters_predictor": False,
            "text_enters_gate": False,
            "text_enters_distance_head": False,
            "text_enters_vmf": False,
            "transport_enabled": bool(args.transport_enabled),
            "inference_inputs": ["raw_sketch_image"],
            "encoder_mode": str(args.encoder_mode),
            "encoder_unfreeze_depth": int(args.encoder_unfreeze_depth),
            "num_positive_photos": int(args.num_positive_photos),
            "photo_target": str(args.photo_target),
            "loss_profile": str(args.loss_profile),
            "lambda_dir": float(args.lambda_dir),
            "lambda_dist": float(args.lambda_dist),
            "lambda_endpoint": float(args.lambda_endpoint),
            "lambda_rank": float(args.lambda_rank),
            "lambda_cls": float(args.lambda_cls),
            "lambda_vmf": float(args.lambda_vmf),
            "lambda_geom": float(args.lambda_geom),
            "rho_max_degrees": float(args.rho_max_degrees),
            "rho_mode": str(args.rho_mode),
            "rho_strategy": str(args.rho_strategy),
            "fixed_rho_degrees": float(args.fixed_rho_degrees),
            "rho_warmup_steps": int(args.rho_warmup_steps),
            "direction_target": str(args.direction_target),
            "text_loss_location": str(args.text_loss_location),
            "score_temperature": float(args.score_temperature),
            "seed": int(args.seed),
            "predictor_learning_rate": float(args.predictor_learning_rate),
            "encoder_learning_rate": float(args.encoder_learning_rate),
            "max_steps": int(args.max_steps),
            "pseudo_validation_seed": split.seed,
            "pseudo_validation_class_ids": list(split.validation_class_ids),
            "pseudo_train_class_ids": list(split.train_class_ids),
            "checkpoint_contains_optimizer": include_optimizer,
            "checkpoint_contains_scheduler": False,
            "checkpoint_contains_rng": True,
            "data_split_identity": data_split_identity,
            "data_manifest_identity": data_manifest_identity,
            "experiment_role": None
            if args.experiment_role is None
            else str(args.experiment_role),
            "experiment_campaign": None
            if args.experiment_campaign is None
            else str(args.experiment_campaign),
            "freeze_encoder_on_resume": bool(args.freeze_encoder_on_resume),
            "stage1_selection_manifest_path": None
            if args.stage1_selection_manifest_path is None
            else str(args.stage1_selection_manifest_path),
            "continuation": continuation_metadata or {},
            "provenance": provenance,
        },
    }
    torch.save(payload, path)


def _metrics_summary(evaluation) -> dict[str, object]:
    return {
        "mAP": evaluation.metrics.mean_average_precision,
        "P@200": evaluation.metrics.precision_at_k.get(200),
        "precision_at_k": evaluation.metrics.precision_at_k,
        "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
        "num_queries": evaluation.metrics.num_queries,
        "num_gallery_items": evaluation.metrics.num_gallery_items,
    }


def _radius_ap_payload(evaluation, rho: Tensor, *, max_items: int = 2048) -> dict[str, list[float]]:
    values = rho if rho.ndim == 1 else rho.mean(dim=-1)
    average_precision = evaluation.average_precision_per_query
    if values.shape[0] != average_precision.shape[0]:
        raise ValueError("rho and per-query AP must have matching lengths")
    if values.shape[0] > max_items:
        indices = torch.linspace(0, values.shape[0] - 1, max_items).long()
        values = values[indices]
        average_precision = average_precision[indices]
    return {
        "rho_degrees": (values.detach().cpu() * (180.0 / math.pi)).tolist(),
        "average_precision": average_precision.detach().cpu().tolist(),
    }


def _probe(
    *,
    model: SpicaPredictiveTransport,
    val_loader: DataLoader,
    val_gallery: EncodedRetrievalSet,
    test_loader: DataLoader,
    test_gallery: EncodedRetrievalSet,
    val_reference: Tensor,
    test_reference: Tensor,
    val_hidden_reference: Tensor,
    test_hidden_reference: Tensor,
    initial_val: TransportFeatureSet,
    initial_test: TransportFeatureSet,
    device: torch.device,
    args: DictConfig,
) -> tuple[dict[str, object], TransportFeatureSet, TransportFeatureSet]:
    val_features = encode_transport_loader(model, val_loader, device=device)
    test_features = encode_transport_loader(model, test_loader, device=device)
    modes = tuple(
        dict.fromkeys(
            ("barycentric", "angular_logsumexp", "max")
            if int(args.K) > 1
            else ("barycentric",)
        )
    )
    val_evaluations = evaluate_transport_features(
        val_features,
        val_gallery,
        modes=modes,
        temperature=float(args.score_temperature),
        precision_at_k=tuple(int(k) for k in args.precision_at_k),
        map_at_k=tuple(int(k) for k in args.map_at_k),
        map_at_k_denominator=str(args.map_at_k_denominator),
        query_chunk_size=int(args.query_chunk_size),
        device=device,
    )
    test_evaluations = evaluate_transport_features(
        test_features,
        test_gallery,
        modes=modes,
        temperature=float(args.score_temperature),
        precision_at_k=tuple(int(k) for k in args.precision_at_k),
        map_at_k=tuple(int(k) for k in args.map_at_k),
        map_at_k_denominator=str(args.map_at_k_denominator),
        query_chunk_size=int(args.query_chunk_size),
        device=device,
    )
    selected_mode = str(args.inference_score_mode)
    if selected_mode not in val_evaluations:
        raise ValueError(f"inference_score_mode {selected_mode!r} is not available")
    common_eval_args = {
        "temperature": float(args.score_temperature),
        "precision_at_k": tuple(int(k) for k in args.precision_at_k),
        "map_at_k": tuple(int(k) for k in args.map_at_k),
        "map_at_k_denominator": str(args.map_at_k_denominator),
        "query_chunk_size": int(args.query_chunk_size),
        "device": device,
    }
    val_base_evaluation = evaluate_base_queries(val_features, val_gallery, **common_eval_args)
    test_base_evaluation = evaluate_base_queries(test_features, test_gallery, **common_eval_args)
    val_probe = transport_probe_dict(
        val_features,
        val_gallery,
        frozen_reference=val_reference,
        kappa_max=float(args.max_kappa),
    )
    test_probe = transport_probe_dict(
        test_features,
        test_gallery,
        frozen_reference=test_reference,
        kappa_max=float(args.max_kappa),
    )
    val_probe["drift"] = _drift_probe(val_features, initial_val)
    test_probe["drift"] = _drift_probe(test_features, initial_test)
    val_probe["transport"]["rho_correlations"] = transport_query_correlations(
        val_features,
        val_gallery,
        val_evaluations[selected_mode].average_precision_per_query,
    )
    test_probe["transport"]["rho_correlations"] = transport_query_correlations(
        test_features,
        test_gallery,
        test_evaluations[selected_mode].average_precision_per_query,
    )
    val_probe["hidden_space_compatibility"] = hidden_space_compatibility(
        val_hidden_reference,
        val_features.h,
        model.photo_projection,
    )
    test_probe["hidden_space_compatibility"] = hidden_space_compatibility(
        test_hidden_reference,
        test_features.h,
        model.photo_projection,
    )
    result = {
        "val": _metrics_summary(val_evaluations[selected_mode]),
        "base_val": _metrics_summary(val_base_evaluation),
        "diagnostic_test": _metrics_summary(test_evaluations[selected_mode]),
        "base_diagnostic_test": _metrics_summary(test_base_evaluation),
        "retrieval_modes": {
            "val": {name: _metrics_summary(value) for name, value in val_evaluations.items()},
            "diagnostic_test": {name: _metrics_summary(value) for name, value in test_evaluations.items()},
            "selected": selected_mode,
            "base_val": _metrics_summary(val_base_evaluation),
            "base_diagnostic_test": _metrics_summary(test_base_evaluation),
        },
        "radius_vs_ap": _radius_ap_payload(val_evaluations[selected_mode], val_features.rho),
        "val_geometry": val_probe,
        "diagnostic_test_geometry": test_probe,
        "protocol": {
            "val_is_pseudo_unseen": str(args.train_class_scope) == "pseudo_train",
            "validation_classes_seen_during_training": str(args.train_class_scope) == "all_seen",
            "official_test_is_diagnostic_only": True,
            "text_used_for_evaluation": False,
            "photo_gallery_reencoded": False,
            "map_at_k_denominator": str(args.map_at_k_denominator),
        },
    }
    return result, val_features, test_features


def _drift_probe(current: TransportFeatureSet, initial: TransportFeatureSet) -> dict[str, float]:
    if current.z0.shape != initial.z0.shape:
        raise ValueError("initial and current probe feature shapes must match")
    base_initial = (F.normalize(current.z0, dim=-1) * F.normalize(initial.z0, dim=-1)).sum(dim=-1)
    q_initial = (F.normalize(current.q, dim=-1) * F.normalize(initial.q, dim=-1)).sum(dim=-1)
    return {
        "base_initial_cosine": base_initial.mean().item(),
        "base_initial_cosine_std": base_initial.std(unbiased=False).item(),
        "query_initial_cosine": q_initial.mean().item(),
        "query_initial_cosine_std": q_initial.std(unbiased=False).item(),
    }


def _flatten_probe_metrics(probe: dict[str, object]) -> dict[str, float]:
    val = probe["val"]
    test = probe["diagnostic_test"]
    val_geometry = probe["val_geometry"]
    if not isinstance(val, dict) or not isinstance(test, dict) or not isinstance(val_geometry, dict):
        raise TypeError("malformed transport probe")
    base_val = probe.get("base_val", {})
    base_test = probe.get("base_diagnostic_test", {})
    metrics: dict[str, float] = {
        "pseudo_val_mAP": float(val["mAP"]),
        "diagnostic_test_mAP": float(test["mAP"]),
    }
    if isinstance(base_val, dict) and isinstance(base_val.get("mAP"), (int, float)):
        metrics["pseudo_val_base_mAP"] = float(base_val["mAP"])
        metrics["pseudo_val_transport_gain"] = metrics["pseudo_val_mAP"] - metrics["pseudo_val_base_mAP"]
    if isinstance(base_test, dict) and isinstance(base_test.get("mAP"), (int, float)):
        metrics["diagnostic_test_base_mAP"] = float(base_test["mAP"])
        metrics["diagnostic_test_transport_gain"] = metrics["diagnostic_test_mAP"] - metrics["diagnostic_test_base_mAP"]
    val_precision = val.get("precision_at_k", {})
    test_precision = test.get("precision_at_k", {})
    if isinstance(val_precision, dict) and 200 in val_precision:
        metrics["pseudo_val_P200"] = float(val_precision[200])
    if isinstance(test_precision, dict) and 200 in test_precision:
        metrics["diagnostic_test_P200"] = float(test_precision[200])
    transport = val_geometry.get("transport", {})
    mixture = val_geometry.get("mixture", {})
    q_geometry = val_geometry.get("q", {})
    semantic = val_geometry.get("semantic", {})
    reference = val_geometry.get("reference", {})
    drift = val_geometry.get("drift", {})
    hidden = val_geometry.get("hidden_space_compatibility", {})
    for source, names in (
        (transport, ("mean_rho", "std_rho", "p95_rho", "mean_direction_cosine", "mean_distance_error", "endpoint_photo_cosine", "base_photo_cosine")),
        (mixture, ("gate_entropy", "responsibility_entropy", "mean_kappa", "kappa_saturation_fraction")),
        (q_geometry, ("effective_rank",)),
        (semantic, ("semantic_margin",)),
        (reference, ("base_reference_cosine", "query_reference_cosine")),
        (drift, ("base_initial_cosine", "query_initial_cosine")),
        (hidden, ("linear_cka", "procrustes_residual", "frozen_projection_mean_cosine", "effective_rank_h_ref", "effective_rank_h_t", "effective_rank_W_h_t")),
    ):
        if isinstance(source, dict):
            for name in names:
                value = source.get(name)
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)
    if isinstance(mixture, dict):
        for prefix, key in (
            ("component_usage", "component_usage"),
            ("component_direction_cosine", "mean_direction_cosine_by_component"),
            ("component_pairwise_direction_cosine", "component_pairwise_direction_cosine"),
        ):
            values = mixture.get(key)
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, (int, float)):
                        metrics[f"{prefix}_{index}"] = float(value)
    angles = transport.get("target_angles", {}) if isinstance(transport, dict) else {}
    if isinstance(angles, dict):
        for frame in ("moving", "fixed"):
            summary = angles.get(frame, {})
            if isinstance(summary, dict):
                for name in ("mean_degrees", "std_degrees", "p05_degrees", "p25_degrees", "p50_degrees", "p75_degrees", "p95_degrees"):
                    value = summary.get(name)
                    if isinstance(value, (int, float)):
                        metrics[f"target_{frame}_{name}"] = float(value)
        for name in ("moving_target_alignment", "fixed_target_alignment", "target_frame_agreement", "fixed_tangent_destination_max_abs_dot", "rho_over_moving_theta"):
            value = angles.get(name)
            if isinstance(value, (int, float)):
                metrics[name] = float(value)
    return metrics


def run(args: DictConfig) -> None:
    _validate_options(args)
    seed = int(args.seed)
    _seed_everything(seed)
    device = _resolve_device(str(args.device))
    data = load_data_config(_resolve_project_path(args.data_config))
    split, all_class_names = _build_split(
        data,
        num_validation_classes=int(args.pseudo_val_num_classes),
        seed=int(args.pseudo_val_seed),
    )
    resolved_config_value = OmegaConf.to_container(args, resolve=True)
    if not isinstance(resolved_config_value, dict):
        raise TypeError("resolved Hydra config must be a dictionary")
    resolved_config = resolved_config_value
    data_split_payload = {
        "dataset": data.name,
        "pseudo_validation_seed": split.seed,
        "train_class_ids": list(split.train_class_ids),
        "validation_class_ids": list(split.validation_class_ids),
        "train_sketches": len(split.train_sketch_entries),
        "train_photos": len(split.train_photo_entries),
        "validation_sketches": len(split.validation_sketch_entries),
        "validation_photos": len(split.validation_photo_entries),
    }
    data_split_identity = {
        **data_split_payload,
        "sha256": hash_json(data_split_payload),
    }
    data_manifest_identity = split_manifest_identity(
        split,
        dataset_name=data.name,
        dataset_root=data.root,
        manifest_paths={
            "train_sketch": data.train.sketch_manifest,
            "train_photo": data.train.photo_manifest,
            "class_map": data.train.class_map,
        },
    )
    if str(args.train_class_scope) == "pseudo_train":
        train_sketch_entries = split.train_sketch_entries
        train_photo_entries = split.train_photo_entries
        train_class_ids = split.train_class_ids
    else:
        train_sketch_entries = tuple(read_manifest(data.train.sketch_manifest, data.root))
        train_photo_entries = tuple(read_manifest(data.train.photo_manifest, data.root))
        train_class_ids = tuple(sorted(all_class_names))
    train_class_names = {class_id: all_class_names[class_id] for class_id in train_class_ids}

    pretrained = None if args.pretrained is None else str(args.pretrained)
    print(f"Loading frozen photo CLIP {args.model_name} ({pretrained}) on {device}...")
    photo_clip = load_frozen_clip(
        model_name=str(args.model_name), pretrained=pretrained, device=device
    )
    sketch_bundle = load_trainable_sketch_hidden_encoder(
        model_name=str(args.model_name),
        pretrained=pretrained,
        device=device,
        mode=str(args.encoder_mode),
        unfreeze_depth=int(args.encoder_unfreeze_depth),
    )
    projection = frozen_visual_projection(photo_clip.encoder).to(device)
    rho_max = math.radians(float(args.rho_max_degrees))
    model = SpicaPredictiveTransport(
        sketch_bundle.encoder,
        projection,
        transport_mode=str(args.transport_mode),
        predictor_hidden_dim=int(args.predictor_hidden_dim),
        num_components=int(args.K),
        use_z0=bool(args.use_z0),
        alpha=float(args.alpha),
        alpha_max=float(args.alpha_max),
        initial_alpha=float(args.initial_alpha),
        rho_max=rho_max,
        initial_rho=math.radians(float(args.initial_rho_degrees)),
        shared_rho=str(args.rho_mode) == "shared",
        rho_strategy=str(args.rho_strategy),
        fixed_rho=math.radians(float(args.fixed_rho_degrees)),
        rho_warmup_steps=int(args.rho_warmup_steps),
        use_vmf=bool(args.use_vmf),
        transport_enabled=bool(args.transport_enabled),
        min_kappa=float(args.min_kappa),
        max_kappa=float(args.max_kappa),
        initial_kappa=float(args.initial_kappa),
    ).to(device)
    model.train()

    train_loader = _build_train_loader(
        train_sketch_entries, train_photo_entries, sketch_bundle.transform, args, seed=seed
    )
    if len(train_loader) == 0:
        raise ValueError("training loader has no batches")
    val_loader = _build_eval_loader(split.validation_sketch_entries, sketch_bundle.transform, args)
    val_photo_loader = _build_eval_loader(split.validation_photo_entries, photo_clip.transform, args)
    test_entries = read_manifest(data.test.sketch_manifest, data.root)
    test_loader = _build_eval_loader(test_entries, sketch_bundle.transform, args)
    print("Encoding pseudo-unseen validation gallery with frozen photo CLIP...")
    val_gallery = encode_retrieval_loader(photo_clip.encoder, val_photo_loader)
    test_gallery = load_encoded_retrieval_set(_resolve_project_path(args.embedding_dir) / "photos.pt")

    # Reference features are encoded in loader order once, only for probes and
    # geometry-loss targets; they are never passed to the model.
    def encode_reference_loader(loader: DataLoader) -> Tensor:
        batches: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batches.append(photo_clip.encoder(batch["image"].to(device)).float().cpu())
        return torch.cat(batches, dim=0)

    def encode_hidden_reference_loader(loader: DataLoader) -> Tensor:
        batches: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                hidden = visual_pre_projection(
                    photo_clip.encoder.model.visual,
                    batch["image"].to(device),
                )
                batches.append(hidden.float().cpu())
        return torch.cat(batches, dim=0)

    val_reference = encode_reference_loader(val_loader)
    test_reference = encode_reference_loader(test_loader)
    val_hidden_reference = encode_hidden_reference_loader(val_loader)
    test_hidden_reference = encode_hidden_reference_loader(test_loader)

    prototype_labels: Tensor | None = None
    prototypes: Tensor | None = None
    if str(args.photo_target) == "class_prototype" or str(args.direction_target) == "class_centroid":
        print("Building train-photo-only class prototypes...")
        prototype_labels, prototypes = _build_class_prototypes(
            photo_clip.encoder, train_photo_entries, photo_clip.transform, args
        )

    text_bank: EncodedTextBank | None = None
    if float(args.lambda_cls) > 0 and str(args.text_loss_location) != "none":
        text_bank = encode_class_text_bank(
            photo_clip.encoder,
            photo_clip.tokenizer,
            train_class_names,
            prompt_template=str(args.prompt_template),
        )
        text_bank_path = Path(HydraConfig.get().runtime.output_dir) / "seen_text_bank.pt"
        text_bank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": text_bank.embeddings,
                "labels": text_bank.labels,
                "class_names": text_bank.class_names,
                "prompts": text_bank.prompts,
                "frozen": True,
                "used_only_as_training_classifier": True,
            },
            text_bank_path,
        )
        print("Text enters predictor: NO (frozen classification bank is loss-only).")
    text_embeddings = None if text_bank is None else text_bank.embeddings.to(device)
    text_labels = None if text_bank is None else text_bank.labels.to(device)

    predictor_parameters = [p for p in model.transport_head.parameters() if p.requires_grad]
    encoder_parameters = [p for p in model.sketch_context_encoder.parameters() if p.requires_grad]
    parameter_groups = [{"params": predictor_parameters, "lr": float(args.predictor_learning_rate)}]
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": float(args.encoder_learning_rate)})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(args.weight_decay))
    resume_step = 0
    payload: dict[str, Any] | None = None
    stage1_selection: dict[str, Any] | None = None
    resume_path = None if args.resume_checkpoint_path is None else _resolve_project_path(args.resume_checkpoint_path)
    if resume_path is not None:
        loaded = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not isinstance(
            loaded.get("model_state_dict"), dict
        ):
            raise ValueError("resume checkpoint is not a predictive transport checkpoint")
        payload = loaded
        branch_roles = {
            "freeze_optimizer_A",
            "freeze_optimizer_B",
            "freeze_optimizer_C",
            "freeze_optimizer_D",
        }
        if str(args.experiment_role) in branch_roles:
            if int(payload.get("step", -1)) != 73:
                raise ValueError(
                    "freeze × optimizer branches must fork the same step-73 checkpoint"
                )
            source_metadata = payload.get("metadata")
            if not isinstance(source_metadata, dict):
                raise ValueError("freeze × optimizer source metadata is missing")
            if source_metadata.get("experiment_role") != "freeze_optimizer_source":
                raise ValueError("freeze × optimizer branches must fork a semantic source checkpoint")
            if source_metadata.get("experiment_campaign") != args.experiment_campaign:
                raise ValueError("freeze × optimizer source campaign differs from branch")
            if source_metadata.get("data_split_identity") != data_split_identity:
                raise ValueError("freeze × optimizer source split differs from branch")
            source_manifest_identity = payload.get("data_manifest_identity")
            if source_manifest_identity is not None and source_manifest_identity != data_manifest_identity:
                raise ValueError("freeze × optimizer source entries differ from branch")
            if payload.get("optimizer_state_dict") is None:
                raise ValueError("freeze × optimizer branches require the source optimizer state")
            source_config = payload.get("resolved_config")
            if not isinstance(source_config, dict) or source_config.get("train_class_scope") != "pseudo_train":
                raise ValueError("freeze × optimizer source must use pseudo-train classes")
        fresh_head = (
            _cpu_state_dict(model.transport_head)
            if bool(args.reset_transport_head_on_resume)
            else None
        )
        scope = str(args.optimizer_reset_scope)
        restore_optimizer = scope != "all" and not bool(args.reset_optimizer_on_resume)
        resume_step = _restore_training_state(
            payload,
            model=model,
            optimizer=optimizer,
            restore_optimizer=restore_optimizer,
            restore_rng=True,
            generator=train_loader.generator,
        )
        if resume_step > int(args.max_steps):
            raise ValueError("resume checkpoint step cannot exceed max_steps")
        if fresh_head is not None:
            model.transport_head.load_state_dict(fresh_head, strict=True)
        if restore_optimizer and scope in {"encoder", "head"}:
            reset_parameters = (
                model.sketch_context_encoder.parameters()
                if scope == "encoder"
                else model.transport_head.parameters()
            )
            for parameter in reset_parameters:
                optimizer.state.pop(parameter, None)
        if str(args.experiment_role) in {
            "two_stage_S1",
            "two_stage_S2",
            "two_stage_S3",
            "two_stage_S4",
        }:
            stage1_selection = _load_stage1_selection(
                _resolve_project_path(args.stage1_selection_manifest_path),
                resume_path=resume_path,
                payload=payload,
                data_split_identity=data_split_identity,
                expected_campaign=str(args.experiment_campaign),
                expected_manifest_identity=data_manifest_identity,
            )
        print(
            f"Resumed transport checkpoint at step {resume_step}: {resume_path} (optimizer reset scope={scope})"
        )
    freeze_step = None if args.freeze_encoder_at_step is None else int(args.freeze_encoder_at_step)
    optimizer, effective_freeze_step, freeze_applied_before_first_step = _apply_resume_controls(
        model,
        optimizer,
        resume_step=resume_step,
        freeze_encoder_on_resume=bool(args.freeze_encoder_on_resume) and resume_path is not None,
        freeze_encoder_at_step=freeze_step,
    )
    if freeze_applied_before_first_step:
        print(f"Sketch encoder frozen before first update at resumed step {resume_step}")
    model.set_schedule_step(resume_step)
    parameter_counts = _parameter_counts(model, photo_clip.encoder)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equivalent_epochs_per_step = int(args.batch_size) / len(train_loader.dataset)
    effective_probe_steps = _effective_probe_steps(
        resume_step=resume_step,
        absolute_steps=args.probe_steps,
        relative_offsets=args.probe_step_offsets,
        max_steps=int(args.max_steps),
        run_probes=bool(args.run_probes),
    )
    optimizer_state_restored = bool(
        resume_path is not None
        and payload is not None
        and payload.get("optimizer_state_dict") is not None
        and str(args.optimizer_reset_scope) != "all"
        and not bool(args.reset_optimizer_on_resume)
    )
    optimizer_state_reset = bool(
        resume_path is not None
        and (bool(args.reset_optimizer_on_resume) or str(args.optimizer_reset_scope) in {"encoder", "head"})
    )
    continuation_metadata: dict[str, Any] = {
        "resumed": resume_path is not None,
        "source_checkpoint": None if resume_path is None else str(resume_path),
        "starting_step": resume_step,
        "steps_since_resume_definition": "global_step - starting_step",
        "freeze_encoder_on_resume": bool(args.freeze_encoder_on_resume),
        "freeze_applied_before_first_update": freeze_applied_before_first_step,
        "effective_freeze_step": effective_freeze_step,
        "encoder_frozen_at_start": not any(
            parameter.requires_grad for parameter in model.sketch_context_encoder.parameters()
        ),
        "transport_head_reinitialized": bool(args.reset_transport_head_on_resume),
        "optimizer_state_restored": optimizer_state_restored,
        "optimizer_state_reset": optimizer_state_reset,
        "optimizer_reset_scope": str(args.optimizer_reset_scope),
        "probe_step_offsets": None
        if args.probe_step_offsets is None
        else [int(value) for value in args.probe_step_offsets],
        "effective_probe_steps": effective_probe_steps,
        "stage1_selection": stage1_selection,
    }
    provenance = _git_provenance(resolved_config)
    run_config: dict[str, Any] = {
        "experiment_role": None
        if args.experiment_role is None
        else str(args.experiment_role),
        "experiment_campaign": None
        if args.experiment_campaign is None
        else str(args.experiment_campaign),
        "model_name": str(args.model_name),
        "pretrained": None if args.pretrained is None else str(args.pretrained),
        "data_config": str(args.data_config),
        "model_family": "predictive_semantic_transport",
        "objective": OBJECTIVE_NAME,
        "transport_mode": str(args.transport_mode),
        "transport_enabled": bool(args.transport_enabled),
        "encoder_mode": str(args.encoder_mode),
        "encoder_unfreeze_depth": int(args.encoder_unfreeze_depth),
        "use_z0": bool(args.use_z0),
        "unfrozen_block_count": int(args.encoder_unfreeze_depth)
        if str(args.encoder_mode) == "partial"
        else (0 if str(args.encoder_mode) == "frozen" else "all"),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "drop_last": bool(args.drop_last),
        "tau_cls": float(args.tau_cls),
        "prompt_template": str(args.prompt_template),
        "scheduler": "none",
        "weight_decay": float(args.weight_decay),
        "margin": float(args.margin),
        "K": int(args.K),
        "shared_or_component_rho": str(args.rho_mode),
        "rho_strategy": str(args.rho_strategy),
        "rho_max": float(args.rho_max_degrees),
        "fixed_rho_degrees": float(args.fixed_rho_degrees),
        "rho_warmup_steps": int(args.rho_warmup_steps),
        "use_vmf": bool(args.use_vmf),
        "use_text_cls": float(args.lambda_cls) > 0
        and str(args.text_loss_location) != "none",
        "text_loss_location": str(args.text_loss_location),
        "direction_target": str(args.direction_target),
        "use_geometry_loss": bool(args.use_geometry_loss),
        "photo_target": str(args.photo_target),
        "loss_profile": str(args.loss_profile),
        "lambda_dir": float(args.lambda_dir),
        "lambda_dist": float(args.lambda_dist),
        "lambda_endpoint": float(args.lambda_endpoint),
        "lambda_rank": float(args.lambda_rank),
        "lambda_cls": float(args.lambda_cls),
        "lambda_vmf": float(args.lambda_vmf),
        "lambda_geom": float(args.lambda_geom),
        "predictor_lr": float(args.predictor_learning_rate),
        "encoder_lr": float(args.encoder_learning_rate),
        "predictor_learning_rate": float(args.predictor_learning_rate),
        "encoder_learning_rate": float(args.encoder_learning_rate),
        "resume_checkpoint_path": None
        if args.resume_checkpoint_path is None
        else str(args.resume_checkpoint_path),
        "reset_optimizer_on_resume": bool(args.reset_optimizer_on_resume),
        "optimizer_reset_scope": str(args.optimizer_reset_scope),
        "reset_transport_head_on_resume": bool(args.reset_transport_head_on_resume),
        "freeze_encoder_on_resume": bool(args.freeze_encoder_on_resume),
        "freeze_encoder_at_step": None
        if args.freeze_encoder_at_step is None
        else int(args.freeze_encoder_at_step),
        "effective_freeze_step": effective_freeze_step,
        "stage1_selection_manifest_path": None
        if args.stage1_selection_manifest_path is None
        else str(args.stage1_selection_manifest_path),
        "gradient_conflict_steps": [
            int(value) for value in args.gradient_conflict_steps
        ],
        "seed": seed,
        "steps": int(args.max_steps),
        "equivalent_epochs": int(args.max_steps) * equivalent_epochs_per_step,
        "num_positive_photos": int(args.num_positive_photos),
        "rho_max_degrees": float(args.rho_max_degrees),
        "inference_score_mode": str(args.inference_score_mode),
        "score_temperature": float(args.score_temperature),
        "train_class_scope": str(args.train_class_scope),
        "pseudo_validation_seed": split.seed,
        "pseudo_train_classes": len(split.train_class_ids),
        "pseudo_validation_classes": len(split.validation_class_ids),
        "probe_steps": [int(value) for value in args.probe_steps],
        "probe_step_offsets": None
        if args.probe_step_offsets is None
        else [int(value) for value in args.probe_step_offsets],
        "effective_probe_steps": effective_probe_steps,
        "map_at_k_denominator": str(args.map_at_k_denominator),
        "text_conditioning": False,
        "text_enters_predictor": False,
        **parameter_counts,
        "wandb_project": str(args.wandb_project),
        "wandb_group": None if args.wandb_group is None else str(args.wandb_group),
        "wandb_mode": str(args.wandb_mode),
        "provenance": provenance,
    }
    initial_parameters = _capture_initial_parameters(model)
    probe_steps = set(effective_probe_steps)
    history: list[dict[str, object]] = []
    training_history: list[dict[str, object]] = []
    experiment_mode = str(args.wandb_mode)
    non_blocking = device.type == "cuda" and bool(args.pin_memory)

    with WandbExperiment(
        project=str(args.wandb_project),
        entity=None if args.wandb_entity is None else str(args.wandb_entity),
        group=None if args.wandb_group is None else str(args.wandb_group),
        name=None if args.wandb_run_name is None else str(args.wandb_run_name),
        config=run_config,
        tags=(
            "spica",
            "predictive-semantic-transport",
            str(args.transport_mode),
            f"K{args.K}",
        ),
        mode=experiment_mode,
        job_type="transport-training",
        directory=output_dir,
    ) as experiment:
        if experiment_mode != "disabled":
            print(f"W&B run: {experiment.run_url}")
        print(
            f"Training Predictive Semantic Transport: mode={args.transport_mode}, "
            f"K={args.K}, encoder={args.encoder_mode}, steps={args.max_steps}"
        )

        def record_probe(
            global_step: int,
            *,
            training_losses: dict[str, float] | None = None,
            origin: str = "scheduled",
        ) -> dict[str, object]:
            """Save one globally named probe with explicit resume coordinates."""
            checkpoint_path = output_dir / "checkpoints" / f"transport_step{global_step}.pt"
            _save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=global_step,
                data_name=data.name,
                args=args,
                split=split,
                parameter_counts=parameter_counts,
                provenance=provenance,
                resolved_config=resolved_config,
                data_split_identity=data_split_identity,
                data_manifest_identity=data_manifest_identity,
                generator=train_loader.generator,
                include_optimizer=bool(args.save_optimizer),
                continuation_metadata=continuation_metadata,
            )
            model.set_schedule_step(global_step)
            if initial_val is None or initial_test is None:
                raise RuntimeError("probe reference features are not initialized")
            probe, _, _ = _probe(
                model=model,
                val_loader=val_loader,
                val_gallery=val_gallery,
                test_loader=test_loader,
                test_gallery=test_gallery,
                val_reference=val_reference,
                test_reference=test_reference,
                val_hidden_reference=val_hidden_reference,
                test_hidden_reference=test_hidden_reference,
                initial_val=initial_val,
                initial_test=initial_test,
                device=device,
                args=args,
            )
            probe.update(
                {
                    "step": global_step,
                    "global_step": global_step,
                    "steps_since_resume": global_step - resume_step,
                    "equivalent_epochs": global_step * equivalent_epochs_per_step,
                    "checkpoint": str(checkpoint_path),
                    "probe_origin": origin,
                    "parameter_drift": _parameter_drift(model, initial_parameters),
                    "training_losses_at_checkpoint": training_losses or {},
                }
            )
            (output_dir / f"probe_step{global_step}.json").write_text(
                json.dumps(probe, indent=2, sort_keys=True) + "\n"
            )
            probe_metrics = _flatten_probe_metrics(probe)
            probe_metrics.update(
                {
                    "effective_rank": probe_metrics.get("effective_rank", 0.0),
                    "mean_rho": probe_metrics.get("mean_rho", 0.0),
                    "std_rho": probe_metrics.get("std_rho", 0.0),
                    "p95_rho": probe_metrics.get("p95_rho", 0.0),
                    "mean_direction_cosine": probe_metrics.get("mean_direction_cosine", 0.0),
                    "endpoint_photo_cosine": probe_metrics.get("endpoint_photo_cosine", 0.0),
                    "base_photo_cosine": probe_metrics.get("base_photo_cosine", 0.0),
                    "semantic_margin": probe_metrics.get("semantic_margin", 0.0),
                    "base_reference_cosine": probe_metrics.get("base_reference_cosine", 0.0),
                    "query_reference_cosine": probe_metrics.get("query_reference_cosine", 0.0),
                    "mean_kappa": probe_metrics.get("mean_kappa", 0.0),
                    "gate_entropy": probe_metrics.get("gate_entropy", 0.0),
                    "responsibility_entropy": probe_metrics.get("responsibility_entropy", 0.0),
                    "global_step": global_step,
                    "steps_since_resume": global_step - resume_step,
                }
            )
            experiment.log_metrics(
                {**probe_metrics, "step": global_step}, step=global_step
            )
            print(
                f"  pseudo_val_mAP={probe_metrics['pseudo_val_mAP']:.6f} "
                f"test_mAP={probe_metrics['diagnostic_test_mAP']:.6f} "
                f"mean_rho={probe_metrics['mean_rho']:.5f} "
                f"direction={probe_metrics['mean_direction_cosine']:.4f}"
            )
            return probe

        # Step zero is a mandatory pretrained-origin evaluation.
        initial_val: TransportFeatureSet | None = None
        initial_test: TransportFeatureSet | None = None
        if 0 in probe_steps:
            # Avoid a special-case in _probe's drift calculation: encode once,
            # then use that same feature set as the initial reference.
            initial_val = encode_transport_loader(model, val_loader, device=device)
            initial_test = encode_transport_loader(model, test_loader, device=device)
            modes = ("barycentric", "angular_logsumexp", "max") if int(args.K) > 1 else ("barycentric",)
            val_evals = evaluate_transport_features(initial_val, val_gallery, modes=modes, temperature=float(args.score_temperature), precision_at_k=tuple(int(k) for k in args.precision_at_k), map_at_k=tuple(int(k) for k in args.map_at_k), map_at_k_denominator=str(args.map_at_k_denominator), query_chunk_size=int(args.query_chunk_size), device=device)
            test_evals = evaluate_transport_features(initial_test, test_gallery, modes=modes, temperature=float(args.score_temperature), precision_at_k=tuple(int(k) for k in args.precision_at_k), map_at_k=tuple(int(k) for k in args.map_at_k), map_at_k_denominator=str(args.map_at_k_denominator), query_chunk_size=int(args.query_chunk_size), device=device)
            val_base = evaluate_base_queries(initial_val, val_gallery, temperature=float(args.score_temperature), precision_at_k=tuple(int(k) for k in args.precision_at_k), map_at_k=tuple(int(k) for k in args.map_at_k), map_at_k_denominator=str(args.map_at_k_denominator), query_chunk_size=int(args.query_chunk_size), device=device)
            test_base = evaluate_base_queries(initial_test, test_gallery, temperature=float(args.score_temperature), precision_at_k=tuple(int(k) for k in args.precision_at_k), map_at_k=tuple(int(k) for k in args.map_at_k), map_at_k_denominator=str(args.map_at_k_denominator), query_chunk_size=int(args.query_chunk_size), device=device)
            selected = str(args.inference_score_mode)
            step0_probe: dict[str, object] = {
                "val": _metrics_summary(val_evals[selected]),
                "base_val": _metrics_summary(val_base),
                "diagnostic_test": _metrics_summary(test_evals[selected]),
                "base_diagnostic_test": _metrics_summary(test_base),
                "retrieval_modes": {"val": {k: _metrics_summary(v) for k, v in val_evals.items()}, "diagnostic_test": {k: _metrics_summary(v) for k, v in test_evals.items()}, "base_val": _metrics_summary(val_base), "base_diagnostic_test": _metrics_summary(test_base), "selected": selected},
                "radius_vs_ap": _radius_ap_payload(val_evals[selected], initial_val.rho),
                "val_geometry": transport_probe_dict(initial_val, val_gallery, frozen_reference=val_reference, kappa_max=float(args.max_kappa)),
                "diagnostic_test_geometry": transport_probe_dict(initial_test, test_gallery, frozen_reference=test_reference, kappa_max=float(args.max_kappa)),
                "hidden_space_compatibility": {
                    "val": hidden_space_compatibility(val_hidden_reference, initial_val.h, model.photo_projection),
                    "diagnostic_test": hidden_space_compatibility(test_hidden_reference, initial_test.h, model.photo_projection),
                },
                "protocol": {"val_is_pseudo_unseen": str(args.train_class_scope) == "pseudo_train", "official_test_is_diagnostic_only": True, "text_used_for_evaluation": False, "photo_gallery_reencoded": False, "map_at_k_denominator": str(args.map_at_k_denominator)},
                "step": 0,
                "global_step": 0,
                "steps_since_resume": 0,
                "probe_origin": "initial",
                "equivalent_epochs": 0.0,
                "checkpoint": None,
                "parameter_drift": {},
                "training_losses_at_checkpoint": {},
            }
            step0_probe["val_geometry"]["drift"] = {"base_initial_cosine": 1.0, "query_initial_cosine": 1.0}  # type: ignore[index]
            step0_probe["diagnostic_test_geometry"]["drift"] = {"base_initial_cosine": 1.0, "query_initial_cosine": 1.0}  # type: ignore[index]
            step0_probe["val_geometry"]["transport"]["rho_correlations"] = transport_query_correlations(  # type: ignore[index]
                initial_val, val_gallery, val_evals[selected].average_precision_per_query
            )
            step0_probe["diagnostic_test_geometry"]["transport"]["rho_correlations"] = transport_query_correlations(  # type: ignore[index]
                initial_test, test_gallery, test_evals[selected].average_precision_per_query
            )
            history.append(step0_probe)
            (output_dir / "probe_step0.json").write_text(
                json.dumps(step0_probe, indent=2, sort_keys=True) + "\n"
            )
            experiment.log_metrics({**_flatten_probe_metrics(step0_probe), "step": 0, "equivalent_epochs": 0.0}, step=0)
            print(f"step=0000 pseudo_val_mAP={float(step0_probe['val']['mAP']):.6f}")  # type: ignore[index]

        if initial_val is None or initial_test is None:
            initial_val = encode_transport_loader(model, val_loader, device=device)
            initial_test = encode_transport_loader(model, test_loader, device=device)

        if resume_path is not None and resume_step in probe_steps:
            history.append(record_probe(resume_step, origin="resume_boundary"))

        step = resume_step
        window: dict[str, float] = {
            "loss_total": 0.0,
            "loss_dir": 0.0,
            "loss_dist": 0.0,
            "loss_endpoint": 0.0,
            "loss_rank": 0.0,
            "loss_cls": 0.0,
            "loss_geom": 0.0,
            "loss_vmf": 0.0,
            "train_direction_cosine": 0.0,
            "train_endpoint_photo_cosine": 0.0,
            "classification_accuracy": 0.0,
            "classification_accuracy_q": 0.0,
            "classification_accuracy_z0": 0.0,
        }
        window_count = 0
        positive_paths_seen: set[str] = set()
        training_angle_values: list[float] = []
        training_angle_fixed_values: list[float] = []
        gradient_conflicts: list[dict[str, object]] = []
        gradient_conflict_steps = {int(value) for value in args.gradient_conflict_steps}
        while step < int(args.max_steps):
            for batch in train_loader:
                sketch_images = batch["sketch"].to(device, non_blocking=non_blocking)
                positive_images = batch["positive_photos"].to(device, non_blocking=non_blocking)
                negative_images = batch["negative_photo"].to(device, non_blocking=non_blocking)
                labels = batch["label"].to(device=device, dtype=torch.long)
                positive_embeddings, negative_embedding = _encode_photo_targets(photo_clip.encoder, positive_images, negative_images)
                target_photo = _target_for_labels(
                    positive_embeddings,
                    labels,
                    photo_target=str(args.photo_target),
                    prototype_labels=prototype_labels,
                    prototypes=prototypes,
                )
                # P4 records the true sampled sketch/photo angle, regardless
                # of whether the optimization target is an instance or a
                # class prototype.  The frozen reference is also the target
                # frame for the fixed-origin direction control.
                sketch_reference = _encode_reference(photo_clip.encoder, sketch_images)
                model.train()
                if not any(parameter.requires_grad for parameter in model.sketch_context_encoder.parameters()):
                    model.sketch_context_encoder.eval()
                optimizer.zero_grad(set_to_none=True)
                model.set_schedule_step(step)
                prediction: SemanticTransportPrediction = model(sketch_images)
                # The base controls use the same encoder and ranking harness,
                # but q is exactly z0 and all transport-specific losses are
                # disabled.  This prevents an unused head from becoming a
                # hidden source of gradients in the factorial experiment.
                query = prediction.q if bool(args.transport_enabled) else prediction.z0
                target_transport = photo_transport_target(prediction.z0, target_photo)
                if str(args.direction_target) == "class_centroid":
                    direction_photo = _target_for_labels(
                        positive_embeddings,
                        labels,
                        photo_target="class_prototype",
                        prototype_labels=prototype_labels,
                        prototypes=prototypes,
                    )
                else:
                    direction_photo = target_photo
                if str(args.direction_target) == "fixed_reference":
                    direction_transport = fixed_origin_transport_target(
                        sketch_reference,
                        direction_photo,
                        prediction.z0,
                    )
                else:
                    direction_transport = photo_transport_target(
                        prediction.z0,
                        direction_photo,
                    )
                positive_targets_for_direction = (
                    target_photo[:, None, :]
                    if str(args.photo_target) == "class_prototype"
                    else positive_embeddings
                )
                if len(training_angle_values) < int(args.training_angle_diagnostic_max):
                    training_angle_values.extend(
                        _angle_values(
                            prediction.z0,
                            positive_embeddings,
                            max_items=int(args.training_angle_diagnostic_max) - len(training_angle_values),
                        ).tolist()
                    )
                    training_angle_fixed_values.extend(
                        _angle_values(
                            sketch_reference,
                            positive_embeddings,
                            max_items=int(args.training_angle_diagnostic_max) - len(training_angle_fixed_values),
                        ).tolist()
                    )
                mixture = None
                if not bool(args.transport_enabled):
                    loss_dir = query.new_zeros(())
                    loss_dist = query.new_zeros(())
                    loss_endpoint = query.new_zeros(())
                    loss_rank = transport_ranking_loss(query, target_photo, negative_embedding, margin=float(args.margin))
                elif int(args.K) == 1:
                    loss_dir = (
                        query.new_zeros(())
                        if str(args.direction_target) == "none"
                        else transport_direction_loss(
                            prediction.direction, direction_transport.direction
                        )
                    )
                    loss_rank = transport_ranking_loss(query, target_photo, negative_embedding, margin=float(args.margin))
                    rho_for_loss = prediction.rho if prediction.rho.ndim == 1 else prediction.rho.mean(dim=-1)
                    loss_dist = transport_distance_loss(rho_for_loss, target_transport.theta)
                    loss_endpoint = transport_endpoint_loss(query, target_photo)
                else:
                    if bool(args.use_vmf):
                        mixture = directional_mixture_loss(
                            prediction,
                            direction_transport.direction,
                            positive_targets_for_direction,
                            negative_embedding,
                            margin=float(args.margin),
                            ranking_weight=1.0,
                            direction_nll_weight=1.0,
                            assignment_temperature=float(args.assignment_temperature),
                        )
                        # Mo-vMF direction likelihood is weighted explicitly by
                        # lambda_vmf below; ranking remains a separate term.
                        loss_dir = query.new_zeros(())
                    else:
                        mixture = deterministic_direction_mixture_loss(
                            prediction,
                            direction_transport.direction,
                            positive_targets_for_direction,
                            negative_embedding,
                            margin=float(args.margin),
                            ranking_weight=1.0,
                            direction_weight=0.0 if str(args.direction_target) == "none" else 1.0,
                            assignment_temperature=float(args.assignment_temperature),
                        )
                        loss_dir = mixture.direction_nll
                    loss_rank = mixture.ranking
                    rho_for_loss = prediction.rho if prediction.rho.ndim == 1 else prediction.rho.mean(dim=-1)
                    loss_dist = transport_distance_loss(rho_for_loss, target_transport.theta)
                    loss_endpoint = transport_endpoint_loss(query, target_photo)
                loss_cls = query.new_zeros(())
                cls_accuracy = query.new_zeros(())
                cls_accuracy_q = query.new_zeros(())
                cls_accuracy_z0 = query.new_zeros(())
                if text_embeddings is not None and text_labels is not None:
                    location = str(args.text_loss_location)
                    if location in {"q", "both"}:
                        loss_cls_q, logits_q = jepa_text_classification_loss(
                            query,
                            text_embeddings,
                            text_labels,
                            labels,
                            temperature=float(args.tau_cls),
                        )
                        cls_accuracy_q = classification_accuracy(logits_q, text_labels, labels)
                    else:
                        loss_cls_q = query.new_zeros(())
                    if location in {"z0", "both"}:
                        loss_cls_z0, logits_z0 = jepa_text_classification_loss(
                            prediction.z0,
                            text_embeddings,
                            text_labels,
                            labels,
                            temperature=float(args.tau_cls),
                        )
                        cls_accuracy_z0 = classification_accuracy(logits_z0, text_labels, labels)
                    else:
                        loss_cls_z0 = query.new_zeros(())
                    if location == "q":
                        loss_cls, cls_accuracy = loss_cls_q, cls_accuracy_q
                    elif location == "z0":
                        loss_cls, cls_accuracy = loss_cls_z0, cls_accuracy_z0
                    elif location == "both":
                        # Equal total text-loss scale: 0.5 CE on each anchor.
                        loss_cls = 0.5 * (loss_cls_q + loss_cls_z0)
                        cls_accuracy = 0.5 * (cls_accuracy_q + cls_accuracy_z0)
                loss_geom = prediction.q.new_zeros(())
                if bool(args.use_geometry_loss):
                    loss_geom = transport_geometry_loss(query, sketch_reference)
                loss_vmf = query.new_zeros(())
                if bool(args.use_vmf):
                    if mixture is None:
                        raise RuntimeError("Mo-vMF loss was not constructed for a K>1 run")
                    loss_vmf = mixture.direction_nll
                dir_weight = float(args.lambda_dir)
                dist_weight = float(args.lambda_dist)
                if str(args.loss_profile) == "endpoint_rank":
                    dir_weight = 0.0
                    dist_weight = 0.0
                gradient_conflict: dict[str, object] | None = None
                if step + 1 in gradient_conflict_steps and bool(args.transport_enabled):
                    conflict_parameters = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    diagnostic_losses = {
                        "L_cls": loss_cls,
                        "L_rank": loss_rank,
                        "L_dir": loss_dir,
                        "L_endpoint": loss_endpoint,
                    }
                    batch_identity = {
                        "sketch_paths": [
                            str(value) for value in batch.get("sketch_path", ())
                        ],
                        "positive_photo_paths": [
                            [str(path) for path in values]
                            for values in batch.get("positive_photo_paths", ())
                        ],
                        "negative_photo_paths": [
                            str(value) for value in batch.get("negative_photo_path", ())
                        ],
                        "labels": labels.detach().cpu().tolist(),
                    }
                    gradient_conflict = {
                        "checkpoint": step + 1,
                        "step": step + 1,
                        "global_step": step + 1,
                        "steps_since_resume": step + 1 - resume_step,
                        "seed": seed,
                        "batch_identity": {
                            **batch_identity,
                            "sha256": hash_json(batch_identity),
                        },
                        "loss_names": list(diagnostic_losses),
                        "spaces": {
                            **_representation_gradient_conflicts(
                                diagnostic_losses, prediction
                            ),
                            "parameter": _parameter_gradient_space(
                                diagnostic_losses, conflict_parameters
                            ),
                        },
                        "optimizer_application": {
                            "L_cls": float(args.lambda_cls),
                            "L_rank": float(args.lambda_rank),
                            "L_dir": dir_weight,
                            "L_endpoint": float(args.lambda_endpoint),
                        },
                        "optimized_loss_names": [
                            name
                            for name, weight in {
                                "L_cls": float(args.lambda_cls),
                                "L_rank": float(args.lambda_rank),
                                "L_dir": dir_weight,
                                "L_endpoint": float(args.lambda_endpoint),
                            }.items()
                            if weight != 0.0
                        ],
                        "diagnostic_only": [
                            name
                            for name, weight in {
                                "L_cls": float(args.lambda_cls),
                                "L_rank": float(args.lambda_rank),
                                "L_dir": dir_weight,
                                "L_endpoint": float(args.lambda_endpoint),
                            }.items()
                            if weight == 0.0
                        ],
                    }
                loss_total = (
                    dir_weight * loss_dir
                    + dist_weight * loss_dist
                    + float(args.lambda_endpoint) * loss_endpoint
                    + float(args.lambda_rank) * loss_rank
                    + float(args.lambda_cls) * loss_cls
                    + float(args.lambda_vmf) * loss_vmf
                    + float(args.lambda_geom) * loss_geom
                )
                values_to_check = {"loss_total": loss_total, "loss_dir": loss_dir, "loss_dist": loss_dist, "loss_endpoint": loss_endpoint, "loss_rank": loss_rank, "loss_cls": loss_cls, "loss_geom": loss_geom, "loss_vmf": loss_vmf, "prediction_q": prediction.q, "prediction_z0": prediction.z0}
                for name, value in values_to_check.items():
                    if not torch.isfinite(value).all().item():
                        raise FloatingPointError(f"{name} contains non-finite values")
                loss_total.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        if parameter.grad is None:
                            raise RuntimeError(f"Trainable parameter has no gradient: {name}")
                        if not torch.isfinite(parameter.grad).all().item():
                            raise FloatingPointError(f"gradient {name} contains non-finite values")
                    elif parameter.grad is not None:
                        raise RuntimeError(f"Frozen parameter received a gradient: {name}")
                optimizer.step()
                if freeze_step is not None and step + 1 == freeze_step:
                    _freeze_encoder(model)
                    optimizer = _rebuild_optimizer_without_encoder(optimizer, model)
                    print(f"Sketch encoder frozen at step {freeze_step}")
                for name, parameter in model.named_parameters():
                    if not torch.isfinite(parameter).all().item():
                        raise FloatingPointError(f"parameter {name} contains non-finite values")
                with torch.no_grad():
                    train_direction = ((prediction.direction * direction_transport.direction).sum(dim=-1).mean() if bool(args.transport_enabled) and int(args.K) == 1 else (prediction.directions * direction_transport.direction[:, None, :]).sum(dim=-1).max(dim=-1).values.mean() if bool(args.transport_enabled) else query.new_zeros(()))
                    train_endpoint = (query * target_transport.z_photo).sum(dim=-1).mean()
                # The dataset samples positive photos afresh in __getitem__; keep
                # an explicit diversity counter so M=1 is not mistaken for a
                # fixed target experiment.
                raw_paths = batch.get("positive_photo_paths", ())
                if isinstance(raw_paths, (list, tuple)):
                    for item in raw_paths:
                        if isinstance(item, str):
                            positive_paths_seen.add(item)
                        elif isinstance(item, (list, tuple)):
                            positive_paths_seen.update(str(path) for path in item)
                step += 1
                if gradient_conflict is not None:
                    gradient_conflicts.append(gradient_conflict)
                    (output_dir / f"gradient_conflict_step{step}.json").write_text(json.dumps(gradient_conflict, indent=2, sort_keys=True) + "\n")
                current = {
                    "loss_total": loss_total.item(),
                    "loss_dir": loss_dir.item(),
                    "loss_dist": loss_dist.item(),
                    "loss_endpoint": loss_endpoint.item(),
                    "loss_rank": loss_rank.item(),
                    "loss_cls": loss_cls.item(),
                    "loss_geom": loss_geom.item(),
                    "loss_vmf": loss_vmf.item(),
                    "train_direction_cosine": train_direction.item(),
                    "train_endpoint_photo_cosine": train_endpoint.item(),
                    "classification_accuracy": cls_accuracy.item(),
                    "classification_accuracy_q": cls_accuracy_q.item(),
                    "classification_accuracy_z0": cls_accuracy_z0.item(),
                }
                for name, value in current.items():
                    window[name] += value
                window_count += 1
                if step % int(args.log_every) == 0 or step == int(args.max_steps):
                    means = {name: value / window_count for name, value in window.items()}
                    equivalent_epochs = step * equivalent_epochs_per_step
                    train_log = {**means, "positive_photo_unique": float(len(positive_paths_seen)), "steps": step, "equivalent_epochs": equivalent_epochs}
                    experiment.log_metrics(train_log, step=step)
                    training_history.append({"step": step, "equivalent_epochs": equivalent_epochs, **means, "positive_photo_unique": len(positive_paths_seen)})
                    print(f"step={step:04d} total={means['loss_total']:.5f} dir={means['loss_dir']:.5f} dist={means['loss_dist']:.5f} endpoint={means['loss_endpoint']:.5f} rank={means['loss_rank']:.5f} unique_photos={len(positive_paths_seen)}")
                    window = dict.fromkeys(window, 0.0)
                    window_count = 0
                if bool(args.run_probes) and step in probe_steps:
                    history.append(
                        record_probe(step, training_losses=current, origin="scheduled")
                    )
                if step >= int(args.max_steps):
                    break

        final_path = _resolve_checkpoint_path(args.checkpoint_path)
        _save_checkpoint(
            path=final_path,
            model=model,
            optimizer=optimizer,
            step=step,
            data_name=data.name,
            args=args,
            split=split,
            parameter_counts=parameter_counts,
            provenance=provenance,
            resolved_config=resolved_config,
            data_split_identity=data_split_identity,
            data_manifest_identity=data_manifest_identity,
            generator=train_loader.generator,
            include_optimizer=bool(args.save_optimizer),
            continuation_metadata=continuation_metadata,
        )
        report = {
            "config": run_config,
            "checkpoint": str(final_path),
            "step": step,
            "equivalent_epochs": step * equivalent_epochs_per_step,
            "parameter_counts": parameter_counts,
            "resolved_config": resolved_config,
            "data_split_identity": data_split_identity,
            "data_manifest_identity": data_manifest_identity,
            "pseudo_split": {
                "seed": split.seed,
                "train_class_ids": list(split.train_class_ids),
                "validation_class_ids": list(split.validation_class_ids),
                "train_sketches": len(split.train_sketch_entries),
                "train_photos": len(split.train_photo_entries),
                "validation_sketches": len(split.validation_sketch_entries),
                "validation_photos": len(split.validation_photo_entries),
            },
            "training_history": training_history,
            "probe_history": history,
            "gradient_conflicts": gradient_conflicts,
            "continuation": continuation_metadata,
            "stage1_selection": stage1_selection,
            "training_target_angles": {
                "theta_train": {
                    **training_angle_summary(torch.tensor(training_angle_values)),
                    "values_degrees": [
                        value * (180.0 / math.pi) for value in training_angle_values
                    ],
                },
                "theta_train_fixed": {
                    **training_angle_summary(torch.tensor(training_angle_fixed_values)),
                    "values_degrees": [
                        value * (180.0 / math.pi)
                        for value in training_angle_fixed_values
                    ],
                },
                "origin_definitions": {
                    "theta_train": "acos(clamp(z0 · z_positive_instance))",
                    "theta_train_fixed": "acos(clamp(z_ref · z_positive_instance))",
                    "positive_pairs": "actual sampled training sketch/photo pairs",
                },
            }
            if training_angle_values and training_angle_fixed_values
            else {},
            "provenance": provenance,
            "resume": {
                "checkpoint": None if resume_path is None else str(resume_path),
                "starting_step": resume_step,
                "freeze_encoder_on_resume": bool(args.freeze_encoder_on_resume),
                "freeze_applied_before_first_update": freeze_applied_before_first_step,
                "encoder_frozen_immediately": continuation_metadata["encoder_frozen_at_start"],
                "effective_freeze_step": effective_freeze_step,
                "freeze_encoder_at_step": freeze_step,
                "optimizer_state_restored": optimizer_state_restored,
                "optimizer_state_reset": optimizer_state_reset,
                "optimizer_reset_scope": str(args.optimizer_reset_scope),
                "scheduler_state_restored": False,
                "rng_state_restored": resume_path is not None,
                "transport_head_reinitialized": bool(
                    args.reset_transport_head_on_resume
                ),
                "steps_since_resume_definition": "global_step - starting_step",
                "effective_probe_steps": effective_probe_steps,
                "probe_step_offsets": continuation_metadata["probe_step_offsets"],
            },
            "photo_sampling": {
                "num_positive_photos_per_step": int(args.num_positive_photos),
                "unique_positive_photo_paths_seen": len(positive_paths_seen),
                "same_sketch_can_see_new_photo_each_epoch": True,
            },
            "inference_contract": {
                "required_inputs": ["raw_sketch_image"],
                "text_required": False,
                "photo_required": False,
                "positive_set_required": False,
                "oracle_class_required": False,
            },
            "text_contract": {
                "enters_predictor": False,
                "enters_gate": False,
                "enters_distance_head": False,
                "enters_vmf": False,
                "loss_location": str(args.text_loss_location),
                "used_only_for_seen_classification": text_bank is not None,
            },
            "wandb": {
                "mode": experiment_mode,
                "project": str(args.wandb_project),
                "group": None if args.wandb_group is None else str(args.wandb_group),
                "run_id": experiment.run_id,
                "run_url": experiment.run_url,
            },
        }
        (output_dir / "training_history.json").write_text(json.dumps(training_history, indent=2, sort_keys=True) + "\n")
        (output_dir / "run_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Checkpoint saved to {final_path}")
        print("Text enters predictor: NO")
        print("Text required at inference: NO")


@hydra.main(
    version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="train_transport"
)
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
