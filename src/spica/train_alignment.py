"""Train the predeclared class-conditional spherical alignment campaign."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .alignment_artifacts import (
    ALIGNMENT_CAMPAIGN,
    ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
    ALIGNMENT_PILOT_CAMPAIGN,
    ALIGNMENT_REPLICATION_CAMPAIGN,
    ALL_ALIGNMENT_ROLES,
    CORRECTED_PILOT_INFERENCE_CONTRACT,
    CORRECTED_PILOT_ROLES,
    canonical_sha256,
    ensure_corrected_run_manifest,
    ensure_manifest,
    manifest_entry_identity,
    treatment_for_role,
    treatment_from_config,
)
from .config.data import load_data_config
from .data.datasets import MultiPositiveRetrievalTrainDataset
from .data.samplers import MatchedClassBatchSampler
from .evaluation.frozen_prompt import (
    cache_identity,
    encode_prompted_loader,
    evaluate_prompted,
    geometry_payload,
    load_prompt_cache,
    save_prompt_cache,
)
from .evaluation.text_bank import SoftPromptTextBank, encode_class_text_bank
from .models.alignment import AlignmentLoss, class_conditional_alignment_loss
from .models.checkpoint import visual_backbone_identity
from .models.clip import load_frozen_clip
from .models.frozen_prompt import FrozenPromptModel
from .models.jepa import classification_accuracy, jepa_text_classification_loss
from .provenance import capture_provenance, capture_rng_state, restore_rng_state
from .train_frozen_prompt import (
    _FrozenEncoderAdapter,
    _assert_clip_policy,
    _assert_optimizer_gradients,
    _check_finite,
    _device,
    _entry_identity,
    _fixed_diagnostic_entries,
    _gradient_norms,
    _loader,
    _metrics,
    _parameter_counts,
    _parameter_gradient_norms,
    _parameter_names,
    _parameter_norms,
    _path,
    _seed,
    _state_hash,
    _load_split,
    build_optimizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")


def _validate(args: DictConfig) -> None:
    role = str(args.experiment_role)
    if role not in ALL_ALIGNMENT_ROLES:
        raise ValueError(f"experiment_role must be exactly one of {ALL_ALIGNMENT_ROLES}")
    campaign = str(args.experiment_campaign)
    if campaign not in {
        ALIGNMENT_PILOT_CAMPAIGN,
        ALIGNMENT_CAMPAIGN,
        ALIGNMENT_REPLICATION_CAMPAIGN,
        ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
    }:
        raise ValueError("unknown alignment campaign")
    if str(args.run_kind) not in {"smoke", "pilot", "primary", "replication"}:
        raise ValueError("run_kind must be smoke, pilot, primary, or replication")
    if args.resume_checkpoint_path is not None:
        raise ValueError("alignment campaign runs are from scratch; resume is not enabled")
    observed = treatment_from_config(OmegaConf.to_container(args, resolve=True))
    expected = treatment_for_role(
        role, seed=int(args.seed), pseudo_val_seed=int(args.pseudo_val_seed)
    )
    mismatches = {
        key: (observed.get(key), value)
        for key, value in expected.items()
        if observed.get(key) != value
        and not (
            campaign == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
            and role in CORRECTED_PILOT_ROLES[1:]
            and key == "lambda_alignment_mean"
        )
    }
    if mismatches:
        raise ValueError(f"{role} has an ambiguous treatment: {mismatches}")
    if (
        campaign == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        and role == "alignment_control"
        and args.alignment_calibration_artifact not in (None, "")
    ):
        raise ValueError("corrected control runs must not reference a calibration artifact")
    if (
        campaign == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        and bool(args.calibration_only)
        and role not in CORRECTED_PILOT_ROLES[1:]
    ):
        raise ValueError("corrected calibration runs must use a mean-only pilot role")
    if (
        campaign == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        and role in CORRECTED_PILOT_ROLES[1:]
        and not bool(args.calibration_only)
        and args.alignment_calibration_artifact in (None, "")
    ):
        raise ValueError("corrected mean-only pilot requires a calibration artifact")
    if str(args.train_class_scope) != "pseudo_train":
        raise ValueError("selection requires pseudo-train classes")
    if bool(args.official_unseen_used_for_selection):
        raise ValueError("official unseen data cannot be used for selection")
    if int(args.max_steps) <= 0 or not args.probe_steps:
        raise ValueError("max_steps and probe_steps must be positive")
    if str(args.run_kind) == "smoke" and not 20 <= int(args.max_steps) <= 50:
        raise ValueError("smoke runs must use 20-50 steps")
    probe_steps = tuple(int(step) for step in args.probe_steps)
    if probe_steps[0] != 0 or probe_steps[-1] != int(args.max_steps):
        raise ValueError("probe_steps must start at 0 and end at max_steps")
    if tuple(sorted(set(probe_steps))) != probe_steps:
        raise ValueError("probe_steps must be strictly increasing")
    if int(args.batch_size) != int(args.classes_per_batch) * int(args.sketches_per_class):
        raise ValueError("batch_size must equal classes_per_batch * sketches_per_class")
    if int(args.classes_per_batch) < 2 or int(args.sketches_per_class) < 2:
        raise ValueError("matched alignment batches need at least two classes and sketches")
    if int(args.num_positive_photos) < 2:
        raise ValueError("alignment needs at least two positive photos per sketch")
    if str(args.alignment_geometry) not in {"log_map", "chordal"}:
        raise ValueError("alignment_geometry must be log_map or chordal")
    if str(args.alignment_anchor) not in {"text", "photo_mean"}:
        raise ValueError("alignment_anchor must be text or photo_mean")
    if str(args.alignment_target_gradient) not in {"detached", "symmetric"}:
        raise ValueError("alignment_target_gradient must be detached or symmetric")
    for name in (
        "visual_prompt_learning_rate",
        "soft_prompt_learning_rate",
        "visual_prompt_weight_decay",
        "soft_prompt_weight_decay",
        "margin",
        "tau_cls",
        "lambda_rank",
        "lambda_cls",
        "lambda_alignment_mean",
        "lambda_alignment_covariance",
    ):
        value = float(args[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in (
        "visual_prompt_learning_rate",
        "soft_prompt_learning_rate",
        "tau_cls",
    ):
        if float(args[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.pseudo_val_num_classes) <= 0 or int(args.diagnostic_num_seen) < 2:
        raise ValueError("diagnostic and pseudo-validation class counts must be positive")
    if int(args.calibration_batches) <= 0:
        raise ValueError("calibration_batches must be positive")
    if not math.isfinite(float(args.calibration_target_ratio)) or float(args.calibration_target_ratio) <= 0:
        raise ValueError("calibration_target_ratio must be finite and positive")
    if int(args.eval_batch_size) <= 0 or int(args.num_workers) < 0:
        raise ValueError("invalid loader settings")
    if int(args.query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    if int(args.pseudo_val_seed) != 3407:
        raise ValueError("the predeclared alignment campaign uses pseudo_val_seed=3407")
    if int(args.seed) not in {42, 123, 3407}:
        raise ValueError("predeclared alignment seeds are 42, 123, and 3407")


def _alignment_checkpoint(
    path: Path,
    *,
    model: FrozenPromptModel,
    text_bank: SoftPromptTextBank,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    args: DictConfig,
    split_identity: dict[str, Any],
    manifest_identity: dict[str, Any],
    entry_identity: dict[str, Any],
    loader_generator: torch.Generator,
    provenance: dict[str, Any],
    optimizer_groups: list[dict[str, Any]],
    initial_hash: str,
    initial_text_bank_hash: str,
    clip_freeze_policy: dict[str, Any],
    full_pseudo_unseen_mAP: float | None = None,
    calibration_identity: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    model_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in model_names
    }
    treatment = treatment_from_config(OmegaConf.to_container(args, resolve=True))
    torch.save(
        {
            "format_version": 1,
            "model_type": "frozen_prompt_alignment",
            "experiment_role": str(args.experiment_role),
            "campaign": str(args.experiment_campaign),
            "run_kind": str(args.run_kind),
            "step": step,
            "training_global_step": step,
            "full_pseudo_unseen_mAP": full_pseudo_unseen_mAP,
            "model_state_dict": model_state,
            "soft_prompt_state_dict": {
                name: value.detach().cpu() for name, value in text_bank.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "optimizer_groups": optimizer_groups,
            "rng_state": capture_rng_state(loader_generator),
            "experiment_code_commit": provenance.get("head_commit"),
            "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"),
            "training_seed": int(args.seed),
            "split_seed": int(args.pseudo_val_seed),
            "training_class_list": list(split_identity["train_class_ids"]),
            "validation_class_list": list(split_identity["validation_class_ids"]),
            "data_split_identity": split_identity,
            "data_manifest_identity": manifest_identity,
            "manifest_entry_identity": entry_identity,
            "model_state_hash": _state_hash(model),
            "initial_model_state_hash": initial_hash,
            "initial_text_bank_state_hash": initial_text_bank_hash,
            "resolved_config": OmegaConf.to_container(args, resolve=True),
            "resolved_treatment": treatment,
            "gradient_calibration_identity": calibration_identity,
            "backbone_identity": visual_backbone_identity(model),
            "clip_freeze_policy": clip_freeze_policy,
            "provenance": provenance,
        },
        path,
    )


def _alignment_metrics(value: AlignmentLoss | None) -> dict[str, float | int | None]:
    if value is None:
        return {
            "total": None,
            "mean": None,
            "covariance": None,
            "num_classes": 0,
            "num_sketches": 0,
            "num_photos": 0,
            "invalid_sketches": 0,
            "invalid_photos": 0,
            "skipped_classes": 0,
        }
    return {
        "total": float(value.total.item()),
        "mean": float(value.mean.item()),
        "covariance": float(value.covariance.item()),
        "num_classes": value.num_classes,
        "num_sketches": value.num_sketches,
        "num_photos": value.num_photos,
        "invalid_sketches": value.invalid_sketches,
        "invalid_photos": value.invalid_photos,
        "skipped_classes": value.skipped_classes,
    }


def _batch_objectives(
    model: FrozenPromptModel,
    text_bank: SoftPromptTextBank,
    hard_text_values: torch.Tensor,
    hard_text_labels: torch.Tensor,
    batch: dict[str, Any],
    args: DictConfig,
    device: torch.device,
    *,
    alignment_mean_weight: float | None = None,
    alignment_covariance_weight: float | None = None,
    alignment_target_gradient: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AlignmentLoss | None]:
    """Compute the exact training objectives for one fixed batch."""
    images = batch["sketch"].to(device, non_blocking=device.type == "cuda")
    positives = batch["positive_photos"].to(device, non_blocking=device.type == "cuda")
    negatives = batch["negative_photo"].to(device, non_blocking=device.type == "cuda")
    labels = batch["label"].long().to(device)
    query = model(images)
    photo_values = model.encode_photo(
        torch.cat((positives.reshape(-1, *positives.shape[2:]), negatives), dim=0)
    )
    positive, negative = photo_values.split(
        (images.shape[0] * positives.shape[1], images.shape[0]), dim=0
    )
    positive = positive.reshape(images.shape[0], positives.shape[1], -1)
    query_normalized = F.normalize(query, dim=-1)
    positive_normalized = F.normalize(positive, dim=-1)
    negative_normalized = F.normalize(negative, dim=-1)
    rank = F.softplus(
        float(args.margin)
        - (query_normalized.unsqueeze(1) * positive_normalized).sum(-1)
        + (query_normalized * negative_normalized).sum(-1, keepdim=True)
    ).mean()
    bank_values = text_bank()
    bank_labels = text_bank.class_labels.to(device)
    cls, logits = jepa_text_classification_loss(
        query,
        bank_values,
        bank_labels,
        labels,
        temperature=float(args.tau_cls),
        detach_text=False,
    )
    accuracy = classification_accuracy(logits, bank_labels, labels)
    mean_weight = (
        float(args.lambda_alignment_mean)
        if alignment_mean_weight is None
        else alignment_mean_weight
    )
    covariance_weight = (
        float(args.lambda_alignment_covariance)
        if alignment_covariance_weight is None
        else alignment_covariance_weight
    )
    target_gradient = (
        str(args.alignment_target_gradient)
        if alignment_target_gradient is None
        else alignment_target_gradient
    )
    alignment = None
    if mean_weight or covariance_weight:
        alignment = class_conditional_alignment_loss(
            query,
            positive,
            labels,
            text_embeddings=hard_text_values,
            text_labels=hard_text_labels,
            mean_weight=mean_weight,
            covariance_weight=covariance_weight,
            geometry=str(args.alignment_geometry),  # type: ignore[arg-type]
            anchor=str(args.alignment_anchor),  # type: ignore[arg-type]
            target_gradient=target_gradient,  # type: ignore[arg-type]
        )
    return rank, cls, accuracy, alignment


_CALIBRATION_SCHEMA_VERSION = 2
_CALIBRATION_EPS = 1e-12
_CALIBRATION_SKETCH_TOLERANCE = 1e-6
_CALIBRATION_RULE = (
    "lambda_alignment_mean = target_ratio * median("
    "base.sketch_gradient_norms / detached.sketch_gradient_norms)"
)


def _gradients_for(
    loss: torch.Tensor, parameters: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor | None, ...]:
    gradients: list[torch.Tensor | None] = [None] * len(parameters)
    active = [(index, parameter) for index, parameter in enumerate(parameters) if parameter.requires_grad]
    if not active or not loss.requires_grad:
        return tuple(gradients)
    values = torch.autograd.grad(
        loss,
        tuple(parameter for _, parameter in active),
        allow_unused=True,
    )
    for (index, _), gradient in zip(active, values, strict=True):
        gradients[index] = gradient
    return tuple(gradients)


def _grad_norms_for(
    gradients: tuple[torch.Tensor | None, ...], parameters: tuple[torch.Tensor, ...]
) -> list[float]:
    return [
        0.0 if gradient is None else float(gradient.detach().norm().item())
        for gradient, _ in zip(gradients, parameters, strict=True)
    ]


def _clone_module_state(module: torch.nn.Module) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        if isinstance(value, torch.Tensor)
        else value
        for name, value in module.state_dict().items()
    }


def _restore_module_state(module: torch.nn.Module, state: dict[str, Any]) -> None:
    module.load_state_dict(state, strict=True)


def _module_state_matches(module: torch.nn.Module, expected: dict[str, Any]) -> bool:
    actual = module.state_dict()
    if actual.keys() != expected.keys():
        return False
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, torch.Tensor):
            if not isinstance(actual_value, torch.Tensor) or not torch.equal(
                actual_value.detach().cpu(), expected_value
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _training_flags(
    *modules: torch.nn.Module,
) -> list[tuple[torch.nn.Module, bool]]:
    result: list[tuple[torch.nn.Module, bool]] = []
    seen: set[int] = set()
    for root in modules:
        for module in root.modules():
            if id(module) not in seen:
                result.append((module, bool(module.training)))
                seen.add(id(module))
    return result


def _restore_training_flags(flags: list[tuple[torch.nn.Module, bool]]) -> None:
    for module, training in flags:
        # Assigning the flag directly restores custom train() overrides exactly.
        module.training = training


def _parameter_grads(
    *modules: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    result: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    seen: set[int] = set()
    for root in modules:
        for parameter in root.parameters():
            if id(parameter) in seen:
                continue
            result.append(
                (
                    parameter,
                    None if parameter.grad is None else parameter.grad.detach().clone(),
                )
            )
            seen.add(id(parameter))
    return result


def _restore_parameter_grads(
    saved: list[tuple[torch.Tensor, torch.Tensor | None]],
) -> None:
    for parameter, gradient in saved:
        parameter.grad = None if gradient is None else gradient.to(parameter.device)


def _rng_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda = left["torch_cuda"]
    right_cuda = right["torch_cuda"]
    if len(left_cuda) != len(right_cuda) or any(
        not torch.equal(a, b) for a, b in zip(left_cuda, right_cuda, strict=True)
    ):
        return False
    left_loader = left["data_loader_generator"]
    right_loader = right["data_loader_generator"]
    if left_loader is None or right_loader is None:
        return left_loader is right_loader
    return torch.equal(left_loader, right_loader)


def _calibration_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous()
        digest = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": digest,
        }
    if isinstance(value, dict):
        return {
            str(key): _calibration_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_calibration_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _calibration_batch_identity(batches: list[dict[str, Any]], sampler_epoch: Any) -> dict[str, Any]:
    identities = [_calibration_json_value(batch) for batch in batches]
    return {
        "count": len(identities),
        "sha256": canonical_sha256(identities),
        "batches": identities,
        "sampler_epoch_before": sampler_epoch,
    }


def _calibration_loader(
    train_loader: DataLoader,
    sampler: MatchedClassBatchSampler,
    loader_generator: torch.Generator,
) -> DataLoader:
    """Clone the loader so worker-seeded sampling matches the training path."""
    if int(getattr(train_loader, "num_workers", 0)) == 0:
        return train_loader
    return DataLoader(
        train_loader.dataset,
        batch_sampler=sampler,
        num_workers=int(train_loader.num_workers),
        pin_memory=bool(train_loader.pin_memory),
        pin_memory_device=getattr(train_loader, "pin_memory_device", ""),
        collate_fn=train_loader.collate_fn,
        worker_init_fn=train_loader.worker_init_fn,
        generator=loader_generator,
        persistent_workers=False,
        prefetch_factor=getattr(train_loader, "prefetch_factor", 2),
        timeout=int(getattr(train_loader, "timeout", 0)),
    )


def _cosine_or_reason(
    base_gradient: torch.Tensor | None,
    alignment_gradient: torch.Tensor | None,
    base_norm: float,
    alignment_norm: float,
) -> tuple[float | None, str | None]:
    if base_norm <= _CALIBRATION_EPS or base_gradient is None:
        return None, "base_gradient_zero"
    if alignment_norm <= _CALIBRATION_EPS or alignment_gradient is None:
        return None, "alignment_gradient_zero"
    return (
        float(
            F.cosine_similarity(
                base_gradient.reshape(1, -1), alignment_gradient.reshape(1, -1)
            ).item()
        ),
        None,
    )


def _weighted_ratio_or_reason(
    alignment_norm: float,
    base_norm: float,
    weight: float | None,
) -> tuple[float | None, str | None]:
    if base_norm <= _CALIBRATION_EPS:
        return None, "base_gradient_zero"
    if weight is None:
        return None, "lambda_alignment_mean_unavailable"
    return weight * alignment_norm / base_norm, None


def _policy_gradient_payload(
    *,
    alignment_norms: list[float],
    alignment_gradients: list[torch.Tensor | None],
    base_norms: list[float],
    base_gradients: list[torch.Tensor | None],
    weight: float | None,
) -> dict[str, Any]:
    ratios: list[float | None] = []
    ratio_reasons: list[str | None] = []
    cosines: list[float | None] = []
    cosine_reasons: list[str | None] = []
    for alignment_norm, alignment_gradient, base_norm, base_gradient in zip(
        alignment_norms,
        alignment_gradients,
        base_norms,
        base_gradients,
        strict=True,
    ):
        ratio, ratio_reason = _weighted_ratio_or_reason(
            alignment_norm, base_norm, weight
        )
        cosine, cosine_reason = _cosine_or_reason(
            base_gradient,
            alignment_gradient,
            base_norm,
            alignment_norm,
        )
        ratios.append(ratio)
        ratio_reasons.append(ratio_reason)
        cosines.append(cosine)
        cosine_reasons.append(cosine_reason)
    return {
        "sketch_gradient_norms": alignment_norms,
        "photo_gradient_norms": [],
        "weighted_sketch_ratios": ratios,
        "weighted_sketch_ratio_reasons": ratio_reasons,
        "weighted_photo_ratios": [],
        "weighted_photo_ratio_reasons": [],
        "sketch_cosines_with_base": cosines,
        "sketch_cosine_reasons": cosine_reasons,
        "photo_cosines_with_base": [],
        "photo_cosine_reasons": [],
    }


def _calibrate_mean_alignment(
    model: FrozenPromptModel,
    text_bank: SoftPromptTextBank,
    hard_text_values: torch.Tensor,
    hard_text_labels: torch.Tensor,
    train_loader: DataLoader,
    sampler: MatchedClassBatchSampler,
    loader_generator: torch.Generator,
    args: DictConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Measure base/detached/symmetric gradients without changing training state."""
    target_ratio = float(args.calibration_target_ratio)
    count = int(args.calibration_batches)
    saved_rng = capture_rng_state(loader_generator)
    saved_epoch = getattr(sampler, "_epoch", None)
    saved_model_state = _clone_module_state(model)
    saved_text_state = _clone_module_state(text_bank)
    saved_flags = _training_flags(model, text_bank)
    saved_grads = _parameter_grads(model, text_bank)
    initial_model_state = dict(saved_model_state)
    initial_text_state = dict(saved_text_state)
    state_restoration_verified = False
    result: dict[str, Any] | None = None
    calibration_iterator: Any = None
    worker_lifecycle_verified = int(getattr(train_loader, "num_workers", 0)) == 0
    try:
        batches: list[dict[str, Any]] = []
        calibration_loader = _calibration_loader(
            train_loader, sampler, loader_generator
        )
        iterator = iter(calibration_loader)
        calibration_iterator = iterator
        for _ in range(count):
            try:
                batches.append(next(iterator))
            except StopIteration as error:
                raise ValueError(
                    "not enough fixed pseudo-train batches for calibration"
                ) from error
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        calibration_iterator = None
        worker_lifecycle_verified = True

        sketch_prompt = model.sketch_prompt
        photo_prompt = model.photo_prompt
        prompt_parameters = (sketch_prompt, photo_prompt)
        base_sketch_norms: list[float] = []
        base_photo_norms: list[float] = []
        detached_sketch_norms: list[float] = []
        detached_photo_norms: list[float] = []
        symmetric_sketch_norms: list[float] = []
        symmetric_photo_norms: list[float] = []
        base_sketch_gradients: list[torch.Tensor | None] = []
        base_photo_gradients: list[torch.Tensor | None] = []
        detached_sketch_gradients: list[torch.Tensor | None] = []
        detached_photo_gradients: list[torch.Tensor | None] = []
        symmetric_sketch_gradients: list[torch.Tensor | None] = []
        symmetric_photo_gradients: list[torch.Tensor | None] = []

        def reset_forward_state() -> None:
            _restore_module_state(model, initial_model_state)
            _restore_module_state(text_bank, initial_text_state)

        for batch in batches:
            reset_forward_state()
            forward_rng = capture_rng_state()
            model.train()
            rank, cls, _, _ = _batch_objectives(
                model,
                text_bank,
                hard_text_values,
                hard_text_labels,
                batch,
                args,
                device,
                alignment_mean_weight=0.0,
                alignment_covariance_weight=0.0,
            )
            base = float(args.lambda_rank) * rank + float(args.lambda_cls) * cls
            base_grads = _gradients_for(base, prompt_parameters)

            reset_forward_state()
            restore_rng_state(forward_rng)
            model.train()
            _, _, _, detached_alignment = _batch_objectives(
                model,
                text_bank,
                hard_text_values,
                hard_text_labels,
                batch,
                args,
                device,
                alignment_mean_weight=1.0,
                alignment_covariance_weight=0.0,
                alignment_target_gradient="detached",
            )
            if detached_alignment is None:
                raise RuntimeError(
                    "mean calibration did not produce a detached alignment loss"
                )
            detached_grads = _gradients_for(
                detached_alignment.mean, prompt_parameters
            )

            reset_forward_state()
            restore_rng_state(forward_rng)
            model.train()
            _, _, _, symmetric_alignment = _batch_objectives(
                model,
                text_bank,
                hard_text_values,
                hard_text_labels,
                batch,
                args,
                device,
                alignment_mean_weight=1.0,
                alignment_covariance_weight=0.0,
                alignment_target_gradient="symmetric",
            )
            if symmetric_alignment is None:
                raise RuntimeError(
                    "mean calibration did not produce a symmetric alignment loss"
                )
            symmetric_grads = _gradients_for(
                symmetric_alignment.mean, prompt_parameters
            )

            base_norms = _grad_norms_for(base_grads, prompt_parameters)
            detached_norms = _grad_norms_for(detached_grads, prompt_parameters)
            symmetric_norms = _grad_norms_for(symmetric_grads, prompt_parameters)
            base_sketch_norms.append(base_norms[0])
            base_photo_norms.append(base_norms[1])
            detached_sketch_norms.append(detached_norms[0])
            detached_photo_norms.append(detached_norms[1])
            symmetric_sketch_norms.append(symmetric_norms[0])
            symmetric_photo_norms.append(symmetric_norms[1])
            base_sketch_gradients.append(
                None
                if base_grads[0] is None
                else base_grads[0].detach().cpu().clone()
            )
            base_photo_gradients.append(
                None
                if base_grads[1] is None
                else base_grads[1].detach().cpu().clone()
            )
            detached_sketch_gradients.append(
                None
                if detached_grads[0] is None
                else detached_grads[0].detach().cpu().clone()
            )
            detached_photo_gradients.append(
                None
                if detached_grads[1] is None
                else detached_grads[1].detach().cpu().clone()
            )
            symmetric_sketch_gradients.append(
                None
                if symmetric_grads[0] is None
                else symmetric_grads[0].detach().cpu().clone()
            )
            symmetric_photo_gradients.append(
                None
                if symmetric_grads[1] is None
                else symmetric_grads[1].detach().cpu().clone()
            )

        detached_raw_ratios: list[float | None] = []
        detached_raw_ratio_reasons: list[str | None] = []
        for base_norm, detached_norm in zip(
            base_sketch_norms, detached_sketch_norms, strict=True
        ):
            if detached_norm <= _CALIBRATION_EPS:
                detached_raw_ratios.append(None)
                detached_raw_ratio_reasons.append("detached_sketch_gradient_zero")
            else:
                detached_raw_ratios.append(base_norm / detached_norm)
                detached_raw_ratio_reasons.append(None)
        valid_raw_ratios = [
            value
            for value in detached_raw_ratios
            if value is not None and math.isfinite(float(value))
        ]
        base_sketch_denominators_valid = all(
            math.isfinite(float(value)) and value > _CALIBRATION_EPS
            for value in base_sketch_norms
        )
        detached_sketch_denominators_valid = all(
            math.isfinite(float(value)) and value > _CALIBRATION_EPS
            for value in detached_sketch_norms
        )
        calibrated = (
            target_ratio * statistics.median(valid_raw_ratios)
            if len(valid_raw_ratios) == count
            and base_sketch_denominators_valid
            and detached_sketch_denominators_valid
            else None
        )

        def policy_payload(
            sketch_norms: list[float],
            photo_norms: list[float],
            sketch_gradients: list[torch.Tensor | None],
            photo_gradients: list[torch.Tensor | None],
        ) -> dict[str, Any]:
            payload = _policy_gradient_payload(
                alignment_norms=sketch_norms,
                alignment_gradients=sketch_gradients,
                base_norms=base_sketch_norms,
                base_gradients=base_sketch_gradients,
                weight=calibrated,
            )
            photo_ratios: list[float | None] = []
            photo_ratio_reasons: list[str | None] = []
            photo_cosines: list[float | None] = []
            photo_cosine_reasons: list[str | None] = []
            for alignment_norm, alignment_gradient, base_norm, base_gradient in zip(
                photo_norms,
                photo_gradients,
                base_photo_norms,
                base_photo_gradients,
                strict=True,
            ):
                ratio, ratio_reason = _weighted_ratio_or_reason(
                    alignment_norm, base_norm, calibrated
                )
                cosine, cosine_reason = _cosine_or_reason(
                    base_gradient,
                    alignment_gradient,
                    base_norm,
                    alignment_norm,
                )
                photo_ratios.append(ratio)
                photo_ratio_reasons.append(ratio_reason)
                photo_cosines.append(cosine)
                photo_cosine_reasons.append(cosine_reason)
            payload.update(
                {
                    "photo_gradient_norms": photo_norms,
                    "weighted_photo_ratios": photo_ratios,
                    "weighted_photo_ratio_reasons": photo_ratio_reasons,
                    "photo_cosines_with_base": photo_cosines,
                    "photo_cosine_reasons": photo_cosine_reasons,
                }
            )
            return payload

        detached = policy_payload(
            detached_sketch_norms,
            detached_photo_norms,
            detached_sketch_gradients,
            detached_photo_gradients,
        )
        symmetric = policy_payload(
            symmetric_sketch_norms,
            symmetric_photo_norms,
            symmetric_sketch_gradients,
            symmetric_photo_gradients,
        )
        sketch_differences = [
            0.0
            if detached_gradient is None and symmetric_gradient is None
            else None
            if detached_gradient is None or symmetric_gradient is None
            else float(
                (detached_gradient - symmetric_gradient).abs().max().item()
            )
            for detached_gradient, symmetric_gradient in zip(
                detached_sketch_gradients,
                symmetric_sketch_gradients,
                strict=True,
            )
        ]
        finite_differences = [value for value in sketch_differences if value is not None]
        sketch_gradients_match = all(
            value <= _CALIBRATION_SKETCH_TOLERANCE for value in finite_differences
        ) and len(finite_differences) == count
        fixed_batch_identity = _calibration_batch_identity(batches, saved_epoch)
        fixed_batch_identity["worker_lifecycle_verified"] = worker_lifecycle_verified
        reset_forward_state()
        initialization_identity = {
            "model_state_hash": _state_hash(model),
            "text_bank_state_hash": _state_hash(text_bank),
        }
        calibration_status = (
            "VALID"
            if calibrated is not None
            and base_sketch_denominators_valid
            and detached_sketch_denominators_valid
            and sketch_gradients_match
            and saved_epoch == 0
            and worker_lifecycle_verified
            else "UNVERIFIED"
        )
        result = {
            "schema_version": _CALIBRATION_SCHEMA_VERSION,
            "schema_name": "corrected_mean_alignment_calibration",
            "status": calibration_status,
            "schema_notes": {
                "legacy_schema_version": 1,
                "legacy_field": "weighted_photo_gradient_ratios_if_symmetric",
                "legacy_field_status": "INVALID",
                "legacy_field_reason": (
                    "it was computed from detached mean-alignment gradients and "
                    "did not measure the symmetric policy"
                ),
            },
            "base": {
                "sketch_gradient_norms": base_sketch_norms,
                "photo_gradient_norms": base_photo_norms,
            },
            "detached": {
                **detached,
                "unweighted_sketch_ratios": detached_raw_ratios,
                "unweighted_sketch_ratio_reasons": detached_raw_ratio_reasons,
            },
            "symmetric": symmetric,
            "calibration": {
                "status": calibration_status,
                "lambda_alignment_mean": calibrated,
                "target_ratio": target_ratio,
                "rule": _CALIBRATION_RULE,
                "batches": count,
                "fixed_batch_identity": fixed_batch_identity,
                "initialization_identity": initialization_identity,
                "state_restoration_verified": False,
                "lambda_selection_status": (
                    "VALID"
                    if (
                        calibrated is not None
                        and base_sketch_denominators_valid
                        and detached_sketch_denominators_valid
                    )
                    else "UNVERIFIED_ZERO_DENOMINATOR"
                ),
                "lambda_selection_reasons": detached_raw_ratio_reasons,
                "sketch_gradient_comparison": {
                    "detached_vs_symmetric_max_abs_difference": (
                        None if not finite_differences else max(finite_differences)
                    ),
                    "tolerance": _CALIBRATION_SKETCH_TOLERANCE,
                    "match": sketch_gradients_match,
                },
            },
        }
    finally:
        if calibration_iterator is not None:
            try:
                shutdown_workers = getattr(calibration_iterator, "_shutdown_workers", None)
                if callable(shutdown_workers):
                    shutdown_workers()
                else:
                    worker_lifecycle_verified = False
            except Exception:  # noqa: BLE001 - worker teardown is best effort
                worker_lifecycle_verified = False
        _restore_module_state(model, saved_model_state)
        _restore_module_state(text_bank, saved_text_state)
        _restore_parameter_grads(saved_grads)
        _restore_training_flags(saved_flags)
        if saved_epoch is not None:
            sampler._epoch = saved_epoch
        restore_rng_state(saved_rng, loader_generator)
        state_restoration_verified = (
            _module_state_matches(model, saved_model_state)
            and _module_state_matches(text_bank, saved_text_state)
            and _rng_matches(saved_rng, capture_rng_state(loader_generator))
            and getattr(sampler, "_epoch", None) == saved_epoch
            and all(
                module.training == training for module, training in saved_flags
            )
            and all(
                (parameter.grad is None and gradient is None)
                or (
                    parameter.grad is not None
                    and gradient is not None
                    and torch.equal(parameter.grad, gradient)
                )
                for parameter, gradient in saved_grads
            )
        )

    if result is None:
        raise RuntimeError("calibration did not produce a result")
    result["calibration"]["state_restoration_verified"] = state_restoration_verified
    result["status"] = (
        "VALID"
        if result["status"] == "VALID"
        and state_restoration_verified
        and saved_epoch == 0
        and result["calibration"]["fixed_batch_identity"].get(
            "worker_lifecycle_verified"
        ) is True
        else "UNVERIFIED"
    )
    result["calibration"]["status"] = result["status"]
    return result


def _validate_calibration_payload(
    payload: dict[str, Any],
    *,
    role: str,
    campaign: str,
    seed: int,
    split_identity: dict[str, Any],
    source_hash: str | None,
    initial_hash: str,
    initial_text_bank_hash: str,
    resolved_config: dict[str, Any],
) -> float:
    if payload.get("schema_version") != _CALIBRATION_SCHEMA_VERSION:
        raise ValueError("alignment calibration artifact has an unsupported schema")
    if payload.get("schema_name") != "corrected_mean_alignment_calibration":
        raise ValueError("alignment calibration artifact schema name is invalid")
    if payload.get("status") != "VALID":
        raise ValueError("alignment calibration artifact is not valid")
    if "weighted_photo_gradient_ratios_if_symmetric" in payload:
        raise ValueError("alignment calibration artifact contains the invalid legacy field")
    if campaign != ALIGNMENT_CORRECTED_PILOT_CAMPAIGN or role not in {
        "alignment_mean_text_log",
        "alignment_mean_text_log_symmetric",
    }:
        raise ValueError("alignment calibration artifacts are only valid for corrected mean pilots")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("status") != "VALID":
        raise ValueError("alignment calibration artifact is not valid")
    if calibration.get("lambda_selection_status") != "VALID":
        raise ValueError("alignment calibration lambda selection is not verified")
    for field, expected in (
        ("experiment_role", payload.get("experiment_role")),
        ("campaign", campaign),
        ("training_seed", seed),
        ("initial_model_state_hash", initial_hash),
        ("source_snapshot_hash", source_hash),
        ("split_identity_hash", split_identity.get("sha256")),
    ):
        if calibration.get(field) != expected:
            raise ValueError(f"alignment calibration nested {field} is stale")
    if calibration.get("state_restoration_verified") is not True:
        raise ValueError("alignment calibration state restoration is not verified")
    comparison = calibration.get("sketch_gradient_comparison")
    if not isinstance(comparison, dict) or comparison.get("match") is not True:
        raise ValueError("detached and symmetric sketch calibration gradients do not match")
    comparison_tolerance = comparison.get("tolerance")
    comparison_difference = comparison.get(
        "detached_vs_symmetric_max_abs_difference"
    )
    if (
        not isinstance(comparison_tolerance, (int, float))
        or isinstance(comparison_tolerance, bool)
        or not math.isfinite(float(comparison_tolerance))
        or not isinstance(comparison_difference, (int, float))
        or isinstance(comparison_difference, bool)
        or not math.isfinite(float(comparison_difference))
        or float(comparison_difference) < 0.0
        or not math.isclose(
            float(comparison_tolerance),
            _CALIBRATION_SKETCH_TOLERANCE,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or float(comparison_difference) > _CALIBRATION_SKETCH_TOLERANCE
    ):
        raise ValueError("detached and symmetric sketch calibration gradients do not match")
    calibrated_value = calibration.get("lambda_alignment_mean")
    if calibrated_value is None or not math.isfinite(float(calibrated_value)):
        raise ValueError("alignment calibration artifact has no finite lambda")
    calibrated_lambda = float(calibrated_value)
    if not math.isclose(
        calibrated_lambda,
        float(resolved_config["lambda_alignment_mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "configured lambda_alignment_mean does not match calibration artifact"
        )
    split_seed = split_identity.get(
        "pseudo_validation_seed", split_identity.get("seed")
    )
    if (
        not source_hash
        or not split_identity.get("sha256")
        or split_identity.get("sha256")
        != canonical_sha256(
            {key: value for key, value in split_identity.items() if key != "sha256"}
        )
        or split_seed is None
    ):
        raise ValueError("alignment calibration run provenance is incomplete")
    if payload.get("campaign") != campaign or payload.get("training_seed") != seed:
        raise ValueError("alignment calibration artifact campaign or seed is stale")
    if payload.get("initial_model_state_hash") != initial_hash:
        raise ValueError("alignment calibration artifact initialization is stale")
    if payload.get("initial_text_bank_state_hash") != initial_text_bank_hash:
        raise ValueError("alignment calibration artifact text-bank initialization is stale")
    identity = calibration.get("initialization_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("model_state_hash") != initial_hash
        or identity.get("text_bank_state_hash") != initial_text_bank_hash
    ):
        raise ValueError("alignment calibration artifact initialization identity is stale")
    if payload.get("source_snapshot_hash") != source_hash:
        raise ValueError("alignment calibration artifact source snapshot is stale")
    artifact_split_identity = payload.get("split_identity")
    if (
        not isinstance(artifact_split_identity, dict)
        or artifact_split_identity != split_identity
        or artifact_split_identity.get("sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in artifact_split_identity.items()
                if key != "sha256"
            }
        )
    ):
        raise ValueError("alignment calibration artifact embedded split identity is stale")
    if payload.get("split_identity_hash") != split_identity.get("sha256"):
        raise ValueError("alignment calibration artifact split identity is stale")
    artifact_role = payload.get("experiment_role")
    if artifact_role not in {"alignment_mean_text_log", "alignment_mean_text_log_symmetric"}:
        raise ValueError("alignment calibration artifact role is invalid")
    detached = payload.get("detached")
    symmetric = payload.get("symmetric")
    if not isinstance(detached, dict) or not isinstance(symmetric, dict):
        raise ValueError("alignment calibration policy diagnostics are missing")
    batches = calibration.get("batches")
    if isinstance(batches, bool) or not isinstance(batches, int) or batches <= 0:
        raise ValueError("alignment calibration batch count is invalid")

    def series(
        policy: dict[str, Any],
        field: str,
        *,
        allow_none: bool = False,
        nonnegative: bool = False,
    ) -> list[Any]:
        values = policy.get(field)
        if not isinstance(values, list) or len(values) != batches:
            raise ValueError(f"alignment calibration field {field} is incomplete")
        for value in values:
            if value is None and allow_none:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (nonnegative and float(value) < 0.0)
            ):
                raise ValueError(f"alignment calibration field {field} is invalid")
        return values

    def ratio_series(
        policy: dict[str, Any],
        field: str,
        reason_field: str,
        *,
        nonnegative: bool = True,
    ) -> list[Any]:
        values = series(policy, field, allow_none=True, nonnegative=nonnegative)
        reasons = policy.get(reason_field)
        if not isinstance(reasons, list) or len(reasons) != batches:
            raise ValueError(f"alignment calibration field {reason_field} is incomplete")
        for value, reason in zip(values, reasons, strict=True):
            if value is None and (not isinstance(reason, str) or not reason):
                raise ValueError(f"alignment calibration field {reason_field} is invalid")
            if value is not None and reason is not None:
                raise ValueError(f"alignment calibration field {reason_field} is invalid")
        return values

    base_sketch = series(
        payload["base"], "sketch_gradient_norms", nonnegative=True
    ) if isinstance(payload.get("base"), dict) else None
    base_photo = series(
        payload["base"], "photo_gradient_norms", nonnegative=True
    ) if isinstance(payload.get("base"), dict) else None
    if base_sketch is None or base_photo is None:
        raise ValueError("alignment calibration base diagnostics are missing")
    if any(float(value) <= _CALIBRATION_EPS for value in base_sketch):
        raise ValueError("alignment calibration has an undefined weighted-ratio denominator")
    detached_sketch = series(
        detached, "sketch_gradient_norms", nonnegative=True
    )
    detached_photo = series(
        detached, "photo_gradient_norms", nonnegative=True
    )
    symmetric_sketch = series(
        symmetric, "sketch_gradient_norms", nonnegative=True
    )
    symmetric_photo = series(
        symmetric, "photo_gradient_norms", nonnegative=True
    )
    norm_difference_lower_bound = max(
        abs(float(detached_value) - float(symmetric_value))
        for detached_value, symmetric_value in zip(
            detached_sketch, symmetric_sketch, strict=True
        )
    )
    if float(comparison_difference) + 1e-12 < norm_difference_lower_bound:
        raise ValueError("alignment calibration sketch comparison is stale")
    if any(float(value) > _CALIBRATION_EPS for value in detached_photo):
        raise ValueError("detached calibration has a photo alignment gradient")
    for policy in (detached, symmetric):
        ratio_series(policy, "weighted_sketch_ratios", "weighted_sketch_ratio_reasons")
        ratio_series(policy, "weighted_photo_ratios", "weighted_photo_ratio_reasons")
        ratio_series(
            policy,
            "sketch_cosines_with_base",
            "sketch_cosine_reasons",
            nonnegative=False,
        )
        ratio_series(
            policy,
            "photo_cosines_with_base",
            "photo_cosine_reasons",
            nonnegative=False,
        )
    raw_ratios = ratio_series(
        detached, "unweighted_sketch_ratios", "unweighted_sketch_ratio_reasons"
    )
    target_ratio = calibration.get("target_ratio")
    if (
        isinstance(target_ratio, bool)
        or not isinstance(target_ratio, (int, float))
        or not math.isfinite(float(target_ratio))
        or float(target_ratio) < 0.0
    ):
        raise ValueError("alignment calibration target ratio is invalid")
    expected_raw_ratios: list[float | None] = []
    for base_norm, detached_norm, observed in zip(
        base_sketch, detached_sketch, raw_ratios, strict=True
    ):
        expected = (
            None
            if float(detached_norm) <= _CALIBRATION_EPS
            else float(base_norm) / float(detached_norm)
        )
        expected_raw_ratios.append(expected)
        if expected is None:
            if observed is not None:
                raise ValueError("alignment calibration zero-denominator ratio is invalid")
        elif observed is None or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("alignment calibration raw ratios are stale")
    if any(value is None for value in expected_raw_ratios):
        raise ValueError("alignment calibration has an undefined lambda denominator")
    expected_lambda = float(target_ratio) * statistics.median(
        [float(value) for value in expected_raw_ratios if value is not None]
    )
    if not math.isclose(
        calibrated_lambda, expected_lambda, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("alignment calibration lambda rule is invalid")
    for policy, sketch_norms, photo_norms in (
        (detached, detached_sketch, detached_photo),
        (symmetric, symmetric_sketch, symmetric_photo),
    ):
        for field, norms, base_norms in (
            ("weighted_sketch_ratios", sketch_norms, base_sketch),
            ("weighted_photo_ratios", photo_norms, base_photo),
        ):
            observed = policy[field]
            for value, norm, base_norm in zip(observed, norms, base_norms, strict=True):
                expected = (
                    None
                    if float(base_norm) <= _CALIBRATION_EPS
                    else calibrated_lambda * float(norm) / float(base_norm)
                )
                if expected is None:
                    if value is not None:
                        raise ValueError("alignment calibration ratio denominator is invalid")
                elif value is None or not math.isclose(
                    float(value), expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("alignment calibration weighted ratios are stale")
    identity = calibration.get("initialization_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("model_state_hash") != initial_hash
        or identity.get("text_bank_state_hash") != initial_text_bank_hash
    ):
        raise ValueError("alignment calibration initialization identity is invalid")
    fixed_identity = calibration.get("fixed_batch_identity")
    if (
        not isinstance(fixed_identity, dict)
        or fixed_identity.get("count") != batches
        or not isinstance(fixed_identity.get("batches"), list)
        or len(fixed_identity["batches"]) != batches
        or not fixed_identity.get("sha256")
        or fixed_identity.get("sha256") != canonical_sha256(fixed_identity["batches"])
        or fixed_identity.get("sampler_epoch_before") != 0
        or fixed_identity.get("worker_lifecycle_verified") is not True
    ):
        raise ValueError("alignment calibration fixed batches are not identified")
    lambda_reasons = calibration.get("lambda_selection_reasons")
    if not isinstance(lambda_reasons, list) or len(lambda_reasons) != batches:
        raise ValueError("alignment calibration lambda reasons are incomplete")
    if lambda_reasons != detached["unweighted_sketch_ratio_reasons"]:
        raise ValueError("alignment calibration lambda reasons are stale")
    calibration_config = payload.get("calibration_config")
    if not isinstance(calibration_config, dict):
        raise ValueError("alignment calibration config evidence is missing")
    if payload.get("config_hash") != canonical_sha256(calibration_config):
        raise ValueError("alignment calibration config hash is invalid")
    if calibration_config.get("experiment_campaign") != campaign:
        raise ValueError("alignment calibration config campaign is stale")
    if calibration.get("target_ratio") != calibration_config.get("calibration_target_ratio"):
        raise ValueError("alignment calibration target ratio is stale")
    if calibration.get("batches") != calibration_config.get("calibration_batches"):
        raise ValueError("alignment calibration batch count is stale")
    if calibration_config.get("seed") != seed:
        raise ValueError("alignment calibration config seed is stale")
    if calibration_config.get("pseudo_val_seed") != split_seed:
        raise ValueError("alignment calibration config split seed is stale")
    if calibration_config.get("alignment_geometry") != "log_map" or calibration_config.get(
        "alignment_anchor"
    ) != "text":
        raise ValueError("alignment calibration geometry or anchor is invalid")
    expected_gradient = (
        "symmetric"
        if artifact_role == "alignment_mean_text_log_symmetric"
        else "detached"
    )
    if calibration_config.get("alignment_target_gradient") != expected_gradient:
        raise ValueError("alignment calibration target-gradient policy is invalid")
    if calibration_config.get("lambda_alignment_covariance") != 0.0:
        raise ValueError("alignment calibration covariance weight must be zero")
    ignored_calibration_config_keys = {
        "alignment_calibration_artifact",
        "alignment_target_gradient",
        "calibration_only",
        "experiment_name",
        "experiment_role",
        "lambda_alignment_mean",
    }
    comparable_calibration_config = {
        key: value
        for key, value in calibration_config.items()
        if key not in ignored_calibration_config_keys
    }
    comparable_resolved_config = {
        key: value
        for key, value in resolved_config.items()
        if key not in ignored_calibration_config_keys
    }
    if canonical_sha256(comparable_calibration_config) != canonical_sha256(
        comparable_resolved_config
    ):
        raise ValueError("alignment calibration config does not match this training run")
    return calibrated_lambda


def run(args: DictConfig) -> None:
    _validate(args)
    seed = int(args.seed)
    _seed(seed)
    device = _device(str(args.device))
    data = load_data_config(_path(args.data_config))
    split, names, split_identity, data_manifest_identity = _load_split(data, args)
    manifest_path = _path(args.experiment_manifest_path)
    if (
        str(args.experiment_campaign) == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        and manifest_path.name == "corrected_pilot_manifest.json"
    ):
        raise ValueError(
            "historical corrected_pilot_manifest.json is immutable; use the v2 manifest path"
        )
    role = str(args.experiment_role)
    train_names = {class_id: names[class_id] for class_id in split.train_class_ids}
    photo_clip = load_frozen_clip(
        model_name=str(args.model_name), pretrained=args.pretrained, device=device
    )
    model = FrozenPromptModel(
        photo_clip.encoder.model.visual,
        prompt_length=int(args.visual_prompt_length),
        train_visual_layernorm=bool(args.train_visual_layernorm),
        train_sketch_prompt=bool(args.train_sketch_prompt),
        train_photo_prompt=bool(args.train_photo_prompt),
    ).to(device)
    model.train(False)
    text_bank = SoftPromptTextBank(
        photo_clip.encoder,
        photo_clip.tokenizer,
        train_names,
        prompt_length=int(args.soft_prompt_length),
    ).to(device)
    hard_text = encode_class_text_bank(
        photo_clip.encoder,
        photo_clip.tokenizer,
        train_names,
        prompt_template=str(args.prompt_template),
    )
    hard_text_values = hard_text.embeddings.to(device)
    hard_text_labels = hard_text.labels.to(device)

    dataset = MultiPositiveRetrievalTrainDataset(
        split.train_sketch_entries,
        split.train_photo_entries,
        photo_clip.transform,
        photo_clip.transform,
        num_positive_photos=int(args.num_positive_photos),
    )
    sampler = MatchedClassBatchSampler(
        [entry.label for entry in split.train_sketch_entries],
        classes_per_batch=int(args.classes_per_batch),
        samples_per_class=int(args.sketches_per_class),
        seed=seed,
        batches_per_epoch=args.batches_per_epoch,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        worker_init_fn=None,
        generator=loader_generator,
        persistent_workers=int(args.num_workers) > 0,
    )
    diagnostic_entries = _fixed_diagnostic_entries(
        split.train_sketch_entries, int(args.diagnostic_num_seen)
    )
    diagnostic_loader = _loader(diagnostic_entries, photo_clip.transform, args)
    val_sketch_loader = _loader(split.validation_sketch_entries, photo_clip.transform, args)
    val_photo_loader = _loader(split.validation_photo_entries, photo_clip.transform, args)
    if len(train_loader) == 0:
        raise ValueError("training loader has no batches")

    vanilla_model = _FrozenEncoderAdapter(photo_clip.encoder)
    vanilla_sketch = encode_prompted_loader(vanilla_model, val_sketch_loader)
    vanilla_photo = encode_prompted_loader(vanilla_model, val_photo_loader, photo=True)
    optimizer, optimizer_groups = build_optimizer(model, text_bank, args)
    if optimizer is None:
        raise RuntimeError("alignment campaign requires trainable prompt parameters")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    initial_hash = _state_hash(model)
    initial_text_bank_hash = _state_hash(text_bank)
    clip_before = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if name.startswith("visual.")
    }
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_provenance(
        PROJECT_ROOT,
        resolved_config=OmegaConf.to_container(args, resolve=True),
        command=[sys.executable, *sys.argv],
    )
    resolved_config = OmegaConf.to_container(args, resolve=True)
    if str(args.experiment_campaign) == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN:
        manifest, manifest_sha256 = ensure_corrected_run_manifest(
            manifest_path,
            dataset=str(data.name),
            data_config=str(args.data_config),
            campaign=str(args.experiment_campaign),
            role=role,
            training_seed=seed,
            pseudo_validation_seed=int(args.pseudo_val_seed),
            split_identity=split_identity,
            resolved_config=resolved_config,
            source_hash=provenance.get("source_snapshot", {}).get("sha256"),
            initial_model_state_hash=initial_hash,
            training_horizon=int(args.max_steps),
            replicate_id=f"{args.run_kind}-seed{seed}-{canonical_sha256(resolved_config)[:12]}",
        )
    else:
        manifest, manifest_sha256 = ensure_manifest(
            manifest_path,
            dataset=str(data.name),
            data_config=str(args.data_config),
            campaign=str(args.experiment_campaign),
        )
    entry_identity = manifest_entry_identity(
        manifest_path,
        manifest,
        role=role,
        manifest_sha256=manifest_sha256,
        training_seed=seed if str(args.experiment_campaign) == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN else None,
        config_hash=canonical_sha256(resolved_config)
        if str(args.experiment_campaign) == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        else None,
    )
    attention_images = next(iter(val_sketch_loader))["image"][:8]
    history: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    step = 0
    probe_steps = {int(value) for value in args.probe_steps}
    last_train: dict[str, Any] = {
        "rank": None,
        "classification": None,
        "accuracy": None,
        "alignment": None,
    }
    last_gradient_norms = {group["name"]: 0.0 for group in optimizer_groups}
    last_parameter_gradient_norms: dict[str, float | None] = {}
    gradient_calibration: dict[str, Any] | None = None
    calibration_batch_identities: list[Any] | None = None
    calibration_batch_replay_index = 0
    calibration_batch_replay_verified: bool | None = None
    calibration_artifact = args.alignment_calibration_artifact
    if (
        str(args.experiment_campaign) == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        and role == "alignment_control"
        and calibration_artifact not in (None, "")
    ):
        raise ValueError("corrected control runs must not reference calibration")
    if calibration_artifact not in (None, ""):
        calibration_path = _path(calibration_artifact)
        if not calibration_path.is_file():
            raise FileNotFoundError(f"alignment calibration artifact not found: {calibration_path}")
        calibration_bytes = calibration_path.read_bytes()
        calibration_payload = json.loads(calibration_bytes)
        calibrated_lambda = _validate_calibration_payload(
            calibration_payload,
            role=role,
            campaign=str(args.experiment_campaign),
            seed=seed,
            split_identity=split_identity,
            source_hash=provenance.get("source_snapshot", {}).get("sha256"),
            initial_hash=initial_hash,
            initial_text_bank_hash=initial_text_bank_hash,
            resolved_config=resolved_config,
        )
        fixed_batch_identity = calibration_payload["calibration"][
            "fixed_batch_identity"
        ]
        calibration_batch_identities = fixed_batch_identity["batches"]
        calibration_batch_replay_verified = False
        gradient_calibration = {
            "artifact": str(calibration_path.resolve()),
            "artifact_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
            "lambda_alignment_mean": calibrated_lambda,
            "schema_version": calibration_payload["schema_version"],
            "campaign": str(args.experiment_campaign),
            "experiment_role": role,
            "training_seed": seed,
            "source_snapshot_hash": calibration_payload.get("source_snapshot_hash"),
            "split_identity_hash": calibration_payload.get("split_identity_hash"),
            "initial_model_state_hash": calibration_payload.get(
                "initial_model_state_hash"
            ),
            "initial_text_bank_state_hash": calibration_payload.get(
                "initial_text_bank_state_hash"
            ),
            "config_hash": calibration_payload.get("config_hash"),
            "fixed_batch_identity_sha256": fixed_batch_identity["sha256"],
            "first_batch_replay_verified": False,
            "calibration_batch_replay_verified": False,
            "calibration_batch_replay_count": 0,
            "calibration_batch_replay_expected_count": len(calibration_batch_identities),
            "calibration_batch_replay_prefix_sha256": canonical_sha256([]),
            "worker_lifecycle_verified": fixed_batch_identity[
                "worker_lifecycle_verified"
            ],
        }
    if bool(args.calibration_only):
        calibration = _calibrate_mean_alignment(
            model,
            text_bank,
            hard_text_values,
            hard_text_labels,
            train_loader,
            sampler,
            loader_generator,
            args,
            device,
        )
        calibration.update(
            {
                "experiment_role": role,
                "campaign": str(args.experiment_campaign),
                "training_seed": seed,
                "initial_model_state_hash": initial_hash,
                "initial_text_bank_state_hash": initial_text_bank_hash,
                "source_snapshot_hash": provenance.get("source_snapshot", {}).get(
                    "sha256"
                ),
                "split_identity": split_identity,
                "split_identity_hash": split_identity.get("sha256"),
                "calibration_config": resolved_config,
                "config_hash": canonical_sha256(resolved_config),
            }
        )
        calibration["calibration"].update(
            {
                "experiment_role": role,
                "campaign": str(args.experiment_campaign),
                "training_seed": seed,
                "initial_model_state_hash": initial_hash,
                "initial_text_bank_state_hash": initial_text_bank_hash,
                "source_snapshot_hash": provenance.get("source_snapshot", {}).get(
                    "sha256"
                ),
                "split_identity_hash": split_identity.get("sha256"),
                "config_hash": canonical_sha256(resolved_config),
            }
        )
        (output_dir / "calibration.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(calibration, indent=2, sort_keys=True))
        return

    def probe(probe_step: int) -> None:
        checkpoint = output_dir / "checkpoints" / f"alignment_step{probe_step}.pt"
        clip_policy = _assert_clip_policy(model, clip_before, role="alignment")
        clip_policy.update(
            {
                "photo_encoder_frozen": True,
                "visual_projection_frozen": True,
                "text_tower_frozen": True,
            }
        )
        current_sketch = encode_prompted_loader(model, val_sketch_loader)
        current_photo = encode_prompted_loader(model, val_photo_loader, photo=True)
        uncached_evaluation = evaluate_prompted(
            current_sketch,
            current_photo,
            query_chunk_size=int(args.query_chunk_size),
            device=device,
        )
        _alignment_checkpoint(
            checkpoint,
            model=model,
            text_bank=text_bank,
            optimizer=optimizer,
            scheduler=scheduler,
            step=probe_step,
            args=args,
            split_identity=split_identity,
            manifest_identity=data_manifest_identity,
            entry_identity=entry_identity,
            loader_generator=loader_generator,
            provenance=provenance,
            optimizer_groups=optimizer_groups,
            initial_hash=initial_hash,
            initial_text_bank_hash=initial_text_bank_hash,
            clip_freeze_policy=clip_policy,
            full_pseudo_unseen_mAP=float(uncached_evaluation["full_mAP"]),
            calibration_identity=(
                None
                if gradient_calibration is None
                else {
                    key: gradient_calibration.get(key)
                    for key in (
                        "artifact",
                        "artifact_sha256",
                        "fixed_batch_identity_sha256",
                        "first_batch_replay_verified",
                        "calibration_batch_replay_verified",
                        "calibration_batch_replay_count",
                        "calibration_batch_replay_expected_count",
                        "calibration_batch_replay_prefix_sha256",
                        "worker_lifecycle_verified",
                    )
                }
            ),
        )
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        identity = cache_identity(
            prompt_checkpoint_hash=checkpoint_hash,
            prompt_length=int(args.visual_prompt_length),
            prompt_mode="prompt_only",
            modality="photo",
            model_name=str(args.model_name),
            pretrained=None if args.pretrained is None else str(args.pretrained),
            data_manifest_identity=data_manifest_identity,
        )
        cache_path = output_dir / "gallery_cache" / f"photo_step{probe_step}.pt"
        save_prompt_cache(current_photo, cache_path, identity=identity)
        loaded_photo = load_prompt_cache(cache_path, expected_identity=identity)
        evaluation = evaluate_prompted(
            current_sketch,
            loaded_photo,
            query_chunk_size=int(args.query_chunk_size),
            device=device,
        )
        if not math.isclose(
            float(uncached_evaluation["full_mAP"]),
            float(evaluation["full_mAP"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("prompt cache changed the evaluated horizon metric")
        geometry = geometry_payload(
            current_sketch,
            loaded_photo,
            sketch_reference=vanilla_sketch,
            photo_reference=vanilla_photo,
            model=model,
            max_samples=512,
        )
        classification = _diagnostic_classification(
            model,
            diagnostic_loader,
            text_bank,
            tau=float(args.tau_cls),
            device=device,
        )
        val_metrics = _metrics(evaluation)
        val_metrics.update(
            {
                "query_identity": _entry_identity(split.validation_sketch_entries),
                "gallery_identity": _entry_identity(split.validation_photo_entries),
            }
        )
        row: dict[str, Any] = {
            "step": probe_step,
            "training_global_step": probe_step,
            "comparison_horizon": {"kind": "training_global_step", "value": probe_step},
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "val": val_metrics,
            "full_pseudo_unseen_mAP": val_metrics["full_mAP"],
            "P@200": val_metrics["P@200"],
            "mAP@200": val_metrics["mAP@200"],
            "last_train_batch_rank_loss": last_train["rank"],
            "last_train_batch_classification_loss": last_train["classification"],
            "last_train_batch_accuracy": last_train["accuracy"],
            "last_train_batch_alignment": last_train["alignment"],
            "diagnostic_seen_classification_accuracy": None
            if classification is None
            else classification["diagnostic_seen_classification_accuracy"],
            "diagnostic_seen_classification_loss": None
            if classification is None
            else classification["diagnostic_seen_classification_loss"],
            "diagnostic_seen_classification_count": None
            if classification is None
            else classification["diagnostic_seen_classification_count"],
            "gradient_norms": dict(last_gradient_norms),
            "gradient_norms_by_parameter": dict(last_parameter_gradient_norms),
            "prompt_gradient_norm": float(
                math.sqrt(sum(value * value for value in last_gradient_norms.values()))
            ),
            "prompt_parameter_norm": _parameter_norms(model, text_bank)["visual_prompts"],
            "soft_prompt_parameter_norm": _parameter_norms(model, text_bank)[
                "soft_text_prompt"
            ],
            "parameter_counts": _parameter_counts(model, text_bank),
            "geometry": geometry,
            "same_class_sketch_photo_cosine": geometry["cross_modal"][
                "same_class_sketch_photo_cosine"
            ],
            "different_class_sketch_photo_cosine": geometry["cross_modal"][
                "different_class_sketch_photo_cosine"
            ],
            "semantic_margin": geometry["cross_modal"]["semantic_margin"],
            "sketch_reference_cosine": geometry["reference_preservation"]["sketch"],
            "photo_reference_cosine": geometry["reference_preservation"]["photo"],
            "effective_rank": geometry["sketch"]["effective_rank"],
            "linear_cka": geometry["representation_alignment"]["sketch"]["linear_cka"],
            "orthogonal_procrustes_residual": geometry["representation_alignment"][
                "sketch"
            ]["orthogonal_procrustes_residual"],
            "visual_embedding_max_abs_delta": max(
                float((current_sketch.embeddings - vanilla_sketch.embeddings).abs().max()),
                float((loaded_photo.embeddings - vanilla_photo.embeddings).abs().max()),
            ),
            "prompt_attention": model.attention_diagnostics_by_block(
                attention_images.to(device), prompt="sketch"
            ),
            "clip_freeze_policy": clip_policy,
            "optimizer_groups": optimizer_groups,
            "trainable_parameter_names": [
                name for name, parameter in _parameter_names(model, text_bank).items()
                if parameter.requires_grad
            ],
            "pseudo_split_identity": split_identity,
            "class_list_hashes": {
                "train": canonical_sha256(split_identity["train_class_ids"]),
                "validation": canonical_sha256(split_identity["validation_class_ids"]),
            },
            "manifest_identity": data_manifest_identity,
            "manifest_entry_identity": entry_identity,
            "official_unseen_used_for_selection": False,
            "protocol": {
                "selection_metric": "full_pseudo_unseen_mAP",
                "official_unseen_used_for_selection": False,
                "text_used_for_inference": False,
                "photo_used_for_inference": False,
                "alignment_targets_from_validation_or_test": False,
                "alignment_anchor": str(args.alignment_anchor),
                "alignment_geometry": str(args.alignment_geometry),
                "gallery_cache_identity": identity,
            },
        }
        history.append(row)
        history.sort(key=lambda value: int(value["training_global_step"]))
        (output_dir / f"probe_step{probe_step}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

    def _diagnostic_classification(
        query_model: Any,
        loader: DataLoader,
        bank: SoftPromptTextBank,
        *,
        tau: float,
        device: torch.device,
    ) -> dict[str, float] | None:
        query_model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=device.type == "cuda")
                labels = batch["label"].long().to(device)
                queries = query_model(images)
                values = bank()
                bank_labels = bank.class_labels.to(device)
                loss, logits = jepa_text_classification_loss(
                    queries, values, bank_labels, labels, temperature=tau, detach_text=True
                )
                total_loss += float(loss.item()) * labels.shape[0]
                total_correct += int(bank_labels[logits.argmax(dim=-1)].eq(labels).sum())
                total += labels.shape[0]
        if total == 0:
            return None
        return {
            "diagnostic_seen_classification_accuracy": total_correct / total,
            "diagnostic_seen_classification_loss": total_loss / total,
            "diagnostic_seen_classification_count": total,
        }

    probe(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    while step < int(args.max_steps):
        for batch in train_loader:
            if step >= int(args.max_steps):
                break
            if (
                calibration_batch_identities is not None
                and calibration_batch_replay_index < len(calibration_batch_identities)
            ):
                if (
                    _calibration_json_value(batch)
                    != calibration_batch_identities[calibration_batch_replay_index]
                ):
                    raise RuntimeError(
                        "training batch does not replay the calibration batch sequence"
                    )
                calibration_batch_replay_index += 1
                if calibration_batch_replay_index == len(calibration_batch_identities):
                    calibration_batch_replay_verified = True
                if gradient_calibration is not None:
                    gradient_calibration["first_batch_replay_verified"] = True
                    gradient_calibration["calibration_batch_replay_verified"] = (
                        calibration_batch_replay_verified
                    )
                    gradient_calibration["calibration_batch_replay_count"] = (
                        calibration_batch_replay_index
                    )
                    gradient_calibration["calibration_batch_replay_prefix_sha256"] = (
                        canonical_sha256(
                            calibration_batch_identities[:calibration_batch_replay_index]
                        )
                    )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            rank, cls, accuracy, alignment = _batch_objectives(
                model,
                text_bank,
                hard_text_values,
                hard_text_labels,
                batch,
                args,
                device,
            )
            alignment_total = rank.new_zeros(()) if alignment is None else alignment.total
            total = (
                float(args.lambda_rank) * rank
                + float(args.lambda_cls) * cls
                + alignment_total
            )
            _check_finite("rank loss", rank)
            _check_finite("classification loss", cls)
            _check_finite("alignment loss", alignment_total)
            _check_finite("total loss", total)
            total.backward()
            _assert_optimizer_gradients(model, text_bank, optimizer_groups)
            last_gradient_norms = _gradient_norms(model, text_bank, optimizer_groups)
            last_parameter_gradient_norms = _parameter_gradient_norms(model, text_bank)
            optimizer.step()
            scheduler.step()
            for name, parameter in _parameter_names(model, text_bank).items():
                _check_finite(f"parameter {name}", parameter)
            step += 1
            last_train = {
                "rank": float(rank.item()),
                "classification": float(cls.item()),
                "accuracy": float(accuracy.item()),
                "alignment": _alignment_metrics(alignment),
            }
            training_history.append(
                {
                    "training_global_step": step,
                    "last_train_batch_rank_loss": last_train["rank"],
                    "last_train_batch_classification_loss": last_train["classification"],
                    "last_train_batch_accuracy": last_train["accuracy"],
                    "last_train_batch_alignment": last_train["alignment"],
                    "gradient_norms": dict(last_gradient_norms),
                    "gradient_norms_by_parameter": dict(last_parameter_gradient_norms),
                }
            )
            if step in probe_steps:
                probe(step)
            if int(args.log_every) and step % int(args.log_every) == 0:
                print(
                    f"step={step} rank={last_train['rank']:.5f} "
                    f"cls={last_train['classification']:.5f} "
                    f"align={last_train['alignment']['total'] if last_train['alignment'] else None}"
                )
    if step != int(args.max_steps):
        raise RuntimeError(f"training stopped at {step}, expected {args.max_steps}")
    if calibration_batch_identities is not None and not calibration_batch_replay_verified:
        raise RuntimeError("training did not verify the calibration batch sequence replay")

    torch.save(
        {
            "format_version": 1,
            "state_dict": {
                name: value.detach().cpu() for name, value in text_bank.state_dict().items()
            },
            "prompt_length": text_bank.prompt_length,
            "class_names_used_for_training": list(text_bank.class_names),
        },
        output_dir / "soft_prompt.pt",
    )
    selected_row = max(
        history,
        key=lambda row: (float(row["full_pseudo_unseen_mAP"]), -int(row["training_global_step"])),
    )
    selected = {
        "selection_metric": "full_pseudo_unseen_mAP",
        "training_global_step": int(selected_row["training_global_step"]),
        "checkpoint": selected_row["checkpoint"],
        "checkpoint_sha256": selected_row["checkpoint_sha256"],
        "full_pseudo_unseen_mAP": selected_row["full_pseudo_unseen_mAP"],
    }
    checkpoints = {
        str(row["training_global_step"]): {
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "training_global_step": row["training_global_step"],
        }
        for row in history
    }
    training_seconds = time.perf_counter() - training_started
    final_clip_policy = _assert_clip_policy(model, clip_before, role="alignment")
    final_clip_policy.update(
        {
            "photo_encoder_frozen": True,
            "visual_projection_frozen": True,
            "text_tower_frozen": True,
        }
    )
    resolved = OmegaConf.to_container(args, resolve=True)
    report = {
        "schema_version": 1,
        "experiment_role": role,
        "campaign": str(args.experiment_campaign),
        "run_kind": str(args.run_kind),
        "dataset": str(data.name),
        "resolved_config": resolved,
        "resolved_treatment": treatment_from_config(resolved),
        "experiment_code_commit": provenance.get("head_commit"),
        "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"),
        "working_tree_state": provenance.get("working_tree_state"),
        "provenance": provenance,
        "seed": seed,
        "training_seed": seed,
        "initial_model_state_hash": initial_hash,
        "initial_text_bank_state_hash": initial_text_bank_hash,
        "initialization_identity": {
            "model_state_hash": initial_hash,
            "text_bank_state_hash": initial_text_bank_hash,
        },
        "pseudo_validation_seed": int(args.pseudo_val_seed),
        "training_class_list": list(split_identity["train_class_ids"]),
        "validation_class_list": list(split_identity["validation_class_ids"]),
        "pseudo_split_identity": split_identity,
        "manifest_identity": data_manifest_identity,
        "manifest_entry_identity": entry_identity,
        "manifest_path": str(manifest_path),
        "class_list_hashes": {
            "train": canonical_sha256(split_identity["train_class_ids"]),
            "validation": canonical_sha256(split_identity["validation_class_ids"]),
        },
        "diagnostic_subset_identity": _entry_identity(diagnostic_entries),
        "diagnostic_subset_selected_before_training": True,
        "official_unseen_used_for_selection": False,
        "optimizer_groups": optimizer_groups,
        "trainable_parameter_names": [
            name for name, parameter in _parameter_names(model, text_bank).items()
            if parameter.requires_grad
        ],
        "frozen_parameter_names": [
            name for name, parameter in _parameter_names(model, text_bank).items()
            if not parameter.requires_grad
        ],
        "clip_freeze_policy": final_clip_policy,
        "parameter_counts": _parameter_counts(model, text_bank),
        "matched_sampler": {
            "type": "MatchedClassBatchSampler",
            "classes_per_batch": int(args.classes_per_batch),
            "sketches_per_class": int(args.sketches_per_class),
            "batch_size": int(args.batch_size),
            "batches_per_epoch": len(train_loader),
            "seed": seed,
            "positive_photos_per_sketch": int(args.num_positive_photos),
            "negative_photos_per_sketch": 1,
        },
        "objective": {
            "name": "class_conditional_spherical_moment_alignment",
            "sketch_distribution": "matched class sketches",
            "photo_distribution": "matched class positive photos",
            "target_gradient": str(args.alignment_target_gradient),
            "text_anchor": str(args.alignment_anchor) == "text",
            "anchor_mode": str(args.alignment_anchor),
            "geometry": str(args.alignment_geometry),
            "mean_weight": float(args.lambda_alignment_mean),
            "covariance_weight": float(args.lambda_alignment_covariance),
            "text_bank_for_anchor": "frozen hard CLIP bank" if str(args.alignment_anchor) == "text" else None,
        },
        "checkpoint_state_fields": [
            "model_state_dict",
            "soft_prompt_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
            "training_global_step",
        ],
        "checkpoints": checkpoints,
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "history": history,
        "training_history": training_history,
        "selection": selected,
        "gradient_calibration": gradient_calibration,
        "gradient_validation": {
            "active_optimizer_groups_have_nonzero_last_update": {
                group["name"]: bool(
                    group["active"] and last_gradient_norms[group["name"]] > 0.0
                )
                for group in optimizer_groups
            },
            "last_update_gradient_norms": dict(last_gradient_norms),
            "last_update_gradient_norms_by_parameter": dict(last_parameter_gradient_norms),
        },
        "runtime": {
            "training_seconds": training_seconds,
            "updates_this_run": step,
            "seconds_per_update": training_seconds / step,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None,
        },
        "inference_contract": dict(CORRECTED_PILOT_INFERENCE_CONTRACT),
        "protocol": {
            "selection_metric": "full_pseudo_unseen_mAP",
            "official_unseen_used_for_selection": False,
            "train_class_scope": "pseudo_train",
            "alignment_fit_scope": "pseudo_train_only",
            "validation_used_for_alignment": False,
            "test_used_for_alignment": False,
            "text_used_for_predictor": False,
            "photo_used_for_predictor": False,
            "ranking_positive_reduction": "mean_over_4_positive_photos",
            "resume": "not_enabled; all campaign runs from scratch",
        },
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "run_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


@hydra.main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="train_alignment")
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
