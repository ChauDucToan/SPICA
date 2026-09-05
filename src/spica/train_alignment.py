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
            and role in CORRECTED_PILOT_ROLES
            and key == "lambda_alignment_mean"
        )
    }
    if mismatches:
        raise ValueError(f"{role} has an ambiguous treatment: {mismatches}")
    if (
        campaign == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN
        and role in {"alignment_mean_text_log", "alignment_mean_text_log_symmetric"}
        and not bool(args.calibration_only)
        and args.alignment_calibration_artifact is None
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
    clip_freeze_policy: dict[str, Any],
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
            "step": step,
            "training_global_step": step,
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
            "resolved_config": OmegaConf.to_container(args, resolve=True),
            "resolved_treatment": treatment,
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


def _grad_norms_for(
    gradients: tuple[torch.Tensor | None, ...], parameters: tuple[torch.Tensor, ...]
) -> list[float]:
    return [
        0.0 if gradient is None else float(gradient.detach().norm().item())
        for gradient, _ in zip(gradients, parameters)
    ]


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
    """Calibrate one fixed mean-loss weight without updating model/RNG state."""
    target_ratio = float(args.calibration_target_ratio)
    count = int(args.calibration_batches)
    saved_rng = capture_rng_state(loader_generator)
    saved_epoch = getattr(sampler, "_epoch", None)
    batches: list[dict[str, Any]] = []
    try:
        iterator = iter(train_loader)
        for _ in range(count):
            try:
                batches.append(next(iterator))
            except StopIteration as error:
                raise ValueError("not enough fixed pseudo-train batches for calibration") from error
    finally:
        restore_rng_state(saved_rng, loader_generator)
        if saved_epoch is not None:
            sampler._epoch = saved_epoch

    sketch_prompt = model.sketch_prompt
    photo_prompt = model.photo_prompt
    prompt_parameters = (sketch_prompt, photo_prompt)
    base_norms: list[float] = []
    mean_norms: list[float] = []
    photo_base_norms: list[float] = []
    photo_mean_norms: list[float] = []
    cosines: list[float] = []
    for batch in batches:
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
        base_grads = torch.autograd.grad(base, prompt_parameters, allow_unused=True)
        base_norm = _grad_norms_for(base_grads, prompt_parameters)[0]
        photo_base_norm = _grad_norms_for(base_grads, prompt_parameters)[1]
        _, _, _, mean_alignment = _batch_objectives(
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
        if mean_alignment is None:
            raise RuntimeError("mean calibration did not produce a mean alignment loss")
        mean_grads = torch.autograd.grad(
            mean_alignment.mean, prompt_parameters, allow_unused=True
        )
        mean_norm = _grad_norms_for(mean_grads, prompt_parameters)[0]
        photo_mean_norm = _grad_norms_for(mean_grads, prompt_parameters)[1]
        if base_norm <= 1e-12 or mean_norm <= 1e-12:
            raise ValueError(
                "mean calibration encountered a near-zero sketch-prompt gradient; "
                "choose a different fixed batch rule"
            )
        base_vector = base_grads[0]
        mean_vector = mean_grads[0]
        assert base_vector is not None and mean_vector is not None
        cosine = float(
            torch.nn.functional.cosine_similarity(
                base_vector.reshape(1, -1), mean_vector.reshape(1, -1)
            ).item()
        )
        base_norms.append(base_norm)
        mean_norms.append(mean_norm)
        photo_base_norms.append(photo_base_norm)
        photo_mean_norms.append(photo_mean_norm)
        cosines.append(cosine)

    ratios = [base / mean for base, mean in zip(base_norms, mean_norms)]
    calibrated = target_ratio * statistics.median(ratios)
    weighted_ratios = [calibrated * mean / base for base, mean in zip(base_norms, mean_norms)]
    photo_ratios = [
        None
        if base <= 1e-12
        else calibrated * mean / base
        for base, mean in zip(photo_base_norms, photo_mean_norms)
    ]
    return {
        "rule": "lambda = target_ratio * median(base_sketch_norm / mean_norm)",
        "target_ratio": target_ratio,
        "lambda_alignment_mean": calibrated,
        "batches": count,
        "base_sketch_gradient_norms": base_norms,
        "mean_sketch_gradient_norms": mean_norms,
        "unweighted_sketch_norm_ratios": ratios,
        "weighted_sketch_gradient_ratios": weighted_ratios,
        "gradient_cosines_sketch": cosines,
        "base_photo_gradient_norms": photo_base_norms,
        "mean_photo_gradient_norms": photo_mean_norms,
        "weighted_photo_gradient_ratios_if_symmetric": photo_ratios,
        "median_actual_sketch_ratio": statistics.median(weighted_ratios),
        "mean_actual_sketch_ratio": statistics.fmean(weighted_ratios),
        "rng_and_sampler_state_restored": True,
    }


def run(args: DictConfig) -> None:
    _validate(args)
    seed = int(args.seed)
    _seed(seed)
    device = _device(str(args.device))
    data = load_data_config(_path(args.data_config))
    split, names, split_identity, data_manifest_identity = _load_split(data, args)
    manifest_path = _path(args.experiment_manifest_path)
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
    calibration_artifact = args.alignment_calibration_artifact
    if calibration_artifact is not None:
        calibration_path = _path(calibration_artifact)
        if not calibration_path.is_file():
            raise FileNotFoundError(f"alignment calibration artifact not found: {calibration_path}")
        calibration_payload = json.loads(calibration_path.read_text())
        calibrated_lambda = float(calibration_payload["lambda_alignment_mean"])
        if not math.isclose(
            calibrated_lambda,
            float(args.lambda_alignment_mean),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "configured lambda_alignment_mean does not match calibration artifact"
            )
        gradient_calibration = {
            "artifact": str(calibration_path.resolve()),
            "artifact_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            "lambda_alignment_mean": calibrated_lambda,
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
            clip_freeze_policy=clip_policy,
        )
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        current_sketch = encode_prompted_loader(model, val_sketch_loader)
        current_photo = encode_prompted_loader(model, val_photo_loader, photo=True)
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
        "inference_contract": {
            "required_inputs": ["raw_sketch_image"],
            "text_required": False,
            "photo_required": False,
            "oracle_class_required": False,
            "text_used_for_predictor": False,
            "photo_prompt_used_for_query": False,
            "photo_prompt_used_for_gallery": True,
        },
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
