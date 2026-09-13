"""Standalone official Sketchy/TU-Berlin/QuickDraw F2 MP-Q trainer.

Legacy mode uses a seen-train probe and a separate final evaluator. Explicit
--periodic-test pauses training for full official test at five 20% boundaries, without
checkpoint selection or changing the model, objective, or training stream.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback
from typing import Any, Mapping

import torch
from torch.optim.lr_scheduler import LambdaLR

from .coupled_predictive_losses import coupled_region_loss
from .data import coupled_benchmark as benchmark_data
from .evaluation.periodic_test import evaluate_periodic_test
from .evaluation.training_probe import TrainingProbe
from .runtime import official_test_steps, runtime_policy
from .models.clip import load_frozen_clip
from .models.coupled_predictive import CoupledPredictiveModel
from .provenance import capture_provenance, capture_rng_state
from .tracking.wandb import WandbExperiment
from .train_coupled_predictive import (
    BATCH_SIZE,
    CLIP_MODEL,
    _batch_to_device,
    _checkpoint_payload,
    _device,
    _append_jsonl,
    _gradient_diagnostics,
    _initialization_hashes,
    _json,
    _loader_cycle,
    _make_optimizer,
    _save_checkpoint,
    _seed,
    _sha256_file,
    _state_hash,
    _verify_clip,
    _verify_source,
)

ROOT = Path(__file__).resolve().parents[2]
METHOD_VERSION = "coupled_predictive_mp_official_v1"
ARM = "F2_MP_Q_OFFICIAL"
SKETCH_REF_ARM = "F2_MP_Q_SREF_OFFICIAL"
SKETCH_REF_METHOD = "coupled_predictive_mp_sketch_ref_official_v1"
MAIN_OBJECTIVE = "multi_positive_supervised_contrastive"
TEMPERATURE = 0.07
SEED = 42
LOCAL_LOSS_LOG_EVERY = 5
DATASETS: dict[str, dict[str, Any]] = {
    "sketchy_104_21": {
        "config": "configs/data/sketchy_104_21.yaml",
        "total_steps": 4446,
        "warmup_steps": 222,
    },
    "tuberlin_220_30": {
        "config": "configs/data/tuberlin_220_30.yaml",
        "total_steps": 1189,
        "warmup_steps": 59,
    },
    "quickdraw_80_30": {
        "config": "configs/data/quickdraw_80_30.yaml",
        "total_steps": 18229,
        "warmup_steps": 911,
    },
}


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_routing(
    dataset: str,
    *,
    main_photo_objective: str = MAIN_OBJECTIVE,
    lambda_sketch_ref: object = None,
) -> dict[str, Any]:
    """Reject accidental routes before device, output, or data side effects."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {tuple(DATASETS)}")
    if main_photo_objective != MAIN_OBJECTIVE:
        raise ValueError(
            f"{ARM} requires the historical F2_MP objective {MAIN_OBJECTIVE!r}; "
            "pooled q multi-positive routing is forbidden"
        )
    if lambda_sketch_ref is None:
        return {
            "arm": ARM,
            "method_version": METHOD_VERSION,
            "main_photo_objective": MAIN_OBJECTIVE,
            "main_query": "mu_i",
            "training_main_query": "mu_i",
            "evaluation_query": "q",
            "evaluation_adapter": "QOnlyAdapter;predictor_forwards=0",
        }
    if dataset != "tuberlin_220_30":
        raise ValueError(f"{SKETCH_REF_ARM} is authorized for TU-Berlin only")
    if isinstance(lambda_sketch_ref, bool) or not isinstance(lambda_sketch_ref, (int, float)):
        raise ValueError("lambda_sketch_ref must be a finite non-negative scalar")
    try:
        valid = math.isfinite(lambda_sketch_ref) and lambda_sketch_ref >= 0.0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("lambda_sketch_ref must be a finite non-negative scalar")
    return {
        "arm": SKETCH_REF_ARM,
        "method_version": SKETCH_REF_METHOD,
        "main_photo_objective": MAIN_OBJECTIVE,
        "main_query": "mu_i",
        "training_main_query": "mu_i",
        "evaluation_query": "q",
        "evaluation_adapter": "QOnlyAdapter;predictor_forwards=0",
        "lambda_sketch_ref": float(lambda_sketch_ref),
    }


def _schedule(total_steps: int, warmup_steps: int):
    if total_steps < 1 or not 1 <= warmup_steps <= total_steps:
        raise ValueError("invalid locked schedule")

    def value(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps == warmup_steps:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    return value


def _optimizer_gate(optimizer: Any) -> dict[str, Any]:
    """Check every AdamW moment once, after the requested updates finish."""
    moment_tensors = [
        value
        for state in optimizer.state.values()
        for name, value in state.items()
        if name != "step" and isinstance(value, torch.Tensor)
    ]
    return {
        "state_tensor_count": len(moment_tensors),
        "finite": bool(moment_tensors) and all(bool(torch.isfinite(value).all()) for value in moment_tensors),
        "nonzero_moment": bool(moment_tensors) and all(bool(value.detach().abs().sum() > 0) for value in moment_tensors),
    }


def _copy_source_archive(output: Path, source: Mapping[str, Any]) -> None:
    snapshot = source.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("provenance source snapshot is unavailable")
    archive = output / "source_snapshot" / "files"
    archive.mkdir(parents=True, exist_ok=False)
    for item in snapshot.get("manifest", ()):
        relative = Path(str(item["path"]))
        source_path = ROOT / relative
        destination = archive / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"source disappeared while archiving: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        if _sha256_file(destination) != str(item["sha256"]):
            raise RuntimeError(f"source archive hash mismatch: {relative}")
    _json(output / "source_snapshot" / "index.json", snapshot)


def _config(
    *,
    args: argparse.Namespace,
    protocol: Any,
    clip_identity: Mapping[str, Any],
    initialization: Mapping[str, Any],
    source_hash: str,
    model: CoupledPredictiveModel,
) -> dict[str, Any]:
    dataset_info = DATASETS[args.dataset]
    train_class_names = {int(k): str(v) for k, v in protocol.train.class_names.items()}
    coefficients = {
        "rank_i": 1.0,
        "ce_i": 1.0,
        "ce_t_aux": 0.25,
        "rank_pool": 0.25,
        "ce_pool": 0.25,
        "align_i": 0.05,
        "align_t": 0.05,
        "anchor_i": 0.5,
        "anchor_t": 0.5,
        "sigreg": 0.0,
    }
    sketch_ref_enabled = getattr(args, "lambda_sketch_ref", None) is not None
    if sketch_ref_enabled:
        coefficients["sketch_ref"] = float(args.lambda_sketch_ref)
    config = _safe({
        "arm": SKETCH_REF_ARM if sketch_ref_enabled else ARM,
        "method_version": SKETCH_REF_METHOD if sketch_ref_enabled else METHOD_VERSION,
        "dataset": args.dataset,
        "dataset_config": str(protocol.config_path),
        "dataset_config_sha256": protocol.config_sha256,
        "device": args.device,
        "wandb_mode": args.wandb_mode,
        "smoke": bool(args.smoke),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True,
        "total_steps": dataset_info["total_steps"],
        "max_steps": 2 if args.smoke else dataset_info["total_steps"],
        "actual_updates": 2 if args.smoke else dataset_info["total_steps"],
        "locked_total_steps": dataset_info["total_steps"],
        "warmup_steps": dataset_info["warmup_steps"],
        "warmup_rounding": "Python round(0.05 * locked_total_steps)",
        "locked_warmup_steps": dataset_info["warmup_steps"],
        "realized_warmup_fraction": dataset_info["warmup_steps"] / dataset_info["total_steps"],
        "schedule": "warmup_then_cosine;5_percent_locked",
        "architecture": "predictive_fusion_v2",
        "positive_pool": "full_official_train_photo_manifest",
        "main_photo_objective": MAIN_OBJECTIVE,
        "main_photo_temperature": TEMPERATURE,
        "objective_identity": "historical_F2_MP_total;main_MP(mu_i)_coefficient1;paired_q_auxiliaries_unchanged",
        "loss_coefficient_identity": coefficients,
        "training_main_query": "mu_i",
        "evaluation_query": "q",
        "evaluation_adapter": "QOnlyAdapter;pooled_head(context.mean(dim=1));predictor_forwards=0",
        "official_unseen_used_for_training": False,
        "official_unseen_used_for_selection": False,
        "official_unseen_evaluation": "final_only_separate_evaluator",
        "selection_policy": "none;final_only",
        "clip_identity": dict(clip_identity),
        "train_class_ids": list(protocol.train.class_ids),
        "train_class_names": train_class_names,
        "classmap_sha256": _sha_json(train_class_names),
        "protocol_identity": _safe(protocol.identity),
        "initialization_hashes": initialization,
        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_total_parameters": sum(p.numel() for p in model.parameters()),
        "frozen_original_state_hash": _state_hash(model, prefix="original_clip."),
        "source_snapshot_hash": source_hash,
        "sampling_identity": {
            "positive_rule": "uniform_photo_within_query_class",
            "negative_rule": "uniform_photo_from_other_train_class",
            "photo_pool": "all_official_train_photos",
            "canonical_pairing": "not_loaded;not_used",
        },
        "observation_passes": {
            "clean": 1, "corrupted": 1, "rows_per_update": BATCH_SIZE,
            "trace_rows_expected": BATCH_SIZE * (2 if args.smoke else dataset_info["total_steps"]),
            "mask_rows_expected": BATCH_SIZE * (2 if args.smoke else dataset_info["total_steps"]),
        },
        "batch_identity": {
            "batch_size": BATCH_SIZE, "full_batch_required": True,
            "drop_last": True, "num_workers": 4, "pin_memory": True,
        },
        "optimizer": {
            "name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
            "groups": [
                {"name": str(group["name"]), "lr": float(group["lr"]),
                 "weight_decay": float(group["weight_decay"]),
                 "parameter_count": sum(parameter.numel() for parameter in group["params"])}
                for group in model.optimizer_parameter_groups()
            ],
        },
        "checkpoint_policy": "step0_and_final_only",
        "metrics": None,
    })
    if sketch_ref_enabled:
        config.update({
            "sketch_ref_lambda": float(args.lambda_sketch_ref),
            "lambda_sketch_ref": float(args.lambda_sketch_ref),
            "official_baseline_seen_before_design": True,
            "blind_holdout_claim": False,
            "objective_identity": "historical_F2_MP_total_plus_masked_original_clip_sketch_ref;clean_q_student",
            "sketch_ref_identity": {
                "coefficient": float(args.lambda_sketch_ref),
                "target": "masked_corrupted_sketch_original_clip_teacher",
                "student": "clean_q",
                "temperature": TEMPERATURE,
                "views": "corrupted_teacher_to_clean_student",
                "direction": "teacher_rows_student_columns",
                "instance_nce": True,
                "official_baseline_seen_before_design": True,
            },
        })
    return config


def _save_run_checkpoint(
    output: Path,
    *,
    step: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: Mapping[str, Any],
    source_hash: str,
    clip_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    loader: Any,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    rolling = config.get("checkpoint_policy") == "rolling_latest"
    path = output / ("checkpoint_latest.pt" if rolling else f"checkpoint_step{step}.pt")
    payload = _checkpoint_payload(
        model, optimizer, scheduler, None,
        step=step, config=config, source_hash=source_hash,
        clip=clip_identity, data_identity=data_identity,
        rng=capture_rng_state(generator=getattr(loader, "generator", None)),
        selections={"latest": {"step": step, "metrics": None}},
        initialization_hashes=initialization,
    )
    if rolling:
        learned = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        payload["model_state_dict"] = {name: value for name, value in payload["model_state_dict"].items() if name in learned}
        payload["state_format"] = "trainable_only_v1"
    # Loader worker/iterator state is not captured: do not claim bit-exact resume.
    payload["exact_data_stream_resume"] = False
    sha = _save_checkpoint(path, payload, overwrite=rolling)
    record = {
        "step": step,
        "path": path.name,
        "sha256": sha,
        "model_state_hash": payload["model_state_hash"],
        "initialization_hashes": initialization,
    }
    if rolling:
        temporary = output / "checkpoint_latest.json.tmp"
        _json(temporary, record)
        temporary.replace(output / "checkpoint_latest.json")
    return record


def _wandb_config(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "arm", "dataset", "seed", "batch_size", "total_steps", "warmup_steps",
        "architecture", "evaluation_query", "source_snapshot_hash", "runtime",
        "lambda_sketch_ref", "official_baseline_seen_before_design", "blind_holdout_claim",
    )
    values = _safe({key: config[key] for key in keys if key in config})
    if config.get("tracking_policy") == "official_test_v1":
        values.update({key: config[key] for key in (
            "tracking_policy", "probe_scope", "official_unseen_evaluation", "test_steps",
        )})
    return values


def _test_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for scope, source in (("cleaned", "clean"), ("masked", "masked_macro")):
        values = report[source]
        metrics.update({
            f"test/{scope}/mAP@200": float(values["mAP@200_min_relevant_k"]),
            f"test/{scope}/mAP@all": float(values["full_mAP"]),
            f"test/{scope}/P@200": float(values["P@200"]),
        })
    return metrics


def _periodic_test(
    *,
    model: Any,
    protocol: Any,
    transform: Any,
    output: Path,
    step: int,
    checkpoint: Mapping[str, Any],
    source_hash: str,
    device: torch.device,
    final: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    test_path = output / "test" / f"step_{step}"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate_periodic_test(
        model, protocol, transform, test_path, device=device,
        source_hash=source_hash, checkpoint_selection=checkpoint, final=final,
    )
    summary_path = test_path / "summary.json"
    summary_sha = _sha256_file(summary_path)
    record = {
        "step": step,
        "path": str(test_path.relative_to(output)),
        "summary": str(summary_path.relative_to(output)),
        "summary_sha256": summary_sha,
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "final": final,
        "evaluation_scope": report.get("evaluation_scope", "official_test_periodic_monitoring_no_selection"),
    }
    return report, record


def _train_impl(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    lambda_sketch_ref = getattr(args, "lambda_sketch_ref", None)
    periodic_test = bool(getattr(args, "periodic_test", False))
    runtime_overrides = getattr(args, "runtime", None)
    if periodic_test:
        policy = runtime_policy({"test_every_percent": 20, **dict(runtime_overrides or {})})
    else:
        policy = runtime_policy(runtime_overrides)
        if "test_every_percent" in policy:
            raise ValueError("--periodic-test is required for official test cadence")
    routing = _validate_routing(args.dataset, lambda_sketch_ref=lambda_sketch_ref)
    dataset_info = DATASETS[args.dataset]
    actual_updates = 2 if args.smoke else int(dataset_info["total_steps"])
    horizon_steps = int(dataset_info["total_steps"])
    warmup_steps = int(dataset_info["warmup_steps"])
    test_steps = official_test_steps(horizon_steps) if periodic_test else []

    # Validate manifests and data identity before touching the requested device.
    protocol = benchmark_data.load_benchmark_protocol(dataset_info["config"], split="train")
    clip_identity = _verify_clip()
    device = _device(args.device)
    _seed(SEED)
    bundle = load_frozen_clip(model_name=CLIP_MODEL, pretrained=str(clip_identity["path"]), device=device)
    model = CoupledPredictiveModel(
        bundle.encoder, bundle.tokenizer,
        {int(k): str(v) for k, v in protocol.train.class_names.items()},
        architecture="predictive_fusion_v2",
    ).to(device)
    model.train(True)
    initialization = _initialization_hashes(model)
    frozen_before = _state_hash(model, prefix="original_clip.")
    config = _config(
        args=args, protocol=protocol, clip_identity=clip_identity,
        initialization=initialization, source_hash="pending", model=model,
    )
    config["total_steps"] = horizon_steps
    config["max_steps"] = actual_updates
    config["actual_updates"] = actual_updates
    config["warmup_steps"] = warmup_steps
    config.update(routing)
    config["runtime"] = policy
    config["test_steps"] = test_steps
    config["tracking_policy"] = "official_test_v1" if periodic_test else "minimal_v1"
    config["checkpoint_policy"] = "step0_and_final_only" if args.smoke else "rolling_latest"
    config["probe_scope"] = (
        "disabled_official_periodic_test" if periodic_test
        else "disabled_smoke" if args.smoke else "seen_train_probe_not_validation"
    )
    config["official_unseen_evaluation"] = (
        "periodic_monitoring_no_selection" if periodic_test
        else "final_only_separate_evaluator"
    )

    source = capture_provenance(ROOT, resolved_config=config)
    source_hash = _verify_source(source, Path(args.campaign_root).expanduser())
    config["source_snapshot_hash"] = source_hash
    source["resolved_config"] = config
    _copy_source_archive(output, source)
    _json(output / "provenance.json", source)
    _json(output / "resolved_config.json", config)
    _json(output / "initialization.json", initialization)

    optimizer = _make_optimizer(model)
    scheduler = LambdaLR(optimizer, lr_lambda=_schedule(horizon_steps, warmup_steps))
    loader = benchmark_data.make_train_loader(
        protocol, bundle.transform, batch_size=BATCH_SIZE, num_workers=4,
        pin_memory=True, drop_last=True, seed=SEED,
    )
    if loader.generator is None:
        raise RuntimeError("official loader must expose its seeded generator")
    probe = None if args.smoke or periodic_test else TrainingProbe.from_protocol(protocol, bundle.transform)
    if probe is not None:
        _json(output / "probe_manifest.json", probe.summary)
    test_evaluations: list[dict[str, Any]] = []
    test_metrics_path = output / "test_metrics.jsonl"
    test_evaluations_path = output / "test_evaluations.jsonl"
    batches = _loader_cycle(loader)
    data_identity = _safe(protocol.identity)
    checkpoints = [_save_run_checkpoint(
        output, step=0, model=model, optimizer=optimizer, scheduler=scheduler,
        config=config, source_hash=source_hash, clip_identity=clip_identity,
        data_identity=data_identity, loader=loader, initialization=initialization,
    )] if args.smoke else []
    history_path = output / "training_history.jsonl"
    lr_path = output / "lr_history_every_step.jsonl"
    trace_path = output / "observation_trace.jsonl"
    mask_path = output / "mask_metadata.jsonl"
    history = history_path.open("w", encoding="utf-8")
    lr_history = lr_path.open("w", encoding="utf-8")
    trace_handle = trace_path.open("w", encoding="utf-8")
    mask_handle = mask_path.open("w", encoding="utf-8")
    test_metrics_handle = test_metrics_path.open("w", encoding="utf-8") if periodic_test and not args.smoke else None
    test_evaluations_handle = test_evaluations_path.open("w", encoding="utf-8") if periodic_test and not args.smoke else None
    wandb_run: WandbExperiment | None = None
    last_losses: dict[str, float] = {}
    last_actual_lr: dict[str, float] = {}
    trace_count = 0
    mask_count = 0
    model_state_before_updates = _state_hash(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_baseline = {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device)}
    else:
        memory_baseline = {}
    try:
        if args.wandb_mode != "disabled":
            wandb_run = WandbExperiment(
                project="spica", entity="a-cctest05187-erd",
                name=f"{config['arm']}-{args.dataset}", group=config["method_version"],
                job_type="train", config=_wandb_config(config),
                mode=args.wandb_mode, directory=output,
            )
            _json(output / "wandb_runtime.json", {"run_id": wandb_run.run_id, "run_url": wandb_run.run_url})
        if wandb_run is not None:
            wandb_run.define_metric("step_train")
            prefix = "test/" if periodic_test else ""
            axis = "progress_percent" if periodic_test else "step_train"
            if periodic_test:
                wandb_run.define_metric(axis)
            for scope in ("cleaned", "masked"):
                wandb_run.define_metric(f"{prefix}{scope}/*", step_metric=axis)
            if periodic_test and not args.smoke:
                photos = Counter(entry.label for entry in protocol.test.photo_entries)
                p_all = sum(photos[entry.label] for entry in protocol.test.sketch_entries) / (len(protocol.test.sketch_entries) * len(protocol.test.photo_entries))
                wandb_run.set_summary({
                    "test/scope": "official_unseen_periodic_monitoring_no_selection",
                    "test/cleaned/P@all": p_all, "test/masked/P@all": p_all,
                    "test/mAP@200_definition": "sum(P(k)*relevance(k), k<=200)/min(R,200)",
                    "test/query_count": len(protocol.test.sketch_entries),
                    "test/gallery_count": len(protocol.test.photo_entries),
                })
            if probe is not None:
                wandb_run.set_summary({
                    "probe/scope": config["probe_scope"], "probe/query_count": 32,
                    "probe/gallery_count": 256, "probe/manifest_hash": probe.manifest_hash,
                    "probe/cleaned/P@all": probe.summary["P@all"]["clean"],
                    "probe/masked/P@all": probe.summary["P@all"]["masked"],
                    "mAP@200_definition": "sum(P(k)*relevance(k), k<=200)/min(R,200)",
                })
        initial_lr = {f"group_{i}": float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}
        _append_jsonl(lr_history, {"step": 0, "lr": initial_lr})
        _append_jsonl(history, {"step": 0, "loss": None, "lr": initial_lr, "gradient_norm": None})
        for step in range(1, actual_updates + 1):
            raw = next(batches)
            batch = benchmark_data.prepare_batch(protocol, raw, step - 1)
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = coupled_region_loss(
                model, batch["clean"], batch["corrupted"], batch["photos"],
                batch["positive_indices"], batch["negative_indices"], batch["labels"],
                batch["photo_labels"], batch["photo_ids"], lambda_sig=0.0, sigreg=None,
                main_photo_objective=MAIN_OBJECTIVE,
                lambda_sketch_ref=lambda_sketch_ref,
            )
            if not bool(torch.isfinite(losses["total"]).item()):
                raise FloatingPointError(f"nonfinite total loss at step {step}")
            losses["total"].backward()
            loss_log_every = policy["probe_every"] if not periodic_test else LOCAL_LOSS_LOG_EVERY
            log_step = step % loss_log_every == 0 or step == actual_updates
            gradient_norms = _gradient_diagnostics(model, optimizer)
            if not all(math.isfinite(value) for value in gradient_norms.values()):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            actual_lr = {f"group_{i}": float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
            last_actual_lr = actual_lr
            _append_jsonl(lr_history, {"step": step, "lr": actual_lr})
            for item in batch.get("trace", ()):
                trace_row = dict(item)
                trace_row.update({"step": step, "actual_lr": actual_lr["group_0"]})
                _append_jsonl(trace_handle, trace_row)
                trace_count += 1
            for item in batch.get("mask_metadata", {}).get("rows", ()):
                mask_row = dict(item)
                mask_row["step"] = step
                _append_jsonl(mask_handle, mask_row)
                mask_count += 1
            if log_step:
                _append_jsonl(history, {"step": step, "loss": last_losses, "lr": actual_lr, "gradient_norm": gradient_norms})
                for handle in (history, lr_history, trace_handle, mask_handle):
                    handle.flush()
            if probe is not None and step % policy["probe_every"] == 0:
                measured = probe(model, device)
                metrics = {f"{scope}/{key}": value for scope in ("cleaned", "masked")
                           for key, value in measured[scope].items()}
                with (output / "probe_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    _append_jsonl(handle, {"step_train": step, **metrics})
                if wandb_run is not None:
                    wandb_run.log_metrics({"step_train": step, **metrics}, step=step)

            checkpoint_due = (
                not args.smoke and step % policy["checkpoint_every"] == 0
                and step < actual_updates
            )
            test_due = periodic_test and not args.smoke and step in test_steps
            checkpoint = None
            if checkpoint_due or test_due:
                checkpoint = _save_run_checkpoint(
                    output, step=step, model=model, optimizer=optimizer, scheduler=scheduler,
                    config=config, source_hash=source_hash, clip_identity=clip_identity,
                    data_identity=data_identity, loader=loader, initialization=initialization,
                )
            if test_due:
                # All train metadata is materialized before releasing the graph/batch.
                del losses, batch, raw
                report, record = _periodic_test(
                    model=model, protocol=protocol, transform=bundle.transform, output=output,
                    step=step, checkpoint=checkpoint, source_hash=source_hash,
                    device=device, final=step == actual_updates,
                )
                test_evaluations.append(record)
                _append_jsonl(test_metrics_handle, {"step_train": step, **_test_metrics(report)})
                _append_jsonl(test_evaluations_handle, record)
                test_metrics_handle.flush()
                test_evaluations_handle.flush()
                if wandb_run is not None:
                    wandb_run.log_test_retrieval(step, report, total_steps=config["total_steps"])
            if checkpoint is not None and not test_due:
                del checkpoint

        final = checkpoint if periodic_test and not args.smoke and test_evaluations and test_evaluations[-1]["step"] == actual_updates else None
        if final is None:
            final = _save_run_checkpoint(
                output, step=actual_updates, model=model, optimizer=optimizer, scheduler=scheduler,
                config=config, source_hash=source_hash, clip_identity=clip_identity,
                data_identity=data_identity, loader=loader, initialization=initialization,
            )
        checkpoints.append(final)
        expected_rows = actual_updates * BATCH_SIZE
        if trace_count != expected_rows or mask_count != expected_rows:
            raise RuntimeError(f"observation/mask count mismatch: trace={trace_count}, mask={mask_count}, expected={expected_rows}")
        optimizer_gate = _optimizer_gate(optimizer)
        if not optimizer_gate["finite"] or not optimizer_gate["nonzero_moment"]:
            raise FloatingPointError(f"final optimizer moment gate failed: {optimizer_gate}")
        model_state_after_updates = _state_hash(model)
        if model_state_after_updates == model_state_before_updates:
            raise RuntimeError("model state did not change after optimizer updates")
        selections = {"latest": {"step": actual_updates, "path": final["path"], "sha256": final["sha256"], "metrics": None}}
        _json(output / "selections.json", selections)
        final_source = capture_provenance(ROOT, resolved_config=config)
        final_source_hash = _verify_source(final_source, Path(args.campaign_root).expanduser())
        _json(output / "provenance_after.json", final_source)
        if final_source_hash != source_hash:
            raise RuntimeError("source snapshot changed during training")
        frozen_after = _state_hash(model, prefix="original_clip.")
        if frozen_after != frozen_before:
            raise RuntimeError("frozen original CLIP state changed")
        result = {
            "status": "COMPLETE", "campaign": config["method_version"], "method_version": config["method_version"],
            "arm": config["arm"], "dataset": args.dataset, "step": actual_updates,
            "completed_steps": actual_updates, "steps_executed": actual_updates,
            "max_steps": actual_updates, "horizon_steps": horizon_steps,
            "source_snapshot_hash": source_hash, "clip_identity": clip_identity,
            "data_identity": data_identity,
            "source_snapshot_hash_after": final_source_hash,
            "checkpoints": checkpoints, "selections": selections, "metrics": None,
            "training_history": history_path.name, "lr_history": lr_path.name,
            "observation_trace": trace_path.name, "trace_count": trace_count,
            "expected_trace_count": expected_rows, "mask_metadata": mask_path.name,
            "mask_count": mask_count, "expected_mask_count": expected_rows,
            "initialization_hashes": initialization,
            "model_state_hash_before_updates": model_state_before_updates,
            "model_state_hash_after_updates": model_state_after_updates,
            "last_actual_lr": last_actual_lr,
            "optimizer": config["optimizer"],
            "warmup_steps": warmup_steps,
            "realized_warmup_fraction": warmup_steps / horizon_steps,
            "frozen_original_state_hash_before": frozen_before,
            "frozen_original_state_hash_after": frozen_after,
            "optimizer_moment_gate": optimizer_gate,
            "memory": {"baseline": memory_baseline, "peak_allocated": int(torch.cuda.max_memory_allocated(device)), "peak_reserved": int(torch.cuda.max_memory_reserved(device))} if device.type == "cuda" else {},
            "official_unseen_used_for_training": False,
            "official_unseen_used_for_selection": False,
            "official_unseen_evaluation": config["official_unseen_evaluation"],
            "official_test_evaluations": test_evaluations,
            "wandb_run_id": None if wandb_run is None else wandb_run.run_id,
            "wandb_url": None if wandb_run is None else wandb_run.run_url,
        }
        _json(output / "run_result.json", result)
        if wandb_run is not None:
            wandb_run.finish()
        return result
    except BaseException:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise
    finally:
        history.close()
        lr_history.close()
        trace_handle.close()
        mask_handle.close()
        if test_metrics_handle is not None:
            test_metrics_handle.close()
        if test_evaluations_handle is not None:
            test_evaluations_handle.close()



def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_routing(args.dataset, lambda_sketch_ref=getattr(args, "lambda_sketch_ref", None))
    output = Path(args.output_dir).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    try:
        return _train_impl(args, output)
    except Exception:
        lambda_sketch_ref = getattr(args, "lambda_sketch_ref", None)
        route = _validate_routing(args.dataset, lambda_sketch_ref=lambda_sketch_ref)
        _json(output / "run_result.json", {"status": "FAIL", "method_version": route["method_version"], "arm": route["arm"], "dataset": args.dataset, "traceback": traceback.format_exc()})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--periodic-test", action="store_true", default=False)
    parser.add_argument("--lambda-sketch-ref", type=float, default=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    train(_parser().parse_args())
