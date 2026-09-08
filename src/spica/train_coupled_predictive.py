"""Sequential online trainer for the approved coupled-predictive V1 campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .coupled_predictive_losses import coupled_region_loss
from .models.clip import load_frozen_clip
from .models.coupled_predictive import CoupledPredictiveModel
from .models.sigreg import SIGReg
from .provenance import capture_provenance, capture_rng_state
from .tracking.wandb import WandbExperiment

ROOT = Path(__file__).resolve().parents[2]
CLIP_PATH = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors"
CLIP_SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
CLIP_MODEL = "ViT-B-32-quickgelu"
BATCH_SIZE = 32
TOTAL_STEPS = 3600
PROBE_STEPS = (0, 600, 1200, 1800, 2400, 3000, 3600)
MASK_FRACTIONS = (0.25, 0.5, 0.75)
MASK_SEEDS = (101, 202, 303)
WARMUP_STEPS = 180
_RETRIEVAL_NAMES = ("full_mAP", "P@200", "mAP@200_prefix_positive", "mAP@200_all_relevant", "mAP@200_min_relevant_k")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(b"\0")
    digest.update(str(value.dtype).encode())
    digest.update(b"\0")
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_hash(module: torch.nn.Module, *, prefix: str | None = None) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if prefix is not None and not name.startswith(prefix):
            continue
        if isinstance(value, Tensor):
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(_tensor_hash(value).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _schedule(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    return 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)))


def _verify_clip() -> dict[str, Any]:
    if not CLIP_PATH.is_file():
        raise FileNotFoundError(f"verified local CLIP cache is missing: {CLIP_PATH}")
    actual = _sha256_file(CLIP_PATH)
    if actual != CLIP_SHA256:
        raise ValueError(f"CLIP cache SHA256 mismatch: expected {CLIP_SHA256}, got {actual}")
    return {"path": str(CLIP_PATH), "bytes": CLIP_PATH.stat().st_size, "sha256": actual}


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, Tensor) else value for key, value in batch.items()}


def _model_state(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if not name.startswith("original_clip.")}


def _initialization_hashes(model: CoupledPredictiveModel) -> dict[str, Any]:
    groups = {
        "student": _state_hash(model, prefix="student_visual."),
        "pooled": _state_hash(model, prefix="pooled_head."),
        "photo": _tensor_hash(model.photo_model.photo_prompt),
        "text": _tensor_hash(model.text_bank.context),
        "T0": _tensor_hash(model.T0),
    }
    return {"groups": groups, "model": hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()}


def _checkpoint_payload(model: Any, optimizer: Any, scheduler: Any, sigreg: Any, *, step: int, config: Mapping[str, Any], source_hash: str, clip: Mapping[str, Any], data_identity: Mapping[str, Any], rng: Mapping[str, Any], selections: Mapping[str, Any], initialization_hashes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "campaign": "coupled_predictive_v1",
        "step": step,
        "model_state_dict": _model_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "sigreg_state_dict": None if sigreg is None else sigreg.state_dict(),
        "rng_state": {**rng, "numpy": np.random.get_state()},
        "source_snapshot_hash": source_hash,
        "clip_identity": dict(clip),
        "data_identity": dict(data_identity),
        "resolved_config": dict(config),
        "initialization_hashes": dict(initialization_hashes),
        "model_state_hash": _state_hash(model),
        "selection_metadata": dict(selections),
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return _sha256_file(path)


def _make_optimizer(model: Any) -> AdamW:
    groups = [{key: value for key, value in group.items() if key in {"params", "lr", "weight_decay"}} for group in model.optimizer_parameter_groups()]
    return AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def _loader_cycle(loader: Any):
    while True:
        yield from loader


def _eval_factory(protocol: Mapping[str, Any], transform: Any, model: Any, device: torch.device):
    split = protocol["split"]
    sketches = tuple(split.validation_sketch_entries)
    photos = tuple(split.validation_photo_entries)
    from .data.datasets import RetrievalEvalDataset
    from .evaluation.coupled_predictive import CoupledPredictiveAdapter, evaluate_views
    from .evaluation.masked_view import _masked_loader, _transform_stats
    data = protocol["data"]
    root = Path(str(getattr(data, "root", ".")))
    mean, std = _transform_stats(transform)
    adapter = CoupledPredictiveAdapter(model, query="q" if model.predictor is None else "mu_i")
    clean = DataLoader(RetrievalEvalDataset(sketches, transform), batch_size=256, shuffle=False, num_workers=4)
    gallery = DataLoader(RetrievalEvalDataset(photos, transform), batch_size=256, shuffle=False, num_workers=4)
    def masked(fraction: float, seed: int):
        return _masked_loader(sketches, transform, root=root, fraction=fraction, seed=seed, batch_size=256, num_workers=4, mean=mean, std=std, ink_threshold=0.9)
    return lambda: evaluate_views(adapter, clean, gallery, masked, query_entries=sketches, device=device)


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_data_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    pairing = protocol["pairing"]
    return {
        "split": _safe_json(protocol["split_identity"]),
        "manifest": _safe_json(protocol["manifest_identity"]),
        "pairing": {
            "sha256": pairing["sha256"],
            "records": int(pairing["records"]),
            "unique_photo_pool": int(pairing["unique_photo_pool"]),
        },
    }


def _diagnostic_payload(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SIGReg diagnostic receipt is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS" or payload.get("verified") is not True:
        raise ValueError("SIGReg diagnostic must have status PASS and verified true")
    for key in ("formula_identity", "architecture_identity", "gradient_batch_source"):
        value = payload.get(key)
        if value in (None, False, "", "FAIL"):
            raise ValueError(f"SIGReg diagnostic is missing verified {key}")
    return payload


def _diagnostic_lambda(path: Path, *, class_ids: list[int], initialization: Mapping[str, Any], source_hash: str, config: Mapping[str, Any]) -> float:
    payload = _diagnostic_payload(path)
    value = payload.get("lambda_sig", payload.get("selection", {}).get("lambda_sig") if isinstance(payload.get("selection"), Mapping) else None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("SIGReg diagnostic receipt has no verified positive lambda_sig")
    reported_ids = payload.get("class_ids", payload.get("train_class_ids"))
    if reported_ids is not None and [int(x) for x in reported_ids] != class_ids:
        raise ValueError("SIGReg diagnostic class IDs differ from this run")
    reported_init = payload.get("initialization_hashes", payload.get("initial_hashes"))
    if reported_init is not None and reported_init != initialization:
        raise ValueError("SIGReg diagnostic initial hashes differ from this run")
    # Diagnostic docs/reporting may precede the final training source snapshot.
    for name, digest in payload["component_sha256"].items():
        if _sha256_file(ROOT / name) != digest:
            raise ValueError(f"SIGReg diagnostic component changed: {name}")
    reported_config = payload.get("resolved_config")
    if isinstance(reported_config, Mapping):
        for key in ("architecture", "batch_size", "mask_policy", "eval_mask_fractions", "eval_mask_seeds"):
            if key in reported_config and reported_config[key] != config.get(key):
                raise ValueError(f"SIGReg diagnostic config differs at {key}")
    return float(value)


def _execution_manifest_hash(campaign_root: Path) -> str | None:
    path = campaign_root / "execution_manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("execution_manifest.json must be an object")
    for key in ("source_snapshot_hash", "source_hash"):
        if isinstance(payload.get(key), str):
            return payload[key]
    for container_key in ("source_snapshot", "executed_source", "source"):
        container = payload.get(container_key)
        if isinstance(container, Mapping) and isinstance(container.get("sha256"), str):
            return container["sha256"]
    return None


def _verify_source(source: Mapping[str, Any], campaign_root: Path) -> str:
    snapshot = source.get("source_snapshot")
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("sha256"), str):
        raise ValueError("provenance did not produce a source snapshot")
    actual = str(snapshot["sha256"])
    expected = _execution_manifest_hash(campaign_root)
    if expected is not None and expected != actual:
        raise ValueError(f"execution manifest source hash mismatch: expected {expected}, got {actual}")
    return actual


def _wandb_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(metrics[name]) for name in _RETRIEVAL_NAMES if name in metrics}


def _wandb_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep W&B config JSON-safe and free of local path representations."""
    allowed = {
        "arm", "campaign_id", "device", "wandb_mode", "max_steps", "smoke",
        "batch_size", "total_steps", "probe_steps", "architecture", "mask_policy",
        "eval_mask_fractions", "eval_mask_seeds", "scheduler", "optimizer",
        "optimizer_betas", "optimizer_eps", "clip_identity", "train_class_ids",
        "class_names", "classmap_sha256", "optimizer_groups",
        "model_trainable_parameters", "model_total_parameters", "initialization_hashes",
        "data_identity", "diagnostic_path_sha256", "lambda_sig", "diagnostic_lambda_source",
    }
    result = {key: config[key] for key in sorted(allowed) if key in config}
    if isinstance(result.get("clip_identity"), Mapping):
        result["clip_identity"] = {
            key: result["clip_identity"][key]
            for key in ("bytes", "sha256")
            if key in result["clip_identity"]
        }
    return _safe_json(result)


def _probe_summary(path: str, sha256: str, metrics: Mapping[str, Any], step: int) -> dict[str, Any]:
    return {
        "step": step,
        "path": path,
        "sha256": sha256,
        "clean": _wandb_scalar_metrics(metrics["clean"]),
        "masked_macro": _wandb_scalar_metrics(metrics["masked_macro"]),
        "masked_by_fraction": {str(key): _wandb_scalar_metrics(value) for key, value in metrics["masked_by_fraction"].items()},
        "conditions": [
            {"fraction": row["fraction"], "seed": row["seed"], **_wandb_scalar_metrics(row)}
            for row in metrics["conditions"]
        ],
        "condition_count": metrics["condition_count"],
    }


def _gradient_diagnostics(model: Any, optimizer: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        values = []
        for parameter in group["params"]:
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient in optimizer group {index}")
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"nonfinite gradient in optimizer group {index}")
            values.append(parameter.grad.detach().float().pow(2).sum())
        result[f"group_{index}"] = float(torch.stack(values).sum().sqrt().cpu())
    return result


def _train_impl(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    if args.max_steps < 1 or args.max_steps > TOTAL_STEPS:
        raise ValueError("max_steps must be between 1 and 3600")
    if args.smoke and args.max_steps != 2:
        raise ValueError("--smoke requires --max-steps 2")
    if not args.smoke and args.max_steps != TOTAL_STEPS:
        raise ValueError("full campaign must run exactly 3600 updates")
    if args.arm not in {"R0", "R1", "R1_SIG"}:
        raise ValueError("arm must be R0, R1, or R1_SIG")
    if args.arm == "R1_SIG" and not args.diagnostic:
        raise ValueError("R1_SIG requires --diagnostic")

    device = _device(args.device)
    from .data.coupled_training import load_protocol_data, make_train_loader, prepare_batch, verify_clip_cache
    protocol = load_protocol_data()
    split = protocol["split"]
    classmap = protocol.get("names") or protocol.get("classmap")
    train_ids = tuple(int(value) for value in split.train_class_ids)
    if not isinstance(classmap, Mapping) or tuple(sorted(train_ids)) != tuple(train_ids) or len(train_ids) != 84:
        raise ValueError("protocol must expose the approved sorted 84 train class IDs")
    normalized_classmap = {int(key): str(value) for key, value in classmap.items()}
    train_classmap = {class_id: normalized_classmap[class_id] for class_id in train_ids}
    clip_identity = verify_clip_cache()
    _seed(42)
    bundle = load_frozen_clip(model_name=CLIP_MODEL, pretrained=str(CLIP_PATH), device=device)
    transform, tokenizer, encoder = bundle.transform, bundle.tokenizer, bundle.encoder
    architecture = "pooled" if args.arm == "R0" else "predictive"
    model = CoupledPredictiveModel(encoder, tokenizer, train_classmap, architecture=architecture).to(device)
    if args.arm == "R0":
        model.predictor = None
    initialization = _initialization_hashes(model)
    frozen_original_before = _state_hash(model, prefix="original_clip.")

    config: dict[str, Any] = _safe_json(vars(args))
    config.update({
        "batch_size": BATCH_SIZE,
        "total_steps": TOTAL_STEPS,
        "probe_steps": list(PROBE_STEPS),
        "architecture": architecture,
        "mask_policy": "region_pair_v1; zero_based_step=step-1",
        "eval_mask_fractions": list(MASK_FRACTIONS),
        "eval_mask_seeds": list(MASK_SEEDS),
        "scheduler": "warmup180_cosine_floor0",
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "clip_path": clip_identity["path"],
        "clip_identity": clip_identity,
        "train_class_ids": list(train_ids),
        "class_names": train_classmap,
        "classmap_sha256": hashlib.sha256(json.dumps(train_classmap, sort_keys=True).encode()).hexdigest(),
        "optimizer_groups": [
            {"name": group["name"], "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])}
            for group in model.optimizer_parameter_groups()
        ],
        "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_total_parameters": sum(p.numel() for p in model.parameters()),
        "initialization_hashes": initialization,
        "frozen_original_state_hash": frozen_original_before,
    })
    source = capture_provenance(ROOT, resolved_config=config)
    source_hash = _verify_source(source, Path(args.campaign_root).expanduser())
    data_identity = _compact_data_identity(protocol)
    config["data_identity"] = data_identity
    config["diagnostic_path_sha256"] = None if not args.diagnostic else _sha256_file(Path(args.diagnostic))
    lambda_sig = 0.0
    if args.arm == "R1_SIG":
        lambda_sig = _diagnostic_lambda(Path(args.diagnostic), class_ids=list(train_ids), initialization=initialization, source_hash=source_hash, config=config)
    config["lambda_sig"] = lambda_sig
    config["diagnostic_lambda_source"] = "verified_receipt" if args.arm == "R1_SIG" else "control_zero"
    source["resolved_config"] = config
    snapshot = source["source_snapshot"]
    snapshot_root = output / "source_snapshot" / "files"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for item in snapshot["manifest"]:
        relative = Path(str(item["path"]))
        source_file = ROOT / relative
        destination = snapshot_root / relative
        if not source_file.is_file():
            raise FileNotFoundError(f"source snapshot file disappeared: {source_file}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination)
        if _sha256_file(destination) != str(item["sha256"]):
            raise RuntimeError(f"source snapshot copy hash mismatch: {relative}")
    _json(output / "source_snapshot" / "index.json", snapshot)
    _json(output / "provenance.json", source)
    _json(output / "resolved_config.json", config)
    _json(output / "initialization.json", initialization)

    optimizer = _make_optimizer(model)
    scheduler = LambdaLR(optimizer, lr_lambda=_schedule)
    sigreg = SIGReg(seed=42).to(device) if args.arm == "R1_SIG" else None
    loader = make_train_loader(protocol, transform)
    if loader.generator is None:
        raise RuntimeError("training loader has no reproducible generator")
    batches = _loader_cycle(loader)
    eval_probe = None if args.smoke else _eval_factory(protocol, transform, model, device)
    if eval_probe is None and not args.smoke:
        raise ValueError("full campaign requires validated pseudo-validation split entries")
    data_root = Path(str(getattr(protocol["data"], "root", args.data_root)))
    selections: dict[str, Any] = {}
    checkpoints: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    trace_count = 0
    training_history = output.joinpath("training_history.jsonl").open("w", encoding="utf-8")
    lr_history = output.joinpath("lr_history_every_step.jsonl").open("w", encoding="utf-8")
    trace_handle = output.joinpath("observation_trace.jsonl").open("w", encoding="utf-8")
    mask_handle = output.joinpath("mask_metadata.jsonl").open("w", encoding="utf-8")
    wandb_run = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_baseline = {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device)}
    else:
        memory_baseline = {}

    try:
        wandb_config = _wandb_config(config)
        wandb_config["source_snapshot_hash"] = source_hash
        if args.wandb_mode != "disabled":
            wandb_run = WandbExperiment(project="spica", entity="a-cctest05187-erd", name=args.arm, group=args.campaign_id, job_type="train", config=wandb_config, mode=args.wandb_mode, directory=output)
            _json(output / "wandb_runtime.json", {"run_id": wandb_run.run_id, "run_url": wandb_run.run_url, "arm": args.arm})
            wandb_run.define_metric("step_train")
            for pattern in ("train/*", "clean/*", "masked/*"):
                wandb_run.define_metric(pattern, step_metric="step_train")

        def save(step: int) -> None:
            payload = _checkpoint_payload(model, optimizer, scheduler, sigreg, step=step, config=config, source_hash=source_hash, clip=clip_identity, data_identity=data_identity, rng=capture_rng_state(generator=loader.generator), selections=selections, initialization_hashes=initialization)
            sha = _save_checkpoint(output / f"checkpoint_step{step}.pt", payload)
            checkpoints.append({"step": step, "path": f"checkpoint_step{step}.pt", "sha256": sha, "model_state_hash": payload["model_state_hash"], "initialization_hashes": initialization})

        save(0)
        initial_lrs = {f"group_{i}": float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}
        _append_jsonl(lr_history, {"step": 0, "lr": initial_lrs})
        _append_jsonl(training_history, {"step": 0, "loss": None, "lr": initial_lrs, "gradient_norm": None})
        if not args.smoke:
            metrics = eval_probe()
            raw_path = output / "probe_step0.json"
            _json(raw_path, metrics)
            raw_sha = _sha256_file(raw_path)
            probe_records.append(_probe_summary(raw_path.name, raw_sha, metrics, 0))
            diagnostics = {f"train/lr_group_{i}": float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}
            if wandb_run is not None:
                wandb_run.log_retrieval_probe(0, _wandb_scalar_metrics(metrics["clean"]), _wandb_scalar_metrics(metrics["masked_macro"]), {float(k): _wandb_scalar_metrics(v) for k, v in metrics["masked_by_fraction"].items()}, conditions=[{**_wandb_scalar_metrics(row), "fraction": row["fraction"], "seed": row["seed"]} for row in metrics["conditions"]], diagnostics=diagnostics)

        for step in range(1, args.max_steps + 1):
            batch = prepare_batch(next(batches), data_root=data_root, step=step - 1, classids=model.classids.detach().cpu().tolist())
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = coupled_region_loss(model, batch["clean"], batch["corrupted"], batch["photos"], batch["positive_indices"], batch["negative_indices"], batch["labels"], batch["photo_labels"], batch["photo_ids"], lambda_sig=lambda_sig, sigreg=sigreg)
            total = losses["total"]
            if not torch.isfinite(total).item():
                raise FloatingPointError(f"nonfinite loss at step {step}")
            total.backward()
            gradient_norms = _gradient_diagnostics(model, optimizer)
            actual_lrs = {f"group_{i}": float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}
            _append_jsonl(lr_history, {"step": step, "lr": actual_lrs})
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            for item in batch.get("trace", ()):
                row = dict(item)
                row["step"] = step
                row["actual_lr"] = actual_lrs["group_0"]
                _append_jsonl(trace_handle, row)
                trace_count += 1
            mask_rows = batch.get("mask_metadata", {}).get("rows", ())
            for item in mask_rows:
                _append_jsonl(mask_handle, item)
            if step % 10 == 0:
                history_row = {"step": step, "loss": {name: float(value.detach().cpu()) for name, value in losses.items()}, "lr": actual_lrs, "gradient_norm": gradient_norms}
                _append_jsonl(training_history, history_row)
            if step in PROBE_STEPS[1:] and (step == args.max_steps or args.max_steps == TOTAL_STEPS):
                metrics = eval_probe()
                raw_path = output / f"probe_step{step}.json"
                _json(raw_path, metrics)
                raw_sha = _sha256_file(raw_path)
                probe_records.append(_probe_summary(raw_path.name, raw_sha, metrics, step))
                clean_score = float(metrics["clean"]["mAP@200_prefix_positive"])
                selection_metrics = {
                    "clean": _wandb_scalar_metrics(metrics["clean"]),
                    "masked_macro": _wandb_scalar_metrics(metrics["masked_macro"]),
                }
                selection_info = {"step": step, "metrics": selection_metrics, "resolved_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
                selections.setdefault("best_clean", {**selection_info, "score": clean_score})
                if clean_score > float(selections["best_clean"]["score"]):
                    selections["best_clean"] = {**selection_info, "score": clean_score}
                masked_score = float(metrics["masked_macro"]["mAP@200_prefix_positive"])
                if masked_score > float(selections.get("best_masked", {"score": -float("inf")})["score"]):
                    selections["best_masked"] = {**selection_info, "score": masked_score}
                diagnostics = {f"train/gradnorm_{key}": value for key, value in gradient_norms.items()}
                diagnostics.update({f"train/lr_{key}": value for key, value in actual_lrs.items()})
                diagnostics.update({f"train/{name}": float(value.detach().cpu()) for name, value in losses.items()})
                if wandb_run is not None:
                    wandb_run.log_retrieval_probe(step, _wandb_scalar_metrics(metrics["clean"]), _wandb_scalar_metrics(metrics["masked_macro"]), {float(k): _wandb_scalar_metrics(v) for k, v in metrics["masked_by_fraction"].items()}, conditions=[{**_wandb_scalar_metrics(row), "fraction": row["fraction"], "seed": row["seed"]} for row in metrics["conditions"]], diagnostics=diagnostics)
                save(step)
            elif step % 10 == 0:
                train_metrics = {"step_train": step, **{f"train/{name}": float(value.detach().cpu()) for name, value in losses.items()}, **{f"train/lr_{key}": value for key, value in actual_lrs.items()}, **{f"train/gradnorm_{key}": value for key, value in gradient_norms.items()}}
                if wandb_run is not None:
                    wandb_run.log_metrics(train_metrics, step=step)
            if step % 10 == 0:
                trace_handle.flush()
                mask_handle.flush()
                training_history.flush()

        if args.smoke or args.max_steps not in PROBE_STEPS:
            save(args.max_steps)
        expected_trace_count = args.max_steps * BATCH_SIZE
        if trace_count != expected_trace_count:
            raise RuntimeError(f"observation trace count mismatch: expected {expected_trace_count}, got {trace_count}")
        latest = checkpoints[-1]
        latest_probe = next((probe for probe in reversed(probe_records) if probe["step"] == latest["step"]), None)
        selections["latest"] = {"step": latest["step"], "path": latest["path"], "sha256": latest["sha256"], "metrics": None if latest_probe is None else {"clean": latest_probe["clean"], "masked_macro": latest_probe["masked_macro"]}, "resolved_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
        for name in ("best_clean", "best_masked"):
            if name in selections:
                selected = next(row for row in checkpoints if row["step"] == selections[name]["step"])
                selections[name].update({"path": selected["path"], "sha256": selected["sha256"]})
                _json(output / f"{name}.json", selections[name])

        for name, selection in selections.items():
            if name not in {"latest", "best_clean", "best_masked"}:
                continue
            _json(output / f"{name}.json", selection)
        if args.wandb_mode != "disabled":
            aliases_by_checkpoint: dict[str, list[str]] = {}
            for name in ("latest", "best_clean", "best_masked"):
                if name in selections:
                    aliases_by_checkpoint.setdefault(selections[name]["path"], []).append(name)
            artifact_root = output / ".wandb_artifact"
            for checkpoint_path, aliases in sorted(aliases_by_checkpoint.items(), key=lambda row: "latest" in row[1]):
                artifact_dir = artifact_root / Path(checkpoint_path).stem
                artifact_dir.mkdir(parents=True)
                _json(artifact_dir / "resolved_config.json", config)
                import os
                os.link(output / checkpoint_path, artifact_dir / Path(checkpoint_path).name)
                wandb_run.log_artifact(artifact_dir, name=f"coupled-{wandb_run.run_id}", artifact_type="model", metadata={"aliases": aliases, "checkpoint_sha256": next(item["sha256"] for item in checkpoints if item["path"] == checkpoint_path), "source_snapshot_hash": source_hash, "large_files": [Path(checkpoint_path).name]}, aliases=aliases)

        final_source = capture_provenance(ROOT, resolved_config=config)
        if _verify_source(final_source, Path(args.campaign_root).expanduser()) != source_hash:
            raise RuntimeError("source snapshot changed during training")
        frozen_original_after = _state_hash(model, prefix="original_clip.")
        if frozen_original_after != frozen_original_before:
            raise RuntimeError("frozen original CLIP state changed")
        result = {"status": "COMPLETE", "campaign": "coupled_predictive_v1", "arm": args.arm, "step": args.max_steps, "completed_steps": args.max_steps, "source_snapshot_hash": source_hash, "clip_identity": clip_identity, "data_identity": data_identity, "checkpoints": checkpoints, "probes": probe_records, "selections": selections, "trace": "observation_trace.jsonl", "trace_count": trace_count, "expected_trace_count": expected_trace_count, "mask_metadata": "mask_metadata.jsonl", "training_history": "training_history.jsonl", "initialization_hashes": initialization, "frozen_original_state_hash_before": frozen_original_before, "frozen_original_state_hash_after": frozen_original_after, "memory": {"baseline": memory_baseline, "peak_allocated": torch.cuda.max_memory_allocated(device), "peak_reserved": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else {}, "wandb_run_id": None if wandb_run is None else wandb_run.run_id, "wandb_url": None if wandb_run is None else wandb_run.run_url}
        _json(output / "run_result.json", result)
        if wandb_run is not None:
            wandb_run.finish()
        return result
    except Exception:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise
    finally:
        trace_handle.close()
        mask_handle.close()
        training_history.close()
        lr_history.close()


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    try:
        return _train_impl(args, output)
    except Exception:
        _json(output / "run_result.json", {"status": "FAIL", "phase": "initialization_or_training", "traceback": traceback.format_exc()})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("R0", "R1", "R1_SIG"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--diagnostic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--max-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--campaign-id", default="coupled_predictive_v1")
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    train(_parser().parse_args())
