"""Versioned coupled trainer: historical V1 arms and explicit experimental F2."""
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

from .coupled_predictive_losses import _validate_photo_ce_coefficient, coupled_region_loss
from .data.coupled_training import _arm_protocol, positive_pool_identity
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
_PHOTO_CE_METADATA = (
    "lambda_photo_ce", "lambda_photo_ce_source", "photo_ce_temperature",
    "photo_ce_reduction", "photo_ce_target", "training_main_query",
    "evaluation_query", "evaluation_adapter",
)


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
    if model.architecture == "predictive_fusion_v2":
        groups["predictor"] = _state_hash(model, prefix="predictor.")
    return {"groups": groups, "model": hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()}


def _checkpoint_payload(model: Any, optimizer: Any, scheduler: Any, sigreg: Any, *, step: int, config: Mapping[str, Any], source_hash: str, clip: Mapping[str, Any], data_identity: Mapping[str, Any], rng: Mapping[str, Any], selections: Mapping[str, Any], initialization_hashes: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "campaign": str(config.get("method_version", "coupled_predictive_v1")),
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
    if config.get("method_version") in {"coupled_predictive_fusion_qmp_v1", "coupled_predictive_fusion_mp_photo_ce_v1"}:
        payload.update({
            "method_version": config["method_version"],
            "main_query": config["main_query"],
            "loss_coefficient_identity": config["loss_coefficient_identity"],
            "sampling_identity": config["sampling_identity"],
            "main_photo_objective": config["main_photo_objective"],
            "main_photo_temperature": config["main_photo_temperature"],
            "objective_identity": config["objective_identity"],
            "primary_comparison": config["primary_comparison"],
        })
    elif config.get("method_version") == "coupled_predictive_fusion_tqmp_v1":
        payload.update({
            "method_version": config["method_version"],
            "main_query": config["main_query"],
            "main_q": config["main_q"],
            "training_main_query": config["training_main_query"],
            "evaluation_query": config["evaluation_query"],
            "evaluation_adapter": config["evaluation_adapter"],
            "loss_coefficient_identity": config["loss_coefficient_identity"],
            "sampling_identity": config["sampling_identity"],
            "main_photo_objective": config["main_photo_objective"],
            "main_photo_temperature": config["main_photo_temperature"],
            "objective_identity": config["objective_identity"],
            "tau": config["tau"],
            "coefficient_identity": config["coefficient_identity"],
            "diagnostic_selection_metadata": config["diagnostic_selection_metadata"],
            "lambda_selection_metadata": config["selection_metadata"],
            "primary_comparison": config["primary_comparison"],
            "diagnostic_identity": config["diagnostic_identity"],
            "diagnostic_sha256": config["diagnostic_sha256"],
            "diagnostic_path_sha256": config["diagnostic_path_sha256"],
            "raw_diagnostic_path_sha256": config["raw_diagnostic_path_sha256"],
            "diagnostic_verified_path_sha256": config["diagnostic_verified_path_sha256"],
            "diagnostic_input_sha256": config["diagnostic_input_sha256"],
            "diagnostic_data_identity": config["diagnostic_data_identity"],
            "diagnostic_initialization_hashes": config["diagnostic_initialization_hashes"],
            "diagnostic_source_snapshot_hash": config["diagnostic_source_snapshot_hash"],
            "lambda_mp_t": config["lambda_mp_t"],
            "lambda_mp_q": config["lambda_mp_q"],
        })
    elif config.get("method_version") in {"coupled_predictive_fusion_v2", "coupled_predictive_fusion_mp_v1"}:
        # Preserve the historical checkpoint schema for F2 and F2_MP exactly.
        payload.update({
            "method_version": config["method_version"],
            "main_query": config["main_query"],
            "loss_coefficient_identity": config["loss_coefficient_identity"],
            "sampling_identity": config["sampling_identity"],
        })
    if config.get("method_version") == "coupled_predictive_fusion_mp_photo_ce_v1":
        payload.update({key: config[key] for key in _PHOTO_CE_METADATA})
    return payload


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


def _eval_factory(protocol: Mapping[str, Any], transform: Any, model: Any, device: torch.device, query: str | None = None):
    split = protocol["split"]
    sketches = tuple(split.validation_sketch_entries)
    photos = tuple(split.validation_photo_entries)
    from .data.datasets import RetrievalEvalDataset
    from .evaluation.coupled_predictive import CoupledPredictiveAdapter, QOnlyAdapter, evaluate_views
    from .evaluation.masked_view import _masked_loader, _transform_stats
    data = protocol["data"]
    root = Path(str(getattr(data, "root", ".")))
    mean, std = _transform_stats(transform)
    adapter = QOnlyAdapter(model) if query == "q" else CoupledPredictiveAdapter(model, query="q" if model.predictor is None else "mu_i")
    clean = DataLoader(RetrievalEvalDataset(sketches, transform), batch_size=256, shuffle=False, num_workers=4)
    gallery = DataLoader(RetrievalEvalDataset(photos, transform), batch_size=256, shuffle=False, num_workers=4)
    def masked(fraction: float, seed: int):
        return _masked_loader(sketches, transform, root=root, fraction=fraction, seed=seed, batch_size=256, num_workers=4, mean=mean, std=std, ink_threshold=0.9)
    return lambda: evaluate_views(adapter, clean, gallery, masked, query_entries=sketches, device=device)


def _evaluation_query(arm: str) -> str | None:
    # Explicit experimental q readouts; historical F2/F2_MP remain mu_i.
    return "q" if arm in {"F2_QMP", "F2_TQMP", "F2_MP_PCE"} else None


def _make_eval_probe(protocol: Mapping[str, Any], transform: Any, model: Any, device: torch.device, arm: str):
    return _eval_factory(protocol, transform, model, device, query=_evaluation_query(arm))


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


def _compact_data_identity(
    protocol: Mapping[str, Any], positive_pool: str = "canonical"
) -> dict[str, Any]:
    pool = positive_pool_identity(protocol, positive_pool)
    pairing = protocol["pairing"]
    result = {
        "split": _safe_json(protocol["split_identity"]),
        "manifest": _safe_json(protocol["manifest_identity"]),
        "pairing": {
            "sha256": pairing["sha256"],
            "records": int(pairing["records"]),
            "unique_photo_pool": int(pairing["unique_photo_pool"]),
        },
    }
    if positive_pool == "full":
        result["positive_pool"] = pool
        result["pairing"]["role"] = "audit_only;not_positive_sampling_source"
        result["sampling"] = {
            "positive_pool": "full",
            "positive_pool_count": pool["count"],
            "negative_pool_count": pool["count"],
            "positive_and_negative_pool_sha256": pool["sha256"],
            "positive_rule": "uniform_photo_within_query_class",
            "negative_rule": "uniform_other_class_then_uniform_photo",
            "same_photo_preprocessing": True,
            "canonical_pairing_records": int(pairing["records"]),
            "canonical_pairing_unique_photo_pool": int(pairing["unique_photo_pool"]),
            "pairing_manifest_role": "audit_only;not_positive_sampling_source",
        }
    return result


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
    if config.get("architecture") == "predictive_fusion_v2":
        expected = {"architecture": "predictive_fusion_v2", "positive_pool": "full",
                    "task_identity": "mean_views(rank_i+ce_i)", "batch_size": 32}
        reported = payload.get("resolved_config", {})
        if any(reported.get(key) != val for key, val in expected.items()):
            raise ValueError("F2_SIG requires a new F2/full-pool/main-mu_i diagnostic, not V1")
        if payload.get("batches") != 4 or payload.get("rho") != 0.1:
            raise ValueError("F2_SIG requires the locked four-batch rho0.1 protocol")
        required_components = {"src/spica/models/coupled_predictive.py", "src/spica/models/sigreg.py",
                               "src/spica/coupled_predictive_losses.py", "src/spica/data/coupled_training.py"}
        if not required_components <= payload.get("component_sha256", {}).keys():
            raise ValueError("F2 diagnostic is missing required source components")
        if payload.get("parameter_scope") != "model.student_visual.transformer.resblocks[-1]":
            raise ValueError("F2 diagnostic gradient scope mismatch")
        measured = Path(str(payload.get("measurement_path", "")))
        if not measured.is_file() or _sha256_file(measured) != payload.get("measurement_sha256"):
            raise ValueError("F2 diagnostic immutable measurement receipt mismatch")
        if payload.get("initialization_hashes") != initialization or payload.get("train_class_ids") != class_ids:
            raise ValueError("F2 diagnostic class/initialization identity mismatch")
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


F2_QMP_READOUT = ROOT / "outputs/fusion_mp_q_evaluation_20260909T145000Z/summary.json"
F2_QMP_READOUT_SHA256 = "f210017b24f93fb659315daa820c1265c698aeb9a52112803543db63cd452afa"
F2_TQMP_DIAGNOSTIC = ROOT / "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/diagnostic_verified.json"
F2_TQMP_DIAGNOSTIC_SHA256 = "d3021b6ecd7a1b0d89b1da5c62761442728f6f095a1c9a4ace572ca9525cd68b"
F2_TQMP_RAW_DIAGNOSTIC = ROOT / "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/diagnostic_result.json"
F2_TQMP_SOURCE_SNAPSHOT_SHA256 = "42485ce4bfc86dc296cba91d997a3d9a810c1859a82d87eb9566bc7d1705b53f"
F2_TQMP_LAMBDA_T = 0.17877235601108843
F2_TQMP_LAMBDA_Q = 0.023774345199536452
F2_TQMP_INPUT_SHA256 = {
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/diagnostic_result.json": "153b47573c0893b794562526dd723ed82f8abbb3c69b94ee5b3e5fe1c6ff633b",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/independent_cpu/receipt.json": "195419b095b6dcf7dcfa0347e50cf693f71feee4db6a552b5392bddcfadf7578",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/independent_cpu/summary.json": "dd49ebcb0d9b0e02bf495046497cdb55a97632b7a127a076ecd63728da5a2fb1",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/independent_cpu/verify_cpu.py": "f9e80a103a72f1d4f7f1c4259778dabe45aac5dd76643b2f221466167943c42f",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/verify_parent.py": "7c7112248481daa9791811b6aa05a4000d0f282fe93ff95b7dc98755984330b2",
    "scripts/diagnose_fusion_mp_tq.py": "d7aeb62b8ab7715731789a0e440491b30263f95bd937377bd4b91191f8cbe9e8",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/raw_gradients_batch00.pt": "815c7f145ccc8d0e303a9b85d4d501c9ba48721e857df214b77f0b7a7a174d30",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/raw_gradients_batch01.pt": "fcd20fdcdb6820869176ffd10b1a45edac8a6d9500ec8a277629d5cddd496432",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/raw_gradients_batch02.pt": "ff6f570946bfb59da09095622cb571276ded1edd615591ba84189f2249e908ba",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/raw_gradients_batch03.pt": "488e8e342a99d7db468e57cba7fafe063230af84ff22865384cc86a2e507f415",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/data_records.json": "6c123f31e7530517da675361d28cede1ab331c57b416bf68d18eed703a474d1d",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/gradient_layout.json": "70a07db6959b61b57cede117f18128851e5bbbc950b1d2c2f4ffa6a4963de028",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/lambda_selection.json": "012eca0655a19a813c9534b87fe0e88eb17ecb750d3a2e330eac797c22671580",
    "outputs/fusion_mp_tq_diagnostic_20260910T001200Z/measured_rows.json": "103d13bc15e221bb3a2e0364c5b01c2f5259098eada812a76c79a63a6495a681",
}


def _validate_f2_tqmp_diagnostic(path: str | Path) -> dict[str, Any]:
    """Validate the immutable, initialization-only TQMP calibration receipt."""
    if len(F2_TQMP_INPUT_SHA256) != 14:
        raise RuntimeError("F2_TQMP diagnostic must bind exactly 14 input files")
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(ROOT)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError("F2_TQMP diagnostic must be an existing path inside the repository") from error
    if _sha256_file(resolved) != F2_TQMP_DIAGNOSTIC_SHA256:
        raise ValueError("F2_TQMP diagnostic receipt SHA256 mismatch")
    verified = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(verified, Mapping) or verified.get("status") != "PASS":
        raise ValueError("F2_TQMP diagnostic verification receipt must have status PASS")
    if verified.get("scope") != "RAW_GRADIENT_CALIBRATION_ONLY; no training or mAP claim":
        raise ValueError("F2_TQMP diagnostic scope mismatch")
    if verified.get("independent_cpu_check_count") != 2041 or verified.get("no_new_gpu_or_optimizer") is not True:
        raise ValueError("F2_TQMP diagnostic independent CPU verification is incomplete")
    if verified.get("input_sha256") != F2_TQMP_INPUT_SHA256:
        raise ValueError("F2_TQMP diagnostic input hash manifest mismatch")
    for name, expected in F2_TQMP_INPUT_SHA256.items():
        input_path = Path(name)
        if input_path.is_absolute() or ".." in input_path.parts:
            raise ValueError("F2_TQMP diagnostic input path is not repository-relative")
        input_resolved = (ROOT / input_path).resolve(strict=True)
        try:
            input_resolved.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("F2_TQMP diagnostic input escaped the repository") from error
        if _sha256_file(input_resolved) != expected:
            raise ValueError(f"F2_TQMP diagnostic input SHA256 mismatch: {name}")

    raw = json.loads(F2_TQMP_RAW_DIAGNOSTIC.read_text(encoding="utf-8"))
    expected_config = {
        "baseline": "F2_MP full total", "batch_size": 32, "batches": 4,
        "diagnostic": "fusion_mp_tq_init_v1", "optimizer_updates": 0,
        "positive_pool": "full", "rho_q": 0.1, "rho_t": 0.1, "seed": 42,
        "sequence": "add lambdaT*MP(muT), then lambdaQ*MP(q)", "tau": 0.07,
        "wandb": "disabled",
    }
    selection = {
        "binding_scope": "pooled_head",
        "lambda_t": F2_TQMP_LAMBDA_T,
        "lambda_q": F2_TQMP_LAMBDA_Q,
        "lambda_q_candidates": {"pooled_head": F2_TQMP_LAMBDA_Q, "student_last_block": 0.2146793430374309},
        "rho_t": 0.1, "rho_q": 0.1,
    }
    if not isinstance(raw, Mapping) or raw.get("status") != "MEASURED_PENDING_REVIEW" or raw.get("verified") is not False:
        raise ValueError("F2_TQMP raw diagnostic status is not the approved pending-review receipt")
    raw_selection = raw.get("selection")
    if raw.get("config") != expected_config or not isinstance(raw_selection, Mapping):
        raise ValueError("F2_TQMP raw diagnostic protocol mismatch")
    if any(raw_selection.get(key) != value for key, value in selection.items()):
        raise ValueError("F2_TQMP raw diagnostic coefficient selection mismatch")
    if raw_selection.get("formula") != "lambdaT=.1 median(norm(base_last)/norm(T_last)); B1=base+lambdaT*T; lambdaQ=.1 min_scopes median(norm(B1_scope)/norm(Q_scope)), scopes=last,pool" or raw_selection.get("scope") != "initialization-only, median-based target; NOT a per-batch cap, mAP optimum or effective AdamW-update ratio":
        raise ValueError("F2_TQMP raw diagnostic selection formula mismatch")
    for key in ("optimizer_updates", "model_hash_before", "model_hash_after", "teacher_hash_before", "teacher_hash_after"):
        if key == "optimizer_updates" and raw.get(key) != 0:
            raise ValueError("F2_TQMP diagnostic contains optimizer updates")
        if key != "optimizer_updates" and not isinstance(raw.get(key), str):
            raise ValueError("F2_TQMP diagnostic is missing state binding")
    if raw.get("model_hash_before") != raw.get("model_hash_after") or raw.get("teacher_hash_before") != raw.get("teacher_hash_after"):
        raise ValueError("F2_TQMP diagnostic model state changed")
    if raw.get("entire_model_unchanged") is not True or raw.get("parameter_grad_all_none") is not True:
        raise ValueError("F2_TQMP diagnostic state guards failed")
    if raw.get("source_snapshot_hash") != F2_TQMP_SOURCE_SNAPSHOT_SHA256:
        raise ValueError("F2_TQMP diagnostic source snapshot mismatch")
    if verified.get("selection") != raw_selection or verified.get("source_snapshot_hash") != raw.get("source_snapshot_hash"):
        raise ValueError("F2_TQMP diagnostic verification binding mismatch")
    if not isinstance(raw.get("data_identity"), Mapping) or not isinstance(raw.get("initialization_hashes"), Mapping):
        raise ValueError("F2_TQMP diagnostic is missing data or initialization identity")
    return {
        "diagnostic_path": str(relative),
        "diagnostic_sha256": F2_TQMP_DIAGNOSTIC_SHA256,
        "diagnostic_path_sha256": F2_TQMP_DIAGNOSTIC_SHA256,
        "raw_diagnostic_path": str(F2_TQMP_RAW_DIAGNOSTIC.relative_to(ROOT)),
        "raw_diagnostic_path_sha256": F2_TQMP_INPUT_SHA256[str(F2_TQMP_RAW_DIAGNOSTIC.relative_to(ROOT))],
        "input_sha256": dict(sorted(F2_TQMP_INPUT_SHA256.items())),
        "lambda_mp_t": F2_TQMP_LAMBDA_T,
        "lambda_mp_q": F2_TQMP_LAMBDA_Q,
        "lambda_q_candidates": dict(selection["lambda_q_candidates"]),
        "binding_scope": "pooled_head",
        "selection_metadata": {
            "formula": raw_selection["formula"],
            "binding_scope": raw_selection["binding_scope"],
            "candidates": dict(raw_selection["lambda_q_candidates"]),
            "rho": {"t": raw_selection["rho_t"], "q": raw_selection["rho_q"]},
            "scope": raw_selection["scope"],
            "lambda_mp_t": raw_selection["lambda_t"],
            "lambda_mp_q": raw_selection["lambda_q"],
        },
        "data_identity": raw["data_identity"],
        "initialization_hashes": raw["initialization_hashes"],
        "source_snapshot_hash": raw["source_snapshot_hash"],
    }


def _validate_f2_qmp_readout() -> dict[str, Any]:
    """Validate the immutable fixed-step q readout used as the QMP baseline."""
    if not F2_QMP_READOUT.is_file():
        raise FileNotFoundError(f"F2_QMP baseline readout is missing: {F2_QMP_READOUT}")
    if _sha256_file(F2_QMP_READOUT) != F2_QMP_READOUT_SHA256:
        raise ValueError("F2_QMP baseline readout SHA256 does not match the approved receipt")
    payload = json.loads(F2_QMP_READOUT.read_text(encoding="utf-8"))
    comparisons = payload.get("comparisons", {})
    clean = comparisons.get("clean", {}).get("q", {})
    masked = [
        comparisons[name]["q"]["full_mAP"]
        for name in (
            "mask_25_101", "mask_25_202", "mask_25_303",
            "mask_50_101", "mask_50_202", "mask_50_303",
            "mask_75_101", "mask_75_202", "mask_75_303",
        )
    ]
    if payload.get("status") != "COMPLETE" or payload.get("step") != 3600 or payload.get("head") != "q":
        raise ValueError("F2_QMP baseline readout is not the approved complete q@3600 evaluation")
    if payload.get("checkpoint_sha256") != "58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e":
        raise ValueError("F2_QMP baseline checkpoint identity mismatch")
    if abs(float(clean["full_mAP"]) - 0.49166918150172645) > 1e-15:
        raise ValueError("F2_QMP baseline clean q mAP mismatch")
    if abs(sum(float(value) for value in masked) / len(masked) - 0.3557525980363653) > 1e-15:
        raise ValueError("F2_QMP baseline masked q mAP mismatch")
    return payload


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
        "data_identity", "diagnostic_sha256", "diagnostic_path_sha256", "lambda_sig", "diagnostic_lambda_source", "lambda_mp_source",
        "method_version", "positive_pool", "main_query", "loss_coefficient_identity",
        "sampling_identity", "sigreg_status", "main_photo_objective", "main_photo_temperature",
        "objective_identity", "primary_comparison", "diagnostic_verified_path_sha256",
        "raw_diagnostic_path_sha256", "diagnostic_input_sha256", "diagnostic_data_identity",
        "diagnostic_initialization_hashes", "lambda_mp_t", "lambda_mp_q", "diagnostic_source_snapshot_hash",
        "diagnostic_selection_metadata", "lambda_selection_metadata", "selection_metadata",
        "diagnostic_identity", "coefficient_identity", "tau", "diagnostic_path", "raw_diagnostic_path",
        "training_main_query", "evaluation_query", "evaluation_adapter", "main_q",
        *_PHOTO_CE_METADATA,
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
    arm_protocol = _arm_protocol(args.arm, args.campaign_id, args.diagnostic)
    lambda_photo_ce = getattr(args, "lambda_photo_ce", None)
    if args.arm == "F2_MP_PCE":
        if lambda_photo_ce is None:
            raise ValueError("F2_MP_PCE requires explicit --lambda-photo-ce; no production value is selected")
        _validate_photo_ce_coefficient(lambda_photo_ce)
    elif lambda_photo_ce is not None:
        raise ValueError("only F2_MP_PCE accepts --lambda-photo-ce")
    tqmp_diagnostic = _validate_f2_tqmp_diagnostic(args.diagnostic) if args.arm == "F2_TQMP" else None
    if args.arm in {"F2_QMP", "F2_TQMP", "F2_MP_PCE"}:
        _validate_f2_qmp_readout()
    main_photo_objective = arm_protocol.get("main_photo_objective", "paired_softplus")
    if args.max_steps < 1 or args.max_steps > TOTAL_STEPS:
        raise ValueError("max_steps must be between 1 and 3600")
    if args.smoke and args.max_steps != 2:
        raise ValueError("--smoke requires --max-steps 2")
    if not args.smoke and args.max_steps != TOTAL_STEPS:
        raise ValueError("full campaign must run exactly 3600 updates")
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
    architecture = arm_protocol["architecture"]
    model = CoupledPredictiveModel(encoder, tokenizer, train_classmap, architecture=architecture).to(device)
    if args.arm == "R0":
        model.predictor = None
    initialization = _initialization_hashes(model)
    if tqmp_diagnostic is not None and initialization != tqmp_diagnostic["initialization_hashes"]:
        raise ValueError("F2_TQMP initialization does not match the approved diagnostic binding")
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
    if args.arm in {"F2", "F2_SIG", "F2_MP", "F2_QMP", "F2_TQMP", "F2_MP_PCE"}:
        config.update({
            "method_version": arm_protocol["method_version"],
            "positive_pool": arm_protocol["positive_pool"],
            "main_query": "q" if args.arm in {"F2_QMP", "F2_MP_PCE"} else "mu_i",
            "main_photo_objective": main_photo_objective,
            "main_photo_temperature": 0.07 if main_photo_objective in {"multi_positive_supervised_contrastive", "multi_positive_pooled_contrastive", "multi_positive_three_head_contrastive"} else None,
            "objective_identity": "rank_q_mp=multi_positive_supervised_contrastive_over_unique_live_photobank" if args.arm == "F2_QMP" else "MP(mu_i)+lambda_mp_t*MP(mu_t)+lambda_mp_q*MP(q);existing_auxiliaries_unchanged" if args.arm == "F2_TQMP" else "rank_i=multi_positive_supervised_contrastive_over_unique_live_photobank" if main_photo_objective == "multi_positive_supervised_contrastive" else "rank_i=paired_softplus",
            "loss_coefficient_identity": ({
                "rank_q_mp": 1.0, "ce_i": 1.0, "ce_t_aux": 0.25,
                "rank_pool": 0.25, "ce_pool": 0.25, "align_i": 0.05, "align_t": 0.05,
                "anchor_i": 0.5, "anchor_t": 0.5, "sigreg": 0.0,
            } if args.arm == "F2_QMP" else ({
                "rank_i": 1.0, "rank_t_mp": F2_TQMP_LAMBDA_T, "rank_q_mp": F2_TQMP_LAMBDA_Q,
                "ce_i": 1.0, "ce_t_aux": 0.25,
                "rank_pool": 0.25, "ce_pool": 0.25, "align_i": 0.05, "align_t": 0.05,
                "anchor_i": 0.5, "anchor_t": 0.5, "sigreg": 0.0,
            } if args.arm == "F2_TQMP" else {
                "rank_i": 1.0, "ce_i": 1.0, "ce_t_aux": 0.25,
                "rank_pool": 0.25, "ce_pool": 0.25, "align_i": 0.05, "align_t": 0.05,
                "anchor_i": 0.5, "anchor_t": 0.5, "sigreg": 0.0,
            })),
            "sampling_identity": {
                "active_positive_pool": "full", "active_positive_pool_count": 58950,
                "active_negative_pool": "full_other_class", "active_negative_pool_count": 58950,
                "same_photo_preprocessing": True,
                "canonical_pairing_records": 46624,
                "canonical_pairing_unique_photo_pool": 8400,
                "pairing_manifest_role": "audit_only;not_positive_sampling_source",
            },
            "sigreg_status": "fixed_f2_diagnostic" if args.arm == "F2_SIG" else "disabled_control",
        })
    if args.arm == "F2_TQMP":
        config.update({
            "lambda_mp_t": tqmp_diagnostic["lambda_mp_t"],
            "lambda_mp_q": tqmp_diagnostic["lambda_mp_q"],
            "diagnostic_sha256": tqmp_diagnostic["diagnostic_sha256"],
            "diagnostic_path_sha256": tqmp_diagnostic["diagnostic_path_sha256"],
            "raw_diagnostic_path_sha256": tqmp_diagnostic["raw_diagnostic_path_sha256"],
            "diagnostic_verified_path_sha256": tqmp_diagnostic["diagnostic_path_sha256"],
            "diagnostic_source_snapshot_hash": tqmp_diagnostic["source_snapshot_hash"],
            "diagnostic_data_identity": tqmp_diagnostic["data_identity"],
            "diagnostic_initialization_hashes": tqmp_diagnostic["initialization_hashes"],
            "diagnostic_input_sha256": tqmp_diagnostic["input_sha256"],
            "diagnostic_selection_metadata": tqmp_diagnostic["selection_metadata"],
            "selection_metadata": tqmp_diagnostic["selection_metadata"],
            "diagnostic_path": tqmp_diagnostic["diagnostic_path"],
            "raw_diagnostic_path": tqmp_diagnostic["raw_diagnostic_path"],
            "diagnostic_identity": {
                "verified_path": tqmp_diagnostic["diagnostic_path"],
                "verified_path_sha256": tqmp_diagnostic["diagnostic_path_sha256"],
                "raw_path": tqmp_diagnostic["raw_diagnostic_path"],
                "raw_path_sha256": tqmp_diagnostic["raw_diagnostic_path_sha256"],
                "source_snapshot_hash": tqmp_diagnostic["source_snapshot_hash"],
                "data_identity": tqmp_diagnostic["data_identity"],
                "initialization_hashes": tqmp_diagnostic["initialization_hashes"],
                "input_sha256": tqmp_diagnostic["input_sha256"],
            },
        })
    if args.arm == "F2_MP":
        config["primary_comparison"] = {
            "metric": "clean/full_mAP", "step": 3600,
            "baseline_arm": "F2", "baseline_wandb_run_id": "1wxvk2lk",
            "selection": "fixed_step;best_clean_and_best_masked_are_auxiliary_prefix_AP200",
        }
    if args.arm == "F2_TQMP":
        config.update({
            "main_query": "q",
            "main_q": "q",
            "training_main_query": "mu_i",
            "evaluation_query": "q",
            "evaluation_adapter": "QOnlyAdapter;pooled_head(context.mean(dim=1));predictor_forwards=0",
            "tau": 0.07,
            "coefficient_identity": {
                **config["loss_coefficient_identity"],
                "lambda_mp_t": F2_TQMP_LAMBDA_T,
                "lambda_mp_q": F2_TQMP_LAMBDA_Q,
            },
            "primary_comparison": {
                "metric": "clean/full_mAP", "step": 3600,
                "baseline_arm": "F2_MP-Q",
                "baseline_wandb_run_id": "y40hu06b",
                "baseline_training_head": "mu_i",
                "baseline_metric_source": "separate_SHA_bound_q_readout_not_training_run_metrics",
                "baseline_readout": "q",
                "baseline_readout_sha256": F2_QMP_READOUT_SHA256,
                "baseline_evaluation": "outputs/fusion_mp_q_evaluation_20260909T145000Z/summary.json",
                "baseline_fixed_values": {
                    "clean/full_mAP@3600": 0.49166918150172645,
                    "masked_macro/full_mAP@3600": 0.3557525980363653,
                },
                "selection": "fixed_step;best_clean_and_best_masked_are_auxiliary_prefix_AP200",
            },
        })
    if args.arm == "F2_MP_PCE":
        config.update({
            "lambda_photo_ce": float(lambda_photo_ce),
            "lambda_photo_ce_source": "explicit_argument_no_calibration_claim",
            "photo_ce_temperature": 0.07,
            "photo_ce_reduction": "mean_unique_live_photos_once_per_update",
            "photo_ce_target": "learned_train_class_text_bank;photos_and_text_not_detached",
            "training_main_query": "mu_i",
            "evaluation_query": "q",
            "evaluation_adapter": "QOnlyAdapter;pooled_head(context.mean(dim=1));predictor_forwards=0",
            "objective_identity": "F2_MP_total+lambda_photo_ce*CE(live_photos,text);existing_auxiliaries_unchanged",
        })
        config["loss_coefficient_identity"]["photo_ce"] = float(lambda_photo_ce)
    if args.arm in {"F2_QMP", "F2_MP_PCE"}:
        config["primary_comparison"] = {
            "metric": "clean/full_mAP", "step": 3600,
            "baseline_arm": "F2_MP-Q", "baseline_wandb_run_id": "y40hu06b",
            "baseline_training_head": "mu_i", "baseline_readout": "q",
            "baseline_readout_sha256": F2_QMP_READOUT_SHA256,
            "baseline_evaluation": "outputs/fusion_mp_q_evaluation_20260909T145000Z/summary.json",
            "baseline_fixed_values": {"clean/full_mAP@3600": 0.49166918150172645, "masked_macro/full_mAP@3600": 0.3557525980363653},
            "selection": "fixed_step;best_clean_and_best_masked_are_auxiliary_prefix_AP200",
        }
    source = capture_provenance(ROOT, resolved_config=config)
    source_hash = _verify_source(source, Path(args.campaign_root).expanduser())
    data_identity = _compact_data_identity(protocol, arm_protocol["positive_pool"])
    if tqmp_diagnostic is not None:
        diagnostic_data = tqmp_diagnostic["data_identity"]
        if (diagnostic_data.get("split") != data_identity["split"]
                or diagnostic_data.get("manifest") != data_identity["manifest"]
                or diagnostic_data.get("pool") != data_identity.get("positive_pool")):
            raise ValueError("F2_TQMP data identity does not match the approved diagnostic binding")
    config["data_identity"] = data_identity
    if args.arm in {"F2", "F2_SIG", "F2_MP", "F2_QMP", "F2_TQMP", "F2_MP_PCE"}:
        config["sampling_identity"]["positive_and_negative_pool_sha256"] = data_identity["positive_pool"]["sha256"]
    if args.arm != "F2_TQMP":
        config["diagnostic_path_sha256"] = None if not args.diagnostic else _sha256_file(Path(args.diagnostic))
    lambda_sig = 0.0
    if args.arm in {"R1_SIG", "F2_SIG"}:
        lambda_sig = _diagnostic_lambda(Path(args.diagnostic), class_ids=list(train_ids), initialization=initialization, source_hash=source_hash, config=config)
    config["lambda_sig"] = lambda_sig
    config["diagnostic_lambda_source"] = (
        "verified_receipt" if args.arm in {"R1_SIG", "F2_SIG"}
        else "not_applicable_tqmp" if args.arm == "F2_TQMP"
        else "control_zero"
    )
    if args.arm == "F2_TQMP":
        config["lambda_mp_source"] = "fixed_two_stage_raw_gradient_calibration_not_optimal"
    if args.arm in {"F2", "F2_SIG"}:
        config["loss_coefficient_identity"]["sigreg"] = lambda_sig
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
    sigreg = SIGReg(seed=42).to(device) if args.arm in {"R1_SIG", "F2_SIG"} else None
    loader = make_train_loader(protocol, transform, positive_pool=arm_protocol["positive_pool"])
    if loader.generator is None:
        raise RuntimeError("training loader has no reproducible generator")
    batches = _loader_cycle(loader)
    eval_probe = None if args.smoke else _make_eval_probe(protocol, transform, model, device, args.arm)
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
            objective_kwargs = {"lambda_mp_t": config["lambda_mp_t"], "lambda_mp_q": config["lambda_mp_q"]} if args.arm == "F2_TQMP" else {}
            if args.arm == "F2_MP_PCE":
                objective_kwargs["lambda_photo_ce"] = config["lambda_photo_ce"]
            losses = coupled_region_loss(model, batch["clean"], batch["corrupted"], batch["photos"], batch["positive_indices"], batch["negative_indices"], batch["labels"], batch["photo_labels"], batch["photo_ids"], lambda_sig=lambda_sig, sigreg=sigreg, main_photo_objective=main_photo_objective, **objective_kwargs)
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
        route_metadata = {}
        if config.get("method_version") in {"coupled_predictive_fusion_v2", "coupled_predictive_fusion_mp_v1", "coupled_predictive_fusion_qmp_v1", "coupled_predictive_fusion_tqmp_v1", "coupled_predictive_fusion_mp_photo_ce_v1"}:
            route_metadata = {"method_version": config["method_version"], "architecture": config["architecture"], "main_query": config["main_query"], "loss_coefficient_identity": config["loss_coefficient_identity"], "sampling_identity": config["sampling_identity"], "main_photo_objective": config["main_photo_objective"], "main_photo_temperature": config["main_photo_temperature"], "objective_identity": config["objective_identity"]}
        if config.get("method_version") in {"coupled_predictive_fusion_qmp_v1", "coupled_predictive_fusion_tqmp_v1", "coupled_predictive_fusion_mp_photo_ce_v1"}:
            route_metadata["primary_comparison"] = config["primary_comparison"]
        if config.get("method_version") == "coupled_predictive_fusion_mp_photo_ce_v1":
            route_metadata.update({key: config[key] for key in _PHOTO_CE_METADATA})
        if config.get("method_version") == "coupled_predictive_fusion_tqmp_v1":
            route_metadata.update({key: config[key] for key in ("tau", "main_q", "training_main_query", "evaluation_query", "evaluation_adapter", "coefficient_identity", "diagnostic_selection_metadata", "selection_metadata", "diagnostic_identity", "diagnostic_path", "raw_diagnostic_path", "lambda_mp_t", "lambda_mp_q", "diagnostic_sha256", "diagnostic_path_sha256", "raw_diagnostic_path_sha256", "diagnostic_verified_path_sha256", "diagnostic_input_sha256", "diagnostic_data_identity", "diagnostic_initialization_hashes", "diagnostic_source_snapshot_hash")})
        result = {"status": "COMPLETE", "campaign": str(config.get("method_version", "coupled_predictive_v1")), "arm": args.arm, "step": args.max_steps, "completed_steps": args.max_steps, "source_snapshot_hash": source_hash, **route_metadata, "clip_identity": clip_identity, "data_identity": data_identity, "checkpoints": checkpoints, "probes": probe_records, "selections": selections, "trace": "observation_trace.jsonl", "trace_count": trace_count, "expected_trace_count": expected_trace_count, "mask_metadata": "mask_metadata.jsonl", "training_history": "training_history.jsonl", "initialization_hashes": initialization, "frozen_original_state_hash_before": frozen_original_before, "frozen_original_state_hash_after": frozen_original_after, "memory": {"baseline": memory_baseline, "peak_allocated": torch.cuda.max_memory_allocated(device), "peak_reserved": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else {}, "wandb_run_id": None if wandb_run is None else wandb_run.run_id, "wandb_url": None if wandb_run is None else wandb_run.run_url}
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
    parser.add_argument("--arm", required=True, choices=("R0", "R1", "R1_SIG", "F2", "F2_SIG", "F2_MP", "F2_QMP", "F2_TQMP", "F2_MP_PCE"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--diagnostic")
    parser.add_argument("--lambda-photo-ce", type=float, default=argparse.SUPPRESS,
                        help="F2_MP_PCE only: explicit finite nonnegative weight; no selected default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--max-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--campaign-id", default="coupled_predictive_v1")
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    train(_parser().parse_args())
