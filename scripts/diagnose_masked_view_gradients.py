"""Offline direction-1 full/masked loss-gradient diagnostic.

This is deliberately not a trainer or evaluator.  It replays four fixed batches
from each arm's recorded training trace, loads fixed prompt checkpoints, and
uses autograd.grad without an optimizer or parameter.grad accumulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from spica.config.data import load_data_config
from spica.data.datasets import _load_rgb_image
from spica.data.manifest import read_class_map, read_manifest
from spica.data.masking import apply_ink_mask, mask_seed
from spica.data.pairing import load_pairing_manifest
from spica.evaluation.masked_view import _selected_record, resolve_pseudo_validation, sha256_file
from spica.evaluation.text_bank import encode_class_text_bank
from spica.models.checkpoint import load_prompt_checkpoint
from spica.models.clip import load_frozen_clip
from spica.models.frozen_prompt import FrozenPromptModel
from spica.models.jepa import jepa_text_classification_loss
from spica.frozen_prompt_artifacts import MASKED_VIEW_CAMPAIGN, MASKED_VIEW_ROLES, MASK_POLICY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = MASKED_VIEW_CAMPAIGN
ROLES = MASKED_VIEW_ROLES
DEFAULT_STEPS = (250, 1800)
DEFAULT_FRACTIONS = (0.25, 0.5, 0.75)
TRAIN_TRACE = "train_observations.jsonl"
SOURCE_FILES = (
    "configs/train_frozen_prompt.yaml",
    "configs/experiments/masked_view_C.yaml",
    "configs/experiments/masked_view_M.yaml",
    "src/spica/config/data.py",
    "src/spica/frozen_prompt_artifacts.py",
    "src/spica/data/datasets.py",
    "src/spica/data/manifest.py",
    "src/spica/data/masking.py",
    "src/spica/data/pairing.py",
    "src/spica/data/splits.py",
    "src/spica/data/transforms.py",
    "src/spica/evaluation/masked_view.py",
    "src/spica/evaluation/text_bank.py",
    "src/spica/models/checkpoint.py",
    "src/spica/models/clip.py",
    "src/spica/models/frozen_prompt.py",
    "src/spica/models/jepa.py",
    "src/spica/train_frozen_prompt.py",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(value, memory_format=torch.contiguous_format)


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def combine_gradient_components(
    rank: Mapping[str, torch.Tensor | None],
    ce: Mapping[str, torch.Tensor | None],
    *,
    lambda_rank: float,
    lambda_cls: float,
) -> dict[str, torch.Tensor]:
    """Combine per-scope gradients without mutating component tensors."""
    result: dict[str, torch.Tensor] = {}
    for scope in ("sketch", "photo"):
        rank_value = rank.get(scope)
        ce_value = ce.get(scope)
        if rank_value is None and ce_value is None:
            raise ValueError(f"both gradient components are missing for {scope}")
        template = rank_value if rank_value is not None else ce_value
        assert template is not None
        result[scope] = lambda_rank * (rank_value if rank_value is not None else _zero_like(template)) + lambda_cls * (ce_value if ce_value is not None else _zero_like(template))
    result["all"] = torch.cat((result["sketch"].reshape(-1), result["photo"].reshape(-1)))
    return result


def check_gradient_linearity(
    direct: Mapping[str, torch.Tensor | None], combined: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Check FP32 autograd additivity by vector error, not near-zero coordinates."""
    result = {}
    for scope in ("sketch", "photo"):
        if direct[scope] is None:
            raise AssertionError(f"missing direct total gradient: {scope}")
        actual = direct[scope].double()
        expected = combined[scope].double()
        error = actual - expected
        relative_error = float(error.norm().item() / max(actual.norm().item(), expected.norm().item(), 1e-12))
        result[scope] = {"relative_l2_error": relative_error, "max_absolute_error": float(error.abs().max().item()), "relative_l2_tolerance": 1e-5}
        if not math.isfinite(relative_error) or relative_error > 1e-5:
            raise AssertionError(f"autograd gradient additivity failed: {scope}: {result[scope]}")
    return result


def gradient_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | None]:
    """Return auditable vector geometry; cosine is null for a zero vector."""
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    dot = float(torch.dot(left, right).item())
    cosine = None if left_norm == 0.0 or right_norm == 0.0 else dot / (left_norm * right_norm)
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "dot": dot,
        "cosine": cosine,
        "right_over_left_norm": None if left_norm == 0.0 else right_norm / left_norm,
        "right_projection_onto_left": None if left_norm == 0.0 else dot / (left_norm * left_norm),
    }


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--mask-seed", type=int, default=4242)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _resolve_recorded(value: object, run_result: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidate = (PROJECT_ROOT / path).resolve()
    return candidate if candidate.exists() else (run_result.parent / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_args(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[float, ...]]:
    steps = tuple(dict.fromkeys(int(step) for step in args.steps))
    fractions = tuple(float(fraction) for fraction in args.fractions)
    if not steps or any(step < 0 for step in steps):
        raise ValueError("steps must be non-negative and non-empty")
    if args.batches != 4:
        raise ValueError("the fixed diagnostic requires exactly four recorded batches")
    if args.batch_size != 32:
        raise ValueError("the actual diagnostic requires batch-size 32")
    if not fractions or any(not math.isfinite(fraction) or not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("fractions must be finite values in (0, 1)")
    if len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return steps, fractions


def _validate_lineage(result: dict[str, Any], run_result: Path, role: str) -> tuple[dict[str, Any], Any, Any, dict[str, Any], Path]:
    if result.get("experiment_role") != role or result.get("campaign") != CAMPAIGN:
        raise ValueError(f"{role}: run role/campaign mismatch")
    config = result.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError(f"{role}: resolved_config is missing")
    expected_view = "full_full" if role.endswith("_C") else "full_masked"
    expected = {
        "experiment_role": role,
        "experiment_campaign": CAMPAIGN,
        "run_kind": "primary",
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "visual_prompt_length": 3,
        "train_class_scope": "pseudo_train",
        "official_unseen_used_for_selection": False,
        "positive_sampling": "same_class",
        "sketch_view_mode": expected_view,
        "text_mode": "hard",
        "prompt_mode": "prompt_only",
        "classification_location": "query",
        "encoder_mode": "frozen",
        "encoder_unfreeze_depth": 0,
        "train_visual_layernorm": False,
        "train_sketch_prompt": True,
        "train_photo_prompt": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{role}: config {key} is not {value!r}")
    if result.get("official_unseen_used_for_selection") is not False:
        raise ValueError(f"{role}: official-unseen selection is not explicitly excluded")
    if config.get("mask_policy") != MASK_POLICY or result.get("mask_policy") != MASK_POLICY:
        raise ValueError(f"{role}: mask_policy differs from approved policy")
    if config.get("lambda_rank") is None or config.get("lambda_cls") is None:
        raise ValueError(f"{role}: loss coefficients are missing")
    data, split, manifest = resolve_pseudo_validation(result, run_result_path=run_result)
    pairing = result.get("pairing_identity")
    if not isinstance(pairing, Mapping) or not pairing.get("sha256"):
        raise ValueError(f"{role}: pairing identity is missing")
    return result, data, split, manifest, _resolve_recorded(config["data_config"], run_result)


def _load_trace(result: dict[str, Any], run_result: Path, *, rows_needed: int) -> list[dict[str, Any]]:
    trace = result.get("observation_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("primary run is missing observation_trace")
    path = _resolve_recorded(trace.get("path"), run_result)
    if not path.is_file() or sha256_file(path) != trace.get("sha256"):
        raise ValueError(f"observation trace missing or hash-mismatched: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("observation trace contains a non-object")
                rows.append(row)
                if len(rows) == rows_needed:
                    break
    if len(rows) != rows_needed:
        raise ValueError(f"trace has fewer than {rows_needed} rows")
    expected_steps = [index // 32 for index in range(rows_needed)]
    actual_steps = [int(row.get("global_step", -1)) for row in rows]
    actual_update_steps = [int(row.get("step", -1)) for row in rows]
    if actual_steps != expected_steps or actual_update_steps != [step + 1 for step in expected_steps]:
        raise ValueError("the fixed trace rows must be contiguous 32-row batches with global steps 0..3 and step=global_step+1")
    return rows


def _entry_maps(data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    by_sketch = {str(entry.path.resolve()): entry for entry in sketches}
    by_photo = {str(entry.path.resolve()): entry for entry in photos}
    return by_sketch, by_photo


def _path_from_trace(value: object, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"trace path escapes dataset root: {value}") from error
    return path


def _selected_checkpoint(result: Mapping[str, Any], step: int) -> tuple[dict[str, Any], Path]:
    row, checkpoint = _selected_record(result, step)
    actual = int(row.get("training_global_step", row.get("step", -1)))
    if actual != step or int(row.get("step", actual)) != step:
        raise ValueError(f"selected record has actual step {actual}, expected {step}")
    return row, checkpoint


def _validate_fixed_rows(rows: Sequence[dict[str, Any]], data: Any, split: Any, *, name: str, canonical_positive_paths: set[str]) -> None:
    train_sketches = {str(entry.path.resolve()): entry for entry in split.train_sketch_entries}
    train_photos = {str(entry.path.resolve()): entry for entry in split.train_photo_entries}
    root = Path(data.root).resolve()
    for row in rows:
        sketch = _path_from_trace(row.get("path"), root)
        positive = _path_from_trace(row.get("positive_photo_path"), root)
        negative = _path_from_trace(row.get("negative_photo_path"), root)
        if str(sketch) not in train_sketches or str(positive) not in train_photos:
            raise ValueError(f"{name}: sketch/positive trace path is not in the canonical pseudo-train manifests")
        if str(negative) not in train_photos:
            raise ValueError(f"{name}: negative trace path is outside pseudo-train photos")
        if not negative.is_file():
            raise FileNotFoundError(f"{name}: negative trace image is missing: {negative}")
        label = int(row["label"])
        if int(train_sketches[str(sketch)].label) != label or int(train_photos[str(positive)].label) != label:
            raise ValueError(f"{name}: positive path/label mismatch")
        if str(positive) not in canonical_positive_paths:
            raise ValueError(f"{name}: positive path is outside the canonical pairing pool")
        if row.get("positive_photo_sha256") != sha256_file(positive):
            raise ValueError(f"{name}: positive photo historical SHA256 mismatch: {positive}")
        negative_sha = row.get("negative_photo_sha256")
        if negative_sha is not None and negative_sha != sha256_file(negative):
            raise ValueError(f"{name}: negative photo historical SHA256 mismatch: {negative}")
        if int(train_photos[str(negative)].label) == label:
            raise ValueError(f"{name}: negative path has the query label")
        if not (sketch.is_file() and positive.is_file()):
            raise FileNotFoundError(f"{name}: trace image is missing")


def _parameter_gradients(loss: torch.Tensor, parameters: Sequence[torch.Tensor]) -> dict[str, torch.Tensor | None]:
    values = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    return {
        "sketch": None if values[0] is None else values[0].detach().clone(),
        "photo": None if values[1] is None else values[1].detach().clone(),
    }


def _validate_gradients(name: str, gradients: Mapping[str, torch.Tensor | None], *, photo_none: bool = False) -> None:
    if gradients["sketch"] is None:
        raise AssertionError(f"{name}: sketch gradient is unexpectedly None")
    _assert_finite(f"{name} sketch gradient", gradients["sketch"])
    if torch.count_nonzero(gradients["sketch"]).item() == 0:
        raise AssertionError(f"{name}: sketch gradient is unexpectedly zero")
    if photo_none:
        if gradients["photo"] is not None:
            raise AssertionError(f"{name}: photo gradient must be structurally None")
    else:
        if gradients["photo"] is None:
            raise AssertionError(f"{name}: photo gradient is unexpectedly None")
        _assert_finite(f"{name} photo gradient", gradients["photo"])
        if torch.count_nonzero(gradients["photo"]).item() == 0:
            raise AssertionError(f"{name}: photo gradient is unexpectedly zero")


def _scope_vector(gradients: Mapping[str, torch.Tensor | None], scope: str, template: Mapping[str, torch.Tensor | None]) -> torch.Tensor:
    value = gradients.get(scope)
    if value is not None:
        return value.reshape(-1)
    other = template.get(scope)
    if other is None:
        raise ValueError(f"cannot materialize missing {scope} gradient")
    return torch.zeros_like(other).reshape(-1)


def _aggregate_rows(rows: Sequence[dict[str, Any]], arrays: Mapping[str, Sequence[np.ndarray]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["arm"]), int(row["step"]), float(row["fraction"])), []).append(row)
    result: dict[str, Any] = {}
    for (arm, step, fraction), values in sorted(grouped.items()):
        cosine = [item["total"]["all"]["cosine"] for item in values if item["total"]["all"]["cosine"] is not None]
        full = np.concatenate([np.concatenate((arrays["full_total_sketch"][item["row_index"]], arrays["full_total_photo"][item["row_index"]])) for item in values])
        masked = np.concatenate([np.concatenate((arrays["masked_total_sketch"][item["row_index"]], arrays["masked_total_photo"][item["row_index"]])) for item in values])
        statuses = sorted({status for item in values for status in item["mask_statuses"]})
        result[f"{arm}/step{step}/fraction{fraction:g}"] = {"batch_count": len(values), "per_batch_mean_cosine": None if not cosine else float(sum(cosine) / len(cosine)), "negative_cosine_count": sum(value is not None and value < 0 for value in cosine), "pooled_dot": float(np.dot(full, masked)), "pooled_full_norm": float(np.linalg.norm(full)), "pooled_masked_norm": float(np.linalg.norm(masked)), "realized_fraction_min": min(item for value in values for item in value["realized_fractions"]), "realized_fraction_max": max(item for value in values for item in value["realized_fractions"]), "status_counts": {status: sum(value["mask_statuses"].count(status) for value in values) for status in statuses}}
    return result

def _losses(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    labels: torch.Tensor,
    text_bank: Any,
    *,
    margin: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_normalized = torch.nn.functional.normalize(query, dim=-1)
    positive_normalized = torch.nn.functional.normalize(positive, dim=-1)
    negative_normalized = torch.nn.functional.normalize(negative, dim=-1)
    rank = torch.nn.functional.softplus(
        margin - (query_normalized * positive_normalized).sum(-1)
        + (query_normalized * negative_normalized).sum(-1)
    ).mean()
    ce, _ = jepa_text_classification_loss(
        query, text_bank.embeddings.to(query.device), text_bank.labels.to(query.device), labels,
        temperature=tau, detach_text=True,
    )
    return rank, ce


def _measurement(
    *,
    full_rank: torch.Tensor,
    full_ce: torch.Tensor,
    masked_rank: torch.Tensor,
    masked_ce: torch.Tensor,
    full_rank_grad: Mapping[str, torch.Tensor | None],
    full_ce_grad: Mapping[str, torch.Tensor | None],
    masked_rank_grad: Mapping[str, torch.Tensor | None],
    masked_ce_grad: Mapping[str, torch.Tensor | None],
    lambda_rank: float,
    lambda_cls: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    full_rank_vec = {scope: _scope_vector(full_rank_grad, scope, full_rank_grad) for scope in ("sketch", "photo")}
    full_ce_vec = {scope: _scope_vector(full_ce_grad, scope, full_rank_grad) for scope in ("sketch", "photo")}
    masked_rank_vec = {scope: _scope_vector(masked_rank_grad, scope, full_rank_grad) for scope in ("sketch", "photo")}
    masked_ce_vec = {scope: _scope_vector(masked_ce_grad, scope, full_rank_grad) for scope in ("sketch", "photo")}
    full_rank_vec["all"] = torch.cat((full_rank_vec["sketch"], full_rank_vec["photo"]))
    full_ce_vec["all"] = torch.cat((full_ce_vec["sketch"], full_ce_vec["photo"]))
    masked_rank_vec["all"] = torch.cat((masked_rank_vec["sketch"], masked_rank_vec["photo"]))
    masked_ce_vec["all"] = torch.cat((masked_ce_vec["sketch"], masked_ce_vec["photo"]))
    full_total = combine_gradient_components(full_rank_vec, full_ce_vec, lambda_rank=lambda_rank, lambda_cls=lambda_cls)
    masked_total = combine_gradient_components(masked_rank_vec, masked_ce_vec, lambda_rank=lambda_rank, lambda_cls=lambda_cls)
    mixed = {scope: 0.5 * (full_total[scope] + masked_total[scope]) for scope in ("sketch", "photo", "all")}
    metrics: dict[str, Any] = {
        "losses": {"full_rank": float(full_rank.item()), "full_ce": float(full_ce.item()), "masked_rank": float(masked_rank.item()), "masked_ce": float(masked_ce.item())},
        "components": {},
        "rank_vs_ce": {},
        "total": {},
        "totalmixed_vs_full": {},
    }
    for name, left, right in (
        ("full_rank_vs_masked_rank", full_rank_vec, masked_rank_vec),
        ("full_ce_vs_masked_ce", full_ce_vec, masked_ce_vec),
    ):
        metrics["components"][name] = {scope: gradient_metrics(left[scope], right[scope]) for scope in ("sketch", "photo", "all")}
    for view, rank, ce in (("full", full_rank_vec, full_ce_vec), ("masked", masked_rank_vec, masked_ce_vec)):
        metrics["rank_vs_ce"][view] = {scope: gradient_metrics(rank[scope], ce[scope]) for scope in ("sketch", "photo", "all")}
    metrics["total"] = {scope: gradient_metrics(full_total[scope], masked_total[scope]) for scope in ("sketch", "photo", "all")}
    metrics["totalmixed_vs_full"] = {scope: gradient_metrics(full_total[scope], mixed[scope]) for scope in ("sketch", "photo", "all")}
    metrics["totalmixed_vs_full_projection"] = {scope: (None if torch.linalg.vector_norm(full_total[scope]) == 0 else float(torch.dot(mixed[scope], full_total[scope]).item() / full_total[scope].square().sum().item())) for scope in ("sketch", "photo", "all")}
    raw = {
        "full_rank_sketch": full_rank_vec["sketch"].cpu().numpy(), "full_rank_photo": full_rank_vec["photo"].cpu().numpy(),
        "full_ce_sketch": full_ce_vec["sketch"].cpu().numpy(), "full_ce_photo": full_ce_vec["photo"].cpu().numpy(),
        "masked_rank_sketch": masked_rank_vec["sketch"].cpu().numpy(), "masked_rank_photo": masked_rank_vec["photo"].cpu().numpy(),
        "masked_ce_sketch": masked_ce_vec["sketch"].cpu().numpy(), "masked_ce_photo": masked_ce_vec["photo"].cpu().numpy(),
        "full_total_sketch": full_total["sketch"].cpu().numpy(), "full_total_photo": full_total["photo"].cpu().numpy(),
        "masked_total_sketch": masked_total["sketch"].cpu().numpy(), "masked_total_photo": masked_total["photo"].cpu().numpy(),
    }
    return metrics, raw


def _cpu_preflight(
    *,
    result: dict[str, Any],
    run_result: Path,
    data: Any,
    split: Any,
    trace: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    row, checkpoint_value = _selected_record(result, 0)
    checkpoint = _resolve_recorded(checkpoint_value, run_result)
    if sha256_file(checkpoint) != row["checkpoint_sha256"]:
        raise ValueError("CPU preflight checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    clip = load_frozen_clip(model_name=str(config["model_name"]), pretrained=config.get("pretrained"), device=torch.device("cpu"))
    model = FrozenPromptModel(clip.encoder.model.visual, prompt_length=int(config["visual_prompt_length"]), train_visual_layernorm=False, train_sketch_prompt=True, train_photo_prompt=True)
    load_prompt_checkpoint(model, payload, expected_config={key: config[key] for key in ("model_name", "pretrained", "visual_prompt_length", "train_visual_layernorm", "train_sketch_prompt", "train_photo_prompt")})
    if any(parameter.requires_grad for parameter in model.visual.parameters()):
        raise AssertionError("CPU preflight: frozen visual has trainable parameters")
    if model.training or any(module.training for module in model.visual.modules()):
        raise AssertionError("CPU preflight: visual contains a training-mode module")
    names = read_class_map(data.train.class_map)
    text_bank = encode_class_text_bank(clip.encoder, clip.tokenizer, {class_id: names[class_id] for class_id in split.train_class_ids}, prompt_template=str(config["prompt_template"]))
    root = data.root.resolve()
    by_sketch, by_photo = _entry_maps(data)
    transform = clip.transform
    def image(value: object, entries: Mapping[str, Any]) -> torch.Tensor:
        path = _path_from_trace(value, root)
        entry = entries[str(path)]
        return transform(_load_rgb_image(entry))
    rows = trace[:2]
    sketches = torch.stack([image(item["path"], by_sketch) for item in rows])
    positives = torch.stack([image(item["positive_photo_path"], by_photo) for item in rows])
    negatives = torch.stack([image(item["negative_photo_path"], by_photo) for item in rows])
    query = model(sketches)
    query_stack = model(torch.cat((sketches, sketches), dim=0))
    if not torch.allclose(query_stack[:2], query, atol=1e-6, rtol=1e-5) or not torch.allclose(query_stack[2:], query, atol=1e-6, rtol=1e-5):
        raise AssertionError("CPU preflight: stacked four-example forward disagrees with separate two-example forward")
    positive, negative = model.encode_photo(torch.cat((positives, negatives), dim=0)).chunk(2, dim=0)
    labels = torch.tensor([int(item["label"]) for item in rows], dtype=torch.long)
    rank, ce = _losses(query, positive, negative, labels, text_bank, margin=float(config["margin"]), tau=float(config["tau_cls"]))
    _assert_finite("CPU preflight rank loss", rank)
    _assert_finite("CPU preflight CE loss", ce)
    params = (model.sketch_prompt, model.photo_prompt)
    rank_grad = _parameter_gradients(rank, params)
    ce_grad = _parameter_gradients(ce, params)
    _validate_gradients("CPU preflight rank", rank_grad)
    _validate_gradients("CPU preflight CE", ce_grad, photo_none=True)
    return {"status": "PASS", "device": "cpu", "rows": 2, "checkpoint_step": int(payload["step"]), "rank_loss": float(rank.item()), "ce_loss": float(ce.item()), "shared_stack_verified": True, "dropout_disabled": True, "visual_frozen": True}


def _source_snapshot(output_dir: Path) -> dict[str, Any]:
    files = [Path(__file__).resolve(), *(PROJECT_ROOT / relative for relative in SOURCE_FILES)]
    rows = []
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(PROJECT_ROOT)
        target = output_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({"path": relative.as_posix(), "sha256": sha256_file(source), "bytes": source.stat().st_size})
    index = {"files": rows, "sha256": canonical_sha256(rows)}
    (output_dir / "source_snapshot" / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    steps, fractions = _validate_args(args)
    pilot_root = args.pilot_root.resolve()
    if not pilot_root.is_dir():
        raise FileNotFoundError(pilot_root)
    run_paths = {role: pilot_root / role[-1] / "run_result.json" for role in ROLES}
    if any(not path.is_file() for path in run_paths.values()):
        raise FileNotFoundError("pilot root must contain C/run_result.json and M/run_result.json")
    results = {role: _read_json(path) for role, path in run_paths.items()}
    lineage = {}
    for role in ROLES:
        lineage[role] = _validate_lineage(results[role], run_paths[role], role)
    data = lineage[ROLES[0]][1]
    split = lineage[ROLES[0]][2]
    data_config = load_data_config(lineage[ROLES[0]][4])
    if data_config.root.resolve() != data.root.resolve():
        raise ValueError("stored data root identity changed")
    traces = {role: _load_trace(results[role], run_paths[role], rows_needed=args.batches * args.batch_size) for role in ROLES}
    pairing_path = _resolve_recorded(results[ROLES[0]]["resolved_config"]["pairing_manifest_path"], run_paths[ROLES[0]])
    canonical_pairing = load_pairing_manifest(pairing_path, dataset_root=data_config.root, sketch_entries=split.train_sketch_entries, photo_entries=split.train_photo_entries)
    canonical_positive_paths = {str(entry.path.resolve()) for entry in canonical_pairing.values()}
    _validate_fixed_rows(traces[ROLES[0]], data_config, split, name="C", canonical_positive_paths=canonical_positive_paths)
    _validate_fixed_rows(traces[ROLES[1]], data_config, split, name="M", canonical_positive_paths=canonical_positive_paths)
    if data.root.resolve() != data_config.root.resolve():
        raise ValueError("resolved data roots differ between run and stored config")
    common_fields = ("model_name", "pretrained", "prompt_template", "prompt_mode", "text_mode", "loss", "lambda_rank", "lambda_cls", "margin", "tau_cls", "batch_size", "seed", "pseudo_val_seed", "positive_sampling", "pairing_manifest_path", "mask_policy", "encoder_mode", "encoder_unfreeze_depth")
    for field in common_fields:
        if results[ROLES[0]]["resolved_config"].get(field) != results[ROLES[1]]["resolved_config"].get(field):
            raise ValueError(f"C/M resolved config mismatch in common field {field}")
    for field in ("source_snapshot_hash", "pseudo_split_identity", "manifest_identity", "pairing_identity", "mask_policy"):
        if results[ROLES[0]].get(field) != results[ROLES[1]].get(field):
            raise ValueError(f"C/M lineage mismatch in common field {field}")
    key_fields = ("path", "global_step", "step", "label", "positive_photo_path", "negative_photo_path", "views")
    if [[row.get(field) for field in key_fields] for row in traces[ROLES[0]]] != [[row.get(field) for field in key_fields] for row in traces[ROLES[1]]]:
        raise ValueError("C/M fixed trace rows differ")
    preflight = _cpu_preflight(result=results[ROLES[0]], run_result=run_paths[ROLES[0]], data=data_config, split=split, trace=traces[ROLES[0]], config=results[ROLES[0]]["resolved_config"])
    if args.preflight_only:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "cpu_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": "CPU_PREFLIGHT_PASS", "cpu_preflight": preflight}
    names = read_class_map(data_config.train.class_map)
    train_names = {class_id: names[class_id] for class_id in split.train_class_ids}
    text_model = None
    tensor_cache: dict[str, torch.Tensor] = {}
    raw_rows: list[dict[str, Any]] = []
    raw_arrays: dict[str, list[np.ndarray]] = {}
    sample_manifest: list[dict[str, Any]] = []
    checkpoint_hash_before: dict[str, str] = {}
    checkpoint_lineage: list[dict[str, Any]] = []
    model_hashes: dict[str, dict[str, str]] = {}
    fixed_trace_manifest: list[dict[str, Any]] = []
    for role in ROLES:
        for index, trace_row in enumerate(traces[role]):
            fixed_trace_manifest.append({
                "arm": role,
                "batch_index": index // args.batch_size,
                "sample_index": index % args.batch_size,
                "global_step": int(trace_row["global_step"]),
                "step": int(trace_row["step"]),
                "path": str(trace_row["path"]),
                "label": int(trace_row["label"]),
                "positive_photo_path": str(trace_row["positive_photo_path"]),
                "positive_photo_sha256": trace_row.get("positive_photo_sha256"),
                "negative_photo_path": str(trace_row["negative_photo_path"]),
                "negative_photo_sha256": trace_row.get("negative_photo_sha256"),
                "sketch_file_sha256": sha256_file(_path_from_trace(trace_row["path"], data_config.root.resolve())),
                "positive_file_sha256": sha256_file(_path_from_trace(trace_row["positive_photo_path"], data_config.root.resolve())),
                "negative_file_sha256": sha256_file(_path_from_trace(trace_row["negative_photo_path"], data_config.root.resolve())),
            })
    for role in ROLES:
        for step in steps:
            row, checkpoint_value = _selected_checkpoint(results[role], step)
            checkpoint = _resolve_recorded(checkpoint_value, run_paths[role])
            if not checkpoint.is_file() or sha256_file(checkpoint) != row.get("checkpoint_sha256"):
                raise ValueError(f"{role} step {step}: checkpoint hash mismatch")
            checkpoint_hash_before[str(checkpoint)] = sha256_file(checkpoint)
            checkpoint_lineage.append({"arm": role, "step": step, "path": str(checkpoint), "sha256": checkpoint_hash_before[str(checkpoint)]})
    start = time.perf_counter()
    for role in ROLES:
        result, run_data, split, manifest, _ = lineage[role]
        config = result["resolved_config"]
        clip = load_frozen_clip(model_name=str(config["model_name"]), pretrained=config.get("pretrained"), device=torch.device(args.device))
        clip_hash_before = state_hash(clip.encoder.model)
        if any(parameter.requires_grad for parameter in clip.encoder.model.parameters()):
            raise AssertionError(f"{role}: original CLIP visual has trainable parameters")
        model = FrozenPromptModel(clip.encoder.model.visual, prompt_length=int(config["visual_prompt_length"]), train_visual_layernorm=False, train_sketch_prompt=True, train_photo_prompt=True).to(args.device)
        model.eval()
        for step in steps:
            row, checkpoint_value = _selected_checkpoint(result, step)
            checkpoint = _resolve_recorded(checkpoint_value, run_paths[role])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            required_payload = {
                "format_version": 2,
                "model_type": "frozen_prompt_v2",
                "step": step,
                "training_global_step": step,
                "campaign": CAMPAIGN,
                "experiment_role": role,
                "run_kind": "primary",
                "data_split_identity": result["pseudo_split_identity"],
                "pairing_identity": result["pairing_identity"],
                "mask_policy": result["mask_policy"],
                "trainable_parameter_names": ["sketch_prompt", "photo_prompt"],
            }
            if any(payload.get(key) != value for key, value in required_payload.items()) or payload.get("resolved_config") != config:
                raise ValueError(f"{role} step {step}: checkpoint lineage mismatch")
            if payload.get("source_snapshot_hash") != result.get("source_snapshot_hash") or payload.get("data_manifest_identity") != result.get("manifest_identity"):
                raise ValueError(f"{role} step {step}: checkpoint provenance mismatch")
            if set(payload.get("model_state_dict", {})) != {"sketch_prompt", "photo_prompt"}:
                raise ValueError(f"{role} step {step}: checkpoint state keys are not prompt-only")
            load_prompt_checkpoint(model, payload, expected_config={key: config[key] for key in ("model_name", "pretrained", "visual_prompt_length", "train_visual_layernorm", "train_sketch_prompt", "train_photo_prompt")})
            step_model_hash_before = state_hash(model)
            if tuple(model.trainable_parameter_names) != ("sketch_prompt", "photo_prompt"):
                raise ValueError("diagnostic model has unexpected trainable parameters")
            if text_model is None:
                text_model = encode_class_text_bank(clip.encoder, clip.tokenizer, train_names, prompt_template=str(config["prompt_template"]))
            transform = clip.transform
            params = (model.sketch_prompt, model.photo_prompt)
            by_sketch, by_photo = _entry_maps(data_config)
            root = data_config.root.resolve()
            def cached_tensor(path: Path) -> torch.Tensor:
                key = str(path)
                if key not in tensor_cache:
                    entry = by_sketch[key] if key in by_sketch else by_photo[key]
                    tensor_cache[key] = transform(_load_rgb_image(entry)).cpu()
                return tensor_cache[key]
            for batch_index in range(args.batches):
                batch = traces[role][batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
                tensors = [cached_tensor(_path_from_trace(row["path"], root)) for row in batch]
                positives = [cached_tensor(_path_from_trace(row["positive_photo_path"], root)) for row in batch]
                negatives = [cached_tensor(_path_from_trace(row["negative_photo_path"], root)) for row in batch]
                full_images = torch.stack(tensors).to(args.device)
                photo_images = torch.cat((torch.stack(positives), torch.stack(negatives))).to(args.device)
                batch_input_hashes = {"full": tensor_hash(full_images), "positive_negative_photos": tensor_hash(photo_images)}
                labels = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long, device=args.device)
                photo_values = model.encode_photo(photo_images)
                positive, negative = photo_values.split((args.batch_size, args.batch_size))
                full_query = model(full_images)
                full_rank, full_ce = _losses(full_query, positive, negative, labels, text_model, margin=float(config["margin"]), tau=float(config["tau_cls"]))
                _assert_finite("full rank loss", full_rank)
                _assert_finite("full CE loss", full_ce)
                full_rank_grad = _parameter_gradients(full_rank, params)
                full_ce_grad = _parameter_gradients(full_ce, params)
                _validate_gradients("full rank", full_rank_grad)
                _validate_gradients("full CE", full_ce_grad, photo_none=True)
                lambda_rank, lambda_cls = float(config["lambda_rank"]), float(config["lambda_cls"])
                full_total_loss = lambda_rank * full_rank + lambda_cls * full_ce
                full_total_grad = _parameter_gradients(full_total_loss, params)
                _validate_gradients("full total", full_total_grad)
                combined_full = combine_gradient_components(full_rank_grad, full_ce_grad, lambda_rank=lambda_rank, lambda_cls=lambda_cls)
                full_linearity = check_gradient_linearity(full_total_grad, combined_full)
                for fraction in fractions:
                    masked_images: list[torch.Tensor] = []
                    for sample_index, (image, trace_row) in enumerate(zip(tensors, batch, strict=True)):
                        relative = _path_from_trace(trace_row["path"], root).relative_to(root).as_posix()
                        masked, metadata = apply_ink_mask(image.cpu(), fraction=fraction, seed=mask_seed(args.mask_seed, relative, view=0), ink_threshold=float(config["mask_policy"]["ink_threshold"]))
                        trace_views = trace_row.get("views")
                        if isinstance(trace_views, list) and trace_views and metadata["input_sha256"] != trace_views[0].get("input_sha256"):
                            raise ValueError(f"{role} step {step} {relative}: diagnostic eval transform differs from recorded training input")
                        masked_images.append(masked)
                        sample_manifest.append({"arm": role, "step": step, "batch_index": batch_index, "sample_index": sample_index, "path": relative, "label": int(trace_row["label"]), "fraction": fraction, "input_sha256": metadata["input_sha256"], "output_sha256": metadata["output_sha256"], "mask": metadata})
                    masked_query = model(torch.stack(masked_images).to(args.device))
                    masked_rank, masked_ce = _losses(masked_query, positive, negative, labels, text_model, margin=float(config["margin"]), tau=float(config["tau_cls"]))
                    _assert_finite("masked rank loss", masked_rank)
                    _assert_finite("masked CE loss", masked_ce)
                    masked_rank_grad = _parameter_gradients(masked_rank, params)
                    masked_ce_grad = _parameter_gradients(masked_ce, params)
                    _validate_gradients("masked rank", masked_rank_grad)
                    _validate_gradients("masked CE", masked_ce_grad, photo_none=True)
                    masked_total_loss = lambda_rank * masked_rank + lambda_cls * masked_ce
                    masked_total_grad = _parameter_gradients(masked_total_loss, params)
                    _validate_gradients("masked total", masked_total_grad)
                    combined_masked = combine_gradient_components(masked_rank_grad, masked_ce_grad, lambda_rank=lambda_rank, lambda_cls=lambda_cls)
                    masked_linearity = check_gradient_linearity(masked_total_grad, combined_masked)
                    metrics, arrays = _measurement(full_rank=full_rank, full_ce=full_ce, masked_rank=masked_rank, masked_ce=masked_ce, full_rank_grad=full_rank_grad, full_ce_grad=full_ce_grad, masked_rank_grad=masked_rank_grad, masked_ce_grad=masked_ce_grad, lambda_rank=lambda_rank, lambda_cls=lambda_cls)
                    measurement_id = len(raw_rows)
                    mask_batch = [item["mask"] for item in sample_manifest[-args.batch_size:]]
                    raw_rows.append({"row_index": measurement_id, "arm": role, "step": step, "batch_index": batch_index, "fraction": fraction, "mask_seed": args.mask_seed, "realized_fractions": [float(item["realized_fraction"]) for item in mask_batch], "mask_statuses": [str(item["status"]) for item in mask_batch], "used_mask_input_hashes": [str(item["input_sha256"]) for item in mask_batch], "used_mask_output_hashes": [str(item["output_sha256"]) for item in mask_batch], "gradient_component_keys": sorted(arrays), "coefficients": {"lambda_rank": lambda_rank, "lambda_cls": lambda_cls}, "batch_input_hashes": batch_input_hashes, "gradient_linearity": {"full": full_linearity, "masked": masked_linearity}, **metrics})
                    for key, array in arrays.items():
                        raw_arrays.setdefault(key, []).append(array)
                del full_query, photo_values
                if args.device == "cuda":
                    torch.cuda.synchronize()
            step_model_hash_after = state_hash(model)
            if step_model_hash_after != step_model_hash_before:
                raise AssertionError(f"{role} step {step}: model state changed during diagnostic")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise AssertionError(f"{role} step {step}: autograd.grad left parameter.grad populated")
            model_hashes[f"{role}/step{step}"] = {"before": step_model_hash_before, "after": step_model_hash_after}
        if state_hash(clip.encoder.model) != clip_hash_before:
            raise AssertionError(f"{role}: CLIP state changed during diagnostic")
        del model, clip
    after_checkpoint_hash = {path: sha256_file(Path(path)) for path in checkpoint_hash_before}
    if after_checkpoint_hash != checkpoint_hash_before:
        raise AssertionError("checkpoint bytes changed during diagnostic")
    args.output_dir.mkdir(parents=True)
    source_index = _source_snapshot(args.output_dir)
    run_result_inputs = {}
    for role, path in run_paths.items():
        target = args.output_dir / f"input_run_result_{role[-1]}.json"
        shutil.copy2(path, target)
        run_result_inputs[role] = {"path": str(path), "snapshot": target.name, "sha256": sha256_file(target)}
    aggregates = _aggregate_rows(raw_rows, raw_arrays)
    np.savez_compressed(args.output_dir / "raw_gradients.npz", **{key: np.stack(values) for key, values in raw_arrays.items()})
    (args.output_dir / "raw_rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows), encoding="utf-8")
    (args.output_dir / "batch_sample_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_manifest), encoding="utf-8")
    (args.output_dir / "fixed_trace_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in fixed_trace_manifest), encoding="utf-8")
    provenance = {
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": args.device,
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
        "elapsed_seconds": time.perf_counter() - start,
        "checkpoint_sha256_before_after": {path: {"before": value, "after": after_checkpoint_hash[path]} for path, value in checkpoint_hash_before.items()},
        "model_state_hash_before_after": model_hashes,
        "source_snapshot": source_index,
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "diagnostic": "direction1_masked_view_loss_gradients",
        "campaign": CAMPAIGN,
        "pilot_root": str(pilot_root),
        "arms": list(ROLES),
        "steps": list(steps),
        "fractions": list(fractions),
        "mask_seed": args.mask_seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "measurement_pairs": len(raw_rows),
        "checkpoint_lineage": checkpoint_lineage,
        "run_result_inputs": run_result_inputs,
        "aggregate_summary": aggregates,
        "original_source_hashes": {role: results[role].get("source_snapshot_hash") for role in ROLES},
        "current_diagnostic_source_snapshot": source_index,
        "config_sha256": {role: canonical_sha256(results[role]["resolved_config"]) for role in ROLES},
        "loss_coefficients": {role: {"lambda_rank": results[role]["resolved_config"]["lambda_rank"], "lambda_cls": results[role]["resolved_config"]["lambda_cls"], "margin": results[role]["resolved_config"]["margin"], "tau_cls": results[role]["resolved_config"]["tau_cls"]} for role in ROLES},
        "model_contract": {"model_type": "frozen_prompt_v2", "trainable_parameters": ["sketch_prompt", "photo_prompt"], "teacher_used": False, "optimizer_used": False, "official_unseen_used": False},
        "data_contract": {"data_config": str(lineage[ROLES[0]][4]), "dataset_root": str(data_config.root), "pseudo_split_identity": results[ROLES[0]].get("pseudo_split_identity"), "manifest_identity": results[ROLES[0]].get("manifest_identity"), "fixed_trace_rows": args.batches * args.batch_size, "transform": "existing CLIP eval_transform; no augmentation"},
        "files": {"raw_gradients": "raw_gradients.npz", "raw_rows": "raw_rows.jsonl", "batch_sample_manifest": "batch_sample_manifest.jsonl", "fixed_trace_manifest": "fixed_trace_manifest.jsonl"},
        "cpu_preflight": preflight,
        "provenance": provenance,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    try:
        run_diagnostic(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILED", "error_type": type(error).__name__, "error": str(error), "argv": sys.argv, "torch": torch.__version__}
        (args.output_dir / "masked_view_gradient_diagnostic_failed.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
