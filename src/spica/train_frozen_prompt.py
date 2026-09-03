"""Train and probe the controlled frozen-prompt v2 campaign."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
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
from .data.splits import make_classwise_retrieval_split, split_manifest_identity
from .evaluation.frozen_prompt import (
    cache_identity,
    encode_prompted_loader,
    evaluate_prompted,
    geometry_payload,
    hash_state,
    load_prompt_cache,
    save_prompt_cache,
)
from .evaluation.text_bank import (
    EncodedTextBank,
    SoftPromptTextBank,
    encode_class_text_bank,
)
from .frozen_prompt_artifacts import (
    CAMPAIGN,
    ROLE_TREATMENTS,
    ROLES,
    SMOKE_CAMPAIGN,
    canonical_sha256,
    ensure_manifest,
    manifest_entry_identity,
    treatment_from_config,
)
from .models.clip import (
    FrozenClipEncoder,
    load_frozen_clip,
    load_trainable_sketch_hidden_encoder,
)
from .models.frozen_prompt import FrozenPromptModel
from .models.jepa import classification_accuracy, jepa_text_classification_loss
from .provenance import capture_provenance, capture_rng_state, restore_rng_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")
PRIMARY_PROBE_STEPS = (0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400)
PROMPT_ROLES = {
    "frozen_prompt_v2_FP1",
    "frozen_prompt_v2_FP1S",
    "frozen_prompt_v2_FP2",
    "frozen_prompt_v2_FP3",
    "frozen_prompt_v2_FP_LN",
}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _path(value: object) -> Path:
    candidate = Path(str(value)).expanduser()
    return (
        candidate
        if candidate.is_absolute() or candidate.exists()
        else PROJECT_ROOT / candidate
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    random.seed(torch.initial_seed() % (2**32) + worker_id)


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _validate(args: DictConfig) -> None:
    role = str(args.experiment_role)
    if role not in ROLES:
        raise ValueError(f"experiment_role must be exactly one of {ROLES}")
    campaign = str(args.experiment_campaign)
    run_kind = str(args.run_kind)
    if run_kind == "primary" and campaign != CAMPAIGN:
        raise ValueError("primary frozen-prompt runs must use the v2 campaign")
    if run_kind == "smoke" and campaign != SMOKE_CAMPAIGN:
        raise ValueError("smoke frozen-prompt runs must use the v2 smoke campaign")
    if run_kind not in {"primary", "smoke"}:
        raise ValueError("run_kind must be primary or smoke")

    observed = treatment_from_config(OmegaConf.to_container(args, resolve=True))
    expected = ROLE_TREATMENTS[role]
    mismatches = {
        key: (observed[key], expected[key])
        for key in expected
        if not _same_value(observed.get(key), expected[key])
    }
    if mismatches:
        raise ValueError(f"{role} has an ambiguous treatment: {mismatches}")

    for name in (
        "visual_prompt_learning_rate",
        "soft_prompt_learning_rate",
        "visual_layernorm_learning_rate",
        "encoder_learning_rate",
        "visual_prompt_weight_decay",
        "soft_prompt_weight_decay",
        "visual_layernorm_weight_decay",
        "encoder_weight_decay",
        "margin",
        "tau_cls",
    ):
        value = float(args[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in (
        "visual_prompt_learning_rate",
        "soft_prompt_learning_rate",
        "visual_layernorm_learning_rate",
        "encoder_learning_rate",
        "tau_cls",
    ):
        if float(args[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(args.encoder_learning_rate) != 1.0e-5:
        raise ValueError("FP5 matching requires encoder_learning_rate=1e-5")
    if float(args.visual_layernorm_learning_rate) != 1.0e-6:
        raise ValueError("FP-LN matching requires visual_layernorm_learning_rate=1e-6")
    if int(args.soft_prompt_length) <= 0:
        raise ValueError("soft_prompt_length must be positive")
    if int(args.pseudo_val_num_classes) <= 0 or int(args.diagnostic_num_seen) < 2:
        raise ValueError(
            "diagnostic and pseudo-validation class counts must be positive"
        )
    if (
        int(args.eval_batch_size) <= 0
        or int(args.batch_size) <= 0
        or int(args.num_workers) < 0
    ):
        raise ValueError("invalid loader settings")
    if int(args.query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    if int(args.pseudo_val_seed) != 3407 or int(args.seed) != 42:
        raise ValueError(
            "v2 primary matching requires training seed 42 and pseudo seed 3407"
        )
    if str(args.train_class_scope) != "pseudo_train":
        raise ValueError("selection requires pseudo-train classes")
    if str(args.classification_location) not in {"none", "query", "z0"}:
        raise ValueError("classification_location must be none, query, or z0")
    if run_kind == "primary" and not bool(args.allow_short_run):
        steps = tuple(int(value) for value in args.probe_steps)
        resume = args.resume_checkpoint_path is not None
        if role == "frozen_prompt_v2_FP0":
            if int(args.max_steps) != 0 or steps != (0,) or resume:
                raise ValueError("FP0 requires one step-0 evaluation and no resume")
        elif role == "frozen_prompt_v2_FP5":
            if int(args.max_steps) != 73 or steps != (0, 15, 44, 73) or resume:
                raise ValueError(
                    "FP5 requires real checkpoints at steps 0, 15, 44, and 73"
                )
        elif resume:
            if int(args.max_steps) != 5400 or steps != (1000, 1800, 5400):
                raise ValueError(
                    "prompt continuation requires probes 1000, 1800, and 5400"
                )
        elif int(args.max_steps) != 500 or steps != (0, 15, 44, 73, 100, 250, 500):
            raise ValueError("prompt primary stage requires probes through step 500")
    elif run_kind == "smoke":
        if int(args.max_steps) < 1 or int(args.max_steps) > 15:
            raise ValueError("smoke runs must use between 1 and 15 updates")


def _load_split(
    data: Any, args: DictConfig
) -> tuple[Any, dict[int, str], dict[str, Any], dict[str, Any]]:
    names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches,
        photos,
        names,
        num_validation_classes=int(args.pseudo_val_num_classes),
        seed=int(args.pseudo_val_seed),
    )
    manifests = {
        "train_sketch": data.train.sketch_manifest,
        "train_photo": data.train.photo_manifest,
        "train_class_map": data.train.class_map,
    }
    manifest_identity = split_manifest_identity(
        split,
        dataset_name=data.name,
        dataset_root=data.root,
        manifest_paths=manifests,
    )
    if set(split.train_class_ids) & set(split.validation_class_ids):
        raise AssertionError("pseudo-train and pseudo-unseen classes overlap")
    split_identity = {
        "dataset": data.name,
        "seed": split.seed,
        "train_class_ids": list(split.train_class_ids),
        "validation_class_ids": list(split.validation_class_ids),
        "train_sketches": len(split.train_sketch_entries),
        "train_photos": len(split.train_photo_entries),
        "validation_sketches": len(split.validation_sketch_entries),
        "validation_photos": len(split.validation_photo_entries),
    }
    split_identity["sha256"] = canonical_sha256(split_identity)
    return split, names, split_identity, manifest_identity


def _loader(
    entries: Any,
    transform: Any,
    args: DictConfig,
    *,
    train: bool = False,
    seed: int = 42,
) -> DataLoader:
    if train:
        dataset = MultiPositiveRetrievalTrainDataset(
            entries[0], entries[1], transform, transform, num_positive_photos=1
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            drop_last=bool(args.drop_last),
            generator=generator,
            worker_init_fn=_worker_seed,
            persistent_workers=int(args.num_workers) > 0,
        )
    return DataLoader(
        RetrievalEvalDataset(entries, transform),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
    )


def _fixed_diagnostic_entries(entries: Any, count: int) -> tuple[Any, ...]:
    ordered = sorted(entries, key=lambda entry: (int(entry.label), str(entry.path)))
    return tuple(ordered[: min(count, len(ordered))])


def _entry_identity(entries: Any) -> dict[str, Any]:
    rows = [[str(entry.path), int(entry.label)] for entry in entries]
    return {
        "count": len(rows),
        "sha256": canonical_sha256(rows),
        "paths_and_labels": rows,
    }


class _FrozenEncoderAdapter:
    def __init__(self, encoder: FrozenClipEncoder) -> None:
        self.encoder = encoder
        self.device = encoder.device

    def eval(self) -> None:
        self.encoder.eval()

    def __call__(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def encode_photo(self, images: Tensor) -> Tensor:
        return self.encoder(images)


class _EarlyAdaptModel(torch.nn.Module):
    """The matched partial depth-4 sketch encoder with frozen W_CLIP."""

    def __init__(self, bundle: Any) -> None:
        super().__init__()
        self.encoder = bundle.encoder
        self.embedding_dim = bundle.encoder.hidden_dim
        self.projection = bundle.projection

    @property
    def device(self) -> torch.device:
        return self.projection.matrix.device

    def forward(self, images: Tensor) -> Tensor:
        return F.normalize(self.projection(self.encoder(images)), dim=-1)

    def encode_photo(self, images: Tensor) -> Tensor:
        return self.forward(images)

    def eval(self) -> "_EarlyAdaptModel":
        super().eval()
        return self


def _prompt_model(
    args: DictConfig, device: torch.device, photo_clip: Any
) -> tuple[torch.nn.Module, Any]:
    role = str(args.experiment_role)
    if role == "frozen_prompt_v2_FP5":
        bundle = load_trainable_sketch_hidden_encoder(
            model_name=str(args.model_name),
            pretrained=args.pretrained,
            device=device,
            mode="partial",
            unfreeze_depth=4,
            train_ln_post=False,
        )
        return _EarlyAdaptModel(bundle).to(device), bundle.transform
    model = FrozenPromptModel(
        photo_clip.encoder.model.visual,
        prompt_length=int(args.visual_prompt_length),
        train_visual_layernorm=bool(args.train_visual_layernorm),
        train_sketch_prompt=bool(args.train_sketch_prompt),
        train_photo_prompt=bool(args.train_photo_prompt),
    ).to(device)
    return model, photo_clip.transform


def _state_hash(model: torch.nn.Module) -> str:
    return hash_state({name: value for name, value in model.state_dict().items()})


def _clip_snapshot(model: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("visual.") or name.startswith("encoder.visual.")
    }


def _clip_changed(model: torch.nn.Module, before: dict[str, Tensor]) -> set[str]:
    changed: set[str] = set()
    current = dict(model.named_parameters())
    for name, value in before.items():
        if name not in current or not torch.equal(value, current[name].detach().cpu()):
            changed.add(name)
    return changed


def _approved_clip_names(model: torch.nn.Module, role: str) -> set[str]:
    if role == "frozen_prompt_v2_FP_LN" and isinstance(model, FrozenPromptModel):
        return set(model.visual_layernorm_parameter_names)
    if role == "frozen_prompt_v2_FP5":
        return {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and (name.startswith("encoder.visual."))
        }
    return set()


def _parameter_names(
    model: torch.nn.Module, text_bank: SoftPromptTextBank | None
) -> dict[str, Tensor]:
    values = dict(model.named_parameters())
    if text_bank is not None:
        values.update(
            {
                f"soft_prompt.{name}": parameter
                for name, parameter in text_bank.named_parameters()
            }
        )
    return values


def build_optimizer_parameter_groups(
    model: torch.nn.Module,
    text_bank: SoftPromptTextBank | None,
    args: DictConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build disjoint named groups and return them with an auditable mapping."""
    role = str(args.experiment_role)
    named = _parameter_names(model, text_bank)
    groups: list[tuple[str, list[str], float, float]] = []
    if role == "frozen_prompt_v2_FP5":
        groups.append(
            (
                "early_adapt_encoder",
                [
                    name
                    for name, parameter in named.items()
                    if parameter.requires_grad and not name.startswith("soft_prompt.")
                ],
                float(args.encoder_learning_rate),
                float(args.encoder_weight_decay),
            )
        )
    else:
        groups.append(
            (
                "visual_prompts",
                [
                    name
                    for name, parameter in named.items()
                    if parameter.requires_grad
                    and name in {"sketch_prompt", "photo_prompt"}
                ],
                float(args.visual_prompt_learning_rate),
                float(args.visual_prompt_weight_decay),
            )
        )
        groups.append(
            (
                "visual_layernorm",
                [
                    name
                    for name, parameter in named.items()
                    if parameter.requires_grad
                    and isinstance(model, FrozenPromptModel)
                    and name in set(model.visual_layernorm_parameter_names)
                ],
                float(args.visual_layernorm_learning_rate),
                float(args.visual_layernorm_weight_decay),
            )
        )
    groups.append(
        (
            "soft_text_prompt",
            [
                name
                for name, parameter in named.items()
                if parameter.requires_grad and name.startswith("soft_prompt.")
            ],
            float(args.soft_prompt_learning_rate),
            float(args.soft_prompt_weight_decay),
        )
    )

    mapping: list[dict[str, Any]] = []
    optimizer_groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    trainable = {
        id(parameter): name
        for name, parameter in named.items()
        if parameter.requires_grad
    }
    for group_name, names, learning_rate, weight_decay in groups:
        parameters = [named[name] for name in names]
        for name, parameter in zip(names, parameters, strict=True):
            if not parameter.requires_grad:
                raise AssertionError(
                    f"frozen parameter assigned to optimizer group: {name}"
                )
            if id(parameter) in seen:
                raise AssertionError(f"optimizer parameter appears twice: {name}")
            seen.add(id(parameter))
        mapping.append(
            {
                "name": group_name,
                "parameter_names": names,
                "parameter_count": sum(named[name].numel() for name in names),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "active": bool(names),
            }
        )
        if names:
            optimizer_groups.append(
                {
                    "name": group_name,
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
    if seen != set(trainable):
        missing = [
            name
            for name, parameter in named.items()
            if parameter.requires_grad and id(parameter) not in seen
        ]
        raise AssertionError(
            f"trainable parameters missing from optimizer groups: {missing}"
        )
    if len(seen) != sum(1 for parameter in named.values() if parameter.requires_grad):
        raise AssertionError("duplicate optimizer parameter identities")
    return optimizer_groups, mapping


def build_optimizer(
    model: torch.nn.Module,
    text_bank: SoftPromptTextBank | None,
    args: DictConfig,
) -> tuple[torch.optim.Optimizer | None, list[dict[str, Any]]]:
    groups, mapping = build_optimizer_parameter_groups(model, text_bank, args)
    if not groups:
        return None, mapping
    return torch.optim.AdamW(groups), mapping


def _optimizer_mapping(
    optimizer: torch.optim.Optimizer | None, mapping: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if optimizer is None:
        return mapping
    active = [group for group in mapping if group["active"]]
    if len(optimizer.param_groups) != len(active):
        raise AssertionError("optimizer and optimizer-group mapping disagree")
    current_by_name = {
        saved["name"]: current
        for current, saved in zip(optimizer.param_groups, active, strict=True)
    }
    return [
        {
            **saved,
            "lr": float(current_by_name[saved["name"]]["lr"])
            if saved["active"]
            else saved["lr"],
            "weight_decay": float(current_by_name[saved["name"]]["weight_decay"])
            if saved["active"]
            else saved["weight_decay"],
        }
        for saved in mapping
    ]


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    text_bank: SoftPromptTextBank | None,
    optimizer: torch.optim.Optimizer | None,
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
    model_trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_names = model_trainable_names + (
        []
        if text_bank is None
        else [
            f"soft_prompt.{name}"
            for name, parameter in text_bank.named_parameters()
            if parameter.requires_grad
        ]
    )
    model_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in set(model_trainable_names)
    }
    text_state = (
        None
        if text_bank is None
        else {
            name: value.detach().cpu() for name, value in text_bank.state_dict().items()
        }
    )
    payload = {
        "format_version": 2,
        "model_type": "frozen_prompt_v2",
        "experiment_role": str(args.experiment_role),
        "campaign": str(args.experiment_campaign),
        "run_kind": str(args.run_kind),
        "step": step,
        "training_global_step": step,
        "comparison_horizon": {"kind": "training_global_step", "value": step},
        "parameters_updated_since_selection": None,
        "model_state_dict": model_state,
        "model_state_complete": False,
        "soft_prompt_state_dict": text_state,
        "optimizer_state_dict": None
        if optimizer is None or not bool(args.save_optimizer)
        else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "optimizer_groups": optimizer_groups,
        "rng_state": capture_rng_state(loader_generator),
        "resolved_config": OmegaConf.to_container(args, resolve=True),
        "resolved_treatment": treatment_from_config(
            OmegaConf.to_container(args, resolve=True)
        ),
        "data_split_identity": split_identity,
        "data_manifest_identity": manifest_identity,
        "manifest_entry_identity": entry_identity,
        "model_state_hash": _state_hash(model),
        "initial_model_state_hash": initial_hash,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        ],
        "clip_freeze_policy": clip_freeze_policy,
        "provenance": provenance,
    }
    torch.save(payload, path)


def _restore_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    text_bank: SoftPromptTextBank | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    args: DictConfig,
    split_identity: dict[str, Any],
    manifest_identity: dict[str, Any],
    loader_generator: torch.Generator,
    optimizer_groups: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("invalid frozen-prompt v2 checkpoint")
    for key, expected in (
        ("model_type", "frozen_prompt_v2"),
        ("experiment_role", str(args.experiment_role)),
        ("campaign", str(args.experiment_campaign)),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {key} does not match this run")
    if (
        payload.get("data_split_identity") != split_identity
        or payload.get("data_manifest_identity") != manifest_identity
    ):
        raise ValueError("checkpoint data identity does not match")
    if payload.get("resolved_treatment") != treatment_from_config(
        OmegaConf.to_container(args, resolve=True)
    ):
        raise ValueError("checkpoint treatment does not match this run")
    if payload.get("optimizer_groups") != optimizer_groups:
        raise ValueError("checkpoint optimizer-group mapping does not match")
    model.load_state_dict(payload.get("model_state_dict", {}), strict=False)
    if text_bank is not None:
        state = payload.get("soft_prompt_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing soft-prompt state")
        text_bank.load_state_dict(state, strict=True)
    if optimizer is not None:
        state = payload.get("optimizer_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing optimizer state")
        optimizer.load_state_dict(state)
    if scheduler is not None:
        state = payload.get("scheduler_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing scheduler state")
        scheduler.load_state_dict(state)
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        raise ValueError("checkpoint is missing RNG state")
    restore_rng_state(rng_state, loader_generator)
    return int(payload.get("training_global_step", payload.get("step", -1))), payload


def _metrics(evaluation: Any) -> dict[str, Any]:
    return {
        "full_mAP": evaluation.metrics.mean_average_precision,
        "P@200": evaluation.metrics.precision_at_k.get(200),
        "mAP@200": evaluation.metrics.mean_average_precision_at_k.get(200),
        "average_precision_per_query": evaluation.average_precision_per_query.tolist(),
        "num_queries": evaluation.metrics.num_queries,
        "num_gallery_items": evaluation.metrics.num_gallery_items,
    }


def _text_bank_values(
    text_bank: EncodedTextBank | SoftPromptTextBank, device: torch.device
) -> tuple[Tensor, Tensor]:
    if isinstance(text_bank, SoftPromptTextBank):
        return text_bank(), text_bank.class_labels.to(device)
    return text_bank.embeddings.to(device), text_bank.labels.to(device)


def _diagnostic_classification(
    query_model: Any,
    loader: DataLoader,
    text_bank: EncodedTextBank | SoftPromptTextBank | None,
    *,
    tau: float,
    device: torch.device,
) -> dict[str, float] | None:
    if text_bank is None:
        return None
    query_model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].long().to(device)
            queries = query_model(images)
            bank, bank_labels = _text_bank_values(text_bank, device)
            loss, logits = jepa_text_classification_loss(
                queries, bank, bank_labels, labels, temperature=tau, detach_text=True
            )
            total_loss += float(loss.item()) * labels.shape[0]
            predicted = bank_labels[logits.argmax(dim=-1)]
            total_correct += int(predicted.eq(labels).sum().item())
            total += labels.shape[0]
    if total == 0:
        raise ValueError("diagnostic classification subset is empty")
    return {
        "diagnostic_seen_classification_accuracy": total_correct / total,
        "diagnostic_seen_classification_loss": total_loss / total,
        "diagnostic_seen_classification_count": total,
    }


def _check_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _gradient_norms(
    model: torch.nn.Module,
    text_bank: SoftPromptTextBank | None,
    mapping: list[dict[str, Any]],
) -> dict[str, float]:
    named = _parameter_names(model, text_bank)
    result: dict[str, float] = {}
    for group in mapping:
        values = [
            named[name].grad.detach().norm().item()
            for name in group["parameter_names"]
            if named[name].grad is not None
        ]
        result[group["name"]] = (
            float(math.sqrt(sum(value * value for value in values))) if values else 0.0
        )
    return result


def _parameter_gradient_norms(
    model: torch.nn.Module, text_bank: SoftPromptTextBank | None
) -> dict[str, float | None]:
    named = _parameter_names(model, text_bank)
    layernorm_names = (
        set(model.visual_layernorm_parameter_names)
        if isinstance(model, FrozenPromptModel)
        else set()
    )
    relevant = {
        name
        for name, parameter in named.items()
        if parameter.requires_grad
        or name in {"sketch_prompt", "photo_prompt"}
        or name in layernorm_names
    }
    return {
        name: None
        if named[name].grad is None
        else float(named[name].grad.detach().norm().item())
        for name in sorted(relevant)
    }


def _parameter_norms(
    model: torch.nn.Module, text_bank: SoftPromptTextBank | None
) -> dict[str, float]:
    named = _parameter_names(model, text_bank)
    result: dict[str, float] = {}
    for prefix, names in (
        ("visual_prompts", ("sketch_prompt", "photo_prompt")),
        (
            "soft_text_prompt",
            tuple(name for name in named if name.startswith("soft_prompt.")),
        ),
    ):
        values = [named[name].detach().flatten() for name in names if name in named]
        result[prefix] = float(torch.cat(values).norm().item()) if values else 0.0
    return result


def _assert_optimizer_gradients(
    model: torch.nn.Module,
    text_bank: SoftPromptTextBank | None,
    mapping: list[dict[str, Any]],
) -> None:
    named = _parameter_names(model, text_bank)
    for name, parameter in named.items():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError(f"trainable parameter has no gradient: {name}")
            _check_finite(f"gradient {name}", parameter.grad)
        elif parameter.grad is not None:
            raise RuntimeError(f"frozen parameter received a gradient: {name}")
    norms = _gradient_norms(model, text_bank, mapping)
    for group in mapping:
        if group["active"] and norms[group["name"]] <= 0.0:
            raise RuntimeError(
                f"active optimizer group received no gradient: {group['name']}"
            )


def _assert_clip_policy(
    model: torch.nn.Module, before: dict[str, Tensor], role: str
) -> dict[str, Any]:
    changed = _clip_changed(model, before)
    allowed = _approved_clip_names(model, role)
    forbidden = sorted(changed - allowed)
    if forbidden:
        raise RuntimeError(f"unexpected CLIP-owned parameter mutation: {forbidden[:5]}")
    if role == "frozen_prompt_v2_FP4" and changed:
        raise RuntimeError(
            f"FP4 mutated frozen visual parameters: {sorted(changed)[:5]}"
        )
    return {
        "clip_owned_parameter_names": sorted(before),
        "approved_trainable_clip_parameter_names": sorted(allowed),
        "changed_clip_parameter_names": sorted(changed),
        "all_clip_owned_parameters_byte_identical": not changed,
        "fully_frozen": not allowed,
    }


def _assert_close(left: float, right: float, tolerance: float, name: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"{name} changed by more than tolerance: {left} vs {right}"
        )


def run(args: DictConfig) -> None:
    _validate(args)
    seed = int(args.seed)
    _seed(seed)
    device = _device(str(args.device))
    data = load_data_config(_path(args.data_config))
    split, names, split_identity, data_manifest_identity = _load_split(data, args)
    manifest_path = _path(args.experiment_manifest_path)
    manifest, manifest_sha256 = ensure_manifest(
        manifest_path,
        dataset=str(data.name),
        data_config=str(args.data_config),
        campaign=str(args.experiment_campaign),
    )
    entry_identity = manifest_entry_identity(
        manifest_path,
        manifest,
        role=str(args.experiment_role),
        manifest_sha256=manifest_sha256,
    )
    train_names = {class_id: names[class_id] for class_id in split.train_class_ids}
    photo_clip = load_frozen_clip(
        model_name=str(args.model_name), pretrained=args.pretrained, device=device
    )
    prompt_model, transform = _prompt_model(args, device, photo_clip)
    role = str(args.experiment_role)
    fixed_photo = role in {
        "frozen_prompt_v2_FP0",
        "frozen_prompt_v2_FP1S",
        "frozen_prompt_v2_FP4",
        "frozen_prompt_v2_FP5",
    }
    photo_model: Any = (
        _FrozenEncoderAdapter(photo_clip.encoder) if fixed_photo else prompt_model
    )
    query_model: Any = (
        _FrozenEncoderAdapter(photo_clip.encoder)
        if role in {"frozen_prompt_v2_FP0", "frozen_prompt_v2_FP4"}
        else prompt_model
    )

    train_loader = _loader(
        (split.train_sketch_entries, split.train_photo_entries),
        transform,
        args,
        train=True,
        seed=seed,
    )
    diagnostic_entries = _fixed_diagnostic_entries(
        split.train_sketch_entries, int(args.diagnostic_num_seen)
    )
    diagnostic_loader = _loader(diagnostic_entries, transform, args)
    val_sketch_loader = _loader(split.validation_sketch_entries, transform, args)
    val_photo_loader = _loader(split.validation_photo_entries, transform, args)
    if len(train_loader) == 0:
        raise ValueError("training loader has no batches")
    attention_images = None
    if isinstance(prompt_model, FrozenPromptModel) and prompt_model.prompt_length:
        attention_images = next(iter(val_sketch_loader))["image"][
            : min(8, int(args.eval_batch_size))
        ]

    vanilla_model = _FrozenEncoderAdapter(photo_clip.encoder)
    vanilla_sketch = encode_prompted_loader(vanilla_model, val_sketch_loader)
    vanilla_photo = encode_prompted_loader(vanilla_model, val_photo_loader, photo=True)
    vanilla_evaluation = evaluate_prompted(
        vanilla_sketch,
        vanilla_photo,
        query_chunk_size=int(args.query_chunk_size),
        device=device,
    )

    text_bank: EncodedTextBank | SoftPromptTextBank | None = None
    if str(args.text_mode) == "soft":
        text_bank = SoftPromptTextBank(
            photo_clip.encoder,
            photo_clip.tokenizer,
            train_names,
            prompt_length=int(args.soft_prompt_length),
        ).to(device)
    elif str(args.text_mode) == "hard":
        text_bank = encode_class_text_bank(
            photo_clip.encoder,
            photo_clip.tokenizer,
            train_names,
            prompt_template=str(args.prompt_template),
        )

    optimizer, optimizer_groups = build_optimizer(
        prompt_model,
        text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
        args,
    )
    scheduler = (
        None
        if optimizer is None
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    )
    initial_hash = _state_hash(prompt_model)
    clip_before = _clip_snapshot(prompt_model)
    photo_before = _clip_snapshot(photo_clip.encoder.model)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_provenance(
        PROJECT_ROOT,
        resolved_config=OmegaConf.to_container(args, resolve=True),
        command=[sys.executable, *sys.argv],
    )
    loader_generator = train_loader.generator
    if loader_generator is None:
        raise RuntimeError("training loader has no reproducible generator")

    history: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    resume_records: list[dict[str, Any]] = []
    step = 0
    parent_payload: dict[str, Any] | None = None
    if args.resume_checkpoint_path is not None:
        step, parent_payload = _restore_checkpoint(
            _path(args.resume_checkpoint_path),
            model=prompt_model,
            text_bank=text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            split_identity=split_identity,
            manifest_identity=data_manifest_identity,
            loader_generator=loader_generator,
            optimizer_groups=optimizer_groups,
        )
        initial_hash = str(parent_payload["initial_model_state_hash"])
        parent_hash = hashlib.sha256(
            _path(args.resume_checkpoint_path).read_bytes()
        ).hexdigest()
        resume_records.append(
            {
                "checkpoint": str(_path(args.resume_checkpoint_path)),
                "checkpoint_sha256": parent_hash,
                "source_training_global_step": step,
                "optimizer_state_restored": optimizer is not None,
            }
        )
        prior_path = output_dir / "run_result.json"
        if prior_path.is_file():
            prior = json.loads(prior_path.read_text())
            if prior.get("experiment_role") != role or prior.get("campaign") != str(
                args.experiment_campaign
            ):
                raise ValueError("existing output directory belongs to another run")
            history = list(prior.get("history", []))
            training_history = list(prior.get("training_history", []))
            resume_records = list(prior.get("resume", [])) + resume_records

    last_train = {"rank": None, "classification": None, "accuracy": None}
    last_gradient_norms = {group["name"]: 0.0 for group in optimizer_groups}
    last_parameter_gradient_norms: dict[str, float | None] = {}
    start_step = step
    existing_steps = {
        int(row["training_global_step"])
        for row in history
        if "training_global_step" in row
    }
    probe_steps = {int(value) for value in args.probe_steps}
    invariance_tolerance = float(args.visual_invariance_tolerance)

    def probe(probe_step: int) -> None:
        nonlocal existing_steps
        if probe_step in existing_steps:
            return
        checkpoint = output_dir / "checkpoints" / f"frozen_prompt_step{probe_step}.pt"
        clip_policy_current = _assert_clip_policy(prompt_model, clip_before, role)
        photo_changed = _clip_changed(photo_clip.encoder.model, photo_before)
        photo_allowed = (
            set(prompt_model.visual_layernorm_parameter_names)
            if role == "frozen_prompt_v2_FP_LN"
            else set()
        )
        if photo_changed - photo_allowed:
            raise RuntimeError(
                f"unexpected photo CLIP mutation: {sorted(photo_changed - photo_allowed)[:5]}"
            )
        clip_policy_current.update(
            {
                "photo_encoder_frozen": role != "frozen_prompt_v2_FP_LN",
                "visual_projection_frozen": True,
                "text_tower_frozen": True,
            }
        )
        _save_checkpoint(
            checkpoint,
            model=prompt_model,
            text_bank=text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
            optimizer=optimizer,
            scheduler=scheduler,
            step=probe_step,
            args=args,
            split_identity=split_identity,
            manifest_identity=data_manifest_identity,
            entry_identity=entry_identity,
            loader_generator=loader_generator,
            provenance=provenance,
            optimizer_groups=_optimizer_mapping(optimizer, optimizer_groups),
            initial_hash=initial_hash,
            clip_freeze_policy=clip_policy_current,
        )
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        current_sketch = encode_prompted_loader(query_model, val_sketch_loader)
        current_photo = encode_prompted_loader(
            photo_model, val_photo_loader, photo=True
        )
        identity = cache_identity(
            prompt_checkpoint_hash=checkpoint_hash,
            prompt_length=int(args.visual_prompt_length) if not fixed_photo else 0,
            prompt_mode=str(args.prompt_mode),
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
            model=prompt_model if isinstance(prompt_model, FrozenPromptModel) else None,
            max_samples=512,
        )
        classification = _diagnostic_classification(
            query_model,
            diagnostic_loader,
            text_bank,
            tau=float(args.tau_cls),
            device=device,
        )
        val_metrics = _metrics(evaluation)
        if role == "frozen_prompt_v2_FP4":
            _assert_close(
                val_metrics["full_mAP"],
                vanilla_evaluation.metrics.mean_average_precision,
                invariance_tolerance,
                "FP4 retrieval mAP",
            )
            _assert_close(
                val_metrics["P@200"],
                vanilla_evaluation.metrics.precision_at_k[200],
                invariance_tolerance,
                "FP4 P@200",
            )
            _assert_close(
                val_metrics["mAP@200"],
                vanilla_evaluation.metrics.mean_average_precision_at_k[200],
                invariance_tolerance,
                "FP4 mAP@200",
            )
            if (
                max(
                    abs(a - b)
                    for a, b in zip(
                        evaluation.average_precision_per_query.tolist(),
                        vanilla_evaluation.average_precision_per_query.tolist(),
                        strict=True,
                    )
                )
                > invariance_tolerance
            ):
                raise AssertionError("FP4 per-query retrieval AP changed")
            if (
                geometry["reference_preservation"]["sketch"]
                < 1.0 - invariance_tolerance
                or geometry["reference_preservation"]["photo"]
                < 1.0 - invariance_tolerance
            ):
                raise AssertionError("FP4 visual embeddings are not invariant")
        sketch_delta = float(
            (current_sketch.embeddings - vanilla_sketch.embeddings).abs().max().item()
        )
        photo_delta = float(
            (loaded_photo.embeddings - vanilla_photo.embeddings).abs().max().item()
        )
        visual_delta = max(sketch_delta, photo_delta)
        if role == "frozen_prompt_v2_FP1S" and photo_delta > invariance_tolerance:
            raise AssertionError(f"FP1S photo branch changed by {photo_delta}")
        if (
            role in {"frozen_prompt_v2_FP4", "frozen_prompt_v2_FP0"}
            and visual_delta > invariance_tolerance
        ):
            raise AssertionError(f"fixed visual branch changed by {visual_delta}")
        prompt_attention = None
        if attention_images is not None:
            prompt_attention = prompt_model.attention_diagnostics_by_block(
                attention_images.to(device),
                prompt="sketch",
            )
        parameter_norms = _parameter_norms(
            prompt_model,
            text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
        )
        row: dict[str, Any] = {
            "step": probe_step,
            "training_global_step": probe_step,
            "comparison_horizon": {"kind": "training_global_step", "value": probe_step},
            "parameters_updated_since_selection": None,
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_sha256": checkpoint_hash,
            "val": val_metrics,
            "full_pseudo_unseen_mAP": val_metrics["full_mAP"],
            "P@200": val_metrics["P@200"],
            "mAP@200": val_metrics["mAP@200"],
            "last_train_batch_rank_loss": last_train["rank"],
            "last_train_batch_classification_loss": last_train["classification"],
            "last_train_batch_accuracy": last_train["accuracy"],
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
            "prompt_parameter_norm": parameter_norms["visual_prompts"],
            "soft_prompt_parameter_norm": parameter_norms["soft_text_prompt"],
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
            "visual_embedding_max_abs_delta": visual_delta,
            "prompt_attention": prompt_attention,
            "clip_freeze_policy": clip_policy_current,
            "optimizer_groups": _optimizer_mapping(optimizer, optimizer_groups),
            "trainable_parameter_names": [
                name
                for name, parameter in _parameter_names(
                    prompt_model,
                    text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
                ).items()
                if parameter.requires_grad
            ],
            "pseudo_split_identity": split_identity,
            "manifest_identity": data_manifest_identity,
            "manifest_entry_identity": entry_identity,
            "official_unseen_used_for_selection": False,
            "protocol": {
                "official_unseen_used_for_selection": False,
                "text_used_for_inference": False,
                "photo_used_for_inference": False,
                "gallery_cache_identity": identity,
            },
        }
        history.append(row)
        history.sort(key=lambda value: int(value["training_global_step"]))
        existing_steps.add(probe_step)
        (output_dir / f"probe_step{probe_step}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

    if role == "frozen_prompt_v2_FP0":
        probe(0)
    elif args.resume_checkpoint_path is None:
        probe(0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    if optimizer is not None:
        while step < int(args.max_steps):
            for batch in train_loader:
                if step >= int(args.max_steps):
                    break
                images = batch["sketch"].to(device, non_blocking=device.type == "cuda")
                positives = batch["positive_photos"][:, 0].to(
                    device, non_blocking=device.type == "cuda"
                )
                negatives = batch["negative_photo"].to(
                    device, non_blocking=device.type == "cuda"
                )
                labels = batch["label"].long().to(device)
                if hasattr(query_model, "train"):
                    query_model.train()
                optimizer.zero_grad(set_to_none=True)
                query = query_model(images)
                with torch.set_grad_enabled(not fixed_photo):
                    photo_values = photo_model.encode_photo(
                        torch.cat((positives, negatives), dim=0)
                    )
                positive, negative = photo_values.split(
                    (images.shape[0], images.shape[0]), dim=0
                )
                if float(args.lambda_rank) > 0:
                    query_normalized = F.normalize(query, dim=-1)
                    positive_normalized = F.normalize(positive, dim=-1)
                    negative_normalized = F.normalize(negative, dim=-1)
                    rank = F.softplus(
                        float(args.margin)
                        - (query_normalized * positive_normalized).sum(-1)
                        + (query_normalized * negative_normalized).sum(-1)
                    ).mean()
                else:
                    rank = query.new_zeros(())
                cls = query.new_zeros(())
                accuracy = query.new_zeros(())
                if text_bank is not None:
                    bank, bank_labels = _text_bank_values(text_bank, device)
                    cls, logits = jepa_text_classification_loss(
                        query,
                        bank,
                        bank_labels,
                        labels,
                        temperature=float(args.tau_cls),
                        detach_text=not isinstance(text_bank, SoftPromptTextBank),
                    )
                    accuracy = classification_accuracy(logits, bank_labels, labels)
                total = float(args.lambda_rank) * rank + float(args.lambda_cls) * cls
                _check_finite("rank loss", rank)
                _check_finite("classification loss", cls)
                _check_finite("total loss", total)
                total.backward()
                _assert_optimizer_gradients(
                    prompt_model,
                    text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
                    optimizer_groups,
                )
                last_gradient_norms = _gradient_norms(
                    prompt_model,
                    text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
                    optimizer_groups,
                )
                last_parameter_gradient_norms = _parameter_gradient_norms(
                    prompt_model,
                    text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
                )
                optimizer.step()
                scheduler.step() if scheduler is not None else None
                for name, parameter in _parameter_names(
                    prompt_model,
                    text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
                ).items():
                    _check_finite(f"parameter {name}", parameter)
                step += 1
                last_train = {
                    "rank": float(rank.item()),
                    "classification": float(cls.item()),
                    "accuracy": float(accuracy.item()),
                }
                training_history.append(
                    {
                        "training_global_step": step,
                        "last_train_batch_rank_loss": last_train["rank"],
                        "last_train_batch_classification_loss": last_train[
                            "classification"
                        ],
                        "last_train_batch_accuracy": last_train["accuracy"],
                        "gradient_norms": dict(last_gradient_norms),
                        "gradient_norms_by_parameter": dict(
                            last_parameter_gradient_norms
                        ),
                    }
                )
                if step in probe_steps:
                    probe(step)
        if role != "frozen_prompt_v2_FP0" and step != int(args.max_steps):
            raise RuntimeError(f"training stopped at {step}, expected {args.max_steps}")

    if isinstance(text_bank, SoftPromptTextBank):
        torch.save(
            {
                "format_version": 2,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in text_bank.state_dict().items()
                },
                "prompt_length": text_bank.prompt_length,
                "class_names_used_for_training": list(text_bank.class_names),
                "optimizer_group": "soft_text_prompt",
            },
            output_dir / "soft_prompt.pt",
        )

    if role == "frozen_prompt_v2_FP5" and str(args.run_kind) == "primary":
        candidates = [
            row for row in history if int(row["training_global_step"]) in {44, 73}
        ]
        if len(candidates) != 2:
            raise RuntimeError(
                "FP5 selection requires both real step-44 and step-73 checkpoints"
            )
        selected_row = max(
            candidates,
            key=lambda row: (
                float(row["full_pseudo_unseen_mAP"]),
                -int(row["training_global_step"]),
            ),
        )
        selected = {
            "selection_metric": "full_pseudo_unseen_mAP",
            "training_global_step": int(selected_row["training_global_step"]),
            "checkpoint": selected_row["checkpoint"],
            "checkpoint_sha256": selected_row["checkpoint_sha256"],
            "full_pseudo_unseen_mAP": selected_row["full_pseudo_unseen_mAP"],
        }
        frozen_hold = []
        for horizon in (500, 1800, 5400):
            frozen_hold.append(
                {
                    "kind": "frozen_hold_evaluation",
                    "comparison_horizon": horizon,
                    "training_global_step": selected["training_global_step"],
                    "parameters_updated_since_selection": 0,
                    "source_probe_step": selected["training_global_step"],
                    "checkpoint": selected["checkpoint"],
                    "checkpoint_sha256": selected["checkpoint_sha256"],
                    "val": selected_row["val"],
                    "geometry": selected_row["geometry"],
                }
            )
    else:
        selected_row = (
            max(
                history,
                key=lambda row: (
                    float(row["full_pseudo_unseen_mAP"]),
                    -int(row["training_global_step"]),
                ),
            )
            if history
            else None
        )
        selected = (
            None
            if selected_row is None
            else {
                "selection_metric": "full_pseudo_unseen_mAP",
                "training_global_step": int(selected_row["training_global_step"]),
                "checkpoint": selected_row["checkpoint"],
                "checkpoint_sha256": selected_row["checkpoint_sha256"],
                "full_pseudo_unseen_mAP": selected_row["full_pseudo_unseen_mAP"],
            }
        )
        frozen_hold = []

    checkpoints = {
        str(row["training_global_step"]): {
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "training_global_step": row["training_global_step"],
        }
        for row in history
    }
    training_seconds = time.perf_counter() - training_started
    updates_this_run = max(0, step - start_step)
    seconds_per_update = (
        training_seconds / updates_this_run if updates_this_run else None
    )
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    final_clip_policy = _assert_clip_policy(prompt_model, clip_before, role)
    final_clip_policy.update(
        {
            "photo_encoder_frozen": role != "frozen_prompt_v2_FP_LN",
            "visual_projection_frozen": True,
            "text_tower_frozen": True,
        }
    )
    report = {
        "schema_version": 2,
        "experiment_role": role,
        "campaign": str(args.experiment_campaign),
        "run_kind": str(args.run_kind),
        "dataset": str(data.name),
        "resolved_config": OmegaConf.to_container(args, resolve=True),
        "resolved_treatment": treatment_from_config(
            OmegaConf.to_container(args, resolve=True)
        ),
        "experiment_code_commit": provenance.get("head_commit"),
        "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"),
        "working_tree_state": provenance.get("working_tree_state"),
        "provenance": provenance,
        "seed": seed,
        "pseudo_validation_seed": int(args.pseudo_val_seed),
        "pseudo_split_identity": split_identity,
        "manifest_identity": data_manifest_identity,
        "manifest_entry_identity": entry_identity,
        "manifest_path": str(manifest_path),
        "diagnostic_subset_identity": _entry_identity(diagnostic_entries),
        "diagnostic_subset_selected_before_training": True,
        "official_unseen_used_for_selection": False,
        "optimizer_groups": _optimizer_mapping(optimizer, optimizer_groups),
        "trainable_parameter_names": [
            name
            for name, parameter in _parameter_names(
                prompt_model,
                text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
            ).items()
            if parameter.requires_grad
        ],
        "frozen_parameter_names": [
            name
            for name, parameter in _parameter_names(
                prompt_model,
                text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
            ).items()
            if not parameter.requires_grad
        ],
        "clip_freeze_policy": final_clip_policy,
        "model_state": {
            "stored_in_checkpoints": True,
            "checkpoint_format_version": 2,
            "compact_trainable_state": True,
        },
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "rng_state": {
            "stored_in_checkpoints": True,
            "checkpoint_steps": sorted(checkpoints, key=int),
        },
        "checkpoints": checkpoints,
        "checkpoint": None if selected is None else selected["checkpoint"],
        "checkpoint_sha256": None
        if selected is None
        else selected["checkpoint_sha256"],
        "history": history,
        "training_history": training_history,
        "resume": resume_records,
        "selection": selected,
        "frozen_hold_evaluation": frozen_hold,
        "runtime": {
            "training_seconds": training_seconds,
            "updates_this_run": updates_this_run,
            "seconds_per_update": seconds_per_update,
            "estimated_seconds_to_step_500": None
            if seconds_per_update is None
            else seconds_per_update * max(0, 500 - start_step),
            "estimated_seconds_to_step_5400": None
            if seconds_per_update is None
            else seconds_per_update * max(0, 5400 - start_step),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
        "inference_contract": {
            "required_inputs": ["raw_sketch_image"],
            "text_required": False,
            "photo_required": False,
            "oracle_class_required": False,
        },
        "protocol": {
            "selection_metric": "full_pseudo_unseen_mAP",
            "official_unseen_used_for_selection": False,
            "transport_enabled": False,
            "direction_supervision": False,
            "distance_prediction": False,
            "num_positive_photos": 1,
        },
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "run_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


@hydra.main(
    version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="train_frozen_prompt"
)
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
