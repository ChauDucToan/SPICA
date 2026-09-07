"""Train and probe the controlled frozen-prompt v2 campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

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
from .data.pairing import load_pairing_manifest
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
    ALL_ROLES,
    CAMPAIGN,
    FINAL_CAMPAIGN,
    FINAL_ROLES,
    PAIRING_PILOT_CAMPAIGN,
    PAIRING_PILOT_ROLES,
    PAIRING_PILOT_STEPS,
    MASKED_VIEW_CAMPAIGN,
    MASKED_VIEW_ROLES,
    MASKED_VIEW_STEPS,
    MASKED_VIEW_3600_CAMPAIGN,
    MASKED_VIEW_3600_ROLES,
    MASKED_VIEW_3600_STEPS,
    MASKED_VIEW_3600_SELECTION_STEPS,
    is_masked_view_campaign,
    is_masked_view_3600_campaign,
    MASK_POLICY,
    PAIRING_MANIFEST_SHA256,
    PAIRING_MANIFEST_PATH,
    FINAL_SMOKE_CAMPAIGN,
    FINAL_SPLIT_SEEDS,
    SMOKE_CAMPAIGN,
    canonical_sha256,
    ensure_manifest,
    expected_probe_steps,
    manifest_entry_identity,
    treatment_for_role,
    treatment_from_config,
)
from .models.clip import (
    FrozenClipEncoder,
    load_frozen_clip,
    load_trainable_sketch_hidden_encoder,
)
from .models.checkpoint import load_trainable_state
from .models.frozen_prompt import FrozenPromptModel
from .models.jepa import classification_accuracy, jepa_text_classification_loss
from .provenance import capture_provenance, capture_rng_state, restore_rng_state
from .tracking.wandb import WandbExperiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")
PRIMARY_PROBE_STEPS = (0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400)
PROMPT_ROLES = {
    "frozen_prompt_v2_FP1",
    "frozen_prompt_v2_FP1S",
    "frozen_prompt_v2_FP2",
    "frozen_prompt_v2_FP3",
    "frozen_prompt_v2_FP_LN",
    "frozen_prompt_final_FP3",
    "frozen_prompt_final_FP3S",
    "frozen_prompt_final_FP2",
    "frozen_prompt_final_FP_LN",
}


def _is_final_role(role: str) -> bool:
    return role in FINAL_ROLES


def _is_fp5(role: str) -> bool:
    return role.endswith("FP5")


def _is_layernorm(role: str) -> bool:
    return role.endswith("FP_LN")


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


@contextmanager
def _preserve_global_rng() -> Any:
    state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.random.set_rng_state(state[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _finite_metric(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number

def _metric_from_view(view: Any, key: str) -> float:
    if not isinstance(view, dict) or key not in view:
        raise ValueError(f"masked-view evaluation is missing {key}")
    return _finite_metric(view[key], key)

def _select_masked_view_3600(
    history: list[dict[str, Any]], *, require_all: bool = True,
    candidate_steps: tuple[int, ...] = MASKED_VIEW_3600_SELECTION_STEPS,
) -> dict[str, Any]:
    candidates = [
        row for row in history
        if int(row.get("training_global_step", -1)) in set(candidate_steps)
    ]
    expected = set(candidate_steps)
    observed = {int(row.get("training_global_step", -1)) for row in candidates}
    if require_all and observed != expected:
        raise RuntimeError("3600 campaign requires one evaluation at every nonzero probe")
    if not candidates:
        raise RuntimeError("3600 campaign has no trained probe available for selection")
    def best(metric: tuple[str, str]) -> dict[str, Any]:
        winner = None
        winner_score = -math.inf
        for row in candidates:
            score = _metric_from_view(row["masked_view"][metric[0]], metric[1])
            if score > winner_score:  # strict greater preserves earliest ties
                winner, winner_score = row, score
        assert winner is not None
        return {
            "training_global_step": int(winner["training_global_step"]),
            "checkpoint": winner["checkpoint"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "score": winner_score,
            "criterion": metric[0] + "." + metric[1],
            "secondary_P@200": _metric_from_view(winner["masked_view"][metric[0]], "P@200"),
            "scores": {
                "clean": winner["masked_view"]["clean"],
                "masked_macro": winner["masked_view"]["masked_macro"],
            },
        }
    return {
        "best_clean": best(("clean", "mAP@200_prefix_positive")),
        "best_masked": best(("masked_macro", "mAP@200_prefix_positive")),
    }

def _validate(args: DictConfig) -> None:
    role = str(args.experiment_role)
    if role not in ALL_ROLES:
        raise ValueError(f"experiment_role must be exactly one of {ALL_ROLES}")
    campaign = str(args.experiment_campaign)
    run_kind = str(args.run_kind)
    pairing_pilot = campaign == PAIRING_PILOT_CAMPAIGN
    masked_view = is_masked_view_campaign(campaign)
    masked_view_3600 = is_masked_view_3600_campaign(campaign)
    final_role = _is_final_role(role)
    if masked_view:
        if role not in (
            MASKED_VIEW_3600_ROLES if masked_view_3600 else MASKED_VIEW_ROLES
        ) or run_kind not in {"primary", "smoke"}:
            raise ValueError("masked-view campaign requires role C or M and primary/smoke")
        if str(args.positive_sampling) != "same_class":
            raise ValueError("masked-view positive_sampling must be same_class")
        if args.pairing_manifest_path is None or not str(args.pairing_manifest_path):
            raise ValueError("masked-view requires a pairing manifest path")
        if args.resume_checkpoint_path is not None:
            raise ValueError("masked-view runs train from scratch; resume is forbidden")
        if int(args.seed) != 42 or int(args.pseudo_val_seed) != 3407:
            raise ValueError("masked-view requires seed 42 and pseudo_val_seed 3407")
        if str(args.sketch_view_mode) != treatment_for_role(role)["sketch_view_mode"]:
            raise ValueError("masked-view role and sketch_view_mode do not match")
        if dict(args.mask_policy) != MASK_POLICY:
            raise ValueError("masked-view mask_policy does not match the known policy")
        if run_kind == "primary":
            if str(args.pairing_manifest_path) != PAIRING_MANIFEST_PATH:
                raise ValueError("masked-view primary requires the approved pairing manifest path")
            expected_steps = MASKED_VIEW_3600_STEPS if masked_view_3600 else MASKED_VIEW_STEPS
            expected_max = 3600 if masked_view_3600 else 1800
            if int(args.max_steps) != expected_max or tuple(int(v) for v in args.probe_steps) != expected_steps:
                raise ValueError(f"masked-view primary requires fixed probes through step {expected_max}")
            if str(args.pretrained) != "openai":
                raise ValueError("masked-view primary requires pretrained=openai")
        else:
            if int(args.batch_size) != 2:
                raise ValueError("masked-view smoke batch size must be 2")
            if str(args.device) != "cpu" or not 1 <= int(args.max_steps) <= 15:
                raise ValueError("masked-view smoke requires CPU and 1..15 steps")
            if args.pretrained not in (None, "openai"):
                raise ValueError("masked-view smoke pretrained must be null or openai")
        if str(args.text_mode) != "hard" or str(args.model_name) != "ViT-B-32-quickgelu":
            raise ValueError("masked-view requires the approved CLIP/text protocol")
        if int(args.batch_size) not in ({32} if run_kind == "primary" else {2}):
            raise ValueError("masked-view batch size must be 32 primary or 2 smoke")
    if pairing_pilot:
        if role not in PAIRING_PILOT_ROLES or run_kind not in {"primary", "smoke"}:
            raise ValueError("pairing pilot requires one of its two primary roles")
        if str(args.positive_sampling) not in {"same_class", "paired"}:
            raise ValueError("pairing pilot positive_sampling must be same_class or paired")
        if str(args.positive_sampling) != treatment_for_role(role)["positive_sampling"]:
            raise ValueError("pairing pilot role and positive_sampling do not match")
        if args.pairing_manifest_path is None or not str(args.pairing_manifest_path):
            raise ValueError("pairing pilot requires pairing_manifest_path")
        if args.resume_checkpoint_path is not None:
            raise ValueError("pairing pilot runs must train from scratch")
        if int(args.seed) != 42 or int(args.pseudo_val_seed) != 3407:
            raise ValueError("pairing pilot requires seed 42 and pseudo_val_seed 3407")
    elif run_kind in {"primary", "split_robustness"} and not masked_view and campaign != (
        FINAL_CAMPAIGN if final_role else CAMPAIGN
    ):
        raise ValueError("frozen-prompt campaign does not match the role")
    if run_kind == "split_robustness" and not final_role:
        raise ValueError("split robustness is only a final-campaign run")
    if run_kind == "smoke" and campaign not in {
        SMOKE_CAMPAIGN,
        FINAL_SMOKE_CAMPAIGN,
        PAIRING_PILOT_CAMPAIGN,
        MASKED_VIEW_CAMPAIGN,
        MASKED_VIEW_3600_CAMPAIGN,
    }:
        raise ValueError("smoke frozen-prompt runs must use a smoke campaign")
    if run_kind not in {"primary", "split_robustness", "smoke"}:
        raise ValueError("run_kind must be primary, split_robustness, or smoke")
    if not pairing_pilot and not masked_view and str(args.positive_sampling) != "same_class":
        raise ValueError("historical frozen-prompt runs require positive_sampling=same_class")
    if not pairing_pilot and not masked_view and args.pairing_manifest_path is not None:
        raise ValueError("historical frozen-prompt runs require pairing_manifest_path=null")
    if masked_view:
        steps = tuple(int(value) for value in args.probe_steps)
        if run_kind == "smoke" and (not steps or steps[0] != 0 or steps[-1] != int(args.max_steps)):
            raise ValueError("masked-view smoke probes must start at 0 and end at max_steps")
    elif pairing_pilot:
        steps = tuple(int(value) for value in args.probe_steps)
        if run_kind == "primary" and (
            int(args.max_steps) != 1800 or steps != PAIRING_PILOT_STEPS
        ):
            raise ValueError("pairing pilot primary requires fixed probes through step 1800")
        if run_kind == "smoke":
            if str(args.device) != "cpu":
                raise ValueError("pairing pilot smoke runs require device=cpu")
            if not 1 <= int(args.max_steps) <= 15:
                raise ValueError("pairing pilot smoke runs must use between 1 and 15 updates")
            if not steps or steps[0] != 0 or steps[-1] != int(args.max_steps):
                raise ValueError(
                    "pairing pilot smoke probes must start at 0 and end at max_steps"
                )

    seed = int(args.seed)
    pseudo_seed = int(args.pseudo_val_seed)
    observed = treatment_from_config(OmegaConf.to_container(args, resolve=True))
    expected = treatment_for_role(
        role, seed=seed if final_role else 42, pseudo_val_seed=pseudo_seed
    )
    smoke_overrides = {"batch_size"} if (pairing_pilot or masked_view) and run_kind == "smoke" else set()
    mismatches = {
        key: (observed[key], expected[key])
        for key in expected
        if key not in smoke_overrides
        and not _same_value(observed.get(key), expected[key])
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
    if run_kind == "split_robustness":
        if seed != 42 or pseudo_seed not in FINAL_SPLIT_SEEDS:
            raise ValueError(
                "split robustness requires training seed 42 and pseudo seed 101, 202, or 303"
            )
        expected_name = (
            f"FP5_split{pseudo_seed}"
            if _is_fp5(role)
            else f"selected_prompt_split{pseudo_seed}"
        )
        if args.get("split_run_name") != expected_name:
            raise ValueError(f"split robustness requires split_run_name={expected_name}")
    elif pseudo_seed != 3407 or (not final_role and seed != 42) or (
        final_role and seed not in {42, 123, 3407}
    ):
        raise ValueError(
            "final matching requires training seed 42, 123, or 3407 and pseudo seed 3407"
            if final_role
            else "v2 primary matching requires training seed 42 and pseudo seed 3407"
        )
    if str(args.train_class_scope) != "pseudo_train":
        raise ValueError("selection requires pseudo-train classes")
    if str(args.classification_location) not in {"none", "query", "z0"}:
        raise ValueError("classification_location must be none, query, or z0")
    if not pairing_pilot and not masked_view and run_kind in {"primary", "split_robustness"} and not bool(args.allow_short_run):
        steps = tuple(int(value) for value in args.probe_steps)
        resume = args.resume_checkpoint_path is not None
        if run_kind == "split_robustness" and resume:
            raise ValueError("split robustness runs must train from scratch")
        if role == "frozen_prompt_v2_FP0":
            if int(args.max_steps) != 0 or steps != (0,) or resume:
                raise ValueError("FP0 requires one step-0 evaluation and no resume")
        elif _is_fp5(role):
            if int(args.max_steps) != 73 or steps != (0, 15, 44, 73) or resume:
                raise ValueError(
                    "FP5 requires real checkpoints at steps 0, 15, 44, and 73"
                )
        elif resume and final_role:
            if int(args.max_steps) != 10800 or steps != (6000, 7200, 9000, 10800):
                raise ValueError(
                    "final prompt continuation requires probes 6000, 7200, 9000, and 10800"
                )
        elif resume:
            if int(args.max_steps) != 5400 or steps != (1000, 1800, 5400):
                raise ValueError(
                    "prompt continuation requires probes 1000, 1800, and 5400"
                )
        elif final_role:
            if run_kind == "split_robustness":
                if _is_fp5(role):
                    expected_steps = (0, 15, 44, 73)
                    expected_max_steps = 73
                else:
                    expected_steps = PRIMARY_PROBE_STEPS
                    expected_max_steps = 5400
                if int(args.max_steps) != expected_max_steps or steps != expected_steps:
                    raise ValueError(
                        "split robustness requires a from-scratch run through its matched horizon"
                    )
            elif int(args.max_steps) != 5400 or steps != PRIMARY_PROBE_STEPS:
                raise ValueError("final prompt primary stage requires probes through step 5400")
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
    positive_pairing: Any = None,
) -> DataLoader:
    if train:
        dataset = MultiPositiveRetrievalTrainDataset(
            entries[0],
            entries[1],
            transform,
            transform,
            num_positive_photos=1,
            positive_pairing=positive_pairing,
            positive_sampling=str(args.positive_sampling),
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


def _relative_data_path(path: str, data_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"data path is outside dataset root: {path}") from error


def _masked_training_views(
    images: Tensor,
    paths: tuple[str, ...] | list[str],
    *,
    args: DictConfig,
    data_root: Path,
    global_step: int,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    """Make both views on CPU; the masking policy owns deterministic details."""
    from .data.masking import apply_ink_mask, mask_seed

    policy = dict(args.mask_policy)
    fractions = tuple(float(value) for value in policy["train_fractions"])
    first: list[Tensor] = []
    second: list[Tensor] = []
    metadata: list[dict[str, Any]] = []
    for image, path in zip(images, paths, strict=True):
        relative = str(Path(path).resolve().relative_to(data_root.resolve()))
        row: dict[str, Any] = {"path": relative, "global_step": global_step, "views": []}
        for view in (0, 1):
            seed = mask_seed(
                int(policy["train_seed"]) + global_step,
                relative,
                view=view,
            )
            fraction = random.Random(seed).choice(fractions)
            masked, meta = apply_ink_mask(
                image.cpu(),
                fraction=fraction,
                seed=seed,
                ink_threshold=float(policy["ink_threshold"]),
            )
            view_meta = dict(meta)
            view_meta["view"] = view
            row["views"].append(view_meta)
            if view == 0:
                first.append(image.cpu())
            else:
                second.append(masked.cpu())
        metadata.append(row)
    return torch.stack(first), torch.stack(second), metadata


def _single_training_views(images: Tensor) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    return images, images, [{} for _ in range(images.shape[0])]


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
    if _is_fp5(role):
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


def _clip_snapshot(
    model: torch.nn.Module, *, all_parameters: bool = False
) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if all_parameters
        or name.startswith("visual.")
        or name.startswith("encoder.visual.")
    }


def _clip_changed(model: torch.nn.Module, before: dict[str, Tensor]) -> set[str]:
    changed: set[str] = set()
    current = dict(model.named_parameters())
    for name, value in before.items():
        if name not in current or not torch.equal(value, current[name].detach().cpu()):
            changed.add(name)
    return changed


def _approved_clip_names(model: torch.nn.Module, role: str) -> set[str]:
    if _is_layernorm(role) and isinstance(model, FrozenPromptModel):
        return set(model.visual_layernorm_parameter_names)
    if _is_fp5(role):
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
    if _is_fp5(role):
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
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
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
        "artifact_identity": {
            "campaign": str(args.experiment_campaign),
            "run_kind": str(args.run_kind),
            "selection_target_step": (
                int(args.max_steps)
                if str(args.experiment_campaign) in {PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN}
                else None
            ),
            "actual_step": step,
        },
        "step": step,
        "training_global_step": step,
        "comparison_horizon": {"kind": "training_global_step", "value": step},
        "parameters_updated_since_selection": None,
        "model_state_dict": model_state,
        "model_state_complete": False,
        "pairing_identity": {
            "path": str(args.pairing_manifest_path),
            "sha256": manifest_identity.get("pairing_manifest_sha256"),
        },
        "mask_policy": (
            OmegaConf.to_container(args.mask_policy, resolve=True)
            if is_masked_view_campaign(str(args.experiment_campaign))
            else None
        ),
        "mask_policy_sha256": (
            canonical_sha256(OmegaConf.to_container(args.mask_policy, resolve=True))
            if is_masked_view_campaign(str(args.experiment_campaign))
            else None
        ),
        "sketch_view_mode": str(args.sketch_view_mode),
        "two_view_budget": is_masked_view_campaign(str(args.experiment_campaign)),
        "soft_prompt_state_dict": text_state,
        "optimizer_state_dict": None
        if optimizer is None or not bool(args.save_optimizer)
        else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "optimizer_groups": optimizer_groups,
        "rng_state": capture_rng_state(loader_generator),
        "experiment_code_commit": provenance.get("head_commit"),
        "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"),
        "training_seed": int(args.seed),
        "split_seed": int(args.pseudo_val_seed),
        "training_class_list": list(split_identity["train_class_ids"]),
        "validation_class_list": list(split_identity["validation_class_ids"]),
        "class_list_hashes": {
            "train": canonical_sha256(split_identity["train_class_ids"]),
            "validation": canonical_sha256(split_identity["validation_class_ids"]),
        },
        "split_specific_training_artifact": str(args.run_kind) == "split_robustness",
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
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    parent_role = payload.get("experiment_role")
    parent_campaign = payload.get("campaign")
    if payload.get("model_type") != "frozen_prompt_v2":
        raise ValueError("checkpoint model_type does not match this run")
    if parent_role != str(args.experiment_role) or parent_campaign != str(
        args.experiment_campaign
    ):
        raise ValueError("checkpoint role/campaign does not match this run")
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
    model_state = payload.get("model_state_dict", {})
    load_trainable_state(
        model,
        model_state,
        required_keys={"sketch_prompt", "photo_prompt"}
        if isinstance(model, FrozenPromptModel)
        else set(),
    )
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


def _parameter_counts(
    model: torch.nn.Module, text_bank: SoftPromptTextBank | None
) -> dict[str, int]:
    named = _parameter_names(model, text_bank)
    trainable = [parameter for parameter in named.values() if parameter.requires_grad]
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in named.values())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "trainable_parameter_name_count": len(trainable),
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
    if role.endswith("FP4") and changed:
        raise RuntimeError(
            f"FP4 mutated frozen visual parameters: {sorted(changed)[:5]}"
        )
    frozen_changed = changed - allowed
    return {
        "role": role,
        "clip_owned_parameter_names": sorted(before),
        "approved_trainable_clip_parameter_names": sorted(allowed),
        "frozen_clip_parameter_names": sorted(set(before) - allowed),
        "changed_clip_parameter_names": sorted(changed),
        "all_clip_owned_parameters_byte_identical": not changed,
        "frozen_clip_parameter_byte_identical": not frozen_changed,
        "fully_frozen": not allowed,
    }


def _assert_close(left: float, right: float, tolerance: float, name: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"{name} changed by more than tolerance: {left} vs {right}"
        )


_WANDB_RETRIEVAL_METRICS = (
    "full_mAP", "P@200", "mAP@200_prefix_positive",
    "mAP@200_all_relevant", "mAP@200_min_relevant_k",
)


def _wandb_metric_values(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise TypeError("masked-view metrics must be a dictionary")
    return {key: float(metrics[key]) for key in _WANDB_RETRIEVAL_METRICS if key in metrics}


def _safe_wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep W&B config useful without exporting paths, commands, or secrets."""
    allowed = (
        "experiment_name", "experiment_role", "experiment_campaign", "run_kind",
        "model_name", "pretrained", "batch_size", "max_steps", "probe_steps",
        "seed", "pseudo_val_seed", "text_mode", "lambda_rank", "lambda_cls",
        "mask_policy", "sketch_view_mode",
    )
    return {key: config[key] for key in allowed if key in config}


def _write_provenance_artifacts(
    output_dir: Path, args: DictConfig, provenance: dict[str, Any]
) -> None:
    """Write small, local lineage files; never hand W&B the output directory."""
    resolved = OmegaConf.to_container(args, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    OmegaConf.save(config=OmegaConf.create(resolved), f=str(output_dir / "resolved_config.yaml"))
    (output_dir / "command.txt").write_text(
        " ".join(str(value) for value in provenance.get("command", sys.argv)) + "\n",
        encoding="utf-8",
    )
    snapshot = provenance.get("source_snapshot")
    if not isinstance(snapshot, dict):
        return
    snapshot_dir = output_dir / "source_snapshot"
    files_dir = snapshot_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for item in snapshot.get("files", snapshot.get("manifest", [])):
        relative = Path(str(item["path"]))
        source = PROJECT_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"source snapshot file disappeared: {source}")
        destination = files_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != str(item["sha256"]):
            raise ValueError(f"source snapshot copy hash mismatch: {relative}")
    (snapshot_dir / "index.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _assert_fresh_masked_view_output(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    ignored = {".hydra", ".wandb", "wandb"}
    existing = [path for path in output_dir.iterdir() if path.name not in ignored]
    if existing:
        raise ValueError(f"masked-view output must be fresh: {output_dir}")


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _new_wandb_experiment(args: DictConfig, output_dir: Path) -> WandbExperiment | None:
    if not is_masked_view_3600_campaign(str(args.experiment_campaign)):
        return None
    resolved = OmegaConf.to_container(args, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    tracking = resolved.get("tracking") or {}
    if not isinstance(tracking, dict):
        raise TypeError("tracking config must be a mapping")
    return WandbExperiment(
        project=str(tracking.get("project", "spica")),
        entity=tracking.get("entity"),
        mode=str(tracking.get("mode", "disabled")),
        group=str(tracking.get("group", resolved.get("experiment_campaign"))),
        tags=tuple(tracking.get("tags", (str(resolved.get("experiment_role")),))),
        name=str(resolved.get("experiment_name")),
        config=_safe_wandb_config(resolved),
        directory=output_dir,
    )


def run(args: DictConfig) -> None:
    _validate(args)
    is_masked_campaign = is_masked_view_campaign(str(args.experiment_campaign))
    if (
        str(args.run_kind) == "smoke"
        and is_masked_campaign
        and (not bool(args.allow_smoke_fixture) or not bool(args.synthetic_fixture))
    ):
        raise ValueError("masked-view smoke requires an explicit synthetic CPU fixture")
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    if is_masked_campaign:
        _assert_fresh_masked_view_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_experiment = _new_wandb_experiment(args, output_dir)
    success = False
    try:
        _run_impl(args, wandb_experiment)
        success = True
    finally:
        if wandb_experiment is not None:
            wandb_experiment.finish(exit_code=0 if success else 1)


def _run_impl(
    args: DictConfig, wandb_experiment: WandbExperiment | None = None
) -> None:
    _validate(args)
    is_masked_campaign = is_masked_view_campaign(str(args.experiment_campaign))
    if (
        str(args.run_kind) == "smoke"
        and is_masked_campaign
        and (not bool(args.allow_smoke_fixture) or not bool(args.synthetic_fixture))
    ):
        raise ValueError("masked-view smoke requires an explicit synthetic CPU fixture")
    seed = int(args.seed)
    _seed(seed)
    device = _device(str(args.device))
    data = load_data_config(_path(args.data_config))
    split, names, split_identity, data_manifest_identity = _load_split(data, args)
    pairing_manifest = None
    pairing_manifest_sha256 = None
    pairing_photo_sha256: dict[str, str] = {}
    if str(args.experiment_campaign) in {PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN}:
        pairing_path = _path(args.pairing_manifest_path)
        if not pairing_path.is_file():
            raise FileNotFoundError(f"pairing manifest not found: {pairing_path}")
        pairing_manifest_sha256 = hashlib.sha256(pairing_path.read_bytes()).hexdigest()
        if (
            is_masked_view_campaign(str(args.experiment_campaign))
            and str(args.run_kind) == "primary"
            and pairing_manifest_sha256 != PAIRING_MANIFEST_SHA256
        ):
            raise ValueError("masked-view primary pairing manifest hash does not match the approved identity")
        pairing_manifest = load_pairing_manifest(
            pairing_path,
            dataset_root=data.root,
            sketch_entries=split.train_sketch_entries,
            photo_entries=split.train_photo_entries,
        )
        pairing_payload = json.loads(pairing_path.read_text(encoding="utf-8"))
        pairing_photo_sha256 = {
            str(record["photo_path"]): str(record["photo_sha256"])
            for record in pairing_payload["records"]
            if isinstance(record, dict)
            and isinstance(record.get("photo_path"), str)
            and isinstance(record.get("photo_sha256"), str)
        }
        data_manifest_identity = {
            **data_manifest_identity,
            "positive_sampling": str(args.positive_sampling),
            "pairing_manifest_path": str(pairing_path),
            "pairing_manifest_sha256": pairing_manifest_sha256,
            **(
                {
                    "mask_policy": OmegaConf.to_container(args.mask_policy, resolve=True),
                    "mask_policy_sha256": canonical_sha256(OmegaConf.to_container(args.mask_policy, resolve=True)),
                    "two_view_budget": True,
                }
                if is_masked_view_campaign(str(args.experiment_campaign))
                else {}
            ),
        }
    manifest_path = _path(args.experiment_manifest_path)
    manifest, manifest_sha256 = ensure_manifest(
        manifest_path,
        dataset=str(data.name),
        data_config=str(args.data_config),
        campaign=str(args.experiment_campaign),
        positive_sampling=args.get("positive_sampling"),
        pairing_manifest_sha256=pairing_manifest_sha256,
        run_kind=str(args.run_kind),
        selection_target_step=(
            int(args.max_steps)
            if str(args.experiment_campaign) in {PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN}
            else None
        ),
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
        "frozen_prompt_final_FP3S",
        "frozen_prompt_final_FP5",
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
        positive_pairing=pairing_manifest,
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
    photo_before = _clip_snapshot(
        photo_clip.encoder.model,
        all_parameters=str(args.experiment_campaign) in {
            PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN
        },
    )
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "train_observations.jsonl"
    if is_masked_campaign and observations_path.exists():
        raise ValueError("masked-view observation trace already exists; refusing overwrite")
    provenance = capture_provenance(
        PROJECT_ROOT,
        resolved_config=OmegaConf.to_container(args, resolve=True),
        command=[sys.executable, *sys.argv],
    )
    if is_masked_view_3600_campaign(str(args.experiment_campaign)):
        _write_provenance_artifacts(output_dir, args, provenance)
    if wandb_experiment is not None:
        wandb_experiment.define_metric("step_train")
        for name in ("clean/*", "masked/*"):
            wandb_experiment.define_metric(name, step_metric="step_train", summary="max")
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
        if str(args.run_kind) == "primary" and _is_final_role(role):
            if step != 5400:
                raise ValueError(
                    "final prompt continuation must restore the new step-5400 checkpoint"
                )
            if parent_payload.get("experiment_code_commit") != provenance.get(
                "head_commit"
            ) or parent_payload.get("source_snapshot_hash") != provenance.get(
                "source_snapshot", {}
            ).get("sha256"):
                raise ValueError(
                    "final prompt continuation checkpoint was produced by another code snapshot"
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
                "source_experiment_role": parent_payload.get("experiment_role")
                if parent_payload is not None
                else None,
                "source_campaign": parent_payload.get("campaign")
                if parent_payload is not None
                else None,
                "source_experiment_code_commit": parent_payload.get(
                    "experiment_code_commit"
                )
                if parent_payload is not None
                else None,
                "source_snapshot_hash": parent_payload.get("source_snapshot_hash")
                if parent_payload is not None
                else None,
                "optimizer_state_restored": optimizer is not None,
            }
        )
        checkpoint_dir = _path(args.resume_checkpoint_path).parent
        prior_candidates = [
            output_dir / "run_result.json",
            checkpoint_dir / "run_result.json",
            checkpoint_dir.parent / "run_result.json",
        ]
        prior_path = next(
            (candidate for candidate in prior_candidates if candidate.is_file()), None
        )
        if prior_path is not None:
            prior = json.loads(prior_path.read_text())
            same_run = prior.get("experiment_role") == role and prior.get(
                "campaign"
            ) == str(args.experiment_campaign)
            if not same_run:
                raise ValueError("resume history belongs to another run")
            if same_run:
                history = list(prior.get("history", []))
                training_history = list(prior.get("training_history", []))
                if str(args.run_kind) == "primary":
                    expected_steps = set(expected_probe_steps(role))
                    if {
                        int(row.get("training_global_step", -1)) for row in history
                    } - expected_steps:
                        raise ValueError("resume history contains an invalid probe step")
                for row in history:
                    checkpoint = _path(row.get("checkpoint"))
                    expected_hash = row.get("checkpoint_sha256")
                    if not checkpoint.is_file() or not isinstance(expected_hash, str):
                        raise ValueError("resume history has a missing checkpoint identity")
                    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    if actual_hash != expected_hash:
                        raise ValueError("resume history checkpoint hash mismatch")
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

    def _probe_impl(probe_step: int) -> None:
        nonlocal existing_steps
        if probe_step in existing_steps:
            return
        checkpoint = output_dir / "checkpoints" / f"frozen_prompt_step{probe_step}.pt"
        pairing_pilot = str(args.experiment_campaign) in {
            PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN
        }
        clip_policy_current = (
            _assert_clip_policy(photo_clip.encoder.model, photo_before, role)
            if pairing_pilot
            else _assert_clip_policy(prompt_model, clip_before, role)
        )
        if not pairing_pilot:
            photo_changed = _clip_changed(photo_clip.encoder.model, photo_before)
            photo_allowed = (
                set(prompt_model.visual_layernorm_parameter_names)
                if _is_layernorm(role)
                else set()
            )
            if photo_changed - photo_allowed:
                raise RuntimeError(
                    f"unexpected photo CLIP mutation: {sorted(photo_changed - photo_allowed)[:5]}"
                )
        clip_policy_current.update(
            {
                "photo_encoder_frozen": not _is_layernorm(role),
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
        if is_masked_view_3600_campaign(str(args.experiment_campaign)):
            _atomic_copy(checkpoint, output_dir / "checkpoints" / "latest.pt")
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
        masked_view_metrics = None
        if is_masked_campaign and is_masked_view_3600_campaign(str(args.experiment_campaign)):
            from .evaluation.masked_view import evaluate_benchmark_views
            masked_view_metrics = evaluate_benchmark_views(
                query_model, current_sketch, loaded_photo,
                split.validation_sketch_entries, photo_clip.transform, Path(data.root),
                device=device, batch_size=int(args.eval_batch_size),
                query_chunk_size=int(args.query_chunk_size), mask_policy=args.mask_policy,
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
        val_metrics.update(
            {
                "query_identity": _entry_identity(split.validation_sketch_entries),
                "gallery_identity": _entry_identity(split.validation_photo_entries),
            }
        )
        if role.endswith("FP4"):
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
        if role in {"frozen_prompt_v2_FP1S", "frozen_prompt_final_FP3S"} and photo_delta > invariance_tolerance:
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
        parameter_counts = _parameter_counts(
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
            "masked_view": None if masked_view_metrics is None else {
                "clean": _wandb_metric_values(masked_view_metrics["clean"]),
                "masked_macro": _wandb_metric_values(masked_view_metrics["masked_macro"]),
                "masked_by_fraction": {
                    fraction: _wandb_metric_values(values)
                    for fraction, values in masked_view_metrics["masked_by_fraction"].items()
                },
                "raw_evaluation": f"masked_view_probe_step{probe_step}.json",
            },
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
            "parameter_counts": parameter_counts,
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
            "class_list_hashes": {
                "train": canonical_sha256(split_identity["train_class_ids"]),
                "validation": canonical_sha256(split_identity["validation_class_ids"]),
            },
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
        if masked_view_metrics is not None:
            (output_dir / f"masked_view_probe_step{probe_step}.json").write_text(
                json.dumps(masked_view_metrics, indent=2, sort_keys=True) + "\n"
            )
        (output_dir / f"probe_step{probe_step}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        if is_masked_view_3600_campaign(str(args.experiment_campaign)):
            checkpoint_root = output_dir / "checkpoints"
            _atomic_copy(checkpoint, checkpoint_root / "latest.pt")
            eligible_steps = (
                MASKED_VIEW_3600_SELECTION_STEPS
                if str(args.run_kind) == "primary"
                else tuple(int(value) for value in args.probe_steps if int(value) > 0)
            )
            eligible = [
                item for item in history
                if int(item["training_global_step"]) in set(eligible_steps)
            ]
            if eligible:
                clean_best = max(eligible, key=lambda item: float(item["masked_view"]["clean"]["mAP@200_prefix_positive"]))
                masked_best = max(eligible, key=lambda item: float(item["masked_view"]["masked_macro"]["mAP@200_prefix_positive"]))
                if clean_best is row:
                    _atomic_copy(checkpoint, checkpoint_root / "best_clean.pt")
                if masked_best is row:
                    _atomic_copy(checkpoint, checkpoint_root / "best_masked.pt")
        if wandb_experiment is not None and masked_view_metrics is not None:
            clean_metrics = _wandb_metric_values(masked_view_metrics["clean"])
            masked_metrics = _wandb_metric_values(masked_view_metrics["masked_macro"])
            fraction_metrics = {
                float(fraction): _wandb_metric_values(metrics)
                for fraction, metrics in masked_view_metrics["masked_by_fraction"].items()
            }
            conditions = [
                {
                    **_wandb_metric_values(condition),
                    **{key: condition[key] for key in ("fraction", "seed") if key in condition},
                }
                for condition in masked_view_metrics["conditions"]
            ]
            wandb_experiment.log_retrieval_probe(
                probe_step, clean_metrics, masked_metrics, fraction_metrics,
                conditions=conditions,
            )
            if is_masked_view_3600_campaign(str(args.experiment_campaign)) and probe_step > 0:
                eligible_steps = (
                    MASKED_VIEW_3600_SELECTION_STEPS
                    if str(args.run_kind) == "primary"
                    else tuple(int(value) for value in args.probe_steps if int(value) > 0)
                )
                candidates = [
                    item for item in history
                    if int(item["training_global_step"]) in set(eligible_steps)
                ]
                aliases = [f"step{probe_step}", "latest"]
                if candidates:
                    best_clean = max(candidates, key=lambda item: float(item["masked_view"]["clean"]["mAP@200_prefix_positive"]))
                    best_masked = max(candidates, key=lambda item: float(item["masked_view"]["masked_macro"]["mAP@200_prefix_positive"]))
                    if best_clean is row:
                        aliases.append("best_clean")
                    if best_masked is row:
                        aliases.append("best_masked")
                artifact_stage = output_dir / ".wandb_artifacts" / f"step{probe_step}"
                artifact_stage.mkdir(parents=True, exist_ok=True)
                (artifact_stage / "resolved_config.json").write_text(
                    json.dumps(
                        _safe_wandb_config(OmegaConf.to_container(args, resolve=True)),
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="utf-8",
                )
                shutil.copyfile(checkpoint, artifact_stage / checkpoint.name)
                wandb_experiment.log_artifact(
                    artifact_stage,
                    name=f"{role}-{getattr(wandb_experiment, 'run_id', 'local')}-checkpoint",
                    artifact_type="model",
                    metadata={
                        "step": probe_step,
                        "checkpoint_sha256": checkpoint_hash,
                        "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"),
                        "resolved_config_sha256": canonical_sha256(_safe_wandb_config(OmegaConf.to_container(args, resolve=True))),
                    },
                    aliases=aliases,
                )
        existing_steps.add(probe_step)

    def probe(probe_step: int) -> None:
        if is_masked_view_3600_campaign(str(args.experiment_campaign)):
            with _preserve_global_rng():
                _probe_impl(probe_step)
        else:
            _probe_impl(probe_step)

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
                raw_images = batch["sketch"]
                masked_campaign = is_masked_view_campaign(str(args.experiment_campaign))
                if masked_campaign:
                    full_view, masked_images, mask_rows = _masked_training_views(
                        raw_images,
                        tuple(str(path) for path in batch["sketch_path"]),
                        args=args,
                        data_root=Path(data.root),
                        global_step=step,
                    )
                    view_mode = str(args.sketch_view_mode)
                    view_images = (
                        (full_view, full_view)
                        if view_mode == "full_full"
                        else (full_view, masked_images)
                    )
                else:
                    view_images, _, mask_rows = _single_training_views(raw_images)
                images = raw_images.to(device, non_blocking=device.type == "cuda")
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
                query_views = (
                    (query_model(images),)
                    if not masked_campaign
                    else tuple(
                        query_model(
                            torch.cat(
                                tuple(view.to(device, non_blocking=device.type == "cuda") for view in view_images),
                                dim=0,
                            )
                        ).chunk(2, dim=0)
                    )
                )
                with torch.set_grad_enabled(not fixed_photo):
                    photo_values = photo_model.encode_photo(
                        torch.cat((positives, negatives), dim=0)
                    )
                positive, negative = photo_values.split(
                    (images.shape[0], images.shape[0]), dim=0
                )
                def _losses(current: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                    if float(args.lambda_rank) > 0:
                        query_normalized = F.normalize(current, dim=-1)
                        positive_normalized = F.normalize(positive, dim=-1)
                        negative_normalized = F.normalize(negative, dim=-1)
                        current_rank = F.softplus(
                            float(args.margin)
                            - (query_normalized * positive_normalized).sum(-1)
                            + (query_normalized * negative_normalized).sum(-1)
                        ).mean()
                    else:
                        current_rank = current.new_zeros(())
                    current_cls = current.new_zeros(())
                    current_accuracy = current.new_zeros(())
                    if text_bank is not None:
                        bank, bank_labels = _text_bank_values(text_bank, device)
                        current_cls, logits = jepa_text_classification_loss(
                            current, bank, bank_labels, labels,
                            temperature=float(args.tau_cls),
                            detach_text=not isinstance(text_bank, SoftPromptTextBank),
                        )
                        current_accuracy = classification_accuracy(logits, bank_labels, labels)
                    return current_rank, current_cls, current_accuracy

                view_losses = [_losses(current) for current in query_views]
                rank = torch.stack([value[0] for value in view_losses]).mean()
                cls = torch.stack([value[1] for value in view_losses]).mean()
                accuracy = torch.stack([value[2] for value in view_losses]).mean()
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
                history_row = {
                    "training_global_step": step,
                    "last_train_batch_rank_loss": last_train["rank"],
                        "last_train_batch_classification_loss": last_train[
                            "classification"
                        ],
                        "last_train_batch_accuracy": last_train["accuracy"],
                    "gradient_norms": dict(last_gradient_norms),
                    "gradient_norms_by_parameter": dict(last_parameter_gradient_norms),
                }
                if masked_campaign:
                    history_row.update({
                        "first_view_rank_loss": float(view_losses[0][0].item()),
                        "second_view_rank_loss": float(view_losses[1][0].item()),
                        "first_view_classification_loss": float(view_losses[0][1].item()),
                        "second_view_classification_loss": float(view_losses[1][1].item()),
                        "averaged_rank_loss": (
                            float(view_losses[0][0].item()) + float(view_losses[1][0].item())
                        ) / 2.0,
                        "averaged_classification_loss": (
                            float(view_losses[0][1].item()) + float(view_losses[1][1].item())
                        ) / 2.0,
                    })
                    with observations_path.open("a", encoding="utf-8") as trace:
                        for index, (row, negative_path, label) in enumerate(zip(
                            mask_rows, batch["negative_photo_path"], batch["label"], strict=True
                        )):
                            positive_path = _relative_data_path(
                                str(batch["positive_photo_paths"][0][index]),
                                Path(data.root),
                            )
                            negative_path = _relative_data_path(str(negative_path), Path(data.root))
                            row.update({
                                "step": step,
                                "label": int(label),
                                "positive_photo_path": positive_path,
                                "negative_photo_path": negative_path,
                                "positive_photo_sha256": pairing_photo_sha256.get(positive_path),
                                "negative_photo_sha256": pairing_photo_sha256.get(negative_path),
                                "mask_metadata_sha256": canonical_sha256(row["views"]),
                            })
                            trace.write(json.dumps(row, sort_keys=True) + "\n")
                training_history.append(history_row)
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

    if _is_fp5(role) and str(args.run_kind) in {"primary", "split_robustness"}:
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
                    "pseudo_split_identity": split_identity,
                    "class_list_hashes": selected_row["class_list_hashes"],
                    "official_unseen_used_for_selection": False,
                }
            )
    else:
        selected_row = None
        if is_masked_view_3600_campaign(str(args.experiment_campaign)):
            eligible_steps = (
                MASKED_VIEW_3600_SELECTION_STEPS
                if str(args.run_kind) == "primary"
                else tuple(int(value) for value in args.probe_steps if int(value) > 0)
            )
            selection = _select_masked_view_3600(
                history, candidate_steps=eligible_steps
            )
            selected = {
                "selection_metric": "mAP@200_prefix_positive",
                "selection_policy": "best_clean_and_best_masked_separately",
                **selection,
            }
            frozen_hold = []
        elif str(args.experiment_campaign) in {
            PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN
        }:
            target_step = 1800 if str(args.run_kind) == "primary" else int(args.max_steps)
            selected_row = next(
                (
                    row
                    for row in history
                    if int(row["training_global_step"]) == target_step
                ),
                None,
            )
            if selected_row is None:
                raise RuntimeError(
                    f"pairing pilot requires a fixed step-{target_step} result"
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
        if not is_masked_view_3600_campaign(str(args.experiment_campaign)):
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

    if is_masked_view_3600_campaign(str(args.experiment_campaign)) and history:
        latest = max(history, key=lambda row: int(row["training_global_step"]))
        for alias in ("latest.pt",):
            _atomic_copy(Path(str(latest["checkpoint"])), output_dir / "checkpoints" / alias)
        chosen = selected or {}
        for name, value in (("best_clean.pt", chosen.get("best_clean")), ("best_masked.pt", chosen.get("best_masked"))):
            if value is not None:
                _atomic_copy(Path(str(value["checkpoint"])), output_dir / "checkpoints" / name)

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
    pairing_pilot = is_masked_campaign or str(args.experiment_campaign) == PAIRING_PILOT_CAMPAIGN
    final_clip_policy = (
        _assert_clip_policy(photo_clip.encoder.model, photo_before, role)
        if pairing_pilot
        else _assert_clip_policy(prompt_model, clip_before, role)
    )
    if not pairing_pilot:
        photo_changed = _clip_changed(photo_clip.encoder.model, photo_before)
        photo_allowed = (
            set(prompt_model.visual_layernorm_parameter_names)
            if _is_layernorm(role)
            else set()
        )
        if photo_changed - photo_allowed:
            raise RuntimeError(
                f"unexpected photo CLIP mutation: {sorted(photo_changed - photo_allowed)[:5]}"
            )
    final_clip_policy.update(
        {
            "photo_encoder_frozen": not _is_layernorm(role),
            "visual_projection_frozen": True,
            "text_tower_frozen": True,
        }
    )
    artifact_status = (
        "CPU_SMOKE"
        if str(args.run_kind) == "smoke" and str(args.experiment_campaign) in {
            PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN
        }
        else "PRIMARY_FIXED_STEP_UNCOMPARED"
        if str(args.run_kind) == "primary" and str(args.experiment_campaign) in {
            PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN
        }
        else str(args.run_kind).upper()
    )
    observation_trace = None
    if is_masked_campaign and observations_path.is_file():
        observation_trace = {
            "path": str(observations_path),
            "sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest(),
            "bytes": observations_path.stat().st_size,
        }
    report = {
        "schema_version": 2,
        "experiment_role": role,
        "campaign": str(args.experiment_campaign),
        "run_kind": str(args.run_kind),
        "artifact_identity": {
            "status": artifact_status,
            "campaign": str(args.experiment_campaign),
            "run_kind": str(args.run_kind),
            "selection_target_step": (
                1800
                if str(args.experiment_campaign) in {
                    PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN
                }
                and str(args.run_kind) == "primary"
                else int(args.max_steps)
                if str(args.experiment_campaign) in {PAIRING_PILOT_CAMPAIGN, MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN}
                else None
            ),
            "actual_final_step": step,
            "device": device.type,
            "pretrained": args.pretrained is not None,
        },
        "dataset": str(data.name),
        "resolved_config": OmegaConf.to_container(args, resolve=True),
        "resolved_treatment": treatment_from_config(
            OmegaConf.to_container(args, resolve=True)
        ),
        "experiment_code_commit": provenance.get("head_commit"),
        "source_snapshot_hash": provenance.get("source_snapshot", {}).get("sha256"),
        "working_tree_state": provenance.get("working_tree_state"),
        "provenance": provenance,
        "wandb": None if wandb_experiment is None else {
            "run_id": getattr(wandb_experiment, "run_id", None),
            "run_url": getattr(wandb_experiment, "run_url", None),
        },
        "seed": seed,
        "training_seed": seed,
        "pseudo_validation_seed": int(args.pseudo_val_seed),
        "split_seed": int(args.pseudo_val_seed),
        "split_run_name": args.get("split_run_name"),
        "split_specific_training_artifact": str(args.run_kind) == "split_robustness",
        "retrained_from_scratch": str(args.run_kind) == "split_robustness"
        and args.resume_checkpoint_path is None,
        "training_class_list": list(split_identity["train_class_ids"]),
        "validation_class_list": list(split_identity["validation_class_ids"]),
        "pseudo_split_identity": split_identity,
        "manifest_identity": data_manifest_identity,
        "manifest_entry_identity": entry_identity,
        "pairing_identity": {
            "path": str(args.pairing_manifest_path),
            "sha256": pairing_manifest_sha256,
        },
        "mask_policy": None if not is_masked_campaign else OmegaConf.to_container(args.mask_policy, resolve=True),
        "mask_policy_sha256": None if not is_masked_campaign else canonical_sha256(OmegaConf.to_container(args.mask_policy, resolve=True)),
        "sketch_view_mode": str(args.sketch_view_mode),
        "two_view_budget": is_masked_campaign,
        "observation_trace": observation_trace,
        "manifest_path": str(manifest_path),
        "class_list_hashes": {
            "train": canonical_sha256(split_identity["train_class_ids"]),
            "validation": canonical_sha256(split_identity["validation_class_ids"]),
        },
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
        "parameter_counts": _parameter_counts(
            prompt_model,
            text_bank if isinstance(text_bank, SoftPromptTextBank) else None,
        ),
        "checkpoint_state_fields": [
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_state",
            "training_global_step",
        ],
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
        "checkpoint": None if selected is None else selected.get("checkpoint"),
        "checkpoint_sha256": None
        if selected is None
        else selected.get("checkpoint_sha256"),
        "history": history,
        "training_history": training_history,
        "resume": resume_records,
        "selection": (
            None
            if is_masked_view_3600_campaign(str(args.experiment_campaign))
            and str(args.run_kind) == "primary"
            else selected
        ),
        "selections": (
            None if not is_masked_view_3600_campaign(str(args.experiment_campaign))
            else {
                "latest": {
                    "training_global_step": int(max(history, key=lambda row: int(row["training_global_step"]))["training_global_step"]),
                    "checkpoint": max(history, key=lambda row: int(row["training_global_step"]))["checkpoint"],
                    "checkpoint_sha256": max(history, key=lambda row: int(row["training_global_step"]))["checkpoint_sha256"],
                    "scores": {
                        "clean": max(history, key=lambda row: int(row["training_global_step"]))["masked_view"]["clean"],
                        "masked_macro": max(history, key=lambda row: int(row["training_global_step"]))["masked_view"]["masked_macro"],
                    },
                },
                "best_clean": None if selected is None else selected["best_clean"],
                "best_masked": None if selected is None else selected["best_masked"],
            }
        ),
        "frozen_hold_evaluation": frozen_hold,
        "gradient_validation": {
            "active_optimizer_groups_have_nonzero_last_update": {
                group["name"]: bool(
                    group["active"] and last_gradient_norms[group["name"]] > 0.0
                )
                for group in optimizer_groups
            },
            "last_update_gradient_norms": dict(last_gradient_norms),
            "last_update_gradient_norms_by_parameter": dict(
                last_parameter_gradient_norms
            ),
        },
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
            "text_used_for_predictor": False,
            "photo_prompt_used_for_query": False,
            "photo_prompt_used_for_gallery": not fixed_photo,
        },
        "protocol": {
            "selection_metric": (
                "mAP@200_prefix_positive"
                if is_masked_view_3600_campaign(str(args.experiment_campaign))
                else "full_pseudo_unseen_mAP"
            ),
            "official_unseen_used_for_selection": False,
            "split_specific_training": str(args.run_kind) == "split_robustness",
            "text_used_for_predictor": False,
            "photo_prompt_used_for_query": False,
            "photo_prompt_used_for_gallery": not fixed_photo,
            "transport_enabled": False,
            "direction_supervision": False,
            "distance_prediction": False,
            "num_positive_photos": 1,
            "sketch_view_mode": str(args.sketch_view_mode),
            "mask_policy": None if not is_masked_campaign else OmegaConf.to_container(args.mask_policy, resolve=True),
            "two_view_budget": is_masked_campaign,
            "pairing_manifest_sha256": pairing_manifest_sha256,
            "metric_denominator": "prefix_positive" if is_masked_view_3600_campaign(str(args.experiment_campaign)) else None,
            "benchmark_status": "public_sketchlvm_not_verified" if is_masked_view_3600_campaign(str(args.experiment_campaign)) else None,
        },
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "run_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if wandb_experiment is not None:
        summary = {
            "best_clean_step": report["selections"]["best_clean"]["training_global_step"],
            "best_masked_step": report["selections"]["best_masked"]["training_global_step"],
            "latest_step": report["selections"]["latest"]["training_global_step"],
            "best_clean_checkpoint_sha256": report["selections"]["best_clean"]["checkpoint_sha256"],
            "best_masked_checkpoint_sha256": report["selections"]["best_masked"]["checkpoint_sha256"],
            "latest_checkpoint_sha256": report["selections"]["latest"]["checkpoint_sha256"],
            "source_snapshot_hash": report["source_snapshot_hash"],
        }
        for selection_name in ("best_clean", "best_masked", "latest"):
            selection = report["selections"][selection_name]
            for metric_name in _WANDB_RETRIEVAL_METRICS:
                view = selection["scores"].get("clean" if selection_name == "best_clean" else "masked_macro", {})
                if metric_name in view:
                    summary[f"{selection_name}/{metric_name}"] = float(view[metric_name])
        wandb_experiment.set_summary(summary)


@hydra.main(
    version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="train_frozen_prompt"
)
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
