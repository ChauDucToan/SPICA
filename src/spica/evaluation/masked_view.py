"""Standalone query-only masked-view evaluation for frozen prompt runs.

The evaluator is deliberately kept separate from training: it loads the full
OpenCLIP backbone, replays the run's pseudo-validation split, reuses the
checkpoint-bound photo cache, and masks normalized query pixels before the
visual encoder is called.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config.data import load_data_config
from ..data.datasets import RetrievalEvalDataset, _load_rgb_image
from ..data.manifest import ManifestEntry, read_class_map, read_manifest
from ..data.masking import MASK_POLICY_VERSION
from ..data.pairing import load_pairing_manifest
from ..data.splits import make_classwise_retrieval_split, split_manifest_identity
from ..models.checkpoint import load_prompt_checkpoint
from ..models.clip import load_frozen_clip
from ..models.frozen_prompt import FrozenPromptModel
from ..frozen_prompt_artifacts import (
    MASKED_VIEW_3600_CAMPAIGN,
    MASKED_VIEW_3600_ROLES,
    MASKED_VIEW_3600_STEPS,
    MASKED_VIEW_CAMPAIGN,
    MASKED_VIEW_ROLES,
)
from .embeddings import EncodedRetrievalSet
from .frozen_prompt import cache_identity, encode_prompted_loader, load_prompt_cache
from .metrics import evaluate_category_retrieval_all_denominators

# These are the actual corrected alignment pseudo-validation cardinalities.
PSEUDO_VALIDATION_SEED = 3407
PSEUDO_QUERY_COUNT = 10_963
PSEUDO_GALLERY_COUNT = 13_999
EVAL_FRACTIONS = (0.25, 0.5, 0.75)
EVAL_SEEDS = (101, 202, 303)
CLEAN_REPLAY_TOLERANCE_CPU = 1e-6
CLEAN_REPLAY_TOLERANCE_CUDA = 1e-5
_ALLOWED_MASK_STATUSES = {"ok", "zero_fraction", "blank_input", "target_unreachable"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_path(value: object, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _entry_identity(entries: Iterable[ManifestEntry]) -> dict[str, Any]:
    rows = [[str(entry.path), int(entry.label)] for entry in entries]
    return {"count": len(rows), "sha256": canonical_sha256(rows), "paths_and_labels": rows}


def _finite(name: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _mask_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the approved plan recorded by the training config."""
    from spica.data.masking import MASK_POLICY_VERSION

    policy = config.get("mask_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("masked-view resolved_config.mask_policy is missing")
    try:
        version = str(policy["version"])
        threshold = _finite("mask_policy.ink_threshold", policy["ink_threshold"])
        train_fractions = tuple(float(value) for value in policy["train_fractions"])
        train_seed = int(policy["train_seed"])
        fractions = tuple(float(value) for value in policy["eval_fractions"])
        seeds = tuple(int(value) for value in policy["eval_seeds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("mask_policy does not contain a valid evaluation plan") from error
    if version != MASK_POLICY_VERSION:
        raise ValueError(f"unsupported mask policy version: {version!r}")
    if (
        train_fractions != EVAL_FRACTIONS
        or train_seed != 4242
        or fractions != EVAL_FRACTIONS
        or seeds != EVAL_SEEDS
        or threshold != 0.9
    ):
        raise ValueError("resolved mask policy differs from the approved evaluation plan")
    return {
        "policy_version": version,
        "train_fractions": list(train_fractions),
        "train_seed": train_seed,
        "fractions": list(fractions),
        "seeds": list(seeds),
        "view": 0,
        "ink_threshold": threshold,
        "sample_key": "dataset-root-relative-path",
    }


def _validate_metrics(evaluation: Any, *, name: str) -> None:
    metrics = evaluation.metrics
    for metric_name, value in (
        ("mAP", metrics.mean_average_precision),
        ("P@200", metrics.precision_at_k.get(200)),
        ("mAP@200", metrics.mean_average_precision_at_k.get(200)),
    ):
        _finite(f"{name}.{metric_name}", value)
    if evaluation.average_precision_per_query.numel() != metrics.num_queries:
        raise ValueError(f"{name}.average_precision_per_query length does not match query count")
    if not torch.isfinite(evaluation.average_precision_per_query).all().item():
        raise ValueError(f"{name}.average_precision_per_query contains NaN or Inf")


def _status_counts(metas: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(_ALLOWED_MASK_STATUSES)}
    for meta in metas:
        status = str(meta.get("status"))
        if status not in counts:
            raise ValueError(f"unknown mask status: {status!r}")
        counts[status] += 1
    return counts


def _resolved_run_path(value: object, run_result: Path) -> Path:
    # Match train_frozen_prompt._path: preserve the recorded spelling for
    # identity, but resolve a separate filesystem path for I/O and hashing.
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[3]
    candidate = (project_root / path).resolve()
    return candidate if candidate.exists() else (run_result.parent / path).resolve()


def _path_resolution_diagnostic(value: object, resolved: Path) -> dict[str, str | bool]:
    """Describe resolution without changing the historical identity value."""
    recorded = str(value)
    return {
        "recorded": recorded,
        "resolved": str(resolved),
        "recorded_is_absolute": Path(recorded).expanduser().is_absolute(),
        "exists": resolved.exists(),
    }


def _manifest_identity_matches(
    recorded: Mapping[str, Any], reconstructed: Mapping[str, Any], *, base: Path | None = None,
) -> bool:
    """Allow equivalent pairing path spellings, while keeping all else strict."""
    if set(recorded) != set(reconstructed):
        return False
    for key, value in reconstructed.items():
        if key == "pairing_manifest_path":
            left = Path(str(recorded.get(key))).expanduser()
            right = Path(str(value)).expanduser()
            if base is not None:
                left = left if left.is_absolute() else base / left
                right = right if right.is_absolute() else base / right
            if left.resolve() != right.resolve():
                return False
        elif recorded.get(key) != value:
            return False
    return True


def _selected_record(
    result: Mapping[str, Any], checkpoint_step: int, *, selection: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Resolve a history row without weakening the historical fixed-step path."""
    history = result.get("history")
    if not isinstance(history, list):
        raise ValueError("run-result history is missing")
    if selection is not None:
        selections = result.get("selections")
        if not isinstance(selections, Mapping):
            raise ValueError("run-result named selections are missing")
        selected = selections.get(selection)
        if not isinstance(selected, Mapping):
            raise ValueError(f"run-result selection {selection!r} is missing")
        selected_step = int(selected.get("training_global_step", -1))
        if selected_step != checkpoint_step:
            raise ValueError("named selection step disagrees with requested checkpoint")
    else:
        selected = result.get("selection")

    rows = [
        row for row in history
        if isinstance(row, dict)
        and int(row.get("training_global_step", row.get("step", -1))) == checkpoint_step
    ]
    if len(rows) != 1:
        raise ValueError(f"run-result has no unique checkpoint at step {checkpoint_step}")
    row = rows[0]
    if isinstance(selected, Mapping) and int(selected.get("training_global_step", -1)) == checkpoint_step:
        if selected.get("checkpoint") != row.get("checkpoint"):
            raise ValueError("selection checkpoint disagrees with step history")
        if selected.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
            raise ValueError("selection checkpoint SHA256 disagrees with step history")
    checkpoint = Path(str(row.get("checkpoint", "")))
    if not str(checkpoint):
        raise ValueError("selected history row has no checkpoint")
    return row, checkpoint


def resolve_pseudo_validation(
    result: Mapping[str, Any], *, run_result_path: Path
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    config = result.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError("run-result resolved_config is missing")
    data_config = _resolved_run_path(config.get("data_config"), run_result_path)
    data = load_data_config(data_config)
    names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches,
        photos,
        names,
        num_validation_classes=int(config.get("pseudo_val_num_classes", 20)),
        seed=int(config.get("pseudo_val_seed", PSEUDO_VALIDATION_SEED)),
    )
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
    manifest_identity = split_manifest_identity(
        split,
        dataset_name=data.name,
        dataset_root=data.root,
        manifest_paths={
            "train_sketch": data.train.sketch_manifest,
            "train_photo": data.train.photo_manifest,
            "train_class_map": data.train.class_map,
        },
    )
    pairing_value = config.get("pairing_manifest_path")
    pairing_path = _resolved_run_path(pairing_value, run_result_path)
    if not pairing_path.is_file():
        raise FileNotFoundError(f"pairing manifest not found: {pairing_path}")
    pairing_sha256 = sha256_file(pairing_path)
    pairing_identity = result.get("pairing_identity", {})
    if pairing_identity.get("sha256") != pairing_sha256:
        raise ValueError("pairing manifest SHA256 differs from run provenance")
    recorded_pairing_path = pairing_identity.get("path")
    if recorded_pairing_path != pairing_value:
        raise ValueError("pairing manifest path differs between resolved config and run provenance")
    load_pairing_manifest(
        pairing_path,
        dataset_root=data.root,
        sketch_entries=split.train_sketch_entries,
        photo_entries=split.train_photo_entries,
    )
    manifest_identity = {
        **manifest_identity,
        "positive_sampling": str(config.get("positive_sampling")),
        "pairing_manifest_path": str(recorded_pairing_path),
        "pairing_manifest_sha256": pairing_sha256,
        "mask_policy": config.get("mask_policy"),
        "mask_policy_sha256": canonical_sha256(config.get("mask_policy")),
        "two_view_budget": True,
    }
    if result.get("pseudo_split_identity") != split_identity:
        raise ValueError("reconstructed pseudo-validation split identity differs")
    # Compare the trainer's historical spelling, not an evaluator-normalized
    # absolute path.  Every other identity field stays an exact comparison.
    recorded_manifest = result.get("manifest_identity")
    if not isinstance(recorded_manifest, Mapping) or not _manifest_identity_matches(
        recorded_manifest, manifest_identity, base=Path(__file__).resolve().parents[3]
    ):
        raise ValueError("reconstructed data manifest identity differs")
    expected_query_count = len(split.validation_sketch_entries)
    expected_gallery_count = len(split.validation_photo_entries)
    if expected_query_count != PSEUDO_QUERY_COUNT and str(result.get("run_kind")) != "smoke":
        raise ValueError("this evaluator requires the 10,963-query pseudo-validation split")
    if expected_gallery_count != PSEUDO_GALLERY_COUNT and str(result.get("run_kind")) != "smoke":
        raise ValueError("this evaluator requires the 13,999-photo pseudo-validation gallery")
    return data, split, manifest_identity


def _mask_image(
    image: Tensor,
    *,
    fraction: float,
    base_seed: int,
    sample_key: str,
    mean: Tensor | tuple[float, ...] | list[float],
    std: Tensor | tuple[float, ...] | list[float],
    ink_threshold: float,
) -> tuple[Tensor, dict[str, Any]]:
    from spica.data.masking import apply_ink_mask, mask_seed

    return apply_ink_mask(
        image.cpu(),
        fraction=fraction,
        seed=mask_seed(base_seed, sample_key, view=0),
        mean=mean,
        std=std,
        ink_threshold=ink_threshold,
    )


def _masked_loader(
    entries: tuple[ManifestEntry, ...],
    transform: Any,
    *,
    root: Path,
    fraction: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    mean: Any,
    std: Any,
    ink_threshold: float,
) -> DataLoader:
    class DatasetImpl(Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return len(entries)

        def __getitem__(self, index: int) -> dict[str, Any]:
            entry = entries[index]
            image = transform(_load_rgb_image(entry))
            key = entry.path.resolve().relative_to(root.resolve()).as_posix()
            masked, meta = _mask_image(
                image,
                fraction=fraction,
                base_seed=seed,
                sample_key=key,
                mean=mean,
                std=std,
                ink_threshold=ink_threshold,
            )
            return {"image": masked, "label": entry.label, "path": str(entry.path), "mask_meta": meta}

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image": torch.stack([sample["image"] for sample in samples]),
            "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
            "path": tuple(sample["path"] for sample in samples),
            "mask_meta": [sample["mask_meta"] for sample in samples],
        }

    return DataLoader(
        DatasetImpl(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
    )


def _encode_masked_loader(model: Any, loader: DataLoader) -> tuple[EncodedRetrievalSet, list[dict[str, Any]]]:
    model.eval()
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    paths: list[str] = []
    metas: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(model.device, non_blocking=model.device.type == "cuda")
            embeddings.append(model(images).float().cpu())
            labels.append(batch["label"].long().cpu())
            paths.extend(str(path) for path in batch["path"])
            metas.extend(batch["mask_meta"])
    if not embeddings:
        raise ValueError("cannot encode an empty masked query loader")
    return EncodedRetrievalSet(torch.cat(embeddings), torch.cat(labels), tuple(paths)), metas


def _transform_stats(transform: Any) -> tuple[Any, Any]:
    # open_clip's Compose exposes Normalize as a nested transform rather than
    # a stable public attribute. Find it without hard-coding a dependency API.
    for item in getattr(transform, "transforms", ()):
        if hasattr(item, "mean") and hasattr(item, "std"):
            return item.mean, item.std
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return mean, std


def evaluate_benchmark_views(
    model: Any,
    clean_queries: EncodedRetrievalSet,
    gallery: EncodedRetrievalSet,
    sketch_entries: tuple[ManifestEntry, ...],
    transform: Any,
    data_root: Path,
    *,
    device: str | torch.device,
    batch_size: int,
    query_chunk_size: int,
    mask_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate clean plus the fixed 3 fractions x 3 seeds masked views."""
    from .metrics import evaluate_category_retrieval_all_denominators

    if not sketch_entries or not clean_queries.embeddings.shape[0] or not gallery.embeddings.shape[0]:
        raise ValueError("benchmark views require non-empty queries, gallery, and entries")
    expected_paths = tuple(str(entry.path) for entry in sketch_entries)
    expected_labels = torch.tensor([entry.label for entry in sketch_entries], dtype=torch.long)
    if clean_queries.paths != expected_paths or not torch.equal(clean_queries.labels.cpu(), expected_labels):
        raise ValueError("clean query paths/labels must match masked sketch entries in order")
    if mask_policy.get("version") != MASK_POLICY_VERSION or float(mask_policy.get("ink_threshold", -1)) != 0.9:
        raise ValueError("benchmark masking version/threshold differs from the approved policy")
    fractions = tuple(float(value) for value in mask_policy.get("eval_fractions", ()))
    seeds = tuple(int(value) for value in mask_policy.get("eval_seeds", ()))
    if fractions != EVAL_FRACTIONS or seeds != EVAL_SEEDS:
        raise ValueError("benchmark mask policy must contain the fixed 3x3 evaluation plan")
    mean, std = _transform_stats(transform)
    kwargs = dict(precision_at_k=(200,), map_at_k=(200,), query_chunk_size=query_chunk_size, top_k=200, device=device)

    def metrics(encoded: EncodedRetrievalSet) -> dict[str, Any]:
        evaluations = evaluate_category_retrieval_all_denominators(encoded, gallery, **kwargs)
        prefix = evaluations["prefix_positive"]
        return {
            "full_mAP": float(prefix.metrics.mean_average_precision),
            "P@200": float(prefix.metrics.precision_at_k[200]),
            "mAP@200_prefix_positive": float(prefix.metrics.mean_average_precision_at_k[200]),
            "mAP@200_all_relevant": float(evaluations["all_relevant"].metrics.mean_average_precision_at_k[200]),
            "mAP@200_min_relevant_k": float(evaluations["min_relevant_k"].metrics.mean_average_precision_at_k[200]),
            "average_precision_per_query": prefix.average_precision_per_query.tolist(),
            "average_precision_at_k_per_query": {
                name: evaluations[name].average_precision_at_k_per_query[200].tolist()
                for name in ("prefix_positive", "all_relevant", "min_relevant_k")
            },
        }

    original_training = bool(model.training)
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        clean = metrics(clean_queries)
        conditions: list[dict[str, Any]] = []
        for fraction in fractions:
            for seed in seeds:
                loader = _masked_loader(
                    sketch_entries, transform, root=data_root, fraction=fraction, seed=seed,
                    batch_size=batch_size, num_workers=0, mean=mean, std=std,
                    ink_threshold=float(mask_policy.get("ink_threshold", 0.9)),
                )
                encoded, metas = _encode_masked_loader(model, loader)
                row = metrics(encoded)
                row.update({
                    "fraction": fraction, "seed": seed,
                    "status_counts": _status_counts(metas),
                    "mask_metadata": {"count": len(metas), "metas": metas},
                })
                conditions.append(row)
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.random.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(original_training)

    keys = (
        "full_mAP", "P@200", "mAP@200_prefix_positive",
        "mAP@200_all_relevant", "mAP@200_min_relevant_k",
    )
    def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in keys}
    return {
        "benchmark": "SPICA_category_retrieval", "benchmark_status": "official_not_verified",
        "evaluation_scope": "pseudo_validation", "official_unseen_used": False,
        "clean": clean, "conditions": conditions, "masked_macro": macro(conditions),
        "masked_by_fraction": {str(fraction): macro([row for row in conditions if row["fraction"] == fraction]) for fraction in fractions},
    }


def evaluate_run(
    run_result_path: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    checkpoint_step: int | None = None,
    selection: str | None = None,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    if selection not in {None, "latest", "best_clean", "best_masked"}:
        raise ValueError(f"unsupported checkpoint selection: {selection!r}")
    if selection is not None and checkpoint_step is not None:
        raise ValueError("--selection and --checkpoint-step are mutually exclusive")

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    result = json.loads(run_result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("run-result must contain a JSON object")
    campaign = str(result.get("campaign", ""))
    is_3600 = campaign == MASKED_VIEW_3600_CAMPAIGN
    if selection is not None and not is_3600:
        raise ValueError("named checkpoint selection is only supported by the 3600 campaign")
    if selection is not None:
        named = result.get("selections")
        if not isinstance(named, Mapping) or not isinstance(named.get(selection), Mapping):
            raise ValueError(f"run-result named selection {selection!r} is missing")
        checkpoint_step = int(named[selection]["training_global_step"])
    elif checkpoint_step is None:
        # Preserve the old evaluator's fixed-step default exactly.
        checkpoint_step = 1800

    run_kind = str(result.get("run_kind", ""))
    if run_kind == "smoke":
        if not allow_smoke:
            raise ValueError("smoke evaluation requires explicit allow_smoke=True")
        if str(result.get("resolved_config", {}).get("device")) != "cpu":
            raise ValueError("smoke evaluation requires device=cpu")
        if not bool(result.get("resolved_config", {}).get("synthetic_fixture")):
            raise ValueError("smoke evaluation requires an explicit synthetic fixture")
        if checkpoint_step > 15:
            raise ValueError("smoke evaluation requires an explicit checkpoint step <= 15")
    row, checkpoint_value = _selected_record(result, checkpoint_step, selection=selection)
    checkpoint = _resolved_run_path(checkpoint_value, run_result_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != row.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA256 does not match selected history row")
    selected = (
        result.get("selections", {}).get(selection)
        if selection is not None and isinstance(result.get("selections"), Mapping)
        else result.get("selection")
    )
    if not is_3600 and not isinstance(selected, Mapping):
        raise ValueError("run-result selection is not the requested fixed evaluation step")
    if isinstance(selected, Mapping):
        if int(selected.get("training_global_step", -1)) != checkpoint_step:
            raise ValueError("run-result selection is not the requested evaluation step")
        if selected.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("checkpoint SHA256 does not match run-result selection")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or int(payload.get("step", payload.get("training_global_step", -1))) != checkpoint_step:
        raise ValueError("checkpoint step is not the requested fixed evaluation step")
    config = result.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError("run-result resolved_config is missing")
    if payload.get("resolved_config") != config:
        raise ValueError("checkpoint resolved_config differs from run-result")
    if payload.get("source_snapshot_hash") != result.get("source_snapshot_hash"):
        raise ValueError("checkpoint source snapshot differs from run-result")
    if payload.get("data_manifest_identity") != result.get("manifest_identity"):
        raise ValueError("checkpoint manifest identity differs from run-result")
    if payload.get("data_split_identity") != result.get("pseudo_split_identity"):
        raise ValueError("checkpoint split identity differs from run-result")
    data, split, manifest_identity = resolve_pseudo_validation(result, run_result_path=run_result_path)
    if payload.get("data_manifest_identity") != manifest_identity:
        raise ValueError("checkpoint manifest identity differs from reconstructed data")
    if payload.get("model_type") != "frozen_prompt_v2":
        raise ValueError("masked-view evaluator requires the existing frozen_prompt_v2 checkpoint format")
    selected_role = str(result.get("experiment_role", payload.get("experiment_role", "")))
    if payload.get("campaign") != campaign or campaign not in {MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN}:
        raise ValueError("run/checkpoint is not an approved masked-view campaign")
    allowed_roles = MASKED_VIEW_3600_ROLES if is_3600 else MASKED_VIEW_ROLES
    if selected_role not in allowed_roles or payload.get("experiment_role") != selected_role:
        raise ValueError("run/checkpoint role is not an approved masked-view role")
    expected_mode = "full_full" if selected_role.endswith("_C") else "full_masked"
    if config.get("sketch_view_mode") != expected_mode:
        raise ValueError("resolved_config sketch_view_mode does not match the selected role")
    if config.get("experiment_role") != selected_role or config.get("experiment_campaign") != campaign:
        raise ValueError("resolved_config role/campaign differs from run-result")
    run_kind = str(result.get("run_kind", payload.get("run_kind", "")))
    if payload.get("run_kind") != run_kind or run_kind not in {"primary", "smoke"}:
        raise ValueError("run/checkpoint run_kind is invalid")
    if run_kind == "primary" and not is_3600 and checkpoint_step != 1800:
        raise ValueError("primary evaluation is fixed to step 1800")
    if run_kind == "primary" and is_3600 and checkpoint_step not in MASKED_VIEW_3600_STEPS:
        raise ValueError("3600 evaluation requires a recorded probe step")
    if run_kind == "primary" and is_3600 and selection in {"best_clean", "best_masked"} and checkpoint_step == 0:
        raise ValueError("3600 named selections cannot select step 0")
    if run_kind == "smoke" and checkpoint_step > 15:
        raise ValueError("smoke evaluation requires an explicit checkpoint step <= 15")
    plan = _mask_plan(config)
    recorded_policy = result.get("mask_policy")
    if recorded_policy != config.get("mask_policy"):
        raise ValueError("run-result mask policy differs from resolved_config")
    if result.get("mask_policy_sha256") != canonical_sha256(recorded_policy):
        raise ValueError("run-result mask policy hash is invalid")
    pairing = result.get("pairing_identity")
    if not isinstance(pairing, Mapping) or not pairing.get("sha256"):
        raise ValueError("run-result pairing identity is missing")
    if payload.get("pairing_identity") is not None and payload.get("pairing_identity") != pairing:
        raise ValueError("checkpoint pairing identity differs from run-result")
    trace = result.get("observation_trace")
    if run_kind == "primary":
        if not isinstance(trace, Mapping):
            raise ValueError("primary masked-view run is missing observation trace")
        trace_path = _resolved_run_path(trace.get("path"), run_result_path)
        if not trace_path.is_file() or sha256_file(trace_path) != trace.get("sha256"):
            raise ValueError("observation trace is missing or has the wrong SHA256")
    actual_device = torch.device(device or config.get("device", "cpu"))
    tolerance = CLEAN_REPLAY_TOLERANCE_CUDA if actual_device.type == "cuda" else CLEAN_REPLAY_TOLERANCE_CPU
    if actual_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    pretrained = config.get("pretrained")
    clip = load_frozen_clip(
        model_name=str(config["model_name"]),
        pretrained=None if pretrained is None else str(pretrained),
        device=actual_device,
    )
    model = FrozenPromptModel(
        clip.encoder.model.visual,
        prompt_length=int(config["visual_prompt_length"]),
        train_visual_layernorm=bool(config.get("train_visual_layernorm", False)),
        train_sketch_prompt=bool(config.get("train_sketch_prompt", True)),
        train_photo_prompt=bool(config.get("train_photo_prompt", True)),
    ).to(actual_device)
    load_info = load_prompt_checkpoint(model, payload, expected_config={
        key: config[key] for key in (
            "model_name", "pretrained", "visual_prompt_length",
            "train_visual_layernorm", "train_sketch_prompt", "train_photo_prompt",
        ) if key in config
    })
    cache_path = checkpoint.parent.parent / "gallery_cache" / f"photo_step{checkpoint_step}.pt"
    identity = cache_identity(
        prompt_checkpoint_hash=checkpoint_hash,
        prompt_length=int(config["visual_prompt_length"]),
        prompt_mode=str(config["prompt_mode"]),
        modality="photo",
        model_name=str(config["model_name"]),
        pretrained=None if pretrained is None else str(pretrained),
        data_manifest_identity=manifest_identity,
    )
    gallery = load_prompt_cache(cache_path, expected_identity=identity)
    expected_gallery = _entry_identity(split.validation_photo_entries)
    if gallery.paths != tuple(str(entry.path) for entry in split.validation_photo_entries):
        raise ValueError("gallery cache path order differs from reconstructed validation gallery")
    if not torch.equal(gallery.labels, torch.tensor([entry.label for entry in split.validation_photo_entries], dtype=torch.long)):
        raise ValueError("gallery cache labels/order differs from reconstructed validation gallery")
    if result.get("history") and isinstance(row.get("val"), Mapping):
        recorded_gallery = row["val"].get("gallery_identity")
        if recorded_gallery is not None and recorded_gallery != expected_gallery:
            raise ValueError("gallery cache identity differs from the selected clean probe")
    query_entries = tuple(split.validation_sketch_entries)
    clean_loader = DataLoader(
        RetrievalEvalDataset(query_entries, clip.transform),
        batch_size=int(config.get("eval_batch_size", 256)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=False,
    )
    clean = encode_prompted_loader(model, clean_loader)
    clean_evaluations = evaluate_category_retrieval_all_denominators(
        clean, gallery, precision_at_k=(1, 5, 10, 100, 200), map_at_k=(200,),
        query_chunk_size=int(config.get("query_chunk_size", 256)), top_k=200, device=actual_device,
    )
    clean_eval = clean_evaluations["prefix_positive"]
    for denominator, variant in clean_evaluations.items():
        _validate_metrics(variant, name=f"clean {denominator}")
    raw_final = row.get("val", {}).get("full_mAP") if isinstance(row.get("val"), dict) else row.get("full_pseudo_unseen_mAP")
    if raw_final is None:
        raw_final = row.get("full_pseudo_unseen_mAP")
    mean, std = _transform_stats(clip.transform)
    conditions: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for fraction in plan["fractions"]:
        for seed in plan["seeds"]:
            loader = _masked_loader(
                query_entries, clip.transform, root=data.root, fraction=fraction, seed=seed,
                batch_size=int(config.get("eval_batch_size", 256)), num_workers=int(config.get("num_workers", 0)),
                mean=mean, std=std, ink_threshold=float(plan["ink_threshold"]),
            )
            encoded, metas = _encode_masked_loader(model, loader)
            evaluations = evaluate_category_retrieval_all_denominators(
                encoded, gallery, precision_at_k=(1, 5, 10, 100, 200), map_at_k=(200,),
                query_chunk_size=int(config.get("query_chunk_size", 256)), top_k=200, device=actual_device,
            )
            evaluation = evaluations["prefix_positive"]
            for denominator, variant in evaluations.items():
                _validate_metrics(variant, name=f"condition fraction={fraction} seed={seed} {denominator}")
            ap = evaluation.average_precision_per_query.tolist()
            status_counts = _status_counts(metas)
            row_condition = {
                "fraction": fraction, "seed": seed,
                "full_mAP": evaluation.metrics.mean_average_precision,
                "P@200": evaluation.metrics.precision_at_k.get(200),
                "mAP@200_prefix_positive": evaluation.metrics.mean_average_precision_at_k.get(200),
                "mAP@200_all_relevant": evaluations["all_relevant"].metrics.mean_average_precision_at_k.get(200),
                "mAP@200_min_relevant_k": evaluations["min_relevant_k"].metrics.mean_average_precision_at_k.get(200),
                "average_precision_per_query": ap,
                "average_precision_at_k_per_query": {
                    name: variant.average_precision_at_k_per_query[200].tolist()
                    for name, variant in evaluations.items()
                },
                "mean_realized_fraction": float(sum(float(m["realized_fraction"]) for m in metas) / len(metas)),
                "realized_fraction_quantiles": {
                    key: float(torch.tensor([float(m["realized_fraction"]) for m in metas]).quantile(q).item())
                    for key, q in (("q0.00", 0.0), ("q0.25", 0.25), ("q0.50", 0.5), ("q0.75", 0.75), ("q1.00", 1.0))
                },
                "status_counts": status_counts,
                "blank_input_count": status_counts["blank_input"],
                "target_unreachable_count": status_counts["target_unreachable"],
                "zero_fraction_count": status_counts["zero_fraction"],
                "ok_count": status_counts["ok"],
            }
            conditions.append(row_condition)
            for entry, meta in zip(query_entries, metas, strict=True):
                relative = entry.path.resolve().relative_to(data.root.resolve()).as_posix()
                manifest_rows.append({
                    "path": relative, "label": int(entry.label),
                    "mask_meta": meta, "fraction": fraction, "seed": seed,
                    "input_sha256": meta.get("input_sha256"), "output_sha256": meta.get("output_sha256"),
                    "bbox": meta.get("bbox"),
                    "ink_pixels_before": meta.get("ink_pixels_before"),
                    "ink_pixels_erased": meta.get("ink_pixels_erased"),
                    "ink_pixels_after": meta.get("ink_pixels_after"),
                    "status": meta.get("status"),
                })
    primary = float(sum(_finite("condition.full_mAP", row["full_mAP"]) for row in conditions) / len(conditions))
    if raw_final is None:
        raise ValueError("selected raw training mAP is missing; clean replay cannot be validated")
    clean_delta = abs(_finite("clean.mAP", clean_eval.metrics.mean_average_precision) - _finite("raw_final_mAP", raw_final))
    if clean_delta > tolerance:
        raise ValueError(
            f"clean replay mAP delta {clean_delta:.9g} exceeds declared tolerance {tolerance:.9g}"
        )
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "query_mask_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8")
    mask_plan = {**plan, "query_identity": _entry_identity(query_entries)}
    aggregate_status_counts = {status: sum(row["status_counts"][status] for row in conditions) for status in sorted(_ALLOWED_MASK_STATUSES)}
    manifest_rows_hash = canonical_sha256(manifest_rows)
    pairing_value = config.get("pairing_manifest_path")
    pairing_path = _resolved_run_path(pairing_value, run_result_path)
    path_resolution = {
        "pairing_manifest": _path_resolution_diagnostic(pairing_value, pairing_path),
        "data_config": _path_resolution_diagnostic(config.get("data_config"), _resolved_run_path(config.get("data_config"), run_result_path)),
        "manifests": {
            name: _path_resolution_diagnostic(path, path.resolve())
            for name, path in {
                "train_sketch": data.train.sketch_manifest,
                "train_photo": data.train.photo_manifest,
                "train_class_map": data.train.class_map,
            }.items()
        },
    }
    report = {
        "schema_version": 2, "status": "COMPLETE", "evaluation_kind": "masked_query_only",
        "primary_mask_score_macro_full_mAP_9_conditions": primary,
        "campaign": campaign, "experiment_role": selected_role, "run_kind": run_kind,
        "selection": None if selection is None else selection,
        "selection_name": selection,
        "selection_record": selected,
        "selection_policy": result.get("selection_policy"),
        "checkpoint_step": checkpoint_step, "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(checkpoint), "run_result": str(run_result_path),
        "source_snapshot_hash": result.get("source_snapshot_hash"),
        "source_checkpoint_hash": checkpoint_hash, "config_sha256": canonical_sha256(config),
        "data_manifest_identity": manifest_identity, "pseudo_split_identity": result.get("pseudo_split_identity"),
        "path_resolution": path_resolution,
        "query_count": len(query_entries), "gallery_count": len(gallery.paths),
        "gallery_cache": str(cache_path), "gallery_cache_identity": identity,
        "gallery_paths_unchanged": True, "official_unseen_used": False,
        "text_used_for_inference": False, "full_clip_weights_loaded": True,
        "load_verification": load_info, "mask_policy_version": MASK_POLICY_VERSION,
        "mask_plan": mask_plan, "mask_plan_sha256": canonical_sha256(mask_plan),
        "query_mask_manifest": str(manifest_path), "query_mask_manifest_sha256": sha256_file(manifest_path),
        "mask_artifact_sha256": manifest_rows_hash,
        "clean_mAP_forward": clean_eval.metrics.mean_average_precision,
        "clean_mAP_raw_final_metric": raw_final, "clean_mAP_sanity_abs_delta": clean_delta,
        "clean_replay_tolerance": tolerance,
        "clean_replay_tolerance_policy": "1e-6 CPU; 1e-5 CUDA",
        "clean_P@200": clean_eval.metrics.precision_at_k.get(200),
        "clean_mAP@200_prefix_positive": clean_eval.metrics.mean_average_precision_at_k.get(200),
        "clean_mAP@200_all_relevant": clean_evaluations["all_relevant"].metrics.mean_average_precision_at_k.get(200),
        "clean_mAP@200_min_relevant_k": clean_evaluations["min_relevant_k"].metrics.mean_average_precision_at_k.get(200),
        "fractions": plan["fractions"], "seeds": plan["seeds"], "conditions": conditions,
        "aggregate_status_counts": aggregate_status_counts,
        "no_confidence_intervals": True,
    }
    (output_dir / "masked_view_evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
