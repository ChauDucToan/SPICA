"""Data-only trust boundary for coupled predictive V1 training.

This module owns protocol composition, manifest/pairing validation, deterministic
loader construction, and conversion of one real loader batch into CPU tensors.
It deliberately does not construct a model, compute a loss, or move tensors to
an accelerator.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from .. import train_frozen_prompt as _legacy
from .coupled_views import region_pair

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"

EXPECTED_SPLIT_SHA256 = "3e02604d2ed315aa254d4264ec440a7e50233c7c9b175be224519feafda88425"
EXPECTED_PAIRING_SHA256 = "545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c"
EXPECTED_CLIP_SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
EXPECTED_CLIP_BYTES = 605143284
EXPECTED_FULL_PHOTO_POOL_SHA256 = "b0dd18492aa9a634ebb04f83ea9f328b8d926301a48886176f5d450533b23973"
CLIP_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai"
    / "snapshots/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors"
)

EXPECTED_COUNTS = {
    "train_class_count": 84,
    "validation_class_count": 20,
    "train_sketches": 46624,
    "train_photos": 58950,
    "validation_sketches": 10963,
    "validation_photos": 13999,
}
EXPECTED_TRAIN_CLASSES = tuple(
    value
    for value in range(104)
    if value not in {7, 13, 16, 27, 28, 31, 33, 34, 39, 42, 45, 51, 52, 53, 60, 65, 75, 86, 90, 99}
)
EXPECTED_VALIDATION_CLASSES = (7, 13, 16, 27, 28, 31, 33, 34, 39, 42, 45, 51, 52, 53, 60, 65, 75, 86, 90, 99)


def _arm_protocol(arm: str, campaign_id: str, diagnostic: object) -> dict[str, str]:
    """Resolve arm routing before device, cache, or data side effects."""
    protocols = {
        "R0": {"architecture": "pooled", "method_version": "coupled_predictive_v1", "positive_pool": "canonical"},
        "R1": {"architecture": "predictive", "method_version": "coupled_predictive_v1", "positive_pool": "canonical"},
        "R1_SIG": {"architecture": "predictive", "method_version": "coupled_predictive_v1", "positive_pool": "canonical"},
        "F2": {"architecture": "predictive_fusion_v2", "method_version": "coupled_predictive_fusion_v2", "positive_pool": "full"},
        "F2_SIG": {"architecture": "predictive_fusion_v2", "method_version": "coupled_predictive_fusion_v2", "positive_pool": "full"},
        "F2_MP": {"architecture": "predictive_fusion_v2", "method_version": "coupled_predictive_fusion_mp_v1", "positive_pool": "full", "main_photo_objective": "multi_positive_supervised_contrastive"},
    }
    if arm not in protocols:
        raise ValueError("arm must be R0, R1, R1_SIG, F2, F2_SIG, or F2_MP")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    if arm in {"F2", "F2_SIG"} and campaign_id != "coupled_predictive_fusion_v2":
        raise ValueError("F2 requires campaign_id='coupled_predictive_fusion_v2'")
    if arm == "F2_MP" and campaign_id != "coupled_predictive_fusion_mp_v1":
        raise ValueError("F2_MP requires campaign_id='coupled_predictive_fusion_mp_v1'")
    if arm in {"F2", "F2_MP"} and diagnostic is not None:
        raise ValueError(f"{arm} does not accept --diagnostic")
    if arm in {"R1_SIG", "F2_SIG"} and not diagnostic:
        raise ValueError(f"{arm} requires --diagnostic")
    return dict(protocols[arm])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_clip_cache() -> dict[str, object]:
    """Verify the already-present CLIP cache; never download or resolve a model."""
    if not CLIP_PATH.is_file():
        raise FileNotFoundError(f"verified local CLIP cache is missing: {CLIP_PATH}")
    size = CLIP_PATH.stat().st_size
    sha256 = _sha256(CLIP_PATH)
    if size != EXPECTED_CLIP_BYTES or sha256 != EXPECTED_CLIP_SHA256:
        raise ValueError(
            f"CLIP cache identity mismatch: bytes={size}, sha256={sha256}"
        )
    return {"path": str(CLIP_PATH), "bytes": size, "sha256": sha256}


def _resolved_config() -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        args = compose(
            config_name="train_frozen_prompt",
            overrides=[
                "+experiments=semantic_text_S0",
                "device=cuda",
                "tracking.mode=disabled",
            ],
        )
    OmegaConf.resolve(args)
    return args


def _validate_protocol_config(args: DictConfig) -> None:
    checks = {
        "role": str(args.experiment_role) == "semantic_text_S0",
        "campaign": str(args.experiment_campaign) == "frozen_prompt_semantic_text_step1_2026-09-07",
        "device": str(args.device) == "cuda",
        "tracking": str(args.tracking.mode) == "disabled",
        "batch_size": int(args.batch_size) == 32,
        "num_workers": int(args.num_workers) == 4,
        "drop_last": bool(args.drop_last),
        "shuffle_protocol": str(args.positive_sampling) == "same_class",
        "seed": int(args.seed) == 42,
        "pseudo_val_seed": int(args.pseudo_val_seed) == 3407,
        "pseudo_val_classes": int(args.pseudo_val_num_classes) == 20,
        "data_config": str(args.data_config) == "configs/data/sketchy_104_21.yaml",
        "pairing_path": str(args.pairing_manifest_path)
        == "outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.json",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"coupled V1 protocol config failed: {failed}")


def _validate_split(split: Any, names: Mapping[int, str], identity: Mapping[str, Any]) -> None:
    if identity.get("sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"pseudo split identity mismatch: {identity.get('sha256')}")
    if tuple(split.train_class_ids) != EXPECTED_TRAIN_CLASSES:
        raise ValueError("pseudo-train class IDs differ from the approved protocol")
    if tuple(split.validation_class_ids) != EXPECTED_VALIDATION_CLASSES:
        raise ValueError("pseudo-validation class IDs differ from the approved protocol")
    if set(names) != set(range(104)):
        raise ValueError("class map must contain exactly the 104 protocol classes")
    actual = {
        "train_class_count": len(split.train_class_ids),
        "validation_class_count": len(split.validation_class_ids),
        "train_sketches": len(split.train_sketch_entries),
        "train_photos": len(split.train_photo_entries),
        "validation_sketches": len(split.validation_sketch_entries),
        "validation_photos": len(split.validation_photo_entries),
    }
    if actual != EXPECTED_COUNTS:
        raise ValueError(f"pseudo split counts differ from protocol: {actual}")


def load_protocol_data() -> dict[str, object]:
    """Compose and validate the approved data protocol without loading a model."""
    args = _resolved_config()
    _validate_protocol_config(args)
    data = _legacy.load_data_config(_legacy._path(args.data_config))
    split, names, split_identity, manifest_identity = _legacy._load_split(data, args)
    _validate_split(split, names, split_identity)

    pairing_path = _legacy._path(args.pairing_manifest_path)
    if _sha256(pairing_path) != EXPECTED_PAIRING_SHA256:
        raise ValueError("pairing manifest hash does not match the approved protocol")
    mapping = _legacy.load_pairing_manifest(
        pairing_path,
        dataset_root=data.root,
        sketch_entries=split.train_sketch_entries,
        photo_entries=split.train_photo_entries,
    )
    pool_paths = {str(entry.path.resolve()) for entry in mapping.values()}
    if len(mapping) != EXPECTED_COUNTS["train_sketches"] or len(pool_paths) != 8400:
        raise ValueError(
            f"pairing coverage mismatch: records={len(mapping)}, pool={len(pool_paths)}"
        )
    pairing = {
        "mapping": mapping,
        "pool_paths": pool_paths,
        "sha256": EXPECTED_PAIRING_SHA256,
        "records": len(mapping),
        "unique_photo_pool": len(pool_paths),
    }
    return {
        "args": args,
        "data": data,
        "names": names,
        "split": split,
        "pairing": pairing,
        "split_identity": split_identity,
        "manifest_identity": manifest_identity,
    }


def positive_pool_identity(context: Mapping[str, object], positive_pool: str) -> dict[str, object]:
    """Validate the active positive pool at the data boundary, not in model/loss math."""
    if positive_pool == "canonical":
        pairing = context["pairing"]
        return {"policy": "canonical", "count": int(pairing["unique_photo_pool"]), "sha256": pairing["sha256"]}
    if positive_pool != "full":
        raise ValueError("positive_pool must be 'canonical' or 'full'")
    train_photo = context["manifest_identity"].get("entry_identity", {}).get("train_photo", {})
    if int(train_photo.get("count", -1)) != EXPECTED_COUNTS["train_photos"]:
        raise ValueError("full positive pool count differs from the approved protocol")
    if train_photo.get("sha256") != EXPECTED_FULL_PHOTO_POOL_SHA256:
        raise ValueError("full positive pool identity differs from the approved ordered photo pool")
    return {"policy": "full", "count": EXPECTED_COUNTS["train_photos"], "sha256": EXPECTED_FULL_PHOTO_POOL_SHA256}


def make_train_loader(
    context: Mapping[str, object],
    transform: Any,
    *,
    positive_pool: str = "canonical",
) -> DataLoader:
    """Build a deterministic loader from the approved positive-pool policy."""
    if positive_pool not in {"canonical", "full"}:
        raise ValueError("positive_pool must be 'canonical' or 'full'")
    args = context["args"]
    split = context["split"]
    pairing = context["pairing"]
    if not isinstance(args, DictConfig):
        raise TypeError("context['args'] must be the composed DictConfig")
    if not isinstance(pairing, Mapping) or not hasattr(split, "train_sketch_entries"):
        raise TypeError("context is missing protocol split/pairing")
    if positive_pool == "full":
        if str(args.positive_sampling) != "same_class":
            raise ValueError("full positive_pool requires positive_sampling='same_class'")
        positive_pool_identity(context, positive_pool)
    return _legacy._loader(
        (split.train_sketch_entries, split.train_photo_entries),
        transform,
        args,
        train=True,
        seed=42,
        positive_pairing=None if positive_pool == "full" else pairing["mapping"],
    )


def _positive_paths(batch: Mapping[str, Any], index: int, batch_size: int) -> tuple[str, ...]:
    values = batch["positive_photo_paths"]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("positive_photo_paths has an invalid collated shape")
    # Default collate transposes [B, 1] paths into [(path_0, ..., path_B)].
    if len(values) == 1:
        row: Any = [values[0][index]]
    elif len(values) == batch_size:
        row = values[index]
    else:
        raise ValueError("positive_photo_paths has an invalid collated shape")
    if isinstance(row, str):
        return (row,)
    if not isinstance(row, (list, tuple)):
        row = (row,)
    return tuple(str(value) for value in row)


def _relative_path(raw: object, root: Path, field: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty path")
    raw_path = Path(raw).expanduser()
    if ".." in raw_path.parts:
        raise ValueError(f"{field} must not contain '..': {raw!r}")
    resolved = raw_path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{field} is outside data_root: {raw!r}") from error
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError(f"{field} is not a canonical relative path")
    if not resolved.is_file():
        raise ValueError(f"{field} does not exist: {resolved}")
    return relative, resolved


def _check_image_batch(name: str, image: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(image, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if image.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if tuple(image.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(image.shape)}, expected {shape}")
    if not image.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f"{name} contains non-finite values")


def _check_labels(name: str, values: Tensor, batch_size: int, known: set[int]) -> None:
    if not isinstance(values, Tensor) or values.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if values.dtype != torch.long or tuple(values.shape) != (batch_size,):
        raise TypeError(f"{name} must be a CPU torch.long tensor of shape [B]")
    if any(int(value) not in known for value in values.tolist()):
        raise ValueError(f"{name} contains an unknown class ID")


def prepare_batch(
    batch: Mapping[str, Any], *, data_root: Path, step: int, classids: Sequence[int]
) -> dict[str, object]:
    """Validate one collated real batch and create clean/masked CPU views.

    The returned photo bank is ordered by all positive rows followed by all
    negative rows, with first-occurrence deduplication.  No model or loss is
    involved; clean labels are trusted supervised annotations, not inferred.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    known_ids = tuple(int(value) for value in classids)
    if not known_ids or known_ids != tuple(sorted(set(known_ids))):
        raise ValueError("classids must be non-empty, sorted, and unique")
    known = set(known_ids)
    required = {
        "sketch", "positive_photos", "negative_photo", "label", "negative_label",
        "sketch_path", "positive_photo_paths", "negative_photo_path",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"batch is missing fields: {sorted(missing)}")

    sketches = batch["sketch"]
    positives = batch["positive_photos"]
    negatives = batch["negative_photo"]
    labels = batch["label"]
    negative_labels = batch["negative_label"]
    if not isinstance(sketches, Tensor) or sketches.ndim != 4 or sketches.shape[1] != 3:
        raise ValueError("sketch must have shape [B, 3, H, W]")
    if sketches.device.type != "cpu":
        raise ValueError("sketch must be on CPU")
    batch_size, _, height, width = map(int, sketches.shape)
    if batch_size <= 0:
        raise ValueError("batch must not be empty")
    _check_image_batch("sketch", sketches, (batch_size, 3, height, width))
    if not isinstance(positives, Tensor) or tuple(positives.shape) != (batch_size, 1, 3, height, width):
        raise ValueError("positive_photos must have shape [B, 1, 3, H, W]")
    if not isinstance(negatives, Tensor) or tuple(negatives.shape) != (batch_size, 3, height, width):
        raise ValueError("negative_photo must have shape [B, 3, H, W]")
    _check_image_batch("positive_photos", positives, tuple(positives.shape))
    _check_image_batch("negative_photo", negatives, tuple(negatives.shape))
    if sketches.dtype != positives.dtype or sketches.dtype != negatives.dtype:
        raise TypeError("sketch and photo tensors must have the same dtype")
    _check_labels("label", labels, batch_size, known)
    _check_labels("negative_label", negative_labels, batch_size, known)
    if any(int(a) == int(b) for a, b in zip(labels.tolist(), negative_labels.tolist())):
        raise ValueError("negative photo labels must differ from query labels")

    query_paths_raw = batch["sketch_path"]
    negative_paths_raw = batch["negative_photo_path"]
    if not isinstance(query_paths_raw, (list, tuple)) or not isinstance(negative_paths_raw, (list, tuple)):
        raise ValueError("collated path fields must be sequences")
    if len(query_paths_raw) != batch_size or len(negative_paths_raw) != batch_size:
        raise ValueError("collated path fields must have one path per row")

    query_ids: list[str] = []
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    query_abs: list[Path] = []
    for index in range(batch_size):
        query_id, query_path = _relative_path(query_paths_raw[index], data_root, "sketch_path")
        positive_paths = _positive_paths(batch, index, batch_size)
        if len(positive_paths) != 1:
            raise ValueError("V1 training requires exactly one positive photo")
        positive_id, positive_path = _relative_path(positive_paths[0], data_root, "positive_photo_path")
        negative_id, negative_path = _relative_path(negative_paths_raw[index], data_root, "negative_photo_path")
        if Path(query_id).parent.name != Path(positive_id).parent.name:
            raise ValueError("positive photo category does not match query category")
        if Path(query_id).parent.name == Path(negative_id).parent.name:
            raise ValueError("negative photo category matches query category")
        query_ids.append(query_id)
        positive_ids.append(positive_id)
        negative_ids.append(negative_id)
        query_abs.append(query_path)
        # The resolved values are intentionally checked here, even though only
        # relative IDs enter the model-facing trace.
        del positive_path, negative_path

    ordered_ids: list[str] = []
    ordered_tensors: list[Tensor] = []
    ordered_labels: list[int] = []
    by_id: dict[str, int] = {}
    positive_indices: list[int] = []
    negative_indices: list[int] = []

    def add(photo_id: str, tensor: Tensor, label: int) -> int:
        previous = by_id.get(photo_id)
        if previous is not None:
            if ordered_labels[previous] != label or not torch.equal(ordered_tensors[previous], tensor):
                raise ValueError(f"duplicate photo identity has conflicting label or pixels: {photo_id}")
            return previous
        by_id[photo_id] = len(ordered_ids)
        ordered_ids.append(photo_id)
        ordered_tensors.append(tensor.detach().clone())
        ordered_labels.append(label)
        return len(ordered_ids) - 1

    # Keep this order stable: every positive row, then every negative row.
    for index, photo_id in enumerate(positive_ids):
        positive_indices.append(add(photo_id, positives[index, 0], int(labels[index])))
    for index, photo_id in enumerate(negative_ids):
        negative_indices.append(add(photo_id, negatives[index], int(negative_labels[index])))
    if set(range(len(ordered_ids))) != set(positive_indices) | set(negative_indices):
        raise AssertionError("a packed photo-bank row is unused")

    clean_rows: list[Tensor] = []
    masked_rows: list[Tensor] = []
    metadata_rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for index in range(batch_size):
        clean, masked, metadata = region_pair(sketches[index], query_ids[index], step)
        clean_rows.append(clean)
        masked_rows.append(masked)
        trace = {
            "zero_based_step": step,
            "step": step,
            "index": index,
            "label": int(labels[index]),
            "negative_label": int(negative_labels[index]),
            "query_relative_path": query_ids[index],
            "positive_photo_id": positive_ids[index],
            "negative_photo_id": negative_ids[index],
            # Explicit aliases make the receipt readable without retaining abs paths.
            "sketch_relative": query_ids[index],
            "positive_photo_path": positive_ids[index],
            "negative_photo_path": negative_ids[index],
        }
        traces.append(trace)
        row = dict(metadata)
        row.update(
            {
                "index": index,
                "query_relative_path": query_ids[index],
                "positive_photo_id": positive_ids[index],
                "negative_photo_id": negative_ids[index],
                "label": int(labels[index]),
                "negative_label": int(negative_labels[index]),
                "masked_status": metadata.get("status"),
            }
        )
        metadata_rows.append(row)

    statuses = Counter(str(row["status"]) for row in metadata_rows)
    return {
        "clean": torch.stack(clean_rows),
        "corrupted": torch.stack(masked_rows),
        "photos": torch.stack(ordered_tensors),
        "positive_indices": torch.tensor(positive_indices, dtype=torch.long),
        "negative_indices": torch.tensor(negative_indices, dtype=torch.long).view(batch_size, 1),
        "labels": labels.detach().clone(),
        "photo_labels": torch.tensor(ordered_labels, dtype=torch.long),
        "photo_ids": tuple(ordered_ids),
        "trace": traces,
        "mask_metadata": {
            "count": batch_size,
            "status_counts": dict(sorted(statuses.items())),
            "input_sha256": [str(row["input_sha256"]) for row in metadata_rows],
            "output_sha256": [str(row["output_sha256"]) for row in metadata_rows],
            "rows": metadata_rows,
            "metas": metadata_rows,
        },
    }


__all__ = [
    "CLIP_PATH",
    "EXPECTED_CLIP_SHA256",
    "EXPECTED_PAIRING_SHA256",
    "EXPECTED_SPLIT_SHA256",
    "_arm_protocol",
    "load_protocol_data",
    "make_train_loader",
    "positive_pool_identity",
    "prepare_batch",
    "verify_clip_cache",
]
