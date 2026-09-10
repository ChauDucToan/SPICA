"""Independent official TU-Berlin/QuickDraw benchmark data boundary.

This adapter is intentionally separate from :mod:`coupled_training`: it does
not compose the Sketchy protocol, load a pairing manifest, or infer pairs.
Training positives are sampled from every official training photo of the
query's class by ``MultiPositiveRetrievalTrainDataset``.

The adapter validates manifests and class maps before exposing immutable
protocol objects.  Validation stats files, but never opens image content;
image decoding happens only when a caller iterates a loader.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import torch
from torch.utils.data import DataLoader

from ..config.data import DataConfig, load_data_config
from .datasets import (
    ImageTransform,
    MultiPositiveRetrievalTrainDataset,
    RetrievalEvalDataset,
)
from .manifest import ManifestEntry, read_class_map, read_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SplitName = Literal["train", "test"]

# These are byte identities of the checked-in official protocol files.  Keeping
# them here prevents a similarly named local dataset from silently becoming an
# experiment input.
_EXPECTED: dict[str, dict[str, Any]] = {
    "tuberlin_220_30": {
        "config_sha256": "f384563c7fb4387f397a22415f844e4bcfbfb5a5610cf9d4d2e5fe28a234cfe2",
        "root_name": "TUBerlin",
        "counts": {
            "train": {"sketch": 15400, "photo": 176081, "classes": 220},
            "test": {"sketch": 2400, "photo": 27989, "classes": 30},
        },
        "class_ids": {
            "train": tuple(range(220)),
            "test": tuple(range(30)),
        },
        "file_sha256": {
            "train_sketch_manifest": "166e3e84ba2a0911ef64537db42567e34e11b5aec8d9ab6552fc17e06ae28647",
            "train_photo_manifest": "15c52b9bc4212f92f2417e2eab476376b46b4b924560bf712d155ca65914283c",
            "train_class_map": "d28ad9a801bc97a2ee4a7eef6d959b1db063bc31864d12e841147b074f8f83e6",
            "test_sketch_manifest": "52ee2c268ef0ba8dd8b768b669627c291b3eb343a4c484568624bd6d0c758abf",
            "test_photo_manifest": "6a0ea982a6acdf2b8b6a30f2ad2f7d88e8aa4de4c19758e41788600fec907b41",
            "test_class_map": "3cdb4a40907a32844d7414b3dcd5178cdbd71bb1f48eab40b9eac183875e354d",
        },
    },
    "quickdraw_80_30": {
        "config_sha256": "75613a99dacd71a0e41ea8b64b1f73ecabe6d1598aa6fc13950ead04ef7977bb",
        "root_name": "QuickDraw",
        "counts": {
            "train": {"sketch": 236080, "photo": 149428, "classes": 80},
            "test": {"sketch": 92291, "photo": 54151, "classes": 30},
        },
        "class_ids": {
            "train": tuple(range(80)),
            "test": tuple(range(30)),
        },
        "file_sha256": {
            "train_sketch_manifest": "e2772ced09ecc95b3da34c73ff839e7661e1d8c9ccac524959e8b8a37e06549a",
            "train_photo_manifest": "1def0e003bc24138f16091c1ba8cd392c5975010663b0897af37385e7c54cd10",
            "train_class_map": "4ddabc5a664960ebb5fa2715e918b1d4c77a8b46f9068cbebddc5fc9697e92e6",
            "test_sketch_manifest": "d7849dd1836967762e8d23b04d0e4074e1f0be271875f46705ad9b60bdee683a",
            "test_photo_manifest": "89940cae8d6de258cc6080882b882c4b4cbdd5d339689ffd1d8ca4a094d4b2f4",
            "test_class_map": "8cdc391d46e1ce4321d6151bb40aa5da001a0f440cdf1b909d3880763cd5ee18",
        },
    },
}
# Keep the public identity read-only without making callers depend on the
# private validation table.
OFFICIAL_IDENTITIES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        name: MappingProxyType(
            {
                "config_sha256": value["config_sha256"],
                "root_name": value["root_name"],
                "counts": MappingProxyType(
                    {split: MappingProxyType(dict(counts)) for split, counts in value["counts"].items()}
                ),
            }
        )
        for name, value in _EXPECTED.items()
    }
)


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry_sha256(entries: tuple[ManifestEntry, ...], root: Path) -> str:
    digest = hashlib.sha256()
    root = root.resolve()
    for entry in entries:
        try:
            relative = entry.path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"manifest path is outside dataset root: {entry.path}") from error
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(entry.label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _resolve_config(config: str | Path) -> Path:
    aliases = {
        "tuberlin": "configs/data/tuberlin_220_30.yaml",
        "tuberlin_220_30": "configs/data/tuberlin_220_30.yaml",
        "quickdraw": "configs/data/quickdraw_80_30.yaml",
        "quickdraw_80_30": "configs/data/quickdraw_80_30.yaml",
    }
    raw = aliases.get(str(config).lower(), str(config))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _validate_path_entries(
    entries: tuple[ManifestEntry, ...],
    *,
    root: Path,
    modality: str,
    class_names: Mapping[int, str],
    require_files: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    seen: set[Path] = set()
    labels: set[int] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry.label, int) or isinstance(entry.label, bool):
            raise ValueError(f"{modality} entry {index} has a malformed label")
        if entry.label not in class_names:
            raise ValueError(f"{modality} entry {index} has unknown label {entry.label}")
        path = entry.path
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{modality} entry {index} is not a safe relative path: {path}")
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{modality} entry {index} escapes dataset root: {path}") from error
        if resolved in seen:
            raise ValueError(f"duplicate resolved {modality} path: {resolved}")
        seen.add(resolved)
        if require_files:
            try:
                if not resolved.stat().st_mode or not resolved.is_file():
                    raise ValueError(f"{modality} path is not a regular file: {resolved}")
            except OSError as error:
                raise ValueError(f"{modality} path cannot be stat'ed: {resolved}") from error
        labels.add(entry.label)
    if labels != set(class_names):
        missing = sorted(set(class_names) - labels)
        extra = sorted(labels - set(class_names))
        raise ValueError(f"{modality} class coverage mismatch; missing={missing}, extra={extra}")
    return {
        "count": len(entries),
        "class_count": len(labels),
        "class_ids": tuple(sorted(labels)),
        "entry_sha256": _entry_sha256(entries, root),
        "resolved_path_count": len(seen),
        "files_stat_scope": "all_entries",
    }


def _cross_path_overlap(splits: Mapping[str, Mapping[str, tuple[ManifestEntry, ...]]]) -> tuple[str, ...]:
    groups: dict[str, set[Path]] = {}
    for split, modalities in splits.items():
        for modality, entries in modalities.items():
            groups[f"{split}_{modality}"] = {entry.path.resolve() for entry in entries}
    overlaps: list[str] = []
    names = tuple(groups)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            shared = groups[left_name] & groups[right_name]
            if shared:
                overlaps.append(f"{left_name}:{right_name}={len(shared)}")
    return tuple(overlaps)


@dataclass(frozen=True, slots=True)
class BenchmarkSplit:
    name: SplitName
    sketch_manifest: Path
    photo_manifest: Path
    class_map: Path
    class_names: Mapping[int, str]
    sketch_entries: tuple[ManifestEntry, ...]
    photo_entries: tuple[ManifestEntry, ...]
    file_sha256: Mapping[str, str]
    identity: Mapping[str, Any]

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.class_names))


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    name: str
    config_path: Path
    config_sha256: str
    root: Path
    train: BenchmarkSplit
    test: BenchmarkSplit
    selected_split: SplitName
    identity: Mapping[str, Any]

    @property
    def selected(self) -> BenchmarkSplit:
        return self.train if self.selected_split == "train" else self.test

    @property
    def class_ids(self) -> tuple[int, ...]:
        return self.selected.class_ids


def _load_split(
    data: DataConfig,
    split: SplitName,
    expected: Mapping[str, Any],
) -> BenchmarkSplit:
    config = data.train if split == "train" else data.test
    class_names = read_class_map(config.class_map)
    sketch_entries = read_manifest(config.sketch_manifest, data.root)
    photo_entries = read_manifest(config.photo_manifest, data.root)
    expected_counts = expected["counts"][split]
    expected_ids = tuple(expected["class_ids"][split])
    actual_counts = {
        "sketch": len(sketch_entries),
        "photo": len(photo_entries),
        "classes": len(class_names),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"{data.name} {split} counts mismatch: {actual_counts} != {expected_counts}")
    if tuple(sorted(class_names)) != expected_ids:
        raise ValueError(f"{data.name} {split} class IDs mismatch")
    for name, entries in (("sketch", sketch_entries), ("photo", photo_entries)):
        if not entries:
            raise ValueError(f"{data.name} {split} {name} manifest is empty")
    sketch_stats = _validate_path_entries(
        sketch_entries, root=data.root, modality=f"{split} sketch", class_names=class_names
    )
    photo_stats = _validate_path_entries(
        photo_entries, root=data.root, modality=f"{split} photo", class_names=class_names
    )
    file_sha = {
        "sketch_manifest": _sha256_bytes(config.sketch_manifest),
        "photo_manifest": _sha256_bytes(config.photo_manifest),
        "class_map": _sha256_bytes(config.class_map),
    }
    expected_sha = {
        "sketch_manifest": expected["file_sha256"][f"{split}_sketch_manifest"],
        "photo_manifest": expected["file_sha256"][f"{split}_photo_manifest"],
        "class_map": expected["file_sha256"][f"{split}_class_map"],
    }
    if file_sha != expected_sha:
        raise ValueError(f"{data.name} {split} file identity mismatch: {file_sha}")
    identity = {
        "counts": actual_counts,
        "class_ids": expected_ids,
        "class_names_sha256": hashlib.sha256(
            "".join(f"{key}\0{class_names[key]}\n" for key in sorted(class_names)).encode()
        ).hexdigest(),
        "file_sha256": file_sha,
        "entry_identity": {"sketch": sketch_stats, "photo": photo_stats},
    }
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BenchmarkSplit(
        name=split,
        sketch_manifest=config.sketch_manifest,
        photo_manifest=config.photo_manifest,
        class_map=config.class_map,
        class_names=MappingProxyType(dict(class_names)),
        sketch_entries=sketch_entries,
        photo_entries=photo_entries,
        file_sha256=MappingProxyType(file_sha),
        identity=_freeze(identity),
    )


def load_benchmark_protocol(config: str | Path, split: SplitName = "train") -> BenchmarkProtocol:
    """Load and validate one official benchmark, selecting train or test data.

    Both splits are validated, including metadata/path ``stat`` checks.  No
    image is opened by this function; test image decoding is the caller's job.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    config_path = _resolve_config(config)
    if not config_path.is_file():
        raise FileNotFoundError(f"benchmark config not found: {config_path}")
    config_sha = _sha256_bytes(config_path)
    data = load_data_config(config_path)
    if data.name not in _EXPECTED:
        raise ValueError(f"unsupported official benchmark: {data.name}")
    expected = _EXPECTED[data.name]
    if config_sha != expected["config_sha256"]:
        raise ValueError(f"{data.name} config identity mismatch: {config_sha}")
    if data.root.name != expected["root_name"]:
        raise ValueError(f"{data.name} dataset root identity mismatch: {data.root}")
    train = _load_split(data, "train", expected)
    test = _load_split(data, "test", expected)
    semantic_overlap = set(train.class_names.values()) & set(test.class_names.values())
    if semantic_overlap:
        raise ValueError(f"train/test semantic class-name overlap: {sorted(semantic_overlap)}")
    path_overlap = _cross_path_overlap(
        {
            "train": {"sketch": train.sketch_entries, "photo": train.photo_entries},
            "test": {"sketch": test.sketch_entries, "photo": test.photo_entries},
        }
    )
    if path_overlap:
        raise ValueError(f"train/test or cross-modality path overlap: {path_overlap[:3]}")
    identity: dict[str, Any] = {
        "config_sha256": config_sha,
        "train": train.identity,
        "test": test.identity,
        "semantic_class_name_overlap": tuple(sorted(semantic_overlap)),
        "path_overlap": path_overlap,
        "file_existence_scope": "train_and_test_stat_only",
        "image_decode_scope": "none",
    }
    identity["sha256"] = hashlib.sha256(
        json.dumps(_plain(identity), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BenchmarkProtocol(
        name=data.name,
        config_path=config_path,
        config_sha256=config_sha,
        root=data.root,
        train=train,
        test=test,
        selected_split=split,
        identity=_freeze(identity),
    )


def make_train_loader(
    protocol: BenchmarkProtocol,
    transform: ImageTransform,
    *,
    photo_transform: ImageTransform | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Build a B32-compatible full-bank same-class training loader.

    There is deliberately no pairing argument: every positive is sampled from
    all official train photos in its class, and negatives come from another
    official train class.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    from ..train_frozen_prompt import _worker_seed

    photo_transform = transform if photo_transform is None else photo_transform
    dataset = MultiPositiveRetrievalTrainDataset(
        sketch_entries=protocol.train.sketch_entries,
        photo_entries=protocol.train.photo_entries,
        sketch_transform=transform,
        photo_transform=photo_transform,
        num_positive_photos=1,
        positive_pairing=None,
        positive_sampling="same_class",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_worker_seed,
        persistent_workers=num_workers > 0,
    )


def build_test_loaders(
    protocol: BenchmarkProtocol,
    transform: ImageTransform,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Mapping[str, DataLoader]:
    """Construct non-shuffled official test loaders without iterating them."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    return MappingProxyType(
        {
            "sketch": DataLoader(
                RetrievalEvalDataset(protocol.test.sketch_entries, transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=num_workers > 0,
            ),
            "photo": DataLoader(
                RetrievalEvalDataset(protocol.test.photo_entries, transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=num_workers > 0,
            ),
        }
    )


def prepare_batch(protocol: BenchmarkProtocol, batch: Mapping[str, Any], step: int) -> dict[str, object]:
    """Convert a generic coupled batch; this wrapper does not load Sketchy data.

    The shared converter only validates tensors/paths and creates clean/masked
    views.  Protocol loading and sampling remain entirely in this module.
    """
    from .coupled_training import prepare_batch as _generic_prepare_batch

    return _generic_prepare_batch(
        batch,
        data_root=protocol.root,
        step=step,
        classids=protocol.train.class_ids,
    )


__all__ = [
    "BenchmarkProtocol",
    "BenchmarkSplit",
    "OFFICIAL_IDENTITIES",
    "build_test_loaders",
    "load_benchmark_protocol",
    "make_train_loader",
    "prepare_batch",
]
