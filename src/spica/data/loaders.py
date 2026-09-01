from dataclasses import dataclass
from typing import Literal

from torch.utils.data import DataLoader

from ..config.data import DataConfig, SplitConfig
from .datasets import (
    ImageTransform,
    MultiPositiveRetrievalTrainDataset,
    RetrievalEvalDataset,
    RetrievalTrainDataset,
)
from .manifest import read_class_map, read_manifest

EvalSplit = Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class RetrievalEvalLoaders:
    sketch: DataLoader
    photo: DataLoader
    class_names: dict[int, str]


def _select_split(config: DataConfig, split: EvalSplit) -> SplitConfig:
    if split == "train":
        return config.train
    if split == "test":
        return config.test

    raise ValueError(f"Unsupported split: {split!r}")


def _validate_eval_labels(
    sketch_dataset: RetrievalEvalDataset,
    photo_dataset: RetrievalEvalDataset,
    class_names: dict[int, str],
) -> None:
    sketch_labels = {entry.label for entry in sketch_dataset.entries}
    photo_labels = {entry.label for entry in photo_dataset.entries}
    known_labels = set(class_names)

    unknown_labels = (sketch_labels | photo_labels) - known_labels
    if unknown_labels:
        raise ValueError(
            f"Manifest labels missing from class map: {sorted(unknown_labels)}"
        )

    labels_without_photos = sketch_labels - photo_labels
    if labels_without_photos:
        raise ValueError(
            "Sketch labels without positive gallery photos: "
            f"{sorted(labels_without_photos)}"
        )


def _validate_train_labels(
    dataset: RetrievalTrainDataset,
    class_names: dict[int, str],
) -> None:
    sketch_labels = {entry.label for entry in dataset.sketch_entries}
    photo_labels = {entry.label for entry in dataset.photo_entries}
    known_labels = set(class_names)

    unknown_labels = (sketch_labels | photo_labels) - known_labels
    if unknown_labels:
        raise ValueError(
            f"Training manifest labels missing from class map: {sorted(unknown_labels)}"
        )


def build_retrieval_eval_loaders(
    config: DataConfig,
    transform: ImageTransform,
    *,
    split: EvalSplit = "test",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> RetrievalEvalLoaders:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers}")

    split_config = _select_split(config, split)
    class_names = read_class_map(split_config.class_map)

    sketch_dataset = RetrievalEvalDataset(
        entries=read_manifest(split_config.sketch_manifest, config.root),
        transform=transform,
    )
    photo_dataset = RetrievalEvalDataset(
        entries=read_manifest(split_config.photo_manifest, config.root),
        transform=transform,
    )

    _validate_eval_labels(sketch_dataset, photo_dataset, class_names)

    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }

    return RetrievalEvalLoaders(
        sketch=DataLoader(sketch_dataset, **loader_options),
        photo=DataLoader(photo_dataset, **loader_options),
        class_names=class_names,
    )


def build_retrieval_train_loader(
    config: DataConfig,
    sketch_transform: ImageTransform,
    photo_transform: ImageTransform,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers}")

    split_config = config.train
    class_names = read_class_map(split_config.class_map)

    dataset = RetrievalTrainDataset(
        photo_entries=read_manifest(split_config.photo_manifest, config.root),
        photo_transform=photo_transform,
        sketch_entries=read_manifest(split_config.sketch_manifest, config.root),
        sketch_transform=sketch_transform,
    )

    _validate_train_labels(dataset, class_names)

    loader_options = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    return DataLoader(dataset, **loader_options)


def build_multi_positive_retrieval_train_loader(
    config: DataConfig,
    sketch_transform: ImageTransform,
    photo_transform: ImageTransform,
    *,
    num_positive_photos: int,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    if num_positive_photos <= 0:
        raise ValueError(
            f"num_positive_photos must be positive, got {num_positive_photos}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers}")

    split_config = config.train
    class_names = read_class_map(split_config.class_map)
    dataset = MultiPositiveRetrievalTrainDataset(
        photo_entries=read_manifest(split_config.photo_manifest, config.root),
        photo_transform=photo_transform,
        sketch_entries=read_manifest(split_config.sketch_manifest, config.root),
        sketch_transform=sketch_transform,
        num_positive_photos=num_positive_photos,
    )
    _validate_train_labels(dataset, class_names)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
