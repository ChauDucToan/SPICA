import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import ManifestEntry

ImageTransform = Callable[[Image.Image], Tensor]


def _load_rgb_image(entry: ManifestEntry) -> Image.Image:
    try:
        with Image.open(entry.path) as image:
            rgb_image = image.convert("RGB")
    except OSError as error:
        raise OSError(f"Could not read image: {entry.path}") from error

    return rgb_image


class EvalSample(TypedDict):
    image: Tensor
    label: int
    path: str


class TrainSample(TypedDict):
    sketch: Tensor  # [3, H, W]
    positive_photo: Tensor  # [3, H, W]
    negative_photo: Tensor  # [3, H, W]
    label: int
    negative_label: int
    sketch_path: str
    positive_photo_path: str
    negative_photo_path: str


class MultiPositiveTrainSample(TypedDict):
    sketch: Tensor  # [3, H, W]
    positive_photos: Tensor  # [num_positives, 3, H, W]
    negative_photo: Tensor  # [3, H, W]
    label: int
    negative_label: int
    sketch_path: str
    positive_photo_paths: tuple[str, ...]
    negative_photo_path: str


class RetrievalEvalDataset(Dataset[EvalSample]):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        transform: ImageTransform,
    ) -> None:
        if not entries:
            raise ValueError("Evaluation dataset cannot be empty")

        self.entries = tuple(entries)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> EvalSample:
        entry = self.entries[index]

        rgb_image = _load_rgb_image(entry)

        return {
            "image": self.transform(rgb_image),
            "label": entry.label,
            "path": str(entry.path),
        }


class RetrievalTrainDataset(Dataset[TrainSample]):
    def __init__(
        self,
        sketch_entries: Sequence[ManifestEntry],
        photo_entries: Sequence[ManifestEntry],
        sketch_transform: ImageTransform,
        photo_transform: ImageTransform,
    ) -> None:
        if not sketch_entries:
            raise ValueError("Training sketch entries cannot be empty")

        if not photo_entries:
            raise ValueError("Training photo entries cannot be empty")

        self.sketch_entries = tuple(sketch_entries)
        self.photo_entries = tuple(photo_entries)
        self.sketch_transform = sketch_transform
        self.photo_transform = photo_transform

        photos_by_label: dict[int, list[ManifestEntry]] = {}
        for entry in self.photo_entries:
            photos_by_label.setdefault(entry.label, []).append(entry)

        self._photos_by_label = {
            label: tuple(entries) for label, entries in photos_by_label.items()
        }

        sketch_labels = {entry.label for entry in self.sketch_entries}
        photo_labels = set(photos_by_label.keys())
        missing_positive_labels = sketch_labels - photo_labels
        if missing_positive_labels:
            raise ValueError(
                "Sketch labels without positive training photos: "
                f"{sorted(missing_positive_labels)}"
            )
        if len(photo_labels) < 2:
            raise ValueError(
                "Negative sampling requires photos from at least two classes"
            )

        self._photo_labels = tuple(sorted(photo_labels))
        self._negative_labels_by_label = {
            label: self._photo_labels[:i] + self._photo_labels[i + 1 :]
            for i, label in enumerate(self._photo_labels)
        }

    def __len__(self) -> int:
        return len(self.sketch_entries)

    def __getitem__(self, index: int) -> TrainSample:
        sketch_entry = self.sketch_entries[index]

        pos_entry = self._sample_positive(sketch_entry.label)
        neg_entry = self._sample_negative(sketch_entry.label)

        sketch_img = _load_rgb_image(sketch_entry)
        pos_img = _load_rgb_image(pos_entry)
        neg_img = _load_rgb_image(neg_entry)

        sketch_tensor = self.sketch_transform(sketch_img)
        pos_tensor = self.photo_transform(pos_img)
        neg_tensor = self.photo_transform(neg_img)

        return TrainSample(
            # Tensor
            sketch=sketch_tensor,
            positive_photo=pos_tensor,
            negative_photo=neg_tensor,
            # Label
            label=sketch_entry.label,
            negative_label=neg_entry.label,
            # Path
            sketch_path=str(sketch_entry.path),
            positive_photo_path=str(pos_entry.path),
            negative_photo_path=str(neg_entry.path),
        )

    def _sample_positive(self, label: int) -> ManifestEntry:
        positive_photos = self._photos_by_label[label]
        return random.choice(positive_photos)

    def _sample_negative(self, label: int) -> ManifestEntry:
        negative_labels = self._negative_labels_by_label[label]
        negative_label = random.choice(negative_labels)
        return random.choice(self._photos_by_label[negative_label])


class MultiPositiveRetrievalTrainDataset(RetrievalTrainDataset):
    def __init__(
        self,
        sketch_entries: Sequence[ManifestEntry],
        photo_entries: Sequence[ManifestEntry],
        sketch_transform: ImageTransform,
        photo_transform: ImageTransform,
        *,
        num_positive_photos: int,
        positive_pairing: Mapping[str, ManifestEntry] | None = None,
        positive_sampling: str = "same_class",
    ) -> None:
        if num_positive_photos <= 0:
            raise ValueError(
                f"num_positive_photos must be positive, got {num_positive_photos}"
            )
        if positive_sampling not in {"same_class", "paired"}:
            raise ValueError(
                "positive_sampling must be 'same_class' or 'paired', got "
                f"{positive_sampling!r}"
            )
        if positive_sampling == "paired" and positive_pairing is None:
            raise ValueError("positive_sampling='paired' requires positive_pairing")
        if positive_pairing is not None and num_positive_photos != 1:
            raise ValueError(
                "positive_pairing requires num_positive_photos=1"
            )

        super().__init__(
            sketch_entries=sketch_entries,
            photo_entries=photo_entries,
            sketch_transform=sketch_transform,
            photo_transform=photo_transform,
        )
        self.num_positive_photos = num_positive_photos
        self.positive_sampling = positive_sampling
        self.positive_pairing = self._validate_positive_pairing(positive_pairing)
        if self.positive_pairing is None:
            self._positive_photos_by_label = self._photos_by_label
        else:
            unique_photos: dict[int, dict[Path, ManifestEntry]] = {}
            for entry in self.positive_pairing.values():
                unique_photos.setdefault(entry.label, {})[entry.path.resolve()] = entry
            self._positive_photos_by_label = {
                label: tuple(by_path.values())
                for label, by_path in unique_photos.items()
            }

    def __getitem__(self, index: int) -> MultiPositiveTrainSample:
        sketch_entry = self.sketch_entries[index]
        positive_entries = self._sample_positives(
            sketch_entry.label, str(sketch_entry.path.resolve())
        )
        negative_entry = self._sample_negative(sketch_entry.label)

        sketch_tensor = self.sketch_transform(_load_rgb_image(sketch_entry))
        positive_tensors = torch.stack(
            [
                self.photo_transform(_load_rgb_image(entry))
                for entry in positive_entries
            ],
            dim=0,
        )
        negative_tensor = self.photo_transform(_load_rgb_image(negative_entry))
        return MultiPositiveTrainSample(
            sketch=sketch_tensor,
            positive_photos=positive_tensors,
            negative_photo=negative_tensor,
            label=sketch_entry.label,
            negative_label=negative_entry.label,
            sketch_path=str(sketch_entry.path),
            positive_photo_paths=tuple(str(entry.path) for entry in positive_entries),
            negative_photo_path=str(negative_entry.path),
        )

    def _validate_positive_pairing(
        self,
        pairing: Mapping[str, ManifestEntry] | None,
    ) -> dict[str, ManifestEntry] | None:
        if pairing is None:
            return None
        sketches = {str(entry.path.resolve()): entry for entry in self.sketch_entries}
        photos = {str(entry.path.resolve()): entry for entry in self.photo_entries}
        if len(sketches) != len(self.sketch_entries):
            raise ValueError("Sketch entries contain duplicate resolved paths")
        if len(photos) != len(self.photo_entries):
            raise ValueError("Photo entries contain duplicate resolved paths")
        normalized = {str(Path(key).resolve()): value for key, value in pairing.items()}
        if set(normalized) != set(sketches):
            raise ValueError("positive_pairing must cover exactly the sketch entries")
        for sketch_path, photo in normalized.items():
            sketch = sketches[sketch_path]
            photo_path = str(photo.path.resolve())
            if photo_path not in photos:
                raise ValueError(f"Paired photo is not in photo entries: {photo.path}")
            if sketch.label != photo.label or photos[photo_path].label != photo.label:
                raise ValueError(f"Paired labels differ for {sketch_path}")
        return normalized

    def _sample_positives(self, label: int, sketch_path: str | None = None) -> tuple[ManifestEntry, ...]:
        positive_photos = self._positive_photos_by_label[label]
        if self.positive_pairing is not None:
            # Both modes consume one identical draw before choosing their target.
            sampled = random.choice(positive_photos)
            if self.positive_sampling == "paired":
                assert sketch_path is not None
                return (self.positive_pairing[sketch_path],)
            return (sampled,)
        if len(positive_photos) >= self.num_positive_photos:
            return tuple(random.sample(positive_photos, self.num_positive_photos))
        return tuple(
            random.choice(positive_photos) for _ in range(self.num_positive_photos)
        )
