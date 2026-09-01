"""Deterministic class-level splits for zero-shot validation diagnostics."""

from dataclasses import dataclass
import random
from collections.abc import Mapping, Sequence

from .manifest import ManifestEntry


@dataclass(frozen=True, slots=True)
class ClasswiseRetrievalSplit:
    train_class_ids: tuple[int, ...]
    validation_class_ids: tuple[int, ...]
    train_sketch_entries: tuple[ManifestEntry, ...]
    train_photo_entries: tuple[ManifestEntry, ...]
    validation_sketch_entries: tuple[ManifestEntry, ...]
    validation_photo_entries: tuple[ManifestEntry, ...]
    seed: int


def _partition_entries(
    entries: Sequence[ManifestEntry],
    class_ids: set[int],
) -> tuple[ManifestEntry, ...]:
    return tuple(entry for entry in entries if entry.label in class_ids)


def make_classwise_retrieval_split(
    sketch_entries: Sequence[ManifestEntry],
    photo_entries: Sequence[ManifestEntry],
    class_names: Mapping[int, str],
    *,
    num_validation_classes: int = 20,
    seed: int = 3407,
) -> ClasswiseRetrievalSplit:
    """Make a reproducible pseudo-unseen split from seen training classes.

    Class IDs are sampled, never individual examples.  The returned validation
    classes are absent from the training entries and can therefore be used for
    early stopping without using the official unseen test classes.
    """
    if not sketch_entries or not photo_entries:
        raise ValueError("Both sketch and photo entries must be non-empty")
    if not class_names:
        raise ValueError("class_names must be non-empty")
    if num_validation_classes <= 0:
        raise ValueError("num_validation_classes must be positive")
    if num_validation_classes >= len(class_names):
        raise ValueError(
            "num_validation_classes must be smaller than the number of classes"
        )

    observed_sketch = {entry.label for entry in sketch_entries}
    observed_photo = {entry.label for entry in photo_entries}
    available = sorted(set(class_names) & observed_sketch & observed_photo)
    if len(available) != len(class_names):
        missing = sorted(set(class_names) - set(available))
        raise ValueError(
            "Every class in the class map must have sketch and photo entries; "
            f"missing {missing}"
        )

    generator = random.Random(seed)
    validation = tuple(
        sorted(generator.sample(available, num_validation_classes))
    )
    validation_set = set(validation)
    training = tuple(class_id for class_id in available if class_id not in validation_set)
    training_set = set(training)

    return ClasswiseRetrievalSplit(
        train_class_ids=training,
        validation_class_ids=validation,
        train_sketch_entries=_partition_entries(sketch_entries, training_set),
        train_photo_entries=_partition_entries(photo_entries, training_set),
        validation_sketch_entries=_partition_entries(sketch_entries, validation_set),
        validation_photo_entries=_partition_entries(photo_entries, validation_set),
        seed=seed,
    )
