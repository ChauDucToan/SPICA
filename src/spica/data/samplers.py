"""Deterministic samplers for matched class-conditional training batches."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence

from torch.utils.data import Sampler


class MatchedClassBatchSampler(Sampler[list[int]]):
    """Yield batches with a fixed number of sketches from each sampled class."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int = 0,
        batches_per_epoch: int | None = None,
    ) -> None:
        if not labels:
            raise ValueError("labels cannot be empty")
        if classes_per_batch <= 0 or samples_per_class <= 0:
            raise ValueError("classes_per_batch and samples_per_class must be positive")
        by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            by_class[int(label)].append(index)
        if len(by_class) < classes_per_batch:
            raise ValueError(
                "classes_per_batch cannot exceed the number of distinct labels"
            )
        self._indices_by_class = {
            label: tuple(indices) for label, indices in sorted(by_class.items())
        }
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        batch_size = classes_per_batch * samples_per_class
        self.batches_per_epoch = (
            max(1, math.ceil(len(labels) / batch_size))
            if batches_per_epoch is None
            else batches_per_epoch
        )
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        self._epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        classes = list(self._indices_by_class)
        class_cursor = 0
        class_order: list[int] = []
        sample_orders = {
            label: list(indices) for label, indices in self._indices_by_class.items()
        }
        sample_cursors = dict.fromkeys(classes, 0)

        def next_classes() -> list[int]:
            nonlocal class_cursor, class_order
            if class_cursor + self.classes_per_batch > len(class_order):
                class_order = classes.copy()
                rng.shuffle(class_order)
                class_cursor = 0
            selected = class_order[class_cursor : class_cursor + self.classes_per_batch]
            class_cursor += self.classes_per_batch
            return selected

        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for label in next_classes():
                order = sample_orders[label]
                cursor = sample_cursors[label]
                if cursor + self.samples_per_class > len(order):
                    order = list(self._indices_by_class[label])
                    rng.shuffle(order)
                    sample_orders[label] = order
                    cursor = 0
                selected = order[cursor : cursor + self.samples_per_class]
                sample_cursors[label] = cursor + self.samples_per_class
                if len(selected) < self.samples_per_class:
                    selected.extend(
                        rng.choice(self._indices_by_class[label])
                        for _ in range(self.samples_per_class - len(selected))
                    )
                batch.extend(selected)
            yield batch
        self._epoch += 1
