from collections import Counter

from spica.data.samplers import MatchedClassBatchSampler


def test_matched_sampler_has_fixed_class_multiplicity_and_is_reproducible() -> None:
    labels = [label for label in range(5) for _ in range(6)]
    first = MatchedClassBatchSampler(
        labels,
        classes_per_batch=3,
        samples_per_class=2,
        seed=41,
        batches_per_epoch=4,
    )
    second = MatchedClassBatchSampler(
        labels,
        classes_per_batch=3,
        samples_per_class=2,
        seed=41,
        batches_per_epoch=4,
    )
    first_batches = list(first)
    second_batches = list(second)
    assert first_batches == second_batches
    for batch in first_batches:
        counts = Counter(labels[index] for index in batch)
        assert len(batch) == 6
        assert all(count == 2 for count in counts.values())


def test_matched_sampler_epoch_changes_order() -> None:
    sampler = MatchedClassBatchSampler(
        [label for label in range(4) for _ in range(4)],
        classes_per_batch=2,
        samples_per_class=2,
        seed=1,
        batches_per_epoch=2,
    )
    first = list(sampler)
    second = list(sampler)
    assert first != second
