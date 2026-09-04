from __future__ import annotations

import pytest

from scripts.analyze_frozen_prompt_final import (
    validate_required_split_artifacts,
    validate_split_checkpoint_uniqueness,
    validate_split_identity,
)
from spica.frozen_prompt_artifacts import canonical_sha256


def _split_run(seed: int, checkpoint_hash: str = "a" * 64) -> dict:
    train = [1, 2]
    validation = [3]
    split = {
        "dataset": "test",
        "seed": seed,
        "train_class_ids": train,
        "validation_class_ids": validation,
        "train_sketches": 1,
        "train_photos": 1,
        "validation_sketches": 1,
        "validation_photos": 1,
    }
    split["sha256"] = canonical_sha256(split)
    return {
        "pseudo_validation_seed": seed,
        "training_class_list": train,
        "validation_class_list": validation,
        "pseudo_split_identity": split,
        "class_list_hashes": {
            "train": canonical_sha256(train),
            "validation": canonical_sha256(validation),
        },
        "history": [{"checkpoint_sha256": checkpoint_hash}],
    }


def test_split_artifact_rejects_checkpoint_from_another_evaluation_split() -> None:
    with pytest.raises(ValueError, match="stored split differs"):
        validate_split_identity(_split_run(101), 202)


def test_split_artifact_rejects_overlapping_training_and_validation_classes() -> None:
    run = _split_run(101)
    run["pseudo_split_identity"]["validation_class_ids"] = [2]
    run["pseudo_split_identity"]["sha256"] = canonical_sha256(
        {key: value for key, value in run["pseudo_split_identity"].items() if key != "sha256"}
    )
    run["validation_class_list"] = [2]
    run["class_list_hashes"]["validation"] = canonical_sha256([2])
    with pytest.raises(ValueError, match="overlap"):
        validate_split_identity(run, 101)


def test_split_artifact_rejects_checkpoint_reuse_across_split_seeds() -> None:
    with pytest.raises(ValueError, match="reused"):
        validate_split_checkpoint_uniqueness([_split_run(101), _split_run(202)])


def test_split_robustness_requires_retrained_split_artifacts() -> None:
    with pytest.raises(ValueError, match="split-specific training artifacts"):
        validate_required_split_artifacts({}, "frozen_prompt_final_FP3")
