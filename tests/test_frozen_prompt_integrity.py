import json

import pytest
import torch

from spica.evaluation.frozen_prompt import cache_identity, load_prompt_cache, save_prompt_cache
from spica.evaluation.embeddings import EncodedRetrievalSet
from scripts.select_frozen_prompt import select


def _encoded() -> EncodedRetrievalSet:
    return EncodedRetrievalSet(
        embeddings=torch.eye(2),
        labels=torch.tensor([0, 1]),
        paths=("a", "b"),
    )


def test_prompt_cache_rejects_incompatible_identity(tmp_path) -> None:
    identity = cache_identity(
        prompt_checkpoint_hash="abc", prompt_length=3, prompt_mode="prompt_only",
        modality="photo", model_name="tiny", pretrained="openai",
        data_manifest_identity={"sha256": "manifest"},
    )
    path = tmp_path / "photo.pt"
    save_prompt_cache(_encoded(), path, identity=identity)
    assert load_prompt_cache(path, expected_identity=identity).paths == ("a", "b")
    bad = {**identity, "prompt_length": 4}
    with pytest.raises(ValueError, match="identity"):
        load_prompt_cache(path, expected_identity=bad)


def test_selection_rejects_duplicate_and_missing_roles(tmp_path) -> None:
    base = {
        "official_unseen_used_for_selection": False,
        "pseudo_split": {"seed": 3407},
        "manifest_identity": {"sha256": "x"},
        "history": [{"step": 1, "val": {"full_mAP": 0.1}, "checkpoint": str(tmp_path / "x.pt")}],
    }
    checkpoint = tmp_path / "x.pt"
    checkpoint.write_bytes(b"checkpoint")
    files = []
    for role in ("frozen_prompt_FP0", "frozen_prompt_FP1"):
        item = {**base, "experiment_role": role}
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(item))
        files.append(path)
    with pytest.raises(ValueError, match="missing required"):
        select(files, tmp_path / "selected.json")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({**base, "experiment_role": "frozen_prompt_FP0"}))
    with pytest.raises(ValueError, match="duplicate"):
        select(files + [duplicate], tmp_path / "selected.json")
