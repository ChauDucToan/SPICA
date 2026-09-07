from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from spica.data.manifest import ManifestEntry
from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.masked_view import evaluate_benchmark_views
from spica.evaluation.metrics import (
    evaluate_category_retrieval_all_denominators,
)


def _sets() -> tuple[EncodedRetrievalSet, EncodedRetrievalSet]:
    query = EncodedRetrievalSet(torch.tensor([[1.0, 0.0]]), torch.tensor([1]), ("q",))
    gallery = EncodedRetrievalSet(
        torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.4, 0.916515]]),
        torch.tensor([1, 2, 1, 2]),
        ("a", "b", "c", "d"),
    )
    return query, gallery


def test_all_ap200_denominators_share_one_sort(monkeypatch) -> None:
    query, gallery = _sets()
    calls = 0
    original = torch.argsort

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "argsort", counted)
    values = evaluate_category_retrieval_all_denominators(
        query, gallery, precision_at_k=(1,), map_at_k=(2,), top_k=2
    )
    assert calls == 1
    assert values["prefix_positive"].metrics.mean_average_precision_at_k[2] == 1.0
    assert values["all_relevant"].metrics.mean_average_precision_at_k[2] == 0.5
    assert values["min_relevant_k"].metrics.mean_average_precision_at_k[2] == 0.5
    assert values["prefix_positive"].metrics.mean_average_precision == pytest.approx(5 / 6)


class _TinyLiveModel(nn.Module):
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(1)[:, :4]


def test_live_benchmark_returns_nine_conditions_and_restores_rng(tmp_path) -> None:
    path = tmp_path / "query.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(path)
    entries = (ManifestEntry(path, 0),)
    def transform(image):
        return torch.from_numpy(np.asarray(image, dtype="float32")).permute(2, 0, 1) / 255
    model = _TinyLiveModel()
    clean = EncodedRetrievalSet(torch.ones((1, 4)), torch.tensor([0]), (str(path),))
    gallery = EncodedRetrievalSet(
        torch.ones((200, 4)),
        torch.tensor([0] + [1] * 199),
        tuple(f"g{i}" for i in range(200)),
    )
    policy = {
        "version": "ink_centered_square_v1",
        "eval_fractions": [0.25, 0.5, 0.75],
        "eval_seeds": [101, 202, 303],
        "ink_threshold": 0.9,
    }

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    result = evaluate_benchmark_views(
        model, clean, gallery, entries, transform, tmp_path,
        device="cpu", batch_size=1, query_chunk_size=1, mask_policy=policy,
    )
    after = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    assert after[0] == before[0]
    assert np.array_equal(after[1][1], before[1][1])
    assert torch.equal(after[2], before[2])
    assert result["benchmark_status"] == "official_not_verified"
    assert len(result["conditions"]) == 9
    assert set(result["clean"]) >= {
        "full_mAP", "P@200", "mAP@200_prefix_positive",
        "mAP@200_all_relevant", "mAP@200_min_relevant_k",
    }
    assert set(result["masked_by_fraction"]) == {"0.25", "0.5", "0.75"}
    assert result["conditions"][0]["status_counts"]["ok"] == 1
    mismatched = EncodedRetrievalSet(clean.embeddings, torch.tensor([1]), clean.paths)
    with pytest.raises(ValueError, match="paths/labels must match"):
        evaluate_benchmark_views(
            model, mismatched, gallery, entries, transform, tmp_path,
            device="cpu", batch_size=1, query_chunk_size=1, mask_policy=policy,
        )
