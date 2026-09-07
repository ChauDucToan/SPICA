from __future__ import annotations

import pytest
import torch
from PIL import Image

from spica.data.manifest import ManifestEntry
from spica.evaluation.masked_view import (
    _encode_masked_loader,
    _mask_plan,
    _masked_loader,
    _status_counts,
    _manifest_identity_matches,
    _validate_metrics,
)
from spica.evaluation.metrics import evaluate_category_retrieval
from spica.evaluation.embeddings import EncodedRetrievalSet


def test_masked_loader_masks_before_encoder_and_replays_seed(tmp_path) -> None:
    image_path = tmp_path / "query.png"
    Image.new("RGB", (16, 16), (255, 255, 255)).save(image_path)
    entries = (ManifestEntry(image_path, 3),)

    def transform(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(__import__("numpy").asarray(image, dtype="float32")).permute(2, 0, 1) / 255

    loader = _masked_loader(
        entries,
        transform,
        root=tmp_path,
        fraction=0.5,
        seed=101,
        batch_size=1,
        num_workers=0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        ink_threshold=0.9,
    )

    class Recorder:
        device = torch.device("cpu")
        seen: list[torch.Tensor] = []

        def eval(self) -> None:
            pass

        def __call__(self, values: torch.Tensor) -> torch.Tensor:
            self.seen.append(values.clone())
            return values.flatten(1)[:, :4]

    first, first_meta = _encode_masked_loader(Recorder(), loader)
    second, second_meta = _encode_masked_loader(Recorder(), _masked_loader(
        entries,
        transform,
        root=tmp_path,
        fraction=0.5,
        seed=101,
        batch_size=1,
        num_workers=0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        ink_threshold=0.9,
    ))
    assert torch.equal(first.embeddings, second.embeddings)
    assert first_meta == second_meta
    assert first.paths == (str(image_path),)


def test_mask_status_counts_are_separate_and_metadata_uses_ink_pixel_fields(tmp_path) -> None:
    statuses = [
        {"status": "ok", "ink_pixels_before": 4, "ink_pixels_erased": 1, "ink_pixels_after": 3},
        {"status": "blank_input", "ink_pixels_before": 0, "ink_pixels_erased": 0, "ink_pixels_after": 0},
        {"status": "target_unreachable", "ink_pixels_before": 1, "ink_pixels_erased": 0, "ink_pixels_after": 1},
    ]
    assert _status_counts(statuses) == {
        "blank_input": 1, "ok": 1, "target_unreachable": 1, "zero_fraction": 0,
    }


def test_mask_plan_is_taken_from_resolved_config() -> None:
    config = {"mask_policy": {
        "version": "ink_centered_square_v1", "ink_threshold": 0.9,
        "train_fractions": [0.25, 0.5, 0.75], "train_seed": 4242,
        "eval_fractions": [0.25, 0.5, 0.75], "eval_seeds": [101, 202, 303],
    }}
    assert _mask_plan(config)["fractions"] == [0.25, 0.5, 0.75]
    config["mask_policy"]["eval_seeds"] = [101, 202, 304]
    with pytest.raises(ValueError):
        _mask_plan(config)


def test_manifest_identity_keeps_path_spelling_but_rejects_other_changes(tmp_path) -> None:
    target = tmp_path / "pairing.json"
    recorded = {"pairing_manifest_path": str(target.relative_to(tmp_path)), "sha256": "data"}
    reconstructed = {"pairing_manifest_path": str(target), "sha256": "data"}
    assert _manifest_identity_matches(recorded, reconstructed, base=tmp_path)
    assert not _manifest_identity_matches(
        recorded, {**reconstructed, "sha256": "wrong"}, base=tmp_path
    )
    assert not _manifest_identity_matches(
        recorded, {**reconstructed, "pairing_manifest_path": "/other/pairing.json"}, base=tmp_path
    )


def test_retrieval_metric_validation_rejects_nonfinite_per_query_ap() -> None:
    queries = EncodedRetrievalSet(torch.eye(2), torch.tensor([0, 1]), ("q0", "q1"))
    gallery = EncodedRetrievalSet(
        torch.vstack([torch.eye(2) for _ in range(100)]),
        torch.tensor([label for _ in range(100) for label in (0, 1)]),
        tuple(f"g{i}" for i in range(200)),
    )
    evaluation = evaluate_category_retrieval(
        queries, gallery, precision_at_k=(1, 5, 10, 100, 200), map_at_k=(200,), top_k=200
    )
    _validate_metrics(evaluation, name="fixture")
    bad = evaluation.average_precision_per_query.clone()
    bad[0] = float("nan")
    object.__setattr__(evaluation, "average_precision_per_query", bad)
    with pytest.raises(ValueError, match="NaN"):
        _validate_metrics(evaluation, name="fixture")
