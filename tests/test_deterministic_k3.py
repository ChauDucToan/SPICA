import pytest
import torch
import torch.nn.functional as F

from spica.data.manifest import ManifestEntry
from spica.evaluate_deterministic import _validate_cache_against_manifest
from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.models.retrieval import (
    DeterministicK3PhotoPredictor,
    DeterministicK3Prediction,
    deterministic_angular_positive_assignment_loss,
    deterministic_gate_weighted_barycenter,
    deterministic_k3_multi_positive_retrieval_loss,
    deterministic_single_direction_multi_positive_retrieval_loss,
)


def test_k3_predictor_has_only_directions_and_gates():
    model = DeterministicK3PhotoPredictor(8, 12)
    output = model(F.normalize(torch.randn(4, 8), dim=-1))
    assert output.directions.shape == (4, 3, 8)
    assert output.gate_logits.shape == (4, 3)
    assert torch.allclose(output.directions.norm(dim=-1), torch.ones(4, 3), atol=1e-6)
    assert not hasattr(output, "concentrations")


def test_k3_loss_is_gate_weighted_barycenter_for_every_positive():
    directions = torch.zeros(1, 3, 4)
    directions[0, 0, 0] = 1
    directions[0, 1, 1] = 1
    directions[0, 2, 2] = 1
    positives = torch.tensor([[[1.0, 0, 0, 0], [0, 1, 0, 0]]])
    negative = torch.tensor([[0.0, 0, 0, 1.0]])
    prediction = DeterministicK3Prediction(directions, torch.tensor([[4.0, 0.0, 0.0]]))
    barycenter = deterministic_gate_weighted_barycenter(
        directions, prediction.gate_logits
    )
    expected = F.softplus(
        0.2 - positives[0] @ barycenter[0] + (negative[0] @ barycenter[0])
    ).mean()
    actual = deterministic_k3_multi_positive_retrieval_loss(
        prediction, positives, negative
    )
    assert torch.allclose(actual, expected)


def test_angular_assignment_keeps_prior_outside_temperature():
    directions = torch.zeros(1, 3, 4)
    directions[0, 0, 0] = 1
    directions[0, 1, 1] = 1
    directions[0, 2, 2] = 1
    positives = torch.tensor([[[1.0, 0, 0, 0], [0, 1, 0, 0]]])
    negative = torch.tensor([[0.0, 0, 0, 1.0]])
    prior_logits = torch.tensor([[0.8, 0.1, 0.1]]).log()
    prediction = DeterministicK3Prediction(directions, prior_logits)

    result = deterministic_angular_positive_assignment_loss(
        prediction,
        positives,
        negative,
        assignment_temperature=0.05,
    )
    cosine = torch.einsum("bkd,bmd->bmk", directions, positives)
    expected = (prior_logits[:, None, :] + cosine / 0.05).softmax(dim=-1)
    assert torch.allclose(result.assignment_responsibilities, expected)
    expected_positive = (expected * cosine).sum(dim=-1)
    negative_cosine = torch.einsum("bkd,bd->bk", directions, negative)
    expected_negative = (expected * negative_cosine[:, None, :]).sum(dim=-1)
    expected_loss = torch.nn.functional.softplus(
        0.2 - expected_positive + expected_negative
    ).mean()
    assert torch.allclose(result.total, expected_loss)


def test_angular_assignment_trains_direction_and_gate_heads():
    torch.manual_seed(24)
    model = DeterministicK3PhotoPredictor(8, 12, initial_dominant_weight=0.8)
    sketches = F.normalize(torch.randn(5, 8), dim=-1)
    positives = F.normalize(torch.randn(5, 3, 8), dim=-1)
    negatives = F.normalize(torch.randn(5, 8), dim=-1)
    result = deterministic_angular_positive_assignment_loss(
        model(sketches), positives, negatives, assignment_temperature=0.05
    )
    result.total.backward()
    for head in (model.direction_head, model.gate_head):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_single_direction_multi_positive_loss_pairs_each_negative():
    predicted = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    positives = torch.tensor(
        [
            [[1.0, 0.0], [0.8, 0.6]],
            [[0.0, 1.0], [0.6, 0.8]],
        ]
    )
    negatives = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = F.softplus(
        0.2
        - torch.einsum("bd,bmd->bm", predicted, positives)
        + (predicted * negatives).sum(dim=-1)[:, None]
    ).mean()
    actual = deterministic_single_direction_multi_positive_retrieval_loss(
        predicted, positives, negatives
    )
    assert torch.allclose(actual, expected)


def test_k3_default_regularizers_are_optional():
    prediction = DeterministicK3Prediction(
        F.normalize(torch.randn(2, 3, 6), dim=-1), torch.randn(2, 3)
    )
    positives = F.normalize(torch.randn(2, 3, 6), dim=-1)
    negative = F.normalize(torch.randn(2, 6), dim=-1)
    assert torch.isfinite(
        deterministic_k3_multi_positive_retrieval_loss(prediction, positives, negative)
    )


def test_evaluation_cache_identity_matches_manifest_exactly(tmp_path):
    entries = (
        ManifestEntry(tmp_path / "first.png", 2),
        ManifestEntry(tmp_path / "second.png", 7),
    )
    encoded = EncodedRetrievalSet(
        embeddings=torch.zeros(2, 4),
        labels=torch.tensor([2, 7]),
        paths=tuple(str(entry.path) for entry in entries),
    )
    identity = _validate_cache_against_manifest(
        modality="sketch", encoded_set=encoded, manifest_entries=entries
    )
    assert len(identity) == 64


def test_evaluation_cache_rejects_manifest_order_or_label_mismatch(tmp_path):
    entries = (
        ManifestEntry(tmp_path / "first.png", 2),
        ManifestEntry(tmp_path / "second.png", 7),
    )
    encoded = EncodedRetrievalSet(
        embeddings=torch.zeros(2, 4),
        labels=torch.tensor([7, 2]),
        paths=(str(entries[1].path), str(entries[0].path)),
    )
    with pytest.raises(ValueError, match="does not match"):
        _validate_cache_against_manifest(
            modality="photo", encoded_set=encoded, manifest_entries=entries
        )
