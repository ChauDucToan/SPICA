import torch
import torch.nn.functional as F

from spica.models.retrieval import (
    DeterministicK3PhotoPredictor,
    DeterministicK3Prediction,
    deterministic_gate_weighted_barycenter,
    deterministic_k3_multi_positive_retrieval_loss,
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


def test_k3_default_regularizers_are_optional():
    prediction = DeterministicK3Prediction(
        F.normalize(torch.randn(2, 3, 6), dim=-1), torch.randn(2, 3)
    )
    positives = F.normalize(torch.randn(2, 3, 6), dim=-1)
    negative = F.normalize(torch.randn(2, 6), dim=-1)
    assert torch.isfinite(
        deterministic_k3_multi_positive_retrieval_loss(prediction, positives, negative)
    )
