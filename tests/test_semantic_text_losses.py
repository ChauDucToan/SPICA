import torch
import pytest

from spica.models.jepa import jepa_ranking_loss, jepa_text_classification_loss
from spica.semantic_text import clone_fixed_text_bank, text_anchor_loss


def _production_batch_loss(parameters: torch.nn.ParameterDict, *, lambda_anchor: float) -> dict[str, torch.Tensor]:
    """Mirror the trainer's per-view loss with production loss helpers."""
    visual = parameters["visual"]
    learned = parameters["context"]
    positive = torch.tensor([[0.9, 0.1, 0.2], [0.2, 0.8, 0.1]])
    negative = torch.tensor([[0.1, 0.2, 0.9], [0.8, 0.1, 0.2]])
    labels = torch.tensor([0, 1])
    class_labels = torch.tensor([0, 1, 2])
    fixed = torch.eye(3)
    views = (visual, visual * 0.75 + 0.1)
    ranks = []
    classifications = []
    for query in views:
        ranks.append(jepa_ranking_loss(query, positive, negative, margin=0.2))
        classification, _ = jepa_text_classification_loss(
            query, learned, class_labels, labels, temperature=0.07,
            detach_text=False,
        )
        classifications.append(classification)
    rank = torch.stack(ranks).mean()
    classification = torch.stack(classifications).mean()
    anchor = text_anchor_loss(learned, fixed)
    total = rank + classification + lambda_anchor * anchor
    return {"rank": rank, "classification": classification, "anchor": anchor, "total": total}


def test_s1_and_s2_lambda_zero_have_exact_production_loss_and_update_parity() -> None:
    torch.manual_seed(7)
    left = torch.nn.ParameterDict({
        "visual": torch.nn.Parameter(torch.randn(2, 3)),
        "context": torch.nn.Parameter(torch.randn(3, 3)),
    })
    right = torch.nn.ParameterDict({
        "visual": torch.nn.Parameter(left["visual"].detach().clone()),
        "context": torch.nn.Parameter(left["context"].detach().clone()),
    })
    left_optimizer = torch.optim.AdamW([
        {"params": [left["visual"]], "lr": 1e-3, "weight_decay": 1e-4},
        {"params": [left["context"]], "lr": 1e-3, "weight_decay": 1e-4},
    ])
    right_optimizer = torch.optim.AdamW([
        {"params": [right["visual"]], "lr": 1e-3, "weight_decay": 1e-4},
        {"params": [right["context"]], "lr": 1e-3, "weight_decay": 1e-4},
    ])
    left_losses = _production_batch_loss(left, lambda_anchor=0.0)
    right_losses = _production_batch_loss(right, lambda_anchor=0.0)
    for name in ("rank", "classification", "anchor", "total"):
        torch.testing.assert_close(left_losses[name], right_losses[name], rtol=0, atol=0)
    left_losses["total"].backward()
    right_losses["total"].backward()
    for name in left:
        torch.testing.assert_close(left[name].grad, right[name].grad, rtol=0, atol=0)
    left_optimizer.step()
    right_optimizer.step()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    left_state = left_optimizer.state_dict()
    right_state = right_optimizer.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for key in left_state["state"]:
        for state_name in left_state["state"][key]:
            torch.testing.assert_close(
                left_state["state"][key][state_name],
                right_state["state"][key][state_name],
                rtol=0,
                atol=0,
            )


def test_anchor_weight_changes_only_total_by_weighted_class_mean() -> None:
    torch.manual_seed(8)
    parameters = torch.nn.ParameterDict({
        "visual": torch.nn.Parameter(torch.randn(2, 3)),
        "context": torch.nn.Parameter(torch.randn(3, 3)),
    })
    unweighted = _production_batch_loss(parameters, lambda_anchor=0.0)
    weighted = _production_batch_loss(parameters, lambda_anchor=1.0)
    torch.testing.assert_close(
        weighted["total"], unweighted["total"] + weighted["anchor"], rtol=0, atol=0
    )


def test_anchor_is_class_mean_not_batch_or_view_scaled() -> None:
    fixed = torch.eye(3)
    learned = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    expected = ((1.0 - 1.0) + (1.0 - 0.0) + (1.0 - 0.0)) / 3.0
    assert torch.allclose(text_anchor_loss(learned, fixed), torch.tensor(expected))


def test_anchor_backpropagates_only_to_learned_bank() -> None:
    fixed = torch.eye(2)
    learned = torch.tensor([[0.8, 0.6], [0.6, 0.8]], requires_grad=True)
    loss = text_anchor_loss(learned, fixed)
    loss.backward()
    assert learned.grad is not None
    assert torch.isfinite(learned.grad).all()
    assert not fixed.requires_grad


def test_inference_tensor_is_cloned_outside_inference_mode() -> None:
    with torch.inference_mode():
        source = torch.ones(2, 3)
    clone = clone_fixed_text_bank(source)
    assert not clone.is_inference()
    assert not clone.requires_grad
    assert torch.equal(clone, source)


def test_anchor_rejects_misaligned_banks() -> None:
    with pytest.raises(ValueError):
        text_anchor_loss(torch.zeros(2, 3), torch.zeros(3, 3))
