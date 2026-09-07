"""CPU-only mathematical checks for the standalone gradient diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "diagnose_masked_view_gradients",
    Path(__file__).parents[1] / "scripts/diagnose_masked_view_gradients.py",
)
assert _SPEC and _SPEC.loader
_DIAG = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DIAG)


def test_gradient_geometry_and_zero_vector_policy() -> None:
    metrics = _DIAG.gradient_metrics(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))
    assert metrics["left_norm"] == 1.0
    assert metrics["right_norm"] == 2.0
    assert metrics["dot"] == 0.0
    assert metrics["cosine"] == 0.0
    zero = _DIAG.gradient_metrics(torch.zeros(2), torch.ones(2))
    assert zero["cosine"] is None
    assert zero["right_over_left_norm"] is None


def test_gradient_linearity_tolerates_roundoff_but_rejects_wrong_components() -> None:
    direct = {"sketch": torch.tensor([10.0, 0.0]), "photo": torch.tensor([1.0])}
    combined = {"sketch": torch.tensor([10.0, 2e-6]), "photo": torch.tensor([1.0])}
    metrics = _DIAG.check_gradient_linearity(direct, combined)
    assert metrics["sketch"]["relative_l2_error"] < 1e-5
    combined["sketch"][0] = 9.0
    with pytest.raises(AssertionError, match="additivity failed"):
        _DIAG.check_gradient_linearity(direct, combined)


def test_component_combination_is_linear_and_does_not_mutate() -> None:
    rank = {"sketch": torch.tensor([1.0, 2.0]), "photo": torch.tensor([3.0])}
    ce = {"sketch": torch.tensor([4.0, 5.0]), "photo": torch.tensor([6.0])}
    before = {key: value.clone() for key, value in rank.items()}
    combined = _DIAG.combine_gradient_components(rank, ce, lambda_rank=2.0, lambda_cls=3.0)
    assert torch.equal(combined["sketch"], torch.tensor([14.0, 19.0]))
    assert torch.equal(combined["photo"], torch.tensor([24.0]))
    assert torch.equal(combined["all"], torch.tensor([14.0, 19.0, 24.0]))
    assert all(torch.equal(rank[key], before[key]) for key in rank)


def test_photo_prompt_has_zero_classification_gradient_and_total_is_linear() -> None:
    sketch = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    photo = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    query = sketch * torch.tensor([2.0, 1.0])
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0])
    logits = query.unsqueeze(0) @ bank.T
    ce = torch.nn.functional.cross_entropy(logits, labels)
    ce_grad = torch.autograd.grad(ce, (sketch, photo), retain_graph=True, allow_unused=True)
    assert ce_grad[1] is None
    rank = (query.square().sum() + photo.square().sum())
    rank_grad = torch.autograd.grad(rank, (sketch, photo), retain_graph=True, allow_unused=True)
    total_grad = torch.autograd.grad(rank + 0.5 * ce, (sketch, photo), allow_unused=True)
    assert torch.allclose(total_grad[0], rank_grad[0] + 0.5 * ce_grad[0])
    assert torch.allclose(total_grad[1], rank_grad[1])


def test_trace_loader_accepts_real_four_batch_step_sequence(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    rows = []
    for global_step in range(4):
        rows.extend({"global_step": global_step, "step": global_step + 1} for _ in range(32))
    trace.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = {"observation_trace": {"path": str(trace), "sha256": _DIAG.sha256_file(trace)}}
    loaded = _DIAG._load_trace(result, tmp_path / "run_result.json", rows_needed=128)
    assert [loaded[index * 32]["global_step"] for index in range(4)] == [0, 1, 2, 3]


def test_mask_input_is_not_mutated() -> None:
    image = torch.ones(3, 8, 8)
    image[:, 2:6, 2:6] = -1
    original = image.clone()
    masked, metadata = _DIAG.apply_ink_mask(image, fraction=0.5, seed=4242)
    assert torch.equal(image, original)
    assert torch.isfinite(masked).all()
    assert metadata["input_sha256"] != metadata["output_sha256"]
