from __future__ import annotations

import statistics

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

import spica.train_alignment as train_alignment
from spica.data.samplers import MatchedClassBatchSampler
from spica.models.alignment import AlignmentLoss


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sketch_prompt = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.photo_prompt = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.register_buffer("running", torch.tensor([5.0]))


class _ToyTextBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context = nn.Parameter(torch.tensor([1.0]))


class _ToyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "value": torch.tensor(float(index + 1)),
            "label": torch.tensor(index % 2),
        }


def _args() -> object:
    return OmegaConf.create(
        {
            "calibration_target_ratio": 0.1,
            "calibration_batches": 2,
            "lambda_rank": 1.0,
            "lambda_cls": 1.0,
        }
    )


def _objective(
    zero_detached: bool = False, zero_base: bool = False
):
    def objective(
        model: _ToyModel,
        text_bank: _ToyTextBank,
        hard_text_values: torch.Tensor,
        hard_text_labels: torch.Tensor,
        batch: dict[str, torch.Tensor],
        args: object,
        device: torch.device,
        *,
        alignment_mean_weight: float | None = None,
        alignment_covariance_weight: float | None = None,
        alignment_target_gradient: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AlignmentLoss | None]:
        del text_bank, hard_text_values, hard_text_labels, args, device
        with torch.no_grad():
            model.running.add_(1.0)
        value = batch["value"].mean()
        rank = (
            model.sketch_prompt.detach().sum() * 0.0
            if zero_base
            else (model.sketch_prompt * value).sum()
        )
        cls = (model.photo_prompt * value * 0.25).sum()
        if not alignment_mean_weight:
            return rank, cls, value.new_zeros(()), None
        if zero_detached and alignment_target_gradient == "detached":
            mean = model.photo_prompt.detach().sum() * 0.0
        else:
            target = (
                model.photo_prompt.detach()
                if alignment_target_gradient == "detached"
                else model.photo_prompt
            )
            mean = (model.sketch_prompt - target).square().sum()
        return rank, cls, value.new_zeros(()), AlignmentLoss(
            mean, mean, mean, 1, 2, 4
        )

    return objective


def _loader() -> tuple[DataLoader, MatchedClassBatchSampler, torch.Generator]:
    sampler = MatchedClassBatchSampler(
        [0, 0, 1, 1, 0, 0, 1, 1],
        classes_per_batch=2,
        samples_per_class=2,
        seed=7,
        batches_per_epoch=4,
    )
    generator = torch.Generator().manual_seed(11)
    return DataLoader(_ToyDataset(), batch_sampler=sampler, generator=generator), sampler, generator


def test_calibration_separates_policies_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_alignment, "_batch_objectives", _objective())
    model = _ToyModel()
    text_bank = _ToyTextBank()
    model.train(False)
    text_bank.train(True)
    model.sketch_prompt.grad = torch.tensor([7.0, 8.0])
    sampler_loader, sampler, generator = _loader()
    sampler._epoch = 3
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    before_text = {name: value.clone() for name, value in text_bank.state_dict().items()}
    before_rng = train_alignment.capture_rng_state(generator)
    before_grads = {
        parameter: None if parameter.grad is None else parameter.grad.clone()
        for parameter in (*model.parameters(), *text_bank.parameters())
    }

    payload = train_alignment._calibrate_mean_alignment(
        model,
        text_bank,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        sampler_loader,
        sampler,
        generator,
        _args(),  # type: ignore[arg-type]
        torch.device("cpu"),
    )

    assert payload["schema_version"] == 2
    assert "weighted_photo_gradient_ratios_if_symmetric" not in payload
    assert all(value == 0.0 for value in payload["detached"]["photo_gradient_norms"])
    assert all(value > 0.0 for value in payload["symmetric"]["photo_gradient_norms"])
    assert torch.allclose(
        torch.tensor(payload["detached"]["sketch_gradient_norms"]),
        torch.tensor(payload["symmetric"]["sketch_gradient_norms"]),
        atol=1e-6,
    )
    raw = [
        base / detached
        for base, detached in zip(
            payload["base"]["sketch_gradient_norms"],
            payload["detached"]["sketch_gradient_norms"],
            strict=True,
        )
    ]
    expected_lambda = 0.1 * statistics.median(raw)
    assert payload["calibration"]["lambda_alignment_mean"] == pytest.approx(
        expected_lambda
    )
    assert payload["calibration"]["state_restoration_verified"] is True

    assert model.training is False
    assert text_bank.training is True
    assert sampler._epoch == 3
    assert train_alignment._rng_matches(
        before_rng, train_alignment.capture_rng_state(generator)
    )
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before_model.items())
    assert all(torch.equal(text_bank.state_dict()[name], value) for name, value in before_text.items())
    assert all(
        (parameter.grad is None and gradient is None)
        or (parameter.grad is not None and gradient is not None and torch.equal(parameter.grad, gradient))
        for parameter, gradient in before_grads.items()
    )


def test_calibration_zero_base_sketch_denominator_is_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_alignment, "_batch_objectives", _objective(zero_base=True))
    model = _ToyModel()
    text_bank = _ToyTextBank()
    loader, sampler, generator = _loader()

    payload = train_alignment._calibrate_mean_alignment(
        model,
        text_bank,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        loader,
        sampler,
        generator,
        _args(),  # type: ignore[arg-type]
        torch.device("cpu"),
    )

    assert payload["status"] == "UNVERIFIED"
    assert payload["calibration"]["lambda_selection_status"] == (
        "UNVERIFIED_ZERO_DENOMINATOR"
    )


def test_calibration_zero_sketch_denominator_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_alignment, "_batch_objectives", _objective(True))
    model = _ToyModel()
    text_bank = _ToyTextBank()
    loader, sampler, generator = _loader()

    payload = train_alignment._calibrate_mean_alignment(
        model,
        text_bank,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        loader,
        sampler,
        generator,
        _args(),  # type: ignore[arg-type]
        torch.device("cpu"),
    )

    assert payload["status"] == "UNVERIFIED"
    assert payload["calibration"]["lambda_alignment_mean"] is None
    assert payload["detached"]["unweighted_sketch_ratios"] == [None, None]
    assert payload["detached"]["unweighted_sketch_ratio_reasons"] == [
        "detached_sketch_gradient_zero",
        "detached_sketch_gradient_zero",
    ]
