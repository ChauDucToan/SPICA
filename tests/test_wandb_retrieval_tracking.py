from __future__ import annotations

from pathlib import Path

import pytest

from src.spica.tracking import wandb as tracking


_METRICS = {
    "full_mAP": 0.8,
    "P@200": 0.4,
    "mAP@200_prefix_positive": 0.5,
    "mAP@200_all_relevant": 0.3,
    "mAP@200_min_relevant_k": 0.2,
}


class FakeRun:
    id = "run-1"
    url = "https://wandb.invalid/run-1"

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, object], int | None]] = []
        self.defined: list[tuple[str, str | None, object]] = []
        self.summary: dict[str, object] = {}
        self.finished: list[int] = []

    def log(self, values: dict[str, object], *, step: int | None = None) -> None:
        self.logs.append((values, step))

    def define_metric(
        self,
        name: str,
        *,
        step_metric: str | None = None,
        summary: object = None,
    ) -> None:
        self.defined.append((name, step_metric, summary))

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class FakeWandb:
    def __init__(self, run: FakeRun) -> None:
        self.run = run
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs: object) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run


def _experiment(monkeypatch: pytest.MonkeyPatch) -> tuple[tracking.WandbExperiment, FakeRun, FakeWandb]:
    run = FakeRun()
    fake_wandb = FakeWandb(run)
    monkeypatch.setattr(tracking, "wandb", fake_wandb)
    experiment = tracking.WandbExperiment(
        project="fixed-project",
        config={"checkpoint": str(Path("/private/checkpoint.pt")), "api_key": "do-not-transform"},
    )
    return experiment, run, fake_wandb


def test_retrieval_probe_uses_one_explicit_training_step(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment, run, fake_wandb = _experiment(monkeypatch)

    experiment.define_metric("clean/*", step_metric="step_train", summary="max")
    experiment.set_summary({"probe_status": "complete"})
    experiment.log_retrieval_probe(
        1800,
        _METRICS,
        _METRICS,
        {0.25: _METRICS, 0.50: _METRICS, 0.75: _METRICS},
        conditions=[{**_METRICS, "fraction": 0.25, "seed": 101}],
        diagnostics={"text/anchor_loss": 0.01, "train/text_gradient_norm": 0.5},
    )
    experiment.finish(exit_code=0)

    assert fake_wandb.init_kwargs is not None
    assert fake_wandb.init_kwargs["config"] == {
        "checkpoint": str(Path("/private/checkpoint.pt")),
        "api_key": "do-not-transform",
    }
    assert run.defined == [("clean/*", "step_train", "max")]
    assert run.summary == {"probe_status": "complete"}
    assert len(run.logs) == 1
    logged, step = run.logs[-1]
    assert logged["text/anchor_loss"] == 0.01
    assert logged["train/text_gradient_norm"] == 0.5
    assert step == 1800
    assert logged["step_train"] == 1800
    assert logged["clean/mAP@200_prefix_positive"] == 0.5
    assert logged["masked/macro/mAP@200_all_relevant"] == 0.3
    assert logged["masked/fraction_025/mAP@200_min_relevant_k"] == 0.2
    assert logged["masked/fraction_050/full_mAP"] == 0.8
    assert logged["masked/fraction_075/P@200"] == 0.4
    assert logged["masked/fraction_025/seed_101/full_mAP"] == 0.8
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in logged.values())
    assert run.finished == [0]


def test_retrieval_probe_rejects_unknown_nested_or_nonfinite_values(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment, run, _ = _experiment(monkeypatch)
    del run

    with pytest.raises(ValueError, match="unknown retrieval metric"):
        experiment.log_retrieval_probe(
            1,
            {"raw_history": [1, 2]},
            _METRICS,
            {0.25: _METRICS, 0.50: _METRICS, 0.75: _METRICS},
        )
    with pytest.raises(TypeError, match="finite scalar"):
        experiment.log_retrieval_probe(
            1,
            {**_METRICS, "full_mAP": True},
            _METRICS,
            {0.25: _METRICS, 0.50: _METRICS, 0.75: _METRICS},
        )
    with pytest.raises(ValueError, match="finite"):
        experiment.log_retrieval_probe(
            1,
            {**_METRICS, "full_mAP": float("nan")},
            _METRICS,
            {0.25: _METRICS, 0.50: _METRICS, 0.75: _METRICS},
        )
