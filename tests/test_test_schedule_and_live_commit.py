from __future__ import annotations

import pytest

from spica.runtime import official_test_steps, runtime_policy
from spica.tracking import wandb as tracking


@pytest.mark.parametrize(
    ("total_steps", "expected"),
    [
        (4446, [890, 1779, 2668, 3557, 4446]),
        (1189, [238, 476, 714, 952, 1189]),
        (18229, [3646, 7292, 10938, 14584, 18229]),
        (3, [1, 2, 3]),
    ],
)
def test_official_test_steps_are_ceil_percent_boundaries(total_steps: int, expected: list[int]) -> None:
    assert official_test_steps(total_steps) == expected


def test_official_runtime_is_distinct_and_rejects_legacy_mixing() -> None:
    assert runtime_policy({"test_every_percent": 20, "checkpoint_every": 100}) == {
        "test_every_percent": 20,
        "checkpoint_every": 100,
    }
    with pytest.raises(ValueError, match="cannot be used together"):
        runtime_policy({"probe_every": 5, "test_every_percent": 20})
    with pytest.raises(ValueError, match="test_every_percent=20"):
        runtime_policy({"test_every_percent": 10})
    with pytest.raises(ValueError, match="unknown runtime options"):
        runtime_policy({"test_every": 1000})


class _Run:
    id = "run"
    url = None

    def __init__(self) -> None:
        self.logs: list[tuple[dict, int | None, bool]] = []

    def log(self, values: dict, *, step: int | None = None, commit: bool = False) -> None:
        self.logs.append((values, step, commit))

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def finish(self, *, exit_code: int) -> None:
        pass


class _Settings:
    def __init__(self, **kwargs) -> None:
        pass


class _Wandb:
    Settings = _Settings

    def __init__(self, run: _Run) -> None:
        self.run = run

    def init(self, **kwargs):
        return self.run


def test_retrieval_logs_commit_at_the_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run()
    monkeypatch.setattr(tracking, "wandb", _Wandb(run))
    experiment = tracking.WandbExperiment(project="test")
    metrics = {"full_mAP": 0.8, "P@200": 0.4, "mAP@200_min_relevant_k": 0.2}

    experiment.log_metrics({"step_train": 3, "cleaned/mAP@all": 0.8}, step=3)
    experiment.log_retrieval_probe(4, metrics, metrics, {})

    assert [(step, commit) for _, step, commit in run.logs] == [(3, True), (4, True)]
