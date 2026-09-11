from __future__ import annotations

from pathlib import Path

import pytest

from src.spica.tracking import wandb as tracking


_METRICS = {
    "full_mAP": 0.8,
    "P@200": 0.4,
    "mAP@200_min_relevant_k": 0.2,
}


class FakeRun:
    id = "run-1"
    url = "https://wandb.invalid/run-1"

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, object], int | None]] = []
        self.defined: list[tuple[str, str | None, object]] = []
        self.summary: dict[str, object] = {}
        self.artifacts: list[tuple[object, list[str]]] = []
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

    def log_artifact(self, artifact: object, *, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class FakeTable:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeArtifact:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def add_dir(self, path: str) -> None:
        pass

    def add_file(self, path: str) -> None:
        pass


class FakeSettings:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeWandb:
    Settings = FakeSettings
    Table = FakeTable
    Artifact = FakeArtifact

    def __init__(self, run: FakeRun) -> None:
        self.run = run
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs: object) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run


def _experiment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_artifacts: bool = False,
) -> tuple[tracking.WandbExperiment, FakeRun, FakeWandb]:
    run = FakeRun()
    fake_wandb = FakeWandb(run)
    monkeypatch.setattr(tracking, "wandb", fake_wandb)
    experiment = tracking.WandbExperiment(
        project="fixed-project",
        config={"checkpoint": str(Path("/private/checkpoint.pt")), "api_key": "do-not-transform"},
        allow_artifacts=allow_artifacts,
    )
    return experiment, run, fake_wandb


def test_retrieval_probe_uses_one_explicit_training_step(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment, run, fake_wandb = _experiment(monkeypatch)

    experiment.define_metric("cleaned/*", step_metric="step_train", summary="max")
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
    settings = fake_wandb.init_kwargs["settings"]
    assert isinstance(settings, FakeSettings)
    assert settings.kwargs == {
        "x_disable_meta": True,
        "x_disable_machine_info": True,
        "x_disable_stats": True,
        "disable_code": True,
        "disable_git": True,
        "x_save_requirements": False,
        "console": "off",
    }
    assert run.defined == [("cleaned/*", "step_train", "max")]
    assert run.summary == {"probe_status": "complete"}
    assert len(run.logs) == 1
    logged, step = run.logs[-1]
    assert step == 1800
    assert logged["step_train"] == 1800
    assert logged["cleaned/mAP@200"] == 0.2
    assert logged["cleaned/mAP@all"] == 0.8
    assert logged["cleaned/P@200"] == 0.4
    assert logged["masked/mAP@200"] == 0.2
    assert logged["masked/mAP@all"] == 0.8
    assert logged["masked/P@200"] == 0.4
    assert set(logged) == {
        "step_train", "cleaned/mAP@200", "cleaned/mAP@all", "cleaned/P@200",
        "masked/mAP@200", "masked/mAP@all", "masked/P@200",
    }
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in logged.values())
    assert run.finished == [0]


def test_metrics_filter_train_values_and_validate_allowed_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment, run, _ = _experiment(monkeypatch)
    experiment.log_metrics({"step_train": 2, "train/loss": 1.0, "cleaned/mAP@all": 0.5})
    assert run.logs == [({"step_train": 2, "cleaned/mAP@all": 0.5}, None)]
    experiment.define_metric("train/*")
    assert run.defined == []


def test_artifact_and_table_upload_require_explicit_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    experiment, run, fake_wandb = _experiment(monkeypatch)
    experiment.log_table("ignored", columns=["x"], rows=[[1]])
    experiment.log_artifact(tmp_path, name="ignored", artifact_type="model")
    assert run.logs == []
    assert run.artifacts == []


def test_artifact_and_table_upload_can_be_explicitly_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    experiment, run, _ = _experiment(monkeypatch, allow_artifacts=True)
    artifact_path = tmp_path / "checkpoint.pt"
    artifact_path.write_bytes(b"checkpoint")
    experiment.log_table("table", columns=["x"], rows=[[1]])
    experiment.log_artifact(artifact_path, name="checkpoint", artifact_type="model")
    assert len(run.logs) == 1
    assert len(run.artifacts) == 1


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
