"""Production-loop 3600-campaign smoke: tiny CPU fixture, real trainer code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir

from scripts.check_pairing_pilot_integration_cpu import _fixture, _tiny_clip
import spica.train_frozen_prompt as trainer
import spica.tracking.wandb as wandb_tracking
import spica.evaluation.masked_view as masked_evaluator

ROOT = Path(__file__).parents[1]


class _FakeArtifact:
    def __init__(self, name: str, type: str, **_: object) -> None:
        self.name, self.type = name, type
        self.paths: list[str] = []

    def add_dir(self, path: str) -> None:
        self.paths.append(path)

    def add_file(self, path: str) -> None:
        self.paths.append(path)


class _FakeRun:
    id = "smoke-run-1"
    url = "https://wandb.invalid/smoke-run-1"

    def __init__(self) -> None:
        self.metrics: list[dict[str, object]] = []
        self.defined: list[str] = []
        self.artifacts: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}
        self.finished: list[int] = []

    def define_metric(self, name: str, **_: object) -> None:
        self.defined.append(name)

    def log(self, values: dict[str, object], *, step: int | None = None) -> None:
        self.metrics.append({"step": step, **values})

    def log_artifact(self, artifact: _FakeArtifact, *, aliases: list[str]) -> None:
        self.artifacts.append({"artifact": artifact, "aliases": aliases})

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class _FakeWandb:
    Table = object
    Artifact = _FakeArtifact

    def __init__(self) -> None:
        self.run = _FakeRun()
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs: object) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run


def _args(fixture_root: Path, output: Path, role: str = "C") -> object:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(
            config_name="train_frozen_prompt",
            overrides=[
                f"+experiments=masked_view_3600_{role}",
                f"data_config={fixture_root / 'toy_data.yaml'}",
                f"pairing_manifest_path={fixture_root / 'pairing_manifest.json'}",
                f"experiment_manifest_path={fixture_root / '3600_manifest.json'}",
                "run_kind=smoke", "device=cpu", "pretrained=null",
                "allow_smoke_fixture=true", "synthetic_fixture=true",
                "num_workers=0", "pin_memory=false", "batch_size=2",
                "eval_batch_size=32", "diagnostic_num_seen=3",
                "max_steps=2", "probe_steps=[0,1,2]", "pseudo_val_num_classes=20",
                f"hydra.run.dir={output}",
            ],
        )


@pytest.mark.parametrize("role", ("C", "M"))
def test_production_3600_smoke_records_all_probe_metrics_and_artifacts(tmp_path: Path, role: str) -> None:
    output = tmp_path / "outputsnewgate"
    fixture = _fixture(tmp_path / "fixture")
    args = _args(fixture.root, output, role)
    tiny = _tiny_clip()
    fake = _FakeWandb()

    with (
        patch.object(trainer, "load_frozen_clip", return_value=tiny),
        patch.object(wandb_tracking.wandb, "init", side_effect=fake.init),
        patch.object(wandb_tracking.wandb, "Artifact", _FakeArtifact),
        patch.object(trainer.HydraConfig, "get", return_value=SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output)))),
    ):
        trainer.run(args)

    report = json.loads((output / "run_result.json").read_text())
    assert report["campaign"] == "frozen_prompt_masked_view_3600_2026-09-07"
    assert report["run_kind"] == "smoke"
    assert [row["training_global_step"] for row in report["history"]] == [0, 1, 2]
    assert report["selection"]["best_clean"]["training_global_step"] in {1, 2}
    assert report["selection"]["best_masked"]["training_global_step"] in {1, 2}
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "command.txt").is_file()
    assert (output / "source_snapshot" / "index.json").is_file()

    tracking = fake.run
    assert tracking.finished == [0]
    assert [item["step"] for item in tracking.metrics] == [0, 1, 2]
    assert all(len([key for key in item if key.startswith("masked/fraction_") and "/seed_" in key]) == 45 for item in tracking.metrics[1:])
    assert len(tracking.artifacts) == 2
    assert all(Path(path).is_dir() for item in tracking.artifacts for path in item["artifact"].paths)
    alias_steps = {}
    for step, item in zip((1, 2), tracking.artifacts, strict=True):
        assert f"step{step}" in item["aliases"] and "latest" in item["aliases"]
        for alias in item["aliases"]:
            alias_steps[alias] = step
    assert alias_steps["latest"] == 2
    for name in ("best_clean", "best_masked"):
        assert alias_steps[name] == report["selections"][name]["training_global_step"]
    assert fake.init_kwargs is not None
    assert all("/" not in str(key) for key in fake.init_kwargs["config"])

    checkpoint_dir = output / "checkpoints"
    for alias in ("latest.pt", "best_clean.pt", "best_masked.pt"):
        path = checkpoint_dir / alias
        assert path.is_file() and path.stat().st_size > 0
    for step in (0, 1, 2):
        checkpoint = checkpoint_dir / f"frozen_prompt_step{step}.pt"
        assert checkpoint.is_file() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoints"][str(step)]["checkpoint_sha256"]
    for name in ("latest", "best_clean", "best_masked"):
        selected = report["selections"][name]
        assert hashlib.sha256((checkpoint_dir / f"{name}.pt").read_bytes()).hexdigest() == selected["checkpoint_sha256"]
        with patch.object(masked_evaluator, "load_frozen_clip", return_value=_tiny_clip()):
            replay = masked_evaluator.evaluate_run(
                output / "run_result.json", tmp_path / f"replay_{name}",
                device="cpu", selection=name, allow_smoke=True,
            )
        assert replay["status"] == "COMPLETE"
        assert replay["checkpoint_sha256"] == selected["checkpoint_sha256"]
        assert replay["clean_mAP_sanity_abs_delta"] == 0
        assert len(replay["conditions"]) == 9
        assert replay["clean_mAP@200_prefix_positive"] == selected["scores"]["clean"]["mAP@200_prefix_positive"]
