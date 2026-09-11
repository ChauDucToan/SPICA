"""CPU-only contract tests for the minimal coupled-benchmark runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import open_clip
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_coupled_benchmark_cpu import (  # noqa: E402
    FrozenClipBundle,
    FrozenClipEncoder,
    _Loader,
    _protocol,
    check_exact_checkpoint,
    safe_numpy_globals,
    synthetic_args,
    tiny_model,
)
import spica.runtime as runtime_module  # noqa: E402
import spica.train_coupled_benchmark as trainer  # noqa: E402
import spica.tracking.wandb as wandb_tracking  # noqa: E402


def _synthetic_batch() -> tuple[torch.Tensor, ...]:
    base = synthetic_args()
    return (
        base[0].repeat(16, 1, 1, 1),
        base[1].repeat(16, 1, 1, 1),
        base[2],
        base[3].repeat(16),
        base[4].repeat(16, 1),
        base[5].repeat(16),
        base[6],
        base[7],
    )


class _FakeProbe:
    calls: list[tuple[int, str]] = []
    manifest_hash = "synthetic-probe"
    summary = {
        "scope": "seen_train_probe_not_validation",
        "query_count": 32,
        "gallery_count": 256,
        "manifest_hash": manifest_hash,
        "P@all": {"clean": 0.125, "masked": 0.125},
    }

    @classmethod
    def from_protocol(cls, protocol: Any, transform: Any) -> "_FakeProbe":
        del protocol, transform
        cls.calls = []
        return cls()

    def __call__(self, model: Any, device: torch.device) -> dict[str, Any]:
        del model
        step = len(self.calls) + 1
        self.calls.append((step, str(device)))
        value = float(step) / 10.0
        metrics = {"mAP@200": value, "mAP@all": value + 0.01, "P@200": value + 0.02}
        return {"cleaned": metrics, "masked": dict(metrics), "summary": self.summary}


class _FakeWandbRun:
    def __init__(self, owner: "_FakeWandb", config: dict[str, Any]) -> None:
        self.owner = owner
        self.config = config
        self.id = "synthetic-run"
        self.url = "https://wandb.invalid/synthetic-run"
        self.rows: list[tuple[int | None, dict[str, Any]]] = []
        self.summary_updates: list[dict[str, Any]] = []
        self.definitions: list[tuple[str, str | None]] = []
        self.finished: list[int] = []


class _FakeWandb:
    instances: list[_FakeWandbRun] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.run = _FakeWandbRun(self, dict(kwargs["config"]))
        self.__class__.instances.append(self.run)

    @property
    def run_id(self) -> str:
        return self.run.id

    @property
    def run_url(self) -> str:
        return self.run.url

    def define_metric(self, name: str, *, step_metric: str | None = None, summary: Any = None) -> None:
        del summary
        self.run.definitions.append((name, step_metric))

    def set_summary(self, values: dict[str, Any]) -> None:
        self.run.summary_updates.append(dict(values))

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        self.run.rows.append((step, dict(metrics)))

    def finish(self, exit_code: int = 0) -> None:
        self.run.finished.append(exit_code)


def _patch_synthetic_trainer(monkeypatch: pytest.MonkeyPatch, batch: tuple[torch.Tensor, ...]) -> list[Any]:
    created: list[Any] = []
    loader = _Loader(batch, torch.Generator().manual_seed(42))
    bundle_model = tiny_model(42)
    bundle = FrozenClipBundle(
        FrozenClipEncoder(bundle_model.original_clip),
        lambda value: value,
        open_clip.get_tokenizer("ViT-B-32"),
        "tiny_cpu",
        None,
    )

    def make_model(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        model = tiny_model(42)
        created.append(model)
        return model

    synthetic_data = SimpleNamespace(
        load_benchmark_protocol=lambda *args, **kwargs: _protocol(),
        make_train_loader=lambda *args, **kwargs: loader,
        prepare_batch=lambda protocol, raw, step: {
            "clean": raw[0],
            "corrupted": raw[1],
            "photos": raw[2],
            "positive_indices": raw[3],
            "negative_indices": raw[4],
            "labels": raw[5],
            "photo_labels": raw[6],
            "photo_ids": raw[7],
            "trace": [{"sample": index} for index in range(32)],
            "mask_metadata": {"rows": [{"sample": index, "masked": True} for index in range(32)]},
        },
    )
    provenance = {"source_snapshot": {"manifest": [], "sha256": "synthetic-source"}}
    monkeypatch.setattr(trainer, "DATASETS", {
        "tuberlin_220_30": {"config": "synthetic.yaml", "total_steps": 10, "warmup_steps": 3},
    })
    monkeypatch.setattr(trainer, "benchmark_data", synthetic_data)
    monkeypatch.setattr(trainer, "_verify_clip", lambda: {"path": "synthetic", "sha256": "synthetic-clip"})
    monkeypatch.setattr(trainer, "load_frozen_clip", lambda **kwargs: bundle)
    monkeypatch.setattr(trainer, "CoupledPredictiveModel", make_model)
    monkeypatch.setattr(trainer, "capture_provenance", lambda *args, **kwargs: provenance)
    monkeypatch.setattr(trainer, "_verify_source", lambda *args, **kwargs: "synthetic-source")
    monkeypatch.setattr(trainer, "_copy_source_archive", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(trainer, "TrainingProbe", _FakeProbe)
    _FakeProbe.calls = []
    _FakeWandb.instances = []
    monkeypatch.setattr(trainer, "WandbExperiment", _FakeWandb)
    return created


def test_tiny_cpu_runtime_updates_probe_tracking_and_exact_rolling_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run ten real CPU updates while every external boundary remains synthetic."""
    created = _patch_synthetic_trainer(monkeypatch, _synthetic_batch())
    args = SimpleNamespace(
        dataset="tuberlin_220_30",
        output_dir=str(tmp_path),
        campaign_root=str(tmp_path),
        device="cpu",
        wandb_mode="offline",
        smoke=False,
        runtime={"probe_every": 5, "checkpoint_every": 4},
    )

    saved_steps: list[int] = []
    original_save = trainer._save_run_checkpoint
    def save_and_record(*call_args: Any, **call_kwargs: Any) -> dict[str, Any]:
        saved_steps.append(int(call_kwargs["step"]))
        return original_save(*call_args, **call_kwargs)
    monkeypatch.setattr(trainer, "_save_run_checkpoint", save_and_record)
    result = trainer._train_impl(args, tmp_path)
    assert saved_steps == [4, 8, 10]
    assert result["step"] == result["completed_steps"] == 10
    assert result["horizon_steps"] == 10
    assert result["warmup_steps"] == 3
    assert len(created) == 1
    model = created[0]
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 2604
    assert len([parameter for parameter in model.parameters() if parameter.requires_grad]) == 47
    assert all(not name.startswith("original_clip.") for name in result["initialization_hashes"]["groups"])

    probe_rows = [json.loads(line) for line in (tmp_path / "probe_metrics.jsonl").read_text().splitlines()]
    assert [row["step_train"] for row in probe_rows] == [5, 10]
    assert all(set(row) == {
        "step_train", "cleaned/mAP@200", "cleaned/mAP@all", "cleaned/P@200",
        "masked/mAP@200", "masked/mAP@all", "masked/P@200",
    } for row in probe_rows)
    assert _FakeProbe.calls == [(1, "cpu"), (2, "cpu")]

    run = _FakeWandb.instances[0]
    assert set(run.config) == {
        "arm", "dataset", "seed", "batch_size", "total_steps", "warmup_steps",
        "architecture", "evaluation_query", "source_snapshot_hash", "runtime",
    }
    assert [step for step, _ in run.rows] == [5, 10]
    assert all(set(row) == {
        "step_train", "cleaned/mAP@200", "cleaned/mAP@all", "cleaned/P@200",
        "masked/mAP@200", "masked/mAP@all", "masked/P@200",
    } for _, row in run.rows)
    summary = {key: value for update in run.summary_updates for key, value in update.items()}
    assert summary["probe/cleaned/P@all"] == 0.125
    assert summary["probe/masked/P@all"] == 0.125
    assert not any(key.startswith("probe/") and ("mAP" in key or "P@200" in key) for key in summary)
    assert not hasattr(run, "artifact")

    assert not (tmp_path / "checkpoint_step0.pt").exists()
    assert not list(tmp_path.glob("checkpoint_step*.pt"))
    assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith("checkpoint")) == [
        "checkpoint_latest.json", "checkpoint_latest.pt",
    ]
    checkpoint = tmp_path / "checkpoint_latest.pt"
    metadata = json.loads((tmp_path / "checkpoint_latest.json").read_text())
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["step"] == 10
    assert metadata["sha256"] == checkpoint_sha
    assert result["selections"]["latest"] == {
        "step": 10, "path": "checkpoint_latest.pt", "sha256": checkpoint_sha, "metrics": None,
    }
    assert result["checkpoints"][-1]["sha256"] == checkpoint_sha
    assert not (tmp_path / "embeddings").exists()

    exact = check_exact_checkpoint(checkpoint)
    assert exact == {
        "weights_only": True,
        "model_load_state_dict_strict": True,
        "optimizer_state_exact": True,
        "scheduler_state_exact": True,
    }
    with safe_numpy_globals():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(not name.startswith("original_clip.") for name in payload["model_state_dict"])
    assert not any(name.startswith("photo_model.visual") for name in payload["model_state_dict"])
    assert payload["state_format"] == "trainable_only_v1"
    assert set(payload["model_state_dict"]) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    from spica.models.checkpoint import load_trainable_state
    restored = tiny_model(42)
    load_trainable_state(restored, payload["model_state_dict"])
    assert trainer._state_hash(restored) == trainer._state_hash(model) == payload["model_state_hash"]
    import evaluate_coupled_benchmark as evaluator
    clip_path = tmp_path / "synthetic_clip_id"
    clip_path.write_bytes(b"synthetic_not_pretrained")
    config = payload["resolved_config"]
    config["clip_identity"] = {"path": str(clip_path), "sha256": evaluator._sha256(clip_path)}
    monkeypatch.setattr(evaluator, "load_frozen_clip", trainer.load_frozen_clip)
    monkeypatch.setattr(evaluator, "CoupledPredictiveModel", lambda *a, **k: tiny_model(42))
    loaded, _ = evaluator._build_model(config, payload, torch.device("cpu"),
                                      {0: "cat", 1: "dog"},
                                      expected_original_hash=config["frozen_original_state_hash"])
    assert trainer._state_hash(loaded) == payload["model_state_hash"]
    with pytest.raises(ValueError, match="unsupported checkpoint state_format"):
        evaluator._build_model(config, {**payload, "state_format": "unknown"},
                               torch.device("cpu"), {0: "cat", 1: "dog"},
                               expected_original_hash=config["frozen_original_state_hash"])


def test_checkpoint_save_failure_preserves_existing_bytes_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "checkpoint_latest.pt"
    original = b"previous-checkpoint-bytes"
    path.write_bytes(original)

    def failing_save(value: Any, destination: Path) -> None:
        del value
        destination.write_bytes(b"partial-new-checkpoint")
        raise OSError("synthetic torch.save failure")

    monkeypatch.setattr(trainer.torch, "save", failing_save)
    with pytest.raises(OSError, match="synthetic torch.save failure"):
        trainer._save_checkpoint(path, {"step": 10}, overwrite=True)
    assert path.read_bytes() == original
    assert not path.with_name(path.name + ".tmp").exists()


def test_runtime_policy_rejects_invalid_intervals_and_keeps_minimal_defaults() -> None:
    assert runtime_module.runtime_policy() == {"probe_every": 5, "checkpoint_every": 100}
    assert runtime_module.runtime_policy({"probe_every": 5, "checkpoint_every": 100}) == {
        "probe_every": 5, "checkpoint_every": 100,
    }
    for overrides in (
        {"probe_every": 0},
        {"checkpoint_every": -1},
        {"probe_every": True},
        {"checkpoint_every": 1.5},
    ):
        with pytest.raises(ValueError, match="positive integers"):
            runtime_module.runtime_policy(overrides)
    with pytest.raises(ValueError, match="unknown runtime options"):
        runtime_module.runtime_policy({"hidden_generated_field": 1})


def test_hydra_runtime_group_is_two_fields_without_training() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = compose(config_name="train_coupled_benchmark")
    runtime = OmegaConf.to_container(config.runtime, resolve=True)
    assert runtime == {"probe_every": 5, "checkpoint_every": 100}
    assert set(config.runtime.keys()) == {"probe_every", "checkpoint_every"}


def test_final_publisher_updates_only_final_metrics_and_rejects_legacy_empty_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApiRun:
        def __init__(self, source: str) -> None:
            self.config = {"source_snapshot_hash": source, "arm": "F2_MP_Q_OFFICIAL",
                           "dataset": "tuberlin_220_30", "total_steps": 10}
            self.summary_updates: list[dict[str, Any]] = []
            self.summary = SimpleNamespace(update=self.summary_updates.append)

    class FakeApi:
        run_object: FakeApiRun
        source = "synthetic-source"

        def run(self, path: str) -> FakeApiRun:
            assert path == "a-cctest05187-erd/spica/synthetic-run"
            self.run_object = FakeApiRun(self.source)
            return self.run_object

    api = FakeApi()
    monkeypatch.setattr(wandb_tracking.wandb, "Api", lambda: api)
    result = {
        "wandb_run_id": "synthetic-run", "arm": "F2_MP_Q_OFFICIAL",
        "dataset": "tuberlin_220_30", "source_snapshot_hash": "synthetic-source",
        "step": 10,
        "selections": {"latest": {"sha256": "checkpoint-sha"}},
    }
    report = {
        "clean": {"full_mAP": 0.1, "mAP@200_min_relevant_k": 0.2, "P@200": 0.3},
        "masked_macro": {"full_mAP": 0.4, "mAP@200_min_relevant_k": 0.5, "P@200": 0.6},
    }
    published = wandb_tracking.publish_final_retrieval(result, report, p_all=0.7)
    values = api.run_object.summary_updates[0]
    assert published["status"] == "PASS"
    assert {
        key for key in values if key.startswith("final/cleaned/") or key.startswith("final/masked/")
    } == {
        "final/cleaned/mAP@all", "final/cleaned/mAP@200", "final/cleaned/P@200",
        "final/cleaned/P@all", "final/masked/mAP@all", "final/masked/mAP@200",
        "final/masked/P@200", "final/masked/P@all",
    }
    assert values["final/cleaned/P@all"] == 0.7
    assert values["final/masked/P@all"] == 0.7
    assert "history" not in values and "artifacts" not in values

    api.source = ""
    with pytest.raises(ValueError, match="source/run identity mismatch"):
        wandb_tracking.publish_final_retrieval(result, report, p_all=0.7)
    assert len(api.run_object.summary_updates) == 0
