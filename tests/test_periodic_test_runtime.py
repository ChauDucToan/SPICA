from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from test_minimal_runtime import _patch_synthetic_trainer, _synthetic_batch
import spica.train_coupled_benchmark as trainer


def test_periodic_official_test_boundary_is_post_checkpoint_and_final_once(monkeypatch, tmp_path: Path) -> None:
    _patch_synthetic_trainer(monkeypatch, _synthetic_batch())
    monkeypatch.setattr(trainer, "DATASETS", {
        "tuberlin_220_30": {"config": "synthetic.yaml", "total_steps": 12, "warmup_steps": 3},
    })
    calls: list[tuple[int, str, bool, bool, bool]] = []

    def evaluate(model: Any, protocol: Any, transform: Any, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        del protocol, transform
        output_dir.mkdir(parents=True)
        step = int(kwargs["checkpoint_selection"]["step"])
        before = torch.random.get_rng_state()
        assert model.training
        assert all(parameter.grad is None for parameter in model.parameters())
        assert (output_dir.parent.parent / "checkpoint_latest.pt").is_file()
        calls.append((step, str(output_dir), bool(kwargs["final"]), model.training, torch.equal(before, torch.random.get_rng_state())))
        values = {"full_mAP": .1, "mAP@200_min_relevant_k": .2, "P@200": .3}
        report = {
            "clean": values,
            "masked_macro": dict(values),
            "evaluation_scope": "official_unseen_periodic_monitoring_no_selection",
        }
        (output_dir / "summary.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(trainer, "evaluate_periodic_test", evaluate)
    args = SimpleNamespace(
        dataset="tuberlin_220_30", output_dir=str(tmp_path), campaign_root=str(tmp_path),
        device="cpu", wandb_mode="disabled", smoke=False, periodic_test=True,
        runtime={"test_every": 5, "checkpoint_every": 4},
    )
    saved: list[int] = []
    original_save = trainer._save_run_checkpoint

    def save(*args: Any, **kwargs: Any) -> dict[str, Any]:
        saved.append(int(kwargs["step"]))
        return original_save(*args, **kwargs)

    monkeypatch.setattr(trainer, "_save_run_checkpoint", save)
    result = trainer._train_impl(args, tmp_path)

    assert [row[0] for row in calls] == [5, 10, 12]
    assert [row[2] for row in calls] == [False, False, True]
    assert all(row[3] and row[4] for row in calls)
    assert saved == [4, 5, 8, 10, 12]
    assert [row["step"] for row in result["official_test_evaluations"]] == [5, 10, 12]
    assert not (tmp_path / "probe_metrics.jsonl").exists()
    assert [json.loads(line)["step_train"] for line in (tmp_path / "test_metrics.jsonl").read_text().splitlines()] == [5, 10, 12]
    assert result["selections"]["latest"]["step"] == 12
    assert json.loads((tmp_path / "resolved_config.json").read_text())["tracking_policy"] == "official_test_v1"
