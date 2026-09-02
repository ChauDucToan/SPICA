from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts.summarize_transport_corrected import _same_checkpoint, build_report
from scripts.transport_artifact_utils import (
    ArtifactIntegrityError,
    MATCHED_FIELDS,
    assert_matched_runs,
    factorial_effects,
    select_unique_role,
    validate_freeze_optimizer_role,
    write_new,
)
from spica.provenance import capture_provenance, capture_rng_state
from spica.train_transport import (
    _parameter_gradient_space,
    _representation_gradient_conflicts,
    _restore_training_state,
)


def _config(**updates):
    value = {field: 1 for field in MATCHED_FIELDS}
    value.update(
        {
            "model_family": "predictive_semantic_transport",
            "transport_enabled": False,
            "use_text_cls": True,
            "lambda_cls": 1.0,
            "lambda_endpoint": 0.0,
            "text_loss_location": "q",
        }
    )
    value.update(updates)
    return value


def _run(**updates):
    config = _config(**updates)
    return {
        "config": config,
        "data_split_identity": {"sha256": "split"},
        "probe_history": [{"step": 73, "val": {"mAP": 0.5}, "base_val": {"mAP": 0.4}}],
    }


def test_causal_decomposition_rejects_different_text_configuration() -> None:
    base = _run()
    transport = _run(transport_enabled=True, text_loss_location="z0")
    with pytest.raises(ArtifactIntegrityError, match="text_loss_location"):
        assert_matched_runs(base, transport)


@pytest.mark.parametrize("field", ["seed", "steps"])
def test_causal_decomposition_rejects_seed_or_training_budget(field: str) -> None:
    base = _run()
    transport = _run(transport_enabled=True, **{field: 2})
    with pytest.raises(ArtifactIntegrityError, match=field):
        assert_matched_runs(base, transport)


def test_causal_decomposition_rejects_different_split() -> None:
    base = _run()
    transport = _run(transport_enabled=True)
    transport["data_split_identity"] = {"sha256": "other"}
    with pytest.raises(ArtifactIntegrityError, match="split"):
        assert_matched_runs(base, transport)


def test_endpoint0_factorial_effects_are_correct() -> None:
    assert factorial_effects({"A": 1.0, "B": 3.0, "C": 4.0, "D": 10.0}) == {
        "text_effect_without_transport": 2.0,
        "text_effect_with_transport": 6.0,
        "transport_effect_without_text": 3.0,
        "transport_effect_with_text": 7.0,
        "interaction": 4.0,
    }


def test_optimizer_reset_only_rejects_frozen_encoder() -> None:
    with pytest.raises(ArtifactIntegrityError, match="keep the encoder trainable"):
        validate_freeze_optimizer_role(
            {
                "experiment_role": "optimizer_reset_only",
                "freeze_encoder_at_step": 73,
                "reset_optimizer_on_resume": True,
                "resume_checkpoint_path": "source.pt",
            }
        )


@pytest.mark.parametrize(
    ("role", "freeze", "reset"),
    [
        ("freeze_optimizer_A", None, False),
        ("freeze_optimizer_B", 73, False),
        ("freeze_optimizer_C", None, True),
        ("freeze_optimizer_D", 73, True),
    ],
)
def test_freeze_reset_branches_resolve_exactly(
    role: str, freeze: int | None, reset: bool
) -> None:
    assert validate_freeze_optimizer_role(
        {
            "experiment_role": role,
            "freeze_encoder_at_step": freeze,
            "reset_optimizer_on_resume": reset,
            "resume_checkpoint_path": "source.pt",
        }
    ) == {"freeze": freeze == 73, "reset": reset}


def test_gradient_outputs_keep_q_z0_and_parameter_spaces_separate() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    z0 = parameter * 2
    q = z0.square()
    prediction = SimpleNamespace(q=q, z0=z0)
    losses = {
        "L_cls": q.sum(),
        "L_rank": q.square().sum(),
        "L_dir": z0.sum(),
        "L_endpoint": (q - 1).square().sum(),
    }
    spaces = {
        **_representation_gradient_conflicts(losses, prediction),
        "parameter": _parameter_gradient_space(losses, [parameter]),
    }
    assert {name: value["space"] for name, value in spaces.items()} == {
        "query": "dL/dq",
        "base": "dL/dz0",
        "parameter": "dL/dtheta",
    }
    assert all(len(value["cosine_matrix"]) == 4 for value in spaces.values())
    assert all(value["loss_names"] == list(losses) for value in spaces.values())


def test_dirty_tree_provenance_contains_diff_and_source_hash(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    source.write_text("value = 2\n")
    provenance = capture_provenance(tmp_path)
    assert provenance["working_tree_state"] == "dirty"
    assert "value = 2" in provenance["git_diff"]
    assert len(provenance["git_diff_sha256"]) == 64
    assert len(provenance["source_snapshot"]["sha256"]) == 64


def test_missing_raw_artifact_stays_not_run(tmp_path: Path) -> None:
    report = build_report(tmp_path, "2099-01-01")
    assert report["causal_decomposition"]["status"] == "not_run"
    assert "peak_mAP" not in report["causal_decomposition"]
    assert report["endpoint0_factorial"]["status"] == "not_run"


def test_duplicate_structured_role_is_fatal() -> None:
    records = [
        (Path("a"), {"config": {"experiment_role": "role"}}),
        (Path("b"), {"config": {"experiment_role": "role"}}),
    ]
    with pytest.raises(ArtifactIntegrityError, match="ambiguous"):
        select_unique_role(records, "role")


def test_generated_artifacts_never_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "historical.json"
    write_new(path, "old")
    with pytest.raises(FileExistsError):
        write_new(path, "new")
    assert path.read_text() == "old"


def test_checkpoint_identity_accepts_relative_and_absolute_paths() -> None:
    relative = Path("outputs/source.pt")
    assert _same_checkpoint(relative, Path.cwd() / relative)


def test_resume_restores_model_optimizer_scheduler_step_and_rng() -> None:
    random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    rng = capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.rand(2)
    payload = {
        "step": 73,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": rng,
    }
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9.0)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=9
    )
    step = _restore_training_state(
        payload,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    assert step == 73
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert random.random() == expected_python
    assert torch.equal(torch.rand(2), expected_torch)


def test_summary_contains_raw_values_not_only_conclusions(tmp_path: Path) -> None:
    run_dir = tmp_path / "group" / "timestamp"
    run_dir.mkdir(parents=True)
    result = _run(
        experiment_role="unrelated_probe",
        experiment_campaign="transport_corrected_2026-09-02_v2",
    )
    (run_dir / "run_result.json").write_text(json.dumps(result))
    report = build_report(tmp_path, "2099-01-01")
    raw = report["run_index"]["unrelated_probe"]
    assert raw["raw_metrics"][0]["val"]["mAP"] == 0.5
    assert raw["resolved_config"]["seed"] == 1
