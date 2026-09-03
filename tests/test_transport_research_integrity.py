from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts.select_transport_stage1 import select_stage1
from scripts.summarize_transport_corrected import (
    _p2_probe_steps,
    _projection_refit,
    _representation_value,
    _same_checkpoint,
    _stability,
    build_report,
)
from scripts.transport_artifact_utils import (
    ArtifactIntegrityError,
    MATCHED_FIELDS,
    assert_matched_runs,
    factorial_effects,
    select_unique_role,
    validate_freeze_optimizer_artifact,
    validate_freeze_optimizer_role,
    write_new,
)
from spica.data.manifest import ManifestEntry
from spica.data.splits import ClasswiseRetrievalSplit, split_manifest_identity
from spica.provenance import capture_provenance, capture_rng_state
from spica.train_transport import (
    _apply_resume_controls,
    _effective_probe_steps,
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


def test_manifest_identity_covers_paths_labels_and_manifest_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    manifests = {
        name: root / f"{name}.txt"
        for name in ("train_sketch", "train_photo", "class_map")
    }
    for path in manifests.values():
        path.write_text("entry\\n")
    split = ClasswiseRetrievalSplit(
        train_class_ids=(0,),
        validation_class_ids=(1,),
        train_sketch_entries=(ManifestEntry(root / "a.png", 0),),
        train_photo_entries=(ManifestEntry(root / "p.jpg", 0),),
        validation_sketch_entries=(ManifestEntry(root / "b.png", 1),),
        validation_photo_entries=(ManifestEntry(root / "q.jpg", 1),),
        seed=3407,
    )
    first = split_manifest_identity(
        split,
        dataset_name="fixture",
        dataset_root=root,
        manifest_paths=manifests,
    )
    changed_path = replace(
        split,
        train_sketch_entries=(ManifestEntry(root / "other.png", 0),),
    )
    changed_label = replace(
        split,
        train_sketch_entries=(ManifestEntry(root / "a.png", 9),),
    )
    for changed in (changed_path, changed_label):
        assert (
            first["sha256"]
            != split_manifest_identity(
                changed,
                dataset_name="fixture",
                dataset_root=root,
                manifest_paths=manifests,
            )["sha256"]
        )
    manifests["train_sketch"].write_text("changed\\n")
    assert (
        first["sha256"]
        != split_manifest_identity(
            split,
            dataset_name="fixture",
            dataset_root=root,
            manifest_paths=manifests,
        )["sha256"]
    )


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


def test_causal_decomposition_rejects_different_manifest_identity() -> None:
    base = _run()
    transport = _run(transport_enabled=True)
    base["data_manifest_identity"] = {"sha256": "base"}
    transport["data_manifest_identity"] = {"sha256": "other"}
    with pytest.raises(ArtifactIntegrityError, match="manifest-entry"):
        assert_matched_runs(base, transport)


def test_endpoint0_factorial_effects_are_correct() -> None:
    assert factorial_effects({"A": 1.0, "B": 3.0, "C": 4.0, "D": 10.0}) == {
        "text_effect_without_transport": 2.0,
        "text_effect_with_transport": 6.0,
        "transport_effect_without_text": 3.0,
        "transport_effect_with_text": 7.0,
        "interaction": 4.0,
    }


def test_freeze_optimizer_artifact_validates_hashes_and_resume_flags(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pt"
    result_path = tmp_path / "result.pt"
    source_path.write_bytes(b"source")
    result_path.write_bytes(b"result")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    source = {"checkpoint": str(source_path), "checkpoint_sha256": source_hash}
    result = {
        "checkpoint": str(result_path),
        "checkpoint_sha256": result_hash,
        "config": {
            "experiment_role": "freeze_optimizer_B",
            "freeze_encoder_at_step": 73,
            "reset_optimizer_on_resume": False,
            "resume_checkpoint_path": str(source_path),
        },
        "resume": {
            "checkpoint": str(source_path),
            "checkpoint_sha256": source_hash,
            "starting_step": 73,
            "freeze_applied_before_first_update": True,
            "encoder_frozen_immediately": True,
            "optimizer_state_restored": True,
            "optimizer_state_reset": False,
            "transport_head_reinitialized": False,
        },
    }
    assert validate_freeze_optimizer_artifact(result, source)["status"] == "validated"
    result["resume"]["optimizer_state_restored"] = False
    with pytest.raises(ArtifactIntegrityError, match="restore state"):
        validate_freeze_optimizer_artifact(result, source)


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


def test_p2_probe_schedule_follows_selected_origin() -> None:
    assert _p2_probe_steps(44) == [44, 100, 150, 250, 500, 1800, 5400]
    assert _p2_probe_steps(73) == [73, 100, 150, 250, 500, 1800, 5400]


def test_stability_uses_actual_probe_steps_and_unknown_transport_is_not_base() -> None:
    result = {
        "config": {"transport_enabled": True},
        "probe_history": [
            {"step": 100, "val": {"mAP": 0.5}},
            {"step": 150, "val": {"mAP": 0.7}},
            {"step": 250, "val": {"mAP": 0.6}},
        ],
    }
    stability = _stability(result)
    assert stability["peak_step"] == 150
    assert stability["late_step"] == 250
    assert stability["status"] == "partial"
    assert _representation_value({"config": {}, "probe_history": []}, 100, "q") is None


def test_missing_raw_artifact_stays_not_run(tmp_path: Path) -> None:
    report = build_report(tmp_path, "2099-01-01")
    assert report["causal_decomposition"]["status"] == "not_run"
    assert "peak_mAP" not in report["causal_decomposition"]
    assert report["endpoint0_factorial"]["status"] == "not_run"


def test_stage1_selection_rejects_duplicate_candidate_steps(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step44.pt"
    split = {"sha256": "split"}
    torch.save(
        {
            "step": 44,
            "metadata": {
                "experiment_role": "two_stage_stage1",
                "transport_enabled": False,
                "data_split_identity": split,
            },
        },
        checkpoint,
    )
    run_result = tmp_path / "run_result.json"
    protocol = {
        "val_is_pseudo_unseen": True,
        "official_test_is_diagnostic_only": True,
    }
    run_result.write_text(
        json.dumps(
            {
                "config": {
                    "experiment_role": "two_stage_stage1",
                    "transport_enabled": False,
                    "train_class_scope": "pseudo_train",
                },
                "data_split_identity": split,
                "provenance": {"status": "valid"},
                "probe_history": [
                    {
                        "step": 44,
                        "protocol": protocol,
                        "checkpoint": str(checkpoint),
                        "val": {"mAP": 0.5},
                    },
                    {
                        "step": 44,
                        "protocol": protocol,
                        "checkpoint": str(checkpoint),
                        "val": {"mAP": 0.5},
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        select_stage1(run_result)


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


def test_resume_probe_offsets_resolve_to_global_steps() -> None:
    assert _effective_probe_steps(
        resume_step=73,
        absolute_steps=[],
        relative_offsets=[0, 27, 77, 427],
        max_steps=5400,
        run_probes=True,
    ) == [73, 100, 150, 500, 5400]
    with pytest.raises(ValueError, match="between resume step"):
        _effective_probe_steps(
            resume_step=73,
            absolute_steps=[44],
            relative_offsets=None,
            max_steps=5400,
            run_probes=True,
        )


def test_resume_freeze_rebuilds_optimizer_before_updates() -> None:
    encoder = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 2)
    model = SimpleNamespace(sketch_context_encoder=encoder, transport_head=head)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=0.1
    )
    rebuilt, freeze_step, applied = _apply_resume_controls(
        model,
        optimizer,
        resume_step=73,
        freeze_encoder_on_resume=True,
        freeze_encoder_at_step=None,
    )
    assert applied is True
    assert freeze_step == 73
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert {
        id(parameter) for group in rebuilt.param_groups for parameter in group["params"]
    } == {id(parameter) for parameter in head.parameters()}


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


def test_projection_refit_rejects_tampered_hashed_source(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"checkpoint")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    methods = {
        name: {"mAP": 0.5}
        for name in ("frozen_original_W_CLIP", "orthogonal_adapter", "ridge_projection")
    }
    artifact = {
        "schema_version": 2,
        "probe": "projection_refit_control",
        "selection_metric": None,
        "fit_split": None,
        "seed": 3407,
        "official_unseen_used": False,
        "data_manifest_identity": {"sha256": "manifest"},
        "provenance": {"status": "valid"},
        "source_artifacts": [{"path": str(source), "sha256": digest}],
        "values": [
            {
                "experiment_role": "freeze_optimizer_source",
                "experiment_campaign": "transport_corrected_2026-09-02_v2",
                "checkpoint_step": 73,
                "fit_split": "pseudo_train only",
                "evaluation_split": "pseudo-unseen only",
                "data_manifest_identity": {"sha256": "manifest"},
                "methods": methods,
                "checkpoint": str(source),
                "checkpoint_sha256": digest,
            }
        ],
    }
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(artifact))
    assert _projection_refit(path)["status"] == "completed"
    source.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="hash/path"):
        _projection_refit(path)


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
