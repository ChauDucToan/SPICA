"""Fail-closed reporting tests using tiny synthetic run artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from scripts.summarize_alignment import write_report


def _write_run(
    root: Path,
    role: str,
    seed: int,
    *,
    horizon: int = 10,
    final_step: int = 10,
    mAP: float = 0.5,
    wrong_hash: bool = False,
) -> None:
    run_dir = root / role / f"seed{seed}" / f"run{seed}_{role}"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    initial = checkpoints / "step0.pt"
    final = checkpoints / f"step{final_step}.pt"
    torch.save({"initial_model_state_hash": f"init-{seed}"}, initial)
    torch.save({"initial_model_state_hash": f"init-{seed}"}, final)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    rows = [
        {
            "training_global_step": 0,
            "checkpoint": str(initial),
            "checkpoint_sha256": digest(initial),
            "full_pseudo_unseen_mAP": 0.1,
        },
    ]
    final_hash = digest(final)
    rows.append(
        {
            "training_global_step": final_step,
            "checkpoint": str(final),
            "checkpoint_sha256": "bad" if wrong_hash else final_hash,
            "full_pseudo_unseen_mAP": mAP,
        }
    )
    config = {
        "max_steps": horizon,
        "experiment_role": role,
        "experiment_name": f"run-{role}",
        "alignment_target_gradient": "detached",
        "lambda_alignment_mean": 0.0 if role == "alignment_control" else 0.2,
        "lambda_alignment_covariance": 0.0,
    }
    result = {
        "campaign": "synthetic",
        "experiment_role": role,
        "run_kind": "pilot",
        "training_seed": seed,
        "seed": seed,
        "source_snapshot_hash": "same-source",
        "resolved_config": config,
        "resolved_treatment": config.copy(),
        "pseudo_split_identity": {"sha256": "same-split"},
        "protocol": {
            "selection_metric": "full_pseudo_unseen_mAP",
            "train_class_scope": "pseudo_train",
            "alignment_fit_scope": "pseudo_train_only",
            "validation_used_for_alignment": False,
            "test_used_for_alignment": False,
            "text_used_for_predictor": False,
            "photo_used_for_predictor": False,
            "ranking_positive_reduction": "mean",
        },
        "official_unseen_used_for_selection": False,
        "history": rows,
    }
    (run_dir / "run_result.json").write_text(json.dumps(result))


def test_missing_same_seed_does_not_fallback_to_control_mean(tmp_path: Path) -> None:
    _write_run(tmp_path, "alignment_control", 42, mAP=0.6)
    _write_run(tmp_path, "alignment_mean_text_log", 123, mAP=0.9)
    output = tmp_path / "report.md"
    payload = write_report(tmp_path, output, horizon=10)
    campaign = payload["campaigns"][0]
    pair = campaign["pairs"]["alignment_mean_text_log"][0]
    assert pair["status"] == "UNMATCHED"
    assert pair["paired_delta"] is None


def test_missing_requested_step_is_incomplete(tmp_path: Path) -> None:
    _write_run(tmp_path, "alignment_control", 42, horizon=10, final_step=5)
    payload = write_report(tmp_path, tmp_path / "report.md", horizon=10)
    run = payload["campaigns"][0]["runs"][0]
    assert run["fixed_step"]["status"] == "INCOMPLETE"
    assert run["fixed_step"]["mAP"] is None


def test_duplicate_control_suppresses_delta(tmp_path: Path) -> None:
    _write_run(tmp_path, "alignment_control", 42)
    source = tmp_path / "alignment_control" / "seed42" / "run42_alignment_control"
    duplicate = tmp_path / "alignment_control" / "seed42" / "duplicate"
    shutil.copytree(source, duplicate)
    _write_run(tmp_path, "alignment_mean_text_log", 42, mAP=0.9)
    payload = write_report(tmp_path, tmp_path / "report.md", horizon=10)
    pair = payload["campaigns"][0]["pairs"]["alignment_mean_text_log"][0]
    assert pair["status"] == "DUPLICATE_CONTROL"
    assert pair["paired_delta"] is None


def test_checkpoint_hash_mismatch_is_invalid(tmp_path: Path) -> None:
    _write_run(tmp_path, "alignment_control", 42, wrong_hash=True)
    payload = write_report(tmp_path, tmp_path / "report.md", horizon=10)
    run = payload["campaigns"][0]["runs"][0]
    assert run["status"] == "ARTIFACT_INVALID"
    assert run["fixed_step"]["mAP"] is None
    assert any("hash mismatch" in error for error in run["errors"])


def test_invalid_candidate_cannot_produce_delta(tmp_path: Path) -> None:
    _write_run(tmp_path, "alignment_control", 42)
    _write_run(tmp_path, "alignment_mean_text_log", 42, wrong_hash=True, mAP=0.9)
    payload = write_report(tmp_path, tmp_path / "report.md", horizon=10)
    pair = payload["campaigns"][0]["pairs"]["alignment_mean_text_log"][0]
    assert pair["status"] == "ARTIFACT_INVALID"
    assert pair["paired_delta"] is None


def test_incomplete_provenance_cannot_produce_delta(tmp_path: Path) -> None:
    _write_run(tmp_path, "alignment_control", 42)
    _write_run(tmp_path, "alignment_mean_text_log", 42, mAP=0.9)
    path = (
        tmp_path
        / "alignment_mean_text_log"
        / "seed42"
        / "run42_alignment_mean_text_log"
        / "run_result.json"
    )
    result = json.loads(path.read_text())
    result.pop("source_snapshot_hash")
    path.write_text(json.dumps(result))
    payload = write_report(tmp_path, tmp_path / "report.md", horizon=10)
    pair = payload["campaigns"][0]["pairs"]["alignment_mean_text_log"][0]
    assert pair["status"] == "UNMATCHED_PROVENANCE"
    assert pair["paired_delta"] is None
