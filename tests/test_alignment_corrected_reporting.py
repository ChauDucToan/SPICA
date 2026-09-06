from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.summarize_alignment import (
    CALIBRATION_RULE,
    CORRECTED_PILOT_REQUIRED_CONFIG_KEYS,
    _canonical_hash,
    write_report,
)
from spica.config.data import load_data_config
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split, split_manifest_identity
from spica.alignment_artifacts import (
    ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
    ensure_corrected_run_manifest,
    make_manifest,
    manifest_identity_sha256,
)


ROLES = (
    "alignment_control",
    "alignment_mean_text_log",
    "alignment_mean_text_log_symmetric",
)


def _base_config(role: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        key: "same" for key in CORRECTED_PILOT_REQUIRED_CONFIG_KEYS
    }
    values.update(
        {
            "experiment_campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
            "experiment_role": role,
            "run_kind": "pilot",
            "experiment_manifest_path": "manifest.json",
            "data_config": "data.yaml",
            "model_name": "toy-model",
            "pretrained": "openai",
            "visual_prompt_length": 3,
            "prompt_mode": "prompt_only",
            "text_mode": "soft",
            "soft_prompt_length": 4,
            "prompt_template": "a photo of a {}",
            "train_visual_layernorm": False,
            "train_sketch_prompt": True,
            "train_photo_prompt": True,
            "classification_location": "query",
            "encoder_mode": "frozen",
            "encoder_unfreeze_depth": 0,
            "encoder_train_ln_post": False,
            "transport_enabled": False,
            "transport_mode": "none",
            "batch_size": 32,
            "classes_per_batch": 16,
            "sketches_per_class": 2,
            "num_positive_photos": 4,
            "lambda_rank": 1.0,
            "lambda_cls": 1.0,
            "margin": 0.2,
            "tau_cls": 0.07,
            "visual_prompt_learning_rate": 1e-3,
            "soft_prompt_learning_rate": 1e-3,
            "visual_layernorm_learning_rate": 1e-6,
            "encoder_learning_rate": 1e-5,
            "visual_prompt_weight_decay": 1e-4,
            "soft_prompt_weight_decay": 1e-4,
            "visual_layernorm_weight_decay": 0.0,
            "encoder_weight_decay": 1e-4,
            "alignment_geometry": "log_map",
            "alignment_anchor": "text",
            "alignment_target_gradient": (
                "symmetric"
                if role == "alignment_mean_text_log_symmetric"
                else "detached"
            ),
            "lambda_alignment_mean": 0.0
            if role == "alignment_control"
            else 0.2,
            "lambda_alignment_covariance": 0.0,
            "calibration_batches": 2,
            "calibration_target_ratio": 0.1,
            "max_steps": 2,
            "probe_steps": [0, 2],
            "eval_batch_size": 256,
            "query_chunk_size": 256,
            "pseudo_val_seed": 3407,
            "pseudo_val_num_classes": 1,
            "seed": 42,
            "train_class_scope": "pseudo_train",
            "official_unseen_used_for_selection": False,
            "alignment_calibration_artifact": None,
        }
    )
    return values


def _data_fixture(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    (dataset_root / "sketches.txt").write_text("sketch0.png 0\nsketch1.png 1\n")
    (dataset_root / "photos.txt").write_text("photo0.png 0\nphoto1.png 1\n")
    (dataset_root / "classes.txt").write_text("zero 0\none 1\n")
    config_path = root / "data.yaml"
    config_path.write_text(
        f"version: 1\nname: toy\nroot: {dataset_root}\n"
        "train:\n  sketch_manifest: sketches.txt\n  photo_manifest: photos.txt\n  class_map: classes.txt\n"
        "test:\n  sketch_manifest: sketches.txt\n  photo_manifest: photos.txt\n  class_map: classes.txt\n"
    )
    data = load_data_config(config_path)
    names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches, photos, names, num_validation_classes=1, seed=3407
    )
    split_identity: dict[str, Any] = {
        "dataset": data.name,
        "seed": split.seed,
        "train_class_ids": list(split.train_class_ids),
        "validation_class_ids": list(split.validation_class_ids),
        "train_sketches": len(split.train_sketch_entries),
        "train_photos": len(split.train_photo_entries),
        "validation_sketches": len(split.validation_sketch_entries),
        "validation_photos": len(split.validation_photo_entries),
    }
    split_identity["sha256"] = _canonical_hash(split_identity)
    manifest_identity = split_manifest_identity(
        split,
        dataset_name=data.name,
        dataset_root=data.root,
        manifest_paths={
            "train_sketch": data.train.sketch_manifest,
            "train_photo": data.train.photo_manifest,
            "train_class_map": data.train.class_map,
        },
    )
    return config_path, split_identity, manifest_identity


def _calibration_payload(split_identity: dict[str, Any]) -> dict[str, Any]:
    split_hash = split_identity["sha256"]
    fixed_batches = [{"batch": 1}, {"batch": 2}]
    return {
        "schema_version": 2,
        "schema_name": "corrected_mean_alignment_calibration",
        "status": "VALID",
        "experiment_role": "alignment_mean_text_log",
        "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
        "training_seed": 42,
        "initial_model_state_hash": "init-42",
        "initial_text_bank_state_hash": "text-init-42",
        "source_snapshot_hash": "source",
        "split_identity_hash": split_hash,
        "split_identity": split_identity,
        "base": {
            "sketch_gradient_norms": [2.0, 4.0],
            "photo_gradient_norms": [1.0, 1.0],
        },
        "detached": {
            "sketch_gradient_norms": [1.0, 2.0],
            "photo_gradient_norms": [0.0, 0.0],
            "weighted_sketch_ratios": [0.1, 0.1],
            "weighted_sketch_ratio_reasons": [None, None],
            "weighted_photo_ratios": [0.0, 0.0],
            "weighted_photo_ratio_reasons": [None, None],
            "sketch_cosines_with_base": [1.0, 1.0],
            "sketch_cosine_reasons": [None, None],
            "photo_cosines_with_base": [None, None],
            "photo_cosine_reasons": ["alignment_gradient_zero"] * 2,
            "unweighted_sketch_ratios": [2.0, 2.0],
            "unweighted_sketch_ratio_reasons": [None, None],
        },
        "symmetric": {
            "sketch_gradient_norms": [1.0, 2.0],
            "photo_gradient_norms": [1.0, 1.0],
            "weighted_sketch_ratios": [0.1, 0.1],
            "weighted_sketch_ratio_reasons": [None, None],
            "weighted_photo_ratios": [0.2, 0.2],
            "weighted_photo_ratio_reasons": [None, None],
            "sketch_cosines_with_base": [1.0, 1.0],
            "sketch_cosine_reasons": [None, None],
            "photo_cosines_with_base": [1.0, 1.0],
            "photo_cosine_reasons": [None, None],
        },
        "calibration": {
            "status": "VALID",
            "lambda_alignment_mean": 0.2,
            "target_ratio": 0.1,
            "rule": CALIBRATION_RULE,
            "batches": 2,
            "fixed_batch_identity": {
                "count": 2,
                "sha256": _canonical_hash(fixed_batches),
                "batches": fixed_batches,
                "sampler_epoch_before": 0,
                "worker_lifecycle_verified": True,
            },
            "initialization_identity": {
                "model_state_hash": "init-42",
                "text_bank_state_hash": "text-init-42",
            },
            "experiment_role": "alignment_mean_text_log",
            "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
            "training_seed": 42,
            "initial_model_state_hash": "init-42",
            "initial_text_bank_state_hash": "text-init-42",
            "source_snapshot_hash": "source",
            "split_identity_hash": split_hash,
            "state_restoration_verified": True,
            "lambda_selection_status": "VALID",
            "lambda_selection_reasons": [None, None],
            "sketch_gradient_comparison": {
                "detached_vs_symmetric_max_abs_difference": 0.0,
                "tolerance": 1e-6,
                "match": True,
            },
        },
    }


def _write_runs(root: Path, roles: tuple[str, ...] = ROLES) -> Path:
    data_config_path, split_identity, manifest_identity = _data_fixture(root)
    calibration_path = root / "calibration.json"
    calibration = _calibration_payload(split_identity)
    calibration_config = _base_config("alignment_mean_text_log")
    calibration_config.update(
        {
            "alignment_calibration_artifact": None,
            "calibration_only": True,
            "data_config": str(data_config_path),
        }
    )
    calibration["calibration_config"] = calibration_config
    calibration["config_hash"] = _canonical_hash(calibration_config)
    calibration["calibration"]["config_hash"] = calibration["config_hash"]
    calibration_path.write_text(json.dumps(calibration))
    calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    for role in roles:
        run_dir = root / role
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        step0 = checkpoint_dir / "step0.pt"
        final = checkpoint_dir / "step2.pt"
        config = _base_config(role)
        config["data_config"] = str(data_config_path)
        if role != "alignment_control":
            config["alignment_calibration_artifact"] = str(calibration_path)
        manifest_path = run_dir / "manifest.json"
        manifest_entry = {
            "run_id": f"{role}-42",
            "experiment_role": role,
            "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
            "training_seed": 42,
            "config_hash": _canonical_hash(config),
        }
        manifest_document = {
            "schema_version": 2,
            "manifest_marker": "spica_corrected_alignment_manifest_v2",
            "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
            "dataset": "toy",
            "data_config": str(data_config_path),
            "roles": list(ROLES),
            "entries": [manifest_entry],
            "protocol": {
                "selection_metric": "full_pseudo_unseen_mAP",
                "official_unseen_used_for_selection": False,
                "text_used_for_predictor": False,
                "photo_used_for_predictor": False,
                "train_only_alignment_targets": True,
            },
        }
        manifest_path.write_text(json.dumps(manifest_document))
        manifest_entry_identity = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _canonical_hash(
                {key: value for key, value in manifest_document.items() if key != "entries"}
            ),
            "entry_sha256": _canonical_hash(manifest_entry),
            "entry_pointer": "/entries/0",
        }
        manifest_entry.update(
            {
                "status": "REGISTERED",
                "pseudo_validation_seed": 3407,
                "replicate_id": f"pilot-seed42-{role}",
                "source_hash": "source",
                "training_horizon": 2,
                "initialization_identity": {"initial_model_state_hash": "init-42"},
                "treatment": dict(config),
                "dataset_identity": {
                    "dataset": "toy",
                    "data_config": str(data_config_path),
                    "split_identity": split_identity,
                },
            }
        )
        manifest_entry["run_id"] = _canonical_hash(
            {
                "campaign": manifest_entry["campaign"],
                "role": manifest_entry["experiment_role"],
                "training_seed": manifest_entry["training_seed"],
                "split_identity": manifest_entry["dataset_identity"]["split_identity"],
                "config_hash": manifest_entry["config_hash"],
                "replicate_id": manifest_entry["replicate_id"],
            }
        )
        manifest_document["entries"] = [manifest_entry]
        manifest_path.write_text(json.dumps(manifest_document))
        manifest_entry_identity["manifest_sha256"] = _canonical_hash(
            {key: value for key, value in manifest_document.items() if key != "entries"}
        )
        manifest_entry_identity["entry_sha256"] = _canonical_hash(manifest_entry)
        optimizer_groups = [
            {
                "name": "visual_prompts",
                "parameter_names": ["sketch_prompt", "photo_prompt"],
                "parameter_count": 10,
                "lr": 1e-3,
                "weight_decay": 1e-4,
            }
        ]
        resolved_treatment = dict(config)
        calibration_identity = (
            None
            if role == "alignment_control"
            else {
                "artifact": str(calibration_path.resolve()),
                "artifact_sha256": calibration_hash,
                "fixed_batch_identity_sha256": calibration["calibration"][
                    "fixed_batch_identity"
                ]["sha256"],
                "first_batch_replay_verified": False,
                "calibration_batch_replay_verified": False,
                "calibration_batch_replay_count": 0,
                "calibration_batch_replay_expected_count": 2,
                "calibration_batch_replay_prefix_sha256": _canonical_hash([]),
                "worker_lifecycle_verified": True,
            }
        )
        checkpoint = {
            "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
            "experiment_role": role,
            "run_kind": "pilot",
            "initial_model_state_hash": "init-42",
            "initial_text_bank_state_hash": "text-init-42",
            "source_snapshot_hash": "source",
            "training_seed": 42,
            "data_split_identity": split_identity,
            "data_manifest_identity": manifest_identity,
            "manifest_entry_identity": manifest_entry_identity,
            "resolved_config": config,
            "resolved_treatment": resolved_treatment,
            "gradient_calibration_identity": calibration_identity,
            "optimizer_groups": optimizer_groups,
            "scheduler_state_dict": {"last_epoch": 0, "_step_count": 1},
            "backbone_identity": {"visual_class": "toy", "width": 8},
        }
        torch.save(
            {**checkpoint, "step": 0, "training_global_step": 0, "full_pseudo_unseen_mAP": 0.1},
            step0,
        )
        final_checkpoint = {
            **checkpoint,
            "step": 2,
            "training_global_step": 2,
            "full_pseudo_unseen_mAP": 0.5
            if role == "alignment_control"
            else 0.6,
        }
        if calibration_identity is not None:
            final_checkpoint["gradient_calibration_identity"] = {
                **calibration_identity,
                "first_batch_replay_verified": True,
                "calibration_batch_replay_verified": True,
                "calibration_batch_replay_count": 2,
                "calibration_batch_replay_prefix_sha256": _canonical_hash(
                    calibration["calibration"]["fixed_batch_identity"]["batches"]
                ),
            }
        torch.save(final_checkpoint, final)

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        result: dict[str, Any] = {
            "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
            "experiment_role": role,
            "run_kind": "pilot",
            "training_seed": 42,
            "seed": 42,
            "dataset": "toy",
            "source_snapshot_hash": "source",
            "initial_model_state_hash": "init-42",
            "initial_text_bank_state_hash": "text-init-42",
            "initialization_identity": {
                "model_state_hash": "init-42",
                "text_bank_state_hash": "text-init-42",
            },
            "resolved_config": config,
            "resolved_treatment": resolved_treatment,
            "pseudo_split_identity": split_identity,
            "manifest_identity": manifest_identity,
            "manifest_path": str(manifest_path),
            "manifest_entry_identity": manifest_entry_identity,
            "matched_sampler": {
                "type": "MatchedClassBatchSampler",
                "batch_size": 32,
                "classes_per_batch": 16,
                "sketches_per_class": 2,
                "positive_photos_per_sketch": 4,
                "negative_photos_per_sketch": 1,
            },
            "optimizer_groups": optimizer_groups,
            "protocol": {
                "selection_metric": "full_pseudo_unseen_mAP",
                "official_unseen_used_for_selection": False,
                "train_class_scope": "pseudo_train",
                "alignment_fit_scope": "pseudo_train_only",
                "validation_used_for_alignment": False,
                "test_used_for_alignment": False,
                "text_used_for_predictor": False,
                "photo_used_for_predictor": False,
                "ranking_positive_reduction": "mean_over_4_positive_photos",
            },
            "official_unseen_used_for_selection": False,
            "inference_contract": {
                "required_inputs": ["raw_sketch_image"],
                "text_required": False,
                "photo_required": False,
                "oracle_class_required": False,
                "text_used_for_predictor": False,
                "photo_prompt_used_for_query": False,
                "photo_prompt_used_for_gallery": True,
            },
            "history": [
                {
                    "training_global_step": 0,
                    "checkpoint": str(step0),
                    "checkpoint_sha256": digest(step0),
                    "full_pseudo_unseen_mAP": 0.1,
                },
                {
                    "training_global_step": 2,
                    "checkpoint": str(final),
                    "checkpoint_sha256": digest(final),
                    "full_pseudo_unseen_mAP": 0.5
                    if role == "alignment_control"
                    else 0.6,
                },
            ],
        }
        if role != "alignment_control":
            result["gradient_calibration"] = {
                "artifact": str(calibration_path),
                "artifact_sha256": calibration_hash,
                "lambda_alignment_mean": 0.2,
                "schema_version": 2,
                "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
                "experiment_role": role,
                "training_seed": 42,
                "source_snapshot_hash": "source",
                "split_identity_hash": split_identity["sha256"],
                "initial_model_state_hash": "init-42",
                "initial_text_bank_state_hash": "text-init-42",
                "config_hash": calibration["config_hash"],
                "fixed_batch_identity_sha256": calibration_identity[
                    "fixed_batch_identity_sha256"
                ],
                "first_batch_replay_verified": True,
                "calibration_batch_replay_verified": True,
                "calibration_batch_replay_count": 2,
                "calibration_batch_replay_expected_count": 2,
                "calibration_batch_replay_prefix_sha256": _canonical_hash(
                    calibration["calibration"]["fixed_batch_identity"]["batches"]
                ),
                "worker_lifecycle_verified": True,
            }
        (run_dir / "run_result.json").write_text(json.dumps(result))
    return calibration_path


def _pair(payload: dict[str, Any], role: str) -> dict[str, Any]:
    return next(
        pair for pair in payload["campaigns"][0]["pairs"][role]
    )


def test_corrected_report_verifies_pair_specific_treatments(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    assert _pair(payload, "alignment_mean_text_log")["status"] == "MATCHED"
    assert _pair(payload, "alignment_mean_text_log_symmetric")["status"] == "MATCHED"
    assert payload["campaigns"][0]["pairwise_comparisons"][0]["status"] == "MATCHED"


def test_corrected_report_rejects_non_treatment_config_difference(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["resolved_config"]["batch_size"] = 64
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    pair = _pair(payload, "alignment_mean_text_log")
    assert pair["status"] == "UNMATCHED_CONFIG"
    assert pair["paired_delta"] is None
    assert "batch_size" in pair["disallowed_config_differences"]


def test_corrected_report_rejects_bad_calibration_but_keeps_other_arm(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["gradient_calibration"]["artifact_sha256"] = "wrong"
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    assert _pair(payload, "alignment_mean_text_log")["status"] == "UNMATCHED_CALIBRATION"
    assert _pair(payload, "alignment_mean_text_log")["paired_delta"] is None
    assert _pair(payload, "alignment_mean_text_log_symmetric")["status"] == "MATCHED"


def test_corrected_report_records_missing_arm_and_control(tmp_path: Path) -> None:
    _write_runs(tmp_path, ("alignment_mean_text_log",))
    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    campaign = payload["campaigns"][0]
    assert _pair(payload, "alignment_mean_text_log")["status"] == "UNMATCHED"
    assert _pair(payload, "alignment_mean_text_log_symmetric")["status"] == "UNMATCHED"
    assert any(
        item["role"] == "alignment_control" for item in campaign["missing_arms"]
    )
    assert any(
        item["pair_id"] == "MD-MS" and item["status"] == "UNMATCHED"
        for item in campaign["pairwise_comparisons"]
    )


def test_corrected_report_rejects_declared_loose_sketch_tolerance(tmp_path: Path) -> None:
    calibration_path = _write_runs(tmp_path)
    calibration = json.loads(calibration_path.read_text())
    comparison = calibration["calibration"]["sketch_gradient_comparison"]
    comparison["detached_vs_symmetric_max_abs_difference"] = 0.5
    comparison["tolerance"] = 0.5
    calibration_path.write_text(json.dumps(calibration))
    artifact_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    for role in ROLES[1:]:
        path = tmp_path / role / "run_result.json"
        result = json.loads(path.read_text())
        result["gradient_calibration"]["artifact_sha256"] = artifact_hash
        path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    assert _pair(payload, "alignment_mean_text_log")["status"] == "UNMATCHED_CALIBRATION"


def test_corrected_report_requires_role_identity_in_resolved_config(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["resolved_config"]["experiment_role"] = "alignment_control"
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    assert _pair(payload, "alignment_mean_text_log")["status"] == "UNMATCHED_CONFIG"


def test_corrected_report_rejects_md_ms_without_same_identity_control(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_control" / "run_result.json"
    result = json.loads(path.read_text())
    result["training_seed"] = 123
    result["seed"] = 123
    result["resolved_config"]["seed"] = 123
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    comparison = payload["campaigns"][0]["pairwise_comparisons"][0]
    assert comparison["status"] == "UNMATCHED"
    assert comparison["paired_delta"] is None


def test_corrected_report_records_duplicate_arm_without_overwriting(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    original = tmp_path / "alignment_mean_text_log" / "run_result.json"
    duplicate = tmp_path / "duplicate_md" / "run_result.json"
    duplicate.parent.mkdir()
    duplicate.write_text(original.read_text())

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    campaign = payload["campaigns"][0]
    assert all(
        pair["status"] == "DUPLICATE_ARM"
        for pair in campaign["pairs"]["alignment_mean_text_log"]
    )
    assert campaign["duplicate_arms"]


def test_corrected_report_missing_horizon_suppresses_peak_delta(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["history"] = result["history"][:1]
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    pair = _pair(payload, "alignment_mean_text_log")
    assert pair["status"] == "INCOMPLETE"
    assert pair["paired_delta"] is None
    assert pair["peak_delta"] is None


def test_corrected_report_binds_history_to_checkpoint_step_and_metric(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "checkpoints" / "step2.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["step"] = 1
    torch.save(checkpoint, path)
    result_path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(result_path.read_text())
    result["history"][1]["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    assert _pair(payload, "alignment_mean_text_log")["status"] == "ARTIFACT_INVALID"


def test_corrected_report_rejects_metric_not_bound_to_checkpoint(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["history"][1]["full_pseudo_unseen_mAP"] = 0.9
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    assert _pair(payload, "alignment_mean_text_log")["status"] == "ARTIFACT_INVALID"


def test_corrected_report_recomputes_split_identity_hash(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["pseudo_split_identity"]["train_class_ids"] = [999]
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    assert _pair(payload, "alignment_mean_text_log")["status"] in {
        "ARTIFACT_INVALID",
        "UNMATCHED_CALIBRATION",
    }


def test_corrected_report_verifies_manifest_entry_bytes(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    manifest = tmp_path / "alignment_control" / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "entries": []}))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    control = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_control"
    )
    assert control["status"] == "ARTIFACT_INVALID"


def test_corrected_report_invalidates_pairs_before_summary_deltas(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path)
    source = tmp_path / "alignment_control" / "run_result.json"
    extra = tmp_path / "unexpected" / "run_result.json"
    extra.parent.mkdir()
    extra.write_text(source.read_text())
    result = json.loads(extra.read_text())
    result["experiment_role"] = "unexpected_role"
    extra.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    campaign = payload["campaigns"][0]
    assert campaign["unexpected_roles"] == ["unexpected_role"]
    assert all(
        summary["paired_fixed_delta"]["count"] == 0
        for summary in campaign["summaries"]
    )


def test_corrected_manifest_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    historical = tmp_path / "corrected_pilot_manifest.json"
    historical.write_text(json.dumps({"schema_version": 2, "entries": []}))
    split = {"sha256": "split"}
    kwargs = {
        "data_config": "data.yaml",
        "dataset": "toy",
        "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
        "role": "alignment_control",
        "training_seed": 42,
        "pseudo_validation_seed": 3407,
        "split_identity": split,
        "resolved_config": {"seed": 42},
        "source_hash": "source",
        "initial_model_state_hash": "init",
        "training_horizon": 2,
        "replicate_id": "pilot-seed42",
    }
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(historical)
    with pytest.raises(ValueError):
        ensure_corrected_run_manifest(symlink, **kwargs)
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(historical)
    with pytest.raises(ValueError):
        ensure_corrected_run_manifest(hardlink, **kwargs)


def test_corrected_manifest_appends_without_invalidating_prior_identity(
    tmp_path: Path,
) -> None:
    config_path, split_identity, _ = _data_fixture(tmp_path)
    path = tmp_path / "corrected_pilot_manifest_v2.json"
    base = {
        "data_config": str(config_path),
        "dataset": "toy",
        "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
        "training_seed": 42,
        "pseudo_validation_seed": 3407,
        "split_identity": split_identity,
        "source_hash": "source",
        "training_horizon": 2,
    }
    config = _base_config("alignment_control")
    first, first_hash = ensure_corrected_run_manifest(
        path,
        role="alignment_control",
        resolved_config=config,
        initial_model_state_hash="init-r",
        replicate_id="pilot-seed42-r",
        **base,
    )
    second, second_hash = ensure_corrected_run_manifest(
        path,
        role="alignment_mean_text_log",
        resolved_config={**config, "experiment_role": "alignment_mean_text_log"},
        initial_model_state_hash="init-md",
        replicate_id="pilot-seed42-md",
        **base,
    )
    assert first_hash == second_hash == manifest_identity_sha256(second)
    assert len(first["entries"]) == 1
    assert len(second["entries"]) == 2
    assert second["entries"][0] == first["entries"][0]
    assert second["entries"][1]["experiment_role"] == "alignment_mean_text_log"


def test_make_manifest_rejects_corrected_campaign() -> None:
    with pytest.raises(ValueError, match="ensure_corrected_run_manifest"):
        make_manifest(
            data_config="data.yaml",
            dataset="toy",
            campaign=ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
        )


def test_corrected_report_rejects_negative_checkpoint_step(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["history"][0]["training_global_step"] = -1
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    pair = _pair(payload, "alignment_mean_text_log")
    assert pair["status"] != "MATCHED"
    assert pair["paired_delta"] is None


def test_corrected_report_requires_calibration_identity_on_each_checkpoint(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path)
    checkpoint_path = tmp_path / "alignment_mean_text_log" / "checkpoints" / "step2.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.pop("gradient_calibration_identity")
    torch.save(checkpoint, checkpoint_path)
    result_path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(result_path.read_text())
    result["history"][1]["checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    result_path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"


def test_corrected_report_rejects_negative_manifest_pointer(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["manifest_entry_identity"]["entry_pointer"] = "/entries/-1"
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"


def test_corrected_report_rejects_shared_invalid_protocol(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    for role in ("alignment_mean_text_log", "alignment_mean_text_log_symmetric"):
        path = tmp_path / role / "run_result.json"
        result = json.loads(path.read_text())
        result["protocol"]["selection_metric"] = "wrong_metric"
        path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    campaign = payload["campaigns"][0]
    assert all(
        summary["paired_fixed_delta"]["count"] == 0
        for summary in campaign["summaries"]
    )
    assert all(
        pair["status"] != "MATCHED"
        for pairs in campaign["pairs"].values()
        for pair in pairs
    )


def test_corrected_report_binds_all_manifest_path_declarations(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["manifest_path"] = str(
        tmp_path / "alignment_control" / "manifest.json"
    )
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert any("path differs" in error for error in run["errors"])


def test_corrected_report_requires_stable_manifest_identity_hash(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    manifest = tmp_path / "alignment_control" / "manifest.json"
    path = tmp_path / "alignment_control" / "run_result.json"
    result = json.loads(path.read_text())
    result["manifest_entry_identity"]["manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_control"
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert "manifest SHA-256 mismatch" in run["errors"]


def test_corrected_report_validates_nested_protocol_and_inference(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["protocol"]["official_unseen_used_for_selection"] = True
    result["inference_contract"]["photo_prompt_used_for_gallery"] = False
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert "corrected protocol mismatch: official_unseen_used_for_selection" in run[
        "errors"
    ]
    assert "corrected inference contract mismatch: photo_prompt_used_for_gallery" in run[
        "errors"
    ]


def test_corrected_mode_cannot_be_downgraded_by_top_level_campaign(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_control" / "run_result.json"
    result = json.loads(path.read_text())
    result["campaign"] = "objective_alignment_pilot_2026-09-05"
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for campaign in payload["campaigns"]
        for run in campaign["runs"]
        if run["artifact_path"] == str(path.resolve())
    )
    assert run["matching_mode"] == "corrected_v2"
    assert run["status"] == "ARTIFACT_INVALID"


def test_corrected_report_hashes_and_rejects_extra_inference_contract_fields(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path)
    control_path = tmp_path / "alignment_control" / "run_result.json"
    control = json.loads(control_path.read_text())
    control["inference_contract"]["unexpected"] = False
    control_path.write_text(json.dumps(control))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    control_run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_control"
    )
    assert any(
        error == "corrected inference contract unexpected:unexpected"
        for error in control_run["errors"]
    )
    assert _pair(payload, "alignment_mean_text_log")["status"] == "UNMATCHED"


def test_corrected_report_requires_calibration_top_level_identity(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    calibration_path = tmp_path / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["initial_model_state_hash"] = "stale-init"
    calibration_path.write_text(json.dumps(calibration))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert any(
        mismatch["field"] == "calibration_artifact.initial_model_state_hash"
        for mismatch in run["calibration_validation"]["mismatches"]
    )


def test_corrected_report_rejects_zero_batch_calibration_without_crashing(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path)
    calibration_path = tmp_path / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["calibration"]["batches"] = 0
    calibration_path.write_text(json.dumps(calibration))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"


def test_corrected_report_rejects_non_chronological_history(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    path = tmp_path / "alignment_mean_text_log" / "run_result.json"
    result = json.loads(path.read_text())
    result["history"].reverse()
    path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert "history steps are not chronological" in run["errors"]


def test_corrected_report_rejects_zero_calibration_denominator(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    calibration_path = tmp_path / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["base"]["sketch_gradient_norms"][0] = 0.0
    calibration["detached"]["unweighted_sketch_ratios"][0] = None
    calibration["detached"]["unweighted_sketch_ratio_reasons"][0] = "zero_denominator"
    calibration_path.write_text(json.dumps(calibration))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert any(
        mismatch["field"] == "calibration_artifact.unweighted_sketch_ratios"
        for mismatch in run["calibration_validation"]["mismatches"]
    )


def test_corrected_report_requires_calibration_split_identity(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    calibration_path = tmp_path / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration.pop("split_identity")
    calibration_path.write_text(json.dumps(calibration))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run
        for run in payload["campaigns"][0]["runs"]
        if run["experiment_role"] == "alignment_mean_text_log"
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert any(
        mismatch["field"] == "calibration_artifact.split_identity"
        for mismatch in run["calibration_validation"]["mismatches"]
    )


def test_report_keeps_malformed_run_result_fail_closed(tmp_path: Path) -> None:
    _write_runs(tmp_path)
    bad_path = tmp_path / "malformed" / "run_result.json"
    bad_path.parent.mkdir()
    bad_path.write_text("{")

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    run = next(
        run for campaign in payload["campaigns"] for run in campaign["runs"]
        if run["artifact_path"] == str(bad_path.resolve())
    )
    assert run["status"] == "ARTIFACT_INVALID"
    assert any("run_result is unreadable" in error for error in run["errors"])


def test_corrected_manifest_rejects_malformed_existing_entries(
    tmp_path: Path,
) -> None:
    config_path, split_identity, _ = _data_fixture(tmp_path)
    path = tmp_path / "corrected_pilot_manifest_v2.json"
    base = {
        "data_config": str(config_path),
        "dataset": "toy",
        "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
        "training_seed": 42,
        "pseudo_validation_seed": 3407,
        "split_identity": split_identity,
        "source_hash": "source",
        "training_horizon": 2,
    }
    config = _base_config("alignment_control")
    ensure_corrected_run_manifest(
        path,
        role="alignment_control",
        resolved_config=config,
        initial_model_state_hash="init-r",
        replicate_id="pilot-seed42-r",
        **base,
    )
    manifest = json.loads(path.read_text())
    manifest["entries"].append(None)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="incomplete entry"):
        ensure_corrected_run_manifest(
            path,
            role="alignment_mean_text_log",
            resolved_config={**config, "experiment_role": "alignment_mean_text_log"},
            initial_model_state_hash="init-md",
            replicate_id="pilot-seed42-md",
            **base,
        )


def test_corrected_manifest_rejects_incomplete_existing_entry(
    tmp_path: Path,
) -> None:
    config_path, split_identity, _ = _data_fixture(tmp_path)
    path = tmp_path / "corrected_pilot_manifest_v2.json"
    base = {
        "data_config": str(config_path),
        "dataset": "toy",
        "campaign": ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
        "training_seed": 42,
        "pseudo_validation_seed": 3407,
        "split_identity": split_identity,
        "source_hash": "source",
        "training_horizon": 2,
    }
    config = _base_config("alignment_control")
    ensure_corrected_run_manifest(
        path,
        role="alignment_control",
        resolved_config=config,
        initial_model_state_hash="init-r",
        replicate_id="pilot-seed42-r",
        **base,
    )
    manifest = json.loads(path.read_text())
    manifest["entries"][0].pop("treatment")
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="incomplete entry"):
        ensure_corrected_run_manifest(
            path,
            role="alignment_mean_text_log",
            resolved_config={**config, "experiment_role": "alignment_mean_text_log"},
            initial_model_state_hash="init-md",
            replicate_id="pilot-seed42-md",
            **base,
        )


def test_historical_corrected_manifest_keeps_legacy_matching_mode(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path)
    for role in ROLES:
        run_dir = tmp_path / role
        current = run_dir / "manifest.json"
        legacy = run_dir / "corrected_pilot_manifest.json"
        current.rename(legacy)
        historical_manifest = json.loads(legacy.read_text())
        historical_manifest.pop("manifest_marker", None)
        legacy.write_text(json.dumps(historical_manifest))
        result_path = run_dir / "run_result.json"
        result = json.loads(result_path.read_text())
        result["manifest_path"] = str(legacy)
        result["manifest_entry_identity"]["manifest_path"] = str(legacy)
        result["resolved_config"]["experiment_manifest_path"] = legacy.name
        result_path.write_text(json.dumps(result))

    payload = write_report(tmp_path, tmp_path / "report.md", horizon=2)
    campaign = payload["campaigns"][0]
    assert campaign["matching_mode"] == "historical"
    assert _pair(payload, "alignment_mean_text_log")["status"] == "MATCHED"
