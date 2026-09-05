"""Predeclared treatments and identities for the alignment campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ALIGNMENT_PILOT_CAMPAIGN = "objective_alignment_pilot_2026-09-05"
ALIGNMENT_CAMPAIGN = "objective_alignment_2026-09-05"
ALIGNMENT_REPLICATION_CAMPAIGN = "objective_alignment_replication_2026-09-05"
ALIGNMENT_CORRECTED_PILOT_CAMPAIGN = "objective_alignment_corrected_pilot_2026-09-05"
ALIGNMENT_ROLES = (
    "alignment_control",
    "alignment_mean_text_log",
    "alignment_cov_text_log",
    "alignment_full_text_log",
    "alignment_full_chordal",
    "alignment_full_photo_anchor",
)
CORRECTED_PILOT_ROLES = (
    "alignment_control",
    "alignment_mean_text_log",
    "alignment_mean_text_log_symmetric",
)
ALL_ALIGNMENT_ROLES = tuple(dict.fromkeys(ALIGNMENT_ROLES + CORRECTED_PILOT_ROLES))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def treatment_for_role(role: str, *, seed: int, pseudo_val_seed: int) -> dict[str, Any]:
    if role not in ALL_ALIGNMENT_ROLES:
        raise ValueError(f"unknown alignment role: {role}")
    geometry = "chordal" if role == "alignment_full_chordal" else "log_map"
    anchor = "photo_mean" if role == "alignment_full_photo_anchor" else "text"
    mean_weight = 0.0 if role == "alignment_cov_text_log" else 1.0
    covariance_weight = 0.0 if role in {"alignment_mean_text_log", "alignment_mean_text_log_symmetric"} else 1.0
    if role == "alignment_control":
        mean_weight = 0.0
        covariance_weight = 0.0
    target_gradient = "symmetric" if role == "alignment_mean_text_log_symmetric" else "detached"
    return {
        "visual_prompt_length": 3,
        "prompt_mode": "prompt_only",
        "text_mode": "soft",
        "train_visual_layernorm": False,
        "train_sketch_prompt": True,
        "train_photo_prompt": True,
        "lambda_rank": 1.0,
        "lambda_cls": 1.0,
        "classification_location": "query",
        "encoder_mode": "frozen",
        "encoder_unfreeze_depth": 0,
        "encoder_train_ln_post": False,
        "transport_enabled": False,
        "transport_mode": "none",
        "num_positive_photos": 4,
        "batch_size": 32,
        "classes_per_batch": 16,
        "sketches_per_class": 2,
        "alignment_geometry": geometry,
        "alignment_anchor": anchor,
        "alignment_target_gradient": target_gradient,
        "lambda_alignment_mean": mean_weight,
        "lambda_alignment_covariance": covariance_weight,
        "seed": seed,
        "pseudo_val_seed": pseudo_val_seed,
        "official_unseen_used_for_selection": False,
    }


def treatment_from_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = treatment_for_role(
        str(config.get("experiment_role")),
        seed=int(config.get("seed")),
        pseudo_val_seed=int(config.get("pseudo_val_seed")),
    )
    return {key: config.get(key) for key in keys}


def make_manifest(
    *, data_config: str, dataset: str, campaign: str
) -> dict[str, Any]:
    if campaign not in {
        ALIGNMENT_PILOT_CAMPAIGN,
        ALIGNMENT_CAMPAIGN,
        ALIGNMENT_REPLICATION_CAMPAIGN,
        ALIGNMENT_CORRECTED_PILOT_CAMPAIGN,
    }:
        raise ValueError(f"unknown alignment campaign: {campaign}")
    result = {
        "schema_version": 1,
        "campaign": campaign,
        "dataset": dataset,
        "data_config": data_config,
        "selection_metric": "full_pseudo_unseen_mAP",
        "official_unseen_used_for_selection": False,
        "objective": "class_conditional_spherical_moment_alignment",
        "roles": list(ALIGNMENT_ROLES),
        "entries": {
            role: {
                "experiment_role": role,
                "campaign": campaign,
                "resolved_treatment": treatment_for_role(
                    role, seed=42, pseudo_val_seed=3407
                ),
                "dataset": dataset,
                "data_config": data_config,
                "training_seed": 42,
                "pseudo_validation_seed": 3407,
                "official_unseen_used_for_selection": False,
            }
            for role in ALIGNMENT_ROLES
        },
        "protocol": {
            "selection_metric": "full_pseudo_unseen_mAP",
            "official_unseen_used_for_selection": False,
            "text_used_for_predictor": False,
            "photo_used_for_predictor": False,
            "train_only_alignment_targets": True,
            "positive_photo_sampling": "matched_class_batch_then_random_without_replacement",
        },
    }
    if campaign == ALIGNMENT_REPLICATION_CAMPAIGN:
        result["predeclared_training_seeds"] = [42, 123, 3407]
        result["replication_campaign"] = True
    return result


def ensure_corrected_run_manifest(
    path: Path,
    *,
    data_config: str,
    dataset: str,
    campaign: str,
    role: str,
    training_seed: int,
    pseudo_validation_seed: int,
    split_identity: dict[str, Any],
    resolved_config: dict[str, Any],
    source_hash: str | None,
    initial_model_state_hash: str,
    training_horizon: int,
    replicate_id: str,
) -> tuple[dict[str, Any], str]:
    """Register one corrected run without touching historical manifests."""
    if campaign != ALIGNMENT_CORRECTED_PILOT_CAMPAIGN:
        raise ValueError("corrected run manifest requires the corrected pilot campaign")
    if role not in CORRECTED_PILOT_ROLES:
        raise ValueError(f"role is not registered for corrected pilot: {role}")
    config_hash = canonical_sha256(resolved_config)
    run_id = canonical_sha256(
        {
            "campaign": campaign,
            "role": role,
            "training_seed": training_seed,
            "split_identity": split_identity,
            "config_hash": config_hash,
            "replicate_id": replicate_id,
        }
    )
    entry = {
        "run_id": run_id,
        "status": "REGISTERED",
        "experiment_role": role,
        "campaign": campaign,
        "training_seed": training_seed,
        "pseudo_validation_seed": pseudo_validation_seed,
        "replicate_id": replicate_id,
        "config_hash": config_hash,
        "source_hash": source_hash,
        "checkpoint_paths": [],
        "checkpoint_hashes": {},
        "training_horizon": training_horizon,
        "initialization_identity": {
            "initial_model_state_hash": initial_model_state_hash,
        },
        "treatment": treatment_for_role(
            role, seed=training_seed, pseudo_val_seed=pseudo_validation_seed
        )
        | {
            "lambda_alignment_mean": resolved_config.get("lambda_alignment_mean"),
            "lambda_alignment_covariance": resolved_config.get("lambda_alignment_covariance"),
        },
        "dataset_identity": {
            "dataset": dataset,
            "data_config": data_config,
            "split_identity": split_identity,
        },
    }
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != 2 or not isinstance(manifest.get("entries"), list):
            raise ValueError(f"corrected manifest has incompatible schema: {path}")
    else:
        manifest = {
            "schema_version": 2,
            "campaign": campaign,
            "dataset": dataset,
            "data_config": data_config,
            "roles": list(CORRECTED_PILOT_ROLES),
            "entries": [],
            "original_manifests": [],
            "protocol": {
                "selection_metric": "full_pseudo_unseen_mAP",
                "official_unseen_used_for_selection": False,
                "text_used_for_predictor": False,
                "photo_used_for_predictor": False,
                "train_only_alignment_targets": True,
            },
        }
    if manifest.get("campaign") != campaign:
        raise ValueError("corrected manifest campaign mismatch")
    existing = [item for item in manifest["entries"] if item.get("run_id") == run_id]
    if existing and existing[0] != entry:
        raise ValueError("corrected manifest entry already differs")
    if not existing:
        manifest["entries"].append(entry)
        manifest["entries"].sort(key=lambda item: item["run_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_manifest(
    path: Path, *, data_config: str, dataset: str, campaign: str
) -> tuple[dict[str, Any], str]:
    expected = make_manifest(
        data_config=data_config, dataset=dataset, campaign=campaign
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = json.loads(path.read_text())
        if actual != expected:
            raise ValueError(f"alignment manifest already differs: {path}")
    else:
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return expected, hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_entry_identity(
    path: Path,
    manifest: dict[str, Any],
    *,
    role: str,
    manifest_sha256: str,
    training_seed: int | None = None,
    config_hash: str | None = None,
) -> dict[str, Any]:
    entries = manifest.get("entries", {})
    if isinstance(entries, dict):
        entry = entries.get(role)
        pointer = f"/entries/{role}"
    elif isinstance(entries, list):
        matches = [
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("experiment_role") == role
            and (training_seed is None or item.get("training_seed") == training_seed)
            and (config_hash is None or item.get("config_hash") == config_hash)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"manifest must have exactly one matching entry for {role}/seed{training_seed}/config{config_hash}, "
                f"got {len(matches)}"
            )
        entry = matches[0]
        pointer = f"/entries/{entry['run_id']}"
    else:
        entry = None
        pointer = ""
    if not isinstance(entry, dict):
        raise ValueError(f"manifest has no entry for {role}")
    return {
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "entry_pointer": pointer,
        "entry_sha256": canonical_sha256(entry),
    }
