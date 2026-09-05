"""Predeclared treatments and identities for the alignment campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ALIGNMENT_PILOT_CAMPAIGN = "objective_alignment_pilot_2026-09-05"
ALIGNMENT_CAMPAIGN = "objective_alignment_2026-09-05"
ALIGNMENT_REPLICATION_CAMPAIGN = "objective_alignment_replication_2026-09-05"
ALIGNMENT_ROLES = (
    "alignment_control",
    "alignment_mean_text_log",
    "alignment_cov_text_log",
    "alignment_full_text_log",
    "alignment_full_chordal",
    "alignment_full_photo_anchor",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def treatment_for_role(role: str, *, seed: int, pseudo_val_seed: int) -> dict[str, Any]:
    if role not in ALIGNMENT_ROLES:
        raise ValueError(f"unknown alignment role: {role}")
    geometry = "chordal" if role == "alignment_full_chordal" else "log_map"
    anchor = "photo_mean" if role == "alignment_full_photo_anchor" else "text"
    mean_weight = 0.0 if role == "alignment_cov_text_log" else 1.0
    covariance_weight = 0.0 if role == "alignment_mean_text_log" else 1.0
    if role == "alignment_control":
        mean_weight = 0.0
        covariance_weight = 0.0
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
    path: Path, manifest: dict[str, Any], *, role: str, manifest_sha256: str
) -> dict[str, Any]:
    entry = manifest.get("entries", {}).get(role)
    if not isinstance(entry, dict):
        raise ValueError(f"manifest has no entry for {role}")
    return {
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "entry_pointer": f"/entries/{role}",
        "entry_sha256": canonical_sha256(entry),
    }
