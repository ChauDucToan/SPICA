"""Predeclared treatments and identities for the alignment campaign."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
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
CORRECTED_MANIFEST_MARKER = "spica_corrected_alignment_manifest_v2"
CORRECTED_PILOT_INFERENCE_CONTRACT: dict[str, Any] = {
    "required_inputs": ["raw_sketch_image"],
    "text_required": False,
    "photo_required": False,
    "oracle_class_required": False,
    "text_used_for_predictor": False,
    "photo_prompt_used_for_query": False,
    "photo_prompt_used_for_gallery": True,
}
ALL_ALIGNMENT_ROLES = tuple(dict.fromkeys(ALIGNMENT_ROLES + CORRECTED_PILOT_ROLES))


def corrected_pilot_inference_contract_mismatches(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["<contract>"]
    expected_keys = set(CORRECTED_PILOT_INFERENCE_CONTRACT)
    mismatches = [
        key
        for key, expected in CORRECTED_PILOT_INFERENCE_CONTRACT.items()
        if key not in value
        or type(value[key]) is not type(expected)
        or value[key] != expected
    ]
    mismatches.extend(
        f"unexpected:{key}" for key in sorted(set(value) - expected_keys)
    )
    return sorted(mismatches)


def validate_corrected_pilot_inference_contract(value: Any) -> None:
    mismatches = corrected_pilot_inference_contract_mismatches(value)
    if mismatches:
        raise ValueError(
            "corrected pilot inference contract mismatch: "
            + ", ".join(mismatches)
        )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def manifest_identity_sha256(manifest: dict[str, Any]) -> str:
    """Hash immutable manifest metadata; corrected entries are append-only."""
    return canonical_sha256(
        {key: value for key, value in manifest.items() if key != "entries"}
    )


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
    if campaign == ALIGNMENT_CORRECTED_PILOT_CAMPAIGN:
        raise ValueError("corrected pilot requires ensure_corrected_run_manifest")
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
    new_entry = entry

    def validate_entry(item: Any) -> None:
        required = {
            "run_id",
            "status",
            "experiment_role",
            "campaign",
            "training_seed",
            "pseudo_validation_seed",
            "replicate_id",
            "config_hash",
            "source_hash",
            "training_horizon",
            "initialization_identity",
            "treatment",
            "dataset_identity",
        }
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("corrected manifest contains an incomplete entry")
        item_role = item["experiment_role"]
        item_seed = item["training_seed"]
        item_pseudo_seed = item["pseudo_validation_seed"]
        if (
            not isinstance(item["run_id"], str)
            or not item["run_id"]
            or item["status"] != "REGISTERED"
            or item_role not in CORRECTED_PILOT_ROLES
            or item["campaign"] != campaign
            or isinstance(item_seed, bool)
            or not isinstance(item_seed, int)
            or isinstance(item_pseudo_seed, bool)
            or not isinstance(item_pseudo_seed, int)
            or not isinstance(item["replicate_id"], str)
            or not item["replicate_id"]
            or not isinstance(item["config_hash"], str)
            or not item["config_hash"]
            or (item["source_hash"] is not None and not isinstance(item["source_hash"], str))
            or isinstance(item["training_horizon"], bool)
            or not isinstance(item["training_horizon"], int)
            or item["training_horizon"] < 0
        ):
            raise ValueError("corrected manifest contains an invalid entry")
        initialization = item["initialization_identity"]
        if (
            not isinstance(initialization, dict)
            or not isinstance(initialization.get("initial_model_state_hash"), str)
            or not initialization["initial_model_state_hash"]
        ):
            raise ValueError("corrected manifest entry initialization is invalid")
        treatment = item["treatment"]
        expected_treatment = treatment_for_role(
            item_role,
            seed=item_seed,
            pseudo_val_seed=item_pseudo_seed,
        )
        if not isinstance(treatment, dict):
            raise ValueError("corrected manifest entry treatment is invalid")
        for key, value in expected_treatment.items():
            if key == "lambda_alignment_mean" and item_role != "alignment_control":
                continue
            if treatment.get(key) != value:
                raise ValueError("corrected manifest entry treatment is invalid")
        if any(
            not isinstance(treatment.get(key), (int, float))
            or isinstance(treatment.get(key), bool)
            or not math.isfinite(float(treatment.get(key)))
            for key in ("lambda_alignment_mean", "lambda_alignment_covariance")
        ):
            raise ValueError("corrected manifest entry treatment is invalid")
        dataset_identity = item["dataset_identity"]
        if (
            not isinstance(dataset_identity, dict)
            or dataset_identity.get("dataset") != dataset
            or dataset_identity.get("data_config") != data_config
            or not isinstance(dataset_identity.get("split_identity"), dict)
        ):
            raise ValueError("corrected manifest entry dataset identity is invalid")
        split_entry = dataset_identity["split_identity"]
        if split_entry.get("sha256") != canonical_sha256(
            {key: value for key, value in split_entry.items() if key != "sha256"}
        ):
            raise ValueError("corrected manifest entry split identity is invalid")
        if item["run_id"] != canonical_sha256(
            {
                "campaign": item["campaign"],
                "role": item_role,
                "training_seed": item_seed,
                "split_identity": split_entry,
                "config_hash": item["config_hash"],
                "replicate_id": item["replicate_id"],
            }
        ):
            raise ValueError("corrected manifest entry run_id is invalid")

    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_symlink() or (
            path.exists() and path.lstat().st_nlink != 1
        ):
            raise ValueError(
                "corrected manifest path must be a regular, single-link file: "
                f"{path}"
            )
        validate_entry(new_entry)
        arm_ids: set[tuple[str, int, str]] = set()
        if path.exists():
            try:
                manifest = json.loads(path.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"corrected manifest is unreadable: {path}") from error
            if not isinstance(manifest, dict):
                raise ValueError(f"corrected manifest is not an object: {path}")
            expected_metadata = {
                "schema_version": 2,
                "manifest_marker": CORRECTED_MANIFEST_MARKER,
                "campaign": campaign,
                "dataset": dataset,
                "data_config": data_config,
                "roles": list(CORRECTED_PILOT_ROLES),
            }
            if any(
                canonical_sha256(manifest.get(key)) != canonical_sha256(value)
                for key, value in expected_metadata.items()
            ):
                raise ValueError(
                    "corrected manifest is not the matching mutable v2 manifest: "
                    f"{path}"
                )
            entries = manifest.get("entries")
            if not isinstance(entries, list):
                raise ValueError(f"corrected manifest entries are invalid: {path}")
            run_ids: set[str] = set()
            for existing_entry in entries:
                validate_entry(existing_entry)
                existing_run_id = existing_entry["run_id"]
                arm_id = (
                    existing_entry["experiment_role"],
                    existing_entry["training_seed"],
                    existing_entry["config_hash"],
                )
                if existing_run_id in run_ids or arm_id in arm_ids:
                    raise ValueError(f"corrected manifest contains duplicate entry: {path}")
                run_ids.add(existing_run_id)
                arm_ids.add(arm_id)
            protocol = manifest.get("protocol")
            expected_protocol = {
                "selection_metric": "full_pseudo_unseen_mAP",
                "official_unseen_used_for_selection": False,
                "text_used_for_predictor": False,
                "photo_used_for_predictor": False,
                "train_only_alignment_targets": True,
            }
            if not isinstance(protocol, dict) or any(
                canonical_sha256(protocol.get(key)) != canonical_sha256(value)
                for key, value in expected_protocol.items()
            ):
                raise ValueError(f"corrected manifest protocol is invalid: {path}")
        else:
            manifest = {
                "schema_version": 2,
                "manifest_marker": CORRECTED_MANIFEST_MARKER,
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
        existing = [item for item in manifest["entries"] if item.get("run_id") == run_id]
        new_arm_id = (role, training_seed, config_hash)
        if not existing and new_arm_id in arm_ids:
            raise ValueError("corrected manifest contains duplicate role/seed/config entry")
        if existing and existing[0] != new_entry:
            raise ValueError("corrected manifest entry already differs")
        if not existing:
            manifest["entries"].append(new_entry)
            _write_json_atomic(path, manifest)
        return manifest, manifest_identity_sha256(manifest)


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
            (index, item)
            for index, item in enumerate(entries)
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
        index, entry = matches[0]
        pointer = f"/entries/{index}"
    else:
        entry = None
        pointer = ""
    if not isinstance(entry, dict):
        raise ValueError(f"manifest has no entry for {role}")
    expected_manifest_hash = (
        manifest_identity_sha256(manifest)
        if manifest.get("manifest_marker") == CORRECTED_MANIFEST_MARKER
        and type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == 2
        else manifest_sha256
    )
    if manifest_sha256 != expected_manifest_hash:
        raise ValueError("manifest identity hash does not match manifest contents")
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": expected_manifest_hash,
        "entry_pointer": pointer,
        "entry_sha256": canonical_sha256(entry),
    }
