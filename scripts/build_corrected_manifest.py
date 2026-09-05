"""Build a provenance-preserving schema-v2 alignment artifact index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _path(value: object) -> Path:
    return Path(str(value)).expanduser()


def _source_hash(result: dict[str, Any]) -> str | None:
    return result.get("source_snapshot_hash") or result.get("provenance", {}).get(
        "source_snapshot", {}
    ).get("sha256")


def _step0(result: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in result.get("history", [])
        if int(row.get("training_global_step", -1)) == 0
    ]
    if not rows:
        return {"status": "UNVERIFIED_NO_STEP0"}
    path = _path(rows[0].get("checkpoint", ""))
    if not path.is_file():
        return {"status": "MISSING", "checkpoint": str(path)}
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - artifact input
        return {"status": "UNREADABLE", "checkpoint": str(path), "error": str(error)}
    if not isinstance(payload, dict):
        return {"status": "INVALID", "checkpoint": str(path)}
    return {
        "status": "VERIFIED",
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "initial_model_state_hash": payload.get("initial_model_state_hash"),
        "backbone_identity": payload.get("backbone_identity"),
    }


def _entry(result: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(result["_artifact_path"])
    split = result.get("pseudo_split_identity", {})
    initialization = _step0(result)
    config = result.get("resolved_config", {})
    config_hash = _hash_json(config)
    seed = result.get("training_seed", result.get("seed"))
    role = str(result.get("experiment_role"))
    campaign = str(result.get("campaign"))
    replicate_id = f"{result.get('run_kind', 'unknown')}:{artifact_path.parent.name}"
    checkpoints: list[dict[str, Any]] = []
    checkpoint_status = "VERIFIED"
    for row in result.get("history", []):
        path = _path(row.get("checkpoint", ""))
        expected = row.get("checkpoint_sha256")
        if not path.is_file():
            checkpoint_status = "MISSING"
            checkpoints.append(
                {"step": row.get("training_global_step"), "path": str(path), "status": "MISSING"}
            )
            continue
        actual = _sha256(path)
        status = "VERIFIED" if expected == actual and expected else "MISMATCH"
        if status != "VERIFIED":
            checkpoint_status = status
        checkpoints.append(
            {
                "step": int(row.get("training_global_step")),
                "path": str(path),
                "metadata_sha256": expected,
                "actual_sha256": actual,
                "status": status,
            }
        )
    source_hash = _source_hash(result)
    if checkpoint_status == "MISMATCH":
        status = "ARTIFACT_HASH_MISMATCH"
    elif checkpoint_status == "MISSING":
        status = "ARTIFACT_MISSING"
    elif source_hash is None:
        status = "UNVERIFIED_SOURCE"
    else:
        status = "UNVERIFIED_SOURCE_SNAPSHOT"
    return {
        "run_id": _hash_json(
            {
                "artifact_path": str(artifact_path),
                "campaign": campaign,
                "role": role,
                "seed": seed,
            }
        ),
        "status": status,
        "experiment_role": role,
        "campaign": campaign,
        "training_seed": None if seed is None else int(seed),
        "pseudo_validation_seed": result.get(
            "pseudo_validation_seed",
            result.get("split_seed", config.get("pseudo_val_seed")),
        ),
        "replicate_id": replicate_id,
        "config_hash": config_hash,
        "source_hash": source_hash,
        "source_snapshot_evidence": {
            "embedded_in_run_result": bool(result.get("provenance", {}).get("source_snapshot")),
            "standalone_snapshot_available": False,
            "verification": "metadata_only; source snapshot bytes are not an artifact",
        },
        "checkpoint_paths": checkpoints,
        "training_horizon": config.get("max_steps"),
        "initialization_identity": initialization,
        "treatment": result.get("resolved_treatment", {}),
        "dataset_identity": {
            "dataset": result.get("dataset"),
            "split_identity": split,
            "data_manifest_identity": result.get("manifest_identity"),
        },
        "evaluation_protocol": result.get("protocol", {}),
        "artifact": {
            "run_result": str(artifact_path),
            "manifest_identity": result.get("manifest_entry_identity"),
        },
        "corrections": [
            "training_seed copied from run_result rather than historical manifest entry",
            "config/source/checkpoint hashes recomputed or explicitly marked unverified",
            "one entry emitted per observed run and replicate",
        ],
    }


def build(input_root: Path, output: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for path in sorted(input_root.glob("**/run_result.json")):
        result = json.loads(path.read_text())
        result["_artifact_path"] = str(path.resolve())
        results.append(result)
    if not results:
        raise ValueError(f"no run_result.json below {input_root}")
    entries = [_entry(result) for result in results]
    group_counts: dict[tuple[Any, ...], int] = {}
    for entry in entries:
        key = (
            entry["campaign"],
            entry["experiment_role"],
            entry["training_seed"],
            entry["dataset_identity"]["split_identity"].get("sha256"),
            entry["initialization_identity"].get("initial_model_state_hash"),
            entry["training_horizon"],
        )
        group_counts[key] = group_counts.get(key, 0) + 1
    duplicates: list[str] = []
    for entry in entries:
        key = (
            entry["campaign"],
            entry["experiment_role"],
            entry["training_seed"],
            entry["dataset_identity"]["split_identity"].get("sha256"),
            entry["initialization_identity"].get("initial_model_state_hash"),
            entry["training_horizon"],
        )
        count = group_counts[key]
        entry["duplicate_count"] = count
        if count > 1:
            entry["status"] = "DUPLICATE"
            duplicates.append(entry["run_id"])
    source_groups: dict[str, set[str | None]] = {}
    for entry in entries:
        source_groups.setdefault(entry["campaign"], set()).add(entry["source_hash"])
    for entry in entries:
        hashes = sorted(value for value in source_groups[entry["campaign"]] if value)
        entry["campaign_source_hashes"] = hashes
        entry["campaign_source_match"] = len(hashes) <= 1 and bool(hashes)
    campaigns = sorted({entry["campaign"] for entry in entries})
    payload = {
        "schema_version": 2,
        "index_type": "corrected_alignment_artifact_index",
        "input_root": str(input_root.resolve()),
        "campaigns": campaigns,
        "entries": sorted(entries, key=lambda entry: entry["run_id"]),
        "duplicate_run_ids": sorted(duplicates),
        "original_manifests": sorted(
            {
                json.dumps(
                    result.get("manifest_identity"), sort_keys=True
                )
                for result in results
                if result.get("manifest_identity")
            }
        ),
        "matching_limitations": [
            "Historical manifest entries are preserved by link; this index uses observed run_result seed/role identities.",
            "Historical source snapshot hashes are recorded metadata; standalone source snapshot bytes were not found, so exact source reconstruction is unverified.",
            "No run is labelled perfectly matched when campaign source hashes differ or source bytes are unavailable.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"entries": len(entries), "campaigns": campaigns, "duplicates": len(duplicates)}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.input_root, args.output)


if __name__ == "__main__":
    main()
