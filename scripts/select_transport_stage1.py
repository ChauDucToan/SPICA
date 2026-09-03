"""Select the corrected two-stage origin from pseudo-unseen validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.transport_artifact_utils import ROOT, write_new
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from transport_artifact_utils import ROOT, write_new

from spica.provenance import capture_provenance

CANDIDATE_STEPS = (44, 73)
SELECTION_METRIC = "pseudo_unseen_validation_mAP"


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("Stage-1 candidate mAP is missing")
    value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("Stage-1 candidate mAP is non-finite")
    return value


def select_stage1(run_result_path: Path) -> dict[str, Any]:
    result = json.loads(run_result_path.read_text())
    if not isinstance(result, dict):
        raise ValueError("Stage-1 run_result must be a JSON object")
    config = result.get("config")
    if (
        not isinstance(config, dict)
        or config.get("experiment_role") != "two_stage_stage1"
    ):
        raise ValueError("run_result is not the corrected Stage-1 role")
    if config.get("transport_enabled") is not False:
        raise ValueError("Stage-1 selection requires transport_enabled=false")
    if config.get("train_class_scope") != "pseudo_train":
        raise ValueError("Stage-1 selection requires pseudo_train scope")
    split = result.get("data_split_identity")
    if not isinstance(split, dict) or not split.get("sha256"):
        raise ValueError("Stage-1 run_result has no split identity")
    manifest_identity = result.get("data_manifest_identity")
    if manifest_identity is not None and (
        not isinstance(manifest_identity, dict) or not manifest_identity.get("sha256")
    ):
        raise ValueError("Stage-1 run_result has an invalid manifest identity")
    provenance = result.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("status") != "valid":
        raise ValueError("Stage-1 provenance is incomplete")

    by_step: dict[int, dict[str, Any]] = {}
    for point in result.get("probe_history", []):
        if not isinstance(point, dict):
            continue
        try:
            step = int(point["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if step not in CANDIDATE_STEPS:
            continue
        if step in by_step:
            raise ValueError(f"duplicate Stage-1 candidate checkpoint step: {step}")
        protocol = point.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("val_is_pseudo_unseen") is not True
        ):
            raise ValueError(
                f"Stage-1 step {step} lacks pseudo-unseen validation protocol"
            )
        if protocol.get("official_test_is_diagnostic_only") is not True:
            raise ValueError(f"Stage-1 step {step} has unsafe official-test protocol")
        checkpoint = _resolve(point.get("checkpoint"))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or int(payload.get("step", -1)) != step:
            raise ValueError(f"Stage-1 checkpoint step mismatch at {checkpoint}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Stage-1 checkpoint metadata missing: {checkpoint}")
        if metadata.get("experiment_role") != "two_stage_stage1":
            raise ValueError(f"checkpoint role mismatch at {checkpoint}")
        if metadata.get("transport_enabled") is not False:
            raise ValueError(f"Stage-1 checkpoint is not a base origin: {checkpoint}")
        if metadata.get("data_split_identity") != split:
            raise ValueError(f"Stage-1 checkpoint split mismatch at {checkpoint}")
        if payload.get("data_manifest_identity") != manifest_identity:
            raise ValueError(
                f"Stage-1 checkpoint manifest identity mismatch at {checkpoint}"
            )
        by_step[step] = {
            "step": step,
            "mAP": _number(
                point.get("val", {}).get("mAP")
                if isinstance(point.get("val"), dict)
                else None
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        }
    if sorted(by_step) != list(CANDIDATE_STEPS):
        raise ValueError(
            f"Stage-1 selection requires exactly candidate checkpoints {list(CANDIDATE_STEPS)}"
        )
    candidates = [by_step[step] for step in CANDIDATE_STEPS]
    selected = max(candidates, key=lambda item: (item["mAP"], -item["step"]))
    return {
        "schema_version": 1,
        "selection_metric": SELECTION_METRIC,
        "source_run_result": str(run_result_path.resolve()),
        "source_role": config["experiment_role"],
        "source_campaign": config.get("experiment_campaign"),
        "candidates": candidates,
        "selected": selected,
        "data_split_identity": split,
        "data_manifest_identity": manifest_identity,
        "official_unseen_used": False,
        "source_provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_result = (
        args.run_result if args.run_result.is_absolute() else ROOT / args.run_result
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    selection = select_stage1(run_result)
    selection["provenance"] = capture_provenance(
        ROOT, command=[str(value) for value in __import__("sys").argv]
    )
    write_new(output, json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
