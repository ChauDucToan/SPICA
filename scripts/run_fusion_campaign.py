#!/usr/bin/env python3
"""Detached-safe sequential F2/F2_SIG launcher; inert unless --launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("F2", "F2_SIG")
CAMPAIGN_ID = "coupled_predictive_fusion_v2"
METRICS = (
    "full_mAP",
    "P@200",
    "mAP@200_prefix_positive",
    "mAP@200_all_relevant",
    "mAP@200_min_relevant_k",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _helpers() -> Any:
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    return __import__("run_coupled_campaign")


def _json(path: Path, value: Any) -> None:
    _helpers()._json_atomic(path, value)


def _now() -> str:
    return _helpers()._now()


def _components(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.component_sha256 must be an object")
    result = {}
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError(f"invalid {label} component path: {name!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid {label} component digest: {name!r}")
        result[name] = digest
    return result


def _diagnostic(path: Path) -> dict[str, Any]:
    payload = _helpers()._diagnostic_payload(path)
    receipt = payload["receipt"]
    config = receipt.get("resolved_config", {})
    if (
        config.get("architecture") != "predictive_fusion_v2"
        or config.get("positive_pool") != "full"
        or config.get("task_identity") != "mean_views(rank_i+ce_i)"
    ):
        raise ValueError("diagnostic is not the verified F2 receipt")
    value = payload["lambda_sig"]
    source = {**config, **receipt}
    if any(
        source.get(k) != v
        for k, v in (("rho", 0.1), ("batches", 4), ("batch_size", 32))
    ):
        raise ValueError("diagnostic must be rho=0.1, 4 batches, B32")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("diagnostic lambda_sig must be positive")
    return {
        "path": str(path),
        "sha256": _sha(path),
        "lambda_sig": value,
        "receipt": receipt,
    }


def _gate(path: Path, diagnostic_sha: str) -> dict[str, Any]:
    value = _read(path)
    if not isinstance(value, Mapping) or value.get("status") != "PASS":
        raise ValueError("gate receipt must have status PASS")
    if value.get("diagnostic_receipt_sha256") != diagnostic_sha:
        raise ValueError("gate diagnostic_receipt_sha256 does not match receipt bytes")
    components = _components(value.get("component_sha256"), "gate")
    if not components:
        raise ValueError("gate receipt requires non-empty component_sha256")
    return {
        "path": str(path),
        "sha256": _sha(path),
        "receipt": dict(value),
        "component_sha256": components,
    }


def _check_components(
    root: Path, snapshot: Mapping[str, Any], groups: Mapping[str, Mapping[str, str]]
) -> None:
    archived = {str(row["path"]): str(row["sha256"]) for row in snapshot["manifest"]}
    for group in groups.values():
        for name, digest in group.items():
            if (
                archived.get(name) != digest
                or _sha(ROOT / name) != digest
                or _sha(root / "source_snapshot/files" / name) != digest
            ):
                raise ValueError(f"source component mismatch: {name}")


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def _selection(value: Any) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("step", "path", "sha256", "resolved_config_sha256")
    } | {
        condition: {
            metric: value.get("metrics", {}).get(condition, {}).get(metric)
            for metric in METRICS
        }
        for condition in ("clean", "masked_macro")
    }


def _summary(root: Path) -> dict[str, Any]:
    arms = {}
    for arm in ARMS:
        result = _read(root / "runs" / arm / "run_result.json")
        arms[arm] = {
            name: _selection(result["selections"][name])
            for name in ("latest", "best_clean", "best_masked")
        }
    a, b = arms["F2"]["best_clean"], arms["F2_SIG"]["best_clean"]
    criteria = {
        "clean_prefix_strict": b["clean"]["mAP@200_prefix_positive"]
        > a["clean"]["mAP@200_prefix_positive"],
        "clean_p200_nondec": b["clean"]["P@200"] >= a["clean"]["P@200"],
        "clean_full_map_nondec": b["clean"]["full_mAP"] >= a["clean"]["full_mAP"],
        "masked_prefix_strict": b["masked_macro"]["mAP@200_prefix_positive"]
        > a["masked_macro"]["mAP@200_prefix_positive"],
    }
    return {
        "status": "ARMS_FINISHED_UNVERIFIED",
        "arms": arms,
        "source_snapshot_hash": _read(root / "execution_manifest.json")[
            "source_snapshot_hash"
        ],
        "wandb_urls": {
            arm: _read(root / "runs" / arm / "run_result.json")["wandb_url"]
            for arm in ARMS
        },
        "promotion": {
            "candidate": "F2_SIG",
            "promote": all(criteria.values()),
            "criteria": criteria,
        },
    }


def _launch(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from spica.provenance import capture_provenance

    helpers = _helpers()
    root = Path(args.output_dir).expanduser().resolve()
    try:
        if not root.relative_to(ROOT / "outputs").parts:
            raise ValueError
    except ValueError as error:
        raise ValueError(
            "--output-dir must be a fresh child of repository outputs/"
        ) from error
    if root.exists():
        raise FileExistsError(f"refusing to use existing output directory: {root}")
    if _free_gib(root.parent) < 40:
        raise RuntimeError("less than 40 GiB free before campaign")
    diagnostic = _diagnostic(Path(args.diagnostic).expanduser().resolve())
    gate = _gate(Path(args.gate).expanduser().resolve(), diagnostic["sha256"])
    resolved = {
        "campaign": CAMPAIGN_ID,
        "arms": list(ARMS),
        "trainer": "spica.train_coupled_predictive",
        "device": "cuda",
        "wandb_mode": "online",
        "max_steps": 3600,
        "diagnostic_receipt_sha256": diagnostic["sha256"],
        "gate_receipt_sha256": gate["sha256"],
    }
    provenance = capture_provenance(
        ROOT, resolved_config=resolved, command=[sys.executable, *sys.argv]
    )
    snapshot = (
        provenance.get("source_snapshot") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("sha256"), str):
        raise RuntimeError("source provenance unavailable")
    root.mkdir(parents=True)
    source_hash = str(snapshot["sha256"])
    runtime = {
        "status": "LAUNCHING",
        "campaign_id": root.name,
        "campaign_root": str(root),
        "arms": list(ARMS),
        "completed_arms": [],
        "current_arm": None,
        "child_pid": None,
        "source_snapshot_hash": source_hash,
        "started_at": _now(),
        "arms_state": {},
    }
    manifest = {
        "status": "LAUNCHING",
        "campaign_id": root.name,
        "campaign_root": str(root),
        "created_at": _now(),
        "head_commit": provenance.get("head_commit"),
        "source_snapshot_hash": source_hash,
        "source_snapshot": snapshot,
        "resolved_config": resolved,
        "diagnostic_receipt": {k: v for k, v in diagnostic.items() if k != "receipt"},
        "gate_receipt": {k: v for k, v in gate.items() if k != "receipt"},
    }
    try:
        _json(root / "runtime.json", runtime)
        _json(root / "execution_manifest.json", manifest)
        helpers._copy_source_archive(root, manifest, provenance)
        helpers._validate_diagnostic_against_archive(diagnostic, snapshot)
        groups = {
            "diagnostic": _components(
                diagnostic["receipt"].get("component_sha256"), "diagnostic"
            ),
            "gate": gate["component_sha256"],
        }
        _check_components(root, snapshot, groups)
        for item, name in (
            (diagnostic, "diagnostic_receipt.json"),
            (gate, "gate_receipt.json"),
        ):
            shutil.copyfile(item["path"], root / name)
            if _sha(root / name) != item["sha256"]:
                raise RuntimeError(f"archived {name} hash mismatch")
        manifest["archived_receipts"] = {
            name[:-5]: {"path": name, "sha256": item["sha256"]}
            for item, name in (
                (diagnostic, "diagnostic_receipt.json"),
                (gate, "gate_receipt.json"),
            )
        }
        _json(root / "execution_manifest.json", manifest)
        _json(root / "source_snapshot/execution_manifest.json", manifest)
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "WANDB_MODE": "online",
                "PYTHONPATH": str(ROOT / "src")
                + os.pathsep
                + env.get("PYTHONPATH", ""),
            }
        )
        logs, runs = root / "logs", root / "runs"
        for arm in ARMS:
            if _free_gib(root.parent) < 15:
                raise RuntimeError(f"less than 15 GiB free before {arm}")
            current = capture_provenance(ROOT).get("source_snapshot")
            if not isinstance(current, Mapping) or current.get("sha256") != source_hash:
                raise RuntimeError(f"source changed before {arm}")
            _check_components(root, snapshot, groups)
            command = [
                sys.executable,
                "-m",
                "spica.train_coupled_predictive",
                "--arm",
                arm,
                "--output-dir",
                str(runs / arm),
                "--campaign-root",
                str(root),
                "--campaign-id",
                CAMPAIGN_ID,
                "--device",
                "cuda",
                "--wandb-mode",
                "online",
                "--max-steps",
                "3600",
            ]
            if arm == "F2_SIG":
                command += ["--diagnostic", str(root / "diagnostic_receipt.json")]
            state = {
                "command": command,
                "start_time": _now(),
                "expected_source_snapshot_hash": source_hash,
                "stdout": str(logs / f"{arm}.stdout.log"),
                "stderr": str(logs / f"{arm}.stderr.log"),
            }
            runtime["arms_state"][arm] = state
            runtime.update({"current_arm": arm, "child_pid": None})
            _json(root / "runtime.json", runtime)
            process, out, err = helpers._start_child(
                command,
                cwd=ROOT,
                env=env,
                stdout=logs / f"{arm}.stdout.log",
                stderr=logs / f"{arm}.stderr.log",
            )
            state["pid"] = process.pid
            runtime["child_pid"] = process.pid
            _json(root / "runtime.json", runtime)
            try:
                code = process.wait()
            finally:
                out.close()
                err.close()
            result = (
                _read(runs / arm / "run_result.json")
                if (runs / arm / "run_result.json").is_file()
                else {}
            )
            state.update(
                {
                    "exit_time": _now(),
                    "exit_code": code,
                    "training_status": result.get("status"),
                    "completed_step": result.get("step"),
                    "reported_source_snapshot_hash": result.get("source_snapshot_hash"),
                }
            )
            runtime["child_pid"] = None
            _json(root / "runtime.json", runtime)
            valid = (
                code == 0
                and result.get("status") == "COMPLETE"
                and result.get("arm") == arm
                and result.get("campaign") == CAMPAIGN_ID
                and result.get("step") == 3600
                and result.get("source_snapshot_hash") == source_hash
            )
            if not valid:
                runtime.update({"status": "FAILED", "failed_arm": arm})
                _json(root / "runtime.json", runtime)
                return code or 1
            runtime["completed_arms"].append(arm)
            runtime["current_arm"] = None
            _json(root / "runtime.json", runtime)
        runtime.update({"status": "ARMS_FINISHED_UNVERIFIED", "finished_at": _now()})
        _json(root / "runtime.json", runtime)
        _json(root / "summary.json", _summary(root))
        manifest.update({"status": "ARMS_FINISHED_UNVERIFIED", "finished_at": _now()})
        _json(root / "execution_manifest.json", manifest)
        return 0
    except BaseException as error:
        runtime.update(
            {"status": "FAILED", "error": repr(error), "finished_at": _now()}
        )
        _json(root / "runtime.json", runtime)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--diagnostic")
    parser.add_argument("--gate")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--cpu-self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_self_check:
        assert ARMS == ("F2", "F2_SIG") and len(METRICS) == 5
        print(json.dumps({"status": "PASS", "check": "pure_runner_contract"}))
        return 0
    if not args.launch:
        print(
            json.dumps(
                {
                    "status": "INERT",
                    "would_launch": False,
                    "output_dir": args.output_dir,
                    "arms": list(ARMS),
                }
            )
        )
        return 0
    if not args.output_dir or not args.diagnostic or not args.gate:
        parser.error("--launch requires --output-dir, --diagnostic, and --gate")
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
