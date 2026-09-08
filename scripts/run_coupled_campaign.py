#!/usr/bin/env python3
"""Tiny launcher for the approved coupled-predictive campaign.

Inert unless --launch is passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spica.provenance import capture_provenance  # noqa: E402

APPROVED_ARMS = ("R0", "R1", "R1_SIG")
PROTOCOL_DOC = ROOT / "docs/designs/coupled_predictive_region_v1/region_first_protocol.md"
ARCHITECTURE_DOC = ROOT / "docs/designs/coupled_predictive_region_v1/architecture.md"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_arms(value: str) -> list[str]:
    arms = [item.strip() for item in value.split(",") if item.strip()]
    if tuple(arms) != APPROVED_ARMS:
        raise argparse.ArgumentTypeError("--arms must be exactly R0,R1,R1_SIG")
    return arms


def _doc_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}


def _diagnostic_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS" or payload.get("verified") is not True:
        raise ValueError("diagnostic receipt must have status PASS and verified true")
    value = payload.get("lambda_sig")
    if value is None and isinstance(payload.get("selection"), Mapping):
        value = payload["selection"].get("lambda_sig")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("diagnostic receipt must contain a verified positive lambda_sig")
    required = {
        "formula_identity", "architecture_identity", "gradient_batch_source",
        "component_sha256", "initialization_hashes", "parameter_scope",
    }
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"diagnostic receipt is missing required proof: {missing}")
    if payload.get("batches") != 4 or payload.get("batch_size") != 32 or payload.get("rho") != 0.1:
        raise ValueError("diagnostic receipt must be the locked 4xB32 rho=0.1 diagnostic")
    if payload.get("parameter_scope") != "model.student_visual.transformer.resblocks[-1]":
        raise ValueError("diagnostic parameter scope is not the approved last student block")
    components = payload["component_sha256"]
    if not isinstance(components, Mapping) or not components:
        raise ValueError("diagnostic component_sha256 is not a mapping")
    for name, digest in components.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"invalid diagnostic component path: {name!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid diagnostic component digest: {name!r}")
    return {"path": str(path), "sha256": _sha256_file(path), "lambda_sig": float(value), "receipt": payload}


def _validate_diagnostic_against_archive(diagnostic: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
    receipt = diagnostic["receipt"]
    manifest = {str(row["path"]): str(row["sha256"]) for row in snapshot.get("manifest", []) if isinstance(row, Mapping)}
    for name, digest in receipt["component_sha256"].items():
        if manifest.get(str(name)) != str(digest):
            raise ValueError(f"diagnostic component is not identical to campaign archive: {name}")


def _make_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["WANDB_MODE"] = "online"
    return env


def _copy_source_archive(root: Path, manifest: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    archive = root / "source_snapshot"
    files_root = archive / "files"
    snapshot = provenance["source_snapshot"]
    for item in snapshot["manifest"]:
        relative = Path(str(item["path"]))
        source = ROOT / relative
        destination = files_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"source snapshot file disappeared: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256_file(destination) != item["sha256"]:
            raise RuntimeError(f"source archive hash mismatch: {relative}")
    _json_atomic(archive / "index.json", snapshot)
    _json_atomic(archive / "provenance.json", provenance)
    _json_atomic(archive / "execution_manifest.json", manifest)


def _runtime_template(root: Path, arms: list[str], source_hash: str | None = None) -> dict[str, Any]:
    return {
        "status": "INERT",
        "campaign_id": root.name,
        "campaign_root": str(root),
        "arms": arms,
        "completed_arms": [],
        "current_arm": None,
        "child_pid": None,
        "source_snapshot_hash": source_hash,
        "started_at": _now(),
        "arms_state": {},
    }


def _arm_reported_source_hash(run_dir: Path) -> str | None:
    result_path = run_dir / "run_result.json"
    if result_path.is_file():
        result = _read_json(result_path)
        if isinstance(result, Mapping):
            for key in ("source_snapshot_hash", "source_hash"):
                if isinstance(result.get(key), str):
                    return result[key]
    provenance_path = run_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = _read_json(provenance_path)
        snapshot = provenance.get("source_snapshot") if isinstance(provenance, Mapping) else None
        if isinstance(snapshot, Mapping) and isinstance(snapshot.get("sha256"), str):
            return snapshot["sha256"]
    return None


def _start_child(command: list[str], *, cwd: Path, env: Mapping[str, str], stdout: Path, stderr: Path) -> tuple[subprocess.Popen[bytes], Any, Any]:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    out = stdout.open("ab")
    err = stderr.open("ab")
    try:
        process = subprocess.Popen(command, cwd=cwd, env=dict(env), stdout=out, stderr=err, start_new_session=True)
    except Exception:
        out.close()
        err.close()
        raise
    return process, out, err


def _launch(args: argparse.Namespace) -> int:
    root = Path(args.output_dir).expanduser().resolve()
    try:
        relative_root = root.relative_to(ROOT)
    except ValueError:
        relative_root = None
    if relative_root is not None and (not relative_root.parts or relative_root.parts[0] != "outputs"):
        raise ValueError("campaign root inside the repository must be under outputs/")
    if root.exists():
        raise FileExistsError(f"refusing to use existing output directory: {root}")
    diagnostic = None
    if "R1_SIG" in args.arms:
        if not args.diagnostic:
            raise ValueError("--diagnostic is required when R1_SIG is selected")
        diagnostic = _diagnostic_payload(Path(args.diagnostic).expanduser().resolve())

    resolved = {
        "campaign": "coupled_predictive_v1",
        "arms": args.arms,
        "trainer": "spica.train_coupled_predictive",
        "device": "cuda",
        "wandb_mode": "online",
        "fixed_protocol_doc": _doc_payload(PROTOCOL_DOC),
        "architecture_doc": _doc_payload(ARCHITECTURE_DOC),
        "diagnostic": None if diagnostic is None else {k: v for k, v in diagnostic.items() if k != "receipt"},
    }
    provenance = capture_provenance(ROOT, resolved_config=resolved, command=[sys.executable, *sys.argv])
    snapshot = provenance.get("source_snapshot") if isinstance(provenance, Mapping) else None
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("sha256"), str):
        raise RuntimeError("provenance.source_snapshot is unavailable")
    source_hash = str(snapshot["sha256"])

    root.mkdir(parents=True)
    runtime_path = root / "runtime.json"
    runtime = _runtime_template(root, args.arms, source_hash)
    _json_atomic(runtime_path, runtime)

    manifest = {
        "status": "LAUNCHING",
        "campaign_id": root.name,
        "campaign_root": str(root),
        "created_at": _now(),
        "head_commit": provenance.get("head_commit"),
        "source_snapshot_hash": source_hash,
        "source_snapshot": snapshot,
        "source_proof": "provenance.source_snapshot",
        "resolved_config": resolved,
        "diagnostic_receipt": diagnostic,
    }
    _json_atomic(root / "execution_manifest.json", manifest)
    _copy_source_archive(root, manifest, provenance)
    if diagnostic is not None:
        _validate_diagnostic_against_archive(diagnostic, snapshot)
    diagnostic_for_child = None
    if diagnostic is not None:
        diagnostic_for_child = root / "diagnostic_receipt.json"
        shutil.copyfile(Path(diagnostic["path"]), diagnostic_for_child)
        diagnostic["archived_path"] = str(diagnostic_for_child)
        manifest["diagnostic_receipt"] = diagnostic
        _json_atomic(root / "execution_manifest.json", manifest)
        _json_atomic(root / "source_snapshot" / "execution_manifest.json", manifest)

    env = _make_env()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    logs = root / "logs"
    runs = root / "runs"
    runtime.update({"status": "LAUNCHING", "launch_started_at": _now()})
    _json_atomic(runtime_path, runtime)

    for arm in args.arms:
        current_source = capture_provenance(ROOT).get("source_snapshot")
        current_hash = current_source.get("sha256") if isinstance(current_source, Mapping) else None
        if current_hash != source_hash:
            runtime.update({"status": "SOURCE_CHANGED", "failed_arm": arm, "current_arm": arm, "current_source_snapshot_hash": current_hash})
            _json_atomic(runtime_path, runtime)
            return 1
        run_dir = runs / arm
        command = [
            sys.executable,
            "-m",
            "spica.train_coupled_predictive",
            "--arm",
            arm,
            "--output-dir",
            str(run_dir),
            "--campaign-root",
            str(root),
            "--campaign-id",
            root.name,
        ]
        if arm == "R1_SIG":
            command += ["--diagnostic", str(diagnostic_for_child or Path(args.diagnostic).expanduser().resolve())]
        command += ["--device", "cuda", "--wandb-mode", "online"]

        state = {
            "command": command,
            "start_time": _now(),
            "expected_source_snapshot_hash": source_hash,
            "stdout": str(logs / f"{arm}.stdout.log"),
            "stderr": str(logs / f"{arm}.stderr.log"),
        }
        runtime.update({"status": "LAUNCHING", "current_arm": arm, "child_pid": None})
        runtime["arms_state"][arm] = state
        _json_atomic(runtime_path, runtime)

        process, out, err = _start_child(command, cwd=ROOT, env=env, stdout=logs / f"{arm}.stdout.log", stderr=logs / f"{arm}.stderr.log")
        state["pid"] = process.pid
        runtime["child_pid"] = process.pid
        _json_atomic(runtime_path, runtime)
        try:
            exit_code = process.wait()
        finally:
            out.close()
            err.close()
        reported_hash = _arm_reported_source_hash(run_dir)
        state.update({"exit_time": _now(), "exit_code": exit_code, "reported_source_snapshot_hash": reported_hash})
        runtime.update({"child_pid": None})
        if exit_code != 0:
            runtime.update({"status": "FAILED", "failed_arm": arm, "current_arm": arm})
            _json_atomic(runtime_path, runtime)
            return exit_code or 1
        if reported_hash != source_hash:
            runtime.update({"status": "SOURCE_MISMATCH", "failed_arm": arm, "current_arm": arm})
            _json_atomic(runtime_path, runtime)
            return 1
        runtime["completed_arms"].append(arm)
        runtime.update({"current_arm": None, "status": "LAUNCHING"})
        _json_atomic(runtime_path, runtime)

    if tuple(args.arms) != APPROVED_ARMS:
        runtime.update({"status": "ARMS_FINISHED_UNVERIFIED", "finished_at": _now()})
        _json_atomic(runtime_path, runtime)
        return 0

    _json_atomic(root / "execution_manifest.json", {**_read_json(root / "execution_manifest.json"), "status": "ARMS_FINISHED"})
    runtime.update({"status": "VERIFYING", "current_arm": None, "child_pid": None, "verification": {"command": [sys.executable, "scripts/verify_coupled_campaign.py", "--campaign-root", str(root), "--replay"], "start_time": _now()}})
    _json_atomic(runtime_path, runtime)
    verify_command = runtime["verification"]["command"]
    _json_atomic(root / "execution_manifest.json", {**_read_json(root / "execution_manifest.json"), "status": "VERIFYING"})
    _json_atomic(runtime_path, runtime)
    process, out, err = _start_child(verify_command, cwd=ROOT, env=env, stdout=logs / "verification.stdout.log", stderr=logs / "verification.stderr.log")
    runtime["child_pid"] = process.pid
    runtime["verification"]["pid"] = process.pid
    _json_atomic(runtime_path, runtime)
    try:
        exit_code = process.wait()
    finally:
        out.close()
        err.close()
    runtime["verification"].update({"exit_time": _now(), "exit_code": exit_code})
    runtime["child_pid"] = None
    final_status = "VERIFIED" if exit_code == 0 else "VERIFICATION_FAILED"
    runtime.update({"status": final_status, "finished_at": _now()})
    _json_atomic(root / "execution_manifest.json", {**_read_json(root / "execution_manifest.json"), "status": final_status})
    _json_atomic(runtime_path, runtime)
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the approved coupled-predictive campaign; inert unless --launch.")
    parser.add_argument("--output-dir", required=True, help="fresh campaign root")
    parser.add_argument("--diagnostic", help="verified SIGReg diagnostic receipt JSON")
    parser.add_argument("--arms", type=_parse_arms, default=list(APPROVED_ARMS), help="approved sequence, default R0,R1,R1_SIG")
    parser.add_argument("--launch", action="store_true", help="actually start training subprocesses")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.launch:
        print(json.dumps({"status": "INERT", "would_launch": False, "output_dir": args.output_dir, "arms": args.arms}, sort_keys=True))
        return 0
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
