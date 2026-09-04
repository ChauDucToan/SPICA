"""Reproducible source, environment, and RNG provenance for experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import torch

_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".lock"}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_snapshot(root: Path) -> dict[str, Any]:
    """Hash every tracked or untracked, non-ignored source/config file."""
    names = _git(
        root, "ls-files", "--cached", "--others", "--exclude-standard"
    ).splitlines()
    names = sorted(
        name
        for name in names
        if not name.startswith("outputs/") and Path(name).suffix in _SOURCE_SUFFIXES
    )
    manifest = []
    aggregate = hashlib.sha256()
    for name in names:
        data = (root / name).read_bytes()
        digest = _sha256(data)
        manifest.append({"path": name, "sha256": digest, "bytes": len(data)})
        aggregate.update(name.encode())
        aggregate.update(b"\0")
        aggregate.update(data)
        aggregate.update(b"\0")
    return {
        "sha256": aggregate.hexdigest(),
        "file_count": len(manifest),
        "manifest": manifest,
    }


def capture_provenance(
    root: Path,
    *,
    resolved_config: dict[str, Any] | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Capture enough information to identify a dirty source tree exactly."""
    try:
        commit = _git(root, "rev-parse", "HEAD").strip()
        tracked_status = _git(
            root, "status", "--porcelain", "--untracked-files=no"
        ).splitlines()
        untracked_files = _git(
            root, "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        status = _git(
            root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        patch = _git(root, "diff", "--binary", "HEAD", "--", ".", ":(exclude)outputs")
        snapshot = source_snapshot(root)
    except (OSError, subprocess.CalledProcessError) as error:
        return {"status": "unavailable", "reason": str(error)}
    dependencies = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return {
        "status": "valid",
        "head_commit": commit,
        "working_tree_state": "dirty" if status else "clean",
        "tracked_working_tree_state": "dirty" if tracked_status else "clean",
        "tracked_dirty_files": tracked_status,
        "untracked_files": sorted(untracked_files),
        "dirty_files": status,
        "git_diff": patch if tracked_status else "",
        "git_diff_sha256": _sha256(patch.encode()),
        "source_snapshot": snapshot,
        "resolved_config": resolved_config,
        "command": command or sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch": str(torch.__version__),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "dependencies": dependencies,
            "environment_variables": {
                key: os.environ[key]
                for key in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")
                if key in os.environ
            },
        },
    }


def capture_rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
        "data_loader_generator": None if generator is None else generator.get_state(),
    }


def restore_rng_state(
    state: dict[str, Any], generator: torch.Generator | None = None
) -> None:
    required = {"python", "torch_cpu", "torch_cuda", "data_loader_generator"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"checkpoint RNG state is missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if generator is not None and state["data_loader_generator"] is not None:
        generator.set_state(state["data_loader_generator"])


def hash_json(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
