"""Trust-boundary loader for canonical filename-derived Sketchy pairings."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .manifest import ManifestEntry

_SCHEMA_VERSION = 1
_PAIRING_KIND = "filename_convention_canonical"
_CANONICAL_PHOTO_PARTS = (
    "256x256",
    "photo",
    "tx_000000000000_ready",
)
_SUFFIX = re.compile(r"-\d+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(raw: Any, *, root: Path, field: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay below dataset root: {raw!r}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes dataset root: {raw!r}") from error
    if not resolved.is_file():
        raise ValueError(f"{field} does not exist: {resolved}")
    return path.as_posix(), resolved


def _entry_index(entries: Sequence[ManifestEntry], *, name: str) -> dict[str, ManifestEntry]:
    result: dict[str, ManifestEntry] = {}
    for entry in entries:
        key = str(entry.path.resolve())
        if key in result:
            raise ValueError(f"{name} entries contain duplicate path: {key}")
        result[key] = entry
    return result


def load_pairing_manifest(
    path: Path,
    *,
    dataset_root: Path,
    sketch_entries: Sequence[ManifestEntry],
    photo_entries: Sequence[ManifestEntry],
) -> dict[str, ManifestEntry]:
    """Load and verify a pairing for exactly the supplied pseudo-train scope."""
    root = dataset_root.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pairing manifest must be a JSON object")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Unsupported pairing manifest schema_version")
    if payload.get("pairing_kind") != _PAIRING_KIND:
        raise ValueError("Pairing manifest has an unsupported pairing_kind")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Pairing manifest records must be a list")

    sketches = _entry_index(sketch_entries, name="Sketch")
    photos = _entry_index(photo_entries, name="Photo")
    result: dict[str, ManifestEntry] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Pairing records must be objects")
        sketch_raw, sketch_path = _relative_path(
            record.get("sketch_path"), root=root, field="sketch_path"
        )
        photo_raw, photo_path = _relative_path(
            record.get("photo_path"), root=root, field="photo_path"
        )
        sketch_key = str(sketch_path)
        photo_key = str(photo_path)
        if sketch_key in result:
            raise ValueError(f"Duplicate sketch pairing: {sketch_raw}")
        if sketch_key not in sketches:
            raise ValueError(f"Pairing sketch is outside supplied scope: {sketch_raw}")
        photo_relative = photo_path.relative_to(root)
        if photo_relative.parts[:3] != _CANONICAL_PHOTO_PARTS:
            raise ValueError(f"Pairing photo is not in canonical branch: {photo_raw}")
        if photo_key not in photos:
            raise ValueError(f"Pairing photo is outside supplied scope: {photo_raw}")
        sketch = sketches[sketch_key]
        photo = photos[photo_key]
        if record.get("label") != sketch.label or photo.label != sketch.label:
            raise ValueError(f"Pairing labels differ for {sketch_raw}")
        if _SUFFIX.sub("", sketch_path.stem) != photo_path.stem:
            raise ValueError(
                f"Pairing filename stems differ for {sketch_raw} -> {photo_raw}"
            )
        if sketch_path.parent.name != photo_path.parent.name:
            raise ValueError(
                f"Pairing class folders differ for {sketch_raw} -> {photo_raw}"
            )
        if record.get("sketch_sha256") != _sha256(sketch_path):
            raise ValueError(f"Sketch SHA256 mismatch: {sketch_raw}")
        if record.get("photo_sha256") != _sha256(photo_path):
            raise ValueError(f"Photo SHA256 mismatch: {photo_raw}")
        result[sketch_key] = photo

    if set(result) != set(sketches):
        missing = sorted(set(sketches) - set(result))
        extra = sorted(set(result) - set(sketches))
        raise ValueError(f"Pairing coverage mismatch; missing={missing[:3]}, extra={extra[:3]}")
    return result
