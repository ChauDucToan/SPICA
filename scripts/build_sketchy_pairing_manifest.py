"""Build a deterministic canonical Sketchy sketch-to-photo pairing manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from spica.config.data import load_data_config
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split, split_manifest_identity

_SUFFIX = re.compile(r"-\d+$")
_SOURCE = Path("/home/oslamelon/Downloads/Research/ZS BIR dataset/Sketchy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_key(path: Path) -> str:
    return _SUFFIX.sub("", path.stem)


def _source_audit(
    *,
    records: list[dict[str, object]],
    dataset_root: Path,
    source_root: Path,
    output: Path,
) -> tuple[Path, str, dict[str, int | str | bool]]:
    audit_path = output.with_name(f"{output.stem}.source_audit.json")
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {audit_path}")
    files_by_path: dict[str, dict[str, str | bool]] = {}
    for record in records:
        for field in ("sketch_path", "photo_path"):
            relative = Path(str(record[field]))
            repo_path = dataset_root / relative
            source_path = source_root / relative
            if not source_path.is_file():
                raise ValueError(f"Missing source counterpart for {repo_path}: {source_path}")
            repo_sha = str(record[field.replace("_path", "_sha256")])
            source_sha = _sha256(source_path)
            if source_sha != repo_sha:
                raise ValueError(
                    f"Source hash mismatch for {repo_path}: {source_path}"
                )
            item = {
                "path": relative.as_posix(),
                "repo_sha256": repo_sha,
                "source_sha256": source_sha,
                "match": True,
            }
            previous = files_by_path.setdefault(relative.as_posix(), item)
            if previous != item:
                raise ValueError(f"Conflicting source hashes for {repo_path}")
    files = list(files_by_path.values())
    audit = {
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "read_only": True,
        "status": "verified",
        "file_count": len(files),
        "matched_count": sum(bool(item["match"]) for item in files),
        "files": files,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit_path, _sha256(audit_path), {
        "status": "verified",
        "file_count": len(files),
        "matched_count": len(files),
        "artifact": audit_path.as_posix(),
    }

def build(
    *,
    config_path: Path,
    output: Path,
    seed: int,
    num_classes: int,
    source_root: Path = _SOURCE,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    config = load_data_config(config_path)
    train_sketches = read_manifest(config.train.sketch_manifest, config.root)
    train_photos = read_manifest(config.train.photo_manifest, config.root)
    class_names = read_class_map(config.train.class_map)
    split = make_classwise_retrieval_split(
        train_sketches,
        train_photos,
        class_names,
        num_validation_classes=num_classes,
        seed=seed,
    )

    canonical: dict[str, list] = {}
    for entry in split.train_photo_entries:
        relative = entry.path.resolve().relative_to(config.root.resolve())
        if relative.parts[:3] == ("256x256", "photo", "tx_000000000000_ready"):
            canonical.setdefault(entry.path.stem, []).append(entry)
    records: list[dict[str, object]] = []
    for sketch in split.train_sketch_entries:
        key = _pair_key(sketch.path)
        candidates = canonical.get(key, [])
        if len(candidates) != 1:
            raise ValueError(f"Expected one canonical photo for {sketch.path}, got {len(candidates)}")
        photo = candidates[0]
        if photo.label != sketch.label:
            raise ValueError(f"Pairing label mismatch: {sketch.path} -> {photo.path}")
        records.append(
            {
                "sketch_path": sketch.path.resolve().relative_to(config.root.resolve()).as_posix(),
                "photo_path": photo.path.resolve().relative_to(config.root.resolve()).as_posix(),
                "label": sketch.label,
                "sketch_sha256": _sha256(sketch.path),
                "photo_sha256": _sha256(photo.path),
            }
        )

    canonical_photo_paths = {
        entry.path.resolve()
        for entries in canonical.values()
        for entry in entries
    }
    mapped_photo_paths = {config.root / record["photo_path"] for record in records}
    if canonical_photo_paths != mapped_photo_paths:
        missing = sorted(str(path) for path in canonical_photo_paths - mapped_photo_paths)
        extra = sorted(str(path) for path in mapped_photo_paths - canonical_photo_paths)
        raise ValueError(
            "Canonical photo pool does not equal mapped photo pool: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    source_audit_path, source_audit_sha256, source_audit = _source_audit(
        records=records,
        dataset_root=config.root.resolve(),
        source_root=source_root.resolve(),
        output=output,
    )

    manifest_identity = split_manifest_identity(
        split,
        dataset_name=config.name,
        dataset_root=config.root,
        manifest_paths={
            "train_sketch": config.train.sketch_manifest,
            "train_photo": config.train.photo_manifest,
            "class_map": config.train.class_map,
        },
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "pairing_kind": "filename_convention_canonical",
        "status": "FILENAME_CONVENTION_CANONICAL",
        "independent_annotation": False,
        "source": {
            "path": str(source_root.resolve()),
            "read_only": True,
            "source_image_hashes_preverified": True,
            "source_hash_evidence": source_audit_path.as_posix(),
            "source_hash_evidence_sha256": source_audit_sha256,
            "source_audit": source_audit,
        },
        "dataset": {"name": config.name, "root": str(config.root)},
        "scope": "pseudo_train_only",
        "split": {
            "pseudo_validation_seed": seed,
            "pseudo_validation_num_classes": num_classes,
            "train_class_ids": list(split.train_class_ids),
            "heldout_class_ids": list(split.validation_class_ids),
            "train_sketch_count": len(split.train_sketch_entries),
            "train_photo_count": len(split.train_photo_entries),
            "heldout_sketch_count": len(split.validation_sketch_entries),
            "heldout_photo_count": len(split.validation_photo_entries),
            "canonical_photo_pool_count": len(canonical_photo_paths),
            "mapped_photo_pool_count": len(mapped_photo_paths),
            "canonical_photo_pool_equals_mapping": True,
            "leakage_check": "heldout classes and rows excluded from records",
        },
        "manifest_identity": manifest_identity,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument("--source-root", type=Path, default=_SOURCE)
    parser.add_argument("--pseudo-val-seed3407", action="store_true")
    parser.add_argument("--pseudo-val-num-classes", type=int, default=20)
    args = parser.parse_args()
    seed = 3407 if args.pseudo_val_seed3407 else 3407
    payload = build(
        config_path=args.data_config,
        output=args.output,
        seed=seed,
        num_classes=args.pseudo_val_num_classes,
        source_root=args.source_root,
    )
    print(json.dumps({"output": str(args.output), "records": len(payload["records"])}, sort_keys=True))


if __name__ == "__main__":
    main()
