#!/usr/bin/env python3
"""CPU-only audit for the independent official benchmark data adapter.

This script validates metadata and path identities for the complete TU-Berlin
220/30 and QuickDraw 80/30 protocols.  It decodes only two training examples
per benchmark and never iterates an official test loader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spica.data.coupled_benchmark import (  # noqa: E402
    _cross_path_overlap,
    _validate_path_entries,
    build_test_loaders,
    load_benchmark_protocol,
    make_train_loader,
    prepare_batch,
)
from spica.data.manifest import ManifestEntry, read_manifest  # noqa: E402
from spica.data.transforms import build_clip_eval_transform  # noqa: E402

OUT = ROOT / "outputs/coupled_benchmark_data_audit_20260910"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _owned_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src/spica/data/coupled_benchmark.py",
        ROOT / "scripts/check_coupled_benchmark_data_cpu.py",
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _expect_error(errors: list[dict[str, str]], name: str, callback: Any) -> None:
    try:
        callback()
    except (OSError, TypeError, ValueError) as error:
        errors.append({"case": name, "error": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError(f"synthetic rejection did not fail: {name}")


def _synthetic_checks(errors: list[dict[str, str]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="coupled_benchmark_audit_") as raw:
        root = Path(raw)
        (root / "class").mkdir()
        (root / "class" / "a.png").write_bytes(b"fixture")
        names = {0: "a", 1: "b"}
        good = (ManifestEntry(root / "class" / "a.png", 0),)
        malformed_manifest = root / "malformed.txt"
        malformed_manifest.write_text("class/a.png nope\n", encoding="utf-8")
        traversal_manifest = root / "traversal.txt"
        traversal_manifest.write_text("../escape.png 0\n", encoding="utf-8")
        _expect_error(errors, "malformed_label", lambda: read_manifest(malformed_manifest, root))
        _expect_error(errors, "manifest_path_traversal", lambda: read_manifest(traversal_manifest, root))
        _expect_error(
            errors,
            "missing_file",
            lambda: _validate_path_entries(
                (ManifestEntry(root / "class" / "missing.png", 0),),
                root=root,
                modality="synthetic sketch",
                class_names={0: "a"},
            ),
        )
        _expect_error(
            errors,
            "duplicate_resolved_path",
            lambda: _validate_path_entries(
                good + (ManifestEntry(root / "class" / "a.png", 0),),
                root=root,
                modality="synthetic sketch",
                class_names={0: "a"},
            ),
        )
        _expect_error(
            errors,
            "unknown_label",
            lambda: _validate_path_entries(
                (ManifestEntry(root / "class" / "a.png", 1),),
                root=root,
                modality="synthetic sketch",
                class_names={0: "a"},
            ),
        )
        _expect_error(
            errors,
            "missing_class_coverage",
            lambda: _validate_path_entries(
                good,
                root=root,
                modality="synthetic sketch",
                class_names=names,
            ),
        )
        traversal = ManifestEntry(root / ".." / "escape.png", 0)
        _expect_error(
            errors,
            "path_traversal",
            lambda: _validate_path_entries(
                (traversal,), root=root, modality="synthetic sketch", class_names={0: "a"}
            ),
        )
        overlap = _cross_path_overlap(
            {
                "train": {"sketch": good, "photo": ()},
                "test": {"sketch": (), "photo": good},
            }
        )
        if not overlap:
            raise AssertionError("synthetic cross-split overlap was not detected")
        errors.append({"case": "cross_split_path_overlap", "error": "ValueError", "message": overlap[0]})
    return {"cases": len(errors), "all_rejections_observed": True}


def _audit_benchmark(name: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    protocol = load_benchmark_protocol(name, split="train")
    transform = build_clip_eval_transform(224)
    random.seed(20260910)
    loader = make_train_loader(protocol, transform, batch_size=2, num_workers=0, drop_last=False)
    raw_batch = next(iter(loader))
    prepared = prepare_batch(protocol, raw_batch, step=0)
    expected_batch_keys = {
        "sketch", "positive_photos", "negative_photo", "label", "negative_label",
        "sketch_path", "positive_photo_paths", "negative_photo_path",
    }
    if set(raw_batch) != expected_batch_keys:
        raise AssertionError(f"unexpected train batch schema: {sorted(raw_batch)}")
    if tuple(raw_batch["sketch"].shape[:2]) != (2, 3):
        raise AssertionError(f"unexpected sketch shape: {tuple(raw_batch['sketch'].shape)}")
    if tuple(raw_batch["positive_photos"].shape[:3]) != (2, 1, 3):
        raise AssertionError(f"unexpected positive shape: {tuple(raw_batch['positive_photos'].shape)}")
    if tuple(prepared["clean"].shape[:2]) != (2, 3):
        raise AssertionError("generic coupled batch converter returned the wrong shape")
    test_loaders = build_test_loaders(protocol, transform, batch_size=2, num_workers=0)
    if len(test_loaders["sketch"].dataset) != len(protocol.test.sketch_entries):
        raise AssertionError("test sketch loader length mismatch")
    if len(test_loaders["photo"].dataset) != len(protocol.test.photo_entries):
        raise AssertionError("test photo loader length mismatch")
    # Deliberately do not call iter(test_loaders[...]): official test is stat-only.
    return {
        "name": protocol.name,
        "config": str(protocol.config_path.relative_to(ROOT)),
        "config_sha256": protocol.config_sha256,
        "root": str(protocol.root),
        "protocol_identity_sha256": protocol.identity["sha256"],
        "counts": {
            "train": protocol.train.identity["counts"],
            "test": protocol.test.identity["counts"],
        },
        "class_ids": {
            "train": list(protocol.train.class_ids),
            "test": list(protocol.test.class_ids),
        },
        "class_names": {
            "train": list(protocol.train.class_names.values()),
            "test": list(protocol.test.class_names.values()),
        },
        "semantic_class_name_overlap": list(protocol.identity["semantic_class_name_overlap"]),
        "path_overlap": list(protocol.identity["path_overlap"]),
        "file_existence_scope": "all train/test manifest paths stat'ed; no image content opened for test",
        "decode_scope": {"train_examples": 2, "test_examples": 0},
        "loader_schema": {
            "train_batch_size": 2,
            "raw_keys": sorted(raw_batch),
            "prepared_keys": sorted(prepared),
            "sketch_shape": list(raw_batch["sketch"].shape),
            "positive_shape": list(raw_batch["positive_photos"].shape),
            "negative_shape": list(raw_batch["negative_photo"].shape),
            "test_loaders_constructed_without_iteration": True,
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    errors: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "format": "coupled_benchmark_data_audit_v1",
        "status": "PASS",
        "protocols": {},
        "synthetic_checks": {},
        "errors_caught": errors,
        "owned_file_sha256": _owned_hashes(),
        "official_test_decode_claim": False,
    }
    try:
        report["synthetic_checks"] = _synthetic_checks(errors)
        for name in ("tuberlin_220_30", "quickdraw_80_30"):
            report["protocols"][name] = _audit_benchmark(name, errors)
    except Exception as error:
        report["status"] = "FAIL"
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error)}
    (OUT / "audit.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = f"status={report['status']} protocols={len(report['protocols'])} synthetic_errors={len(errors)} test_decode=0\n"
    (OUT / "logs" / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(OUT / "audit.json")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
