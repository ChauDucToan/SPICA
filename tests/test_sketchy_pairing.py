from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from spica.data.datasets import MultiPositiveRetrievalTrainDataset
from spica.data.manifest import ManifestEntry
from spica.data.pairing import load_pairing_manifest


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (value, value, value)).save(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy(tmp_path: Path):
    root = tmp_path / "data"
    sketches = []
    photos = []
    records = []
    for label in (0, 1):
        sketch = root / "256x256" / "sketch" / "tx_000000000000_ready" / f"class{label}" / f"item{label}-1.png"
        photo = root / "256x256" / "photo" / "tx_000000000000_ready" / f"class{label}" / f"item{label}.jpg"
        other = root / "256x256" / "photo" / "tx_000000000000_ready" / f"class{label}" / f"other{label}.jpg"
        _write_image(sketch, label + 1)
        _write_image(photo, label + 3)
        _write_image(other, label + 5)
        sketches.append(ManifestEntry(sketch, label))
        photos.extend((ManifestEntry(photo, label), ManifestEntry(other, label)))
        records.append(
            {
                "sketch_path": sketch.relative_to(root).as_posix(),
                "photo_path": photo.relative_to(root).as_posix(),
                "label": label,
                "sketch_sha256": _sha(sketch),
                "photo_sha256": _sha(photo),
            }
        )
    return root, tuple(sketches), tuple(photos), records


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "pairing_kind": "filename_convention_canonical", "records": records}),
        encoding="utf-8",
    )


def test_loader_validates_scope_labels_branch_and_hash(tmp_path: Path) -> None:
    root, sketches, photos, records = _toy(tmp_path)
    manifest = tmp_path / "pairing.json"
    _write_manifest(manifest, records)
    loaded = load_pairing_manifest(
        manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos
    )
    assert loaded[str(sketches[0].path.resolve())].path == photos[0].path

    tampered = json.loads(manifest.read_text())
    tampered["records"][0]["photo_sha256"] = "0" * 64
    _write_manifest(manifest, tampered["records"])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_pairing_manifest(manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos)

    tampered = records[:1]
    _write_manifest(manifest, tampered)
    with pytest.raises(ValueError, match="coverage"):
        load_pairing_manifest(manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos)

    duplicate = records + [records[0]]
    _write_manifest(manifest, duplicate)
    with pytest.raises(ValueError, match="Duplicate"):
        load_pairing_manifest(manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos)


def test_loader_rejects_extended_photo_and_label_mismatch(tmp_path: Path) -> None:
    root, sketches, photos, records = _toy(tmp_path)
    manifest = tmp_path / "pairing.json"
    extended = root / "EXTEND_image_sketchy_ready" / "class0" / "item0.jpg"
    _write_image(extended, 9)
    bad = [dict(record) for record in records]
    bad[0]["photo_path"] = extended.relative_to(root).as_posix()
    bad[0]["photo_sha256"] = _sha(extended)
    _write_manifest(manifest, bad)
    with pytest.raises(ValueError, match="canonical"):
        load_pairing_manifest(manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos)

    bad = [dict(record) for record in records]
    bad[0]["label"] = 1
    _write_manifest(manifest, bad)
    with pytest.raises(ValueError, match="labels"):
        load_pairing_manifest(manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos)

    bad = [dict(record) for record in records]
    bad[0]["photo_path"] = photos[1].path.relative_to(root).as_posix()
    bad[0]["photo_sha256"] = _sha(photos[1].path)
    _write_manifest(manifest, bad)
    with pytest.raises(ValueError, match="filename stems"):
        load_pairing_manifest(manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos)


def _dataset(root: Path, sketches, photos, **kwargs):
    return MultiPositiveRetrievalTrainDataset(
        sketches,
        photos,
        lambda image: torch.tensor([image.getpixel((0, 0))[0]]),
        lambda image: torch.tensor([image.getpixel((0, 0))[0]]),
        num_positive_photos=1,
        **kwargs,
    )


def test_paired_and_same_class_consume_matching_rng_and_keep_negatives(tmp_path: Path) -> None:
    root, sketches, photos, records = _toy(tmp_path)
    manifest = tmp_path / "pairing.json"
    _write_manifest(manifest, records)
    pairing = load_pairing_manifest(manifest, dataset_root=root, sketch_entries=sketches, photo_entries=photos)
    paired = _dataset(root, sketches, photos, positive_pairing=pairing, positive_sampling="paired")
    same_class = _dataset(root, sketches, photos, positive_pairing=pairing, positive_sampling="same_class")

    random.seed(17)
    paired_sample = paired[0]
    random.seed(17)
    same_sample = same_class[0]
    assert paired_sample["positive_photo_paths"] == (str(photos[0].path),)
    assert same_sample["positive_photo_paths"] == (str(photos[0].path),)
    assert paired_sample["negative_photo_path"] == same_sample["negative_photo_path"]
    assert paired_sample["negative_label"] == same_sample["negative_label"]


def test_pairing_requires_k_one_and_paired_requires_mapping(tmp_path: Path) -> None:
    root, sketches, photos, records = _toy(tmp_path)
    with pytest.raises(ValueError, match="requires num_positive_photos=1"):
        MultiPositiveRetrievalTrainDataset(
            sketches, photos, lambda _: torch.zeros(1), lambda _: torch.zeros(1),
            num_positive_photos=2, positive_pairing={str(sketches[0].path): photos[0]},
        )
    with pytest.raises(ValueError, match="requires positive_pairing"):
        MultiPositiveRetrievalTrainDataset(
            sketches, photos, lambda _: torch.zeros(1), lambda _: torch.zeros(1),
            num_positive_photos=1, positive_sampling="paired",
        )


def test_default_sampling_is_unchanged(tmp_path: Path) -> None:
    root, sketches, photos, _ = _toy(tmp_path)
    first = _dataset(root, sketches, photos)
    second = _dataset(root, sketches, photos)
    random.seed(91)
    a = first[0]
    random.seed(91)
    b = second[0]
    assert a["positive_photo_paths"] == b["positive_photo_paths"]
    assert a["negative_photo_path"] == b["negative_photo_path"]
