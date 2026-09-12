from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from spica.data import coupled_benchmark as benchmark
from spica.data.coupled_benchmark import load_benchmark_protocol
from spica.data.manifest import ManifestEntry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/sketchy_104_21.yaml"


@pytest.fixture(scope="module")
def sketchy_protocol():
    return load_benchmark_protocol(CONFIG, split="train")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sketchy_metadata_identity_and_noncontiguous_guard(
    sketchy_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = sketchy_protocol

    assert protocol.name == "sketchy_104_21"
    assert protocol.train.identity["counts"] == {"sketch": 57587, "photo": 72949, "classes": 104}
    assert protocol.test.identity["counts"] == {"sketch": 12694, "photo": 12553, "classes": 21}
    assert protocol.train.class_ids == tuple(range(104))
    assert protocol.test.class_ids == tuple(range(21))
    assert set(protocol.train.class_names.values()).isdisjoint(protocol.test.class_names.values())
    assert protocol.identity["path_overlap"] == ()
    assert protocol.identity["image_decode_scope"] == "none"
    assert protocol.train.sketch_entries[0].label == 0  # labels are passed through, never remapped

    # The adapter validates IDs but does not remap manifest labels.
    original = benchmark._EXPECTED["sketchy_104_21"]["class_ids"]["train"]
    monkeypatch.setitem(benchmark._EXPECTED["sketchy_104_21"]["class_ids"], "train", (0, 2, *range(3, 104)))
    with pytest.raises(ValueError, match="class IDs mismatch"):
        load_benchmark_protocol(CONFIG)
    monkeypatch.setitem(benchmark._EXPECTED["sketchy_104_21"]["class_ids"], "train", original)


def test_sketchy_identity_hashes_and_all_paths_are_statted(sketchy_protocol) -> None:
    protocol = sketchy_protocol
    expected = benchmark._EXPECTED["sketchy_104_21"]["file_sha256"]
    actual = {
        f"{split}_{kind}": _sha(path)
        for split, part in (("train", protocol.train), ("test", protocol.test))
        for kind, path in (
            ("sketch_manifest", part.sketch_manifest),
            ("photo_manifest", part.photo_manifest),
            ("class_map", part.class_map),
        )
    }
    assert actual == expected
    for split in (protocol.train, protocol.test):
        for entries in (split.sketch_entries, split.photo_entries):
            assert split.identity["entry_identity"]["sketch" if entries is split.sketch_entries else "photo"]["resolved_path_count"] == len(entries)
            assert all(entry.path.is_file() for entry in entries)


def test_sketchy_loader_construction_does_not_decode_images(
    sketchy_protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = sketchy_protocol

    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("metadata-only protocol load decoded an image")

    monkeypatch.setattr("PIL.Image.open", fail_open)
    assert len(benchmark.build_test_loaders(protocol, lambda image: image)) == 2
    assert benchmark.make_train_loader(
        protocol, lambda image: image, batch_size=32, num_workers=0, drop_last=True
    ).dataset.sketch_entries == protocol.train.sketch_entries


def test_sketchy_path_validation_rejects_mismatched_root_and_bad_entry(sketchy_protocol) -> None:
    protocol = sketchy_protocol
    entry = protocol.train.sketch_entries[0]
    with pytest.raises(ValueError, match="escapes dataset root"):
        benchmark._validate_path_entries(
            (ManifestEntry(ROOT / "not_sketchy.png", entry.label),),
            root=protocol.root,
            modality="sketch",
            class_names=protocol.train.class_names,
            require_files=False,
        )


def test_sketchy_exposure_ratio_and_sampling_contract(sketchy_protocol) -> None:
    assert load_benchmark_protocol(CONFIG, split="test").selected_split == "test"
    with pytest.raises(ValueError, match="split must"):
        load_benchmark_protocol(CONFIG, split="validation")
    ratio = 3600 * 32 / 46624
    assert ratio == pytest.approx(2.4708304735758406)
    assert round(ratio) == 2
    assert 1 <= 1 <= 64  # one positive and one negative, at most 64 unique photos
    assert len(sketchy_protocol.train.class_ids) == 104
