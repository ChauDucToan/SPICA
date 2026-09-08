"""CPU checks for the no-update preflight's data packing and receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts import preflight_coupled_predictive as preflight


def _batch(root):
    images = torch.stack((torch.zeros(3, 4, 4), torch.ones(3, 4, 4)))
    labels = torch.arange(32) % 2
    batch = {
        "positive_photos": images[labels, None],
        "negative_photo": images[1 - labels],
    }
    trace = [{"positive_photo_paths": [str(root / f"{i % 2}.jpg")],
              "negative_photo_path": str(root / f"{1 - i % 2}.jpg"),
              "label": i % 2, "negative_label": 1 - i % 2} for i in range(32)]
    return batch, trace


def test_unique_photo_pack_preserves_gathers_and_rejects_conflicts(tmp_path):
    batch, trace = _batch(tmp_path)
    packed = preflight.pack_unique_photos(batch, trace, tmp_path)
    assert packed["photo_ids"] == ("0.jpg", "1.jpg")
    assert packed["bank_tensors"].shape == (2, 3, 4, 4)
    assert torch.equal(packed["bank_tensors"][packed["positive_indices"]], batch["positive_photos"][:, 0])
    assert torch.equal(packed["bank_tensors"][packed["negative_indices"][:, 0]], batch["negative_photo"])
    assert packed["bank_labels"].tolist() == [0, 1]
    assert not packed["bank_tensors"].requires_grad
    batch["positive_photos"][2, 0, 0, 0, 0] = 3
    with pytest.raises(AssertionError, match="different transformed"):
        preflight.pack_unique_photos(batch, trace, tmp_path)
    batch, trace = _batch(tmp_path)
    trace[2]["label"] = 1
    with pytest.raises(AssertionError, match="conflicting labels"):
        preflight.pack_unique_photos(batch, trace, tmp_path)
    with pytest.raises(ValueError, match="32 rows"):
        preflight.pack_unique_photos(batch, trace[:1], tmp_path)
    batch, trace = _batch(tmp_path)
    with pytest.raises(ValueError):
        preflight.pack_unique_photos(batch, trace, tmp_path / "not-the-root")


def test_fresh_outputs_source_archive_and_failure_receipt(tmp_path, monkeypatch):
    out = preflight._output_dir(str(tmp_path / "gate"))
    with pytest.raises(FileExistsError):
        preflight._output_dir(str(out))
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("x = 1\n")
    monkeypatch.setattr(preflight, "ROOT", source)
    digest = hashlib.sha256((source / "example.py").read_bytes()).hexdigest()
    inventory = {"sha256": "aggregate-recorded-elsewhere", "file_count": 1,
                 "manifest": [{"path": "example.py", "sha256": digest}]}
    preflight._archive_source_snapshot(out, inventory)
    assert (out / "source_snapshot/files/example.py").read_bytes() == (source / "example.py").read_bytes()
    assert json.loads((out / "source_snapshot/index.json").read_text())["file_count"] == 1
    with pytest.raises(AssertionError, match="empty or incomplete"):
        preflight._archive_source_snapshot(out, {"manifest": [], "file_count": 0})

    def fail():
        raise RuntimeError("preserve this failure")

    with pytest.raises(RuntimeError):
        preflight._phase(out, "cpu_fixture", fail)
    failure = json.loads((out / "phase_cpu_fixture.json").read_text())
    assert failure["status"] == "FAIL"
    assert failure["error"]["message"] == "preserve this failure"


def test_help_is_inert_and_cpu_fallback_is_forbidden():
    path = Path(preflight.__file__)
    help_run = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0 and "--output-dir" in help_run.stdout
    invalid = subprocess.run([sys.executable, str(path), "--device", "cpu", "--output-dir", "/unused"], capture_output=True, text=True)
    assert invalid.returncode == 2 and "invalid choice" in invalid.stderr
