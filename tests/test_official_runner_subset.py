"""CPU-only orchestration test: selected datasets only, no training."""
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_coupled_benchmarks as runner


def test_subset_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    gate = tmp_path / "gate.json"
    gate.write_text("{}")
    order = ["tuberlin_220_30", "quickdraw_80_30"]
    calls = []
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(runner, "gate_check", lambda *a, **k: {"smoke_roots": {d: d for d in order}})
    monkeypatch.setattr(runner, "capture_provenance", lambda *a, **k: {"head_commit": "cpu", "source_snapshot": {"sha256": "fixed"}})
    monkeypatch.setattr(runner, "_copy_source_archive", lambda *a: None)
    def child(command, **kwargs):
        calls.append(command[command.index("--dataset") + 1])
        assert "--periodic-test" in command
        return SimpleNamespace(pid=0, wait=lambda: 0), SimpleNamespace(close=lambda: None), SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(runner, "_start_child", child)
    monkeypatch.setattr(runner, "check_finished_train", lambda *a, **k: {"step": 1, "selections": {"latest": {"sha256": "synthetic"}}})
    monkeypatch.setattr(runner, "_periodic_final_result", lambda *a: {"synthetic": True})
    output = tmp_path / "outputs" / "new"
    args = SimpleNamespace(output=str(output), gate=str(gate), periodic_test=True, datasets=order)
    assert runner.launch(args) == 0
    assert calls == order
    assert runner._read(output / "execution_manifest.json")["resolved_config"]["order"] == order
    assert runner._read(output / "runtime.json")["completed"] == order
    for invalid in [[order[0], order[0]], ["unknown"]]:
        args.datasets = invalid
        with pytest.raises(ValueError, match="nonduplicated subset"):
            runner.launch(args)
