"""Runner contracts use tiny metadata fixtures, not GPU or official images."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_coupled_benchmarks as runner  # noqa: E402


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def gate_fixture(tmp, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp)
    monkeypatch.setattr(runner, "COMPONENTS", ("source.py",))
    (tmp / "source.py").write_text("synthetic source")
    components = {"source.py": runner._sha(tmp / "source.py")}
    required, evidence, roots = {}, [], {}
    for name in runner.REQUIRED_EVIDENCE:
        receipt = {"status": "PASS", "scope": "CPU_SYNTHETIC_FIXTURE_ONLY", "post_smoke_component_sha256": {}}
        if ":" in name:
            dataset = name.split(":", 1)[1]
            receipt.update(dataset=dataset, step=2, actual_updates=2, smoke=True,
                           runtime_policy=runner.RUNTIME, optimizer_states=179, moment_tensors=358,
                           restore_exact=True, probe_rows=320, official_images_opened=0,
                           compact_whole_model_cuda_restore_exact=True,
                           scope="CPU_SAVED_SMOKE_RESTORE_AND_RUNTIME_PROBE",
                           source_snapshot_hash="synthetic", checkpoint_sha256="synthetic-ckpt")
            root = tmp / dataset
            roots[dataset] = str(root)
            put(root / "resolved_config.json", {"dataset": dataset, "arm": runner.ARM,
                "method_version": runner.METHOD_VERSION, "smoke": True, "actual_updates": 2,
                "total_steps": runner.DATASETS[dataset]["total_steps"], "runtime": runner.RUNTIME,
                "evaluation_query": "q", "training_main_query": "mu_i", "source_snapshot_hash": "synthetic"})
            put(root / "run_result.json", {"status": "COMPLETE", "step": 2,
                "source_snapshot_hash": "synthetic", "selections": {"latest": {"sha256": "synthetic-ckpt"}}})
            put(root / "source_snapshot/index.json", {"sha256": "synthetic",
                "manifest": [{"path": k, "sha256": v} for k, v in components.items()]})
        p = put(tmp / "evidence" / (name.replace(":", "_") + ".json"), receipt)
        required[name] = {"path": str(p), "sha256": runner._sha(p)}
        evidence.append(required[name])
    gate = {"status": "PASS", "method": runner.METHOD_VERSION, "arm": runner.ARM,
            "datasets": runner.DATASETS, "dataset_identities": runner.EXPECTED_IDENTITIES,
            "runtime_policy": runner.RUNTIME, "component_sha256": components,
            "required_evidence": required, "evidence": evidence, "smoke_roots": roots,
            "smoke_source_snapshot_hash": "synthetic"}
    return gate


def test_valid_directory_gate_and_reject_mutated_evidence(tmp_path, monkeypatch):
    gate = gate_fixture(tmp_path, monkeypatch)
    path = put(tmp_path / "gate.json", gate)
    assert runner.gate_check(path) == gate
    evidence = Path(gate["evidence"][0]["path"])
    evidence.write_text('{}')
    with pytest.raises(ValueError, match="changed gate evidence"):
        runner.gate_check(path)


def test_gate_requires_each_dataset_and_current_probe_source(tmp_path, monkeypatch):
    gate = gate_fixture(tmp_path, monkeypatch)
    del gate["smoke_roots"]["quickdraw_80_30"]
    with pytest.raises(ValueError, match="fresh smoke root"):
        runner.gate_check(put(tmp_path / "gate.json", gate))
    assert list(runner.DATASETS) == ["sketchy_104_21", "tuberlin_220_30", "quickdraw_80_30"]
    assert runner.DATASETS["sketchy_104_21"]["total_steps"] == 4446


def test_probe_only_multiples_of_five(tmp_path):
    put(tmp_path / "probe_manifest.json", {"scope": "seen_train_probe_not_validation", "query_count": 32, "gallery_count": 256})
    rows = [{"step_train": step, **{f"{scope}/{metric}": .02 for scope in ("cleaned", "masked")
             for metric in ("mAP@200", "mAP@all", "P@200")}} for step in (5, 10)]
    path = tmp_path / "probe_metrics.jsonl"
    path.write_text('\n'.join(map(json.dumps, rows)))
    runner._finite_probe(tmp_path, 12)
    rows.append({**rows[-1], "step_train": 12})
    path.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="cadence"):
        runner._finite_probe(tmp_path, 12)


def test_old_gate_rejected():
    with pytest.raises(ValueError):
        runner.gate_check(ROOT / 'outputs/coupled_benchmark_gate_20260910T162500Z/launch_gate.json')
