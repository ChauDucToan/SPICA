"""CPU-only periodic runner contract checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_coupled_benchmarks as runner  # noqa: E402


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream_files(root: Path, *, salt: str) -> None:
    rows = [{"row": index, "salt": salt} for index in range(64)]
    for name in ("observation_trace.jsonl", "mask_metadata.jsonl"):
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "lr_history_every_step.jsonl").write_text(
        "".join(json.dumps({"step": step, "lr": {"group_0": 1.0}}) + "\n" for step in (0, 1, 2))
    )


def _checkpoint(path: Path, step: int) -> str:
    """Use a real small tensor serialization and bind its actual file SHA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model_state_dict": {"weight": torch.tensor([step])}}, path)
    return _file_sha(path)


def _periodic_fixture(tmp_path, monkeypatch, horizon: int = 2000):
    dataset = "fixture"
    monkeypatch.setattr(runner, "DATASETS", {dataset: {"total_steps": horizon}})
    monkeypatch.setattr(
        runner,
        "EXPECTED_IDENTITIES",
        {dataset: {"counts": {"test": {"sketch": 4, "photo": 300}}}},
    )
    run = tmp_path / "run"
    smoke = tmp_path / "smoke"
    run.mkdir()
    smoke.mkdir()
    source = "source-hash"
    common = {
        "protocol_identity": {"fixture": True},
        "initialization_hashes": {"model": "init-state"},
        "loss_coefficient_identity": {"rank_i": 1.0},
        "seed": 42,
        "batch_size": 32,
        "optimizer": {"name": "AdamW"},
        "sampling_identity": {"fixture": "seeded"},
        "total_steps": horizon,
        "warmup_steps": max(1, horizon // 5),
    }
    cfg = {
        **common,
        "runtime": runner.OFFICIAL_RUNTIME,
        "test_steps": runner.official_test_steps(horizon),
        "tracking_policy": "official_test_v1",
        "selection_policy": "none;final_only",
        "probe_scope": "disabled_official_periodic_test",
        "official_unseen_evaluation": "periodic_monitoring_no_selection",
        "checkpoint_policy": "rolling_latest",
        "evaluation_query": "q",
        "training_main_query": "mu_i",
        "smoke": False,
    }
    control = {**common, "runtime": runner.OFFICIAL_RUNTIME, "smoke": True,
               "evaluation_query": "q", "training_main_query": "mu_i"}
    _write(run / "resolved_config.json", cfg)
    _write(smoke / "resolved_config.json", control)
    _write(run / "initialization.json", common["initialization_hashes"])
    _write(smoke / "initialization.json", common["initialization_hashes"])
    _stream_files(run, salt="same-stream")
    _stream_files(smoke, salt="same-stream")

    clean = {"full_mAP": 0.1, "mAP@200_min_relevant_k": 0.2, "P@200": 0.3}
    expected_steps = runner.official_test_steps(horizon)
    records = []
    metric_rows = []
    state_by_step = {}
    for index, step in enumerate(expected_steps):
        final = index == len(expected_steps) - 1
        state = f"model-state-{step}"
        state_by_step[step] = state
        if final:
            checkpoint_path = run / "checkpoint_latest.pt"
        else:
            checkpoint_path = run / "checkpoint_records" / f"step_{step}.pt"
        checkpoint_sha = _checkpoint(checkpoint_path, step)
        relative = Path("test") / f"step_{step}"
        summary = {
            "status": "COMPLETE",
            "step": step,
            "source_snapshot_hash": source,
            "predictor_forwards": 0,
            "model_state_before": state,
            "model_state_after": state,
            "checkpoint_selection": {
                "step": step,
                "sha256": checkpoint_sha,
                "model_state_hash": state,
            },
            "conditions": [{}] * 10,
            "query_count": 4,
            "gallery_count": 300,
            "official_test_evaluated": True,
            "official_unseen_used_for_selection": False,
            "official_unseen_used_for_training": False,
            "evaluation_scope": "official_unseen_final" if final else "official_unseen_periodic_monitoring_no_selection",
            "checkpoint_selections": {"latest": {"sha256": checkpoint_sha}},
            "artifact_policy": "full" if final else "metrics_only",
            "clean": dict(clean),
            "masked_macro": dict(clean),
            "masked_by_fraction": {},
        }
        summary_path = run / relative / "summary.json"
        _write(summary_path, summary)
        if final:
            (summary_path.parent / "gallery_embeddings.npy").write_bytes(b"small-final-gallery")
        record = {
            "step": step,
            "path": str(relative),
            "summary": str(relative / "summary.json"),
            "summary_sha256": runner._sha(summary_path),
            "checkpoint_sha256": checkpoint_sha,
            "final": final,
        }
        records.append(record)
        metric_rows.append({
            "step_train": step,
            "test/cleaned/mAP@200": clean["mAP@200_min_relevant_k"],
            "test/cleaned/mAP@all": clean["full_mAP"],
            "test/cleaned/P@200": clean["P@200"],
            "test/masked/mAP@200": clean["mAP@200_min_relevant_k"],
            "test/masked/mAP@all": clean["full_mAP"],
            "test/masked/P@200": clean["P@200"],
        })

    final_record = records[-1]
    final_state = state_by_step[expected_steps[-1]]
    _write(run / "checkpoint_latest.json", {
        "path": "checkpoint_latest.pt",
        "step": horizon,
        "sha256": final_record["checkpoint_sha256"],
        "model_state_hash": final_state,
    })
    (run / "test_evaluations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (run / "test_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metric_rows)
    )
    result = {
        **common,
        "status": "COMPLETE",
        "step": horizon,
        "source_snapshot_hash": source,
        "source_snapshot_hash_after": source,
        "arm": runner.ARM,
        "wandb_run_id": "run",
        "model_state_hash_after_updates": final_state,
        "selections": {"latest": {
            "step": horizon,
            "path": "checkpoint_latest.pt",
            "sha256": final_record["checkpoint_sha256"],
        }},
        "official_test_evaluations": records,
        "frozen_original_state_hash_before": "frozen",
        "frozen_original_state_hash_after": "frozen",
    }
    _write(run / "run_result.json", result)
    _write(run / "wandb_runtime.json", {"run_id": "run"})
    return run, smoke, result, cfg, source


def test_periodic_finished_train_rejects_stale_probe(tmp_path, monkeypatch):
    run, smoke, result, cfg, source = _periodic_fixture(tmp_path, monkeypatch)
    (run / "probe_metrics.jsonl").write_text("stale\n")
    _write(run / "run_result.json", result)
    _write(run / "resolved_config.json", cfg)
    with pytest.raises(ValueError, match="stale train-probe"):
        runner.check_finished_train(run, smoke, "fixture", source, periodic_test=True)


@pytest.mark.parametrize("horizon", (5, 2, 19))
def test_periodic_finished_train_accepts_valid_cadence_budgets(tmp_path, monkeypatch, horizon):
    run, smoke, result, cfg, source = _periodic_fixture(tmp_path, monkeypatch, horizon)
    _write(run / "run_result.json", result)
    _write(run / "resolved_config.json", cfg)
    assert runner.check_finished_train(run, smoke, "fixture", source, periodic_test=True) == result


def test_periodic_finished_train_rejects_duplicate_final(tmp_path, monkeypatch):
    run, smoke, result, cfg, source = _periodic_fixture(tmp_path, monkeypatch)
    result["official_test_evaluations"].append(result["official_test_evaluations"][-1])
    _write(run / "run_result.json", result)
    _write(run / "resolved_config.json", cfg)
    with pytest.raises(ValueError, match="cadence"):
        runner.check_finished_train(run, smoke, "fixture", source, periodic_test=True)


@pytest.mark.parametrize("mutation", ("checkpoint", "source", "initialization", "stream"))
def test_periodic_finished_train_rejects_mutated_final_inputs(tmp_path, monkeypatch, mutation):
    run, smoke, result, cfg, source = _periodic_fixture(tmp_path, monkeypatch)
    if mutation == "checkpoint":
        with (run / "checkpoint_latest.pt").open("ab") as handle:
            handle.write(b"mutation")
    elif mutation == "source":
        result["source_snapshot_hash"] = "mutated-source"
        _write(run / "run_result.json", result)
    elif mutation == "initialization":
        mutated = dict(cfg)
        mutated["initialization_hashes"] = {"model": "mutated-init"}
        _write(run / "resolved_config.json", mutated)
    else:
        stream = (run / "observation_trace.jsonl").read_text().splitlines()
        stream[0] = json.dumps({"row": 0, "salt": "mutated-stream"})
        (run / "observation_trace.jsonl").write_text("\n".join(stream) + "\n")
    with pytest.raises(ValueError):
        runner.check_finished_train(run, smoke, "fixture", source, periodic_test=True)


def _periodic_gate_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    evaluator_sha = "evaluator-component-sha"
    paths = {}
    receipts = {}
    for name in runner.EVIDENCE_NAMES:
        filename = name.replace(":", "_") + ".json"
        path = tmp_path / filename
        if not name.startswith("gpu_smoke_restore_probe:"):
            receipt = {"status": "PASS", "scope": name}
        else:
            dataset = name.split(":", 1)[1]
            receipt = {
                "status": "PASS", "scope": "CPU_SAVED_SMOKE_RESTORE_AND_PERIODIC_TEST",
                "dataset": dataset, "step": 2, "actual_updates": 2, "smoke": True,
                "runtime_policy": runner.OFFICIAL_RUNTIME,
                "optimizer_states": 179, "moment_tensors": 358,
                "restore_exact": True, "official_images_opened": 0,
                "test_rows": 2560,
                "test_boundary_resume_exact": True,
                "no_update_after_step1_evaluation": True,
                "step2_parity_baseline": True,
                "compact_whole_model_cuda_restore_exact": True,
                "evaluator_sha256": evaluator_sha,
            }
        _write(path, receipt)
        relative = path.relative_to(tmp_path).as_posix()
        paths[name] = relative
        receipts[relative] = {"path": relative, "sha256": runner._sha(path)}
    gate = {
        "required_evidence": paths,
        "evidence": list(receipts.values()),
        "component_sha256": {"scripts/evaluate_coupled_benchmark.py": evaluator_sha},
    }
    return gate


def test_periodic_smoke_gate_requires_periodic_restore_contract(tmp_path, monkeypatch):
    gate = _periodic_gate_fixture(tmp_path, monkeypatch)
    assert "src/spica/evaluation/periodic_test.py" in runner.COMPONENTS
    assert "configs/runtime/official_test.yaml" in runner.COMPONENTS
    assert runner._required_evidence(tmp_path / "gate.json", gate, periodic_test=True)


@pytest.mark.parametrize(
    "missing",
    ("test_boundary_resume_exact", "no_update_after_step1_evaluation", "step2_parity_baseline"),
)
def test_periodic_smoke_gate_rejects_missing_periodic_key(tmp_path, monkeypatch, missing):
    gate = _periodic_gate_fixture(tmp_path, monkeypatch)
    key = "gpu_smoke_restore_probe:sketchy_104_21"
    receipt_path = tmp_path / gate["required_evidence"][key]
    receipt = json.loads(receipt_path.read_text())
    del receipt[missing]
    _write(receipt_path, receipt)
    for row in gate["evidence"]:
        if row["path"] == gate["required_evidence"][key]:
            row["sha256"] = runner._sha(receipt_path)
    with pytest.raises(ValueError, match="incomplete saved restore evidence"):
        runner._required_evidence(tmp_path / "gate.json", gate, periodic_test=True)
