"""Inert-by-default official MP-Q runner.

Three locked sets: fresh CUDA smoke-2 gates; sequential Sketchy/TU-Berlin/QuickDraw
training; final-only official evaluation.  No checkpoint loading, retries, resume,
or probe/evaluation is performed by this parent process.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import islice
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

from run_coupled_campaign import _copy_source_archive, _json_atomic, _now, _start_child
from run_fusion_campaign import _read, _sha
from spica.provenance import capture_provenance
from spica.data.coupled_benchmark import OFFICIAL_IDENTITIES, _plain
from spica.runtime import runtime_policy
from spica.train_coupled_benchmark import ARM, DATASETS, METHOD_VERSION

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {"probe_every": 5, "checkpoint_every": 100}
OFFICIAL_RUNTIME = {"test_every": 1000, "checkpoint_every": 100}
PERIODIC_STATUS = "TRAIN_AND_PERIODIC_OFFICIAL_EVAL_FINISHED_UNVERIFIED"
EXPECTED_IDENTITIES = _plain(OFFICIAL_IDENTITIES)
COMPONENTS = tuple(
    sorted(
        set(
            (
                "scripts/run_coupled_benchmarks.py",
                "scripts/evaluate_coupled_benchmark.py",
                "scripts/check_coupled_benchmark_cpu.py",
                "scripts/check_coupled_benchmark_data_cpu.py",
                "src/spica/train_coupled_benchmark.py",
                "src/spica/data/coupled_benchmark.py",
                "src/spica/evaluation/coupled_benchmark.py",
                "src/spica/train_coupled_predictive.py",
                "src/spica/models/coupled_predictive.py",
                "src/spica/models/clip.py",
                "src/spica/models/checkpoint.py",
                "src/spica/coupled_predictive_losses.py",
                "src/spica/data/coupled_training.py",
                "src/spica/data/datasets.py",
                "src/spica/data/coupled_views.py",
                "src/spica/data/masking.py",
                "src/spica/evaluation/coupled_predictive.py",
                "src/spica/evaluation/metrics.py",
                "src/spica/evaluation/periodic_test.py",
                "src/spica/evaluation/masked_view.py",
                "src/spica/provenance.py",
                "src/spica/runtime.py",
                "src/spica/tracking/wandb.py",
                "src/spica/evaluation/training_probe.py",
                "src/spica/evaluation/embeddings.py",
                "src/spica/evaluation/text_bank.py",
                "src/spica/models/frozen_prompt.py",
                "src/spica/data/transforms.py",
                "src/spica/data/manifest.py",
                "src/spica/config/data.py",
                "src/spica/semantic_text.py",
                "scripts/train.py",
                "configs/train_coupled_benchmark.yaml",
                "scripts/run_coupled_campaign.py",
                "scripts/run_fusion_campaign.py",
                "configs/runtime/minimal.yaml",
                "configs/runtime/official_test.yaml",
                "configs/data/sketchy_104_21.yaml",
                "configs/data/tuberlin_220_30.yaml",
                "configs/data/quickdraw_80_30.yaml",
                "uv.lock",
            )
        )
    )
)
EVIDENCE_NAMES = (
    "cpu_test_receipt",
    "data_audit_receipt",
    *(f"gpu_smoke_restore_probe:{d}" for d in DATASETS),
)
REQUIRED_EVIDENCE = EVIDENCE_NAMES  # parent gate assembly contract
# Post-smoke fixes here require fresh CPU runner and CUDA compact-restore evidence,
# not another optimizer run; every training/probe component remains archive-exact.
RESTORE_COMPONENTS = {"scripts/run_coupled_benchmarks.py", "scripts/evaluate_coupled_benchmark.py"}


def _records(gate):
    rows = gate.get("evidence")
    if not isinstance(rows, list):
        raise ValueError("gate.evidence must be a list")
    out = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
        ):
            raise ValueError("every gate evidence record requires path and SHA256")
        if (
            row["path"] in out
            or _sha(_evidence_path(row["path"], None)) != row["sha256"]
        ):
            raise ValueError("duplicate or changed gate evidence")
        out[row["path"]] = row["sha256"]
    return out


def _evidence_path(path, gate_path):
    p = Path(path).expanduser()
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"evidence outside repository: {path}")
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _required_evidence(gate_path, gate, periodic_test=False):
    records = _records(gate)
    required = gate.get("required_evidence")
    if not isinstance(required, dict) or set(required) != set(EVIDENCE_NAMES):
        raise ValueError(
            "required_evidence must name CPU, data-audit, and each dataset smoke receipt"
        )
    for name in EVIDENCE_NAMES:
        value = required[name]
        path, digest = (
            (value, None)
            if isinstance(value, str)
            else (value.get("path"), value.get("sha256"))
            if isinstance(value, dict)
            else (None, None)
        )
        if not isinstance(path, str):
            raise ValueError(f"invalid required evidence: {name}")
        actual = _sha(_evidence_path(path, gate_path))
        if digest is not None and actual != digest:
            raise ValueError(f"required evidence hash mismatch: {name}")
        if path not in records or records[path] not in (None, actual):
            raise ValueError(f"required evidence is not bound by gate.evidence: {name}")
        receipt = _read(_evidence_path(path, gate_path))
        if receipt.get("status") != "PASS" or not isinstance(receipt.get("scope"), str):
            raise ValueError(f"evidence status/scope is not PASS: {name}")
        if name.startswith("gpu_smoke_restore_probe:"):
            dataset = name.split(":", 1)[1]
            if (
                receipt.get("dataset") != dataset
                or receipt.get("step", receipt.get("actual_updates")) != 2
            ):
                raise ValueError(f"smoke receipt identity mismatch: {dataset}")
            if receipt.get("actual_updates", 2) != 2 or receipt.get("smoke") not in (
                None,
                True,
            ):
                raise ValueError(f"smoke receipt is not smoke-2: {dataset}")
            expected_runtime = OFFICIAL_RUNTIME if periodic_test else RUNTIME
            if receipt.get("runtime_policy") != expected_runtime:
                raise ValueError(f"smoke runtime policy mismatch: {dataset}")
            expected = {
                "optimizer_states": 179,
                "moment_tensors": 358,
                "restore_exact": True,
                "official_images_opened": 0,
                "scope": (
                    "CPU_SAVED_SMOKE_RESTORE_AND_PERIODIC_TEST"
                    if periodic_test else "CPU_SAVED_SMOKE_RESTORE_AND_RUNTIME_PROBE"
                ),
            }
            expected["test_rows" if periodic_test else "probe_rows"] = 2560 if periodic_test else 320
            if periodic_test:
                expected.update({
                    "test_boundary_resume_exact": True,
                    "no_update_after_step1_evaluation": True,
                    "step2_parity_baseline": True,
                })
            if not receipt.get("compact_whole_model_cuda_restore_exact"):
                raise ValueError(f"missing actual compact CUDA restore: {dataset}")
            if receipt.get("evaluator_sha256") != gate['component_sha256'].get('scripts/evaluate_coupled_benchmark.py'):
                raise ValueError(f"stale evaluator restore gate: {dataset}")
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError(f"incomplete saved restore evidence: {dataset}")
    return {name: required[name] for name in EVIDENCE_NAMES}


def gate_check(path: Path, periodic_test: bool = False):
    gate = _read(path)
    if (
        gate.get("status") != "PASS"
        or gate.get("method") != METHOD_VERSION
        or gate.get("arm") != ARM
    ):
        raise ValueError("official MP-Q PASS gate method/arm required")
    if (
        gate.get("datasets") != DATASETS
        or gate.get("dataset_identities") != EXPECTED_IDENTITIES
    ):
        raise ValueError(
            "official dataset identities/order do not match the locked protocol"
        )
    expected_runtime = OFFICIAL_RUNTIME if periodic_test else RUNTIME
    if periodic_test:
        if gate.get("evaluation_policy") != "official_every1000_and_final_no_selection":
            raise ValueError("periodic gate requires official_every1000_and_final_no_selection")
    elif gate.get("evaluation_policy") == "official_every1000_and_final_no_selection":
        raise ValueError("periodic gate cannot be used by the legacy runner")
    if (
        gate.get("runtime_policy", gate.get("runtime")) != expected_runtime
        or runtime_policy(expected_runtime) != expected_runtime
    ):
        raise ValueError("runtime policy does not match the selected runner")
    components = gate.get("component_sha256")
    if not isinstance(components, dict) or set(components) != set(COMPONENTS):
        raise ValueError(
            "gate component coverage is not the current production contract"
        )
    for name, digest in components.items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"gate source changed: {name}")
    _required_evidence(path, gate, periodic_test=periodic_test)
    cpu_record = gate['required_evidence']['cpu_test_receipt']
    cpu_path = cpu_record if isinstance(cpu_record, str) else cpu_record['path']
    cpu = _read(_evidence_path(cpu_path, path))
    if cpu.get('post_smoke_component_sha256') != {k: v for k, v in components.items() if k in RESTORE_COMPONENTS}:
        raise ValueError('fresh CPU runner/restore tests are not bound')
    smoke_roots = gate.get("smoke_roots")
    if not isinstance(smoke_roots, dict) or set(smoke_roots) != set(DATASETS):
        raise ValueError("gate must contain one fresh smoke root per dataset")
    source_hashes = set()
    for dataset, value in smoke_roots.items():
        root = (ROOT / value).resolve() if isinstance(value, str) else None
        if root is None or not root.is_relative_to(ROOT) or not root.is_dir():
            raise ValueError(f"invalid smoke root: {dataset}")
        cfg = _read(root / "resolved_config.json")
        result = _read(root / "run_result.json")
        if (
            cfg.get("dataset"),
            cfg.get("arm"),
            cfg.get("method_version"),
            cfg.get("smoke"),
        ) != (dataset, ARM, METHOD_VERSION, True):
            raise ValueError(f"smoke config identity mismatch: {dataset}")
        if (
            cfg.get("actual_updates") != 2
            or cfg.get("total_steps") != DATASETS[dataset]["total_steps"]
            or cfg.get("runtime") != expected_runtime
        ):
            raise ValueError(f"smoke schedule/runtime mismatch: {dataset}")
        if (
            cfg.get("evaluation_query") != "q"
            or cfg.get("training_main_query") != "mu_i"
            or "sketch_ref" in cfg
            or cfg.get("lambda_sketch_ref") is not None
        ):
            raise ValueError(f"smoke is not MP-Q without SREF: {dataset}")
        if (
            result.get("status") != "COMPLETE"
            or result.get("step") != 2
            or result.get("source_snapshot_hash") != cfg.get("source_snapshot_hash")
        ):
            raise ValueError(f"smoke result mismatch: {dataset}")
        required = gate["required_evidence"][f"gpu_smoke_restore_probe:{dataset}"]
        receipt_path = required if isinstance(required, str) else required["path"]
        receipt = _read(_evidence_path(receipt_path, path))
        if receipt.get("source_snapshot_hash") != cfg[
            "source_snapshot_hash"
        ] or receipt.get("checkpoint_sha256") != result.get("selections", {}).get(
            "latest", {}
        ).get("sha256"):
            raise ValueError(
                f"CPU receipt does not bind this smoke checkpoint/source: {dataset}"
            )
        index = root / "source_snapshot" / "index.json"
        if not index.is_file() or _read(index).get("sha256") != cfg.get(
            "source_snapshot_hash"
        ):
            raise ValueError(f"smoke source manifest mismatch: {dataset}")
        manifest = {
            row.get("path"): row.get("sha256")
            for row in _read(index).get("manifest", ())
        }
        if any(manifest.get(name) != digest for name, digest in components.items() if name not in RESTORE_COMPONENTS):
            raise ValueError(
                f"smoke source components do not bind current gate: {dataset}"
            )
        source_hashes.add(cfg["source_snapshot_hash"])
    if (
        len(source_hashes) != 1
        or gate.get("smoke_source_snapshot_hash") not in source_hashes
    ):
        raise ValueError("smoke roots do not share the gate source binding")
    return gate


def _finite_probe(run, horizon):
    probe = _read(run / "probe_manifest.json")
    if (
        probe.get("scope") != "seen_train_probe_not_validation"
        or probe.get("query_count") != 32
        or probe.get("gallery_count") != 256
    ):
        raise ValueError(
            "probe scope is not the fixed 32-query/256-gallery train probe"
        )
    rows = [
        json.loads(line)
        for line in (run / "probe_metrics.jsonl").read_text().splitlines()
    ]
    expected = list(range(5, horizon + 1, 5))
    if [row.get("step_train") for row in rows] != expected:
        raise ValueError("probe cadence/history is not every five updates")
    names = (
        "cleaned/mAP@200",
        "cleaned/mAP@all",
        "cleaned/P@200",
        "masked/mAP@200",
        "masked/mAP@all",
        "masked/P@200",
    )
    for row in rows:
        if set(row) != {"step_train", *names} or any(
            isinstance(row[n], bool) or not math.isfinite(float(row[n])) for n in names
        ):
            raise ValueError("probe row is not six finite retrieval scalars")


def _check_periodic_train(run: Path, result: dict, cfg: dict, dataset: str, source_hash: str):
    horizon = DATASETS[dataset]["total_steps"]
    if (
        cfg.get("smoke")
        or cfg.get("runtime") != OFFICIAL_RUNTIME
        or cfg.get("tracking_policy") != "official_test_v1"
        or cfg.get("selection_policy") != "none;final_only"
        or cfg.get("probe_scope") != "disabled_official_periodic_test"
        or cfg.get("official_unseen_evaluation") != "periodic_monitoring_no_selection"
    ):
        raise ValueError(f"periodic official runtime/config mismatch: {dataset}")
    if any((run / name).exists() for name in ("probe_manifest.json", "probe_metrics.jsonl")):
        raise ValueError(f"periodic run contains stale train-probe artifacts: {dataset}")
    expected_steps = list(range(1000, horizon + 1, 1000))
    if not expected_steps or expected_steps[-1] != horizon:
        expected_steps.append(horizon)
    if not isinstance(result.get("wandb_run_id"), str) or not result["wandb_run_id"]:
        raise ValueError(f"periodic W&B run identity is missing: {dataset}")
    records = result.get("official_test_evaluations")
    if not isinstance(records, list) or len({row.get("step") for row in records if isinstance(row, dict)}) != len(records) or [row.get("step") for row in records] != expected_steps:
        raise ValueError(f"periodic test cadence/history mismatch: {dataset}")
    wandb_runtime = _read(run / "wandb_runtime.json")
    if wandb_runtime.get("run_id") != result.get("wandb_run_id"):
        raise ValueError(f"periodic W&B runtime identity mismatch: {dataset}")
    evaluation_rows = [json.loads(line) for line in (run / "test_evaluations.jsonl").read_text().splitlines()]
    if evaluation_rows != records:
        raise ValueError(f"periodic evaluation records are not bound to run result: {dataset}")
    metrics_rows = [json.loads(line) for line in (run / "test_metrics.jsonl").read_text().splitlines()]
    metric_names = tuple(
        f"test/{scope}/{metric}"
        for scope in ("cleaned", "masked")
        for metric in ("mAP@200", "mAP@all", "P@200")
    )
    if [row.get("step_train") for row in metrics_rows] != expected_steps or any(
        set(row) != {"step_train", *metric_names}
        or any(isinstance(row[name], bool) or not math.isfinite(float(row[name])) for name in metric_names)
        for row in metrics_rows
    ):
        raise ValueError(f"periodic test metric history is invalid: {dataset}")
    counts = EXPECTED_IDENTITIES[dataset]["counts"]["test"]
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("step") != expected_steps[index]:
            raise ValueError(f"periodic test record is invalid: {dataset}")
        summary_path = run / str(record.get("path")) / "summary.json"
        if (
            record.get("summary") != str(summary_path.relative_to(run))
            or record.get("summary_sha256") != _sha(summary_path)
            or not isinstance(record.get("checkpoint_sha256"), str)
            or record.get("final") != (index == len(records) - 1)
        ):
            raise ValueError(f"periodic checkpoint/summary binding mismatch: {dataset}/{record.get('step')}")
        summary = _read(summary_path)
        if (
            summary.get("status") != "COMPLETE"
            or summary.get("step") != record["step"]
            or summary.get("source_snapshot_hash") != source_hash
            or summary.get("predictor_forwards") != 0
            or summary.get("official_test_evaluated") is not True
            or summary.get("model_state_before") != summary.get("model_state_after")
            or summary.get("model_state_before") != summary.get("checkpoint_selection", {}).get("model_state_hash")
            or (record["final"] and (record["checkpoint_sha256"] != result["selections"]["latest"]["sha256"]
                                    or summary.get("model_state_after") != result["model_state_hash_after_updates"]))
            or len(summary.get("conditions", ())) != 10
            or summary.get("query_count") != counts["sketch"]
            or summary.get("gallery_count") != counts["photo"]
            or summary.get("official_unseen_used_for_selection") is not False
            or summary.get("official_unseen_used_for_training") is not False
            or summary.get("evaluation_scope") != (
                "official_unseen_final" if record["final"]
                else "official_unseen_periodic_monitoring_no_selection"
            )
            or summary.get("checkpoint_selections", {}).get("latest", {}).get("sha256") != record["checkpoint_sha256"]
            or summary.get("checkpoint_selection", {}).get("sha256", record["checkpoint_sha256"]) != record["checkpoint_sha256"]
        ):
            raise ValueError(f"periodic summary identity/count/state mismatch: {dataset}/{record.get('step')}")
        values = {"cleaned": "clean", "masked": "masked_macro"}
        for scope, source in values.items():
            for metric, key in (("mAP@200", "mAP@200_min_relevant_k"), ("mAP@all", "full_mAP"), ("P@200", "P@200")):
                if metrics_rows[index][f"test/{scope}/{metric}"] != summary[source][key]:
                    raise ValueError(f"periodic metric/summary mismatch: {dataset}/{record['step']}")
        files = [path for path in summary_path.parent.iterdir() if path.name != "summary.json"]
        if record["final"]:
            if summary.get("artifact_policy") != "full" or not (summary_path.parent / "gallery_embeddings.npy").is_file():
                raise ValueError(f"periodic final artifacts are incomplete: {dataset}")
        elif summary.get("artifact_policy") != "metrics_only" or any(path.suffix in {".npy", ".npz", ".jsonl"} for path in files):
            raise ValueError(f"periodic non-final artifacts are not metrics-only: {dataset}")
    return result


def check_finished_train(run: Path, smoke: Path, dataset: str, source_hash: str, periodic_test: bool = False):
    result, cfg, control = (
        _read(run / "run_result.json"),
        _read(run / "resolved_config.json"),
        _read(smoke / "resolved_config.json"),
    )
    horizon = DATASETS[dataset]["total_steps"]
    if (
        result.get("status") != "COMPLETE"
        or result.get("step") != horizon
        or result.get("source_snapshot_hash") != source_hash
        or result.get("source_snapshot_hash_after") != source_hash
        or result.get("arm") != ARM
    ):
        raise ValueError(f"production result identity mismatch: {dataset}")
    if (
        cfg.get("smoke")
        or cfg.get("runtime") != (OFFICIAL_RUNTIME if periodic_test else RUNTIME)
        or cfg.get("checkpoint_policy") != "rolling_latest"
        or cfg.get("tracking_policy") != ("official_test_v1" if periodic_test else "minimal_v1")
        or cfg.get("selection_policy") != "none;final_only"
    ):
        raise ValueError(f"production minimal runtime/config mismatch: {dataset}")
    if (
        cfg.get("evaluation_query") != "q"
        or cfg.get("training_main_query") != "mu_i"
        or cfg.get("lambda_sketch_ref") is not None
        or "sketch_ref" in cfg
    ):
        raise ValueError(f"production route is not MP-Q: {dataset}")
    for key in (
        "protocol_identity",
        "initialization_hashes",
        "loss_coefficient_identity", "seed", "batch_size", "optimizer",
        "sampling_identity", "total_steps", "warmup_steps",
    ):
        if cfg.get(key) != control.get(key):
            raise ValueError(f"smoke binding mismatch: {dataset}/{key}")
    for name, count in (
        ("observation_trace.jsonl", 64),
        ("mask_metadata.jsonl", 64),
        ("lr_history_every_step.jsonl", 3),
    ):
        with (run / name).open() as handle:
            prefix = list(islice(handle, count))
        with (smoke / name).open() as handle:
            control_prefix = list(islice(handle, count))
        if len(prefix) != count or prefix != control_prefix:
            raise ValueError(f"initial stream mismatch: {dataset}/{name}")
    if (
        list(result.get("selections", {})) != ["latest"]
        or (run / "checkpoint_latest.pt").is_file() is False
    ):
        raise ValueError(f"rolling latest checkpoint missing: {dataset}")
    final = result["selections"]["latest"]
    stream = _read(run / "checkpoint_latest.json")
    if (
        final.get("path") != "checkpoint_latest.pt"
        or final.get("step") != horizon
        or any(stream.get(key) != final.get(key) for key in ("path", "step", "sha256"))
        or _sha(run / final["path"]) != final.get("sha256")
        or stream.get("model_state_hash")
        != result.get("model_state_hash_after_updates")
    ):
        raise ValueError(f"final rolling checkpoint mismatch: {dataset}")
    if result.get("frozen_original_state_hash_before") != result.get(
        "frozen_original_state_hash_after"
    ):
        raise ValueError("frozen original state changed during training")
    if any(run.glob("checkpoint_step*.pt")):
        raise ValueError(f"production retained non-rolling checkpoints: {dataset}")
    if periodic_test:
        return _check_periodic_train(run, result, cfg, dataset, source_hash)
    _finite_probe(run, horizon)
    return result


def _periodic_final_result(run: Path, train: dict[str, Any], dataset: str) -> dict[str, Any]:
    records = train["official_test_evaluations"]
    summary = _read(run / records[-1]["path"] / "summary.json")
    return {
        "step": train["step"],
        "checkpoint_sha256": records[-1]["checkpoint_sha256"],
        "wandb_url": train.get("wandb_url"),
        "clean": summary["clean"],
        "masked_macro": summary["masked_macro"],
        "masked_by_fraction": summary["masked_by_fraction"],
        "evaluation_summary_sha256": _sha(run / records[-1]["path"] / "summary.json"),
    }


def _check_evaluation(evaluation, train, dataset, source_hash):
    stats = _read(evaluation / "summary.json")
    counts = EXPECTED_IDENTITIES[dataset]["counts"]["test"]
    if (
        stats.get("status") != "COMPLETE"
        or stats.get("official_test_evaluated") is not True
        or stats.get("step") != train["step"]
        or stats.get("source_snapshot_hash") != source_hash
        or stats.get("predictor_forwards") != 0
        or stats.get("model_state_before")
        != train.get("model_state_hash_after_updates")
        or stats.get("model_state_after") != stats.get("model_state_before")
        or len(stats.get("conditions", ())) != 10
        or stats.get("query_count") != counts["sketch"]
        or stats.get("gallery_count") != counts["photo"]
    ):
        raise ValueError(f"final evaluation identity/count/state mismatch: {dataset}")
    published = _read(evaluation / "wandb_final.json")
    summary = published.get("summary", {})
    if (
        published.get("status") != "PASS"
        or published.get("run_id") != train.get("wandb_run_id")
        or summary.get("final/step") != train["step"]
        or summary.get("final/checkpoint_sha256")
        != train["selections"]["latest"]["sha256"]
    ):
        raise ValueError(f"W&B final identity mismatch: {dataset}")
    for scope, source in (("cleaned", "clean"), ("masked", "masked_macro")):
        for metric, key in (
            ("mAP@200", "mAP@200_min_relevant_k"),
            ("mAP@all", "full_mAP"),
            ("P@200", "P@200"),
        ):
            if summary.get(f"final/{scope}/{metric}") != stats[source][key]:
                raise ValueError(
                    f"W&B final metric mismatch: {dataset}/{scope}/{metric}"
                )
    return {
        "step": train["step"],
        "checkpoint_sha256": train["selections"]["latest"]["sha256"],
        "wandb_url": train["wandb_url"],
        "clean": stats["clean"],
        "masked_macro": stats["masked_macro"],
        "masked_by_fraction": stats["masked_by_fraction"],
        "evaluation_summary_sha256": _sha(evaluation / "summary.json"),
    }


def launch(args):
    output = Path(args.output).resolve()
    periodic_test = bool(args.periodic_test)
    if (
        output.exists()
        or not output.is_relative_to(ROOT / "outputs")
        or output == ROOT / "outputs"
    ):
        raise ValueError("fresh child of outputs/ required")
    minimum_free = 16 * 1024**3 if periodic_test else 24 * 1024**3
    if shutil.disk_usage(ROOT).free < minimum_free:
        raise RuntimeError(
            f"at least {minimum_free // (1024**3)}GiB free required; no artifact cleanup permitted"
        )
    gate = gate_check(Path(args.gate), periodic_test=periodic_test)
    config = {
        "method": METHOD_VERSION,
        "arm": ARM,
        "datasets": DATASETS,
        "dataset_identities": EXPECTED_IDENTITIES,
        "runtime_policy": OFFICIAL_RUNTIME if periodic_test else RUNTIME,
        "order": list(DATASETS),
        "evaluation": (
            "official_every1000_and_final_no_selection"
            if periodic_test else "final_only_clean_plus_9_masks"
        ),
        "evaluation_policy": (
            "official_every1000_and_final_no_selection"
            if periodic_test else "final_only_clean_plus_9_masks"
        ),
        "official_unseen_used_for_selection": False,
        "no_retry_or_resume": True,
        "gate_sha256": _sha(Path(args.gate)),
    }
    provenance = capture_provenance(ROOT, resolved_config=config)
    source_hash = provenance["source_snapshot"]["sha256"]
    manifest = {
        "head_commit": provenance["head_commit"],
        "source_snapshot_hash": source_hash,
        "source_snapshot": provenance["source_snapshot"],
        "resolved_config": config,
    }
    output.mkdir()
    _copy_source_archive(output, manifest, provenance)
    _json_atomic(output / "execution_manifest.json", manifest)
    shutil.copyfile(args.gate, output / "launch_gate.json")
    runtime = {
        "status": "LAUNCHING",
        "source_snapshot_hash": source_hash,
        "started_at": _now(),
        "child_pid": None,
        "completed": [],
        "dataset": None,
        "phase": None,
    }
    _json_atomic(output / "runtime.json", runtime)
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "WANDB_MODE": "online",
        "PYTHONPATH": str(ROOT / "src"),
    }
    results = {}
    try:
        for dataset in DATASETS:
            run, evaluation = output / "runs" / dataset, output / "evaluation" / dataset
            commands = {
                "train": [
                    sys.executable,
                    "-m",
                    "spica.train_coupled_benchmark",
                    "--dataset",
                    dataset,
                    "--output-dir",
                    str(run),
                    "--campaign-root",
                    str(output),
                    "--device",
                    "cuda",
                    "--wandb-mode",
                    "online",
                ] + (["--periodic-test"] if periodic_test else []),
                "evaluate": [
                    sys.executable,
                    "scripts/evaluate_coupled_benchmark.py",
                    "--run-dir",
                    str(run),
                    "--output-dir",
                    str(evaluation),
                    "--device",
                    "cuda",
                ],
            }
            phases = ("train",) if periodic_test else tuple(commands)
            for phase in phases:
                command = commands[phase]
                if capture_provenance(ROOT)["source_snapshot"]["sha256"] != source_hash:
                    raise RuntimeError("source changed before child")
                runtime.update(
                    status="RUNNING", dataset=dataset, phase=phase, command=command
                )
                child, stdout, stderr = _start_child(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=output / "logs" / f"{dataset}.{phase}.stdout.log",
                    stderr=output / "logs" / f"{dataset}.{phase}.stderr.log",
                )
                runtime["child_pid"] = child.pid
                _json_atomic(output / "runtime.json", runtime)
                try:
                    code = child.wait()
                finally:
                    stdout.close()
                    stderr.close()
                runtime.update(child_pid=None, exit_code=code)
                if code:
                    raise RuntimeError(f"{dataset}/{phase} exited {code}; no retry")
                if capture_provenance(ROOT)["source_snapshot"]["sha256"] != source_hash:
                    raise RuntimeError("source changed during child")
                if phase == "train":
                    train = check_finished_train(
                        run, ROOT / gate["smoke_roots"][dataset], dataset, source_hash,
                        periodic_test=periodic_test,
                    )
                    _json_atomic(
                        run / "parent_final_gate.json",
                        {
                            "status": "PASS",
                            "scope": "final_checkpoint_init_inputs_before_official_evaluation",
                            "step": train["step"],
                            "checkpoint_sha256": train["selections"]["latest"][
                                "sha256"
                            ],
                        },
                    )
                elif not periodic_test:
                    results[dataset] = _check_evaluation(
                        evaluation, train, dataset, source_hash
                    )
            if periodic_test:
                results[dataset] = _periodic_final_result(run, train, dataset)
            runtime["completed"].append(dataset)
            _json_atomic(output / "runtime.json", runtime)
        _json_atomic(
            output / "results.json",
            {
                "status": (PERIODIC_STATUS if periodic_test else "LOCAL_RESULTS_PENDING_INDEPENDENT_REVIEW"),
                "evaluation_policy": config["evaluation_policy"],
                "source_snapshot_hash": source_hash,
                "datasets": results,
            },
        )
        runtime.update(
            status=(PERIODIC_STATUS if periodic_test else "TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED"), finished_at=_now()
        )
        _json_atomic(output / "runtime.json", runtime)
        return 0
    except BaseException:
        runtime.update(
            status="FAILED_NO_RETRY",
            finished_at=_now(),
            traceback=traceback.format_exc(),
        )
        _json_atomic(output / "runtime.json", runtime)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--gate")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--periodic-test", action="store_true", help="run official test every 1000 updates and once at final")
    args = parser.parse_args()
    if not args.launch:
        print(
            json.dumps(
                {
                    "status": "INERT",
                    "would_launch": False,
                    "datasets": DATASETS,
                    "runtime_policy": OFFICIAL_RUNTIME if args.periodic_test else RUNTIME,
                }
            )
        )
        return 0
    if not args.output or not args.gate:
        parser.error("--launch requires --output and --gate")
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
