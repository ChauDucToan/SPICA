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

from run_coupled_campaign import _copy_source_archive, _json_atomic, _now, _start_child
from run_fusion_campaign import _read, _sha
from spica.provenance import capture_provenance
from spica.data.coupled_benchmark import OFFICIAL_IDENTITIES, _plain
from spica.runtime import runtime_policy
from spica.train_coupled_benchmark import ARM, DATASETS, METHOD_VERSION

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {"probe_every": 5, "checkpoint_every": 100}
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


def _required_evidence(gate_path, gate):
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
            if receipt.get("runtime_policy") != RUNTIME:
                raise ValueError(f"smoke runtime policy mismatch: {dataset}")
            expected = {
                "optimizer_states": 179,
                "moment_tensors": 358,
                "restore_exact": True,
                "probe_rows": 320,
                "official_images_opened": 0,
                "scope": "CPU_SAVED_SMOKE_RESTORE_AND_RUNTIME_PROBE",
            }
            if not receipt.get("compact_whole_model_cuda_restore_exact"):
                raise ValueError(f"missing actual compact CUDA restore: {dataset}")
            if receipt.get("evaluator_sha256") != gate['component_sha256'].get('scripts/evaluate_coupled_benchmark.py'):
                raise ValueError(f"stale evaluator restore gate: {dataset}")
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError(f"incomplete saved restore/probe evidence: {dataset}")
    return {name: required[name] for name in EVIDENCE_NAMES}


def gate_check(path: Path):
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
    if (
        gate.get("runtime_policy", gate.get("runtime")) != RUNTIME
        or runtime_policy() != RUNTIME
    ):
        raise ValueError("minimal runtime policy must be probe=5/checkpoint=100")
    components = gate.get("component_sha256")
    if not isinstance(components, dict) or set(components) != set(COMPONENTS):
        raise ValueError(
            "gate component coverage is not the current production contract"
        )
    for name, digest in components.items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"gate source changed: {name}")
    _required_evidence(path, gate)
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
            or cfg.get("runtime") != RUNTIME
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


def check_finished_train(run: Path, smoke: Path, dataset: str, source_hash: str):
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
        or cfg.get("runtime") != RUNTIME
        or cfg.get("checkpoint_policy") != "rolling_latest"
        or cfg.get("tracking_policy") != "minimal_v1"
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
    _finite_probe(run, horizon)
    return result


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
    if (
        output.exists()
        or not output.is_relative_to(ROOT / "outputs")
        or output == ROOT / "outputs"
    ):
        raise ValueError("fresh child of outputs/ required")
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise RuntimeError(
            "at least 24GiB free required; no artifact cleanup permitted"
        )
    gate = gate_check(Path(args.gate))
    config = {
        "method": METHOD_VERSION,
        "arm": ARM,
        "datasets": DATASETS,
        "dataset_identities": EXPECTED_IDENTITIES,
        "runtime_policy": RUNTIME,
        "order": list(DATASETS),
        "evaluation": "final_only_clean_plus_9_masks",
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
                ],
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
            for phase, command in commands.items():
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
                        run, ROOT / gate["smoke_roots"][dataset], dataset, source_hash
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
                else:
                    results[dataset] = _check_evaluation(
                        evaluation, train, dataset, source_hash
                    )
            runtime["completed"].append(dataset)
            _json_atomic(output / "runtime.json", runtime)
        _json_atomic(
            output / "results.json",
            {
                "status": "LOCAL_RESULTS_PENDING_INDEPENDENT_REVIEW",
                "source_snapshot_hash": source_hash,
                "datasets": results,
            },
        )
        runtime.update(
            status="TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED", finished_at=_now()
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
    args = parser.parse_args()
    if not args.launch:
        print(
            json.dumps(
                {
                    "status": "INERT",
                    "would_launch": False,
                    "datasets": DATASETS,
                    "runtime_policy": RUNTIME,
                }
            )
        )
        return 0
    if not args.output or not args.gate:
        parser.error("--launch requires --output and --gate")
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
