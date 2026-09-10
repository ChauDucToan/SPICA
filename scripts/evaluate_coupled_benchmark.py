#!/usr/bin/env python3
"""Final-only TU-Berlin/QuickDraw evaluator for F2_MP_Q_OFFICIAL.

The default route performs all provenance/checkpoint gates before constructing an
official test loader.  ``--train-probe`` is an actual pretrained CUDA smoke-2
probe over deterministic official *training* entries; it never evaluates the
official test split.  ``--cpu-self-check`` needs no run at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spica.data.coupled_benchmark import (  # noqa: E402
    _plain,
    build_test_loaders,
    load_benchmark_protocol,
)
from spica.data.datasets import RetrievalEvalDataset  # noqa: E402
from spica.evaluation.coupled_benchmark import (  # noqa: E402
    cpu_self_check,
    evaluate_coupled_benchmark,
    module_state_hash,
)
from spica.evaluation.coupled_predictive import QOnlyAdapter  # noqa: E402
from spica.evaluation.masked_view import (  # noqa: E402
    _masked_loader,
    _transform_stats,
)
from spica.models.checkpoint import load_trainable_state  # noqa: E402
from spica.models.clip import load_frozen_clip  # noqa: E402
from spica.models.coupled_predictive import CoupledPredictiveModel  # noqa: E402
from spica.provenance import source_snapshot  # noqa: E402

METHOD = "coupled_predictive_mp_official_v1"
ARM = "F2_MP_Q_OFFICIAL"
ARCHITECTURE = "predictive_fusion_v2"
EXPECTED_STEPS = {"tuberlin_220_30": 1189, "quickdraw_80_30": 18229}


def _json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve(value: Any, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _config(result: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    embedded = result.get("resolved_config")
    embedded_config = dict(embedded) if isinstance(embedded, Mapping) else None
    path = run_dir / "resolved_config.json"
    recorded_config = dict(_json(path)) if path.is_file() else None
    if embedded_config is not None and recorded_config is not None and embedded_config != recorded_config:
        raise ValueError("embedded and recorded resolved_config differ")
    if embedded_config is not None:
        return embedded_config
    if recorded_config is not None:
        return recorded_config
    raise ValueError("run result has no resolved_config and resolved_config.json is absent")


def _benchmark_name(config: Mapping[str, Any]) -> str:
    value = config.get("dataset")
    if not isinstance(value, str) or value not in EXPECTED_STEPS:
        raise ValueError("config.dataset must explicitly be tuberlin_220_30 or quickdraw_80_30")
    return value


def _class_names(config: Mapping[str, Any]) -> dict[int, str]:
    raw = config.get("train_class_names", config.get("class_names"))
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("config.train_class_names is required")
    result = {int(key): str(value) for key, value in raw.items()}
    if sorted(result) != list(range(len(result))) or len(set(result.values())) != len(result):
        raise ValueError("config.class_names must be a complete ordered train semantic class map")
    return result


def _run_result(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_dir = run_dir.expanduser().resolve()
    result_path = run_dir / "run_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing {result_path}")
    result = dict(_json(result_path))
    config = _config(result, run_dir)
    if result.get("status") != "COMPLETE":
        raise ValueError("official evaluation requires a COMPLETE trainer result")
    if result.get("campaign") != METHOD or result.get("method_version", METHOD) != METHOD:
        raise ValueError("run is not coupled_predictive_mp_official_v1")
    if result.get("arm") != ARM or config.get("arm") not in {None, ARM}:
        raise ValueError("run is not F2_MP_Q_OFFICIAL")
    if config.get("method_version") != METHOD or config.get("architecture") != ARCHITECTURE:
        raise ValueError("run method or architecture is not the official F2_MP route")
    if config.get("official_unseen_used_for_selection", False) is not False:
        raise ValueError("official test data was used for selection")
    if config.get("official_unseen_used_for_training", False) is not False:
        raise ValueError("official test data was used for training")
    if config.get("selection_policy") != "none;final_only":
        raise ValueError("official run must declare selection_policy=none;final_only")
    return result, config, run_dir


def _safe_checkpoint_load(path: Path) -> Mapping[str, Any]:
    """Load trusted trainer tensors with a scoped NumPy allowlist only."""
    try:
        reconstruct = np._core.multiarray._reconstruct
    except AttributeError:  # NumPy < 2 exposes the same symbol here.
        reconstruct = np.core.multiarray._reconstruct
    safe = [
        reconstruct, np.ndarray, np.dtype,
        np.float64, np.uint32, np.int64,
        type(np.dtype(np.float64)), type(np.dtype(np.uint32)), type(np.dtype(np.int64)),
    ]
    try:
        with torch.serialization.safe_globals(safe):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint cannot be loaded with scoped weights_only=True; no fallback is allowed") from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def _gate_checkpoint(result: Mapping[str, Any], config: Mapping[str, Any], run_dir: Path, benchmark: str, *, smoke: bool) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    final_step = 2 if smoke else EXPECTED_STEPS[benchmark]
    recorded_step = result.get("step", result.get("completed_steps"))
    if int(recorded_step) != final_step or int(result.get("completed_steps", recorded_step)) != final_step:
        raise ValueError(f"result is not the final {final_step}-step run")
    expected_selection = result.get("selections")
    if not isinstance(expected_selection, Mapping) or not isinstance(expected_selection.get("latest"), Mapping):
        raise ValueError("run result selections.latest is required")
    latest = dict(expected_selection["latest"])
    if int(latest.get("step", -1)) != final_step:
        raise ValueError("selections.latest does not point at the final step")
    for key in ("path", "sha256"):
        if not isinstance(latest.get(key), str) or not latest[key]:
            raise ValueError(f"selections.latest.{key} is missing")
    checkpoint = _resolve(latest["path"], base=run_dir)
    try:
        checkpoint.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("latest checkpoint escapes run directory") from error
    if not checkpoint.is_file() or _sha256(checkpoint) != latest["sha256"]:
        raise ValueError("latest checkpoint is absent or has the wrong SHA256")
    rows = result.get("checkpoints")
    if not isinstance(rows, list) or not any(
        isinstance(row, Mapping) and int(row.get("step", -1)) == final_step
        and row.get("path") == latest["path"] and row.get("sha256") == latest["sha256"]
        for row in rows
    ):
        raise ValueError("latest selection is not bound to the final checkpoint record")
    # There is intentionally no pickle fallback.  A trainer checkpoint must be
    # consumable by PyTorch's restricted weights-only loader.
    payload = _safe_checkpoint_load(checkpoint)
    if int(payload.get("step", -1)) != final_step:
        raise ValueError("checkpoint step does not match selections.latest")
    if payload.get("source_snapshot_hash") != result.get("source_snapshot_hash"):
        raise ValueError("checkpoint source hash differs from run result")
    if result.get("frozen_original_state_hash_before") != result.get("frozen_original_state_hash_after"):
        raise ValueError("trainer result reports a changed frozen original CLIP")
    expected_clip = config.get("clip_identity")
    if not isinstance(expected_clip, Mapping) or payload.get("clip_identity") != expected_clip:
        raise ValueError("checkpoint frozen CLIP identity differs from config")
    if payload.get("resolved_config") != dict(config):
        raise ValueError("checkpoint resolved_config differs from run result")
    expected_data = config.get("protocol_identity")
    if not isinstance(expected_data, Mapping) or payload.get("data_identity") != expected_data:
        raise ValueError("checkpoint data_identity differs from config.protocol_identity")
    if payload.get("initialization_hashes") != config.get("initialization_hashes"):
        raise ValueError("checkpoint initialization hashes differ from config")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint model_state_dict is missing")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint model_state_dict contains a non-tensor value")
    for key in ("model_state_hash", "initialization_hashes"):
        if key not in payload:
            raise ValueError(f"checkpoint frozen/state identity is missing: {key}")
    return latest, checkpoint, dict(payload)


def _source_gate(result: Mapping[str, Any]) -> str:
    recorded = result.get("source_snapshot_hash", result.get("source_hash"))
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("run result source_snapshot_hash is missing")
    if result.get("source_snapshot_hash_after") != recorded:
        raise ValueError("trainer source snapshot changed during training")
    current = source_snapshot(ROOT)
    if current.get("sha256") != recorded:
        raise ValueError("current source snapshot does not match the trainer result")
    return recorded


def _protocol_gate(config: Mapping[str, Any], benchmark: str):
    # This function only validates/stat-checks manifests.  No image loader is
    # constructed or iterated until every run/checkpoint/source gate is done.
    protocol = load_benchmark_protocol(benchmark, split="test")
    expected_identity = _plain(protocol.identity)
    recorded_identity = config.get("protocol_identity")
    if recorded_identity != expected_identity:
        raise ValueError("config protocol_identity is not data.coupled_benchmark._plain(protocol.identity)")
    names = _class_names(config)
    if names != dict(protocol.train.class_names):
        raise ValueError("config.class_names must be the TRAIN semantic class names")
    return protocol, names


def _round_robin_entries(entries: tuple[Any, ...], *, count: int) -> tuple[Any, ...]:
    grouped: dict[int, list[Any]] = {}
    for entry in entries:
        grouped.setdefault(int(entry.label), []).append(entry)
    selected = [grouped[label][0] for label in sorted(grouped)[:count]]
    if len(selected) != count:
        raise ValueError(f"train probe needs {count} classes, found {len(selected)}")
    return tuple(selected)


def _probe_gallery(entries: tuple[Any, ...], query_classes: set[int], *, count: int) -> tuple[Any, ...]:
    selected: list[Any] = []
    seen_paths: set[str] = set()
    seen_labels: set[int] = set()
    for entry in entries:
        path = str(entry.path)
        if entry.label in query_classes and entry.label not in seen_labels:
            selected.append(entry)
            seen_paths.add(path)
            seen_labels.add(entry.label)
    for entry in entries:
        path = str(entry.path)
        if len(selected) == count:
            break
        if entry.label not in query_classes and path not in seen_paths:
            selected.append(entry)
            seen_paths.add(path)
    if len(selected) != count or not query_classes <= {int(entry.label) for entry in selected}:
        raise ValueError("train probe gallery must contain 256 unique entries and every positive")
    return tuple(selected)


def _check_q_bypass(model: Any, loader: Any, device: torch.device) -> None:
    adapter = QOnlyAdapter(model)
    before = module_state_hash(model)
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            if not torch.equal(model(images).q, adapter(images)):
                raise ValueError("q full-output and predictor-bypass outputs differ")
            seen += int(images.shape[0])
            break
    if seen < 32 or module_state_hash(model) != before:
        raise RuntimeError("q bypass check failed: expected at least 32 real queries and unchanged state")


def _build_model(
    config: Mapping[str, Any], checkpoint: Mapping[str, Any], device: torch.device,
    names: Mapping[int, str], *, expected_original_hash: str,
):
    clip = config.get("clip_identity")
    if not isinstance(clip, Mapping) or not clip.get("path") or not isinstance(clip.get("sha256"), str):
        raise ValueError("config.clip_identity.path and sha256 are required")
    clip_path = Path(str(clip["path"])).expanduser()
    if not clip_path.is_file():
        raise FileNotFoundError(f"frozen CLIP cache is missing: {clip_path}")
    if _sha256(clip_path) != clip["sha256"]:
        raise ValueError("frozen CLIP cache SHA256 mismatch")
    model_name = str(config.get("model_name", "ViT-B-32-quickgelu"))
    bundle = load_frozen_clip(model_name=model_name, pretrained=str(clip_path), device=device)
    kwargs = {
        key: int(config[key])
        for key in ("photo_prompt_length", "text_prompt_length", "predictor_width", "predictor_heads")
        if key in config
    }
    model = CoupledPredictiveModel(
        bundle.encoder,
        bundle.tokenizer,
        dict(names),
        architecture=ARCHITECTURE,
        **kwargs,
    ).to(device)
    load_info = load_trainable_state(model, checkpoint["model_state_dict"])
    missing = load_info.get("missing_frozen_keys", [])
    if any(not str(key).startswith("original_clip.") for key in missing):
        raise ValueError(f"checkpoint omits non-original frozen state: {sorted(missing)}")
    model.eval()
    if tuple(int(x) for x in model.classids.detach().cpu().tolist()) != tuple(sorted(names)):
        raise ValueError("model text bank contains classes outside the TRAIN class map")
    if module_state_hash(model) != checkpoint["model_state_hash"]:
        raise ValueError("restored model state hash does not match checkpoint")
    original_hash = module_state_hash(model, prefix="original_clip.")
    if original_hash != expected_original_hash:
        raise ValueError("restored original CLIP hash differs from config/result")
    return model, bundle


def _official(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    result, config, run_dir = _run_result(run_dir)
    benchmark = _benchmark_name(config)
    latest, checkpoint, payload = _gate_checkpoint(result, config, run_dir, benchmark, smoke=False)
    source_hash = _source_gate(result)
    protocol, names = _protocol_gate(config, benchmark)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    expected_original_hash = config.get("frozen_original_state_hash")
    if not isinstance(expected_original_hash, str):
        raise ValueError("config.frozen_original_state_hash is required")
    if result.get("frozen_original_state_hash_before") != expected_original_hash or result.get("frozen_original_state_hash_after") != expected_original_hash:
        raise ValueError("result frozen original CLIP hashes differ from config")
    model, bundle = _build_model(config, payload, device, names, expected_original_hash=expected_original_hash)
    del payload
    loaders = build_test_loaders(protocol, bundle.transform, batch_size=256, num_workers=0)
    mean, std = _transform_stats(bundle.transform)
    entries = tuple(protocol.test.sketch_entries)
    data_root = protocol.root

    def masked(fraction: float, seed: int):
        return _masked_loader(
            entries,
            bundle.transform,
            root=data_root,
            fraction=fraction,
            seed=seed,
            batch_size=256,
            num_workers=0,
            mean=mean,
            std=std,
            ink_threshold=0.9,
        )

    return evaluate_coupled_benchmark(
        model,
        loaders["sketch"],
        loaders["photo"],
        masked,
        Path(args.output_dir),
        device=device,
        benchmark=benchmark,
        protocol_identity=_plain(protocol.identity),
        class_names=names,
        source_hash=source_hash,
        checkpoint_selection={"latest": latest, "step": int(result["step"]), "path": str(checkpoint), "sha256": latest["sha256"]},
        data_identity=config["protocol_identity"],
        status="COMPLETE",
    )


def _train_probe(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    result, config, run_dir = _run_result(run_dir)
    requested_device = torch.device(args.device)
    if requested_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("--train-probe requires an available CUDA device")
    if config.get("smoke") is not True or torch.device(str(config.get("device"))).type != "cuda":
        raise ValueError("--train-probe requires an actual pretrained CUDA smoke-2 run")
    benchmark = _benchmark_name(config)
    latest, checkpoint, payload = _gate_checkpoint(result, config, run_dir, benchmark, smoke=True)
    if config.get("official_test_loaded", False) or config.get("official_unseen_used", False):
        raise ValueError("smoke run metadata claims official data access")
    source_hash = _source_gate(result)
    protocol, names = _protocol_gate(config, benchmark)
    expected_original_hash = config.get("frozen_original_state_hash")
    if not isinstance(expected_original_hash, str):
        raise ValueError("config.frozen_original_state_hash is required")
    if result.get("frozen_original_state_hash_before") != expected_original_hash or result.get("frozen_original_state_hash_after") != expected_original_hash:
        raise ValueError("result frozen original CLIP hashes differ from config")
    model, bundle = _build_model(config, payload, requested_device, names, expected_original_hash=expected_original_hash)
    del payload
    query_entries = _round_robin_entries(protocol.train.sketch_entries, count=32)
    query_classes = {entry.label for entry in query_entries}
    if len(query_classes) != 32:
        raise ValueError("train probe must select 32 distinct train classes")
    gallery_entries = _probe_gallery(protocol.train.photo_entries, query_classes, count=256)
    _check_q_bypass(model, DataLoader(
        RetrievalEvalDataset(query_entries, bundle.transform), batch_size=32, shuffle=False, num_workers=0
    ), requested_device)
    transform = bundle.transform
    clean_loader = DataLoader(
        RetrievalEvalDataset(query_entries, transform), batch_size=32, shuffle=False, num_workers=0
    )
    gallery_loader = DataLoader(
        RetrievalEvalDataset(tuple(gallery_entries), transform), batch_size=64, shuffle=False, num_workers=0
    )
    mean, std = _transform_stats(transform)

    def masked(fraction: float, seed: int):
        return _masked_loader(
            query_entries, transform, root=protocol.root, fraction=fraction, seed=seed,
            batch_size=32, num_workers=0, mean=mean, std=std, ink_threshold=0.9,
        )

    report = evaluate_coupled_benchmark(
        model, clean_loader, gallery_loader, masked, Path(args.output_dir), device=requested_device,
        benchmark=benchmark, protocol_identity=_plain(protocol.identity), class_names=names,
        source_hash=source_hash,
        checkpoint_selection={"latest": latest, "step": int(result["step"]), "path": str(checkpoint), "sha256": latest["sha256"]},
        data_identity=config["protocol_identity"],
        status="TRAIN_ONLY_PREFLIGHT_NOT_OFFICIAL",
    )
    report.update({
        "official_image_loader_opened": False,
        "official_test_loader_opened": False,
        "official_test_evaluated": False,
        "official_unseen_used_for_selection": False,
        "train_only_query_count": len(query_entries),
        "train_only_gallery_count": len(gallery_entries),
        "train_only_query_classes": sorted(query_classes),
        "positive_gallery_coverage": True,
        "scope": "first32_train_queries;gallery_at_most_256;all_clean_and_9_masks",
    })
    _atomic(Path(args.output_dir) / 'train_probe_scope.json', {key: value for key, value in report.items() if key.startswith(('official_', 'train_only_')) or key in {'scope', 'positive_gallery_coverage'}})
    return report


def _write_failure(output_value: str | None, error: BaseException) -> None:
    if not output_value:
        return
    output = Path(output_value).expanduser().resolve()
    if output.exists():
        path = output / "failure.json"
        if path.exists():
            return
    else:
        output.mkdir(parents=True)
    _atomic(output / "failure.json", {
        "status": "FAIL",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-probe", action="store_true")
    parser.add_argument("--cpu-self-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_check:
        print(json.dumps(cpu_self_check(), sort_keys=True))
        return 0
    if args.run_dir is None or args.output_dir is None:
        parser.error("--run-dir and --output-dir are required unless --cpu-self-check is used")
    if args.output_dir.expanduser().resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    try:
        report = _train_probe(args) if args.train_probe else _official(args)
    except Exception as error:
        _write_failure(str(args.output_dir), error)
        raise
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
