#!/usr/bin/env python3
"""Initial-state photo-CE lambda calibration; no optimizer update or evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import median
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from diagnose_coupled_sigreg import (  # noqa: E402
    _all_float32, _artifact_record, _file_sha256, _fresh_output,
    _global_rng_state, _module_hash, _move_batch, _seed42, _tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "outputs/fusion_mp_execution_20260909T115500Z/runs/F2_MP"
INITIALIZATION = HISTORICAL / "initialization.json"
BATCHES, BATCH_SIZE, RHO, TAU = 4, 32, 0.1, 0.07
OBJECTIVE = "multi_positive_supervised_contrastive"
SCOPES = ("photo_model.photo_prompt", "text_bank.context")


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _first128(path: Path) -> list[dict[str, object]]:
    import itertools
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in itertools.islice(handle, 128)]
    if len(rows) != 128:
        raise AssertionError(f"{path} does not contain the required 128 rows")
    return rows


def _strip_trace(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"step", "actual_lr"}}


def _prompt_scopes(model: object) -> dict[str, list[tuple[str, object]]]:
    names = dict(model.named_parameters())
    if not set(SCOPES) <= names.keys():
        raise AssertionError("required prompt parameters are not registered")
    result = {name: [(name, names[name])] for name in SCOPES}
    if any(not rows[0][1].requires_grad for rows in result.values()):
        raise AssertionError("photo/text prompt scopes must be trainable")
    return result


def _losses(model: object, batch: dict[str, object]) -> dict[str, object]:
    import torch
    from spica.coupled_predictive_losses import coupled_region_loss

    terms = coupled_region_loss(
        model, *(batch[key] for key in ("clean", "corrupted", "photos", "positive_indices", "negative_indices", "labels", "photo_labels", "photo_ids")),
        lambda_sig=0.0, sigreg=None, main_photo_objective=OBJECTIVE, lambda_photo_ce=0.0,
    )
    result = {"base": terms["total"], "photo_ce": terms["photo_ce"]}
    if not all(torch.isfinite(value).item() for value in result.values()):
        raise FloatingPointError("nonfinite diagnostic loss")
    return result


def _gradients(loss: object, scopes: dict[str, list[tuple[str, object]]], *, retain: bool) -> tuple[dict[str, object], dict[str, list[str]]]:
    import torch
    rows = [(name, parameter) for scope in SCOPES for name, parameter in scopes[scope]]
    values = iter(torch.autograd.grad(loss, [parameter for _, parameter in rows], retain_graph=retain, allow_unused=True))
    vectors, unused = {}, {}
    for scope in SCOPES:
        name, parameter = scopes[scope][0]
        gradient = next(values)
        if gradient is None:
            unused[scope] = [name]
            gradient = torch.zeros_like(parameter)
        else:
            unused[scope] = []
        if gradient.dtype != torch.float32 or gradient.shape != parameter.shape or not torch.isfinite(gradient).all().item():
            raise FloatingPointError(f"invalid {scope} gradient")
        vectors[scope] = gradient.detach().cpu().contiguous()
    return vectors, unused


def _norm(value: object, *, positive: bool = False) -> float:
    import torch
    result = float(torch.linalg.vector_norm(value.to(dtype=torch.float64)))
    if not math.isfinite(result) or (positive and result <= 0):
        raise FloatingPointError("nonfinite or zero calibration norm")
    return result


def _cosine(left: object, right: object) -> float:
    import torch
    a, b = left.to(dtype=torch.float64).flatten(), right.to(dtype=torch.float64).flatten()
    na, nb = _norm(a), _norm(b)
    value = float(torch.dot(a, b) / (na * nb))
    if not torch.isfinite(torch.tensor(value, dtype=torch.float64)).item():
        raise FloatingPointError("nonfinite gradient cosine")
    return value


def _select(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != BATCHES:
        raise ValueError("photo-CE calibration requires exactly four batches")
    candidates = {}
    per_scope = {}
    for scope in SCOPES:
        ratios = []
        for row in rows:
            base = row["gradients"]["base"][scope]
            photo = row["gradients"]["photo_ce"][scope]
            ratios.append(_norm(base, positive=True) / _norm(photo, positive=True))
        middle = median(ratios)
        candidates[scope] = RHO * middle
        per_scope[scope] = {"ratios_base_over_photo_ce": ratios, "median_ratio": middle, "rho": RHO}
    binding = min(SCOPES, key=lambda scope: candidates[scope])
    value = float(candidates[binding])
    if not all(math.isfinite(x) and x > 0 for x in candidates.values()):
        raise FloatingPointError("photo-CE lambda candidates must be finite and positive")
    return {
        "lambda_photo_ce": value,
        "lambda_candidates": candidates,
        "binding_scope": binding,
        "scope_selection": per_scope,
        "rho": RHO,
        "batches": BATCHES,
        "formula": "lambda_photo_ce=.1*min_s median_b(||g_base_s||_2/||g_photo_ce_s||_2)",
        "scope_notes": "photo_model.photo_prompt and text_bank.context only; target median, not a per-batch cap, optimum, or effective AdamW-update ratio",
    }


def _rows(raw: list[dict[str, object]], selection: dict[str, object]) -> list[dict[str, object]]:
    lam = float(selection["lambda_photo_ce"])
    result = []
    for row in raw:
        scopes = {}
        for scope in SCOPES:
            base = row["gradients"]["base"][scope].double()
            photo = row["gradients"]["photo_ce"][scope].double()
            nb, np = _norm(base, positive=True), _norm(photo, positive=True)
            scopes[scope] = {
                "base_norm_l2_float64": nb,
                "photo_ce_norm_l2_float64": np,
                "base_over_photo_ce_float64": nb / np,
                "cosine_base_photo_ce_float64": _cosine(base, photo),
                "realized_photo_ce_over_base_float64": lam * np / nb,
                "weighted_norm_l2_float64": _norm(base + lam * photo),
            }
        result.append({"batch_index": row["batch_index"], "losses": row["losses"], "scopes": scopes})
    return result


def _run(output: Path) -> dict[str, object]:
    import torch
    from spica.data.coupled_training import load_protocol_data, make_train_loader, prepare_batch, positive_pool_identity, verify_clip_cache
    from spica.models.clip import load_frozen_clip
    from spica.models.coupled_predictive import CoupledPredictiveModel
    from spica.provenance import capture_provenance
    from spica.train_coupled_predictive import _initialization_hashes
    from run_coupled_campaign import _copy_source_archive

    if not torch.cuda.is_available():
        raise RuntimeError("calibration requires the parent CUDA environment; refusing CPU fallback")
    config = {"diagnostic": "fusion_mp_photo_ce_init_v1", "seed": 42, "batches": BATCHES, "batch_size": BATCH_SIZE,
              "rho": RHO, "tau": TAU, "positive_pool": "full", "optimizer_updates": 0, "wandb": "disabled"}
    provenance = capture_provenance(ROOT, resolved_config=config)
    _copy_source_archive(output, config, provenance)
    _json(output / "provenance.json", provenance)
    protocol, clip = load_protocol_data(), verify_clip_cache()
    _seed42()
    split, names = protocol["split"], protocol["names"]
    classmap = {int(i): str(names[int(i)]) for i in split.train_class_ids}
    assert len(classmap) == 84
    bundle = load_frozen_clip(model_name="ViT-B-32-quickgelu", pretrained=str(clip["path"]), device=torch.device("cuda"))
    model = CoupledPredictiveModel(bundle.encoder, bundle.tokenizer, classmap, architecture="predictive_fusion_v2").to("cuda")
    model.train(True)
    _all_float32(model, "model")
    initialization = _initialization_hashes(model)
    expected_initialization = json.loads(INITIALIZATION.read_text(encoding="utf-8"))
    if initialization != expected_initialization:
        raise AssertionError("initialization hashes do not equal historical F2_MP initialization")
    scopes = _prompt_scopes(model)
    layout = {scope: [{"name": name, "shape": list(parameter.shape), "numel": int(parameter.numel()), "dtype": str(parameter.dtype)} for name, parameter in rows] for scope, rows in scopes.items()}
    _json(output / "gradient_layout.json", layout)
    before, teacher_before = _module_hash(model), _module_hash(model.original_clip)
    traces = _first128(HISTORICAL / "observation_trace.jsonl")
    masks = _first128(HISTORICAL / "mask_metadata.jsonl")
    loader = make_train_loader(protocol, bundle.transform, positive_pool="full")
    if not torch.equal(loader.generator.get_state(), torch.Generator().manual_seed(42).get_state()):
        raise AssertionError("full-pool loader generator is not seed 42")
    records, raw = [], []
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    for batch_index in range(BATCHES):
        batch = prepare_batch(next(iterator), data_root=Path(str(protocol["data"].root)), step=batch_index, classids=model.classids.cpu().tolist())
        expected = [_strip_trace(row) for row in traces[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]]
        actual = [_strip_trace(row) for row in batch["trace"]]
        expected_masks = masks[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        actual_masks = batch["mask_metadata"]["rows"]
        if actual != expected or actual_masks != expected_masks:
            raise AssertionError(f"historical trace/mask mismatch at batch {batch_index}")
        if batch["clean"].shape[0] != BATCH_SIZE or len(batch["photos"]) > 64:
            raise AssertionError("unique live photo bank exceeds 64")
        batch_record = {"batch_index": batch_index, "trace": batch["trace"], "mask_metadata": batch["mask_metadata"], "photo_ids": list(batch["photo_ids"]),
                        "tensors": {key: {"shape": list(value.shape), "sha256": _tensor_sha256(value)} for key, value in batch.items() if isinstance(value, torch.Tensor)},
                        "labels": batch["labels"].tolist(), "photo_labels": batch["photo_labels"].tolist()}
        records.append(batch_record)
        batch = _move_batch(batch, torch.device("cuda"))
        model_hash, teacher_hash, rng = _module_hash(model), _module_hash(model.original_clip), _global_rng_state()
        losses = _losses(model, batch)
        gradients, unused_base = _gradients(losses["base"], scopes, retain=True)
        photo_gradients, unused_photo = _gradients(losses["photo_ce"], scopes, retain=False)
        for scope in SCOPES:
            if unused_base[scope] or unused_photo[scope]:
                raise AssertionError(f"unexpected disconnected prompt gradient: {scope}")
        if _module_hash(model) != model_hash or _module_hash(model.original_clip) != teacher_hash or _global_rng_state() != rng:
            raise AssertionError(f"state changed during batch {batch_index}")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("autograd.grad populated a model .grad")
        gradients_all = {"base": gradients, "photo_ce": photo_gradients}
        raw_path = output / f"raw_gradients_batch{batch_index:02d}.pt"
        torch.save({"batch_index": batch_index, "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
                    "gradients": gradients_all, "unused": {"base": unused_base, "photo_ce": unused_photo}, "layout": layout}, raw_path)
        raw.append({"batch_index": batch_index, "losses": {key: float(value.detach().cpu()) for key, value in losses.items()}, "gradients": gradients_all})
        del batch, losses, gradients, photo_gradients
    selection = _select(raw)
    measured = _rows(raw, selection)
    _json(output / "batch_records.json", records)
    _json(output / "lambda_selection.json", selection)
    _json(output / "measured_rows.json", measured)
    if _module_hash(model) != before or _module_hash(model.original_clip) != teacher_before or not all(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("model or teacher changed after no-update diagnostic")
    if model.original_clip.training or any(parameter.requires_grad for parameter in model.original_clip.parameters()):
        raise AssertionError("teacher is not frozen/eval")
    historical = {str(path.relative_to(ROOT)): _file_sha256(path) for path in (INITIALIZATION, HISTORICAL / "observation_trace.jsonl", HISTORICAL / "mask_metadata.jsonl")}
    data_identity = {"split": protocol["split_identity"], "manifest": protocol["manifest_identity"], "pool": positive_pool_identity(protocol, "full")}
    artifacts = ["provenance.json", "gradient_layout.json", "batch_records.json", "lambda_selection.json", "measured_rows.json", *(f"raw_gradients_batch{i:02d}.pt" for i in range(BATCHES))]
    if capture_provenance(ROOT)["source_snapshot"]["sha256"] != provenance["source_snapshot"]["sha256"]:
        raise RuntimeError("source changed during calibration")
    input_hashes = {str((output / name).relative_to(ROOT)): _file_sha256(output / name) for name in (*artifacts, "source_snapshot/index.json")}
    input_hashes.update(historical)
    result = {"status": "MEASURED_PENDING_REVIEW", "verified": False, "config": config, "input_sha256": input_hashes,
              "selection": selection, "rows": measured, "initialization_hashes": initialization, "historical_bindings": historical, "data_identity": data_identity, "clip_identity": clip,
              "source_snapshot_hash": provenance["source_snapshot"]["sha256"], "model_hash_before": before, "model_hash_after": _module_hash(model), "teacher_hash_before": teacher_before, "teacher_hash_after": _module_hash(model.original_clip),
              "model_unchanged": True, "teacher_unchanged": True, "parameter_grad_all_none": True, "optimizer_updates": 0, "no_map_claim": True,
              "memory": {"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()},
              "artifacts": [_artifact_record(output, name) for name in artifacts], "finished_utc": datetime.now(timezone.utc).isoformat()}
    _json(output / "diagnostic_result.json", result)
    return result


def _self_check(output: Path) -> dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import torch.nn.functional as F
    from check_fusion_photo_ce_cpu import photo_batch, tiny_model, set_noncontiguous_classes
    from spica.coupled_predictive_losses import coupled_region_loss
    model = tiny_model(42)
    set_noncontiguous_classes(model)
    args = photo_batch()
    batch = dict(zip(("clean", "corrupted", "photos", "positive_indices", "negative_indices", "labels", "photo_labels", "photo_ids"), args, strict=True))
    before = _module_hash(model)
    control = tiny_model(42)
    set_noncontiguous_classes(control)
    rng = _global_rng_state()
    losses = _losses(model, batch)
    old = coupled_region_loss(control, *args, lambda_sig=0.0, sigreg=None, main_photo_objective=OBJECTIVE)["total"]
    assert torch.equal(losses["base"], old)
    positive = args[6][:, None] == model.classids[None, :]
    logits = F.normalize(model.encode_photo(args[2]), dim=-1) @ F.normalize(model.text_bank(), dim=-1).T / TAU
    expected = -(F.log_softmax(logits, dim=-1) * positive).sum(dim=-1)
    expected = (expected / positive.sum(dim=-1)).mean()
    torch.testing.assert_close(losses["photo_ce"], expected, rtol=0, atol=0)
    scopes = _prompt_scopes(model)
    base, unused_base = _gradients(losses["base"], scopes, retain=True)
    photo, unused_photo = _gradients(losses["photo_ce"], scopes, retain=False)
    control_scopes = _prompt_scopes(control)
    old_grad, old_unused = _gradients(old, control_scopes, retain=False)
    assert not any(unused_base.values()) and not any(unused_photo.values()) and not any(old_unused.values())
    assert all(torch.equal(base[scope], old_grad[scope]) for scope in SCOPES)
    assert all(_norm(photo[scope], positive=True) > 0 for scope in SCOPES)
    sample = {"gradients": {"base": base, "photo_ce": photo}, "batch_index": 0, "losses": {}}
    measured = _rows([sample], _select([sample] * BATCHES))  # Real 2D prompt shapes, not just flat toy vectors.
    assert all(math.isfinite(row["cosine_base_photo_ce_float64"]) for row in measured[0]["scopes"].values())
    assert _module_hash(model) == before and _global_rng_state() == rng and all(parameter.grad is None for parameter in model.parameters())
    toy = []
    for i in range(4):
        toy.append({"gradients": {"base": {SCOPES[0]: torch.tensor([2. + i, 0.]), SCOPES[1]: torch.tensor([1. + i, 0.])}, "photo_ce": {SCOPES[0]: torch.tensor([1., 0.]), SCOPES[1]: torch.tensor([1., 0.])}}, "batch_index": i, "losses": {}})
    selected = _select(toy)
    assert selected["binding_scope"] == SCOPES[1] and abs(selected["lambda_photo_ce"] - 0.25) < 1e-12
    try:
        _norm(torch.zeros(2), positive=True)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("zero denominator accepted")
    result = {"status": "PASS", "verified": True, "device": "cpu", "optimizer_updates": 0,
              "checks": ["production photo-CE oracle", "F2_MP base scalar/gradient parity", "two prompt scopes, 2D cosine and four-batch selection", "zero rejection/state unchanged"],
              "source_sha256": _file_sha256(Path(__file__))}
    _json(output / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    output = None
    try:
        output = _fresh_output(args.output_dir)
        result = _self_check(output) if args.self_check else _run(output)
        print(json.dumps({"status": result["status"], "output": str(output), "lambda_photo_ce": result.get("selection", {}).get("lambda_photo_ce")}, sort_keys=True))
        return 0
    except Exception as error:
        if output is not None:
            _json(output / ("receipt.json" if args.self_check else "diagnostic_result.json"), {"status": "FAIL", "verified": False, "error": str(error), "traceback": traceback.format_exc()})
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
