#!/usr/bin/env python3
"""Independent CPU contract gate for the TU-only sketch-reference loss.

This checker is deliberately synthetic: it uses the tiny model and mocked data
helpers, never official data, pretrained weights, network, GPU, or W&B.  The
archived F2_MP loss is used only as a local historical algebra oracle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

os.environ.update(
    CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", WANDB_MODE="disabled",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

from check_coupled_benchmark_cpu import (  # noqa: E402
    _same,
    active,
    images,
    safe_numpy_globals,
    tiny_model,
)
from spica.coupled_predictive_losses import (  # noqa: E402
    _sketch_reference_ce,
    _validate_sketch_ref_coefficient,
    coupled_region_loss,
)
import spica.train_coupled_benchmark as trainer  # noqa: E402
from spica.models.clip import FrozenClipBundle, FrozenClipEncoder  # noqa: E402

OBJECTIVE = "multi_positive_supervised_contrastive"
LAMBDA = 0.37  # synthetic algebra fixture only; no production coefficient selected.
ARCHIVED_LOSS = ROOT / "outputs/fusion_mp_execution_20260909T115500Z/runs/F2_MP/source_snapshot/files/src/spica/coupled_predictive_losses.py"


def batch() -> tuple[Tensor, ...]:
    return (
        images(2, 101), images(2, 102), images(6, 103),
        torch.tensor([0, 1]), torch.tensor([[2], [3]]),
        torch.tensor([0, 1]), torch.tensor([0, 0, 0, 1, 1, 1]),
        ["p0", "p1", "p2", "p3", "p4", "p5"],
    )


def run(
    model: nn.Module, coefficient: float | None,
) -> tuple[dict[str, Tensor], dict[str, int], dict[str, Any]]:
    """Run one loss while recording every relevant call and teacher input."""
    counts = {"student": 0, "photo": 0, "teacher": 0, "text": 0}
    captured: dict[str, Any] = {"teacher_inputs": [], "teacher_outputs": []}
    original_forward = model.forward
    original_photo = model.encode_photo
    original_reference = model.photo_reference
    original_text = model.text_bank.forward

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["student"] += 1
        value = original_forward(*args, **kwargs)
        captured["clean_q"] = value.q[: args[0].shape[0] // 2]
        return value

    def photo(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["photo"] += 1
        value = original_photo(*args, **kwargs)
        captured["photos"] = value
        return value

    def reference(self: Any, values: Tensor, *args: Any, **kwargs: Any) -> Any:
        counts["teacher"] += 1
        captured["teacher_inputs"].append(values.detach().clone())
        value = original_reference(values, *args, **kwargs)
        captured["teacher_outputs"].append(value.detach().clone())
        return value

    def text(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["text"] += 1
        value = original_text(*args, **kwargs)
        captured["text"] = value
        return value

    model.forward = types.MethodType(forward, model)  # type: ignore[method-assign]
    model.encode_photo = types.MethodType(photo, model)  # type: ignore[method-assign]
    model.photo_reference = types.MethodType(reference, model)  # type: ignore[method-assign]
    model.text_bank.forward = types.MethodType(text, model.text_bank)  # type: ignore[method-assign]
    try:
        result = coupled_region_loss(
            model, *batch(), lambda_sig=0.0, sigreg=None,
            main_photo_objective=OBJECTIVE, lambda_sketch_ref=coefficient,
        )
    finally:
        model.forward = original_forward  # type: ignore[method-assign]
        model.encode_photo = original_photo  # type: ignore[method-assign]
        model.photo_reference = original_reference  # type: ignore[method-assign]
        model.text_bank.forward = original_text  # type: ignore[method-assign]
    return result, counts, captured


def gradient_map(loss: Tensor, model: nn.Module, *, strict: bool = False) -> dict[str, Tensor | None]:
    values = torch.autograd.grad(
        loss, tuple(active(model).values()), allow_unused=True, retain_graph=True,
    )
    result: dict[str, Tensor | None] = dict(zip(active(model), values))
    if strict:
        assert len(result) == 47, len(result)
        assert all(value is not None for value in result.values())
        assert all(torch.isfinite(value).all() for value in result.values() if value is not None)
    return result


def assert_exact_maps(left: dict[str, Tensor | None], right: dict[str, Tensor | None]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert (left[name] is None) == (right[name] is None), name
        if left[name] is not None:
            assert torch.equal(left[name], right[name]), name


def archived_loss_module() -> Any:
    if not ARCHIVED_LOSS.is_file():
        raise FileNotFoundError(ARCHIVED_LOSS)
    spec = importlib.util.spec_from_file_location("spica._archived_f2_mp_sketch_ref_check", ARCHIVED_LOSS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "spica"
    spec.loader.exec_module(module)
    return module


def check_archived_loss_counts_and_keys() -> dict[str, Any]:
    """Compare the repaired None/zero paths with the actual archived F2_MP graph."""
    archived = archived_loss_module()
    args = batch()
    current_model = tiny_model(42)
    historical_model = tiny_model(42)
    current = coupled_region_loss(
        current_model, *args, lambda_sig=0.0, sigreg=None,
        main_photo_objective=OBJECTIVE, lambda_sketch_ref=None,
    )
    historical = archived.coupled_region_loss(
        historical_model, *args, lambda_sig=0.0, sigreg=None,
        main_photo_objective=OBJECTIVE,
    )
    assert current.keys() == historical.keys()
    assert len(active(current_model)) == 47
    for name in current:
        assert torch.equal(current[name], historical[name]), name
    current_grads = gradient_map(current["total"], current_model, strict=True)
    historical_grads = gradient_map(historical["total"], historical_model, strict=True)
    assert_exact_maps(current_grads, historical_grads)

    none_model = tiny_model(43)
    zero_model = tiny_model(43)
    none, none_counts, none_capture = run(none_model, None)
    zero, zero_counts, zero_capture = run(zero_model, 0.0)
    assert none.keys() == zero.keys() - {"sketch_ref"}
    for name in none:
        assert torch.equal(none[name], zero[name]), name
    assert torch.equal(none["total"], zero["total"])
    assert none_counts == {"student": 1, "photo": 1, "teacher": 1, "text": 1}
    assert zero_counts == {"student": 1, "photo": 1, "teacher": 2, "text": 1}
    assert len(none_capture["teacher_inputs"]) == 1
    assert len(zero_capture["teacher_inputs"]) == 2
    torch.testing.assert_close(none_capture["teacher_inputs"][0], args[2], rtol=0, atol=0)
    torch.testing.assert_close(zero_capture["teacher_inputs"][0], args[2], rtol=0, atol=0)
    torch.testing.assert_close(zero_capture["teacher_inputs"][1], args[1], rtol=0, atol=0)
    assert torch.equal(zero["sketch_ref"], zero["sketch_ref"])
    return {
        "archived_f2_mp_loss_keys_exact": True,
        "archived_f2_mp_active_gradients": len(current_grads),
        "archived_f2_mp_gradient_exact": True,
        "none_zero_common_keys_and_total_exact": True,
        "none_counts": none_counts,
        "explicit_zero_counts": zero_counts,
        "teacher_inputs_exact": True,
    }


def independent_ce_oracle(teacher: Tensor, student: Tensor) -> Tensor:
    """The independent log-sum-exp form; intentionally not F.cross_entropy."""
    logits = F.normalize(teacher.detach(), dim=-1) @ F.normalize(student, dim=-1).T / 0.07
    labels = torch.arange(logits.shape[0], device=logits.device)
    return -(logits[torch.arange(logits.shape[0]), labels] - torch.logsumexp(logits, dim=1)).mean()


def check_formula_and_weighted_gradients() -> dict[str, Any]:
    model = tiny_model(44)
    result, counts, captured = run(model, LAMBDA)
    assert counts == {"student": 1, "photo": 1, "teacher": 2, "text": 1}
    assert result["sketch_ref"].grad_fn is not None
    teacher = captured["teacher_outputs"][1]
    clean_q = captured["clean_q"]
    expected = independent_ce_oracle(teacher, clean_q)
    torch.testing.assert_close(result["sketch_ref"], expected, rtol=0, atol=1e-7)
    assert not teacher.requires_grad and teacher.grad_fn is None

    base_model = tiny_model(45)
    ref_model = tiny_model(45)
    full_model = tiny_model(45)
    base = run(base_model, None)[0]["total"]
    ref_result = run(ref_model, LAMBDA)[0]
    full = run(full_model, LAMBDA)[0]["total"]
    # The ref-only graph is evaluated separately so this is a real gradient oracle.
    ref_only = ref_result["sketch_ref"]
    base_grad = gradient_map(base, base_model, strict=True)
    ref_grad = gradient_map(ref_only, ref_model)
    full_grad = gradient_map(full, full_model, strict=True)
    assert set(ref_grad) == set(base_grad)
    for name in full_grad:
        assert full_grad[name] is not None and base_grad[name] is not None
        if ref_grad[name] is None:
            expected_grad = base_grad[name]
        else:
            expected_grad = base_grad[name] + LAMBDA * ref_grad[name]
        torch.testing.assert_close(full_grad[name], expected_grad, rtol=2e-6, atol=2e-6, msg=name)

    # Backward on the raw reference term must touch only the student visual and pooled head.
    raw_model = tiny_model(46)
    raw_result = run(raw_model, LAMBDA)[0]
    raw_result["sketch_ref"].backward()
    for name, parameter in raw_model.named_parameters():
        if name in active(raw_model) and name.startswith(("student_visual.", "pooled_head.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        elif name.startswith(("predictor.", "photo_model.photo_prompt", "text_bank.context", "original_clip.")) or not parameter.requires_grad:
            assert parameter.grad is None, name
    return {
        "oracle": "normalized_teacher_student_logsumexp_instance_ce",
        "weighted_total_gradient_linear_combination": True,
        "raw_reference_student_pooled_only": True,
        "raw_reference_predictor_prompts_teacher_none": True,
        "explicit_counts": counts,
    }


def check_detached_helper() -> dict[str, Any]:
    teacher = torch.randn(2, 6, requires_grad=True)
    student = torch.randn(2, 6, requires_grad=True)
    loss = _sketch_reference_ce(teacher, student)
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    return {"raw_helper_teacher_detached": True, "raw_helper_student_gradient": True}


def check_validation_and_incompatibilities() -> dict[str, Any]:
    invalid = (True, "1", torch.tensor(1.0), float("nan"), float("inf"), -1,
               -float("inf"), 10**400, [1], None)
    for value in invalid:
        try:
            _validate_sketch_ref_coefficient(value)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            raise AssertionError(f"invalid coefficient accepted: {value!r}")
    for value in (0, 0.25, 1.0, 10**200):
        _validate_sketch_ref_coefficient(value)

    args = batch()
    checks: list[tuple[str, dict[str, Any]]] = [
        ("photo_ce", {"lambda_photo_ce": 0.1}),
        ("sig", {"lambda_sig": 1.0, "sigreg": lambda value: value.mean()}),
        ("architecture", {"architecture": "predictive"}),
        ("qmp", {"main_photo_objective": "multi_positive_pooled_contrastive"}),
    ]
    for name, overrides in checks:
        model = tiny_model(50)
        architecture = overrides.pop("architecture", None)
        if architecture is not None:
            model.architecture = architecture
        kwargs: dict[str, Any] = dict(
            lambda_sig=0.0, sigreg=None, main_photo_objective=OBJECTIVE,
            lambda_sketch_ref=LAMBDA,
        )
        kwargs.update(overrides)
        try:
            coupled_region_loss(model, *args, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"incompatible {name} route was accepted")
    return {
        "invalid_finite_overflow_values_rejected": True,
        "low_level_photo_ce_sig_architecture_qmp_rejected": True,
    }


def check_routing_and_metadata() -> dict[str, Any]:
    assert trainer._validate_routing("tuberlin_220_30")["arm"] == trainer.ARM
    route = trainer._validate_routing("tuberlin_220_30", lambda_sketch_ref=LAMBDA)
    assert route["arm"] == trainer.SKETCH_REF_ARM and route["method_version"] == trainer.SKETCH_REF_METHOD
    assert trainer._validate_routing("tuberlin_220_30", lambda_sketch_ref=0)["arm"] == trainer.SKETCH_REF_ARM
    for kwargs in (
        {"dataset": "quickdraw_80_30", "lambda_sketch_ref": LAMBDA},
        {"dataset": "tuberlin_220_30", "lambda_sketch_ref": True},
        {"dataset": "tuberlin_220_30", "lambda_sketch_ref": -1.0},
        {"dataset": "tuberlin_220_30", "lambda_sketch_ref": 10**400},
    ):
        try:
            trainer._validate_routing(**kwargs)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            raise AssertionError(f"invalid route accepted: {kwargs}")
    common = {
        "dataset": "quickdraw_80_30", "output_dir": "unused", "campaign_root": "unused",
        "device": "cuda", "wandb_mode": "disabled", "smoke": True,
        "lambda_sketch_ref": LAMBDA,
    }
    with patch.object(trainer, "_device", side_effect=AssertionError("device reached")) as device:
        try:
            trainer._train_impl(argparse.Namespace(**common), ROOT / "unused-sref-output")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid TU-only route reached data/device")
        device.assert_not_called()
    parsed_old = trainer._parser().parse_args(["--dataset", "tuberlin_220_30", "--output-dir", "x", "--campaign-root", "x"])
    parsed_new = trainer._parser().parse_args(["--dataset", "tuberlin_220_30", "--output-dir", "x", "--campaign-root", "x", "--lambda-sketch-ref", "0.25"])
    assert not hasattr(parsed_old, "lambda_sketch_ref") and parsed_new.lambda_sketch_ref == 0.25

    model = tiny_model(51)
    protocol = SimpleNamespace(
        config_path="synthetic.yaml", config_sha256="synthetic-config",
        train=SimpleNamespace(class_names={0: "cat", 1: "dog"}, class_ids=(0, 1)),
        identity={"provenance": "SYNTHETIC_UNVERIFIED"},
    )
    config_args = argparse.Namespace(dataset="tuberlin_220_30", device="cpu", wandb_mode="disabled", smoke=True, lambda_sketch_ref=LAMBDA)
    config = trainer._config(args=config_args, protocol=protocol, clip_identity={}, initialization={}, source_hash="synthetic-source-UNVERIFIED", model=model)
    wandb = trainer._wandb_config(config)
    assert config["arm"] == trainer.SKETCH_REF_ARM and config["method_version"] == trainer.SKETCH_REF_METHOD
    assert config["sketch_ref_lambda"] == LAMBDA and config["lambda_sketch_ref"] == LAMBDA
    identity = config["sketch_ref_identity"]
    assert identity["target"] == "masked_corrupted_sketch_original_clip_teacher"
    assert identity["student"] == "clean_q"
    assert identity["views"] == "corrupted_teacher_to_clean_student"
    assert identity["direction"] == "teacher_rows_student_columns"
    assert identity["temperature"] == 0.07
    assert wandb["lambda_sketch_ref"] == LAMBDA
    assert wandb["arm"] == trainer.SKETCH_REF_ARM
    return {
        "invalid_values_and_routes_rejected_predevice": True,
        "old_namespace_shape_unchanged": True,
        "new_arm_config_and_wandb_lambda_identity": True,
    }


class _Loader:
    def __init__(self, value: tuple[Tensor, ...]) -> None:
        self.batch = value
        self.generator = torch.Generator().manual_seed(42)

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        while True:
            yield self.batch


def synthetic_protocol() -> Any:
    return SimpleNamespace(
        config_path="synthetic.yaml", config_sha256="synthetic-config",
        train=SimpleNamespace(class_names={0: "cat", 1: "dog"}, class_ids=(0, 1)),
        identity={"provenance": "SYNTHETIC_UNVERIFIED", "photos": 6},
    )


def restore_checkpoint(path: Path) -> dict[str, Any]:
    """Load only with weights_only=True, then restore actual model/optimizer/scheduler."""
    with safe_numpy_globals():
        payload = torch.load(path, map_location="cpu", weights_only=True)
    restored = tiny_model(42)
    full_state = dict(restored.state_dict())
    full_state.update(payload["model_state_dict"])
    result = restored.load_state_dict(full_state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    optimizer = trainer._make_optimizer(restored)
    scheduler = LambdaLR(optimizer, lr_lambda=trainer._schedule(1189, 59))
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    assert _same(payload["optimizer_state_dict"], optimizer.state_dict())
    assert _same(payload["scheduler_state_dict"], scheduler.state_dict())
    return {"weights_only": True, "strict_model_restore": True, "optimizer_restore_exact": True, "scheduler_restore_exact": True}


def check_mocked_train_impl(output: Path) -> dict[str, Any]:
    """Run the real trainer loop twice on synthetic mocked input, without fixed OUT writes."""
    mock_output = output / "mock_train"
    mock_output.mkdir()
    base = batch()
    train_batch = (
        base[0].repeat(16, 1, 1, 1), base[1].repeat(16, 1, 1, 1), base[2],
        base[3].repeat(16), base[4].repeat(16, 1), base[5].repeat(16), base[6], base[7],
    )
    loader = _Loader(train_batch)
    synthetic_clip = tiny_model(42).original_clip
    bundle = FrozenClipBundle(
        FrozenClipEncoder(synthetic_clip), lambda value: value,
        __import__("open_clip").get_tokenizer("ViT-B-32"), "tiny_cpu", None,
    )
    created: list[nn.Module] = []
    original_module = trainer.benchmark_data
    original_values = {
        "_verify_clip": trainer._verify_clip,
        "load_frozen_clip": trainer.load_frozen_clip,
        "capture_provenance": trainer.capture_provenance,
        "_verify_source": trainer._verify_source,
        "_copy_source_archive": trainer._copy_source_archive,
        "_device": trainer._device,
        "CoupledPredictiveModel": trainer.CoupledPredictiveModel,
    }

    def make_model(*args: Any, **kwargs: Any) -> nn.Module:
        model = tiny_model(42)
        created.append(model)
        return model

    def no_official(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("official benchmark loader was called")

    synthetic_module = SimpleNamespace(
        load_benchmark_protocol=lambda *args, **kwargs: synthetic_protocol(),
        make_train_loader=lambda *args, **kwargs: loader,
        prepare_batch=lambda protocol, raw, step: {
            "clean": raw[0], "corrupted": raw[1], "photos": raw[2],
            "positive_indices": raw[3], "negative_indices": raw[4], "labels": raw[5],
            "photo_labels": raw[6], "photo_ids": raw[7],
            "trace": [{"sample": i} for i in range(32)],
            "mask_metadata": {"rows": [{"sample": i, "masked": True} for i in range(32)]},
        },
    )
    import spica.data.coupled_benchmark as official
    old_official_loader = official.make_train_loader
    trainer.benchmark_data = synthetic_module
    trainer._verify_clip = lambda: {"path": "synthetic", "sha256": "synthetic"}
    trainer.load_frozen_clip = lambda **kwargs: bundle
    trainer.CoupledPredictiveModel = make_model
    trainer.capture_provenance = lambda *args, **kwargs: {
        "source_snapshot": {"manifest": []}, "resolved_config": kwargs.get("resolved_config", {}),
    }
    trainer._verify_source = lambda *args, **kwargs: "synthetic-source-UNVERIFIED"
    trainer._copy_source_archive = lambda *args, **kwargs: None
    trainer._device = lambda value: torch.device("cpu")
    try:
        official.make_train_loader = no_official
        with tempfile.TemporaryDirectory(prefix="spica-sketch-ref-no-official-"):
            args = SimpleNamespace(
                dataset="tuberlin_220_30", output_dir=str(mock_output), campaign_root=str(mock_output),
                device="cpu", wandb_mode="disabled", smoke=True, lambda_sketch_ref=LAMBDA,
            )
            run_result = trainer._train_impl(args, mock_output)
        assert run_result["status"] == "COMPLETE"
        assert run_result["arm"] == trainer.SKETCH_REF_ARM
        assert run_result["method_version"] == trainer.SKETCH_REF_METHOD
        assert run_result["step"] == 2 and run_result["horizon_steps"] == 1189
        assert run_result["warmup_steps"] == 59 and run_result["trace_count"] == 64 and run_result["mask_count"] == 64
        resolved = json.loads((mock_output / "resolved_config.json").read_text())
        assert resolved["arm"] == trainer.SKETCH_REF_ARM
        assert resolved["method_version"] == trainer.SKETCH_REF_METHOD
        assert resolved["lambda_sketch_ref"] == LAMBDA and resolved["sketch_ref_lambda"] == LAMBDA
        wandb_config = trainer._wandb_config(resolved)
        assert wandb_config["lambda_sketch_ref"] == LAMBDA
        assert wandb_config["arm"] == trainer.SKETCH_REF_ARM
        (mock_output / "wandb_config.json").write_text(json.dumps(wandb_config, indent=2, sort_keys=True) + "\n")
        trace_rows = [json.loads(line) for line in (mock_output / "observation_trace.jsonl").read_text().splitlines()]
        mask_rows = [json.loads(line) for line in (mock_output / "mask_metadata.jsonl").read_text().splitlines()]
        assert len(trace_rows) == len(mask_rows) == 64
        assert all(row["step"] in (1, 2) and "actual_lr" in row for row in trace_rows)
        assert all(row["step"] in (1, 2) for row in mask_rows)
        checkpoint = mock_output / "checkpoint_step2.pt"
        restore = restore_checkpoint(checkpoint)
        with safe_numpy_globals():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["step"] == 2
        assert payload["resolved_config"]["lambda_sketch_ref"] == LAMBDA
        return {
            "source_provenance": "SYNTHETIC_UNVERIFIED",
            "new_run_result_identity": True,
            "arm_method_lambda_config_identity": True,
            "wandb_lambda_identity": True,
            "horizon_steps": 1189,
            "warmup_steps": 59,
            "first_64_trace_rows": 64,
            "first_64_mask_rows": 64,
            "actual_two_updates": True,
            "checkpoint_safe_actual_restore": restore,
        }
    finally:
        official.make_train_loader = old_official_loader
        trainer.benchmark_data = original_module
        for name, value in original_values.items():
            setattr(trainer, name, value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    checks: list[dict[str, Any]] = []
    functions = (
        ("archived_loss_counts_and_keys", check_archived_loss_counts_and_keys),
        ("formula_weighted_gradients", check_formula_and_weighted_gradients),
        ("detached_helper", check_detached_helper),
        ("validation_and_incompatibilities", check_validation_and_incompatibilities),
        ("routing_and_metadata", check_routing_and_metadata),
        ("mocked_train_impl", lambda: check_mocked_train_impl(output)),
    )
    for name, function in functions:
        try:
            checks.append({"name": name, "status": "PASS", "details": function()})
        except Exception as error:  # noqa: BLE001
            import traceback
            # Keep every failure in the receipt; do not stop before the actual trainer check.
            checks.append({"name": name, "status": "FAIL", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
    failed = [row for row in checks if row["status"] != "PASS"]
    result = {
        "schema_version": 2,
        "status": "FAIL" if failed else "PASS",
        "verified": not failed,
        "gate": "coupled_sketch_ref_cpu",
        "device": "cpu",
        "pretrained_weights": False,
        "network": False,
        "campaign_training": False,
        "source_provenance": "SYNTHETIC_UNVERIFIED",
        "checks": checks,
        "summary": {"pass": len(checks) - len(failed), "fail": len(failed), "skip": 0},
        "component_sha256": {name: sha256(ROOT / name) for name in (
            "scripts/check_coupled_sketch_ref_cpu.py",
            "src/spica/coupled_predictive_losses.py",
            "src/spica/train_coupled_benchmark.py",
        )},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "failures": len(failed)}))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
