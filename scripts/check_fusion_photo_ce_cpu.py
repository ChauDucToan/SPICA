#!/usr/bin/env python3
"""CPU-only checker for the bounded predictive-fusion photo CE term."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import types
from typing import Any
from unittest.mock import patch

os.environ.update(
    CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", WANDB_MODE="disabled",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
)
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor  # noqa: E402

from check_fusion_qmp_cpu import (  # noqa: E402
    MP_LOSS,
    active,
    batch,
    images,
    tiny_model,
    _loss_parity,
)
from spica.coupled_predictive_losses import (  # noqa: E402
    _validate_photo_ce_coefficient,
    coupled_region_loss,
)
import spica.train_coupled_predictive as trainer  # noqa: E402

OBJECTIVE = "multi_positive_supervised_contrastive"
CAMPAIGN = "coupled_predictive_fusion_mp_photo_ce_v1"
SYNTHETIC_LAMBDA = 0.37  # CPU algebra fixture only; no production weight selected.


def photo_batch() -> tuple[Tensor, ...]:
    values = list(batch())
    # Three queries share a positive; all four unique bank rows are used.
    values[0], values[1], values[2] = images(3), images(3).flip(-1), images(4)
    values[3] = torch.tensor([0, 0, 2])
    values[4] = torch.tensor([[2], [3], [1]])
    values[5] = torch.tensor([2, 2, 7])
    values[6] = torch.tensor([2, 2, 7, 7])
    values[7] = ["p0", "p1", "p2", "p3"]
    return tuple(values)  # type: ignore[return-value]


def set_noncontiguous_classes(model: torch.nn.Module) -> None:
    model.text_bank.class_labels.copy_(torch.tensor([2, 7]))


def run(model: torch.nn.Module, coefficient: float | None) -> tuple[dict[str, Tensor], dict[str, int], dict[str, Tensor]]:
    counts = {"model": 0, "photo": 0, "reference": 0, "text": 0}
    captured: dict[str, Tensor] = {}

    def wrap(name: str, method: Any, capture: str | None = None) -> Any:
        def call(self: Any, *args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            result = method(*args, **kwargs)
            if capture:
                captured[capture] = result
            return result
        return types.MethodType(call, model if name != "text" else model.text_bank)

    model_forward = model.forward
    photo_encode = model.encode_photo
    photo_reference = model.photo_reference
    text_forward = model.text_bank.forward
    model.forward = wrap("model", model_forward)  # type: ignore[method-assign]
    model.encode_photo = wrap("photo", photo_encode, "photos")  # type: ignore[method-assign]
    model.photo_reference = wrap("reference", photo_reference)  # type: ignore[method-assign]
    def text_call(self: Any, *args: Any, **kwargs: Any) -> Tensor:
        counts["text"] += 1
        value = text_forward(*args, **kwargs)
        captured["text"] = value
        return value

    model.text_bank.forward = types.MethodType(text_call, model.text_bank)  # type: ignore[method-assign]
    try:
        result = coupled_region_loss(
            model, *photo_batch(), lambda_sig=0.0, sigreg=None,
            main_photo_objective=OBJECTIVE, lambda_photo_ce=coefficient,
        )
    finally:
        model.forward = model_forward  # type: ignore[method-assign]
        model.encode_photo = photo_encode  # type: ignore[method-assign]
        model.photo_reference = photo_reference  # type: ignore[method-assign]
        model.text_bank.forward = text_forward  # type: ignore[method-assign]
    return result, counts, captured


def gradients(loss: Tensor, model: torch.nn.Module) -> dict[str, Tensor]:
    values = torch.autograd.grad(loss, tuple(active(model).values()), allow_unused=True, retain_graph=True)
    return {
        name: torch.zeros_like(parameter) if value is None else value.detach().clone()
        for (name, parameter), value in zip(active(model).items(), values)
    }


def assert_maps_equal(left: dict[str, Tensor], right: dict[str, Tensor]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def oracle(bank: Tensor, text: Tensor, classids: Tensor, labels: Tensor) -> Tensor:
    positions = torch.searchsorted(classids, labels)
    logits = F.normalize(bank, dim=-1) @ F.normalize(text, dim=-1).T
    return F.cross_entropy(logits / 0.07, positions)


def check_legacy_archive() -> dict[str, Any]:
    _loss_parity(MP_LOSS, "spica.photo_ce_legacy_archive", OBJECTIVE)
    return {"archive": str(MP_LOSS), "scalar_exact": True, "gradient_exact": True}


def check_photo_ce() -> dict[str, Any]:
    legacy = tiny_model(42)
    set_noncontiguous_classes(legacy)
    base, base_counts, _ = run(legacy, None)
    zero_model = tiny_model(42)
    set_noncontiguous_classes(zero_model)
    zero, zero_counts, _ = run(zero_model, 0.0)
    assert "photo_ce" in zero and base.keys() | {"photo_ce"} == zero.keys()
    assert torch.equal(base["total"], zero["total"])
    assert all(torch.equal(base[key], zero[key]) for key in base)
    assert_maps_equal(gradients(base["total"], legacy), gradients(zero["total"], zero_model))

    model = tiny_model(42)
    set_noncontiguous_classes(model)
    result, counts, captured = run(model, SYNTHETIC_LAMBDA)
    assert counts == base_counts == zero_counts == {"model": 1, "photo": 1, "reference": 1, "text": 1}
    expected = oracle(captured["photos"], captured["text"], model.classids, photo_batch()[6])
    torch.testing.assert_close(result["photo_ce"], expected, rtol=0, atol=0)
    assert all(torch.equal(base[key], result[key]) for key in base if key != "total")
    torch.testing.assert_close(result["total"], base["total"] + SYNTHETIC_LAMBDA * result["photo_ce"], rtol=1e-6, atol=1e-7)
    base_grad, ce_grad = gradients(base["total"], legacy), gradients(expected, model)
    for name, value in gradients(result["total"], model).items():
        torch.testing.assert_close(value, base_grad[name] + SYNTHETIC_LAMBDA * ce_grad[name],
                                   rtol=2e-5, atol=2e-6)

    isolated = tiny_model(43)
    set_noncontiguous_classes(isolated)
    isolated_result, _, _ = run(isolated, SYNTHETIC_LAMBDA)
    isolated.zero_grad(set_to_none=True)
    isolated_result["photo_ce"].backward()
    allowed = {"photo_model.photo_prompt", "text_bank.context"}
    isolated_grads = {
        name: parameter.grad for name, parameter in active(isolated).items()
    }
    assert all(
        (value is not None and torch.isfinite(value).all() and float(value.abs().sum()) > 0)
        if name in allowed else value is None
        for name, value in isolated_grads.items()
    )

    full = tiny_model(44)
    set_noncontiguous_classes(full)
    full_result, _, _ = run(full, SYNTHETIC_LAMBDA)
    full_grads = gradients(full_result["total"], full)
    assert all(torch.isfinite(value).all() and float(value.abs().sum()) > 0 for value in full_grads.values())
    frozen = {name: value.detach().clone() for name, value in full.original_clip.state_dict().items()}
    optimizer = trainer._make_optimizer(full)
    optimizer.zero_grad(set_to_none=True)
    full_result["total"].backward()
    optimizer.step()
    assert all(torch.equal(value, full.original_clip.state_dict()[name]) for name, value in frozen.items())
    restored = tiny_model(44)
    missing, unexpected = restored.load_state_dict(full.state_dict(), strict=False)
    assert not unexpected and all(name.startswith("original_clip.") for name in missing)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in full.state_dict().items())
    restored_optimizer = trainer._make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert optimizer.state_dict()["param_groups"] == restored_optimizer.state_dict()["param_groups"]
    for index, state in optimizer.state_dict()["state"].items():
        for key, value in state.items():
            assert torch.equal(value, restored_optimizer.state_dict()["state"][index][key])
    return {
        "independent_oracle": True, "noncontiguous_labels": [2, 7],
        "shared_positive_and_negative_rows": True, "lambda_zero_total_and_gradient_exact": True,
        "weighted_gradient_oracle": True, "synthetic_lambda_not_production": SYNTHETIC_LAMBDA,
        "photo_ce_gradient_scope": sorted(allowed), "full_gradients_finite_nonzero": True,
        "hook_counts": counts, "bank_encoded_once": True, "adamw_step_reload_exact": True,
    }


def check_validation() -> dict[str, Any]:
    for value in (None, True, "1", torch.tensor(1.0), float("nan"), float("inf"), -1, [1], 10**400):
        try:
            _validate_photo_ce_coefficient(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid coefficient accepted: {value!r}")
    for value in (0, 0.25, 1):
        _validate_photo_ce_coefficient(value)
    incompatible = tiny_model(45)
    for kwargs in (
        {"lambda_sig": 1.0, "sigreg": object()},
        {"main_photo_objective": "multi_positive_pooled_contrastive"},
    ):
        try:
            call_kwargs = dict(lambda_sig=0.0, lambda_photo_ce=0.1, sigreg=None,
                               main_photo_objective=OBJECTIVE)
            call_kwargs.update(kwargs)
            coupled_region_loss(incompatible, *photo_batch(), **call_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("incompatible photo CE route accepted")
    return {"invalid_values_rejected": True, "finite_nonnegative_values_accepted": True,
            "incompatible_variants_rejected": True}


def check_route() -> dict[str, Any]:
    from spica.data.coupled_training import _arm_protocol
    from spica.evaluation.coupled_predictive import QOnlyAdapter

    expected = {"architecture": "predictive_fusion_v2", "method_version": CAMPAIGN,
                "positive_pool": "full", "main_photo_objective": OBJECTIVE}
    assert _arm_protocol("F2_MP_PCE", CAMPAIGN, None) == expected
    common = dict(arm="F2_MP_PCE", campaign_id=CAMPAIGN, diagnostic=None,
                  max_steps=2, smoke=True, device="cuda")
    invalid = [dict(lambda_photo_ce=value) for value in (None, True, -1., float("nan"), float("inf"), "1")]
    invalid += [{"campaign_id": "wrong", "lambda_photo_ce": 1.},
                {"diagnostic": "not-a-photo-ce-diagnostic", "lambda_photo_ce": 1.},
                {"arm": "F2_MP", "campaign_id": "coupled_predictive_fusion_mp_v1", "lambda_photo_ce": 0.}]
    with patch.object(trainer, "_device", side_effect=AssertionError("device reached")) as device:
        for override in [{}] + invalid:
            try:
                trainer._train_impl(argparse.Namespace(**(common | override)), ROOT / "unused-photo-ce-output")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid photo-CE route accepted")
        device.assert_not_called()
    for value in (0., SYNTHETIC_LAMBDA):
        with patch.object(trainer, "_device", side_effect=StopIteration("CPU boundary check")) as device:
            try:
                trainer._train_impl(argparse.Namespace(**(common | {"lambda_photo_ce": value})), ROOT / "unused-photo-ce-output")
            except StopIteration:
                device.assert_called_once_with("cuda")
            else:
                raise AssertionError("valid route did not reach the device boundary")
    parsed = trainer._parser().parse_args(["--arm", "F2_MP", "--output-dir", "unused", "--campaign-root", "unused"])
    assert not hasattr(parsed, "lambda_photo_ce")  # Historical config shape unchanged.
    assert trainer._evaluation_query("F2_MP_PCE") == "q"
    assert trainer._evaluation_query("F2_MP") is None
    model = tiny_model(42).eval()
    with torch.no_grad():
        expected_q = model(images(3)).q
        with patch.object(model.predictor, "forward", side_effect=AssertionError("predictor reached")):
            assert torch.equal(expected_q, QOnlyAdapter(model)(images(3)))
    return {"protocol": expected, "invalid_rejected_before_device": len(invalid) + 1,
            "no_production_default": True, "q_bypass_exact": True, "predictor_calls": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or Path("/tmp") / f"fusion_photo_ce_cpu_{stamp}").resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    checks = []
    for name, function in (("legacy_archive", check_legacy_archive),
                           ("photo_ce", check_photo_ce), ("validation", check_validation),
                           ("route", check_route)):
        try:
            checks.append({"name": name, "status": "PASS", "details": function()})
        except Exception as error:  # noqa: BLE001
            import traceback
            checks.append({"name": name, "status": "FAIL", "error": f"{type(error).__name__}: {error}",
                           "traceback": traceback.format_exc()})
            break
    failed = [row for row in checks if row["status"] != "PASS"]
    result = {"schema_version": 1, "status": "FAIL" if failed else "PASS",
              "verified": not failed, "gate": "fusion_photo_ce_cpu", "device": "cpu",
              "pretrained_weights": False, "network": False, "campaign_training": False,
              "checks": checks, "summary": {"pass": len(checks) - len(failed),
              "fail": len(failed), "skip": 0},
              "component_sha256": {name: trainer._sha256_file(ROOT / name) for name in (
                  "scripts/check_fusion_photo_ce_cpu.py", "scripts/check_fusion_qmp_cpu.py",
                  "src/spica/coupled_predictive_losses.py", "src/spica/data/coupled_training.py",
                  "src/spica/train_coupled_predictive.py", "src/spica/models/coupled_predictive.py",
                  "src/spica/evaluation/coupled_predictive.py")}}
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
