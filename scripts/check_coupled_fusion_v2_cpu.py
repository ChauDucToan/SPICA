#!/usr/bin/env python3
"""Standalone CPU gate for the coupled-predictive contextual fusion V2 path.

This gate uses a randomly initialised, tiny OpenCLIP model only.  It never
resolves pretrained weights, reads images, trains a campaign, or changes the
production sources.  Every check is recorded so a source defect is evidence,
not silently patched by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Callable
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import open_clip  # noqa: E402
import torch  # noqa: E402
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

from spica.coupled_predictive_losses import (  # noqa: E402
    _view_terms,
    coupled_region_loss,
    task_loss,
)
from spica.data.coupled_training import _arm_protocol  # noqa: E402
from spica.models.clip import FrozenClipBundle, FrozenClipEncoder  # noqa: E402
from spica.models.coupled_predictive import (  # noqa: E402
    CoupledPredictiveModel,
)
import spica.train_coupled_predictive as trainer  # noqa: E402

ARCHIVE_ROOT = ROOT / "outputs/coupled_predictive_execution_20260908T153000Z/source_snapshot/files"
ARCHIVE_MODEL = ARCHIVE_ROOT / "src/spica/models/coupled_predictive.py"
ARCHIVE_LOSS = ARCHIVE_ROOT / "src/spica/coupled_predictive_losses.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(repr((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def finite_tensor(value: Tensor, label: str) -> None:
    assert isinstance(value, Tensor), f"{label} is not a tensor"
    assert bool(torch.isfinite(value).all()), f"{label} is non-finite"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class Receipt:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.log_path = root / "gate.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        self.checks: list[dict[str, Any]] = []
        self.started = time.time()

    def line(self, message: str) -> None:
        print(message, file=self.log, flush=True)

    def check(
        self,
        name: str,
        function: Callable[[], Any],
        *,
        skip_if: Callable[[BaseException], bool] | None = None,
    ) -> Any:
        self.line(f"START {name}")
        started = time.time()
        try:
            result = function()
        except Exception as error:  # noqa: BLE001 - receipt must retain all failures
            status = "SKIP" if skip_if is not None and skip_if(error) else "FAIL"
            row = {
                "name": name,
                "status": status,
                "seconds": round(time.time() - started, 6),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            self.checks.append(row)
            self.line(f"{status} {name}: {row['error']}")
            return None
        row = {"name": name, "status": "PASS", "seconds": round(time.time() - started, 6)}
        if isinstance(result, dict):
            row["details"] = result
        self.checks.append(row)
        self.line(f"PASS {name}")
        return result

    def close(self, *, source_hash: dict[str, Any]) -> dict[str, Any]:
        self.log.close()
        failed = [row for row in self.checks if row["status"] == "FAIL"]
        skipped = [row for row in self.checks if row["status"] == "SKIP"]
        receipt = {
            "schema_version": 1,
            "status": "FAIL" if failed else ("INCOMPLETE" if skipped else "PASS"),
            "verified": not failed and not skipped,
            "gate": "coupled_predictive_fusion_v2_cpu",
            "device": "cpu",
            "pretrained_weights": False,
            "campaign_training": False,
            "started_unix": self.started,
            "finished_unix": time.time(),
            "source_hash": source_hash,
            "checks": self.checks,
            "summary": {"pass": len(self.checks) - len(failed) - len(skipped), "fail": len(failed), "skip": len(skipped)},
            "log": str(self.log_path),
        }
        write_json(self.root / "receipt.json", receipt)
        return receipt


def tiny_bundle(seed: int = 42) -> FrozenClipBundle:
    """Build the exact no-download tiny OpenCLIP fixture used by this gate."""
    torch.manual_seed(seed)
    model = CLIP(
        embed_dim=6,
        vision_cfg=CLIPVisionCfg(layers=1, width=8, head_width=4, patch_size=4, image_size=8),
        text_cfg=CLIPTextCfg(context_length=77, vocab_size=49408, width=8, heads=2, layers=1),
    ).float().cpu()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return FrozenClipBundle(
        encoder=FrozenClipEncoder(model),
        transform=lambda value: value,
        tokenizer=tokenizer,
        model_name="tiny_cpu_fixture",
        pretrained=None,
    )


def make_model(architecture: str = "predictive_fusion_v2", seed: int = 42) -> CoupledPredictiveModel:
    bundle = tiny_bundle(seed)
    model = CoupledPredictiveModel(
        bundle.encoder,
        bundle.tokenizer,
        {0: "cat", 1: "dog"},
        architecture=architecture,
        photo_prompt_length=3,
        text_prompt_length=4,
        predictor_width=8,
        predictor_heads=2,
    ).cpu()
    return model


def active_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def original_state(model: CoupledPredictiveModel) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.original_clip.state_dict().items()}


def assert_same_state(before: dict[str, Tensor], after: dict[str, Tensor]) -> None:
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


def images(batch: int = 2) -> Tensor:
    torch.manual_seed(123)
    return torch.randn(batch, 3, 8, 8)


def check_routing() -> dict[str, Any]:
    # Exercise the trainer entry boundary with a device function that must not
    # be reached when F2 campaign/diagnostic arguments are invalid.
    original_device = trainer._device
    calls = 0
    def forbidden_device(_value: str) -> torch.device:
        nonlocal calls
        calls += 1
        raise AssertionError("invalid F2 routing reached device selection")
    trainer._device = forbidden_device
    try:
        invalid = (("wrong", None), ("coupled_predictive_fusion_v2", object()))
        for campaign_id, diagnostic in invalid:
            namespace = argparse.Namespace(
                arm="F2", campaign_id=campaign_id, diagnostic=diagnostic,
                max_steps=2, smoke=True, device="cuda",
            )
            try:
                trainer._train_impl(namespace, ROOT / "this-output-is-never-created")
            except (TypeError, ValueError):
                continue
            raise AssertionError(f"invalid F2 CLI routing was accepted: {campaign_id!r}")
    finally:
        trainer._device = original_device
    assert calls == 0
    resolved = _arm_protocol("F2", "coupled_predictive_fusion_v2", None)
    assert resolved == {"architecture": "predictive_fusion_v2", "method_version": "coupled_predictive_fusion_v2", "positive_pool": "full"}
    assert _arm_protocol("F2_SIG", "coupled_predictive_fusion_v2", "new-receipt.json") == resolved
    try:
        _arm_protocol("F2_SIG", "coupled_predictive_fusion_v2", None)
    except ValueError:
        pass
    else:
        raise AssertionError("F2_SIG accepted a missing diagnostic")
    # A V1 PASS label alone must never enable the F2 regularizer scale.
    old_receipt = {"status": "PASS", "verified": True, "formula_identity": "old",
                   "architecture_identity": "predictive", "gradient_batch_source": "old", "lambda_sig": .0078}
    with patch.object(trainer, "_diagnostic_payload", return_value=old_receipt):
        try:
            trainer._diagnostic_lambda(Path("unused"), class_ids=[], initialization={}, source_hash="",
                                       config={"architecture": "predictive_fusion_v2"})
        except ValueError:
            pass
        else:
            raise AssertionError("F2_SIG accepted the old V1 diagnostic")
    from spica.evaluation.coupled_predictive import CoupledPredictiveAdapter
    adapter = CoupledPredictiveAdapter(object(), query="mu_i")
    assert adapter.query == "mu_i"
    return {"resolved": resolved, "misuse_cases": len(invalid), "device_calls_for_invalid": calls,
            "eval_adapter_query": adapter.query}


def check_forward_backward(model: CoupledPredictiveModel) -> dict[str, Any]:
    before = original_state(model)
    output = model(images())
    for name in ("g", "q", "mu_i", "mu_t"):
        value = getattr(output, name)
        assert value is not None
        finite_tensor(value, name)
    model.zero_grad(set_to_none=True)
    (output.q.square().mean() + output.mu_i.square().mean() + output.mu_t.square().mean()).backward()
    active = active_parameters(model)
    missing = [name for name, parameter in active.items() if parameter.grad is None]
    assert not missing, f"active parameters without gradient: {missing}"
    for name, parameter in active.items():
        finite_tensor(parameter.grad, f"gradient/{name}")
    frozen = [name for name, parameter in model.original_clip.named_parameters() if parameter.requires_grad]
    assert not frozen, f"original CLIP trainability leaked: {frozen}"
    assert all(parameter.grad is None for parameter in model.original_clip.parameters())
    assert_same_state(before, original_state(model))
    student = list(model.student_visual.parameters())
    original = list(model.original_clip.parameters())
    assert student and original
    assert not any(a.data_ptr() == b.data_ptr() for a in student for b in original)
    return {"active_parameters": len(active), "active_gradients": len(active), "student_parameters": len(student)}


def check_hooks_and_context(model: CoupledPredictiveModel) -> dict[str, Any]:
    events: list[str] = []
    values: list[Tensor] = []

    def mark(name: str) -> Callable[..., None]:
        def hook(*_args: Any) -> None:
            events.append(name)
        return hook

    def capture_self_attention(_module: nn.Module, inputs: Any, output: Any) -> None:
        events.append("fusion_self_attn")
        values.append(inputs[0].detach().clone())

    handles = [
        model.predictor.attention.register_forward_hook(mark("CA")),
        model.predictor.fusion_attention.register_forward_hook(capture_self_attention),
        model.predictor.ffn.register_forward_hook(mark("FFN")),
    ]
    try:
        model(images())
    finally:
        for handle in handles:
            handle.remove()
    assert events == ["CA", "fusion_self_attn", "FFN"], events
    assert len(values) == 1 and values[0].shape[0] == 2
    assert not torch.allclose(values[0][0], values[0][1], atol=1e-7, rtol=1e-6)
    return {"order": events, "self_attention_batch_difference": float((values[0][0] - values[0][1]).abs().max())}


def check_no_leakage(model: CoupledPredictiveModel) -> dict[str, Any]:
    batch = images(3)
    batched = model(batch)
    maxima: dict[str, float] = {}
    for index in range(3):
        single = model(batch[index : index + 1])
        for name in ("g", "q", "mu_i", "mu_t"):
            left = getattr(batched, name)[index]
            right = getattr(single, name)[0]
            delta = float((left - right).abs().max().detach())
            maxima[name] = max(maxima.get(name, 0.0), delta)
            assert torch.allclose(left, right, atol=2e-5, rtol=2e-5), f"batch leakage in {name}: {delta}"
    return {"max_abs_delta": maxima, "tolerance": 2e-5}


def check_prompt_gradients(model: CoupledPredictiveModel) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    model(images()).mu_i.sum().backward()
    text_grad = model.text_bank.context.grad
    assert text_grad is not None and float(text_grad.abs().sum()) > 0
    model.zero_grad(set_to_none=True)
    model(images()).mu_t.sum().backward()
    photo_grad = model.photo_model.photo_prompt.grad
    assert photo_grad is not None and float(photo_grad.abs().sum()) > 0
    legacy = make_model("predictive")
    old = legacy(images())
    old_text = torch.autograd.grad(old.mu_i.sum(), legacy.text_bank.context, retain_graph=True)[0]
    old_photo = torch.autograd.grad(old.mu_t.sum(), legacy.photo_model.photo_prompt)[0]
    assert torch.count_nonzero(old_text) == torch.count_nonzero(old_photo) == 0
    return {"mu_i_to_text_context": float(text_grad.abs().sum()), "mu_t_to_photo_prompt": float(photo_grad.abs().sum()),
            "V1_cross_group_forward_gradients": 0.0}


def check_prompt_ownership(model: CoupledPredictiveModel) -> dict[str, Any]:
    names = dict(model.named_parameters())
    assert "photo_model.photo_prompt" in names
    assert "text_bank.context" in names
    assert names["photo_model.photo_prompt"] is model.photo_model.photo_prompt
    assert names["text_bank.context"] is model.text_bank.context
    assert not any(name.startswith("original_clip.") and p.requires_grad for name, p in names.items())
    pointers = [p.data_ptr() for p in names.values()]
    assert len(pointers) == len(set(pointers)), "a parameter has multiple registered owners"
    return {"prompt_parameter_names": ["photo_model.photo_prompt", "text_bank.context"], "registered_parameters": len(names)}


def check_view_terms_and_loss() -> dict[str, Any]:
    output = type("Output", (), {
        "q": torch.randn(2, 6, requires_grad=True), "mu_i": torch.randn(2, 6, requires_grad=True),
        "mu_t": torch.randn(2, 6, requires_grad=True),
    })()
    positive = torch.randn(2, 6)
    negative = torch.randn(2, 2, 6)
    text = torch.randn(2, 6)
    classids = torch.tensor([0, 1])
    labels = torch.tensor([0, 1])
    terms = _view_terms(output, positive, negative, text, classids, labels, "predictive_fusion_v2")
    assert set(terms) == {"rank_pool", "ce_pool", "rank_i", "ce_t", "align_i", "align_t", "ce_i"}
    direct = torch.autograd.grad(terms["ce_i"], (output.mu_i, output.mu_t, output.q), allow_unused=True)
    assert direct[0].abs().sum() > 0 and direct[1] is None and direct[2] is None
    values = {
        "clean_rank_i": torch.tensor(1.0), "clean_ce_i": torch.tensor(2.0),
        "masked_rank_i": torch.tensor(3.0), "masked_ce_i": torch.tensor(4.0),
        "clean_rank_pool": torch.tensor(100.0), "clean_ce_pool": torch.tensor(100.0),
        "masked_rank_pool": torch.tensor(100.0), "masked_ce_pool": torch.tensor(100.0),
        "clean_ce_t": torch.tensor(100.0), "masked_ce_t": torch.tensor(100.0),
        "clean_align_i": torch.tensor(100.0), "clean_align_t": torch.tensor(100.0),
        "masked_align_i": torch.tensor(100.0), "masked_align_t": torch.tensor(100.0),
    }
    assert torch.equal(task_loss(values, "predictive_fusion_v2"), torch.tensor(5.0))

    # Keep this coefficient oracle independent of the production graph: each
    # leaf is a differentiable scalar, so its derivative is the exact weight.
    leaves = {name: torch.ones((), requires_grad=True) for name in (
        "clean_rank_i", "clean_ce_i", "masked_rank_i", "masked_ce_i",
        "clean_ce_t", "masked_ce_t", "clean_rank_pool", "clean_ce_pool",
        "masked_rank_pool", "masked_ce_pool", "clean_align_i", "clean_align_t",
        "masked_align_i", "masked_align_t", "anchor_i", "anchor_t",
    )}
    view_leaves = [{key.removeprefix(view + "_"): value for key, value in leaves.items()
                    if key.startswith(view + "_")} for view in ("clean", "masked")]
    with patch("spica.coupled_predictive_losses._view_terms", side_effect=view_leaves), \
         patch("spica.coupled_predictive_losses._align", side_effect=[leaves["anchor_i"], leaves["anchor_t"]]):
        # Differentiate the PRODUCTION total, not a copied coefficient formula.
        mock_total = real_loss(make_model())["total"]
    assert torch.equal(mock_total.detach(), torch.tensor(3.85))
    mock_total.backward()
    expected_weights = {
        "clean_rank_i": .5, "clean_ce_i": .5, "masked_rank_i": .5,
        "masked_ce_i": .5, "clean_ce_t": .125, "masked_ce_t": .125,
        "clean_rank_pool": .125, "clean_ce_pool": .125,
        "masked_rank_pool": .125, "masked_ce_pool": .125,
        "clean_align_i": .025, "clean_align_t": .025,
        "masked_align_i": .025, "masked_align_t": .025,
        "anchor_i": .5, "anchor_t": .5,
    }
    assert all(torch.equal(leaves[name].grad, torch.tensor(weight))
               for name, weight in expected_weights.items())
    return {"view_term_keys": sorted(terms), "main_only_task_loss": 5.0,
            "all_ones_total": float(mock_total.detach()),
            "derivative_weights": expected_weights}


def real_loss(model: CoupledPredictiveModel) -> dict[str, Tensor]:
    clean = images(2)
    corrupted = clean + 0.03 * torch.randn_like(clean)
    photos = images(4)
    return coupled_region_loss(
        model, clean, corrupted, photos,
        torch.tensor([0, 1]), torch.tensor([[2], [3]]), torch.tensor([0, 1]),
        torch.tensor([0, 1, 1, 0]), ["p0", "p1", "n0", "n1"],
        lambda_sig=0.0, sigreg=None,
    )


def check_real_loss_and_coefficients(model: CoupledPredictiveModel) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    calls: dict[str, int] = {}
    def count(name):
        def hook(*_args):
            calls[name] = calls.get(name, 0) + 1
        return hook
    handles = [module.register_forward_hook(count(name)) for name, module in (
        ("query", model), ("student", model.student_visual.transformer),
        ("predictor", model.predictor), ("text_bank", model.text_bank))]
    try:
        with patch.object(model, "encode_photo", wraps=model.encode_photo) as photos, \
             patch.object(model, "photo_reference", wraps=model.photo_reference) as reference:
            terms = real_loss(model)
            assert photos.call_count == reference.call_count == 1
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {"query": 1, "student": 1, "predictor": 1, "text_bank": 1}, calls
    for name, value in terms.items():
        finite_tensor(value, f"loss/{name}")
    expected = task_loss(terms, "predictive_fusion_v2")
    expected = expected + 0.125 * (terms["clean_rank_pool"] + terms["clean_ce_pool"] + terms["masked_rank_pool"] + terms["masked_ce_pool"])
    expected = expected + 0.025 * (terms["clean_align_i"] + terms["clean_align_t"] + terms["masked_align_i"] + terms["masked_align_t"])
    expected = expected + 0.125 * (terms["clean_ce_t"] + terms["masked_ce_t"])
    expected = expected + 0.5 * terms["anchor_i"] + 0.5 * terms["anchor_t"]
    assert torch.allclose(terms["total"], expected, atol=1e-6, rtol=1e-6)
    terms["total"].backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0)
               for p in active_parameters(model).values())
    return {"term_count": len(terms), "total": float(terms["total"].detach()), "coefficient_identity": "task + pooled*.125 + align*.025 + ce_t*.125 + anchors*.5",
            "single_query_and_bank_calls": calls}


def check_optimizer(model: CoupledPredictiveModel) -> dict[str, Any]:
    groups = model.optimizer_parameter_groups()
    assert groups
    ownership: dict[int, str] = {}
    for group in groups:
        assert float(group["lr"]) in {1e-5, 1e-4}
        assert float(group["weight_decay"]) in {0.0, 1e-2, 1e-4}
        for name, parameter in zip(group["parameter_names"], group["params"], strict=True):
            assert parameter.requires_grad and not name.startswith("original_clip.")
            assert id(parameter) not in ownership, f"duplicate optimizer ownership: {name}"
            ownership[id(parameter)] = name
    assert set(ownership.values()) == set(active_parameters(model))
    assert not any(name.startswith("original_clip") for name in ownership.values())
    original = original_state(model)
    active_before = {name: parameter.detach().clone() for name, parameter in active_parameters(model).items()}
    optimizer = AdamW([{k: v for k, v in group.items() if k in {"params", "lr", "weight_decay"}} for group in groups], betas=(0.9, 0.999), eps=1e-8)
    optimizer.zero_grad(set_to_none=True)
    loss = real_loss(model)["total"]
    loss.backward()
    optimizer.step()
    assert_same_state(original, original_state(model))
    changed = [name for name, parameter in active_parameters(model).items() if not torch.equal(parameter, active_before[name])]
    assert changed
    return {"groups": [{"name": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"], "parameters": len(g["params"])} for g in groups], "changed_after_one_step": len(changed)}


def check_serialization(model: CoupledPredictiveModel) -> dict[str, Any]:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, map_location="cpu", weights_only=True)
    restored = make_model("predictive_fusion_v2", seed=999)
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        left, right = model(images()), restored(images())
    for name in ("g", "q", "mu_i", "mu_t"):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    v1 = make_model("predictive", seed=999)
    try:
        v1.load_state_dict(state, strict=True)
    except RuntimeError as error:
        mismatch = str(error)
    else:
        raise AssertionError("strict F2 state unexpectedly loaded into V1")
    assert "fusion" in mismatch
    return {"state_bytes": stream.tell(), "reload_parity": True, "strict_v1_mismatch": mismatch.splitlines()[0]}


def check_trainer_metadata(model: CoupledPredictiveModel) -> dict[str, Any]:
    groups = model.optimizer_parameter_groups()
    optimizer = AdamW([{k: v for k, v in group.items() if k in {"params", "lr", "weight_decay"}} for group in groups])
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    config = {
        "method_version": "coupled_predictive_fusion_v2",
        "main_query": "mu_i",
        "loss_coefficient_identity": {"rank_i": 1.0, "ce_i": 1.0, "ce_t_aux": 0.25},
        "sampling_identity": {"active_positive_pool": "full"},
        "positive_pool": "full",
        "sigreg_status": "unavailable_pending_new_diagnostic",
        "architecture": "predictive_fusion_v2",
    }
    payload = trainer._checkpoint_payload(
        model, optimizer, scheduler, None, step=0, config=config,
        source_hash="a" * 64, clip={"sha256": "b" * 64}, data_identity={"fixture": True},
        rng={}, selections={}, initialization_hashes={"fixture": True},
    )
    assert payload["method_version"] == config["method_version"]
    assert payload["main_query"] == "mu_i"
    assert payload["loss_coefficient_identity"] == config["loss_coefficient_identity"]
    assert payload["sigreg_state_dict"] is None
    assert payload["resolved_config"]["positive_pool"] == "full"
    assert payload["resolved_config"]["sigreg_status"] == "unavailable_pending_new_diagnostic"
    safe = trainer._wandb_config({**config, "optimizer": "AdamW", "device": "cpu"})
    assert safe["method_version"] == config["method_version"] and safe["main_query"] == "mu_i"
    assert safe["positive_pool"] == "full" and safe["sigreg_status"] == "unavailable_pending_new_diagnostic"
    return {"payload_keys": sorted(k for k in payload if k in {"method_version", "main_query", "loss_coefficient_identity", "sampling_identity"}), "wandb_metadata": safe, "sigreg": "none"}


def check_archived_v1_parity() -> dict[str, Any]:
    if not ARCHIVE_MODEL.is_file() or not ARCHIVE_LOSS.is_file():
        raise FileNotFoundError("archived V1 model/loss snapshot is incomplete")
    name = "spica.models._old_v1"
    spec = importlib.util.spec_from_file_location(name, ARCHIVE_MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    loss_name = "spica._old_v1_losses"
    loss_spec = importlib.util.spec_from_file_location(loss_name, ARCHIVE_LOSS)
    assert loss_spec is not None and loss_spec.loader is not None
    old_loss_module = importlib.util.module_from_spec(loss_spec)
    sys.modules[loss_name] = old_loss_module
    loss_spec.loader.exec_module(old_loss_module)

    archived = module.CoupledPredictiveModel
    current = make_model("predictive", seed=71)
    old_bundle = tiny_bundle(71)
    old = archived(old_bundle.encoder, old_bundle.tokenizer, {0: "cat", 1: "dog"}, architecture="predictive", photo_prompt_length=3, text_prompt_length=4, predictor_width=8, predictor_heads=2).cpu()
    assert current.state_dict().keys() == old.state_dict().keys()
    common = sorted(set(current.state_dict()) & set(old.state_dict()))
    assert common and all(torch.equal(current.state_dict()[name], old.state_dict()[name]) for name in common)
    x = images(2)
    with torch.no_grad():
        a, b = current(x), old(x)
    for field in ("g", "q", "mu_i", "mu_t"):
        assert torch.equal(getattr(a, field), getattr(b, field)), field

    # Same inputs and byte-identical V1 initialization must produce identical
    # loss leaves and gradients in the archived/current implementations.
    clean, corrupted, photos = images(2), images(2), images(4)
    args = (clean, corrupted, photos, torch.tensor([0, 1]), torch.tensor([[2], [3]]),
            torch.tensor([0, 1]), torch.tensor([0, 1, 1, 0]), ["p0", "p1", "n0", "n1"])
    current_terms = coupled_region_loss(current, *args, lambda_sig=0.0, sigreg=None)
    old_terms = old_loss_module.coupled_region_loss(old, *args, lambda_sig=0.0, sigreg=None)
    assert current_terms.keys() == old_terms.keys()
    assert all(torch.equal(current_terms[key], old_terms[key]) for key in current_terms)
    current_terms["total"].backward()
    old_terms["total"].backward()
    current_grads = dict(current.named_parameters())
    old_grads = dict(old.named_parameters())
    grad_keys = sorted(set(current_grads) & set(old_grads))
    grad_keys = [key for key in grad_keys if current_grads[key].requires_grad or old_grads[key].requires_grad]
    assert grad_keys
    for key in grad_keys:
        left, right = current_grads[key].grad, old_grads[key].grad
        assert (left is None) == (right is None), key
        if left is not None:
            assert torch.equal(left, right), key

    f2 = make_model("predictive_fusion_v2", seed=71)
    f2_common = sorted(set(f2.state_dict()) & set(old.state_dict()))
    assert f2_common and all(torch.equal(f2.state_dict()[name], old.state_dict()[name]) for name in f2_common)
    fusion_keys = [name for name in f2.state_dict() if "fusion_attention" in name or "fusion_norm" in name]
    assert set(f2.state_dict()) - set(old.state_dict()) == set(fusion_keys)
    # The historical pooled control also retains its exact forward/loss path.
    pooled = make_model("pooled", seed=71)
    old_bundle = tiny_bundle(71)
    old_pooled = archived(old_bundle.encoder, old_bundle.tokenizer, {0: "cat", 1: "dog"},
                          architecture="pooled", predictor_width=8, predictor_heads=2).cpu()
    assert pooled.state_dict().keys() == old_pooled.state_dict().keys()
    assert all(torch.equal(v, old_pooled.state_dict()[k]) for k, v in pooled.state_dict().items())
    assert torch.equal(pooled(x).q, old_pooled(x).q)
    new_pooled_terms = coupled_region_loss(pooled, *args, lambda_sig=0.0)
    old_pooled_terms = old_loss_module.coupled_region_loss(old_pooled, *args, lambda_sig=0.0)
    assert all(torch.equal(v, old_pooled_terms[k]) for k, v in new_pooled_terms.items())
    new_pooled_terms["total"].backward()
    old_pooled_terms["total"].backward()
    for key, parameter in active_parameters(pooled).items():
        assert torch.equal(parameter.grad, dict(old_pooled.named_parameters())[key].grad), key
    fusion_parameter_count = sum(p.numel() for n, p in f2.named_parameters() if "fusion_attention" in n or "fusion_norm" in n)
    production_attention = nn.MultiheadAttention(256, 4, dropout=0.0, batch_first=True)
    production_norm = nn.LayerNorm(256)
    production_count = sum(p.numel() for m in (production_attention, production_norm) for p in m.parameters())
    assert production_count == 263680
    tokens = torch.randn(2, 7, 256)
    fused, _ = production_attention(tokens, tokens, tokens, need_weights=False)
    assert production_norm(tokens + fused).shape == (2, 7, 256)
    return {"archive": str(ARCHIVE_MODEL), "module_registered": name in sys.modules,
            "loss_module_registered": loss_name in sys.modules, "forward_parity": True,
            "loss_and_gradient_parity": True, "R0_pooled_state_forward_loss_gradient_parity": True,
            "common_state_keys": len(common),
            "f2_common_state_keys": len(f2_common), "new_fusion_keys": len(fusion_keys),
            "new_fusion_parameter_count": fusion_parameter_count,
            "production_added_fusion_parameters": production_count,
            "production_self_attention_shape": [2, 7, 256],
            "new_parameter_difference_formula": "F2 state keys minus V1 state keys = fusion_attention + fusion_norm"}


def check_data_protocol() -> dict[str, Any]:
    from spica.data.coupled_training import load_protocol_data, make_train_loader
    from spica import train_frozen_prompt as legacy
    config = ROOT / "configs/data/sketchy_104_21.yaml"
    pairing = ROOT / "outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.json"
    if not config.is_file() or not pairing.is_file():
        raise FileNotFoundError("approved data config or pairing manifest is absent")
    protocol = load_protocol_data()
    assert len(protocol["pairing"]["mapping"]) == 46624
    assert len(protocol["pairing"]["pool_paths"]) == 8400
    loader = make_train_loader(protocol, lambda value: value, positive_pool="full")
    assert len(loader) > 0
    dataset = loader.dataset
    assert len(dataset.photo_entries) == 58950
    assert dataset.positive_pairing is None
    assert sum(len(rows) for rows in dataset._positive_photos_by_label.values()) == 58950
    canonical = {str(path) for path in protocol["pairing"]["pool_paths"]}
    import random
    random.seed(123)
    sampled = []
    for _ in range(256):
        label = random.choice(tuple(dataset._positive_photos_by_label))
        positives = dataset._sample_positives(label, None)
        negative = dataset._sample_negative(label)
        assert positives and all(entry.label == label for entry in positives)
        assert negative.label != label
        sampled.extend(str(entry.path.resolve()) for entry in positives)
    assert any(path not in canonical for path in sampled)
    legacy_loader = legacy._loader(
        (protocol["split"].train_sketch_entries, protocol["split"].train_photo_entries),
        lambda value: value, protocol["args"], train=True, seed=42, positive_pairing=None,
    )
    assert len(legacy_loader) == len(loader)
    assert len(legacy_loader.dataset.photo_entries) == len(dataset.photo_entries)
    default = make_train_loader(protocol, lambda value: value)
    assert sum(len(v) for v in default.dataset._positive_photos_by_label.values()) == 8400
    assert dataset._positive_photos_by_label is dataset._photos_by_label
    identity = trainer._compact_data_identity(protocol, "full")
    assert identity["positive_pool"]["count"] == identity["sampling"]["negative_pool_count"] == 58950
    assert identity["pairing"]["role"] == "audit_only;not_positive_sampling_source"
    return {"records": len(protocol["pairing"]["mapping"]), "canonical_unique_photo_pool": len(canonical),
            "full_photo_pool": len(dataset.photo_entries), "loader_length": len(loader),
            "sampled_pairs": 256, "image_reads": False,
            "extended_positive_sample": True, "legacy_loader_length": len(legacy_loader)}


def source_receipt() -> dict[str, Any]:
    paths = [
        ROOT / "src/spica/models/coupled_predictive.py",
        ROOT / "src/spica/coupled_predictive_losses.py",
        ROOT / "src/spica/train_coupled_predictive.py",
        ROOT / "src/spica/data/coupled_training.py",
        Path(__file__).resolve(),
    ]
    for archived in (ARCHIVE_MODEL, ARCHIVE_LOSS):
        if archived.is_file():
            paths.append(archived)
    rows = [{"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths]
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(str(row["path"]).encode())
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode())
        aggregate.update(b"\0")
    result = {"aggregate_sha256": aggregate.hexdigest(), "files": rows}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh output root; an existing path is refused")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(tempfile.mkdtemp(prefix="coupled_fusion_v2_cpu_"))
    else:
        args.output = args.output.expanduser().resolve()
        if args.output.exists():
            parser.error(f"--output must not already exist: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.mkdir()
    return args


def main() -> int:
    args = parse_args()
    root = args.output
    hashes = source_receipt()
    write_json(root / "source_hash.json", hashes)
    receipt = Receipt(root)
    receipt.line(f"output={root}")
    receipt.line(f"source_aggregate_sha256={hashes['aggregate_sha256']}")
    failures_before_model = receipt.check("protocol routing before model/data", check_routing)
    del failures_before_model
    receipt.check("CPU real manifest and pool protocol", check_data_protocol, skip_if=lambda error: isinstance(error, FileNotFoundError))
    model = make_model()
    receipt.check("F2 forward/backward/freeze/independent student", lambda: check_forward_backward(model))
    receipt.check("CA -> fusion self-attention -> FFN hooks and conditioning", lambda: check_hooks_and_context(model))
    receipt.check("batched versus separate no-context-leakage", lambda: check_no_leakage(model))
    receipt.check("mu_i to text and mu_t to photo prompt gradients", lambda: check_prompt_gradients(model))
    receipt.check("prompt bank parameter ownership", lambda: check_prompt_ownership(model))
    receipt.check("mock _view_terms identity and main-only task loss", check_view_terms_and_loss)
    receipt.check("real F2 loss and coefficient identity", lambda: check_real_loss_and_coefficients(model))
    receipt.check("AdamW ownership, hyperparameters, and one CPU step", lambda: check_optimizer(model))
    receipt.check("state reload parity and strict V1 mismatch", lambda: check_serialization(model))
    receipt.check("trainer checkpoint and W&B metadata", lambda: check_trainer_metadata(model))
    receipt.check("archived V1 sys.modules parity", check_archived_v1_parity, skip_if=lambda error: isinstance(error, FileNotFoundError))
    final = receipt.close(source_hash=hashes)
    if final["status"] != "PASS":
        print(f"CPU gate FAIL; receipt: {root / 'receipt.json'}", file=sys.stderr)
        return 1
    print(f"CPU gate PASS; receipt: {root / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
