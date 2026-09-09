#!/usr/bin/env python3
"""Standalone, no-download CPU gate for the F2_MP multi-positive objective."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

# This must precede every torch/open_clip import.
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
    _main_photo_multi_positive,
    coupled_region_loss,
)
from spica.data.coupled_training import _arm_protocol  # noqa: E402
from spica.models.clip import FrozenClipBundle, FrozenClipEncoder  # noqa: E402
from spica.models.coupled_predictive import CoupledPredictiveModel  # noqa: E402
import spica.train_coupled_predictive as trainer  # noqa: E402

ARCHIVE = ROOT / "outputs/fusion_execution_20260909T064000Z/source_snapshot/files/src/spica"
ARCHIVE_MODEL = ARCHIVE / "models/coupled_predictive.py"
ARCHIVE_LOSS = ARCHIVE / "coupled_predictive_losses.py"
TEMP = 0.07


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def finite(value: Tensor, name: str) -> None:
    assert isinstance(value, Tensor) and bool(torch.isfinite(value).all()), name


class Receipt:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.log_path = root / "gate.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        self.checks: list[dict[str, Any]] = []
        self.started = time.time()

    def check(self, name: str, function: Callable[[], Any], *, skip_if: Callable[[BaseException], bool] | None = None) -> Any:
        print(f"START {name}", file=self.log, flush=True)
        started = time.time()
        try:
            result = function()
        except Exception as error:  # noqa: BLE001
            status = "SKIP" if skip_if is not None and skip_if(error) else "FAIL"
            row = {"name": name, "status": status, "seconds": round(time.time() - started, 6), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            self.checks.append(row)
            print(f"{status} {name}: {row['error']}", file=self.log, flush=True)
            return None
        row = {"name": name, "status": "PASS", "seconds": round(time.time() - started, 6)}
        if isinstance(result, dict):
            row["details"] = result
        self.checks.append(row)
        print(f"PASS {name}", file=self.log, flush=True)
        return result

    def close(self, source_hash: dict[str, Any]) -> dict[str, Any]:
        self.log.close()
        failed = [row for row in self.checks if row["status"] == "FAIL"]
        skipped = [row for row in self.checks if row["status"] == "SKIP"]
        receipt = {
            "schema_version": 1,
            "status": "FAIL" if failed else "INCOMPLETE" if skipped else "PASS",
            "verified": not failed and not skipped,
            "gate": "fusion_multipositive_cpu",
            "device": "cpu",
            "pretrained_weights": False,
            "network": False,
            "training": False,
            "source_hash": source_hash,
            "checks": self.checks,
            "summary": {"pass": len(self.checks) - len(failed) - len(skipped), "fail": len(failed), "skip": len(skipped)},
            "log": str(self.log_path),
            "started_unix": self.started,
            "finished_unix": time.time(),
        }
        write_json(self.root / "receipt.json", receipt)
        return receipt


def tiny_bundle(seed: int = 42) -> FrozenClipBundle:
    torch.manual_seed(seed)
    model = CLIP(
        embed_dim=6,
        vision_cfg=CLIPVisionCfg(layers=1, width=8, head_width=4, patch_size=4, image_size=8),
        text_cfg=CLIPTextCfg(context_length=77, vocab_size=49408, width=8, heads=2, layers=1),
    ).float().cpu()
    return FrozenClipBundle(
        encoder=FrozenClipEncoder(model),
        transform=lambda value: value,
        tokenizer=open_clip.get_tokenizer("ViT-B-32"),
        model_name="tiny_cpu_fixture",
        pretrained=None,
    )


def make_model(seed: int = 42) -> CoupledPredictiveModel:
    bundle = tiny_bundle(seed)
    return CoupledPredictiveModel(
        bundle.encoder, bundle.tokenizer, {0: "cat", 1: "dog"},
        architecture="predictive_fusion_v2", photo_prompt_length=3,
        text_prompt_length=4, predictor_width=8, predictor_heads=2,
    ).cpu()


def images(batch: int) -> Tensor:
    torch.manual_seed(123 + batch)
    return torch.randn(batch, 3, 8, 8)


def multi_positive_oracle(query: Tensor, bank: Tensor, bank_labels: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    logits = (query / query.norm(dim=-1, keepdim=True)) @ (bank / bank.norm(dim=-1, keepdim=True)).T / TEMP
    log_probability = torch.log_softmax(logits, dim=-1)
    positive = bank_labels[None, :] == labels[:, None]
    count = positive.sum(dim=-1)
    assert bool((count > 0).all())
    loss = (-(log_probability * positive).sum(dim=-1) / count).mean()
    return loss, logits, positive


def check_multi_positive_formula() -> dict[str, Any]:
    torch.manual_seed(7)
    query = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    bank = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    bank_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    labels = torch.tensor([0, 1, 2])
    expected, logits, positive = multi_positive_oracle(query, bank, bank_labels, labels)
    actual = _main_photo_multi_positive(query, bank, bank_labels, labels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert positive[0].sum() == 3 and positive[1].sum() == 3

    extreme = torch.tensor([[10000.0, -10000.0, 0.0], [-9000.0, 8000.0, 7000.0]], dtype=torch.float64)
    assert bool(torch.isfinite(torch.log_softmax(extreme, dim=-1)).all())
    huge_query = torch.tensor([[1e150, -2e150, 3e149]], dtype=torch.float64)
    huge_bank = torch.tensor([[2e150, 1e150, -1e149], [-1e150, 2e150, 4e149]], dtype=torch.float64)
    huge = _main_photo_multi_positive(huge_query, huge_bank, torch.tensor([0, 1]), torch.tensor([0]))
    finite(huge, "extreme multi-positive loss")

    actual.backward()
    finite(query.grad, "multi-positive query gradient")
    finite(bank.grad, "multi-positive bank gradient")
    assert float(query.grad.abs().sum()) > 0 and float(bank.grad.abs().sum()) > 0

    # The logit-level oracle makes the all-positive direction explicit.
    raw = logits.detach().requires_grad_()
    logp = torch.log_softmax(raw, dim=-1)
    counts = positive.sum(dim=-1)
    logit_loss = (-(logp * positive).sum(dim=-1) / counts).mean()
    gradient = torch.autograd.grad(logit_loss, raw)[0]
    positive_float = positive.to(raw.dtype)
    expected_gradient = (torch.softmax(raw, dim=-1) * positive_float.sum(dim=-1, keepdim=True) - positive_float) / counts[:, None] / query.shape[0]
    # The identity is exact analytically; floating-point autograd may differ by
    # a few ulps after the reduction, so keep this a strict float64 check.
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert bool((gradient * positive).sum(dim=-1).lt(0).all())

    permutation = torch.tensor([6, 2, 5, 0, 4, 1, 3])
    permuted = _main_photo_multi_positive(query.detach(), bank.detach()[permutation], bank_labels[permutation], labels)
    torch.testing.assert_close(permuted, actual.detach(), rtol=1e-14, atol=1e-14)
    relabeled = bank_labels.clone()
    relabeled[0] = 1
    changed, _, changed_mask = multi_positive_oracle(query.detach(), bank.detach(), relabeled, labels)
    assert not torch.equal(changed_mask, positive) and not torch.equal(changed, expected)
    return {"dtype": "float64", "temperature": TEMP, "positive_counts": positive.sum(dim=-1).tolist(), "permutation_invariant": True, "relabel_changed_entries": True}


def load_archive() -> tuple[Any, Any]:
    if not ARCHIVE_MODEL.is_file() or not ARCHIVE_LOSS.is_file():
        raise FileNotFoundError("archived F2 source snapshot is incomplete")
    model_name = "spica.models._fusion_f2_archive"
    spec = importlib.util.spec_from_file_location(model_name, ARCHIVE_MODEL)
    assert spec is not None and spec.loader is not None
    old_model = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = old_model
    spec.loader.exec_module(old_model)
    loss_name = "spica._fusion_f2_archive_losses"
    loss_spec = importlib.util.spec_from_file_location(loss_name, ARCHIVE_LOSS)
    assert loss_spec is not None and loss_spec.loader is not None
    old_loss = importlib.util.module_from_spec(loss_spec)
    sys.modules[loss_name] = old_loss
    loss_spec.loader.exec_module(old_loss)
    return old_model, old_loss


def legacy_args() -> tuple[Tensor, ...]:
    return (
        images(2), images(2), images(6), torch.tensor([0, 1]),
        torch.tensor([[2], [3]]), torch.tensor([0, 1]),
        torch.tensor([0, 0, 1, 1, 0, 1]), ["p0", "p1", "p2", "p3", "p4", "p5"],
    )


def check_legacy_parity() -> dict[str, Any]:
    old_model_module, old_loss = load_archive()
    current = make_model(71)
    old_bundle = tiny_bundle(71)
    archived = old_model_module.CoupledPredictiveModel(
        old_bundle.encoder, old_bundle.tokenizer, {0: "cat", 1: "dog"},
        architecture="predictive_fusion_v2", photo_prompt_length=3,
        text_prompt_length=4, predictor_width=8, predictor_heads=2,
    ).cpu()
    assert current.state_dict().keys() == archived.state_dict().keys()
    assert all(torch.equal(current.state_dict()[key], archived.state_dict()[key]) for key in current.state_dict())
    args = legacy_args()
    current_terms = coupled_region_loss(current, *args, lambda_sig=0.0, sigreg=None)
    archived_terms = old_loss.coupled_region_loss(archived, *args, lambda_sig=0.0, sigreg=None)
    assert current_terms.keys() == archived_terms.keys()
    assert all(torch.equal(current_terms[key], archived_terms[key]) for key in current_terms)
    current_terms["total"].backward()
    archived_terms["total"].backward()
    old_parameters = dict(archived.named_parameters())
    for name, parameter in current.named_parameters():
        other = old_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad), name
    return {"archive_model": str(ARCHIVE_MODEL), "archive_loss": str(ARCHIVE_LOSS), "terms": len(current_terms), "forward_loss_gradient_exact": True}


def mp_batch() -> tuple[Tensor, ...]:
    return (
        images(2), images(2), images(6), torch.tensor([0, 3]),
        torch.tensor([[3], [0]]), torch.tensor([0, 1]),
        torch.tensor([0, 0, 0, 1, 1, 1]), ["p0", "p1", "p2", "p3", "p4", "p5"],
    )


def active_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def state_copy(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.original_clip.state_dict().items()}


def same_state(before: dict[str, Tensor], model: CoupledPredictiveModel) -> None:
    after = model.original_clip.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[name], after[name]) for name in before)


def check_f2_mp_loss_and_step(root: Path) -> dict[str, Any]:
    model = make_model(42)
    before_original = state_copy(model)
    args = mp_batch()
    calls = {"model": 0, "photo": 0, "reference": 0, "text": 0}
    model_handle = model.register_forward_hook(lambda *_: calls.__setitem__("model", calls["model"] + 1))
    text_handle = model.text_bank.register_forward_hook(lambda *_: calls.__setitem__("text", calls["text"] + 1))
    original_photo = model.encode_photo
    original_reference = model.photo_reference
    def photo(value: Tensor) -> Tensor:
        calls["photo"] += 1
        return original_photo(value)
    def reference(value: Tensor) -> Tensor:
        calls["reference"] += 1
        return original_reference(value)
    model.encode_photo = photo  # type: ignore[method-assign]
    model.photo_reference = reference  # type: ignore[method-assign]
    try:
        paired = coupled_region_loss(model, *args, lambda_sig=0.0, sigreg=None, main_photo_objective="paired_softplus")
        mp = coupled_region_loss(model, *args, lambda_sig=0.0, sigreg=None, main_photo_objective="multi_positive_supervised_contrastive")
    finally:
        model_handle.remove()
        text_handle.remove()
    assert calls == {"model": 2, "photo": 2, "reference": 2, "text": 2}, calls
    changed = {"clean_rank_i", "masked_rank_i", "total"}
    for name in paired:
        if name not in changed:
            assert torch.equal(paired[name], mp[name]), name
    assert not torch.equal(paired["clean_rank_i"], mp["clean_rank_i"])
    expected_total = 0.5 * (mp["clean_rank_i"] + mp["clean_ce_i"] + mp["masked_rank_i"] + mp["masked_ce_i"])
    expected_total = expected_total + 0.125 * (mp["clean_rank_pool"] + mp["clean_ce_pool"] + mp["masked_rank_pool"] + mp["masked_ce_pool"])
    expected_total = expected_total + 0.025 * (mp["clean_align_i"] + mp["clean_align_t"] + mp["masked_align_i"] + mp["masked_align_t"])
    expected_total = expected_total + 0.125 * (mp["clean_ce_t"] + mp["masked_ce_t"])
    expected_total = expected_total + 0.5 * mp["anchor_i"] + 0.5 * mp["anchor_t"]
    torch.testing.assert_close(mp["total"], expected_total, rtol=0, atol=0)

    model.zero_grad(set_to_none=True)
    mp["total"].backward()
    active = active_parameters(model)
    assert active and all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.abs().sum()) > 0 for parameter in active.values())
    same_state(before_original, model)

    optimizer = AdamW([{key: value for key, value in group.items() if key in {"params", "lr", "weight_decay"}} for group in model.optimizer_parameter_groups()], betas=(0.9, 0.999), eps=1e-8)
    active_before_step = {name: parameter.detach().clone() for name, parameter in active.items()}
    optimizer.step()
    assert any(not torch.equal(parameter.detach(), active_before_step[name]) for name, parameter in active.items())
    same_state(before_original, model)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    config = {
        "method_version": "coupled_predictive_fusion_mp_v1",
        "architecture": "predictive_fusion_v2",
        "main_photo_objective": "multi_positive_supervised_contrastive",
        "main_photo_temperature": TEMP,
        "objective_identity": "rank_i=multi_positive_supervised_contrastive_over_unique_live_photobank",
        "main_query": "mu_i",
        "lambda_sig": 0.0,
        "sigreg_status": "disabled_control",
        "loss_coefficient_identity": {"rank_i": 1.0, "ce_i": 1.0, "ce_t_aux": 0.25, "rank_pool": 0.25, "ce_pool": 0.25, "align_i": 0.05, "align_t": 0.05, "anchor_i": 0.5, "anchor_t": 0.5, "sigreg": 0.0},
        "sampling_identity": {"active_positive_pool": "full"},
    }
    payload = trainer._checkpoint_payload(
        model, optimizer, scheduler, None, step=1, config=config,
        source_hash="a" * 64, clip={"sha256": "b" * 64}, data_identity={"fixture": True},
        rng={}, selections={}, initialization_hashes={"fixture": True},
    )
    resolved = payload["resolved_config"]
    for key in ("method_version", "architecture", "main_photo_objective", "main_photo_temperature", "objective_identity"):
        assert resolved[key] == config[key]
    assert payload["sigreg_state_dict"] is None and resolved["lambda_sig"] == 0.0
    payload["model_state_dict"] = model.state_dict()
    checkpoint = root / "checkpoint_step1.pt"
    torch.save(payload, checkpoint)
    restored = make_model(999)
    restored_optimizer = AdamW([{key: value for key, value in group.items() if key in {"params", "lr", "weight_decay"}} for group in restored.optimizer_parameter_groups()], betas=(0.9, 0.999), eps=1e-8)
    import numpy as np
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]):
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    assert json.dumps(optimizer.state_dict(), sort_keys=True, default=str) == json.dumps(restored_optimizer.state_dict(), sort_keys=True, default=str)
    moments = sum(1 for state in optimizer.state.values() if "exp_avg" in state and "exp_avg_sq" in state)
    assert moments > 0
    return {"terms": len(mp), "coefficients_exact": True, "forward_calls": calls, "active_gradients": len(active), "optimizer_moment_states": moments, "checkpoint": str(checkpoint), "restore_exact": True}


def check_routing_and_metadata() -> dict[str, Any]:
    expected = {"architecture": "predictive_fusion_v2", "method_version": "coupled_predictive_fusion_mp_v1", "positive_pool": "full", "main_photo_objective": "multi_positive_supervised_contrastive"}
    assert _arm_protocol("F2_MP", "coupled_predictive_fusion_mp_v1", None) == expected
    calls = 0
    original_device = trainer._device
    def forbidden(value: str) -> torch.device:
        nonlocal calls
        calls += 1
        raise AssertionError(f"device reached for invalid route: {value}")
    trainer._device = forbidden
    try:
        for campaign_id, diagnostic in (("wrong", None), ("coupled_predictive_fusion_mp_v1", object())):
            namespace = argparse.Namespace(arm="F2_MP", campaign_id=campaign_id, diagnostic=diagnostic, max_steps=2, smoke=True, device="cuda")
            try:
                trainer._train_impl(namespace, ROOT / "never-created-f2-mp-output")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid F2_MP route was accepted")
    finally:
        trainer._device = original_device
    assert calls == 0
    return {"protocol": expected, "invalid_routes": 2, "device_calls_for_invalid": calls}


def check_full_pool_metadata() -> dict[str, Any]:
    from spica.data.coupled_training import load_protocol_data, make_train_loader
    protocol = load_protocol_data()
    loader = make_train_loader(protocol, lambda value: value, positive_pool="full")
    dataset = loader.dataset
    assert len(dataset.photo_entries) == 58950
    assert dataset.positive_sampling == "same_class" and dataset.num_positive_photos == 1
    assert dataset.positive_pairing is None
    assert dataset._positive_photos_by_label is dataset._photos_by_label
    return {"loader_length": len(loader), "full_photo_pool": len(dataset.photo_entries), "num_positive_photos": dataset.num_positive_photos, "image_pixels_read": False}


def source_receipt() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/spica/coupled_predictive_losses.py",
        ROOT / "src/spica/data/coupled_training.py",
        ROOT / "src/spica/train_coupled_predictive.py",
        ROOT / "src/spica/models/coupled_predictive.py",
        ARCHIVE_MODEL,
        ARCHIVE_LOSS,
    ]
    rows = []
    for path in paths:
        if path.is_file():
            rows.append({"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(f"{row['path']}\0{row['sha256']}\0".encode())
    return {"aggregate_sha256": aggregate.hexdigest(), "files": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "outputs" / f"fusion_multipositive_cpu_{stamp}"
    else:
        args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()
    return args


def main() -> int:
    args = parse_args()
    hashes = source_receipt()
    write_json(args.output / "source_hash.json", hashes)
    receipt = Receipt(args.output)
    receipt.check("F2_MP routing before device/data", check_routing_and_metadata)
    receipt.check("float64 multi-positive oracle and invariances", check_multi_positive_formula)
    receipt.check("paired_softplus current versus archived F2 exact parity", check_legacy_parity, skip_if=lambda error: isinstance(error, FileNotFoundError))
    receipt.check("F2_MP production loss, hooks, gradients, AdamW, checkpoint restore", lambda: check_f2_mp_loss_and_step(args.output))
    receipt.check("full-pool loader metadata without image pixels", check_full_pool_metadata, skip_if=lambda error: isinstance(error, FileNotFoundError))
    final = receipt.close(hashes)
    if final["status"] != "PASS":
        print(f"CPU gate FAIL; receipt: {args.output / 'receipt.json'}", file=sys.stderr)
        return 1
    print(f"CPU gate PASS; receipt: {args.output / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
