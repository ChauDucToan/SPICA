#!/usr/bin/env python3
"""No-download CPU gate for the production F2_QMP loss and q readout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

from spica.coupled_predictive_losses import (  # noqa: E402
    coupled_region_loss,
    _main_photo_multi_positive,
    _view_terms,
)
from spica.data.coupled_training import _arm_protocol  # noqa: E402
from spica.models.clip import FrozenClipBundle, FrozenClipEncoder  # noqa: E402
from spica.models.coupled_predictive import (  # noqa: E402
    CoupledPredictiveModel,
    CoupledPredictiveOutput,
)
from spica.evaluation.coupled_predictive import QOnlyAdapter  # noqa: E402
import spica.train_coupled_predictive as trainer  # noqa: E402

ARCHIVE = (
    ROOT / "outputs/fusion_execution_20260909T064000Z/source_snapshot/files/src/spica"
)
ARCHIVE_LOSS = ARCHIVE / "coupled_predictive_losses.py"
MP_ROOT = ROOT / "outputs/fusion_mp_execution_20260909T115500Z"
MP_ARCHIVE = MP_ROOT / "runs/F2_MP/source_snapshot/files/src/spica"
MP_LOSS = MP_ARCHIVE / "coupled_predictive_losses.py"
MP_MANIFEST = MP_ROOT / "source_snapshot/index.json"
MP_EXECUTION = MP_ROOT / "execution_manifest.json"
MP_GATE = MP_ROOT / "launch_gate.json"
MP_ARCHIVE_FILES = MP_ROOT / "runs/F2_MP/source_snapshot/files"
MP_ARCHIVE_MODEL = MP_ARCHIVE_FILES / "src/spica/models/coupled_predictive.py"
MP_ARCHIVE_DATA = MP_ARCHIVE_FILES / "src/spica/data/coupled_training.py"
MP_ARCHIVE_EVAL = MP_ARCHIVE_FILES / "src/spica/evaluation/coupled_predictive.py"
MP_ARCHIVE_TRAINER = MP_ARCHIVE_FILES / "src/spica/train_coupled_predictive.py"
MP_ARCHIVE_LAUNCHER = MP_ARCHIVE_FILES / "scripts/run_fusion_multipositive.py"
TEMP = 0.07


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tiny_model(seed: int = 42) -> CoupledPredictiveModel:
    torch.manual_seed(seed)
    clip = (
        CLIP(
            embed_dim=6,
            vision_cfg=CLIPVisionCfg(
                layers=1, width=8, head_width=4, patch_size=4, image_size=8
            ),
            text_cfg=CLIPTextCfg(
                context_length=77, vocab_size=49408, width=8, heads=2, layers=1
            ),
        )
        .float()
        .cpu()
    )
    bundle = FrozenClipBundle(
        encoder=FrozenClipEncoder(clip),
        transform=lambda x: x,
        tokenizer=open_clip.get_tokenizer("ViT-B-32"),
        model_name="tiny_cpu",
        pretrained=None,
    )
    return CoupledPredictiveModel(
        bundle.encoder,
        bundle.tokenizer,
        {0: "cat", 1: "dog"},
        architecture="predictive_fusion_v2",
        photo_prompt_length=3,
        text_prompt_length=4,
        predictor_width=8,
        predictor_heads=2,
    ).cpu()


def images(n: int) -> Tensor:
    torch.manual_seed(100 + n)
    return torch.randn(n, 3, 8, 8)


def batch() -> tuple[Tensor, ...]:
    return (
        images(2),
        images(2),
        images(6),
        torch.tensor([0, 3]),
        torch.tensor([[3], [0]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 0, 0, 1, 1, 1]),
        ["p0", "p1", "p2", "p3", "p4", "p5"],
    )


def active(model: nn.Module) -> dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def oracle(q: Tensor, bank: Tensor, bank_labels: Tensor, labels: Tensor) -> Tensor:
    logits = (
        torch.nn.functional.normalize(q, dim=-1)
        @ torch.nn.functional.normalize(bank, dim=-1).T
    )
    positive = bank_labels[None, :] == labels[:, None]
    return (
        -(torch.log_softmax(logits / TEMP, dim=-1) * positive).sum(-1)
        / positive.sum(-1)
    ).mean()


def load_archive_loss(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"archived loss is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args_legacy() -> tuple[Tensor, ...]:
    return (
        images(2),
        images(2),
        images(6),
        torch.tensor([0, 1]),
        torch.tensor([[2], [3]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 0, 1, 1, 0, 1]),
        ["p0", "p1", "p2", "p3", "p4", "p5"],
    )


def check_formula() -> dict[str, Any]:
    torch.manual_seed(7)
    q = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    bank = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    bank_labels, labels = torch.tensor([0, 0, 0, 1, 1, 1, 2]), torch.tensor([0, 1, 2])
    expected = oracle(q, bank, bank_labels, labels)
    actual = _main_photo_multi_positive(q, bank, bank_labels, labels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    try:
        _main_photo_multi_positive(
            q.detach(),
            bank.detach(),
            torch.tensor([0, 0, 1, 1, 1, 1, 1]),
            torch.tensor([2, 0, 1]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero-positive labels were accepted")
    actual.backward()
    assert q.grad is not None and bank.grad is not None
    assert float(q.grad.abs().sum()) > 0 and float(bank.grad.abs().sum()) > 0
    return {
        "temperature": TEMP,
        "positive_counts": [3, 3, 1],
        "rounding_tolerance": "torch float64 exact; production checks use rtol=1e-6, atol=1e-7",
    }


def check_loss_and_step(output: Path) -> dict[str, Any]:
    model = tiny_model(42)
    original = {
        n: p.detach().clone() for n, p in model.original_clip.state_dict().items()
    }
    args = batch()
    old = coupled_region_loss(
        model,
        *args,
        lambda_sig=0.0,
        sigreg=None,
        main_photo_objective="multi_positive_supervised_contrastive",
    )
    qmp = coupled_region_loss(
        model,
        *args,
        lambda_sig=0.0,
        sigreg=None,
        main_photo_objective="multi_positive_pooled_contrastive",
    )
    assert "clean_rank_q_mp" in qmp and "masked_rank_q_mp" in qmp
    assert "clean_rank_i" not in qmp and "masked_rank_i" not in qmp
    for key in qmp:
        if key not in {"clean_rank_q_mp", "masked_rank_q_mp", "total"}:
            assert torch.equal(qmp[key], old[key]), key
    expected = 0.5 * (
        qmp["clean_rank_q_mp"]
        + qmp["clean_ce_i"]
        + qmp["masked_rank_q_mp"]
        + qmp["masked_ce_i"]
    )
    expected += 0.125 * sum(
        qmp[f"{view}_{name}"]
        for view in ("clean", "masked")
        for name in ("rank_pool", "ce_pool")
    )
    expected += 0.025 * sum(
        qmp[f"{view}_{name}"]
        for view in ("clean", "masked")
        for name in ("align_i", "align_t")
    )
    expected += 0.125 * (qmp["clean_ce_t"] + qmp["masked_ce_t"])
    expected += 0.5 * qmp["anchor_i"] + 0.5 * qmp["anchor_t"]
    torch.testing.assert_close(qmp["total"], expected, rtol=1e-6, atol=1e-7)
    expected_delta = old["total"] - 0.5 * (old["clean_rank_i"] + old["masked_rank_i"])
    expected_delta = expected_delta + 0.5 * (
        qmp["clean_rank_q_mp"] + qmp["masked_rank_q_mp"]
    )
    torch.testing.assert_close(qmp["total"], expected_delta, rtol=0.0, atol=2e-6)

    model.zero_grad(set_to_none=True)
    qmp["total"].backward()
    trainable = active(model)
    assert trainable and all(
        p.grad is not None
        and bool(torch.isfinite(p.grad).all())
        and float(p.grad.abs().sum()) > 0
        for p in trainable.values()
    )
    assert all(
        torch.equal(value, model.original_clip.state_dict()[name])
        for name, value in original.items()
    )

    # The production view graph, not detached dummy μ tensors, must isolate q-MP.
    qv = torch.randn(2, 5, requires_grad=True)
    bank = torch.randn(4, 5, requires_grad=True)
    mu_i = torch.randn(2, 5, requires_grad=True)
    mu_t = torch.randn(2, 5, requires_grad=True)
    view_output = CoupledPredictiveOutput(
        g=torch.randn(2, 5), q=qv, mu_i=mu_i, mu_t=mu_t
    )
    terms = _view_terms(
        view_output,
        bank[:2],
        bank[2:],
        torch.randn(2, 5),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        "predictive_fusion_v2",
        live_photos=bank,
        photo_labels=torch.tensor([0, 0, 1, 1]),
        main_photo_objective="multi_positive_pooled_contrastive",
    )
    assert "rank_q_mp" in terms and "rank_i" not in terms and "muT_MP" not in terms
    gradients = torch.autograd.grad(
        terms["rank_q_mp"], (qv, bank, mu_i, mu_t), allow_unused=True
    )
    assert gradients[0] is not None and float(gradients[0].abs().sum()) > 0
    assert gradients[1] is not None and float(gradients[1].abs().sum()) > 0
    assert gradients[2] is None and gradients[3] is None

    optimizer = trainer._make_optimizer(model)
    expected_groups = model.optimizer_parameter_groups()
    assert len(optimizer.param_groups) == len(expected_groups)
    for actual, expected in zip(optimizer.param_groups, expected_groups):
        assert (
            actual["lr"] == expected["lr"]
            and actual["weight_decay"] == expected["weight_decay"]
        )
        assert actual["params"] == expected["params"]
    before = {n: p.detach().clone() for n, p in trainable.items()}
    optimizer.step()
    assert any(not torch.equal(p.detach(), before[n]) for n, p in trainable.items())
    assert all(
        torch.equal(value, model.original_clip.state_dict()[name])
        for name, value in original.items()
    )

    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    config = {
        "method_version": "coupled_predictive_fusion_qmp_v1",
        "architecture": "predictive_fusion_v2",
        "main_query": "q",
        "main_photo_objective": "multi_positive_pooled_contrastive",
        "main_photo_temperature": TEMP,
        "objective_identity": "rank_q_mp=multi_positive_supervised_contrastive_over_unique_live_photobank",
        "loss_coefficient_identity": {
            "rank_q_mp": 1.0,
            "ce_i": 1.0,
            "ce_t_aux": 0.25,
            "rank_pool": 0.25,
            "ce_pool": 0.25,
            "align_i": 0.05,
            "align_t": 0.05,
            "anchor_i": 0.5,
            "anchor_t": 0.5,
            "sigreg": 0.0,
        },
        "sampling_identity": {"active_positive_pool": "full"},
        "primary_comparison": {
            "baseline_arm": "F2_MP",
            "baseline_wandb_run_id": "y40hu06b",
        },
    }
    payload = trainer._checkpoint_payload(
        model,
        optimizer,
        scheduler,
        None,
        step=1,
        config=config,
        source_hash="a" * 64,
        clip={"sha256": "b" * 64},
        data_identity={"fixture": True},
        rng={},
        selections={},
        initialization_hashes={"fixture": True},
    )
    assert payload["method_version"] == config["method_version"]
    assert payload["main_query"] == "q"
    assert payload["main_photo_objective"] == config["main_photo_objective"]
    assert payload["main_photo_temperature"] == TEMP
    assert payload["objective_identity"] == config["objective_identity"]
    assert payload["loss_coefficient_identity"] == config["loss_coefficient_identity"]
    assert payload["sampling_identity"] == config["sampling_identity"]
    assert payload["primary_comparison"] == config["primary_comparison"]
    assert payload["resolved_config"] == config
    path = output / "checkpoint_cpu.pt"
    torch.save(payload, path)
    restored = tiny_model(42)
    restored_optimizer = trainer._make_optimizer(restored)
    safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype]
    if hasattr(np, "dtypes"):
        safe.append(np.dtypes.UInt32DType)
    with torch.serialization.safe_globals(safe):
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = restored.load_state_dict(
        loaded["model_state_dict"], strict=False
    )
    expected_missing = [
        name for name in restored.state_dict() if name.startswith("original_clip.")
    ]
    assert sorted(missing) == sorted(expected_missing) and not unexpected
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    assert json.dumps(
        optimizer.state_dict(), sort_keys=True, default=str
    ) == json.dumps(restored_optimizer.state_dict(), sort_keys=True, default=str)
    assert sum("exp_avg" in state for state in optimizer.state.values()) > 0
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
        if not name.startswith("original_clip.")
    )
    expected_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("original_clip.")
    }
    restored_state = restored.state_dict()
    assert all(
        torch.equal(value, restored_state[name])
        for name, value in expected_state.items()
    )
    assert all(
        name.startswith("original_clip.")
        for name in set(restored_state) - set(loaded["model_state_dict"])
    )
    return {
        "terms": len(qmp),
        "all_other_terms_exact": True,
        "active_gradients": len(trainable),
        "optimizer_groups": len(optimizer.param_groups),
        "optimizer_moment_states": sum(
            "exp_avg" in state for state in optimizer.state.values()
        ),
        "optimizer_step": True,
        "safe_weights_only_restore": True,
        "production_checkpoint_schema": True,
    }


def _loss_parity(path: Path, name: str, objective: str | None = None) -> None:
    old = load_archive_loss(path, name)
    current_model, old_model = tiny_model(71), tiny_model(71)
    kwargs = {} if objective is None else {"main_photo_objective": objective}
    current = coupled_region_loss(
        current_model, *args_legacy(), lambda_sig=0.0, sigreg=None, **kwargs
    )
    archived = old.coupled_region_loss(
        old_model, *args_legacy(), lambda_sig=0.0, sigreg=None, **kwargs
    )
    assert current.keys() == archived.keys()
    assert all(torch.equal(current[k], archived[k]) for k in current)
    current["total"].backward()
    archived["total"].backward()
    for name_, parameter in current_model.named_parameters():
        other = dict(old_model.named_parameters())[name_]
        assert (parameter.grad is None) == (other.grad is None), name_
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad), name_


def check_legacy() -> dict[str, Any]:
    _loss_parity(ARCHIVE_LOSS, "spica._f2_qmp_archive_loss")
    _loss_parity(
        MP_LOSS, "spica._f2_mp_archive_loss", "multi_positive_supervised_contrastive"
    )
    return {
        "archives": [str(ARCHIVE_LOSS), str(MP_LOSS)],
        "default_f2_and_explicit_mp_scalar_and_gradient_exact": True,
    }


def check_routing() -> dict[str, Any]:
    expected = _arm_protocol("F2_QMP", "coupled_predictive_fusion_qmp_v1", None)
    assert expected == {
        "architecture": "predictive_fusion_v2",
        "method_version": "coupled_predictive_fusion_qmp_v1",
        "positive_pool": "full",
        "main_photo_objective": "multi_positive_pooled_contrastive",
    }
    calls = 0
    original = trainer._device
    trainer._device = lambda value: (_ for _ in ()).throw(
        AssertionError("device reached before route rejection")
    )
    try:
        for campaign, diagnostic in (
            ("wrong", None),
            ("coupled_predictive_fusion_qmp_v1", object()),
        ):
            namespace = argparse.Namespace(
                arm="F2_QMP",
                campaign_id=campaign,
                diagnostic=diagnostic,
                max_steps=2,
                smoke=True,
                device="cuda",
            )
            try:
                trainer._train_impl(namespace, ROOT / "never-created-qmp-output")
            except ValueError:
                calls += 0
            else:
                raise AssertionError("invalid route accepted")
    finally:
        trainer._device = original
    assert calls == 0
    return {
        "protocol": expected,
        "invalid_campaign_and_diagnostic_rejected_before_device_data": True,
    }


def check_archive_bindings() -> dict[str, Any]:
    execution = json.loads(MP_EXECUTION.read_text(encoding="utf-8"))
    index = json.loads(MP_MANIFEST.read_text(encoding="utf-8"))
    gate = json.loads(MP_GATE.read_text(encoding="utf-8"))
    assert (
        execution["source_snapshot"]["sha256"]
        == index["sha256"]
        == "ab064ced3b11c95f142ac94b336072d88312f5e14d776a386416bbf3cf70f29e"
    )
    assert (
        execution["resolved_config"]["arm"] == "F2_MP"
        and execution["resolved_config"]["campaign_id"]
        == "coupled_predictive_fusion_mp_v1"
    )
    assert gate["arm"] == "F2_MP" and gate["no_sig"] is True
    assert index["file_count"] == len(index["manifest"])
    for row in index["manifest"]:
        path = MP_ROOT / "source_snapshot/files" / row["path"]
        assert path.is_file(), row["path"]
        assert path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"], (
            row["path"]
        )
    expected = gate["component_sha256"]
    for relative, digest in expected.items():
        path = MP_ROOT / "source_snapshot/files" / relative
        assert path.is_file() and sha256(path) == digest, relative
    current_loader = (ROOT / "src/spica/data/coupled_training.py").read_text(
        encoding="utf-8"
    )
    archived_loader = MP_ARCHIVE_DATA.read_text(encoding="utf-8")
    current_tree, archived_tree = ast.parse(current_loader), ast.parse(archived_loader)
    names = {"make_train_loader", "prepare_batch", "_positive_paths"}

    def selected(tree: ast.Module) -> dict[str, str]:
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        }

    assert selected(current_tree) == selected(archived_tree)
    assert (
        'positive_pairing=None if positive_pool == "full"' in current_loader
        and "seed=42" in current_loader
    )
    return {
        "archive_source_hash": index["sha256"],
        "manifest_files": index["file_count"],
        "component_hashes_verified": len(expected),
        "sampler_ast_exact": True,
    }


def check_metadata_and_eval() -> dict[str, Any]:
    source = (ROOT / "src/spica/data/coupled_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_loader"
    ]
    assert calls and 'positive_pairing=None if positive_pool == "full"' in source
    model = tiny_model(13)
    adapter = QOnlyAdapter(model)
    images_ = images(2)
    expected = model(images_).q
    calls_predictor = 0
    assert model.predictor is not None
    original = model.predictor.forward

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls_predictor
        calls_predictor += 1
        raise AssertionError("q-only adapter called predictor")

    model.predictor.forward = forbidden  # type: ignore[method-assign]
    try:
        actual = adapter(images_)
    finally:
        model.predictor.forward = original  # type: ignore[method-assign]
    assert torch.equal(actual, expected)
    assert calls_predictor == 0
    assert trainer._eval_factory.__defaults__ == (None,)
    selected: list[str | None] = []
    old_factory = trainer._eval_factory
    trainer._eval_factory = lambda *args, **kwargs: (
        selected.append(kwargs.get("query")) or (lambda: {})
    )
    try:
        trainer._make_eval_probe({}, None, None, torch.device("cpu"), "F2_QMP")
        trainer._make_eval_probe({}, None, None, torch.device("cpu"), "F2_MP")
    finally:
        trainer._eval_factory = old_factory
    assert selected == ["q", None]
    baseline = trainer._validate_f2_qmp_readout()
    assert (
        baseline["status"] == "COMPLETE"
        and baseline["step"] == 3600
        and baseline["head"] == "q"
    )
    config = trainer._wandb_config(
        {
            "arm": "F2_QMP",
            "campaign_id": "coupled_predictive_fusion_qmp_v1",
            "main_query": "q",
            "main_photo_objective": "multi_positive_pooled_contrastive",
            "objective_identity": "rank_q_mp=multi_positive_supervised_contrastive_over_unique_live_photobank",
            "loss_coefficient_identity": {"rank_q_mp": 1.0},
            "primary_comparison": {"baseline_arm": "F2_MP"},
        }
    )
    assert (
        config["main_query"] == "q"
        and config["primary_comparison"]["baseline_arm"] == "F2_MP"
    )
    try:
        _arm_protocol("F2_QMP", "coupled_predictive_fusion_qmp_v1", None)
        _arm_protocol("F2_QMP", "coupled_predictive_fusion_qmp_v1", object())
    except ValueError:
        pass
    else:
        raise AssertionError("diagnostic was accepted")
    try:
        coupled_region_loss(
            model, *batch(), lambda_sig=0.0, sigreg=None, main_photo_objective="invalid"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid objective was accepted")
    try:
        trainer._device("not-a-real-device")
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("invalid device was accepted")
    return {
        "full_pool_loader_contract": True,
        "q_forward_parity": True,
        "predictor_calls": calls_predictor,
        "metadata_allowlist": True,
        "baseline_readout_validation": True,
        "invalid_objective_and_device_rejected": True,
    }


class Receipt:
    def __init__(self, root: Path) -> None:
        self.root, self.started = root, time.time()
        self.log = (root / "gate.log").open("w", encoding="utf-8")
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, fn: Callable[[], Any]) -> None:
        started = time.time()
        print(f"START {name}", file=self.log, flush=True)
        try:
            details = fn()
            row = {
                "name": name,
                "status": "PASS",
                "seconds": round(time.time() - started, 6),
                "details": details,
            }
        except Exception as error:  # noqa: BLE001
            row = {
                "name": name,
                "status": "FAIL",
                "seconds": round(time.time() - started, 6),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        self.checks.append(row)
        print(row["status"], name, file=self.log, flush=True)

    def close(self, source: dict[str, Any]) -> dict[str, Any]:
        self.log.close()
        failed = [x for x in self.checks if x["status"] != "PASS"]
        result = {
            "schema_version": 1,
            "status": "FAIL" if failed else "PASS",
            "verified": not failed,
            "gate": "fusion_qmp_cpu",
            "device": "cpu",
            "pretrained_weights": False,
            "network": False,
            "training": False,
            "source_hash": source,
            "checks": self.checks,
            "summary": {
                "pass": len(self.checks) - len(failed),
                "fail": len(failed),
                "skip": 0,
            },
            "log": str(self.root / "gate.log"),
            "started_unix": self.started,
            "finished_unix": time.time(),
        }
        (self.root / "receipt.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        (args.output or ROOT / "outputs" / f"fusion_qmp_cpu_{stamp}")
        .expanduser()
        .resolve()
    )
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    paths = [
        Path(__file__),
        ROOT / "src/spica/coupled_predictive_losses.py",
        ROOT / "src/spica/data/coupled_training.py",
        ROOT / "src/spica/train_coupled_predictive.py",
        ROOT / "src/spica/evaluation/coupled_predictive.py",
        ROOT / "src/spica/models/coupled_predictive.py",
        ROOT / "src/spica/models/clip.py",
        ARCHIVE_LOSS,
        MP_LOSS,
        MP_ARCHIVE_MODEL,
        MP_ARCHIVE_DATA,
        MP_ARCHIVE_EVAL,
        MP_ARCHIVE_TRAINER,
        MP_ARCHIVE_LAUNCHER,
        MP_MANIFEST,
        MP_EXECUTION,
        MP_GATE,
        ROOT / "scripts/run_fusion_multipositive.py",
    ]
    source = {
        "files": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in paths
        ]
    }
    receipt = Receipt(output)
    receipt.check(
        "F2_MP archive bindings and sampler source exactness", check_archive_bindings
    )
    receipt.check("F2_QMP formula and q/photo-bank gradients", check_formula)
    receipt.check(
        "F2_QMP production coefficients, head gradients, optimizer, safe restore",
        lambda: check_loss_and_step(output),
    )
    receipt.check("legacy default scalar and gradient parity", check_legacy)
    receipt.check("source-bound routing before device/data", check_routing)
    receipt.check(
        "full-pool contract, q-only eval, metadata and invalid objective",
        check_metadata_and_eval,
    )
    final = receipt.close(source)
    print(f"CPU gate {final['status']}; receipt: {output / 'receipt.json'}")
    return 0 if final["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
