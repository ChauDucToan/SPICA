#!/usr/bin/env python3
"""Independent synthetic CPU contract gate for the benchmark trainer.

This gate never loads official data, images, pretrained weights, W&B, or the
network.  Its synthetic provenance is explicitly unverified and makes no
production claim.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Iterator

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

import spica.train_coupled_benchmark as trainer  # noqa: E402
from spica.coupled_predictive_losses import coupled_region_loss  # noqa: E402
from spica.models.clip import FrozenClipBundle, FrozenClipEncoder  # noqa: E402
from spica.models.coupled_predictive import CoupledPredictiveModel  # noqa: E402

OUT = ROOT / "outputs/coupled_benchmark_independent_cpu_20260910"
TEMP = 0.07


def tiny_model(seed: int = 42) -> CoupledPredictiveModel:
    torch.manual_seed(seed)
    clip = CLIP(
        embed_dim=6,
        vision_cfg=CLIPVisionCfg(layers=1, width=8, head_width=4, patch_size=4, image_size=8),
        text_cfg=CLIPTextCfg(context_length=77, vocab_size=49408, width=8, heads=2, layers=1),
    ).float().cpu()
    bundle = FrozenClipBundle(
        encoder=FrozenClipEncoder(clip), transform=lambda value: value,
        tokenizer=open_clip.get_tokenizer("ViT-B-32"), model_name="tiny_cpu", pretrained=None,
    )
    return CoupledPredictiveModel(
        bundle.encoder, bundle.tokenizer, {0: "cat", 1: "dog"},
        architecture="predictive_fusion_v2", photo_prompt_length=3,
        text_prompt_length=4, predictor_width=8, predictor_heads=2,
    ).cpu()


def images(count: int, seed: int) -> Tensor:
    return torch.randn(count, 3, 8, 8, generator=torch.Generator().manual_seed(seed))


def synthetic_args() -> tuple[Tensor, ...]:
    return (
        images(2, 101), images(2, 102), images(6, 103),
        torch.tensor([0, 1]), torch.tensor([[2], [3]]),
        torch.tensor([0, 1]), torch.tensor([0, 0, 0, 1, 1, 1]),
        ["p0", "p1", "p2", "p3", "p4", "p5"],
    )


def active(model: nn.Module) -> dict[str, nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_routing_and_schedule() -> dict[str, Any]:
    assert trainer._validate_routing("tuberlin_220_30")["main_query"] == "mu_i"
    assert trainer._validate_routing("quickdraw_80_30")["evaluation_query"] == "q"
    try:
        trainer._validate_routing("tuberlin_220_30", main_photo_objective="multi_positive_pooled_contrastive")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid routing was accepted")
    assert trainer._schedule(1189, 59)(0) == 1 / 59
    assert trainer._schedule(18229, 911)(0) == 1 / 911
    assert trainer.DATASETS["tuberlin_220_30"] == {"config": "configs/data/tuberlin_220_30.yaml", "total_steps": 1189, "warmup_steps": 59}
    assert trainer.DATASETS["quickdraw_80_30"] == {"config": "configs/data/quickdraw_80_30.yaml", "total_steps": 18229, "warmup_steps": 911}
    return {"wrong_routing_rejected_before_device": True, "legacy_schedule_defaults_unchanged": True}


def _archived_loss() -> Any:
    path = ROOT / "outputs/fusion_mp_execution_20260909T115500Z/runs/F2_MP/source_snapshot/files/src/spica/coupled_predictive_losses.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("spica._archived_f2_mp_loss", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "spica"
    spec.loader.exec_module(module)
    return module


def check_archived_gradients() -> dict[str, Any]:
    current = tiny_model(7)
    archived = tiny_model(7)
    args = synthetic_args()
    kwargs = dict(lambda_sig=0.0, sigreg=None, main_photo_objective=trainer.MAIN_OBJECTIVE)
    current_loss = coupled_region_loss(current, *args, **kwargs)["total"]
    archived_loss = _archived_loss().coupled_region_loss(archived, *args, **kwargs)["total"]
    torch.testing.assert_close(current_loss, archived_loss, rtol=0.0, atol=0.0)
    current_loss.backward()
    archived_loss.backward()
    left, right = active(current), active(archived)
    assert left.keys() == right.keys() and left
    compared = 0
    for name in left:
        assert left[name].grad is not None and right[name].grad is not None
        torch.testing.assert_close(left[name].grad, right[name].grad, rtol=0.0, atol=0.0)
        assert bool(torch.isfinite(left[name].grad).all()) and float(left[name].grad.abs().sum()) > 0
        compared += 1
    original = {n: v.detach().clone() for n, v in current.original_clip.state_dict().items()}
    assert all(torch.equal(v, current.original_clip.state_dict()[n]) for n, v in original.items())
    return {"archived_f2_mp_gradient_parity": True, "active_parameters_compared": compared, "loss": float(current_loss)}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@contextmanager
def safe_numpy_globals() -> Iterator[None]:
    # The scope is deliberately narrow; there is no unsafe torch.load fallback.
    from torch.serialization import safe_globals
    globals_ = [
        np._core.multiarray._reconstruct, np._core.multiarray.scalar,
        np.ndarray, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.uint32)),
    ]
    with safe_globals(globals_):
        yield


def check_exact_checkpoint(path: Path) -> dict[str, Any]:
    with safe_numpy_globals():
        payload = torch.load(path, map_location="cpu", weights_only=True)
    restored = tiny_model(42)
    full_state = dict(restored.state_dict())
    full_state.update(payload["model_state_dict"])
    result = restored.load_state_dict(full_state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    restored_optimizer = trainer._make_optimizer(restored)
    restored_scheduler = LambdaLR(restored_optimizer, lr_lambda=trainer._schedule(1189, 59))
    restored_optimizer.load_state_dict(payload["optimizer_state_dict"])
    restored_scheduler.load_state_dict(payload["scheduler_state_dict"])
    assert _same(payload["optimizer_state_dict"], restored_optimizer.state_dict())
    assert _same(payload["scheduler_state_dict"], restored_scheduler.state_dict())
    assert all(group.keys() == other.keys() and group["lr"] == other["lr"] and group["weight_decay"] == other["weight_decay"]
               for group, other in zip(payload["optimizer_state_dict"]["param_groups"], restored_optimizer.state_dict()["param_groups"]))
    return {"weights_only": True, "model_load_state_dict_strict": True, "optimizer_state_exact": True, "scheduler_state_exact": True}


@dataclass
class _Loader:
    batch: tuple[Tensor, ...]
    generator: torch.Generator
    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        while True:
            yield self.batch


def _protocol() -> Any:
    train = SimpleNamespace(class_names={0: "cat", 1: "dog"}, class_ids=(0, 1))
    return SimpleNamespace(config_path="synthetic.yaml", config_sha256="synthetic-config", train=train,
                           identity={"provenance": "SYNTHETIC_UNVERIFIED", "photos": 6})


def check_mocked_train_impl() -> dict[str, Any]:
    created: list[CoupledPredictiveModel] = []
    base_batch = synthetic_args()
    batch = (
        base_batch[0].repeat(16, 1, 1, 1), base_batch[1].repeat(16, 1, 1, 1), base_batch[2],
        base_batch[3].repeat(16), base_batch[4].repeat(16, 1), base_batch[5].repeat(16),
        base_batch[6], base_batch[7],
    )
    loader = _Loader(batch, torch.Generator().manual_seed(42))
    synthetic_clip = tiny_model(42).original_clip
    bundle = FrozenClipBundle(FrozenClipEncoder(synthetic_clip), lambda x: x,
                              open_clip.get_tokenizer("ViT-B-32"), "tiny_cpu", None)
    original_module = trainer.benchmark_data
    original = {
        "verify_clip": trainer._verify_clip,
        "load_clip": trainer.load_frozen_clip,
        "capture": trainer.capture_provenance,
        "verify_source": trainer._verify_source,
        "copy_archive": trainer._copy_source_archive,
        "device": trainer._device,
        "model_class": trainer.CoupledPredictiveModel,
    }
    def make_model(*args: Any, **kwargs: Any) -> CoupledPredictiveModel:
        model = tiny_model(42)
        created.append(model)
        return model
    def no_official(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("official benchmark loader was called")
    synthetic_module = SimpleNamespace(
        load_benchmark_protocol=lambda *a, **k: _protocol(),
        make_train_loader=lambda *a, **k: loader,
        prepare_batch=lambda protocol, raw, step: {
            "clean": raw[0], "corrupted": raw[1], "photos": raw[2], "positive_indices": raw[3],
            "negative_indices": raw[4], "labels": raw[5], "photo_labels": raw[6], "photo_ids": raw[7],
            "trace": [{"sample": i} for i in range(32)],
            "mask_metadata": {"rows": [{"sample": i, "masked": True} for i in range(32)]},
        },
    )
    trainer.benchmark_data = synthetic_module
    trainer._verify_clip = lambda: {"path": "synthetic", "sha256": "synthetic"}
    trainer.load_frozen_clip = lambda **kwargs: bundle
    trainer.CoupledPredictiveModel = make_model
    trainer.capture_provenance = lambda *a, **k: {"source_snapshot": {"manifest": []}, "resolved_config": k.get("resolved_config", {})}
    trainer._verify_source = lambda *a, **k: "synthetic-source-UNVERIFIED"
    trainer._copy_source_archive = lambda *a, **k: None
    trainer._device = lambda value: torch.device("cpu")
    import spica.data.coupled_benchmark as official
    old_official_loader = official.make_train_loader
    old_open = None
    try:
        official.make_train_loader = no_official
        try:
            from PIL import Image
            old_open = Image.open
            Image.open = no_official
        except ImportError:
            pass
        with tempfile.TemporaryDirectory(prefix="spica-benchmark-smoke-") as directory:
            args = SimpleNamespace(dataset="tuberlin_220_30", output_dir=directory,
                                   campaign_root=directory, device="cpu", wandb_mode="disabled", smoke=True)
            before = trainer._state_hash(created[0], prefix="original_clip.") if created else None
            result = trainer._train_impl(args, Path(directory))
            assert result["step"] == 2 and result["completed_steps"] == 2
            assert result["horizon_steps"] == 1189 and result["warmup_steps"] == 59
            assert before is None or before == trainer._state_hash(created[0], prefix="original_clip.")
            trace = Path(directory) / "observation_trace.jsonl"
            masks = Path(directory) / "mask_metadata.jsonl"
            assert sum(1 for _ in trace.open()) == 64
            assert sum(1 for _ in masks.open()) == 64
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            assert all("step" in row and "actual_lr" in row for row in rows)
            checkpoint = Path(directory) / "checkpoint_step2.pt"
            exact = check_exact_checkpoint(checkpoint)
            receipt = {"status": "PASS", "source_provenance": "SYNTHETIC_UNVERIFIED",
                       "steps_executed": 2, "trace_rows": 64, "mask_rows": 64, **exact}
            (OUT / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            return receipt
    finally:
        official.make_train_loader = old_official_loader
        if old_open is not None:
            from PIL import Image
            Image.open = old_open
        trainer.benchmark_data = original_module
        for name, value in original.items():
            setattr(trainer, {"verify_clip": "_verify_clip", "load_clip": "load_frozen_clip", "capture": "capture_provenance", "verify_source": "_verify_source", "copy_archive": "_copy_source_archive", "device": "_device", "model_class": "CoupledPredictiveModel"}[name], value)


def main() -> dict[str, Any]:
    result = {
        "status": "PASS",
        "scope": "independent synthetic CPU checker; no production identity claim",
        "source_provenance": "SYNTHETIC_UNVERIFIED",
        "source_hashes": {name: sha256(path) for name, path in {
            "checker": Path(__file__), "trainer": ROOT / "src/spica/train_coupled_benchmark.py",
            "archived_f2_mp_loss": ROOT / "outputs/fusion_mp_execution_20260909T115500Z/runs/F2_MP/source_snapshot/files/src/spica/coupled_predictive_losses.py",
        }.items()},
        "routing_schedule": check_routing_and_schedule(),
        "archived_gradients": check_archived_gradients(),
        "mocked_train_impl": check_mocked_train_impl(),
    }
    (OUT / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
