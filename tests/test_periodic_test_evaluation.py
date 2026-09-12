from __future__ import annotations

from types import SimpleNamespace
import random

import numpy as np
import torch

from spica.evaluation import periodic_test
from spica.evaluation.coupled_benchmark import _fixture_loaders, module_state_hash


def test_periodic_metrics_only_matches_full_without_large_artifacts(monkeypatch, tmp_path):
    model, clean, gallery, masked = _fixture_loaders()
    entries = tuple(range(32))
    protocol = SimpleNamespace(
        name="synthetic_fixture",
        root=tmp_path,
        identity={"fixture": True},
        train=SimpleNamespace(class_names={i: f"class_{i}" for i in range(8)}),
        test=SimpleNamespace(sketch_entries=entries),
    )
    monkeypatch.setattr(
        periodic_test.benchmark_data,
        "build_test_loaders",
        lambda protocol, transform, batch_size, num_workers: {"sketch": clean, "photo": gallery},
    )
    monkeypatch.setattr(periodic_test, "_masked_loader", lambda *args, **kwargs: masked(*[kwargs[k] for k in ("fraction", "seed")]))
    monkeypatch.setattr(periodic_test, "_transform_stats", lambda transform: ((0.0,) * 3, (1.0,) * 3))

    model.train()
    model.predictor.eval()
    before_state = module_state_hash(model)
    before_modes = {module: module.training for module in model.modules()}
    before_python_rng = random.getstate()
    before_numpy_rng = np.random.get_state()
    before_rng = torch.random.get_rng_state()
    periodic = periodic_test.evaluate_periodic_test(
        model, protocol, lambda image: image, tmp_path / "periodic",
        device="cpu", source_hash="source", checkpoint_selection={"step": 100},
    )
    full = periodic_test.evaluate_periodic_test(
        model, protocol, lambda image: image, tmp_path / "final",
        device="cpu", source_hash="source", checkpoint_selection={"step": 100}, final=True,
    )

    assert periodic["artifact_policy"] == "metrics_only"
    assert periodic["evaluation_scope"] == "official_unseen_periodic_monitoring_no_selection"
    assert full["artifact_policy"] == "full"
    assert full["evaluation_scope"] == "official_unseen_final"
    assert periodic["clean"] == full["clean"]
    assert periodic["masked_macro"] == full["masked_macro"]
    assert len(periodic["conditions"]) == 10
    assert not list((tmp_path / "periodic").glob("*.npy"))
    assert not list((tmp_path / "periodic").glob("*.npz"))
    assert not list((tmp_path / "periodic").glob("*mask_metadata.jsonl"))
    assert (tmp_path / "final" / "gallery_embeddings.npy").exists()
    assert (tmp_path / "final" / "mask_f25_s101_mask_metadata.jsonl").exists()
    assert module_state_hash(model) == before_state
    assert {module: module.training for module in model.modules()} == before_modes
    assert random.getstate() == before_python_rng
    after_numpy_rng = np.random.get_state()
    assert after_numpy_rng[0] == before_numpy_rng[0]
    assert np.array_equal(after_numpy_rng[1], before_numpy_rng[1])
    assert after_numpy_rng[2:] == before_numpy_rng[2:]
    assert torch.equal(torch.random.get_rng_state(), before_rng)


def test_periodic_uses_full_test_loader_contract(monkeypatch, tmp_path):
    model, clean, gallery, masked = _fixture_loaders()
    calls = []
    protocol = SimpleNamespace(
        name="synthetic_fixture", root=tmp_path, identity={"fixture": True},
        train=SimpleNamespace(class_names={i: f"class_{i}" for i in range(8)}),
        test=SimpleNamespace(sketch_entries=tuple(range(32))),
    )

    def build(protocol, transform, batch_size, num_workers):
        calls.append((batch_size, num_workers))
        return {"sketch": clean, "photo": gallery}

    monkeypatch.setattr(periodic_test.benchmark_data, "build_test_loaders", build)
    monkeypatch.setattr(periodic_test, "_masked_loader", lambda *args, **kwargs: masked(kwargs["fraction"], kwargs["seed"]))
    monkeypatch.setattr(periodic_test, "_transform_stats", lambda transform: ((0.0,) * 3, (1.0,) * 3))
    periodic_test.evaluate_periodic_test(
        model, protocol, lambda image: image, tmp_path / "eval",
        device="cpu", source_hash="source", checkpoint_selection={"step": 1},
    )
    assert calls == [(256, 0)]
