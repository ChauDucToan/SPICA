"""Regression test: diagnose_alignment_geometry must route photos through encode_photo."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class _FakeDataset(Dataset):
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": torch.randn(3, 224, 224),
            "label": torch.tensor(0),
            "path": f"fake_{idx}.jpg",
        }


class _RoutingTracker:
    """Fake model that records which branch was called."""

    def __init__(self) -> None:
        self.sketch_calls = 0
        self.photo_calls = 0
        self.device = torch.device("cpu")

    def eval(self) -> None:
        pass

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        self.sketch_calls += images.shape[0]
        return torch.randn(images.shape[0], 512)

    def encode_photo(self, images: torch.Tensor) -> torch.Tensor:
        self.photo_calls += images.shape[0]
        return torch.randn(images.shape[0], 512) * 10  # distinct scale


def test_photo_routing_distinct_outputs() -> None:
    """Photos must go through encode_photo, sketches through __call__."""
    from spica.evaluation.frozen_prompt import encode_prompted_loader

    tracker = _RoutingTracker()
    loader = DataLoader(_FakeDataset(8), batch_size=4, shuffle=False)

    sketch_result = encode_prompted_loader(tracker, loader, photo=False)
    assert tracker.sketch_calls == 8
    assert tracker.photo_calls == 0

    tracker.sketch_calls = 0
    photo_result = encode_prompted_loader(tracker, loader, photo=True)
    assert tracker.sketch_calls == 0
    assert tracker.photo_calls == 8

    # Outputs should differ because encode_photo returns 10x scale
    assert not torch.allclose(
        sketch_result.embeddings.abs().mean(),
        photo_result.embeddings.abs().mean(),
        atol=1.0,
    )


def test_diagnose_encode_helper_passes_photo_flag() -> None:
    """The _encode helper in diagnose script must forward the photo kwarg.

    We verify by inspecting the source: _encode signature has photo kwarg
    and passes it to encode_prompted_loader.
    """
    import inspect
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "diagnose_mod",
        Path(__file__).resolve().parents[1] / "scripts" / "diagnose_alignment_geometry.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    sig = inspect.signature(mod._encode)
    assert "photo" in sig.parameters, "_encode must accept photo kwarg"

    source = inspect.getsource(mod._encode)
    assert "photo=photo" in source, "_encode must forward photo to encode_prompted_loader"


if __name__ == "__main__":
    test_photo_routing_distinct_outputs()
    test_diagnose_encode_helper_passes_photo_flag()
    print("PASS: photo routing regression tests")
