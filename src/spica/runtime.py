"""Shared run I/O policy, independent of model/loss configuration."""
from pathlib import Path
from collections.abc import Mapping

from omegaconf import OmegaConf


def runtime_policy(overrides: Mapping | None = None) -> dict[str, int]:
    overrides = dict(overrides or {})
    if "probe_every" in overrides and "test_every" in overrides:
        raise ValueError("probe_every and test_every cannot be used together")
    profile = "official_test.yaml" if "test_every" in overrides else "minimal.yaml"
    path = Path(__file__).resolve().parents[2] / "configs/runtime" / profile
    values = dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"unknown runtime options: {sorted(unknown)}")
    values.update(overrides)
    if any(type(v) is not int or v <= 0 for v in values.values()):
        raise ValueError("runtime intervals must be positive integers")
    return values
