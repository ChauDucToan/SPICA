"""Shared run I/O policy, independent of model/loss configuration."""
from pathlib import Path
from collections.abc import Mapping

from omegaconf import OmegaConf


def official_test_steps(total_steps: int, every_percent: int = 20) -> list[int]:
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if type(every_percent) is not int or every_percent <= 0 or 100 % every_percent:
        raise ValueError("every_percent must be a positive divisor of 100")
    steps: list[int] = []
    for percent in range(every_percent, 101, every_percent):
        step = (total_steps * percent + 99) // 100
        if not steps or step != steps[-1]:
            steps.append(step)
    return steps


def runtime_policy(overrides: Mapping | None = None) -> dict[str, int]:
    overrides = dict(overrides or {})
    if "probe_every" in overrides and "test_every_percent" in overrides:
        raise ValueError("probe_every and test_every_percent cannot be used together")
    profile = "official_test.yaml" if "test_every_percent" in overrides else "minimal.yaml"
    path = Path(__file__).resolve().parents[2] / "configs/runtime" / profile
    values = dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"unknown runtime options: {sorted(unknown)}")
    values.update(overrides)
    if any(type(v) is not int or v <= 0 for v in values.values()):
        raise ValueError("runtime intervals must be positive integers")
    if "test_every_percent" in values and values["test_every_percent"] != 20:
        raise ValueError("official test cadence must use test_every_percent=20")
    return values
