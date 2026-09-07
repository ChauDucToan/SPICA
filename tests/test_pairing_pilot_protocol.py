from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
import torch
from torch import nn

from spica.frozen_prompt_artifacts import (
    PAIRING_PILOT_CAMPAIGN,
    PAIRING_PILOT_ROLES,
    ensure_manifest,
)
from spica.models.clip import FrozenClipEncoder
from spica.train_frozen_prompt import _assert_clip_policy, _clip_snapshot, _validate

ROOT = Path(__file__).parents[1]


def _config(role: str):
    experiment = {
        "frozen_prompt_pairing_pilot_A": "pairing_pilot_A",
        "frozen_prompt_pairing_pilot_B": "pairing_pilot_B",
    }[role]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(
            config_name="train_frozen_prompt",
            overrides=[f"+experiments={experiment}", "device=cpu"],
        )


@pytest.mark.parametrize("role", PAIRING_PILOT_ROLES)
def test_pairing_pilot_config_validates_on_cpu(role: str) -> None:
    args = _config(role)
    _validate(args)
    assert str(args.device) == "cpu"
    assert args.experiment_role == role
    assert "experiments" not in args
    assert int(args.max_steps) == 1800
    assert tuple(args.probe_steps)[-1] == 1800


@pytest.mark.parametrize(
    ("role", "field", "value"),
    [
        (PAIRING_PILOT_ROLES[0], "positive_sampling", "paired"),
        (PAIRING_PILOT_ROLES[1], "positive_sampling", "same_class"),
        (PAIRING_PILOT_ROLES[0], "pairing_manifest_path", None),
        (PAIRING_PILOT_ROLES[1], "experiment_campaign", "frozen_prompt_final_2026-09-04"),
        (PAIRING_PILOT_ROLES[0], "max_steps", 1799),
    ],
)
def test_pairing_pilot_mutations_fail_closed(role: str, field: str, value) -> None:
    args = _config(role)
    args[field] = value
    with pytest.raises(ValueError):
        _validate(args)


@pytest.mark.parametrize("run_kind", ["primary", "smoke"])
@pytest.mark.parametrize("role", PAIRING_PILOT_ROLES)
def test_pairing_pilot_rejects_resume_for_primary_and_smoke(
    role: str, run_kind: str
) -> None:
    args = _config(role)
    args.run_kind = run_kind
    args.resume_checkpoint_path = "outputs/old/checkpoint.pt"
    if run_kind == "smoke":
        args.max_steps = 2
        args.probe_steps = [0, 1, 2]
    with pytest.raises(ValueError, match="train from scratch"):
        _validate(args)


class _SmallClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Linear(2, 2)
        self.text = nn.Linear(2, 2)
        self.visual_projection = nn.Parameter(torch.ones(2, 2))
        self.logit_scale = nn.Parameter(torch.ones(()))


@pytest.mark.parametrize(
    "parameter_name", [
        "visual.weight",
        "text.weight",
        "visual_projection",
        "logit_scale",
    ]
)
def test_pairing_clip_policy_covers_all_frozen_clip_parameters(
    parameter_name: str,
) -> None:
    encoder = FrozenClipEncoder(_SmallClip())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    before = _clip_snapshot(encoder.model, all_parameters=True)
    assert set(before) == set(dict(encoder.model.named_parameters()))
    with torch.no_grad():
        dict(encoder.model.named_parameters())[parameter_name].add_(1)
    with pytest.raises(RuntimeError, match=parameter_name):
        _assert_clip_policy(
            encoder.model,
            before,
            PAIRING_PILOT_ROLES[0],
        )


def test_pairing_manifest_identity_is_campaign_and_role_specific(tmp_path: Path) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    a, sha_a = ensure_manifest(
        path_a,
        dataset="toy",
        data_config="toy.yaml",
        campaign=PAIRING_PILOT_CAMPAIGN,
        positive_sampling="same_class",
        pairing_manifest_sha256="pairing-sha",
    )
    b, sha_b = ensure_manifest(
        path_b,
        dataset="toy",
        data_config="toy.yaml",
        campaign=PAIRING_PILOT_CAMPAIGN,
        positive_sampling="paired",
        pairing_manifest_sha256="pairing-sha",
    )
    assert a["campaign"] == b["campaign"] == PAIRING_PILOT_CAMPAIGN
    assert a["entries"][PAIRING_PILOT_ROLES[0]]["positive_sampling"] == "same_class"
    assert b["entries"][PAIRING_PILOT_ROLES[1]]["positive_sampling"] == "paired"
    assert sha_a != sha_b
