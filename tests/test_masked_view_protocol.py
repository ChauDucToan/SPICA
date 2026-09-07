import ast
import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from spica.frozen_prompt_artifacts import (
    MASKED_VIEW_CAMPAIGN,
    MASKED_VIEW_ROLES,
    make_manifest,
)
from spica.train_frozen_prompt import _run_impl, _validate

ROOT = Path(__file__).parents[1]


def config(role: str):
    name = "masked_view_C" if role.endswith("_C") else "masked_view_M"
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train_frozen_prompt", overrides=[f"+experiments={name}", "device=cpu"])


@pytest.mark.parametrize("role", MASKED_VIEW_ROLES)
def test_new_masked_view_configs_compose_and_validate(role: str) -> None:
    args = config(role)
    _validate(args)
    assert args.experiment_campaign == MASKED_VIEW_CAMPAIGN
    assert args.batch_size == 32
    assert args.mask_policy.version == "ink_centered_square_v1"
    assert args.pairing_manifest_path.endswith("sketchy_pseudo_train_pairing.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [("sketch_view_mode", "full_masked"), ("seed", 43), ("max_steps", 1799), ("resume_checkpoint_path", "x.pt")],
)
def test_masked_view_mutations_fail_closed(field: str, value) -> None:
    args = config("frozen_prompt_masked_view_C")
    args[field] = value
    with pytest.raises(ValueError):
        _validate(args)


def test_masked_view_manifest_binds_view_mode_per_role() -> None:
    manifest = make_manifest(
        dataset="toy",
        data_config="toy.yaml",
        campaign=MASKED_VIEW_CAMPAIGN,
        positive_sampling="same_class",
        pairing_manifest_sha256="pairing-sha",
        run_kind="primary",
        selection_target_step=1800,
    )
    assert "sketch_view_mode" not in manifest
    assert manifest["entries"]["frozen_prompt_masked_view_C"]["sketch_view_mode"] == "full_full"
    assert manifest["entries"]["frozen_prompt_masked_view_M"]["sketch_view_mode"] == "full_masked"


def test_run_keeps_campaign_flag_separate_from_mask_tensor() -> None:
    tree = ast.parse(inspect.getsource(_run_impl))
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assigned_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert "is_masked_campaign" in assigned_names
    assert "masked_images" in assigned_names
    assert "masked_view" not in loaded_names


def test_masked_view_smoke_requires_two_item_batch() -> None:
    args = config("frozen_prompt_masked_view_M")
    args.run_kind = "smoke"
    args.max_steps = 2
    args.probe_steps = [0, 1, 2]
    args.batch_size = 32
    with pytest.raises(ValueError, match="batch size"):
        _validate(args)
    args.batch_size = 2
    _validate(args)
