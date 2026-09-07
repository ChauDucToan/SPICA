from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from spica.frozen_prompt_artifacts import (
    MASKED_VIEW_3600_CAMPAIGN,
    MASKED_VIEW_3600_ROLES,
    MASKED_VIEW_3600_SELECTION_STEPS,
    MASKED_VIEW_3600_STEPS,
    make_manifest,
)
from spica.train_frozen_prompt import _select_masked_view_3600, _validate

ROOT = Path(__file__).parents[1]


def config(role: str):
    name = "masked_view_3600_C" if role.endswith("_C") else "masked_view_3600_M"
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train_frozen_prompt", overrides=[f"+experiments={name}", "device=cpu"])


@pytest.mark.parametrize("role", MASKED_VIEW_3600_ROLES)
def test_3600_configs_are_strict_and_from_scratch(role: str) -> None:
    args = config(role)
    _validate(args)
    assert args.experiment_campaign == MASKED_VIEW_3600_CAMPAIGN
    assert args.experiment_role == role
    assert tuple(args.probe_steps) == MASKED_VIEW_3600_STEPS
    assert args.max_steps == 3600
    assert args.resume_checkpoint_path is None
    assert args.batch_size == 32
    assert args.num_workers == 4
    assert args.text_mode == "hard"
    assert args.lambda_rank == 1.0
    assert args.lambda_cls == 1.0


def test_3600_manifest_has_explicit_selection_protocol() -> None:
    manifest = make_manifest(
        dataset="toy",
        data_config="toy.yaml",
        campaign=MASKED_VIEW_3600_CAMPAIGN,
        positive_sampling="same_class",
        pairing_manifest_sha256="pairing-sha",
        run_kind="primary",
        selection_target_step=3600,
    )
    assert manifest["selection_metric"] == "mAP@200_prefix_positive"
    assert manifest["selection_policy"]["candidate_steps"] == list(MASKED_VIEW_3600_SELECTION_STEPS)
    assert manifest["selection_policy"]["step_zero_selectable"] is False
    assert set(manifest["entries"]) == set(MASKED_VIEW_3600_ROLES)


def _row(step: int, clean: float, masked: float, clean_p: float = 0.1, masked_p: float = 0.2):
    return {
        "training_global_step": step,
        "checkpoint": f"step{step}.pt",
        "checkpoint_sha256": f"sha{step}",
        "masked_view": {
            "clean": {"mAP@200_prefix_positive": clean, "P@200": clean_p},
            "masked_macro": {"mAP@200_prefix_positive": masked, "P@200": masked_p},
        },
    }


def test_3600_selection_is_separate_stable_and_excludes_step_zero() -> None:
    rows = [_row(0, 99.0, 99.0)] + [_row(step, 0.5, 0.5) for step in MASKED_VIEW_3600_SELECTION_STEPS]
    rows[1]["masked_view"]["clean"]["mAP@200_prefix_positive"] = 0.8
    rows[2]["masked_view"]["clean"]["mAP@200_prefix_positive"] = 0.8
    rows[-1]["masked_view"]["masked_macro"]["mAP@200_prefix_positive"] = 0.9
    selected = _select_masked_view_3600(rows)
    assert selected["best_clean"]["training_global_step"] == 600
    assert selected["best_masked"]["training_global_step"] == 3600
    assert selected["best_clean"]["secondary_P@200"] == 0.1


def test_3600_selection_rejects_nonfinite_or_missing_probe() -> None:
    rows = [_row(step, 0.5, 0.5) for step in MASKED_VIEW_3600_SELECTION_STEPS]
    rows[-1]["masked_view"]["clean"]["mAP@200_prefix_positive"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _select_masked_view_3600(rows)
    with pytest.raises(RuntimeError, match="every nonzero probe"):
        _select_masked_view_3600(rows[:-1])
