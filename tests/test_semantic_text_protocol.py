from hydra import compose, initialize_config_dir
import pytest

from spica.frozen_prompt_artifacts import (
    MASKED_VIEW_3600_CAMPAIGN,
    SEMANTIC_TEXT_CAMPAIGN,
    SEMANTIC_TEXT_ROLES,
    SEMANTIC_TEXT_SELECTION_STEPS,
    is_masked_view_campaign,
    is_retrieval_probe_campaign,
    make_manifest,
    treatment_for_role,
)
from spica.train_frozen_prompt import _validate

ROOT = __import__("pathlib").Path(__file__).parents[1]


def test_semantic_campaign_is_probe_not_training_mask_campaign() -> None:
    assert is_retrieval_probe_campaign(MASKED_VIEW_3600_CAMPAIGN)
    assert is_retrieval_probe_campaign(SEMANTIC_TEXT_CAMPAIGN)
    assert not is_masked_view_campaign(SEMANTIC_TEXT_CAMPAIGN)


def test_semantic_manifest_and_treatments_are_counterfactual_roles() -> None:
    manifest = make_manifest(
        dataset="toy",
        data_config="toy.yaml",
        campaign=SEMANTIC_TEXT_CAMPAIGN,
        positive_sampling="same_class",
        pairing_manifest_sha256="pairing-sha",
    )
    assert tuple(manifest["entries"]) == SEMANTIC_TEXT_ROLES
    assert manifest["selection_policy"]["candidate_steps"] == list(SEMANTIC_TEXT_SELECTION_STEPS)
    assert manifest["selection_policy"]["step_zero_selectable"] is False
    assert treatment_for_role("semantic_text_S0")["lambda_anchor"] == 0.0
    assert treatment_for_role("semantic_text_S2")["lambda_anchor"] == 1.0
    assert treatment_for_role("semantic_text_S0")["sketch_view_mode"] == "full_full"


def _config(role: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(
            config_name="train_frozen_prompt",
            overrides=[f"+experiments={role}", "device=cpu"],
        )


@pytest.mark.parametrize("role", SEMANTIC_TEXT_ROLES)
def test_semantic_primary_config_is_exact_and_fail_closed(role: str) -> None:
    args = _config(role)
    _validate(args)
    assert args.data_config == "configs/data/sketchy_104_21.yaml"
    assert args.pretrained == "openai"
    assert args.prompt_template == "a photo of a {}"
    assert args.pseudo_val_num_classes == 20
    assert args.batch_size == 32
    assert args.num_workers == 4
    assert args.eval_batch_size == 256
    assert args.query_chunk_size == 256
    assert args.margin == 0.2
    assert args.tau_cls == 0.07
    assert args.visual_prompt_learning_rate == 1.0e-3
    assert args.soft_prompt_learning_rate == 1.0e-3
    assert args.visual_prompt_weight_decay == 1.0e-4
    assert args.soft_prompt_weight_decay == 1.0e-4


@pytest.mark.parametrize("field", ["data_config", "pretrained", "prompt_template", "margin"])
def test_semantic_primary_protocol_mutations_fail(field: str) -> None:
    args = _config("semantic_text_S1")
    args[field] = {
        "data_config": "configs/data/other.yaml",
        "pretrained": None,
        "prompt_template": "a sketch of a {}",
        "margin": 0.3,
    }[field]
    with pytest.raises(ValueError):
        _validate(args)
