import pytest
import torch

from spica.evaluation.masked_view import validate_semantic_text_checkpoint
from spica.frozen_prompt_artifacts import SEMANTIC_TEXT_CAMPAIGN, SEMANTIC_TEXT_SELECTION_STEPS
from spica.semantic_text import semantic_text_identity, tensor_sha256
from spica.train_frozen_prompt import _select_masked_view_3600


def _history():
    return [
        {
            "training_global_step": step,
            "checkpoint": f"step{step}.pt",
            "checkpoint_sha256": f"sha{step}",
            "masked_view": {
                "clean": {"mAP@200_prefix_positive": 0.5, "P@200": 0.5, "full_mAP": 0.5},
                "masked_macro": {"mAP@200_prefix_positive": step / 1000, "P@200": step / 1000, "full_mAP": step / 1000},
            },
        }
        for step in (0, *SEMANTIC_TEXT_SELECTION_STEPS)
    ]


def test_semantic_selection_excludes_step_zero_and_keeps_earliest_tie() -> None:
    selected = _select_masked_view_3600(_history(), candidate_steps=SEMANTIC_TEXT_SELECTION_STEPS)
    assert selected["best_clean"]["training_global_step"] == 600
    assert selected["best_masked"]["training_global_step"] == 3600


def test_semantic_selection_rejects_missing_probe() -> None:
    with pytest.raises(RuntimeError, match="every nonzero"):
        _select_masked_view_3600(_history()[:-1], candidate_steps=SEMANTIC_TEXT_SELECTION_STEPS)


def test_semantic_checkpoint_metadata_is_fail_closed() -> None:
    bank = torch.zeros(84, 512)
    class_ids = list(range(84))
    identity = semantic_text_identity(
        class_ids=class_ids,
        class_names=[f"class_{value}" for value in class_ids],
        token_ids=torch.zeros(84, 8, dtype=torch.long),
        eot_positions=[5] * 84,
        model_name="ViT-B-32-quickgelu", pretrained="openai", fixed_bank=bank,
        initial_context=torch.zeros(4, 768), initial_learned_bank=bank, lambda_anchor=0.0,
    )
    digest = tensor_sha256(bank)
    payload = {
        "campaign": SEMANTIC_TEXT_CAMPAIGN, "experiment_role": "semantic_text_S1",
        "lambda_anchor": 0.0, "semantic_text_identity": identity,
        "semantic_text_fixed_bank": {"embeddings": bank, "sha256": digest, "class_ids": class_ids},
        "semantic_text_fixed_bank_sha256": digest,
        "soft_prompt_state_dict": {"context": torch.zeros(4, 768)},
    }
    assert validate_semantic_text_checkpoint(payload, role="semantic_text_S1")["fixed_bank_shape"] == [84, 512]
    payload["semantic_text_fixed_bank"]["sha256"] = "wrong"  # type: ignore[index]
    with pytest.raises(ValueError, match="hash"):
        validate_semantic_text_checkpoint(payload, role="semantic_text_S1")


def test_semantic_checkpoint_rejects_wrong_class_order() -> None:
    bank = torch.zeros(2, 3)
    ids = [4, 9]
    identity = semantic_text_identity(
        class_ids=ids, class_names=["a", "b"], token_ids=torch.zeros(2, 4, dtype=torch.long),
        eot_positions=[2, 2], model_name="x", pretrained="openai", fixed_bank=bank,
        initial_context=torch.zeros(4, 3), lambda_anchor=0.0,
    )
    payload = {
        "campaign": SEMANTIC_TEXT_CAMPAIGN, "experiment_role": "semantic_text_S1", "lambda_anchor": 0.0,
        "semantic_text_identity": {**identity, "class_names": ["b", "a"]},
        "semantic_text_fixed_bank": {"embeddings": bank, "sha256": tensor_sha256(bank), "class_ids": ids},
        "semantic_text_fixed_bank_sha256": tensor_sha256(bank),
        "soft_prompt_state_dict": {"context": torch.zeros(4, 3)},
    }
    with pytest.raises(ValueError, match="class-order"):
        validate_semantic_text_checkpoint(payload, role="semantic_text_S1")
