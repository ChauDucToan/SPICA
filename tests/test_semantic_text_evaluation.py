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


def _checkpoint(*, role: str = "semantic_text_S1", step: int = 0, run_kind: str = "primary") -> dict[str, object]:
    class_ids = list(range(84))
    width = 512 if run_kind == "primary" else 64
    token_length = 77 if run_kind == "primary" else 8
    bank = torch.ones(84, width)
    token_ids = torch.zeros(84, token_length, dtype=torch.long)
    token_ids[:, 5] = 7
    context = torch.zeros(4, width)
    identity = semantic_text_identity(
        class_ids=class_ids,
        class_names=[f"class_{value}" for value in class_ids],
        token_ids=token_ids,
        eot_positions=[5] * 84,
        model_name="ViT-B-32-quickgelu" if run_kind == "primary" else "tiny_cpu_fixture",
        pretrained="openai" if run_kind == "primary" else None, fixed_bank=bank,
        initial_context=context, initial_learned_bank=bank,
        lambda_anchor=1.0 if role == "semantic_text_S2" else 0.0,
    )
    identity.update({"context_width": width, "token_context_length": token_length, "eot_token_id": 7})
    digest = tensor_sha256(bank)
    config = {
        "experiment_campaign": SEMANTIC_TEXT_CAMPAIGN,
        "experiment_role": role,
        "model_name": identity["model_name"],
        "pretrained": identity["pretrained"],
    }
    payload: dict[str, object] = {
        "campaign": SEMANTIC_TEXT_CAMPAIGN,
        "experiment_role": role,
        "run_kind": run_kind,
        "step": step,
        "lambda_anchor": 1.0 if role == "semantic_text_S2" else 0.0,
        "semantic_text_identity": identity,
        "semantic_text_fixed_bank": {"embeddings": bank, "sha256": digest, "class_ids": class_ids},
        "semantic_text_fixed_bank_sha256": digest,
        "soft_prompt_state_dict": {"context": context},
        "current_context_sha256": tensor_sha256(context),
        "resolved_config": config,
        "training_class_list": class_ids,
        "data_split_identity": {"train_class_ids": class_ids},
    }
    return payload


def test_semantic_checkpoint_metadata_is_fail_closed() -> None:
    payload = _checkpoint()
    assert validate_semantic_text_checkpoint(payload, role="semantic_text_S1")["fixed_bank_shape"] == [84, 512]
    payload["semantic_text_fixed_bank"]["sha256"] = "wrong"  # type: ignore[index]
    with pytest.raises(ValueError, match="hash"):
        validate_semantic_text_checkpoint(payload, role="semantic_text_S1")


def test_semantic_checkpoint_rejects_wrong_class_order() -> None:
    payload = _checkpoint()
    identity = payload["semantic_text_identity"]
    identity["class_names"] = ["wrong", *identity["class_names"][1:]]  # type: ignore[index]
    with pytest.raises(ValueError, match="class-order"):
        validate_semantic_text_checkpoint(payload, role="semantic_text_S1")


def test_semantic_checkpoint_rejects_context_hash_drift() -> None:
    payload = _checkpoint(step=1)
    payload["soft_prompt_state_dict"]["context"][0, 0] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="current context hash"):
        validate_semantic_text_checkpoint(payload, role="semantic_text_S1")


def test_semantic_checkpoint_rejects_primary_wrong_width() -> None:
    payload = _checkpoint(run_kind="smoke")
    payload["run_kind"] = "primary"
    with pytest.raises(ValueError, match="context length"):
        validate_semantic_text_checkpoint(payload, role="semantic_text_S1")
