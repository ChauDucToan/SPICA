import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from open_clip.model import VisionTransformer
from torch import nn

from scripts.summarize_frozen_prompt import summarize, validate_run
from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.frozen_prompt import (
    linear_cka,
    orthogonal_procrustes_residual,
    prompt_token_geometry,
    representation_alignment,
)
from spica.frozen_prompt_artifacts import ROLES, canonical_sha256
from spica.models.frozen_prompt import FrozenPromptModel
from spica.train_frozen_prompt import build_optimizer_parameter_groups


def _tiny_vit() -> VisionTransformer:
    torch.manual_seed(11)
    return VisionTransformer(
        image_size=8,
        patch_size=4,
        width=8,
        layers=3,
        heads=2,
        mlp_ratio=2,
        output_dim=6,
    )


def _args(role: str) -> object:
    return OmegaConf.create(
        {
            "experiment_role": role,
            "visual_prompt_learning_rate": 1.0e-3,
            "soft_prompt_learning_rate": 1.0e-3,
            "visual_layernorm_learning_rate": 1.0e-6,
            "encoder_learning_rate": 1.0e-5,
            "visual_prompt_weight_decay": 1.0e-4,
            "soft_prompt_weight_decay": 1.0e-4,
            "visual_layernorm_weight_decay": 0.0,
            "encoder_weight_decay": 1.0e-4,
        }
    )


def test_optimizer_groups_are_disjoint_and_consume_soft_prompt_lr() -> None:
    model = FrozenPromptModel(_tiny_vit(), prompt_length=2)
    text = nn.Module()
    text.context = nn.Parameter(torch.randn(4, 8))
    groups, mapping = build_optimizer_parameter_groups(
        model, text, _args("frozen_prompt_v2_FP3")
    )
    assert [group["name"] for group in mapping] == [
        "visual_prompts",
        "visual_layernorm",
        "soft_text_prompt",
    ]
    assert mapping[-1]["lr"] == 1.0e-3
    names = [name for group in mapping for name in group["parameter_names"]]
    assert len(names) == len(set(names))
    assert set(names) == {
        "sketch_prompt",
        "photo_prompt",
        "soft_prompt.context",
    }
    assert len(groups) == 2


def test_sketch_only_prompt_excludes_photo_prompt() -> None:
    model = FrozenPromptModel(_tiny_vit(), prompt_length=2, train_photo_prompt=False)
    _, mapping = build_optimizer_parameter_groups(
        model, None, _args("frozen_prompt_v2_FP1S")
    )
    assert mapping[0]["parameter_names"] == ["sketch_prompt"]
    assert "photo_prompt" not in mapping[0]["parameter_names"]
    assert not model.photo_prompt.requires_grad


def test_layernorm_has_a_separate_matched_rate_and_zero_decay() -> None:
    model = FrozenPromptModel(_tiny_vit(), prompt_length=1, train_visual_layernorm=True)
    _, mapping = build_optimizer_parameter_groups(
        model, None, _args("frozen_prompt_v2_FP_LN")
    )
    layernorm = next(group for group in mapping if group["name"] == "visual_layernorm")
    assert layernorm["lr"] == 1.0e-6
    assert layernorm["weight_decay"] == 0.0
    assert layernorm["parameter_names"]
    assert not set(layernorm["parameter_names"]) & set(mapping[0]["parameter_names"])


def test_early_adapt_group_uses_exact_encoder_rate() -> None:
    model = nn.Linear(3, 2)
    _, mapping = build_optimizer_parameter_groups(
        model, None, _args("frozen_prompt_v2_FP5")
    )
    assert mapping[0]["name"] == "early_adapt_encoder"
    assert mapping[0]["lr"] == 1.0e-5
    assert mapping[1]["active"] is False


def test_cka_procrustes_and_prompt_token_geometry_are_real_probes() -> None:
    current = torch.eye(4)
    reference = torch.eye(4)
    assert linear_cka(current, reference) == pytest.approx(1.0)
    assert orthogonal_procrustes_residual(current, reference) == pytest.approx(
        0.0, abs=1e-6
    )
    encoded = EncodedRetrievalSet(
        embeddings=current,
        labels=torch.tensor([0, 0, 1, 1]),
        paths=("a", "b", "c", "d"),
    )
    assert representation_alignment(encoded, encoded)["linear_cka"] == pytest.approx(
        1.0
    )
    prompt_model = FrozenPromptModel(_tiny_vit(), prompt_length=2)
    geometry = prompt_token_geometry(prompt_model)
    assert len(geometry["sketch_token_norms"]) == 2
    assert len(geometry["photo_token_norms"]) == 2
    assert len(geometry["pairwise_sketch_prompt_token_cosine"]) == 2


def test_attention_probe_labels_multiple_exact_blocks() -> None:
    model = FrozenPromptModel(_tiny_vit(), prompt_length=1)
    result = model.attention_diagnostics_by_block(torch.randn(2, 3, 8, 8))
    assert [row["block_index"] for row in result["blocks"]] == [0, 1, 2]
    for row in result["blocks"]:
        assert {
            "cls_to_prompt_mass",
            "patch_to_prompt_mass",
            "prompt_to_cls_mass",
            "prompt_to_patch_mass",
        } <= row.keys()


def test_report_generation_never_deletes_existing_files(tmp_path: Path) -> None:
    sentinel = tmp_path / "historical.md"
    sentinel.write_text("historical")
    report = summarize(
        [],
        tmp_path / "plots",
        tmp_path / "summary.md",
        tmp_path / "summary.json",
        selection_json=tmp_path / "selection.json",
    )
    assert report["status"] == "incomplete"
    assert sentinel.read_text() == "historical"
    assert all(Path(path).is_file() for path in report["plots"].values())


def test_strict_summarizer_rejects_legacy_and_wrong_campaign(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"schema_version": 1, "experiment_role": "frozen_prompt_FP0"})
    )
    with pytest.raises(ValueError, match="legacy or invalid"):
        validate_run(legacy)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "experiment_role": ROLES[0],
                "campaign": "frozen_prompt_smoke_v2_2026-09-04",
                "run_kind": "smoke",
            }
        )
    )
    with pytest.raises(ValueError, match="wrong campaign"):
        validate_run(wrong)


def test_split_identity_hash_is_deterministic() -> None:
    value = {"classes": [1, 2], "seed": 3407}
    assert canonical_sha256(value) == canonical_sha256(json.loads(json.dumps(value)))
