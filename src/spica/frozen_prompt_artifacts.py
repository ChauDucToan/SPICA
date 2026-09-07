"""Shared frozen-prompt v2 treatment and manifest identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CAMPAIGN = "frozen_prompt_probe_v2_2026-09-04"
SMOKE_CAMPAIGN = "frozen_prompt_smoke_v2_2026-09-04"
FINAL_CAMPAIGN = "frozen_prompt_final_2026-09-04"
FINAL_SMOKE_CAMPAIGN = "frozen_prompt_final_smoke_2026-09-04"
PAIRING_PILOT_CAMPAIGN = "frozen_prompt_pairing_pilot_2026-09-06"
PAIRING_PILOT_ROLES = (
    "frozen_prompt_pairing_pilot_A",
    "frozen_prompt_pairing_pilot_B",
)
PAIRING_PILOT_STEPS = (0, 15, 44, 73, 100, 250, 500, 1000, 1800)
MASKED_VIEW_CAMPAIGN = "frozen_prompt_masked_view_pilot_2026-09-07"
MASKED_VIEW_ROLES = (
    "frozen_prompt_masked_view_C",
    "frozen_prompt_masked_view_M",
)
MASKED_VIEW_STEPS = (0, 15, 44, 73, 100, 250, 500, 1000, 1800)
# The pilot protocol above is historical and intentionally stays unchanged.
MASKED_VIEW_3600_CAMPAIGN = "frozen_prompt_masked_view_3600_2026-09-07"
MASKED_VIEW_3600_ROLES = (
    "frozen_prompt_masked_view_3600_C",
    "frozen_prompt_masked_view_3600_M",
)
MASKED_VIEW_3600_STEPS = (0, 600, 1200, 1800, 2400, 3000, 3600)
MASKED_VIEW_3600_SELECTION_STEPS = MASKED_VIEW_3600_STEPS[1:]
MASKED_VIEW_CAMPAIGNS = (MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN)

def is_masked_view_campaign(campaign: str) -> bool:
    return campaign in MASKED_VIEW_CAMPAIGNS

def is_masked_view_3600_campaign(campaign: str) -> bool:
    return campaign == MASKED_VIEW_3600_CAMPAIGN
MASK_POLICY = {
    "version": "ink_centered_square_v1",
    "ink_threshold": 0.9,
    "train_fractions": [0.25, 0.5, 0.75],
    "eval_fractions": [0.25, 0.5, 0.75],
    "train_seed": 4242,
    "eval_seeds": [101, 202, 303],
}
PAIRING_MANIFEST_SHA256 = "545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c"
PAIRING_MANIFEST_PATH = "outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.json"
FINAL_SPLIT_SEEDS = (101, 202, 303)
MANIFEST_PATH = Path(
    "outputs/experiment_manifest_frozen_prompt_probe_v2_2026-09-04.json"
)
FINAL_MANIFEST_PATH = Path(
    "outputs/experiment_manifest_frozen_prompt_final_2026-09-04.json"
)
ROLES = (
    "frozen_prompt_v2_FP0",
    "frozen_prompt_v2_FP1",
    "frozen_prompt_v2_FP1S",
    "frozen_prompt_v2_FP2",
    "frozen_prompt_v2_FP3",
    "frozen_prompt_v2_FP4",
    "frozen_prompt_v2_FP5",
    "frozen_prompt_v2_FP_LN",
)
FINAL_ROLES = (
    "frozen_prompt_final_FP3",
    "frozen_prompt_final_FP3S",
    "frozen_prompt_final_FP2",
    "frozen_prompt_final_FP_LN",
    "frozen_prompt_final_FP5",
)
ALL_ROLES = (
    ROLES
    + FINAL_ROLES
    + PAIRING_PILOT_ROLES
    + MASKED_VIEW_ROLES
    + MASKED_VIEW_3600_ROLES
)


def _treatment(
    *,
    visual_prompt_length: int,
    prompt_mode: str,
    text_mode: str,
    train_visual_layernorm: bool,
    train_sketch_prompt: bool,
    train_photo_prompt: bool,
    lambda_rank: float,
    lambda_cls: float,
    classification_location: str,
    encoder_mode: str,
    encoder_unfreeze_depth: int,
    encoder_train_ln_post: bool,
) -> dict[str, Any]:
    return {
        "visual_prompt_length": visual_prompt_length,
        "prompt_mode": prompt_mode,
        "text_mode": text_mode,
        "train_visual_layernorm": train_visual_layernorm,
        "train_sketch_prompt": train_sketch_prompt,
        "train_photo_prompt": train_photo_prompt,
        "lambda_rank": lambda_rank,
        "lambda_cls": lambda_cls,
        "classification_location": classification_location,
        "encoder_mode": encoder_mode,
        "encoder_unfreeze_depth": encoder_unfreeze_depth,
        "encoder_train_ln_post": encoder_train_ln_post,
        "transport_enabled": False,
        "transport_mode": "none",
        "num_positive_photos": 1,
        "batch_size": 32,
        "seed": 42,
        "pseudo_val_seed": 3407,
        "official_unseen_used_for_selection": False,
    }


ROLE_TREATMENTS: dict[str, dict[str, Any]] = {
    "frozen_prompt_v2_FP0": _treatment(
        visual_prompt_length=0,
        prompt_mode="vanilla",
        text_mode="none",
        train_visual_layernorm=False,
        train_sketch_prompt=False,
        train_photo_prompt=False,
        lambda_rank=0.0,
        lambda_cls=0.0,
        classification_location="none",
        encoder_mode="frozen",
        encoder_unfreeze_depth=0,
        encoder_train_ln_post=False,
    ),
    "frozen_prompt_v2_FP1": _treatment(
        visual_prompt_length=3,
        prompt_mode="prompt_only",
        text_mode="none",
        train_visual_layernorm=False,
        train_sketch_prompt=True,
        train_photo_prompt=True,
        lambda_rank=1.0,
        lambda_cls=0.0,
        classification_location="none",
        encoder_mode="frozen",
        encoder_unfreeze_depth=0,
        encoder_train_ln_post=False,
    ),
    "frozen_prompt_v2_FP1S": _treatment(
        visual_prompt_length=3,
        prompt_mode="sketch_prompt_only",
        text_mode="none",
        train_visual_layernorm=False,
        train_sketch_prompt=True,
        train_photo_prompt=False,
        lambda_rank=1.0,
        lambda_cls=0.0,
        classification_location="none",
        encoder_mode="frozen",
        encoder_unfreeze_depth=0,
        encoder_train_ln_post=False,
    ),
    "frozen_prompt_v2_FP2": _treatment(
        visual_prompt_length=3,
        prompt_mode="prompt_only",
        text_mode="hard",
        train_visual_layernorm=False,
        train_sketch_prompt=True,
        train_photo_prompt=True,
        lambda_rank=1.0,
        lambda_cls=1.0,
        classification_location="query",
        encoder_mode="frozen",
        encoder_unfreeze_depth=0,
        encoder_train_ln_post=False,
    ),
    "frozen_prompt_v2_FP3": _treatment(
        visual_prompt_length=3,
        prompt_mode="prompt_only",
        text_mode="soft",
        train_visual_layernorm=False,
        train_sketch_prompt=True,
        train_photo_prompt=True,
        lambda_rank=1.0,
        lambda_cls=1.0,
        classification_location="query",
        encoder_mode="frozen",
        encoder_unfreeze_depth=0,
        encoder_train_ln_post=False,
    ),
    "frozen_prompt_v2_FP4": _treatment(
        visual_prompt_length=0,
        prompt_mode="vanilla",
        text_mode="soft",
        train_visual_layernorm=False,
        train_sketch_prompt=False,
        train_photo_prompt=False,
        lambda_rank=0.0,
        lambda_cls=1.0,
        classification_location="query",
        encoder_mode="frozen",
        encoder_unfreeze_depth=0,
        encoder_train_ln_post=False,
    ),
    "frozen_prompt_v2_FP5": _treatment(
        visual_prompt_length=0,
        prompt_mode="early_adapt_then_freeze",
        text_mode="hard",
        train_visual_layernorm=False,
        train_sketch_prompt=False,
        train_photo_prompt=False,
        lambda_rank=1.0,
        lambda_cls=1.0,
        classification_location="z0",
        encoder_mode="partial",
        encoder_unfreeze_depth=4,
        encoder_train_ln_post=False,
    ),
    "frozen_prompt_v2_FP_LN": _treatment(
        visual_prompt_length=3,
        prompt_mode="prompt_plus_layernorm",
        text_mode="hard",
        train_visual_layernorm=True,
        train_sketch_prompt=True,
        train_photo_prompt=True,
        lambda_rank=1.0,
        lambda_cls=1.0,
        classification_location="query",
        encoder_mode="frozen",
        encoder_unfreeze_depth=0,
        encoder_train_ln_post=False,
    ),
}

FINAL_ROLE_TREATMENTS: dict[str, dict[str, Any]] = {
    "frozen_prompt_final_FP3": {
        **ROLE_TREATMENTS["frozen_prompt_v2_FP3"],
    },
    "frozen_prompt_final_FP3S": {
        **ROLE_TREATMENTS["frozen_prompt_v2_FP3"],
        "prompt_mode": "sketch_prompt_only",
        "train_photo_prompt": False,
    },
    "frozen_prompt_final_FP2": {
        **ROLE_TREATMENTS["frozen_prompt_v2_FP2"],
    },
    "frozen_prompt_final_FP_LN": {
        **ROLE_TREATMENTS["frozen_prompt_v2_FP_LN"],
    },
    "frozen_prompt_final_FP5": {
        **ROLE_TREATMENTS["frozen_prompt_v2_FP5"],
    },
}


MASKED_VIEW_TREATMENTS: dict[str, dict[str, Any]] = {
    role: {
        **FINAL_ROLE_TREATMENTS["frozen_prompt_final_FP2"],
        "positive_sampling": "same_class",
        "sketch_view_mode": "full_full" if role.endswith("_C") else "full_masked",
        "mask_policy": MASK_POLICY,
    }
    for role in MASKED_VIEW_ROLES
}
MASKED_VIEW_3600_TREATMENTS: dict[str, dict[str, Any]] = {
    role: {
        **FINAL_ROLE_TREATMENTS["frozen_prompt_final_FP2"],
        "positive_sampling": "same_class",
        "sketch_view_mode": "full_full" if role.endswith("_C") else "full_masked",
        "mask_policy": MASK_POLICY,
    }
    for role in MASKED_VIEW_3600_ROLES
}

PAIRING_PILOT_TREATMENTS: dict[str, dict[str, Any]] = {
    "frozen_prompt_pairing_pilot_A": {
        **FINAL_ROLE_TREATMENTS["frozen_prompt_final_FP2"],
        "positive_sampling": "same_class",
    },
    "frozen_prompt_pairing_pilot_B": {
        **FINAL_ROLE_TREATMENTS["frozen_prompt_final_FP2"],
        "positive_sampling": "paired",
    },
}


def treatment_for_role(
    role: str, *, seed: int | None = None, pseudo_val_seed: int | None = None
) -> dict[str, Any]:
    if role in MASKED_VIEW_3600_ROLES:
        treatments = MASKED_VIEW_3600_TREATMENTS
    elif role in MASKED_VIEW_ROLES:
        treatments = MASKED_VIEW_TREATMENTS
    elif role in PAIRING_PILOT_ROLES:
        treatments = PAIRING_PILOT_TREATMENTS
    elif role in FINAL_ROLES:
        treatments = FINAL_ROLE_TREATMENTS
    else:
        treatments = ROLE_TREATMENTS
    if role not in treatments:
        raise ValueError(f"unknown frozen-prompt role: {role}")
    result = dict(treatments[role])
    if seed is not None:
        result["seed"] = int(seed)
    if pseudo_val_seed is not None:
        result["pseudo_val_seed"] = int(pseudo_val_seed)
    return result


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def treatment_from_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = next(iter(ROLE_TREATMENTS.values())).keys()
    result = {key: config.get(key) for key in keys}
    if config.get("experiment_campaign") in {
        PAIRING_PILOT_CAMPAIGN,
        MASKED_VIEW_CAMPAIGN,
        MASKED_VIEW_3600_CAMPAIGN,
    }:
        result["positive_sampling"] = config.get("positive_sampling")
    if config.get("experiment_campaign") in {
        MASKED_VIEW_CAMPAIGN,
        MASKED_VIEW_3600_CAMPAIGN,
    }:
        result["sketch_view_mode"] = config.get("sketch_view_mode")
        result["mask_policy"] = config.get("mask_policy")
    return result


def expected_probe_steps(role: str, *, extended: bool = False) -> tuple[int, ...]:
    if role in MASKED_VIEW_3600_ROLES:
        return MASKED_VIEW_3600_STEPS
    if role in MASKED_VIEW_ROLES:
        return MASKED_VIEW_STEPS
    if role in PAIRING_PILOT_ROLES:
        return PAIRING_PILOT_STEPS
    if role == "frozen_prompt_v2_FP0":
        return (0,)
    if role.endswith("FP5"):
        return (0, 15, 44, 73)
    if extended and role in FINAL_ROLES:
        return (6000, 7200, 9000, 10800)
    if role in ALL_ROLES:
        return (0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400)
    raise ValueError(f"unknown frozen-prompt role: {role}")


def make_manifest(
    *,
    dataset: str,
    data_config: str,
    campaign: str = CAMPAIGN,
    positive_sampling: str | None = None,
    pairing_manifest_sha256: str | None = None,
    run_kind: str = "primary",
    selection_target_step: int | None = None,
) -> dict[str, Any]:
    final_roles = campaign in {FINAL_CAMPAIGN, FINAL_SMOKE_CAMPAIGN}
    pairing_pilot = campaign == PAIRING_PILOT_CAMPAIGN
    masked_view = campaign in {MASKED_VIEW_CAMPAIGN, MASKED_VIEW_3600_CAMPAIGN}
    masked_view_3600 = campaign == MASKED_VIEW_3600_CAMPAIGN
    roles = (
        MASKED_VIEW_3600_ROLES
        if masked_view_3600
        else MASKED_VIEW_ROLES
        if masked_view
        else PAIRING_PILOT_ROLES if pairing_pilot else FINAL_ROLES if final_roles else ROLES
    )
    treatments = (
        MASKED_VIEW_3600_TREATMENTS
        if masked_view_3600
        else MASKED_VIEW_TREATMENTS
        if masked_view
        else PAIRING_PILOT_TREATMENTS if pairing_pilot else FINAL_ROLE_TREATMENTS if final_roles else ROLE_TREATMENTS
    )
    result = {
        "schema_version": 5 if masked_view_3600 else 4 if masked_view else 3 if pairing_pilot else 1 if campaign == FINAL_CAMPAIGN else 2,
        "campaign": campaign,
        "dataset": dataset,
        "data_config": data_config,
        **(
            {
                "positive_sampling": positive_sampling,
                "pairing_manifest_sha256": pairing_manifest_sha256,
                "run_kind": run_kind,
                "selection_target_step": selection_target_step,
            }
            if pairing_pilot
            else {
                "positive_sampling": positive_sampling,
                "pairing_manifest_sha256": pairing_manifest_sha256,
                "run_kind": run_kind,
                "mask_policy": MASK_POLICY,
            }
            if masked_view
            else {}
        ),
        "selection_metric": (
            "mAP@200_prefix_positive"
            if masked_view_3600
            else "full_pseudo_unseen_mAP"
        ),
        "official_unseen_used_for_selection": False,
        "entries": {
            role: {
                "experiment_role": role,
                "campaign": campaign,
                "resolved_treatment": treatments[role],
                "dataset": dataset,
                "data_config": data_config,
                "training_seed": 42,
                "pseudo_validation_seed": 3407,
                "official_unseen_used_for_selection": False,
                **(
                    {
                        "positive_sampling": positive_sampling,
                        "pairing_manifest_sha256": pairing_manifest_sha256,
                        "run_kind": run_kind,
                        "selection_target_step": selection_target_step,
                    }
                    if pairing_pilot
                    else {
                        "positive_sampling": positive_sampling,
                        "pairing_manifest_sha256": pairing_manifest_sha256,
                        "run_kind": run_kind,
                        "sketch_view_mode": treatments[role].get("sketch_view_mode", "single"),
                        "mask_policy": treatments[role].get("mask_policy"),
                    }
                    if masked_view
                    else {}
                ),
            }
            for role in roles
        },
    }
    if campaign == MASKED_VIEW_3600_CAMPAIGN:
        result["selection_policy"] = {
            "candidate_steps": list(MASKED_VIEW_3600_SELECTION_STEPS),
            "primary_clean": "clean.mAP@200_prefix_positive",
            "primary_masked": "masked_macro.mAP@200_prefix_positive",
            "denominator": "prefix_positive",
            "tie_break": "strict_greater_earliest_step",
            "step_zero_selectable": False,
            "official_unseen_used_for_selection": False,
        }
    if campaign == FINAL_CAMPAIGN:
        result.update(
            {
                "predeclared_training_seeds": [42, 123, 3407],
                "predeclared_split_seeds": [101, 202, 303],
                "finalist_roles": [
                    "frozen_prompt_final_FP3",
                    "frozen_prompt_final_FP3S",
                ],
                "selection_policy": {
                    "official_unseen_used_for_selection": False,
                    "bootstrap_draws_minimum": 2000,
                    "simpler_model_margin": 0.003,
                    "photo_prompt_threshold": 0.005,
                    "layernorm_threshold": 0.005,
                },
            }
        )
    return result


def ensure_manifest(
    path: Path,
    *,
    dataset: str,
    data_config: str,
    campaign: str = CAMPAIGN,
    positive_sampling: str | None = None,
    pairing_manifest_sha256: str | None = None,
    run_kind: str = "primary",
    selection_target_step: int | None = None,
) -> tuple[dict[str, Any], str]:
    expected = make_manifest(
        dataset=dataset,
        data_config=data_config,
        campaign=campaign,
        positive_sampling=positive_sampling,
        pairing_manifest_sha256=pairing_manifest_sha256,
        run_kind=run_kind,
        selection_target_step=selection_target_step,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = json.loads(path.read_text())
        if actual != expected:
            raise ValueError(
                f"frozen-prompt v2 manifest already exists but differs: {path}"
            )
    else:
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return expected, hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_entry_identity(
    path: Path,
    manifest: dict[str, Any],
    *,
    role: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or role not in entries:
        raise ValueError(f"manifest has no entry for {role}")
    entry = entries[role]
    return {
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "entry_pointer": f"/entries/{role}",
        "entry_sha256": canonical_sha256(entry),
    }
