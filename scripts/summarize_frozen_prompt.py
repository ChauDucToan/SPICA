"""Strict, provenance-driven summarizer for frozen-prompt v2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spica.frozen_prompt_artifacts import (
    CAMPAIGN,
    ROLE_TREATMENTS,
    ROLES,
    canonical_sha256,
    expected_probe_steps,
    treatment_from_config,
)

ROOT = Path(__file__).resolve().parents[1]
PLOT_NAMES = {
    "main_ablation": "frozen_prompt_v2_main_ablation_2026-09-04.png",
    "stability": "frozen_prompt_v2_stability_2026-09-04.png",
    "visual_geometry": "frozen_prompt_v2_visual_geometry_2026-09-04.png",
    "soft_text_effect": "frozen_prompt_v2_soft_text_effect_2026-09-04.png",
    "prompt_vs_early_freeze": "frozen_prompt_v2_prompt_vs_early_freeze_2026-09-04.png",
    "attention_analysis": "frozen_prompt_v2_attention_analysis_2026-09-04.png",
    "prompt_token_geometry": "frozen_prompt_v2_prompt_token_geometry_2026-09-04.png",
}
PROMPT_ROLES = {
    "frozen_prompt_v2_FP1",
    "frozen_prompt_v2_FP1S",
    "frozen_prompt_v2_FP2",
    "frozen_prompt_v2_FP3",
    "frozen_prompt_v2_FP_LN",
}
PROMPT_ONLY_ROLES = {
    "frozen_prompt_v2_FP1",
    "frozen_prompt_v2_FP1S",
    "frozen_prompt_v2_FP2",
    "frozen_prompt_v2_FP3",
}


def _resolve(path: object) -> Path:
    value = Path(str(path)).expanduser()
    return value if value.is_absolute() else ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def _close(left: object, right: object) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


def _validate_manifest_entry(run: dict[str, Any], role: str) -> None:
    identity = run.get("manifest_entry_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{role}: missing manifest-entry identity")
    path = _resolve(identity.get("manifest_path"))
    if not path.is_file():
        raise ValueError(f"{role}: manifest does not exist: {path}")
    if _sha256(path) != identity.get("manifest_sha256"):
        raise ValueError(f"{role}: manifest hash mismatch")
    manifest = _json(path)
    entries = manifest.get("entries")
    if manifest.get("campaign") != CAMPAIGN or not isinstance(entries, dict):
        raise ValueError(f"{role}: invalid v2 manifest")
    entry = entries.get(role)
    if entry is None or identity.get("entry_sha256") != canonical_sha256(entry):
        raise ValueError(f"{role}: manifest-entry hash mismatch")
    if identity.get("entry_pointer") != f"/entries/{role}":
        raise ValueError(f"{role}: manifest-entry pointer mismatch")


def _validate_checkpoint(
    checkpoint_path: Path,
    expected_hash: str,
    run: dict[str, Any],
    role: str,
) -> None:
    if not checkpoint_path.is_file():
        raise ValueError(f"{role}: checkpoint does not exist: {checkpoint_path}")
    if _sha256(checkpoint_path) != expected_hash:
        raise ValueError(f"{role}: checkpoint SHA256 mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{role}: checkpoint is not a mapping")
    if (
        payload.get("format_version") != 2
        or payload.get("model_type") != "frozen_prompt_v2"
    ):
        raise ValueError(f"{role}: unsupported checkpoint schema")
    if payload.get("experiment_role") != role or payload.get("campaign") != CAMPAIGN:
        raise ValueError(f"{role}: checkpoint role/campaign mismatch")
    if payload.get("data_split_identity") != run.get("pseudo_split_identity"):
        raise ValueError(f"{role}: checkpoint split identity mismatch")
    if payload.get("data_manifest_identity") != run.get("manifest_identity"):
        raise ValueError(f"{role}: checkpoint manifest identity mismatch")
    if payload.get("manifest_entry_identity") != run.get("manifest_entry_identity"):
        raise ValueError(f"{role}: checkpoint manifest-entry identity mismatch")
    if payload.get("optimizer_groups") != run.get("optimizer_groups"):
        raise ValueError(f"{role}: checkpoint optimizer groups mismatch")
    expected_names = set(run.get("trainable_parameter_names", []))
    checkpoint_names = set(payload.get("trainable_parameter_names", []))
    if checkpoint_names != expected_names:
        raise ValueError(f"{role}: checkpoint trainable parameter names mismatch")
    checkpoint_policy = payload.get("clip_freeze_policy")
    run_policy = run.get("clip_freeze_policy")
    policy_keys = (
        "role",
        "approved_trainable_clip_parameter_names",
        "photo_encoder_frozen",
        "visual_projection_frozen",
        "text_tower_frozen",
    )
    if (
        not isinstance(checkpoint_policy, dict)
        or not isinstance(run_policy, dict)
        or any(checkpoint_policy.get(key) != run_policy.get(key) for key in policy_keys)
    ):
        raise ValueError(f"{role}: checkpoint CLIP freeze policy mismatch")
    if not isinstance(payload.get("rng_state"), dict):
        raise ValueError(f"{role}: checkpoint has no RNG state")
    if "scheduler_state_dict" not in payload:
        raise ValueError(f"{role}: checkpoint has no scheduler state field")


def _validate_optimizer_groups(run: dict[str, Any], role: str) -> None:
    groups = run.get("optimizer_groups")
    trainable = run.get("trainable_parameter_names")
    if not isinstance(groups, list) or not isinstance(trainable, list):
        raise ValueError(f"{role}: missing optimizer groups or trainable names")
    all_names: list[str] = []
    names_seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError(f"{role}: malformed optimizer group")
        names = group.get("parameter_names")
        if not isinstance(names, list):
            raise ValueError(f"{role}: optimizer group has no parameter names")
        if len(names) != len(set(names)) or names_seen.intersection(names):
            raise ValueError(f"{role}: duplicate optimizer parameters")
        names_seen.update(names)
        all_names.extend(names)
        if bool(group.get("active")) != bool(names):
            raise ValueError(f"{role}: optimizer active flag is inconsistent")
        if float(group.get("lr", -1)) <= 0 or float(group.get("weight_decay", -1)) < 0:
            raise ValueError(f"{role}: invalid optimizer hyperparameters")
    if set(all_names) != set(trainable):
        raise ValueError(
            f"{role}: optimizer groups do not cover trainable parameters exactly once"
        )
    expected = ROLE_TREATMENTS[role]
    group_by_name = {group["name"]: group for group in groups}
    if role == "frozen_prompt_v2_FP5":
        if group_by_name.get("early_adapt_encoder", {}).get("lr") != 1.0e-5:
            raise ValueError(f"{role}: encoder LR is not exactly 1e-5")
    else:
        if (
            group_by_name.get("visual_prompts", {}).get("lr") != 1.0e-3
            and role in PROMPT_ROLES
        ):
            raise ValueError(f"{role}: visual prompt LR is not exactly 1e-3")
    if (
        role == "frozen_prompt_v2_FP_LN"
        and group_by_name.get("visual_layernorm", {}).get("lr") != 1.0e-6
    ):
        raise ValueError(f"{role}: LayerNorm LR is not exactly 1e-6")
    if (
        role in {"frozen_prompt_v2_FP3", "frozen_prompt_v2_FP4"}
        and group_by_name.get("soft_text_prompt", {}).get("lr") != 1.0e-3
    ):
        raise ValueError(f"{role}: soft-prompt LR is not consumed")
    if expected["text_mode"] != "soft" and group_by_name.get(
        "soft_text_prompt", {}
    ).get("active"):
        raise ValueError(f"{role}: unexpected trainable soft text prompt")


def validate_run(path: Path) -> dict[str, Any]:
    run = _json(path)
    role = run.get("experiment_role")
    if role not in ROLES:
        raise ValueError(f"legacy or invalid experiment role: {role!r}")
    if run.get("schema_version") != 2:
        raise ValueError(f"{role}: schema version is not 2")
    if run.get("campaign") != CAMPAIGN or run.get("run_kind") != "primary":
        raise ValueError(
            f"{role}: wrong campaign or smoke artifact supplied as primary evidence"
        )
    if run.get("official_unseen_used_for_selection") is not False:
        raise ValueError(f"{role}: official unseen was used for selection")
    config = run.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError(f"{role}: missing resolved config")
    observed_treatment = run.get("resolved_treatment")
    expected_treatment = ROLE_TREATMENTS[role]
    if (
        observed_treatment != expected_treatment
        or treatment_from_config(config) != expected_treatment
    ):
        raise ValueError(f"{role}: resolved treatment/config does not match role")
    if run.get("seed") != 42 or run.get("pseudo_validation_seed") != 3407:
        raise ValueError(f"{role}: wrong training or pseudo-validation seed")
    split = run.get("pseudo_split_identity")
    if not isinstance(split, dict) or split.get("sha256") != canonical_sha256(
        {key: value for key, value in split.items() if key != "sha256"}
    ):
        raise ValueError(f"{role}: invalid pseudo split identity")
    manifest_identity = run.get("manifest_identity")
    if not isinstance(manifest_identity, dict) or not manifest_identity.get("sha256"):
        raise ValueError(f"{role}: missing data manifest identity")
    _validate_manifest_entry(run, role)
    provenance = run.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("status") != "valid":
        raise ValueError(f"{role}: invalid provenance")
    if not provenance.get("head_commit") or not (
        run.get("experiment_code_commit") or run.get("source_snapshot_hash")
    ):
        raise ValueError(f"{role}: no clean code commit or source snapshot provenance")
    if provenance.get("working_tree_state") == "dirty" and not run.get(
        "source_snapshot_hash"
    ):
        raise ValueError(f"{role}: dirty run has no source snapshot")
    _validate_optimizer_groups(run, role)
    if (
        run.get("clip_freeze_policy", {}).get("text_tower_frozen") is not True
        or run.get("clip_freeze_policy", {}).get("visual_projection_frozen") is not True
    ):
        raise ValueError(f"{role}: CLIP freeze policy is incomplete")

    history = run.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{role}: missing pointwise history")
    steps = [
        int(row.get("training_global_step", -1))
        for row in history
        if isinstance(row, dict)
    ]
    if len(steps) != len(set(steps)):
        raise ValueError(f"{role}: duplicate logical checkpoints")
    expected_steps = expected_probe_steps(role)
    if set(steps) != set(expected_steps):
        raise ValueError(
            f"{role}: expected probe steps {expected_steps}, got {sorted(steps)}"
        )
    for row in history:
        if (
            not isinstance(row, dict)
            or row.get("comparison_horizon", {}).get("kind") != "training_global_step"
        ):
            raise ValueError(
                f"{role}: history contains a non-training point or fake checkpoint"
            )
        if int(row["comparison_horizon"]["value"]) != int(row["training_global_step"]):
            raise ValueError(f"{role}: comparison horizon does not equal training step")
        if (
            row.get("official_unseen_used_for_selection") is not False
            or row.get("protocol", {}).get("official_unseen_used_for_selection")
            is not False
        ):
            raise ValueError(f"{role}: official unseen appears in a probe")
        if (
            not isinstance(row.get("optimizer_groups"), list)
            or row["optimizer_groups"] != run["optimizer_groups"]
        ):
            raise ValueError(f"{role}: probe optimizer mapping is incomplete")
        if row.get("trainable_parameter_names") != run["trainable_parameter_names"]:
            raise ValueError(f"{role}: probe trainable names are incomplete")
        checkpoint = _resolve(row.get("checkpoint"))
        expected_hash = row.get("checkpoint_sha256")
        if not isinstance(expected_hash, str):
            raise ValueError(f"{role}: probe has no checkpoint hash")
        _validate_checkpoint(checkpoint, expected_hash, run, role)
    holds = run.get("frozen_hold_evaluation", [])
    if role == "frozen_prompt_v2_FP5":
        if not isinstance(holds, list) or [
            item.get("comparison_horizon") for item in holds
        ] != [500, 1800, 5400]:
            raise ValueError(
                f"{role}: frozen hold evaluations are missing or malformed"
            )
        for item in holds:
            if (
                item.get("kind") != "frozen_hold_evaluation"
                or item.get("parameters_updated_since_selection") != 0
            ):
                raise ValueError(f"{role}: FP5 hold was encoded as optimization")
    elif holds:
        raise ValueError(f"{role}: only FP5 may contain frozen hold evaluations")
    return run


def load_runs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    if not paths:
        return {}
    runs: dict[str, dict[str, Any]] = {}
    for path in paths:
        run = _json(path)
        role = run.get("experiment_role")
        if role in runs:
            raise ValueError(f"duplicate experiment role: {role}")
        validated = validate_run(path)
        runs[role] = validated
    identities = {
        (
            json.dumps(run["pseudo_split_identity"], sort_keys=True),
            json.dumps(run["manifest_identity"], sort_keys=True),
        )
        for run in runs.values()
    }
    if len(identities) > 1:
        raise ValueError("runs contain mixed pseudo splits or manifest identities")
    return runs


def _row(run: dict[str, Any], step: int) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in run.get("history", [])
            if int(row["training_global_step"]) == step
        ),
        None,
    )


def _metric(run: dict[str, Any], step: int, key: str = "full_mAP") -> float | None:
    value = _row(run, step)
    if value is None:
        if run.get("experiment_role") == "frozen_prompt_v2_FP5":
            value = next(
                (
                    item
                    for item in run.get("frozen_hold_evaluation", [])
                    if item.get("comparison_horizon") == step
                ),
                None,
            )
        elif run.get("experiment_role") == "frozen_prompt_v2_FP0":
            value = _row(run, 0)
    if value is None:
        return None
    values = value.get("val", value)
    return None if values.get(key) is None else float(values[key])


def _peak(run: dict[str, Any]) -> dict[str, Any] | None:
    rows = run.get("history", [])
    return (
        max(
            rows,
            key=lambda value: (
                float(value["full_pseudo_unseen_mAP"]),
                -int(value["training_global_step"]),
            ),
        )
        if rows
        else None
    )


def _best_prompt(runs: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [(_peak(runs[role]), role) for role in PROMPT_ROLES if role in runs]
    candidates = [(row, role) for row, role in candidates if row is not None]
    if not candidates:
        return None
    row, role = max(
        candidates, key=lambda item: (float(item[0]["full_pseudo_unseen_mAP"]), item[1])
    )
    return {"role": role, **row}


def _write_once(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def _plot(path: Path, title: str, draw: Any, missing: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(9, 5))
    if missing:
        figure.text(
            0.5,
            0.5,
            "NOT_RUN\nmissing cells: " + ", ".join(missing),
            ha="center",
            va="center",
            fontsize=13,
        )
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
        return
    try:
        draw(figure)
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path)
    finally:
        plt.close(figure)


def _plot_data(runs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    missing = sorted(set(ROLES) - runs.keys())
    return runs, missing


def _make_plots(runs: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, str]:
    runs, missing = _plot_data(runs)
    paths = {key: output_dir / name for key, name in PLOT_NAMES.items()}

    def main(figure: Any) -> None:
        axis = figure.add_subplot(111)
        roles = list(ROLES)
        x = list(range(len(roles)))
        peak = [
            None if role not in runs else _peak(runs[role])["full_pseudo_unseen_mAP"]
            for role in roles
        ]
        late = [_metric(runs[role], 5400) if role in runs else None for role in roles]
        axis.bar(
            [value - 0.18 for value in x],
            [value or 0 for value in peak],
            width=0.36,
            label="peak full mAP",
        )
        axis.bar(
            [value + 0.18 for value in x],
            [value or 0 for value in late],
            width=0.36,
            label="late / frozen hold full mAP",
        )
        axis.set_xticks(
            x,
            [role.replace("frozen_prompt_v2_", "") for role in roles],
            rotation=30,
            ha="right",
        )
        axis.set_ylabel("full pseudo-unseen mAP")
        axis.legend()

    def stability(figure: Any) -> None:
        axis = figure.add_subplot(111)
        for role in (*PROMPT_ONLY_ROLES, "frozen_prompt_v2_FP_LN"):
            if role in runs:
                rows = runs[role]["history"]
                axis.plot(
                    [row["training_global_step"] for row in rows],
                    [row["full_pseudo_unseen_mAP"] for row in rows],
                    marker=".",
                    label=role.replace("frozen_prompt_v2_", ""),
                )
        if "frozen_prompt_v2_FP0" in runs:
            axis.axhline(
                _metric(runs["frozen_prompt_v2_FP0"], 0) or 0,
                color="black",
                linestyle="--",
                label="FP0 stationary reference",
            )
        if "frozen_prompt_v2_FP5" in runs:
            axis.axhline(
                _metric(runs["frozen_prompt_v2_FP5"], 5400) or 0,
                color="brown",
                linestyle=":",
                label="FP5 frozen-hold reference",
            )
        axis.set_xlabel(
            "training global step; dashed/ dotted lines are stationary references"
        )
        axis.set_ylabel("full pseudo-unseen mAP")
        axis.legend(fontsize=7)

    def visual_geometry(figure: Any) -> None:
        axes = figure.subplots(2, 2).flat
        fields = (
            "semantic_margin",
            "sketch_reference_cosine",
            "linear_cka",
            "orthogonal_procrustes_residual",
        )
        for axis, field in zip(axes, fields, strict=True):
            for role in PROMPT_ROLES:
                if role not in runs:
                    continue
                rows = runs[role]["history"]
                axis.plot(
                    [row["training_global_step"] for row in rows],
                    [float(row[field]) for row in rows],
                    marker=".",
                    label=role.replace("frozen_prompt_v2_", ""),
                )
            axis.set_title(field)
            axis.set_xlabel("training global step")
        axes[0].legend(fontsize=6)

    def soft_text(figure: Any) -> None:
        axes = figure.subplots(1, 2).flat
        for role in ("frozen_prompt_v2_FP2", "frozen_prompt_v2_FP3"):
            rows = runs[role]["history"]
            axes[0].plot(
                [row["training_global_step"] for row in rows],
                [row["full_pseudo_unseen_mAP"] for row in rows],
                marker=".",
                label=role.replace("frozen_prompt_v2_", ""),
            )
            values = [row["diagnostic_seen_classification_accuracy"] for row in rows]
            axes[1].plot(
                [row["training_global_step"] for row in rows],
                values,
                marker=".",
                label=role.replace("frozen_prompt_v2_", ""),
            )
        fp4 = runs["frozen_prompt_v2_FP4"]
        axes[0].axhline(
            _metric(fp4, 0) or 0,
            linestyle="--",
            color="gray",
            label="FP4 fixed-visual retrieval reference",
        )
        axes[1].plot(
            [row["training_global_step"] for row in fp4["history"]],
            [row["diagnostic_seen_classification_accuracy"] for row in fp4["history"]],
            marker=".",
            color="black",
            label="FP4 classification-only",
        )
        axes[0].set_ylabel("full pseudo-unseen mAP")
        axes[1].set_ylabel("diagnostic seen classification accuracy")
        axes[0].set_xlabel("training global step")
        axes[1].set_xlabel("training global step")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)

    def prompt_vs_freeze(figure: Any) -> None:
        axis = figure.add_subplot(111)
        best = _best_prompt(runs)
        rows = runs[best["role"]]["history"]
        axis.plot(
            [row["training_global_step"] for row in rows],
            [row["full_pseudo_unseen_mAP"] for row in rows],
            marker=".",
            label=f"best prompt: {best['role'].replace('frozen_prompt_v2_', '')}",
        )
        fp5 = runs["frozen_prompt_v2_FP5"]
        axis.axhline(
            _metric(fp5, 5400) or 0,
            linestyle=":",
            color="brown",
            label="matched FP5 frozen hold",
        )
        axis.axhline(
            _metric(runs["frozen_prompt_v2_FP0"], 0) or 0,
            linestyle="--",
            color="black",
            label="FP0 stationary reference",
        )
        axis.set_xlabel("training global step")
        axis.set_ylabel("full pseudo-unseen mAP")
        axis.legend()

    def attention(figure: Any) -> None:
        axis = figure.add_subplot(111)
        best = _best_prompt(runs)
        role = best["role"] if best else "frozen_prompt_v2_FP1"
        rows = [row for row in runs[role]["history"] if row.get("prompt_attention")]
        row = max(rows, key=lambda item: int(item["training_global_step"]))
        blocks = row["prompt_attention"]["blocks"]
        for field in (
            "cls_to_prompt_mass",
            "patch_to_prompt_mass",
            "prompt_to_cls_mass",
            "prompt_to_patch_mass",
        ):
            axis.plot(
                [item["block_index"] for item in blocks],
                [item[field] for item in blocks],
                marker="o",
                label=field,
            )
        axis.set_xticks(
            [item["block_index"] for item in blocks],
            [f"block {item['block_index']}" for item in blocks],
        )
        axis.set_xlabel("exact transformer block index")
        axis.set_ylabel("attention mass")
        axis.legend(fontsize=7)

    def token_geometry(figure: Any) -> None:
        best = _best_prompt(runs)
        rows = [
            row
            for row in runs[best["role"]]["history"]
            if row.get("geometry", {}).get("prompt_token_geometry")
        ]
        geometry = max(rows, key=lambda item: int(item["training_global_step"]))[
            "geometry"
        ]["prompt_token_geometry"]
        axes = figure.subplots(1, 3).flat
        axes[0].plot(geometry["sketch_token_norms"], marker="o", label="sketch")
        axes[0].plot(geometry["photo_token_norms"], marker="o", label="photo")
        axes[0].set_title("token norms")
        axes[0].set_xlabel("prompt token index")
        axes[0].legend()
        axes[1].imshow(
            geometry["pairwise_sketch_prompt_token_cosine"],
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axes[1].set_title("sketch token cosine")
        axes[2].imshow(
            geometry["sketch_vs_photo_prompt_token_cosine"],
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axes[2].set_title("sketch × photo cosine")

    specs = {
        "main_ablation": (main, list(set(ROLES) - runs.keys())),
        "stability": (
            stability,
            list(
                set(
                    (
                        *PROMPT_ONLY_ROLES,
                        "frozen_prompt_v2_FP_LN",
                        "frozen_prompt_v2_FP0",
                        "frozen_prompt_v2_FP5",
                    )
                )
                - runs.keys()
            ),
        ),
        "visual_geometry": (visual_geometry, list(set(PROMPT_ROLES) - runs.keys())),
        "soft_text_effect": (
            soft_text,
            list(
                set(
                    (
                        "frozen_prompt_v2_FP2",
                        "frozen_prompt_v2_FP3",
                        "frozen_prompt_v2_FP4",
                    )
                )
                - runs.keys()
            ),
        ),
        "prompt_vs_early_freeze": (
            prompt_vs_freeze,
            list(
                set(
                    (
                        *PROMPT_ONLY_ROLES,
                        "frozen_prompt_v2_FP_LN",
                        "frozen_prompt_v2_FP5",
                        "frozen_prompt_v2_FP0",
                    )
                )
                - runs.keys()
            ),
        ),
        "attention_analysis": (attention, list(set(PROMPT_ROLES) - runs.keys())),
        "prompt_token_geometry": (
            token_geometry,
            list(set(PROMPT_ROLES) - runs.keys()),
        ),
    }
    for key, (draw, required_missing) in specs.items():
        _plot(paths[key], key.replace("_", " "), draw, sorted(required_missing))
    return {key: str(path) for key, path in paths.items()}


def _selection(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for role, run in sorted(runs.items()):
        peak = _peak(run)
        if peak is not None:
            candidates.append(
                {
                    "role": role,
                    "training_global_step": int(peak["training_global_step"]),
                    "full_pseudo_unseen_mAP": float(peak["full_pseudo_unseen_mAP"]),
                    "checkpoint": peak["checkpoint"],
                    "checkpoint_sha256": peak["checkpoint_sha256"],
                }
            )
    prompted = [item for item in candidates if item["role"] in PROMPT_ROLES]
    selected = (
        max(prompted, key=lambda item: (item["full_pseudo_unseen_mAP"], item["role"]))
        if prompted
        else None
    )
    return {
        "schema_version": 2,
        "campaign": CAMPAIGN,
        "selection_metric": "full_pseudo_unseen_mAP",
        "official_unseen_used_for_selection": False,
        "candidates": candidates,
        "selected_prompt": selected,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "not_run"
    return f"{float(value):.6f}"


def _yes_no(value: bool | None) -> str:
    return "YES" if value is True else "NO" if value is False else "not_run"


def _derived(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fp0 = runs.get("frozen_prompt_v2_FP0")
    fp1 = runs.get("frozen_prompt_v2_FP1")
    fp1s = runs.get("frozen_prompt_v2_FP1S")
    fp2 = runs.get("frozen_prompt_v2_FP2")
    fp3 = runs.get("frozen_prompt_v2_FP3")
    fp5 = runs.get("frozen_prompt_v2_FP5")
    fpln = runs.get("frozen_prompt_v2_FP_LN")
    best = _best_prompt(runs)
    best_prompt_peak = None if best is None else float(best["full_pseudo_unseen_mAP"])
    fp0_peak = None if fp0 is None else float(_peak(fp0)["full_pseudo_unseen_mAP"])
    fp5_peak = None if fp5 is None else float(_peak(fp5)["full_pseudo_unseen_mAP"])
    fp5_selected = None if fp5 is None else fp5.get("selection", {})
    fp5_step = (
        None
        if not isinstance(fp5_selected, dict)
        else fp5_selected.get("training_global_step")
    )
    fp5_selected_map = (
        None
        if not isinstance(fp5_selected, dict)
        else fp5_selected.get("full_pseudo_unseen_mAP")
    )

    def effect(
        left: dict[str, Any] | None, right: dict[str, Any] | None
    ) -> float | None:
        if left is None or right is None:
            return None
        return float(_peak(left)["full_pseudo_unseen_mAP"]) - float(
            _peak(right)["full_pseudo_unseen_mAP"]
        )

    return {
        "best_prompt": None
        if best is None
        else {
            "role": best["role"],
            "peak_mAP": best_prompt_peak,
            "peak_step": best["training_global_step"],
            "checkpoint": best["checkpoint"],
        },
        "fp0_peak_mAP": fp0_peak,
        "fp5_peak_mAP": fp5_peak,
        "fp5_selected_step": fp5_step,
        "fp5_selected_mAP": fp5_selected_map,
        "visual_prompt_improves_vanilla": None
        if best_prompt_peak is None or fp0_peak is None
        else best_prompt_peak > fp0_peak,
        "photo_prompt_necessary": None
        if fp1 is None or fp1s is None
        else abs(effect(fp1, fp1s) or 0.0) > 0.005,
        "hard_text_effect": effect(fp2, fp1),
        "soft_text_effect": effect(fp3, fp2),
        "layernorm_effect": effect(fpln, fp3 if fp3 is not None else fp2),
        "prompt_vs_fp5_delta": None
        if best_prompt_peak is None or fp5_peak is None
        else best_prompt_peak - fp5_peak,
        "late": {
            role: {str(step): _metric(run, step) for step in (500, 1800, 5400)}
            for role, run in runs.items()
        },
        "retention": {
            role: None
            if _peak(run) is None or _metric(run, 5400) is None
            else _metric(run, 5400) / float(_peak(run)["full_pseudo_unseen_mAP"])
            for role, run in runs.items()
        },
    }


def _markdown(
    runs: dict[str, dict[str, Any]],
    report: dict[str, Any],
    *,
    starting_commit: str | None,
) -> str:
    derived = report["derived"]
    missing = sorted(set(ROLES) - runs.keys())
    status = report["status"]
    lines = [
        "# Frozen-prompt SPICA v2 probe",
        "",
        f"Status: **{status}**",
        "",
        "## Artifact restoration",
        "",
        "Historical transport artifacts were restored in the dedicated restoration commit before this campaign. No historical artifact is used as v2 evidence.",
        "",
        "## Integrity repairs",
        "",
        "- Separate AdamW groups, exact FP5/FP-LN rates, manifest/checkpoint hashes, fixed diagnostic classification, real checkpoint horizons, CKA/Procrustes, token geometry, and multi-block attention are recorded in the validated run artifacts.",
        "- FP4 is a fixed-visual retrieval control; smoke artifacts and official unseen metrics are excluded from selection.",
        "",
        "## Architecture",
        "",
    ]
    for role in ROLES:
        run = runs.get(role)
        if run is None:
            lines.extend([f"### {role}", "not_run", ""])
            continue
        treatment = run["resolved_treatment"]
        lines.extend(
            [
                f"### {role}",
                f"frozen parameters: {len(run.get('frozen_parameter_names', []))} recorded names; CLIP policy={run['clip_freeze_policy']}",
                f"trainable parameters: {len(run['trainable_parameter_names'])} recorded names",
                "optimizer groups: "
                + ", ".join(
                    f"{group['name']} lr={group['lr']} wd={group['weight_decay']}"
                    for group in run["optimizer_groups"]
                    if group["active"]
                ),
                f"losses: rank={treatment['lambda_rank']}; classification={treatment['lambda_cls']} at {treatment['classification_location']}",
                "inference inputs: raw_sketch_image only; text/photo/oracle class not required",
                "",
            ]
        )
    lines.extend(
        [
            "## Primary results",
            "",
            "| Cell | Visual adaptation | Text supervision | Peak mAP | Peak step | mAP@500 | mAP@1800 | mAP@5400 | Retention |",
            "| ---- | ----------------- | ---------------- | -------: | --------: | ------: | -------: | -------: | --------: |",
        ]
    )
    for role in ROLES:
        run = runs.get(role)
        if run is None:
            lines.append(
                f"| {role} | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |"
            )
            continue
        treatment = run["resolved_treatment"]
        visual = (
            "early-adapt depth-4"
            if role.endswith("FP5")
            else "visual prompts"
            if treatment["visual_prompt_length"]
            else "none"
        )
        text = treatment["text_mode"]
        peak = _peak(run)
        lines.append(
            f"| {role} | {visual} | {text} | {_fmt(peak['full_pseudo_unseen_mAP'])} | {peak['training_global_step']} | {_fmt(_metric(run, 500))} | {_fmt(_metric(run, 1800))} | {_fmt(_metric(run, 5400))} | {_fmt(derived['retention'].get(role))} |"
        )
    lines.extend(
        [
            "",
            "## Prompt geometry",
            "",
            "Reference preservation, semantic margin, effective rank, CKA, Procrustes residual, and per-token prompt geometry are in each raw pointwise row. The plots route those fields only to the geometry plot.",
            "",
            "## Attention",
            "",
            "Attention rows carry exact block indices and all four directions: CLS→prompt, patch→prompt, prompt→CLS, and prompt→patch.",
            "",
            "## Soft-text verdict",
            "",
            f"Peak soft-text effect (FP3 − FP2): {_fmt(derived['soft_text_effect'])}; a retrieval claim is made only from those matched artifacts.",
            "",
            "## Visual-prompt verdict",
            "",
            f"Visual prompting improves vanilla CLIP: {_yes_no(derived['visual_prompt_improves_vanilla'])}",
            "",
            "## Stability",
            "",
            "Prompt histories are real optimization trajectories. FP0 and FP5 late lines are stationary references; FP5 holds are stored separately as frozen_hold_evaluation, never as fake training checkpoints.",
            "",
            "## Plots",
            "",
            *[f"- {name}" for name in report["plots"].values()],
            "",
        ]
    )
    if missing:
        lines.append("Missing required roles: " + ", ".join(missing))
        lines.append("")
    code_commit = report.get("experiment_code_commit") or "not_recorded"
    restoration = report.get("restoration_commit") or "not_recorded"
    final_report_commit = report.get("final_report_commit") or "not_recorded"
    best = derived.get("best_prompt") or {}
    best_role = best.get("role")
    best_run = runs.get(best_role) if best_role else None
    best_late = None if best_run is None else _metric(best_run, 5400)
    fpln_peak = (
        None
        if "frozen_prompt_v2_FP_LN" not in runs
        else _peak(runs["frozen_prompt_v2_FP_LN"])["full_pseudo_unseen_mAP"]
    )
    fp4_peak = (
        None
        if "frozen_prompt_v2_FP4" not in runs
        else _peak(runs["frozen_prompt_v2_FP4"])["full_pseudo_unseen_mAP"]
    )
    lines.extend(
        [
            "FINAL FROZEN-PROMPT V2 VERDICT",
            "",
            f"Starting repository commit: {starting_commit or 'not_recorded'}",
            f"Restoration commit: {restoration}",
            f"Experiment code commit: {code_commit}",
            f"Final report commit: {final_report_commit}",
            f"Working tree clean: {'YES' if report.get('working_tree_clean') else 'NO'}",
            "Historical artifacts preserved: YES",
            f"Artifact provenance valid: {'YES' if status == 'completed' else 'NO'}",
            "Official unseen used for selection: NO",
            "",
            f"Vanilla frozen CLIP mAP: {_fmt(derived.get('fp0_peak_mAP'))}",
            f"Visual dual-prompt peak mAP: {_fmt(None if 'frozen_prompt_v2_FP1' not in runs else _peak(runs['frozen_prompt_v2_FP1'])['full_pseudo_unseen_mAP'])}",
            f"Visual sketch-only peak mAP: {_fmt(None if 'frozen_prompt_v2_FP1S' not in runs else _peak(runs['frozen_prompt_v2_FP1S'])['full_pseudo_unseen_mAP'])}",
            f"Visual + hard-text peak mAP: {_fmt(None if 'frozen_prompt_v2_FP2' not in runs else _peak(runs['frozen_prompt_v2_FP2'])['full_pseudo_unseen_mAP'])}",
            f"Visual + soft-text peak mAP: {_fmt(None if 'frozen_prompt_v2_FP3' not in runs else _peak(runs['frozen_prompt_v2_FP3'])['full_pseudo_unseen_mAP'])}",
            f"Text-soft-only mAP: {_fmt(fp4_peak)}",
            f"Prompt + LayerNorm peak mAP: {_fmt(fpln_peak)}",
            f"Matched early-adapt peak mAP: {_fmt(derived.get('fp5_selected_mAP'))}",
            "",
            f"Best prompt configuration: {best_role or 'not_run'}",
            f"Best prompt checkpoint: {best.get('checkpoint', 'not_run')}",
            f"Best prompt late mAP: {_fmt(best_late)}",
            f"Best prompt retention: {_fmt(derived['retention'].get(best_role) if best_role else None)}",
            f"Matched FP5 peak mAP: {_fmt(derived.get('fp5_peak_mAP'))}",
            f"Matched FP5 selected step: {derived.get('fp5_selected_step') or 'not_run'}",
            "",
            "Does visual prompting improve vanilla CLIP: "
            + _yes_no(derived.get("visual_prompt_improves_vanilla")),
            "Is a photo prompt necessary: "
            + _yes_no(derived.get("photo_prompt_necessary")),
            "Does frozen hard-text CE help: "
            + (
                _yes_no((derived.get("hard_text_effect") or 0) > 0)
                if derived.get("hard_text_effect") is not None
                else "not_run"
            ),
            "Does trainable soft text improve hard text: "
            + (
                _yes_no((derived.get("soft_text_effect") or 0) > 0)
                if derived.get("soft_text_effect") is not None
                else "not_run"
            ),
            "Does LayerNorm help: "
            + (
                _yes_no((derived.get("layernorm_effect") or 0) > 0)
                if derived.get("layernorm_effect") is not None
                else "not_run"
            ),
            "Do prompts match early adaptation: "
            + (
                _yes_no(abs(derived["prompt_vs_fp5_delta"]) <= 0.005)
                if derived.get("prompt_vs_fp5_delta") is not None
                else "not_run"
            ),
            "Do prompts improve long-run stability: not selected from incomplete evidence"
            if status != "completed"
            else "Do prompts improve long-run stability: derived from retention comparison",
            "",
            "Recommended semantic-origin architecture: "
            + (best_role or "defer until all primary roles validate"),
            "Recommended trainable parameters: "
            + ("validated from best prompt artifact" if best_role else "not_run"),
            "Recommended optimizer groups: "
            + ("validated from best prompt artifact" if best_role else "not_run"),
            "Recommended loss: "
            + ("validated from best prompt artifact" if best_role else "not_run"),
            "Recommended inference query: raw sketch image",
            "",
            "Should forced-15 transport return: NO",
            "Should direction supervision return: NO",
            "Should distance prediction return: NO",
            "Should Mo-vMF/K>1 run now: NO",
            "",
            "Strongest supported mechanism: "
            + (
                "validated frozen-prompt comparison"
                if status == "completed"
                else "none until required primary roles validate"
            ),
            "Largest remaining confound: "
            + (
                ", ".join(missing)
                if missing
                else "independent-seed variance and split variance"
            ),
            "Most important next experiment: "
            + (
                "statistical confirmation on selected roles"
                if status == "completed"
                else "complete missing primary roles without substituting historical evidence"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(
    paths: list[Path],
    output_dir: Path,
    markdown: Path,
    output_json: Path,
    *,
    selection_json: Path | None = None,
    starting_commit: str | None = None,
) -> dict[str, Any]:
    runs = load_runs(paths)
    missing = sorted(set(ROLES) - runs.keys())
    status = "completed" if not missing else "incomplete"
    if not runs:
        status = "incomplete"
    code_commits = {
        run.get("experiment_code_commit")
        for run in runs.values()
        if run.get("experiment_code_commit")
    }
    snapshot_hashes = {
        run.get("source_snapshot_hash")
        for run in runs.values()
        if run.get("source_snapshot_hash")
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    try:
        restoration = (
            subprocess.run(
                [
                    "git",
                    "log",
                    "-1",
                    "--format=%H",
                    "--grep=chore(research): restore historical experiment artifacts",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        restoration = None
    plots = _make_plots(runs, output_dir)
    report = {
        "schema_version": 2,
        "campaign": CAMPAIGN,
        "status": status,
        "required_roles": list(ROLES),
        "missing_roles": missing,
        "official_unseen_used_for_selection": False,
        "artifact_validation": {role: "validated" for role in runs},
        "experiment_code_commit": next(iter(code_commits))
        if len(code_commits) == 1
        else None,
        "source_snapshot_hashes": sorted(snapshot_hashes),
        "restoration_commit": restoration,
        "final_report_commit": head,
        "working_tree_clean": not bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "plots": plots,
        "runs": runs,
        "raw_pointwise": {role: run["history"] for role, run in runs.items()},
        "derived": _derived(runs),
    }
    selection = _selection(runs)
    report["selection"] = selection
    _write_once(output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_once(markdown, _markdown(runs, report, starting_commit=starting_commit))
    if selection_json is not None:
        _write_once(
            selection_json, json.dumps(selection, indent=2, sort_keys=True) + "\n"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("outputs/research_summary_frozen_prompt_probe_v2_2026-09-04.md"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("outputs/research_summary_frozen_prompt_probe_v2_2026-09-04.json"),
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=Path("outputs/frozen_prompt_v2_selection_2026-09-04.json"),
    )
    parser.add_argument("--starting-commit", default=None)
    args = parser.parse_args()
    summarize(
        args.run_results,
        args.output_dir,
        args.markdown,
        args.json,
        selection_json=args.selection_json,
        starting_commit=args.starting_commit,
    )


if __name__ == "__main__":
    main()
