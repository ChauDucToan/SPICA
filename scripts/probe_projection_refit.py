"""P3: refit sketch-to-CLIP compatibility maps on pseudo-train only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.data.splits import split_manifest_identity
from spica.evaluation.embeddings import EncodedRetrievalSet, encode_retrieval_loader
from spica.evaluation.metrics import evaluate_category_retrieval
from spica.evaluation.transport import hidden_space_compatibility
from spica.evaluate_transport import _load_model
from spica.models.clip import (
    frozen_visual_projection,
    load_frozen_clip,
    visual_pre_projection,
)
from spica.provenance import capture_provenance, hash_json
from spica.train_transport import PROJECT_ROOT, _build_split

CONTROL_ROLES = {"freeze_optimizer_B", "freeze_optimizer_D"}
P3_CAMPAIGN = "transport_corrected_2026-09-02_v2"
P3_ROLES = {
    "freeze_optimizer_source",
    "freeze_optimizer_A",
    "freeze_optimizer_B",
    "freeze_optimizer_C",
    "freeze_optimizer_D",
}
CONTROL_FOR_BRANCH = {
    "freeze_optimizer_A": "freeze_optimizer_B",
    "freeze_optimizer_C": "freeze_optimizer_D",
}


def _loader(entries, transform, batch_size: int) -> DataLoader:
    return DataLoader(
        RetrievalEvalDataset(entries, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


@torch.inference_mode()
def _hidden(loader: DataLoader, model, reference_visual, device: torch.device):
    adapted, reference, labels, paths = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        adapted.append(model.sketch_context_encoder(images).float().cpu())
        reference.append(visual_pre_projection(reference_visual, images).float().cpu())
        labels.append(batch["label"].long())
        paths.extend(str(path) for path in batch["path"])
    return torch.cat(adapted), torch.cat(reference), torch.cat(labels), tuple(paths)


def _effective_rank(values: torch.Tensor) -> float:
    centered = values.float() - values.float().mean(0, keepdim=True)
    if centered.shape[0] > 4096:
        # ponytail: sample the rank diagnostic; use full SVD only if rank itself is a result.
        centered = centered[:: max(1, centered.shape[0] // 4096)]
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    return float(energy.sum().square().div(energy.square().sum().clamp_min(1e-12)))


def _split_identity(data, split) -> dict[str, Any]:
    payload = {
        "dataset": data.name,
        "pseudo_validation_seed": split.seed,
        "train_class_ids": list(split.train_class_ids),
        "validation_class_ids": list(split.validation_class_ids),
        "train_sketches": len(split.train_sketch_entries),
        "train_photos": len(split.train_photo_entries),
        "validation_sketches": len(split.validation_sketch_entries),
        "validation_photos": len(split.validation_photo_entries),
    }
    return {**payload, "sha256": hash_json(payload)}


def _evaluate(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    paths: tuple[str, ...],
    gallery: EncodedRetrievalSet,
    device: torch.device,
) -> dict[str, object]:
    evaluation = evaluate_category_retrieval(
        EncodedRetrievalSet(
            embeddings=F.normalize(embeddings, dim=-1), labels=labels, paths=paths
        ),
        gallery,
        precision_at_k=(1, 5, 10, 100, 200),
        map_at_k=(200,),
        map_at_k_denominator="prefix_positive",
        top_k=200,
        device=device,
    )
    return {
        "mAP": evaluation.metrics.mean_average_precision,
        "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
        "precision_at_k": evaluation.metrics.precision_at_k,
        "per_query_AP": evaluation.average_precision_per_query.tolist(),
        "effective_rank": _effective_rank(embeddings),
    }


def _alignment(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = predicted.float()
    target = target.float()
    if predicted.shape != target.shape:
        raise ValueError("alignment tensors must have matching shapes")
    cosine = F.cosine_similarity(predicted, target, dim=-1)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(unbiased=False)),
        "mse": float((predicted - target).square().mean()),
    }


def _ridge_fit(
    train_h: torch.Tensor, target: torch.Tensor, strength: float
) -> torch.Tensor:
    x = torch.cat((train_h.float(), torch.ones(train_h.shape[0], 1)), dim=1)
    gram = x.T @ x
    regularizer = torch.eye(gram.shape[0], dtype=x.dtype) * strength
    regularizer[-1, -1] = 0
    return torch.linalg.solve(gram + regularizer, x.T @ target.float())


def _singular_summary(matrix: torch.Tensor) -> dict[str, float]:
    values = torch.linalg.svdvals(matrix.float())
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def _fit_methods(
    train_h: torch.Tensor,
    train_ref: torch.Tensor,
    val_h: torch.Tensor,
    projection,
    ridge_strength: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    cross = train_h.T @ train_ref
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    train_target = F.normalize(projection(train_ref), dim=-1)
    ridge = _ridge_fit(train_h, train_target, ridge_strength)
    val_x = torch.cat((val_h.float(), torch.ones(val_h.shape[0], 1)), dim=1)
    projected = {
        "frozen_original_W_CLIP": projection(val_h),
        "orthogonal_adapter": projection(val_h @ rotation),
        "ridge_projection": val_x @ ridge,
    }
    diagnostics = {
        "orthogonal": {
            "rotation_singular_values": _singular_summary(rotation),
            "rotation_frobenius_displacement": float(
                (rotation - torch.eye(rotation.shape[0])).norm()
            ),
            "train_hidden_alignment": _alignment(train_h @ rotation, train_ref),
        },
        "ridge": {
            "weight_singular_values": _singular_summary(ridge[:-1]),
            "weight_frobenius_norm": float(ridge[:-1].norm()),
            "bias_norm": float(ridge[-1].norm()),
            "train_target_alignment": _alignment(
                torch.cat((train_h, torch.ones(train_h.shape[0], 1)), dim=1) @ ridge,
                train_target,
            ),
        },
        "fit_target": "frozen photo-CLIP visual hidden state of the same raw sketch",
    }
    return projected, {
        "rotation": rotation,
        "ridge_matrix": ridge,
        "diagnostics": diagnostics,
    }


def _method_results(
    methods: dict[str, torch.Tensor],
    *,
    val_ref: torch.Tensor,
    val_labels: torch.Tensor,
    val_paths: tuple[str, ...],
    gallery: EncodedRetrievalSet,
    projection,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    reference = F.normalize(projection(val_ref), dim=-1)
    results: dict[str, dict[str, object]] = {}
    for name, embeddings in methods.items():
        result = _evaluate(embeddings, val_labels, val_paths, gallery, device)
        result["frozen_W_compatibility_cosine"] = float(
            F.cosine_similarity(
                F.normalize(embeddings, dim=-1), reference, dim=-1
            ).mean()
        )
        results[name] = result
    reference_result = _evaluate(
        projection(val_ref), val_labels, val_paths, gallery, device
    )
    reference_result["frozen_W_compatibility_cosine"] = 1.0
    results["frozen_sketch_reference_W_CLIP"] = reference_result
    return results


def _add_control_comparisons(values: list[dict[str, Any]]) -> None:
    controls = {
        (row["experiment_role"], int(row["checkpoint_step"])): row
        for row in values
        if row.get("experiment_role") in CONTROL_ROLES
    }
    for row in values:
        control_role = CONTROL_FOR_BRANCH.get(str(row.get("experiment_role")))
        if control_role is None:
            continue
        control = controls.get((control_role, int(row["checkpoint_step"])))
        if control is None:
            row["matched_frozen_control"] = {
                "status": "not_available",
                "required_role": control_role,
            }
            continue
        control_methods = {
            method: float(control["methods"][method]["mAP"])
            for method in row["methods"]
            if method in control["methods"]
        }
        original_gap = (
            float(row["methods"]["frozen_original_W_CLIP"]["mAP"])
            - control_methods["frozen_original_W_CLIP"]
        )
        row["matched_frozen_control"] = {
            "roles": [control_role],
            "mAP_by_method": control_methods,
            "mAP_minus_control": {
                method: float(row["methods"][method]["mAP"]) - value
                for method, value in control_methods.items()
            },
            "original_gap": original_gap,
            "recovery_fraction_relative_to_matched_method_control": {
                method: None
                if abs(original_gap) < 1e-12
                else 1.0
                - (float(row["methods"][method]["mAP"]) - control_methods[method])
                / original_gap
                for method in row["methods"]
                if method in control_methods
            },
        }


def _plot(values: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    figure, axis = plt.subplots(figsize=(9, 5))
    for row in values:
        role = row.get("experiment_role") or "unknown"
        step = int(row["checkpoint_step"])
        for method in (
            "frozen_original_W_CLIP",
            "orthogonal_adapter",
            "ridge_projection",
        ):
            axis.scatter(
                step,
                float(row["methods"][method]["mAP"]),
                label=f"{role}/{method}"
                if step == int(row["checkpoint_step"])
                else None,
            )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    axis.set_xlabel("checkpoint global step")
    axis.set_ylabel("pseudo-unseen mAP")
    axis.set_title("P3 projection-refit controls")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ridge <= 0:
        raise ValueError("ridge must be positive")
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    plot_path = output.with_suffix(".png")
    if output.exists() or plot_path.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {plot_path}")
    device = torch.device(args.device)
    data_path = (
        args.data_config
        if args.data_config.is_absolute()
        else PROJECT_ROOT / args.data_config
    )
    data = load_data_config(data_path)
    split, _ = _build_split(data, num_validation_classes=20, seed=3407)
    expected_split = _split_identity(data, split)
    expected_manifest_identity = split_manifest_identity(
        split,
        dataset_name=data.name,
        dataset_root=data.root,
        manifest_paths={
            "train_sketch": data.train.sketch_manifest,
            "train_photo": data.train.photo_manifest,
            "class_map": data.train.class_map,
        },
    )
    seen_checkpoints: set[tuple[str, int]] = set()

    first_checkpoint = args.checkpoints[0]
    first_checkpoint = (
        first_checkpoint
        if first_checkpoint.is_absolute()
        else PROJECT_ROOT / first_checkpoint
    )
    first_payload = torch.load(first_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(first_payload, dict) or not isinstance(
        first_payload.get("model_config"), dict
    ):
        raise ValueError("invalid first transport checkpoint")
    model_config = first_payload["model_config"]
    model_name = str(model_config["encoder_model_name"])
    pretrained_value = model_config.get("encoder_pretrained")
    pretrained = None if pretrained_value is None else str(pretrained_value)
    photo_clip = load_frozen_clip(
        model_name=model_name, pretrained=pretrained, device=device
    )
    photo_projection = frozen_visual_projection(photo_clip.encoder).to(device)
    gallery = encode_retrieval_loader(
        photo_clip.encoder,
        _loader(split.validation_photo_entries, photo_clip.transform, args.batch_size),
    )
    values: list[dict[str, Any]] = []
    for configured_checkpoint in args.checkpoints:
        checkpoint = (
            configured_checkpoint
            if configured_checkpoint.is_absolute()
            else PROJECT_ROOT / configured_checkpoint
        )
        model, payload, transform = _load_model(checkpoint, device=device)
        checkpoint_model_config = payload.get("model_config")
        if not isinstance(checkpoint_model_config, dict):
            raise ValueError(f"checkpoint model_config is missing: {checkpoint}")
        for key in (
            "hidden_dim",
            "embedding_dim",
            "encoder_model_name",
            "encoder_pretrained",
            "encoder_mode",
            "encoder_unfreeze_depth",
        ):
            if checkpoint_model_config.get(key) != model_config.get(key):
                raise ValueError(f"checkpoint model identity differs: {checkpoint}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"checkpoint metadata is missing: {checkpoint}")
        split_identity = payload.get("data_split_identity")
        if split_identity != expected_split:
            raise ValueError(
                f"checkpoint split differs from the current pseudo split: {checkpoint}"
            )
        resolved = payload.get("resolved_config")
        if (
            not isinstance(resolved, dict)
            or resolved.get("train_class_scope") != "pseudo_train"
        ):
            raise ValueError(
                f"checkpoint was not trained on pseudo-train classes: {checkpoint}"
            )
        if metadata.get("data_split_identity") != expected_split:
            raise ValueError(
                f"checkpoint metadata split differs from the current pseudo split: {checkpoint}"
            )
        checkpoint_manifest_identity = payload.get("data_manifest_identity")
        if (
            checkpoint_manifest_identity is not None
            and checkpoint_manifest_identity != expected_manifest_identity
        ):
            raise ValueError(
                f"checkpoint manifest entries differ from the current pseudo split: {checkpoint}"
            )
        metadata_manifest_identity = metadata.get("data_manifest_identity")
        if (
            metadata_manifest_identity is not None
            and metadata_manifest_identity != expected_manifest_identity
        ):
            raise ValueError(
                f"checkpoint metadata manifest entries differ from the current pseudo split: {checkpoint}"
            )
        if metadata.get("pseudo_validation_class_ids") != list(
            split.validation_class_ids
        ):
            raise ValueError(
                f"checkpoint pseudo-validation classes differ: {checkpoint}"
            )
        role = metadata.get("experiment_role")
        campaign = metadata.get("experiment_campaign")
        if role not in P3_ROLES or campaign != P3_CAMPAIGN:
            raise ValueError(
                f"checkpoint role/campaign is not a corrected P1 artifact: {checkpoint}"
            )
        step = int(payload.get("step", -1))
        identity = (str(role), step)
        if identity in seen_checkpoints:
            raise ValueError(f"duplicate P3 role/checkpoint: {identity}")
        seen_checkpoints.add(identity)
        if role == "freeze_optimizer_source" and step != 73:
            raise ValueError("P3 source must be the step-73 checkpoint")
        if role != "freeze_optimizer_source" and step not in {500, 1800, 5400}:
            raise ValueError(
                "P3 branch checkpoints must be at steps 500, 1800, or 5400"
            )
        bias = model.photo_projection.bias
        reference_bias = photo_projection.bias
        bias_matches = (bias is None and reference_bias is None) or (
            bias is not None
            and reference_bias is not None
            and torch.allclose(bias.cpu(), reference_bias.cpu(), atol=1e-6, rtol=0)
        )
        if not (
            torch.allclose(
                model.photo_projection.matrix.cpu(),
                photo_projection.matrix.cpu(),
                atol=1e-6,
                rtol=0,
            )
            and bias_matches
        ):
            raise ValueError(f"checkpoint/photo CLIP projection mismatch: {checkpoint}")
        train_h, train_ref, train_labels, _ = _hidden(
            _loader(split.train_sketch_entries, transform, args.batch_size),
            model,
            photo_clip.encoder.model.visual,
            device,
        )
        val_h, val_ref, val_labels, val_paths = _hidden(
            _loader(split.validation_sketch_entries, transform, args.batch_size),
            model,
            photo_clip.encoder.model.visual,
            device,
        )
        if set(train_labels.tolist()) & set(val_labels.tolist()):
            raise ValueError("projection refit train and validation classes overlap")
        projection = model.photo_projection.cpu()
        methods, fit = _fit_methods(
            train_h,
            train_ref,
            val_h,
            projection,
            float(args.ridge),
        )
        method_results = _method_results(
            methods,
            val_ref=val_ref,
            val_labels=val_labels,
            val_paths=val_paths,
            gallery=gallery,
            projection=projection,
            device=device,
        )
        val_x = torch.cat((val_h, torch.ones(val_h.shape[0], 1)), dim=1)
        rotation = fit["rotation"]
        ridge = fit["ridge_matrix"]
        fit_diagnostics = {
            "orthogonal": {
                **fit["diagnostics"]["orthogonal"],
                "validation_hidden_alignment": _alignment(val_h @ rotation, val_ref),
            },
            "ridge": {
                **fit["diagnostics"]["ridge"],
                "validation_target_alignment": _alignment(
                    val_x @ ridge, F.normalize(projection(val_ref), dim=-1)
                ),
            },
            "fit_target": fit["diagnostics"]["fit_target"],
        }
        values.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_step": step,
                "experiment_role": role,
                "experiment_campaign": campaign,
                "fit_split": "pseudo_train only",
                "evaluation_split": "pseudo_unseen only",
                "fit_class_ids": list(split.train_class_ids),
                "evaluation_class_ids": list(split.validation_class_ids),
                "data_split_identity": split_identity,
                "data_manifest_identity": expected_manifest_identity,
                "methods": method_results,
                "fit_diagnostics": fit_diagnostics,
                # This helper fits an oracle rotation on validation rows only as
                # a descriptive geometry diagnostic; it is never used by a map.
                "hidden_geometry_oracle_validation": hidden_space_compatibility(
                    val_ref, val_h, projection
                ),
                "checkpoint_provenance": metadata.get("provenance"),
            }
        )
    _add_control_comparisons(values)
    result = {
        "schema_version": 2,
        "probe": "projection_refit_control",
        "selection_metric": None,
        "seed": 3407,
        "official_unseen_used": False,
        "target_space": "frozen photo-CLIP retrieval embedding space",
        "fit_target": "frozen photo-CLIP visual hidden state of the same raw sketch; no pseudo-validation rows",
        "data_manifest_identity": expected_manifest_identity,
        "ridge": args.ridge,
        "source_artifacts": [
            {
                "path": str(
                    (
                        checkpoint
                        if checkpoint.is_absolute()
                        else PROJECT_ROOT / checkpoint
                    ).resolve()
                ),
                "sha256": _sha256(
                    checkpoint
                    if checkpoint.is_absolute()
                    else PROJECT_ROOT / checkpoint
                ),
            }
            for checkpoint in args.checkpoints
        ],
        "values": values,
        "controls": {
            "roles": sorted(CONTROL_ROLES),
            "comparison": "A↔B and C↔D at the identical global checkpoint step; no pooling across reset conditions",
        },
        "provenance": capture_provenance(
            PROJECT_ROOT, command=[str(value) for value in __import__("sys").argv]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _plot(values, plot_path)
    print(output)
    print(plot_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
