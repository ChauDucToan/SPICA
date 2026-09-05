"""Measure class-conditional CLIP-sphere geometry without fitting anything."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.data.manifest import read_class_map, read_manifest
from spica.data.splits import make_classwise_retrieval_split
from spica.evaluation.frozen_prompt import encode_prompted_loader
from spica.evaluation.text_bank import encode_class_text_bank
from spica.models.alignment import class_conditional_geometry_diagnostics
from spica.models.checkpoint import load_prompt_checkpoint
from spica.models.clip import load_frozen_clip
from spica.models.frozen_prompt import FrozenPromptModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _select(entries: Any, max_per_class: int) -> tuple[Any, ...]:
    grouped: dict[int, list[Any]] = {}
    for entry in sorted(entries, key=lambda value: (int(value.label), str(value.path))):
        grouped.setdefault(int(entry.label), []).append(entry)
    return tuple(
        entry
        for label in sorted(grouped)
        for entry in grouped[label][:max_per_class]
    )


def _encode(
    entries: Any,
    transform: Any,
    model: Any,
    batch_size: int,
    *,
    photo: bool,
) -> Any:
    loader = DataLoader(
        RetrievalEvalDataset(entries, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return encode_prompted_loader(model, loader, photo=photo)


def _encode_diagnostic_sets(
    train_sketches: Any,
    train_photos: Any,
    val_sketches: Any,
    val_photos: Any,
    transform: Any,
    model: Any,
    batch_size: int,
) -> dict[str, Any]:
    """Encode every set with explicit modality routing, including bug controls."""
    return {
        "train_sketch": _encode(
            train_sketches, transform, model, batch_size, photo=False
        ),
        "train_photo": _encode(
            train_photos, transform, model, batch_size, photo=True
        ),
        "validation_sketch": _encode(
            val_sketches, transform, model, batch_size, photo=False
        ),
        "validation_photo": _encode(
            val_photos, transform, model, batch_size, photo=True
        ),
        "train_photo_wrong_routing": _encode(
            train_photos, transform, model, batch_size, photo=False
        ),
        "validation_photo_wrong_routing": _encode(
            val_photos, transform, model, batch_size, photo=False
        ),
    }


def _sample_identity(entries: Any) -> dict[str, Any]:
    rows = [
        {"id": str(entry.path), "label": int(entry.label)}
        for entry in entries
    ]
    ids = [row["id"] for row in rows]
    return {
        "rows": rows,
        "count": len(rows),
        "unique_count": len(set(ids)),
        "duplicate_count": len(ids) - len(set(ids)),
    }


def _legacy_log_map(anchor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Historical diagnostic formula; never used by training."""
    base = torch.nn.functional.normalize(anchor, dim=-1)
    values = torch.nn.functional.normalize(points, dim=-1)
    raw_cosine = (values * base).sum(dim=-1)
    cosine = raw_cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cosine)
    sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
    factor = angle / sine.clamp_min(1e-6)
    tangent = values - raw_cosine.unsqueeze(-1) * base
    near_anchor = (1.0 - raw_cosine).le(1e-6)
    return torch.where(near_anchor.unsqueeze(-1), torch.zeros_like(tangent), tangent * factor.unsqueeze(-1))


def _legacy_moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    centered = values - mean
    return mean, centered.T @ centered / values.shape[0]


def _legacy_geometry(
    sketches: Any,
    photos: Any,
    text: Any,
) -> dict[str, Any]:
    """Recompute the old routing/estimator diagnostic on the same subset."""
    sketch_values = torch.nn.functional.normalize(sketches.embeddings.float(), dim=-1)
    photo_values = torch.nn.functional.normalize(photos.embeddings.float(), dim=-1)
    text_values = torch.nn.functional.normalize(text.embeddings.float(), dim=-1)
    positions = {int(label): index for index, label in enumerate(text.labels.tolist())}
    mean_distances: list[torch.Tensor] = []
    covariance_norms: list[torch.Tensor] = []
    sketch_angles: list[torch.Tensor] = []
    photo_angles: list[torch.Tensor] = []
    orthogonality: list[torch.Tensor] = []
    class_count = 0
    for label in torch.unique(sketches.labels, sorted=True):
        sketch = sketch_values[sketches.labels == label]
        photo = photo_values[photos.labels == label]
        if sketch.shape[0] < 2 or photo.shape[0] < 2:
            continue
        anchor = text_values[positions[int(label)]]
        sketch_log = _legacy_log_map(anchor, sketch)
        photo_log = _legacy_log_map(anchor, photo)
        sketch_mean, sketch_cov = _legacy_moments(sketch_log)
        photo_mean, photo_cov = _legacy_moments(photo_log)
        mean_distances.append((sketch_mean - photo_mean).norm())
        covariance_norms.append((sketch_cov - photo_cov).norm())
        sketch_angles.append(sketch_log.norm(dim=-1).mean())
        photo_angles.append(photo_log.norm(dim=-1).mean())
        orthogonality.append((sketch_log * anchor).sum(dim=-1).abs().mean())
        class_count += 1
    if not class_count:
        return {
            "class_count": 0,
            "mean_distance": None,
            "covariance_frobenius_distance": None,
            "sketch_log_angle": None,
            "photo_log_angle": None,
            "sketch_anchor_orthogonality": None,
            "estimator": "population_covariance_n",
            "covariance_distance": "squared_frobenius_norm",
        }
    return {
        "class_count": class_count,
        "mean_distance": float(torch.stack(mean_distances).mean().item()),
        "covariance_frobenius_distance": float(torch.stack(covariance_norms).mean().item()),
        "sketch_log_angle": float(torch.stack(sketch_angles).mean().item()),
        "photo_log_angle": float(torch.stack(photo_angles).mean().item()),
        "sketch_anchor_orthogonality": float(torch.stack(orthogonality).mean().item()),
        "estimator": "population_covariance_n",
        "covariance_distance": "squared_frobenius_norm",
    }


def _historical_reference(
    checkpoint_path: Path, role: str | None
) -> dict[str, Any] | None:
    candidates = {
        "frozen_prompt_v2_FP3": Path("outputs/alignment_geometry_baseline_2026-09-05.json"),
        "alignment_control": Path("outputs/alignment_geometry_control_selected_2026-09-05.json"),
        "alignment_full_text_log": Path("outputs/alignment_geometry_full_selected_2026-09-05.json"),
    }
    path = candidates.get(str(role))
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return {
        "status": "HISTORICAL_REFERENCE_ONLY",
        "artifact": str(path),
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "same_checkpoint": str(Path(payload.get("checkpoint", "")).resolve())
        == str(checkpoint_path.resolve()),
        "geometry": payload.get("train"),
        "pseudo_unseen_geometry": payload.get("pseudo_unseen"),
    }


def _companion_checkpoint_hash(checkpoint_path: Path) -> str | None:
    run_result = checkpoint_path.parent.parent / "run_result.json"
    if not run_result.is_file():
        return None
    payload = json.loads(run_result.read_text())
    candidates = list(payload.get("history", []))
    candidates.extend(payload.get("checkpoints", {}).values())
    for row in candidates:
        if str(Path(row.get("checkpoint", "")).resolve()) == str(checkpoint_path.resolve()):
            value = row.get("checkpoint_sha256")
            return str(value) if value else None
    return None


def run(
    checkpoint_path: Path,
    data_config_path: Path,
    output_path: Path,
    device: str,
    max_per_class: int,
    batch_size: int,
) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")

    checkpoint_hash = _sha256(checkpoint_path)
    metadata_hash = _companion_checkpoint_hash(checkpoint_path)
    if metadata_hash is None:
        hash_status = "UNVERIFIED_NO_COMPANION_METADATA"
    elif metadata_hash != checkpoint_hash:
        hash_status = "MISMATCH"
        raise ValueError(
            f"checkpoint hash mismatch: metadata={metadata_hash}, actual={checkpoint_hash}"
        )
    else:
        hash_status = "VERIFIED"

    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, dict):
        raise ValueError("checkpoint resolved_config must be a mapping")
    device_value = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else device
    )
    clip = load_frozen_clip(
        model_name=str(resolved.get("model_name", "ViT-B-32-quickgelu")),
        pretrained=resolved.get("pretrained", "openai"),
        device=device_value,
    )
    model = FrozenPromptModel(
        clip.encoder.model.visual,
        prompt_length=int(resolved.get("visual_prompt_length", 3)),
        train_visual_layernorm=bool(resolved.get("train_visual_layernorm", False)),
        train_sketch_prompt=bool(resolved.get("train_sketch_prompt", True)),
        train_photo_prompt=bool(resolved.get("train_photo_prompt", True)),
    ).to(device_value)
    checkpoint_load = load_prompt_checkpoint(
        model,
        checkpoint,
        expected_config={
            "model_name": str(resolved["model_name"]),
            "pretrained": resolved.get("pretrained"),
            "visual_prompt_length": int(resolved["visual_prompt_length"]),
            "train_visual_layernorm": bool(resolved.get("train_visual_layernorm", False)),
            "train_sketch_prompt": bool(resolved.get("train_sketch_prompt", True)),
            "train_photo_prompt": bool(resolved.get("train_photo_prompt", True)),
        },
    )
    model.eval()

    data = load_data_config(data_config_path)
    names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches,
        photos,
        names,
        num_validation_classes=int(resolved.get("pseudo_val_num_classes", 20)),
        seed=int(resolved.get("pseudo_val_seed", 3407)),
    )

    train_sketches = _select(split.train_sketch_entries, max_per_class)
    train_photos = _select(split.train_photo_entries, max_per_class)
    val_sketches = _select(split.validation_sketch_entries, max_per_class)
    val_photos = _select(split.validation_photo_entries, max_per_class)
    encoded = _encode_diagnostic_sets(
        train_sketches,
        train_photos,
        val_sketches,
        val_photos,
        clip.transform,
        model,
        batch_size,
    )
    train_sketch_values = encoded["train_sketch"]
    train_photo_values = encoded["train_photo"]
    val_sketch_values = encoded["validation_sketch"]
    val_photo_values = encoded["validation_photo"]
    train_photo_wrong_routing = encoded["train_photo_wrong_routing"]
    val_photo_wrong_routing = encoded["validation_photo_wrong_routing"]
    train_names = {label: names[label] for label in split.train_class_ids}
    val_names = {label: names[label] for label in split.validation_class_ids}
    prompt_template = str(resolved.get("prompt_template", "a photo of a {}"))
    train_text = encode_class_text_bank(
        clip.encoder, clip.tokenizer, train_names, prompt_template=prompt_template
    )
    val_text = encode_class_text_bank(
        clip.encoder, clip.tokenizer, val_names, prompt_template=prompt_template
    )

    corrected_train = class_conditional_geometry_diagnostics(
        train_sketch_values.embeddings,
        train_photo_values.embeddings,
        train_sketch_values.labels,
        photo_labels=train_photo_values.labels,
        text_embeddings=train_text.embeddings,
        text_labels=train_text.labels,
    )
    corrected_val = class_conditional_geometry_diagnostics(
        val_sketch_values.embeddings,
        val_photo_values.embeddings,
        val_sketch_values.labels,
        photo_labels=val_photo_values.labels,
        text_embeddings=val_text.embeddings,
        text_labels=val_text.labels,
    )
    legacy_train = _legacy_geometry(train_sketch_values, train_photo_values, train_text)
    legacy_val = _legacy_geometry(val_sketch_values, val_photo_values, val_text)
    wrong_route_train = _legacy_geometry(
        train_sketch_values, train_photo_wrong_routing, train_text
    )
    wrong_route_val = _legacy_geometry(
        val_sketch_values, val_photo_wrong_routing, val_text
    )

    text_bank_hash = hashlib.sha256()
    for bank in (train_text, val_text):
        text_bank_hash.update(json.dumps({
            "labels": bank.labels.tolist(),
            "class_names": list(bank.class_names),
            "prompts": list(bank.prompts),
        }, sort_keys=True).encode())
        text_bank_hash.update(bank.embeddings.detach().cpu().contiguous().numpy().tobytes())
    result = {
        "schema_version": 2,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_verification": {
            "status": hash_status,
            "metadata_sha256": metadata_hash,
            "actual_sha256": checkpoint_hash,
        },
        "experiment_role": checkpoint.get("experiment_role"),
        "campaign": checkpoint.get("campaign"),
        "experiment_code_commit": checkpoint.get("experiment_code_commit"),
        "source_snapshot_hash": checkpoint.get("source_snapshot_hash"),
        "resolved_config_hash": hashlib.sha256(
            json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "device": str(device_value),
        "max_per_class": max_per_class,
        "checkpoint_loading": checkpoint_load,
        "backbone": {
            "model_name": clip.model_name,
            "pretrained": clip.pretrained,
            "identity": checkpoint_load["backbone_identity"],
            "preprocessing": repr(clip.transform),
        },
        "text_anchor": {
            "template": prompt_template,
            "train_bank_hash": _tensor_sha256(train_text.embeddings),
            "validation_bank_hash": _tensor_sha256(val_text.embeddings),
            "combined_text_bank_hash": text_bank_hash.hexdigest(),
        },
        "pseudo_split": {
            "seed": split.seed,
            "train_class_ids": list(split.train_class_ids),
            "validation_class_ids": list(split.validation_class_ids),
            "train_class_names": {str(label): names[label] for label in split.train_class_ids},
            "validation_class_names": {str(label): names[label] for label in split.validation_class_ids},
            "train_class_count": len(split.train_class_ids),
            "validation_class_count": len(split.validation_class_ids),
        },
        "samples": {
            "train_sketch": _sample_identity(train_sketches),
            "train_photo": _sample_identity(train_photos),
            "validation_sketch": _sample_identity(val_sketches),
            "validation_photo": _sample_identity(val_photos),
        },
        "modality_routing": {
            "train_sketch": "sketch_prompt",
            "train_photo": "photo_prompt",
            "validation_sketch": "sketch_prompt",
            "validation_photo": "photo_prompt",
            "historical_wrong_photo_branch": "sketch_prompt",
        },
        "variants": {
            "historical_reported": _historical_reference(
                checkpoint_path, checkpoint.get("experiment_role")
            ),
            "photo_routing_corrected_legacy_estimator": {
                "train": legacy_train,
                "pseudo_unseen": legacy_val,
                "estimator": "population_covariance_n",
                "covariance_distance": "squared_frobenius_norm",
            },
            "fully_corrected_geometry": {
                "train": corrected_train,
                "pseudo_unseen": corrected_val,
                "estimator": "sample_covariance_n_minus_1",
                "covariance_distance": "frobenius_norm",
                "invalid_policy": "near-antipodal samples excluded via validity mask and counted",
            },
            "historical_wrong_photo_routing_recomputed": {
                "train": wrong_route_train,
                "pseudo_unseen": wrong_route_val,
                "estimator": "population_covariance_n",
                "covariance_distance": "squared_frobenius_norm",
                "note": "comparison-only recreation of the original photo routing bug",
            },
        },
        "protocol": {
            "train_only_fitting": True,
            "validation_used_for_fitting": False,
            "official_unseen_used_for_selection": False,
            "text_used_only_as_diagnostic_anchor": True,
            "text_used_for_predictor": False,
            "photo_used_for_predictor": False,
            "anchor_template_source": "checkpoint resolved_config.prompt_template",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["variants"]["fully_corrected_geometry"]["pseudo_unseen"], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-per-class", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.max_per_class < 2:
        raise ValueError("--max-per-class must be at least 2")
    run(
        args.checkpoint,
        args.data_config,
        args.output,
        args.device,
        args.max_per_class,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
