"""Sketch-only evaluation for Predictive Semantic Transport checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from .config.data import load_data_config
from .data.datasets import RetrievalEvalDataset
from .data.manifest import read_manifest
from .evaluation.embeddings import load_encoded_retrieval_set
from .evaluation.transport import encode_transport_loader, evaluate_transport_features, transport_probe_dict
from .evaluate_deterministic import (
    _resolve_device,
    _validate_cache_against_manifest,
    _validate_provenance,
    _validate_zero_shot_split,
)
from .models.clip import FrozenVisualProjection, load_trainable_sketch_hidden_encoder
from .models.transport import SpicaPredictiveTransport
from .train_transport import HYDRA_CONFIG_DIR, PROJECT_ROOT, _radius_ap_payload


def _resolve_project_path(configured_path: object) -> Path:
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[SpicaPredictiveTransport, dict[str, object], object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"transport checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("invalid transport checkpoint payload")
    required = {"format_version", "model_type", "step", "model_config", "model_state_dict", "metadata"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"transport checkpoint is missing keys: {sorted(missing)}")
    if payload["format_version"] != 1 or payload["model_type"] != "predictive_semantic_transport":
        raise ValueError(f"unsupported transport checkpoint type/version: {payload.get('model_type')!r}")
    config = payload["model_config"]
    state = payload["model_state_dict"]
    metadata = payload["metadata"]
    if not isinstance(config, dict) or not isinstance(state, dict) or not isinstance(metadata, dict):
        raise TypeError("transport checkpoint config, state, and metadata must be dictionaries")
    if metadata.get("text_enters_predictor") is not False or metadata.get("text_enters_gate") is not False or metadata.get("text_enters_distance_head") is not False or metadata.get("text_enters_vmf") is not False:
        raise ValueError("transport checkpoint violates the text-free predictor contract")
    try:
        hidden_dim = int(config["hidden_dim"])
        embedding_dim = int(config["embedding_dim"])
        predictor_hidden_dim = int(config["predictor_hidden_dim"])
        mode = str(config["encoder_mode"])
        depth = int(config["encoder_unfreeze_depth"])
        model_name = str(config["encoder_model_name"])
        pretrained_value = config["encoder_pretrained"]
        pretrained = None if pretrained_value is None else str(pretrained_value)
        transport_mode = str(config["transport_mode"])
        num_components = int(config["K"])
        use_z0 = bool(config["use_z0"])
        rho_mode = str(config["rho_mode"])
        use_vmf = bool(config["use_vmf"])
    except KeyError as error:
        raise ValueError(f"transport model_config is missing {error.args[0]!r}") from error
    matrix = state.get("photo_projection.matrix")
    bias = state.get("photo_projection.bias")
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("transport checkpoint does not contain frozen photo projection")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise ValueError("transport checkpoint photo projection bias is invalid")
    projection = FrozenVisualProjection(matrix, bias)
    sketch_bundle = load_trainable_sketch_hidden_encoder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        mode=mode,
        unfreeze_depth=depth,
    )
    if sketch_bundle.encoder.hidden_dim != hidden_dim or projection.embedding_dim != embedding_dim:
        raise ValueError("checkpoint dimensions do not match the CLIP visual architecture")
    model = SpicaPredictiveTransport(
        sketch_bundle.encoder,
        projection,
        transport_mode=transport_mode,
        predictor_hidden_dim=predictor_hidden_dim,
        num_components=num_components,
        use_z0=use_z0,
        alpha=float(config.get("alpha", 1.0)),
        alpha_max=float(config.get("alpha_max", 0.5)),
        initial_alpha=float(config.get("initial_alpha", 0.0)),
        rho_max=float(config.get("rho_max", 0.25)),
        initial_rho=float(config.get("initial_rho", 0.0)),
        shared_rho=rho_mode == "shared",
        use_vmf=use_vmf,
        transport_enabled=bool(config.get("transport_enabled", True)),
        min_kappa=float(config.get("min_kappa", 1e-4)),
        max_kappa=float(config.get("max_kappa", 2048.0)),
        initial_kappa=float(config.get("initial_kappa", 64.0)),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("transport checkpoint state does not match model configuration") from error
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"transport checkpoint contains a non-finite parameter: {name}")
    model.to(device).eval()
    return model, payload, sketch_bundle.transform


def _metrics_dict(evaluation) -> dict[str, object]:
    return {
        "mAP": evaluation.metrics.mean_average_precision,
        "precision_at_k": evaluation.metrics.precision_at_k,
        "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
        "num_queries": evaluation.metrics.num_queries,
        "num_gallery_items": evaluation.metrics.num_gallery_items,
    }


@hydra.main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="evaluate_transport")
def main(args: DictConfig) -> None:
    device = _resolve_device(str(args.device))
    checkpoint_path = _resolve_project_path(args.checkpoint_path)
    embedding_dir = _resolve_project_path(args.embedding_dir)
    data_path = _resolve_project_path(args.data_config)
    model, checkpoint, sketch_transform = _load_model(checkpoint_path, device=device)
    gallery = load_encoded_retrieval_set(embedding_dir / "photos.pt")
    sketches_cache = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
    data = load_data_config(data_path)
    split_identities = _validate_zero_shot_split(
        data_path,
        dataset_name=str(args.dataset_name),
        sketches=sketches_cache,
        photos=gallery,
    )
    pretrained_value = checkpoint["metadata"].get("pretrained")
    _validate_provenance(
        checkpoint=checkpoint,
        sketches=sketches_cache,
        photos=gallery,
        dataset_name=str(args.dataset_name),
        model_name=str(args.model_name),
        pretrained=None if pretrained_value is None else str(pretrained_value),
    )
    test_entries = read_manifest(data.test.sketch_manifest, data.root)
    _validate_cache_against_manifest(
        modality="sketch", encoded_set=sketches_cache, manifest_entries=test_entries
    )
    query_loader = DataLoader(
        RetrievalEvalDataset(test_entries, sketch_transform),
        batch_size=int(args.prediction_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
    )
    features = encode_transport_loader(model, query_loader, device=device)
    modes = ("barycentric", "angular_logsumexp", "max") if model.num_components > 1 else ("barycentric",)
    evaluations = evaluate_transport_features(
        features,
        gallery,
        modes=modes,
        temperature=float(args.score_temperature),
        precision_at_k=tuple(int(k) for k in args.precision_at_k),
        map_at_k=tuple(int(k) for k in args.map_at_k),
        map_at_k_denominator=str(args.map_at_k_denominator),
        query_chunk_size=int(args.query_chunk_size),
        device=device,
    )
    selected = str(args.inference_score_mode)
    if selected not in evaluations:
        raise ValueError(f"inference_score_mode {selected!r} is not available")
    probe = transport_probe_dict(
        features,
        gallery,
        frozen_reference=sketches_cache.embeddings,
        kappa_max=float(checkpoint["model_config"].get("max_kappa", 2048.0)),
    )
    output = Path(HydraConfig.get().runtime.output_dir) / "metrics.json"
    result = {
        "protocol": "standard_zs_sbir_sketch_only_inference",
        "diagnostic_test_evaluation": True,
        "model_family": "predictive_semantic_transport",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "training_metadata": checkpoint["metadata"],
        "split_identities": split_identities,
        "inference_score_mode": selected,
        "metrics": _metrics_dict(evaluations[selected]),
        "retrieval_modes": {name: _metrics_dict(value) for name, value in evaluations.items()},
        "radius_vs_ap": _radius_ap_payload(evaluations[selected], features.rho),
        "feature_geometry": probe,
        "leakage_flags": {
            "text_used_at_inference": False,
            "text_enters_predictor": False,
            "text_enters_gate": False,
            "text_enters_distance_head": False,
            "text_enters_vmf": False,
            "photo_used_at_inference": False,
            "positive_set_used_at_inference": False,
            "oracle_class_used_at_inference": False,
            "gallery_reencoded": False,
            "test_labels_used_for_scoring": False,
        },
        "inference_contract": {
            "inputs": ["raw_sketch_image"],
            "query_path": [
                "raw sketch image",
                "trainable CLIP-initialized pre-projection sketch context encoder",
                "frozen photo-CLIP semantic projection to z0",
                "predictive tangent/residual transport",
                "L2-normalized photo-compatible query",
            ],
        },
        "map_at_k_denominator": str(args.map_at_k_denominator),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"mAP ({selected}): {evaluations[selected].metrics.mean_average_precision:.6f}")
    if 200 in evaluations[selected].metrics.precision_at_k:
        print(f"P@200 ({selected}): {evaluations[selected].metrics.precision_at_k[200]:.6f}")
    print("Text required at inference: NO")
    print("Photo required at inference: NO")
    print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
