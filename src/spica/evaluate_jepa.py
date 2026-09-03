"""Sketch-only inference and diagnostics for cross-modal JEPA checkpoints."""

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
from .evaluation.jepa import (
    encode_jepa_loader,
    evaluate_jepa_features,
    feature_probe_dict,
)
from .evaluate_deterministic import (
    _resolve_device,
    _validate_cache_against_manifest,
    _validate_provenance,
    _validate_zero_shot_split,
)
from .models.clip import load_trainable_sketch_encoder
from .models.jepa import SketchPhotoJepa, SpicaJepaPredictor
from .train_deterministic import HYDRA_CONFIG_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(configured_path: object) -> Path:
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[SketchPhotoJepa, dict[str, object], object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"JEPA checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Invalid JEPA checkpoint payload")
    required = {
        "format_version",
        "model_type",
        "step",
        "model_config",
        "model_state_dict",
        "metadata",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"JEPA checkpoint is missing keys: {sorted(missing)}")
    if payload["format_version"] != 1:
        raise ValueError("Unsupported JEPA checkpoint format")
    if payload["model_type"] != "cross_modal_jepa":
        raise ValueError(f"Unsupported JEPA model type: {payload['model_type']!r}")
    config = payload["model_config"]
    metadata = payload["metadata"]
    state = payload["model_state_dict"]
    if (
        not isinstance(config, dict)
        or not isinstance(metadata, dict)
        or not isinstance(state, dict)
    ):
        raise TypeError(
            "JEPA checkpoint config, metadata, and state must be dictionaries"
        )
    if (
        metadata.get("text_enters_predictor") is not False
        or metadata.get("text_conditioning") is not False
    ):
        raise ValueError("JEPA checkpoint violates the text-free predictor contract")
    try:
        mode = str(config["encoder_mode"])
        depth = int(config["encoder_unfreeze_depth"])
        model_name = str(config["encoder_model_name"])
        pretrained_value = config["encoder_pretrained"]
        pretrained = None if pretrained_value is None else str(pretrained_value)
        embedding_dim = int(config["embedding_dim"])
        hidden_dim = int(config["hidden_dim"])
    except KeyError as error:
        raise ValueError(f"JEPA model_config is missing {error.args[0]!r}") from error

    sketch_bundle = load_trainable_sketch_encoder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        mode=mode,
        unfreeze_depth=depth,
    )
    if sketch_bundle.encoder.embedding_dim != embedding_dim:
        raise ValueError(
            "Checkpoint embedding dimension does not match CLIP visual output"
        )
    predictor = SpicaJepaPredictor(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model = SketchPhotoJepa(sketch_bundle.encoder, predictor)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "JEPA checkpoint state does not match its model configuration"
        ) from error
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"JEPA checkpoint contains a non-finite parameter: {name}")
    model.to(device).eval()
    return model, payload, sketch_bundle.transform


def _metrics_dict(evaluation) -> dict[str, object]:
    return {
        "mAP": evaluation.metrics.mean_average_precision,
        "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
        "precision_at_k": evaluation.metrics.precision_at_k,
        "num_queries": evaluation.metrics.num_queries,
        "num_gallery_items": evaluation.metrics.num_gallery_items,
    }


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="evaluate_jepa",
)
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
    _validate_provenance(
        checkpoint=checkpoint,
        sketches=sketches_cache,
        photos=gallery,
        dataset_name=str(args.dataset_name),
        model_name=str(args.model_name),
        pretrained=None if args.pretrained is None else str(args.pretrained),
    )
    test_entries = read_manifest(data.test.sketch_manifest, data.root)
    # Recheck the query order independently of the cached CLIP sketch set.  The
    # cache is used only for protocol identity; inference below consumes images.
    _validate_cache_against_manifest(
        modality="sketch",
        encoded_set=sketches_cache,
        manifest_entries=test_entries,
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
    features = encode_jepa_loader(model, query_loader, device=device)
    evaluation = evaluate_jepa_features(
        features,
        gallery,
        precision_at_k=tuple(int(k) for k in args.precision_at_k),
        map_at_k=tuple(int(k) for k in args.map_at_k),
        map_at_k_denominator=str(args.map_at_k_denominator),
        query_chunk_size=int(args.query_chunk_size),
        device=device,
    )
    counts = checkpoint["model_config"].get("parameter_counts", {})
    result = {
        "protocol": "standard_zs_sbir_sketch_only_inference",
        "diagnostic_test_evaluation": True,
        "model_family": "cross_modal_jepa",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "training_metadata": checkpoint["metadata"],
        "split_identities": split_identities,
        "metrics": _metrics_dict(evaluation),
        "feature_geometry": feature_probe_dict(features, gallery),
        "parameter_counts": counts,
        "leakage_flags": {
            "text_used_at_inference": False,
            "text_enters_predictor": False,
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
                "trainable CLIP-initialized sketch context encoder",
                "JEPA predictor",
                "L2-normalized predicted photo-semantic query",
            ],
        },
        "map_at_k_denominator": str(args.map_at_k_denominator),
    }
    output = Path(HydraConfig.get().runtime.output_dir) / "metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"mAP: {evaluation.metrics.mean_average_precision:.6f}")
    print(f"P@200: {evaluation.metrics.precision_at_k[200]:.6f}")
    print("Text enters predictor: NO")
    print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
