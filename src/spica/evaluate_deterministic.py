import json
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.manifest import read_class_map
from .evaluation.embeddings import EncodedRetrievalSet, load_encoded_retrieval_set
from .evaluation.metrics import CategoryRetrievalMetrics, evaluate_category_retrieval
from .models.retrieval import DeterministicPhotoPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_project_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _load_predictor(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[DeterministicPhotoPredictor, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Predictor checkpoint not found: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid predictor checkpoint: {checkpoint_path}")

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
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    if payload["format_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint format version: {payload['format_version']}"
        )
    if payload["model_type"] != "deterministic_photo_predictor":
        raise ValueError(f"Unsupported model type: {payload['model_type']!r}")

    model_config = payload["model_config"]
    model_state_dict = payload["model_state_dict"]
    metadata = payload["metadata"]
    if not isinstance(model_config, dict):
        raise TypeError("Checkpoint model_config must be a dictionary")
    if not isinstance(model_state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a dictionary")
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a dictionary")

    try:
        embedding_dim = int(model_config["embedding_dim"])
        hidden_dim = int(model_config["hidden_dim"])
    except KeyError as error:
        raise ValueError(
            f"Checkpoint model_config is missing {error.args[0]!r}"
        ) from error

    predictor = DeterministicPhotoPredictor(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    predictor.load_state_dict(model_state_dict, strict=True)
    for name, parameter in predictor.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"Checkpoint contains a non-finite parameter: {name}")
    predictor.to(device).eval()
    return predictor, payload


def _validate_provenance(
    *,
    checkpoint: dict[str, object],
    sketches: EncodedRetrievalSet,
    photos: EncodedRetrievalSet,
    dataset_name: str,
    model_name: str,
    pretrained: str | None,
) -> None:
    checkpoint_metadata = checkpoint["metadata"]
    if not isinstance(checkpoint_metadata, dict):
        raise TypeError("Checkpoint metadata must be a dictionary")

    expected_checkpoint = {
        "dataset": dataset_name,
        "split": "train",
        "model_name": model_name,
        "pretrained": pretrained,
    }
    checkpoint_mismatches = {
        key: (checkpoint_metadata.get(key), expected)
        for key, expected in expected_checkpoint.items()
        if key not in checkpoint_metadata or checkpoint_metadata[key] != expected
    }
    if checkpoint_mismatches:
        raise ValueError(
            "Checkpoint provenance mismatch (observed, expected): "
            f"{checkpoint_mismatches}"
        )

    common_cache_metadata = {
        "format_version": 1,
        "dataset": dataset_name,
        "split": "test",
        "model_name": model_name,
        "pretrained": pretrained,
    }
    for modality, encoded_set in (("sketch", sketches), ("photo", photos)):
        expected_cache = {**common_cache_metadata, "modality": modality}
        mismatches = {
            key: (encoded_set.metadata.get(key), expected)
            for key, expected in expected_cache.items()
            if key not in encoded_set.metadata or encoded_set.metadata[key] != expected
        }
        if mismatches:
            raise ValueError(
                f"{modality.capitalize()} cache provenance mismatch "
                f"(observed, expected): {mismatches}"
            )


def _validate_zero_shot_split(
    data_config_path: Path,
    *,
    dataset_name: str,
    sketches: EncodedRetrievalSet,
    photos: EncodedRetrievalSet,
) -> None:
    data_config = load_data_config(data_config_path)
    if data_config.name != dataset_name:
        raise ValueError(
            f"Data config name must be {dataset_name!r}, got {data_config.name!r}"
        )

    train_classes = read_class_map(data_config.train.class_map)
    test_classes = read_class_map(data_config.test.class_map)
    semantic_overlap = set(train_classes.values()) & set(test_classes.values())
    if semantic_overlap:
        raise ValueError(
            "Train and test class names must be disjoint; overlap: "
            f"{sorted(semantic_overlap)}"
        )

    known_test_labels = set(test_classes)
    observed_labels = set(sketches.labels.tolist()) | set(photos.labels.tolist())
    unknown_labels = observed_labels - known_test_labels
    if unknown_labels:
        raise ValueError(
            "Test cache labels missing from the test class map: "
            f"{sorted(unknown_labels)}"
        )


def _predict_photo_directions(
    predictor: DeterministicPhotoPredictor,
    sketches: EncodedRetrievalSet,
    *,
    batch_size: int,
    device: torch.device,
) -> EncodedRetrievalSet:
    if batch_size <= 0:
        raise ValueError(f"prediction_batch_size must be positive, got {batch_size}")
    if sketches.embeddings.shape[1] != predictor.embedding_dim:
        raise ValueError(
            "Sketch cache and predictor embedding dimensions must match, got "
            f"{sketches.embeddings.shape[1]} and {predictor.embedding_dim}"
        )
    if not torch.isfinite(sketches.embeddings).all().item():
        raise ValueError("Sketch cache contains non-finite embeddings")

    predicted_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for embeddings in sketches.embeddings.split(batch_size, dim=0):
            predicted_batches.append(predictor(embeddings.to(device)).float().cpu())

    predicted_embeddings = torch.cat(predicted_batches, dim=0)
    if not torch.isfinite(predicted_embeddings).all().item():
        raise FloatingPointError("Predictor returned non-finite embeddings")

    return EncodedRetrievalSet(
        embeddings=predicted_embeddings,
        labels=sketches.labels,
        paths=sketches.paths,
        metadata={**sketches.metadata, "representation": "deterministic_prediction"},
    )


def _metrics_dict(metrics: CategoryRetrievalMetrics) -> dict[str, object]:
    return {
        "mAP": metrics.mean_average_precision,
        "precision_at_k": metrics.precision_at_k,
        "num_queries": metrics.num_queries,
        "num_gallery_items": metrics.num_gallery_items,
    }


def _print_comparison(
    baseline: CategoryRetrievalMetrics,
    deterministic: CategoryRetrievalMetrics,
    ks: tuple[int, ...],
) -> None:
    rows = [
        (
            "mAP",
            baseline.mean_average_precision,
            deterministic.mean_average_precision,
        ),
        *(
            (
                f"P@{k}",
                baseline.precision_at_k[k],
                deterministic.precision_at_k[k],
            )
            for k in ks
        ),
    ]

    print("\nmetric       sketch-only   deterministic   delta")
    print("------------------------------------------------")
    for name, baseline_value, deterministic_value in rows:
        print(
            f"{name:<10} "
            f"{baseline_value:>11.6f} "
            f"{deterministic_value:>15.6f} "
            f"{deterministic_value - baseline_value:>9.6f}"
        )


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="evaluate_deterministic",
)
def main(args: DictConfig) -> None:
    device = _resolve_device(str(args.device))
    checkpoint_path = _resolve_project_path(str(args.checkpoint_path))
    embedding_dir = _resolve_project_path(str(args.embedding_dir))
    data_config_path = _resolve_project_path(str(args.data_config))
    pretrained = None if args.pretrained is None else str(args.pretrained)
    ks = tuple(sorted({int(k) for k in args.precision_at_k}))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"precision_at_k must contain positive integers, got {ks}")

    print(
        "Warning: this unseen-test evaluation is diagnostic only; do not select "
        "training hyperparameters from it."
    )
    predictor, checkpoint = _load_predictor(checkpoint_path, device=device)
    sketches = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
    photos = load_encoded_retrieval_set(embedding_dir / "photos.pt")
    if not torch.isfinite(photos.embeddings).all().item():
        raise ValueError("Photo cache contains non-finite embeddings")
    _validate_zero_shot_split(
        data_config_path,
        dataset_name=str(args.dataset_name),
        sketches=sketches,
        photos=photos,
    )
    _validate_provenance(
        checkpoint=checkpoint,
        sketches=sketches,
        photos=photos,
        dataset_name=str(args.dataset_name),
        model_name=str(args.model_name),
        pretrained=pretrained,
    )

    print(
        f"Evaluating checkpoint step {checkpoint['step']} on {device}: "
        f"{checkpoint_path}"
    )
    predicted_sketches = _predict_photo_directions(
        predictor,
        sketches,
        batch_size=int(args.prediction_batch_size),
        device=device,
    )

    evaluation_options = {
        "precision_at_k": ks,
        "query_chunk_size": int(args.query_chunk_size),
        "top_k": max(ks),
        "device": device,
    }
    baseline = evaluate_category_retrieval(
        sketches,
        photos,
        **evaluation_options,
    ).metrics
    deterministic = evaluate_category_retrieval(
        predicted_sketches,
        photos,
        **evaluation_options,
    ).metrics
    _print_comparison(baseline, deterministic, ks)

    output_path = Path(HydraConfig.get().runtime.output_dir) / "metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "protocol": "standard_zs_sbir_sketch_only_inference",
                "diagnostic_test_evaluation": True,
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": int(checkpoint["step"]),
                "sketch_only": _metrics_dict(baseline),
                "deterministic": _metrics_dict(deterministic),
                "delta_mAP": (
                    deterministic.mean_average_precision
                    - baseline.mean_average_precision
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
