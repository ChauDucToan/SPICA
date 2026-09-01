import json
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .evaluate_deterministic import (
    _metrics_dict,
    _resolve_device,
    _resolve_project_path,
    _validate_provenance,
    _validate_zero_shot_split,
)
from .evaluation.embeddings import EncodedRetrievalSet, load_encoded_retrieval_set
from .evaluation.metrics import CategoryRetrievalMetrics, evaluate_category_retrieval
from .models.retrieval import K1VmfPhotoPredictor
from .models.vmf import LOG_NORMALIZER_VERSION, log_vmf_normalizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")


def _load_predictor(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[K1VmfPhotoPredictor, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"K=1 vMF checkpoint not found: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid K=1 vMF checkpoint: {checkpoint_path}")

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
    if payload["model_type"] != "k1_vmf_photo_predictor":
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

    expected_metadata = {
        "frozen_clip": True,
        "num_components": 1,
        "objective": "positive_vmf_nll_plus_cosine_ranking",
        "log_normalizer": LOG_NORMALIZER_VERSION,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if key not in metadata or metadata[key] != expected
    }
    if mismatches:
        raise ValueError(
            f"K=1 vMF checkpoint metadata mismatch (observed, expected): {mismatches}"
        )

    try:
        predictor = K1VmfPhotoPredictor(
            embedding_dim=int(model_config["embedding_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            min_concentration=float(model_config["min_concentration"]),
            max_concentration=float(model_config["max_concentration"]),
            initial_concentration=float(model_config["initial_concentration"]),
        )
    except KeyError as error:
        raise ValueError(
            f"Checkpoint model_config is missing {error.args[0]!r}"
        ) from error

    predictor.load_state_dict(model_state_dict, strict=True)
    for name, parameter in predictor.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"Checkpoint contains a non-finite parameter: {name}")
    predictor.to(device).eval()
    return predictor, payload


def _predict_vmf_parameters(
    predictor: K1VmfPhotoPredictor,
    sketches: EncodedRetrievalSet,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[EncodedRetrievalSet, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"prediction_batch_size must be positive, got {batch_size}")
    if sketches.embeddings.shape[1] != predictor.embedding_dim:
        raise ValueError(
            "Sketch cache and predictor embedding dimensions must match, got "
            f"{sketches.embeddings.shape[1]} and {predictor.embedding_dim}"
        )
    if not torch.isfinite(sketches.embeddings).all().item():
        raise ValueError("Sketch cache contains non-finite embeddings")

    direction_batches: list[torch.Tensor] = []
    concentration_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for embeddings in sketches.embeddings.split(batch_size, dim=0):
            prediction = predictor(embeddings.to(device))
            direction_batches.append(prediction.mean_direction.float().cpu())
            concentration_batches.append(prediction.concentration.float().cpu())

    directions = torch.cat(direction_batches, dim=0)
    concentrations = torch.cat(concentration_batches, dim=0)
    if not torch.isfinite(directions).all().item():
        raise FloatingPointError("Predictor returned non-finite mean directions")
    if not torch.isfinite(concentrations).all().item():
        raise FloatingPointError("Predictor returned non-finite concentrations")
    if (
        torch.any(concentrations <= predictor.min_concentration).item()
        or torch.any(concentrations >= predictor.max_concentration).item()
    ):
        raise RuntimeError("Predicted concentration reached a configured bound")

    return (
        EncodedRetrievalSet(
            embeddings=directions,
            labels=sketches.labels,
            paths=sketches.paths,
            metadata={**sketches.metadata, "representation": "k1_vmf_mean_direction"},
        ),
        concentrations,
    )


def _concentration_statistics(
    concentrations: torch.Tensor,
    *,
    min_concentration: float,
    max_concentration: float,
) -> dict[str, float | int]:
    quantile_levels = torch.tensor(
        [0.05, 0.25, 0.5, 0.75, 0.95],
        dtype=concentrations.dtype,
    )
    quantiles = torch.quantile(concentrations, quantile_levels)
    bound_tolerance = 0.01 * (max_concentration - min_concentration)

    return {
        "count": concentrations.numel(),
        "mean": concentrations.double().mean().item(),
        "std": concentrations.double().std(unbiased=False).item(),
        "min": concentrations.min().item(),
        "p05": quantiles[0].item(),
        "p25": quantiles[1].item(),
        "p50": quantiles[2].item(),
        "p75": quantiles[3].item(),
        "p95": quantiles[4].item(),
        "max": concentrations.max().item(),
        "fraction_near_lower_bound": (
            concentrations.le(min_concentration + bound_tolerance)
            .double()
            .mean()
            .item()
        ),
        "fraction_near_upper_bound": (
            concentrations.ge(max_concentration - bound_tolerance)
            .double()
            .mean()
            .item()
        ),
    }


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left.double() - left.double().mean()
    right_centered = right.double() - right.double().mean()
    denominator = left_centered.std(unbiased=False) * right_centered.std(unbiased=False)
    if denominator == 0:
        raise ValueError("Cannot correlate a constant tensor")
    return ((left_centered * right_centered).mean() / denominator).item()


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(values, stable=True)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(dim=0)
    starts = ends - counts
    average_group_ranks = (starts + ends - 1).double() / 2.0
    sorted_ranks = torch.repeat_interleave(average_group_ranks, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _calibration_statistics(
    predicted_sketches: EncodedRetrievalSet,
    concentrations: torch.Tensor,
    photos: EncodedRetrievalSet,
    average_precision_per_query: torch.Tensor,
) -> dict[str, float]:
    directions = F.normalize(predicted_sketches.embeddings.float(), dim=-1)
    gallery = F.normalize(photos.embeddings.float(), dim=-1)
    observed_positive_cosine = torch.empty(directions.shape[0])
    for label in torch.unique(predicted_sketches.labels):
        query_mask = predicted_sketches.labels.eq(label)
        gallery_mask = photos.labels.eq(label)
        class_mean = gallery[gallery_mask].mean(dim=0)
        observed_positive_cosine[query_mask] = (
            directions[query_mask] * class_mean
        ).sum(dim=-1)

    differentiable_kappa = concentrations.double().requires_grad_(True)
    relative_log_normalizer = log_vmf_normalizer(
        differentiable_kappa,
        dimension=directions.shape[1],
        relative_to_uniform=True,
    )
    predicted_resultant_length = (
        -torch.autograd.grad(
            relative_log_normalizer.sum(),
            differentiable_kappa,
        )[0]
        .detach()
        .float()
    )

    calibration_error = predicted_resultant_length - observed_positive_cosine
    query_ap = average_precision_per_query.float()
    kappa_ranks = _average_ranks(concentrations)
    observed_ranks = _average_ranks(observed_positive_cosine)
    ap_ranks = _average_ranks(query_ap)

    return {
        "observed_positive_cosine_mean": (
            observed_positive_cosine.double().mean().item()
        ),
        "predicted_resultant_length_mean": (
            predicted_resultant_length.double().mean().item()
        ),
        "mean_absolute_calibration_error": (
            calibration_error.abs().double().mean().item()
        ),
        "mean_signed_calibration_error": (calibration_error.double().mean().item()),
        "pearson_kappa_observed_cosine": _pearson_correlation(
            concentrations,
            observed_positive_cosine,
        ),
        "spearman_kappa_observed_cosine": _pearson_correlation(
            kappa_ranks,
            observed_ranks,
        ),
        "pearson_kappa_average_precision": _pearson_correlation(
            concentrations,
            query_ap,
        ),
        "spearman_kappa_average_precision": _pearson_correlation(
            kappa_ranks,
            ap_ranks,
        ),
    }


def _print_comparison(
    baseline: CategoryRetrievalMetrics,
    vmf: CategoryRetrievalMetrics,
    ks: tuple[int, ...],
) -> None:
    rows = [
        ("mAP", baseline.mean_average_precision, vmf.mean_average_precision),
        *((f"P@{k}", baseline.precision_at_k[k], vmf.precision_at_k[k]) for k in ks),
    ]

    print("\nmetric       sketch-only       K=1 vMF   delta")
    print("------------------------------------------------")
    for name, baseline_value, vmf_value in rows:
        print(
            f"{name:<10} "
            f"{baseline_value:>11.6f} "
            f"{vmf_value:>13.6f} "
            f"{vmf_value - baseline_value:>9.6f}"
        )


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="evaluate_vmf",
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
    print(
        "K=1 note: positive concentration and the per-query log normalizer do "
        "not change gallery order; retrieval ranks by the learned mean direction."
    )
    predictor, checkpoint = _load_predictor(checkpoint_path, device=device)
    sketches = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
    photos = load_encoded_retrieval_set(embedding_dir / "photos.pt")
    if not torch.isfinite(photos.embeddings).all().item():
        raise ValueError("Photo cache contains non-finite embeddings")
    split_identities = _validate_zero_shot_split(
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
    predicted_sketches, concentrations = _predict_vmf_parameters(
        predictor,
        sketches,
        batch_size=int(args.prediction_batch_size),
        device=device,
    )
    concentration_stats = _concentration_statistics(
        concentrations,
        min_concentration=predictor.min_concentration,
        max_concentration=predictor.max_concentration,
    )

    evaluation_options = {
        "precision_at_k": ks,
        "map_at_k": tuple(int(k) for k in args.map_at_k),
        "map_at_k_denominator": str(args.map_at_k_denominator),
        "query_chunk_size": int(args.query_chunk_size),
        "top_k": max(max(ks), *(int(k) for k in args.map_at_k)),
        "device": device,
    }
    baseline = evaluate_category_retrieval(
        sketches,
        photos,
        **evaluation_options,
    ).metrics
    vmf_evaluation = evaluate_category_retrieval(
        predicted_sketches,
        photos,
        **evaluation_options,
    )
    vmf = vmf_evaluation.metrics
    calibration_stats = _calibration_statistics(
        predicted_sketches,
        concentrations,
        photos,
        vmf_evaluation.average_precision_per_query,
    )
    _print_comparison(baseline, vmf, ks)
    print(
        "\nkappa: "
        f"mean={concentration_stats['mean']:.3f}, "
        f"std={concentration_stats['std']:.3f}, "
        f"p05={concentration_stats['p05']:.3f}, "
        f"p50={concentration_stats['p50']:.3f}, "
        f"p95={concentration_stats['p95']:.3f}, "
        f"range=[{concentration_stats['min']:.3f}, "
        f"{concentration_stats['max']:.3f}]"
    )
    print(
        "calibration: "
        f"observed_cos={calibration_stats['observed_positive_cosine_mean']:.4f}, "
        f"predicted_A(kappa)="
        f"{calibration_stats['predicted_resultant_length_mean']:.4f}, "
        f"MAE={calibration_stats['mean_absolute_calibration_error']:.4f}, "
        f"rho(kappa, observed)="
        f"{calibration_stats['spearman_kappa_observed_cosine']:.4f}, "
        f"rho(kappa, AP)="
        f"{calibration_stats['spearman_kappa_average_precision']:.4f}"
    )

    output_path = Path(HydraConfig.get().runtime.output_dir) / "metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "protocol": "standard_zs_sbir_sketch_only_inference",
                "diagnostic_test_evaluation": True,
                "model": "k1_vmf",
                "ranking_semantics": "equivalent_to_cosine_of_mean_direction",
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": int(checkpoint["step"]),
                "split_identities": split_identities,
                "map_at_k_denominator": str(args.map_at_k_denominator),
                "sketch_only": _metrics_dict(baseline),
                "k1_vmf": _metrics_dict(vmf),
                "delta_mAP": vmf.mean_average_precision
                - baseline.mean_average_precision,
                "concentration": concentration_stats,
                "calibration": calibration_stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
