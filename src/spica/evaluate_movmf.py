import json
import math
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
from .evaluate_vmf import (
    _average_ranks,
    _concentration_statistics,
    _pearson_correlation,
)
from .evaluation.embeddings import EncodedRetrievalSet, load_encoded_retrieval_set
from .evaluation.metrics import (
    CategoryRetrievalEvaluation,
    CategoryRetrievalMetrics,
    _average_precision_from_relevance,
    evaluate_category_retrieval,
)
from .models.retrieval import MoVmfPhotoPredictor, MoVmfPrediction
from .models.vmf import (
    LOG_NORMALIZER_VERSION,
    SCORE_NORMALIZATION_VERSION,
    log_vmf_normalizer,
    mo_vmf_gallery_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")
OBJECTIVE_NAME = "positive_mixture_vmf_nll_plus_normalized_density_ranking"
SUPPORTED_OBJECTIVES = {
    OBJECTIVE_NAME,
    "multi_positive_mixture_vmf_anticollapse",
}


def _load_predictor(
    checkpoint_path: Path,
    *,
    expected_num_components: int,
    device: torch.device,
) -> tuple[MoVmfPhotoPredictor, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Mo-vMF checkpoint not found: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid Mo-vMF checkpoint: {checkpoint_path}")

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
    if payload["model_type"] != "mo_vmf_photo_predictor":
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

    objective = metadata.get("objective")
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported Mo-vMF objective: {objective!r}")
    positives_per_anchor = metadata.get("positives_per_anchor_per_step")
    if not isinstance(positives_per_anchor, int) or positives_per_anchor <= 0:
        raise ValueError(
            "Checkpoint positives_per_anchor_per_step must be a positive integer"
        )

    expected_metadata = {
        "frozen_clip": True,
        "num_components": expected_num_components,
        "log_normalizer": LOG_NORMALIZER_VERSION,
        "score_normalization": SCORE_NORMALIZATION_VERSION,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if key not in metadata or metadata[key] != expected
    }
    if mismatches:
        raise ValueError(
            f"Mo-vMF checkpoint metadata mismatch (observed, expected): {mismatches}"
        )

    try:
        predictor = MoVmfPhotoPredictor(
            embedding_dim=int(model_config["embedding_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            num_components=int(model_config["num_components"]),
            min_concentration=float(model_config["min_concentration"]),
            max_concentration=float(model_config["max_concentration"]),
            initial_concentration=float(model_config["initial_concentration"]),
            component_init_std=float(model_config["component_init_std"]),
            initial_dominant_weight=(
                None
                if model_config.get("initial_dominant_weight") is None
                else float(model_config["initial_dominant_weight"])
            ),
            concentration_mode=str(model_config.get("concentration_mode", "learned")),
            fixed_concentration=(
                None
                if model_config.get("fixed_concentration") is None
                else float(model_config["fixed_concentration"])
            ),
        )
    except KeyError as error:
        raise ValueError(
            f"Checkpoint model_config is missing {error.args[0]!r}"
        ) from error
    if predictor.num_components != expected_num_components:
        raise ValueError(
            "Checkpoint model_config num_components does not match evaluation: "
            f"{predictor.num_components} != {expected_num_components}"
        )

    predictor.load_state_dict(model_state_dict, strict=True)
    for name, parameter in predictor.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"Checkpoint contains a non-finite parameter: {name}")
    predictor.to(device).eval()
    return predictor, payload


def _predict_parameters(
    predictor: MoVmfPhotoPredictor,
    sketches: EncodedRetrievalSet,
    *,
    batch_size: int,
    device: torch.device,
) -> MoVmfPrediction:
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
    logit_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for embeddings in sketches.embeddings.split(batch_size, dim=0):
            prediction = predictor(embeddings.to(device))
            direction_batches.append(prediction.mean_directions.float().cpu())
            concentration_batches.append(prediction.concentrations.float().cpu())
            logit_batches.append(prediction.mixture_logits.float().cpu())

    prediction = MoVmfPrediction(
        mean_directions=torch.cat(direction_batches, dim=0),
        concentrations=torch.cat(concentration_batches, dim=0),
        mixture_logits=torch.cat(logit_batches, dim=0),
    )
    if not torch.isfinite(prediction.mean_directions).all().item():
        raise FloatingPointError("Predictor returned non-finite mean directions")
    if not torch.isfinite(prediction.concentrations).all().item():
        raise FloatingPointError("Predictor returned non-finite concentrations")
    if not torch.isfinite(prediction.mixture_logits).all().item():
        raise FloatingPointError("Predictor returned non-finite mixture logits")
    if (
        torch.any(prediction.concentrations < predictor.min_concentration).item()
        or torch.any(prediction.concentrations > predictor.max_concentration).item()
    ):
        raise RuntimeError("Predicted concentration escaped a configured bound")

    norms = prediction.mean_directions.norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Predicted Mo-vMF directions are not unit normalized")
    return prediction


def _slice_prediction(
    prediction: MoVmfPrediction,
    start: int,
    stop: int,
    device: torch.device,
) -> MoVmfPrediction:
    return MoVmfPrediction(
        mean_directions=prediction.mean_directions[start:stop].to(device),
        concentrations=prediction.concentrations[start:stop].to(device),
        mixture_logits=prediction.mixture_logits[start:stop].to(device),
    )


@torch.inference_mode()
def _evaluate_movmf_retrieval(
    prediction: MoVmfPrediction,
    query_labels: torch.Tensor,
    photos: EncodedRetrievalSet,
    *,
    precision_at_k: tuple[int, ...],
    query_chunk_size: int,
    top_k: int,
    device: torch.device,
    map_at_k: tuple[int, ...] = (),
    map_at_k_denominator: str = "prefix_positive",
) -> CategoryRetrievalEvaluation:
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    ks = tuple(sorted(set(precision_at_k)))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"precision_at_k must contain positive values, got {ks}")

    num_queries = prediction.mean_directions.shape[0]
    num_gallery_items = photos.embeddings.shape[0]
    if query_labels.shape != (num_queries,):
        raise ValueError(
            f"query_labels must have shape ({num_queries},), got {query_labels.shape}"
        )
    if num_queries == 0 or num_gallery_items == 0:
        raise ValueError("Queries and gallery must both be non-empty")
    if max(ks) > num_gallery_items:
        raise ValueError(
            f"P@{max(ks)} requires {max(ks)} gallery items, got {num_gallery_items}"
        )
    map_ks = tuple(sorted(set(map_at_k)))
    if any(k <= 0 or k > num_gallery_items for k in map_ks):
        raise ValueError(f"map_at_k values must be in [1, {num_gallery_items}]")
    if not 0 < top_k <= num_gallery_items:
        raise ValueError(
            f"top_k must be between 1 and {num_gallery_items}, got {top_k}"
        )

    gallery_embeddings = F.normalize(photos.embeddings.to(device), dim=-1)
    gallery_labels = photos.labels.to(device)
    rank_positions = torch.arange(
        1,
        num_gallery_items + 1,
        device=device,
        dtype=torch.float32,
    )
    average_precision_batches: list[torch.Tensor] = []
    precision_batches: dict[int, list[torch.Tensor]] = {k: [] for k in ks}
    map_batches: dict[int, list[torch.Tensor]] = {k: [] for k in map_ks}
    top_index_batches: list[torch.Tensor] = []
    top_score_batches: list[torch.Tensor] = []

    for start in range(0, num_queries, query_chunk_size):
        stop = min(start + query_chunk_size, num_queries)
        chunk = _slice_prediction(prediction, start, stop, device)
        scores = mo_vmf_gallery_scores(
            chunk,
            gallery_embeddings,
            normalized=True,
        )
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("Mo-vMF gallery scores are not finite")
        ranked_indices = torch.argsort(
            scores,
            dim=1,
            descending=True,
            stable=True,
        )
        ranked_labels = gallery_labels[ranked_indices]
        labels = query_labels[start:stop].to(device)
        relevant = ranked_labels.eq(labels[:, None])
        num_positives = relevant.sum(dim=1)
        if torch.any(num_positives == 0).item():
            local_indices = torch.nonzero(num_positives == 0).flatten()
            query_indices = (local_indices + start).tolist()
            raise ValueError(
                "Every query must have at least one positive gallery item; "
                f"queries without positives: {query_indices}"
            )

        average_precision, truncated = _average_precision_from_relevance(
            relevant,
            rank_positions,
            map_ks,
            map_at_k_denominator=map_at_k_denominator,
        )
        average_precision_batches.append(average_precision.cpu())
        for k, values in truncated.items():
            map_batches[k].append(values.cpu())
        relevant_float = relevant.float()
        for k in ks:
            precision_batches[k].append(relevant_float[:, :k].mean(dim=1).cpu())

        chunk_top_indices = ranked_indices[:, :top_k]
        top_index_batches.append(chunk_top_indices.cpu())
        top_score_batches.append(
            scores.gather(dim=1, index=chunk_top_indices).float().cpu()
        )

    average_precision_per_query = torch.cat(average_precision_batches)
    return CategoryRetrievalEvaluation(
        metrics=CategoryRetrievalMetrics(
            mean_average_precision=(average_precision_per_query.double().mean().item()),
            precision_at_k={
                k: torch.cat(batches).double().mean().item()
                for k, batches in precision_batches.items()
            },
            num_queries=num_queries,
            num_gallery_items=num_gallery_items,
            mean_average_precision_at_k={
                k: torch.cat(values).double().mean().item()
                for k, values in map_batches.items()
            },
        ),
        average_precision_per_query=average_precision_per_query,
        top_indices=torch.cat(top_index_batches),
        top_scores=torch.cat(top_score_batches),
    )


def _component_statistics(
    prediction: MoVmfPrediction,
    average_precision_per_query: torch.Tensor,
    *,
    min_concentration: float,
    max_concentration: float,
) -> dict[str, object]:
    probabilities = prediction.mixture_logits.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    effective_components = entropy.exp()
    weighted_concentration = (probabilities * prediction.concentrations).sum(dim=-1)
    num_components = probabilities.shape[1]

    if num_components > 1:
        upper_triangle = torch.triu(
            torch.ones(num_components, num_components, dtype=torch.bool),
            diagonal=1,
        )
        pairwise_cosines = torch.einsum(
            "bkd,bjd->bkj",
            prediction.mean_directions,
            prediction.mean_directions,
        )[:, upper_triangle]
        mean_pairwise_cosine = pairwise_cosines.double().mean().item()
        min_pairwise_cosine = pairwise_cosines.min().item()
        p05_pairwise_cosine = torch.quantile(pairwise_cosines, 0.05).item()
        normalized_mean_entropy = entropy.double().mean().item() / math.log(
            num_components
        )
        entropy_ap_correlation: float | None = _pearson_correlation(
            _average_ranks(entropy),
            _average_ranks(average_precision_per_query.float()),
        )
    else:
        mean_pairwise_cosine = 1.0
        min_pairwise_cosine = 1.0
        p05_pairwise_cosine = 1.0
        normalized_mean_entropy = 0.0
        entropy_ap_correlation = None
    hard_assignments = probabilities.argmax(dim=-1)
    hard_fractions = (
        torch.bincount(
            hard_assignments,
            minlength=num_components,
        ).double()
        / hard_assignments.numel()
    )
    query_ap = average_precision_per_query.float()

    weighted_kappa_ranks = _average_ranks(weighted_concentration)
    ap_ranks = _average_ranks(query_ap)
    return {
        "concentration": _concentration_statistics(
            prediction.concentrations.flatten(),
            min_concentration=min_concentration,
            max_concentration=max_concentration,
        ),
        "mean_component_weights": probabilities.double().mean(dim=0).tolist(),
        "mean_component_concentrations": (
            prediction.concentrations.double().mean(dim=0).tolist()
        ),
        "hard_prior_assignment_fractions": hard_fractions.tolist(),
        "mean_prior_entropy": entropy.double().mean().item(),
        "normalized_mean_prior_entropy": normalized_mean_entropy,
        "mean_effective_components": effective_components.double().mean().item(),
        "mean_max_component_weight": (
            probabilities.max(dim=-1).values.double().mean().item()
        ),
        "mean_pairwise_direction_cosine": mean_pairwise_cosine,
        "min_pairwise_direction_cosine": min_pairwise_cosine,
        "p05_pairwise_direction_cosine": p05_pairwise_cosine,
        "spearman_weighted_kappa_average_precision": _pearson_correlation(
            weighted_kappa_ranks,
            ap_ranks,
        ),
        "spearman_prior_entropy_average_precision": entropy_ap_correlation,
    }


@torch.inference_mode()
def _positive_responsibility_statistics(
    prediction: MoVmfPrediction,
    query_labels: torch.Tensor,
    photos: EncodedRetrievalSet,
    *,
    query_chunk_size: int,
    device: torch.device,
) -> dict[str, object]:
    if query_chunk_size <= 0:
        raise ValueError(
            f"responsibility_query_chunk_size must be positive, got {query_chunk_size}"
        )
    num_components = prediction.mean_directions.shape[1]
    responsibility_sums = torch.zeros(num_components, dtype=torch.float64)
    hard_counts = torch.zeros(num_components, dtype=torch.float64)
    entropy_sum = 0.0
    effective_components_sum = 0.0
    max_responsibility_sum = 0.0
    query_usage_effective_components_sum = 0.0
    query_max_component_usage_sum = 0.0
    pair_count = 0
    query_count = 0

    unique_labels = torch.unique(query_labels)
    for label in unique_labels:
        query_indices = torch.nonzero(query_labels.eq(label)).flatten()
        class_gallery = F.normalize(
            photos.embeddings[photos.labels.eq(label)].to(device),
            dim=-1,
        )
        for offset in range(0, query_indices.numel(), query_chunk_size):
            indices = query_indices[offset : offset + query_chunk_size]
            chunk = MoVmfPrediction(
                mean_directions=prediction.mean_directions[indices].to(device),
                concentrations=prediction.concentrations[indices].to(device),
                mixture_logits=prediction.mixture_logits[indices].to(device),
            )
            log_weights = F.log_softmax(chunk.mixture_logits, dim=-1)
            log_normalizers = log_vmf_normalizer(
                chunk.concentrations,
                dimension=chunk.mean_directions.shape[2],
                relative_to_uniform=True,
            )
            cosines = torch.einsum(
                "bkd,gd->bkg",
                chunk.mean_directions,
                class_gallery,
            )
            component_terms = (
                log_weights[:, :, None]
                + log_normalizers[:, :, None]
                + chunk.concentrations[:, :, None] * cosines
            )
            responsibilities = component_terms.softmax(dim=1)
            entropy = -(responsibilities * responsibilities.clamp_min(1e-12).log()).sum(
                dim=1
            )
            hard = responsibilities.argmax(dim=1)
            mean_query_responsibilities = responsibilities.mean(dim=2)
            query_usage_entropy = -(
                mean_query_responsibilities
                * mean_query_responsibilities.clamp_min(1e-12).log()
            ).sum(dim=1)

            responsibility_sums += responsibilities.sum(dim=(0, 2)).double().cpu()
            hard_counts += (
                torch.bincount(
                    hard.flatten(),
                    minlength=num_components,
                )
                .double()
                .cpu()
            )
            entropy_sum += entropy.double().sum().item()
            effective_components_sum += entropy.exp().double().sum().item()
            max_responsibility_sum += (
                responsibilities.max(dim=1).values.double().sum().item()
            )
            query_usage_effective_components_sum += (
                query_usage_entropy.exp().double().sum().item()
            )
            query_max_component_usage_sum += (
                mean_query_responsibilities.max(dim=1).values.double().sum().item()
            )
            pair_count += entropy.numel()
            query_count += mean_query_responsibilities.shape[0]

    if pair_count == 0 or query_count == 0:
        raise ValueError("No positive query-gallery pairs were available")
    return {
        "uses_ground_truth_test_labels": True,
        "num_positive_pairs": pair_count,
        "mean_responsibilities": (responsibility_sums / pair_count).tolist(),
        "hard_assignment_fractions": (hard_counts / pair_count).tolist(),
        "mean_entropy": entropy_sum / pair_count,
        "normalized_mean_entropy": (
            entropy_sum / pair_count / math.log(num_components)
            if num_components > 1
            else 0.0
        ),
        "mean_effective_components": effective_components_sum / pair_count,
        "mean_max_responsibility": max_responsibility_sum / pair_count,
        "mean_query_usage_effective_components": (
            query_usage_effective_components_sum / query_count
        ),
        "mean_query_max_component_usage": (query_max_component_usage_sum / query_count),
    }


def _print_comparison(
    baseline: CategoryRetrievalMetrics,
    movmf: CategoryRetrievalMetrics,
    ks: tuple[int, ...],
    *,
    num_components: int,
) -> None:
    rows = [
        ("mAP", baseline.mean_average_precision, movmf.mean_average_precision),
        *((f"P@{k}", baseline.precision_at_k[k], movmf.precision_at_k[k]) for k in ks),
    ]
    print(f"\nmetric       sketch-only      K={num_components} Mo-vMF   delta")
    print("---------------------------------------------------")
    for name, baseline_value, movmf_value in rows:
        print(
            f"{name:<10} "
            f"{baseline_value:>11.6f} "
            f"{movmf_value:>15.6f} "
            f"{movmf_value - baseline_value:>9.6f}"
        )


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="evaluate_movmf",
)
def main(args: DictConfig) -> None:
    device = _resolve_device(str(args.device))
    checkpoint_path = _resolve_project_path(str(args.checkpoint_path))
    embedding_dir = _resolve_project_path(str(args.embedding_dir))
    data_config_path = _resolve_project_path(str(args.data_config))
    pretrained = None if args.pretrained is None else str(args.pretrained)
    num_components = int(args.num_components)
    scoring_rule = str(args.scoring_rule)
    ks = tuple(sorted({int(k) for k in args.precision_at_k}))
    map_ks = tuple(sorted({int(k) for k in args.map_at_k}))
    if num_components < 1:
        raise ValueError(f"num_components must be at least 1, got {num_components}")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"precision_at_k must contain positive integers, got {ks}")
    if scoring_rule not in {"density", "semantic_barycenter", "dominant"}:
        raise ValueError(
            "scoring_rule must be density, semantic_barycenter, or dominant"
        )
    if not map_ks or any(k <= 0 for k in map_ks):
        raise ValueError(f"map_at_k must contain positive integers, got {map_ks}")

    print(
        "Warning: this unseen-test evaluation is diagnostic only; do not select "
        "training hyperparameters from it."
    )
    print(
        "Mo-vMF scoring uses only query sketches, gallery photos, and learned "
        "parameters. Test labels are used only after ranking for metrics and "
        "component diagnostics."
    )
    predictor, checkpoint = _load_predictor(
        checkpoint_path,
        expected_num_components=num_components,
        device=device,
    )
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
        f"Evaluating K={num_components} checkpoint step {checkpoint['step']} "
        f"on {device}: {checkpoint_path}"
    )
    prediction = _predict_parameters(
        predictor,
        sketches,
        batch_size=int(args.prediction_batch_size),
        device=device,
    )
    baseline_evaluation = evaluate_category_retrieval(
        sketches,
        photos,
        precision_at_k=ks,
        map_at_k=map_ks,
        query_chunk_size=int(args.baseline_query_chunk_size),
        top_k=max(max(ks), *map_ks),
        device=device,
    )
    if scoring_rule == "density":
        movmf_evaluation = _evaluate_movmf_retrieval(
            prediction,
            sketches.labels,
            photos,
            precision_at_k=ks,
            map_at_k=map_ks,
            map_at_k_denominator=str(args.map_at_k_denominator),
            query_chunk_size=int(args.query_chunk_size),
            top_k=max(max(ks), *map_ks),
            device=device,
        )
        ranking_semantics = "normalized_score_preserves_mixture_density_order"
    else:
        if scoring_rule == "semantic_barycenter":
            probabilities = prediction.mixture_logits.softmax(dim=-1)
            query_embeddings = F.normalize(
                (probabilities[:, :, None] * prediction.mean_directions).sum(dim=1),
                dim=-1,
            )
            ranking_semantics = "cosine_of_gate_weighted_semantic_barycenter"
        else:
            query_embeddings = prediction.mean_directions[:, 0]
            ranking_semantics = "cosine_of_fixed_dominant_component"
        scoring_queries = EncodedRetrievalSet(
            embeddings=query_embeddings,
            labels=sketches.labels,
            paths=sketches.paths,
            metadata={**sketches.metadata, "representation": scoring_rule},
        )
        movmf_evaluation = evaluate_category_retrieval(
            scoring_queries,
            photos,
            precision_at_k=ks,
            map_at_k=map_ks,
            map_at_k_denominator=str(args.map_at_k_denominator),
            query_chunk_size=int(args.baseline_query_chunk_size),
            top_k=max(max(ks), *map_ks),
            device=device,
        )
    component_stats = _component_statistics(
        prediction,
        movmf_evaluation.average_precision_per_query,
        min_concentration=predictor.min_concentration,
        max_concentration=predictor.max_concentration,
    )
    responsibility_stats = _positive_responsibility_statistics(
        prediction,
        sketches.labels,
        photos,
        query_chunk_size=int(args.responsibility_query_chunk_size),
        device=device,
    )

    baseline = baseline_evaluation.metrics
    movmf = movmf_evaluation.metrics
    print(f"Scoring rule: {scoring_rule} ({ranking_semantics})")
    _print_comparison(baseline, movmf, ks, num_components=num_components)
    concentration = component_stats["concentration"]
    assert isinstance(concentration, dict)
    print(
        "\nmixture: "
        f"effective_K={component_stats['mean_effective_components']:.3f}, "
        f"max_weight={component_stats['mean_max_component_weight']:.3f}, "
        f"mean_mu_cos={component_stats['mean_pairwise_direction_cosine']:.4f}"
    )
    print(
        "positive responsibilities (label-based diagnostic): "
        f"pair_effective_K={responsibility_stats['mean_effective_components']:.3f}, "
        f"query_usage_K="
        f"{responsibility_stats['mean_query_usage_effective_components']:.3f}, "
        f"max={responsibility_stats['mean_max_responsibility']:.3f}"
    )
    print(
        "kappa: "
        f"mean={concentration['mean']:.3f}, std={concentration['std']:.3f}, "
        f"range=[{concentration['min']:.3f}, {concentration['max']:.3f}]"
    )

    output_path = Path(HydraConfig.get().runtime.output_dir) / "metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "protocol": "standard_zs_sbir_sketch_only_inference",
                "diagnostic_test_evaluation": True,
                "uses_test_labels_for_scoring": False,
                "model": "mo_vmf",
                "num_components": num_components,
                "scoring_rule": scoring_rule,
                "ranking_semantics": ranking_semantics,
                "map_at_k_denominator": str(args.map_at_k_denominator),
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": int(checkpoint["step"]),
                "training_metadata": checkpoint["metadata"],
                "sketch_only": _metrics_dict(baseline),
                "mo_vmf": _metrics_dict(movmf),
                "delta_mAP": movmf.mean_average_precision
                - baseline.mean_average_precision,
                "component_statistics": component_stats,
                "positive_responsibility_diagnostic": responsibility_stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
