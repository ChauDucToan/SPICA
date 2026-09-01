#!/usr/bin/env python3
"""Evaluate one Stage-E checkpoint with all retrieval representations.

This is an inference-only diagnostic.  Test labels are used only to compute
category retrieval metrics and positive-responsibility/component-role
statistics after rankings have been produced.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from spica.evaluate_deterministic import (
    _load_predictor as _load_k1_predictor,
    _predict_photo_directions,
    _resolve_device,
    _validate_provenance,
    _validate_zero_shot_split,
)
from spica.evaluate_deterministic_k3 import (
    _load_predictor as _load_deterministic_predictor,
    _predict as _predict_deterministic,
)
from spica.evaluate_movmf import _load_predictor as _load_movmf_predictor
from spica.evaluate_movmf import _predict_parameters as _predict_movmf_parameters
from spica.evaluate_vmf import _average_ranks, _pearson_correlation
from spica.evaluation.embeddings import load_encoded_retrieval_set
from spica.models.retrieval import MoVmfPrediction
from spica.models.vmf import log_vmf_normalizer, mo_vmf_gallery_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() == 0 or right.numel() != left.numel():
        return None
    if left.double().std(unbiased=False) == 0 or right.double().std(unbiased=False) == 0:
        return None
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float64).mean().item())


def _metric_accumulator() -> dict[str, Any]:
    return {
        "ap": [],
        "map_at_200": {
            "prefix_positive": [],
            "all_relevant": [],
            "min_relevant_k": [],
        },
        "precision": {1: [], 5: [], 10: [], 100: [], 200: []},
        "top1_correct": [],
    }


def _finish_metrics(
    accumulator: dict[str, Any],
    *,
    num_queries: int,
    num_gallery_items: int,
) -> dict[str, Any]:
    ap = torch.cat(accumulator["ap"]).double()
    map_at_200 = {
        name: torch.cat(values).double().mean().item()
        for name, values in accumulator["map_at_200"].items()
    }
    precision = {
        str(k): torch.cat(values).double().mean().item()
        for k, values in accumulator["precision"].items()
    }
    return {
        "mAP": ap.mean().item(),
        "mAP@200_prefix_positive": map_at_200["prefix_positive"],
        "mAP@200_all_relevant": map_at_200["all_relevant"],
        "mAP@200_min_relevant_k": map_at_200["min_relevant_k"],
        "P@1": precision["1"],
        "P@5": precision["5"],
        "P@10": precision["10"],
        "P@100": precision["100"],
        "P@200": precision["200"],
        "top1_accuracy": torch.cat(accumulator["top1_correct"])
        .double()
        .mean()
        .item(),
        "num_queries": num_queries,
        "num_gallery_items": num_gallery_items,
    }


def _component_pair_cosines(directions: torch.Tensor) -> torch.Tensor:
    pairs = []
    for first in range(directions.shape[1]):
        for second in range(first + 1, directions.shape[1]):
            pairs.append(
                (directions[:, first] * directions[:, second]).sum(dim=-1)
            )
    return torch.stack(pairs, dim=-1) if pairs else directions.new_ones((directions.shape[0], 1))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--embedding-dir",
        default="outputs/sketchy_104_21/clip_openai_quickgelu",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/sketchy_104_21.yaml",
    )
    parser.add_argument("--dataset-name", default="sketchy_104_21")
    parser.add_argument("--model-name", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument(
        "--assignment-temperature",
        type=float,
        default=None,
        help="Override the checkpoint training temperature for post-hoc angular diagnostics",
    )
    parser.add_argument("--angular-logsumexp-temperature", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    embedding_dir = Path(args.embedding_dir).expanduser()
    if not embedding_dir.is_absolute() and not embedding_dir.exists():
        embedding_dir = PROJECT_ROOT / embedding_dir
    data_config = Path(args.data_config).expanduser()
    if not data_config.is_absolute() and not data_config.exists():
        data_config = PROJECT_ROOT / data_config

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    model_type = payload.get("model_type")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config must be a dictionary")
    max_concentration: float | None = None
    if model_type == "mo_vmf_photo_predictor":
        num_components = int(model_config["num_components"])
        max_concentration = float(model_config["max_concentration"])
        predictor, checkpoint = _load_movmf_predictor(
            checkpoint_path,
            expected_num_components=num_components,
            device=device,
        )
        sketches = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
        photos = load_encoded_retrieval_set(embedding_dir / "photos.pt")
        prediction = _predict_movmf_parameters(
            predictor,
            sketches,
            batch_size=args.prediction_batch_size,
            device=device,
        )
        is_movmf = True
    elif model_type == "deterministic_k3_photo_predictor":
        predictor, checkpoint = _load_deterministic_predictor(checkpoint_path, device)
        num_components = 3
        sketches = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
        photos = load_encoded_retrieval_set(embedding_dir / "photos.pt")
        directions, mixture_logits = _predict_deterministic(
            predictor,
            sketches,
            device,
            args.prediction_batch_size,
        )
        prediction = MoVmfPrediction(
            mean_directions=directions,
            concentrations=torch.ones(directions.shape[:2]),
            mixture_logits=mixture_logits,
        )
        is_movmf = False
    elif model_type == "deterministic_photo_predictor":
        predictor, checkpoint = _load_k1_predictor(
            checkpoint_path,
            device=device,
        )
        sketches = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
        photos = load_encoded_retrieval_set(embedding_dir / "photos.pt")
        predicted = _predict_photo_directions(
            predictor,
            sketches,
            batch_size=args.prediction_batch_size,
            device=device,
        )
        num_components = 1
        prediction = MoVmfPrediction(
            mean_directions=predicted.embeddings[:, None, :],
            concentrations=torch.ones((predicted.embeddings.shape[0], 1)),
            mixture_logits=torch.zeros((predicted.embeddings.shape[0], 1)),
        )
        is_movmf = False
    else:
        raise ValueError(f"unsupported checkpoint model_type: {model_type!r}")

    split_identities = _validate_zero_shot_split(
        data_config,
        dataset_name=args.dataset_name,
        sketches=sketches,
        photos=photos,
    )
    checkpoint_metadata = checkpoint.get("metadata", {})
    if args.assignment_temperature is None:
        metadata_temperature = (
            checkpoint_metadata.get("assignment_temperature")
            if isinstance(checkpoint_metadata, dict)
            else None
        )
        args.assignment_temperature = (
            float(metadata_temperature) if metadata_temperature is not None else 0.05
        )
    _validate_provenance(
        checkpoint=checkpoint,
        sketches=sketches,
        photos=photos,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )
    if args.query_chunk_size <= 0:
        raise ValueError("query-chunk-size must be positive")
    if args.assignment_temperature <= 0:
        raise ValueError("assignment-temperature must be positive")
    if args.angular_logsumexp_temperature <= 0:
        raise ValueError("angular-logsumexp-temperature must be positive")

    query_count = sketches.embeddings.shape[0]
    gallery_count = photos.embeddings.shape[0]
    gallery = F.normalize(photos.embeddings.to(device), dim=-1)
    gallery_labels = photos.labels.to(device)
    sketch_embeddings = F.normalize(sketches.embeddings, dim=-1)
    directions = F.normalize(prediction.mean_directions, dim=-1)
    probabilities = prediction.mixture_logits.softmax(dim=-1)
    gate_barycenter = F.normalize(
        (probabilities[:, :, None] * directions).sum(dim=1), dim=-1
    )
    uniform_barycenter = F.normalize(directions.mean(dim=1), dim=-1)

    modes = [f"mu{component}" for component in range(num_components)]
    modes.extend(
        [
            "gate_barycenter",
            "uniform_barycenter",
            "max_component",
            "angular_logsumexp",
        ]
    )
    if is_movmf:
        modes.append("density")
    accumulators = {mode: _metric_accumulator() for mode in modes}
    primary_ap = torch.empty(query_count)
    primary_positive_similarity = torch.empty(query_count)
    semantic_margin = torch.empty(query_count)
    top1_correct = torch.empty(query_count)
    prior_entropy_per_query = torch.empty(query_count)
    posterior_entropy_per_query = torch.empty(query_count)
    posterior_effective_per_query = torch.empty(query_count)
    component_disagreement_per_query = torch.empty(query_count)
    kappa_per_query = torch.empty(query_count)
    class_size_per_query = torch.empty(query_count)

    positive_responsibility_sum = torch.zeros(num_components, dtype=torch.float64)
    positive_hard_counts = torch.zeros(num_components, dtype=torch.float64)
    positive_entropy_sum = 0.0
    positive_effective_sum = 0.0
    positive_max_sum = 0.0
    positive_pair_count = 0
    query_usage_effective_sum = 0.0
    query_usage_max_sum = 0.0
    positive_similarity_sum = torch.zeros(num_components, dtype=torch.float64)
    positive_centroid_similarity_sum = torch.zeros(num_components, dtype=torch.float64)
    centroid_query_count = 0
    sketch_similarity_sum = torch.zeros(num_components, dtype=torch.float64)
    pairwise_sum = torch.zeros(num_components, dtype=torch.float64)
    pairwise_count = 0

    rank_positions = torch.arange(1, gallery_count + 1, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, query_count, args.query_chunk_size):
            stop = min(start + args.query_chunk_size, query_count)
            d = directions[start:stop].to(device)
            p = probabilities[start:stop].to(device)
            score_batches: dict[str, torch.Tensor] = {
                "mu0": d[:, 0] @ gallery.T,
                "gate_barycenter": gate_barycenter[start:stop].to(device) @ gallery.T,
                "uniform_barycenter": uniform_barycenter[start:stop].to(device) @ gallery.T,
            }
            for component in range(1, num_components):
                score_batches[f"mu{component}"] = d[:, component] @ gallery.T
            component_cosines = torch.einsum("bkd,gd->bkg", d, gallery)
            score_batches["max_component"] = component_cosines.max(dim=1).values
            score_batches["angular_logsumexp"] = args.angular_logsumexp_temperature * torch.logsumexp(
                component_cosines / args.angular_logsumexp_temperature,
                dim=1,
            )
            if is_movmf:
                movmf_chunk = MoVmfPrediction(
                    mean_directions=d,
                    concentrations=prediction.concentrations[start:stop].to(device),
                    mixture_logits=prediction.mixture_logits[start:stop].to(device),
                )
                score_batches["density"] = mo_vmf_gallery_scores(
                    movmf_chunk,
                    gallery,
                    normalized=True,
                )

            labels = sketches.labels[start:stop].to(device)
            for mode, scores in score_batches.items():
                ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
                relevant = gallery_labels[ranked].eq(labels[:, None])
                positives = relevant.sum(dim=1).float()
                precision_at_rank = relevant.float().cumsum(dim=1) / rank_positions
                ap = (precision_at_rank * relevant.float()).sum(dim=1) / positives.clamp_min(1)
                prefix = relevant[:, :200].float()
                prefix_precision = precision_at_rank[:, :200]
                prefix_hits = prefix.sum(dim=1)
                values = {
                    "prefix_positive": (prefix_precision * prefix).sum(dim=1) / prefix_hits.clamp_min(1),
                    "all_relevant": (prefix_precision * prefix).sum(dim=1) / positives.clamp_min(1),
                    "min_relevant_k": (prefix_precision * prefix).sum(dim=1) / torch.minimum(
                        positives, torch.full_like(positives, 200)
                    ).clamp_min(1),
                }
                accumulator = accumulators[mode]
                accumulator["ap"].append(ap.cpu())
                for name, value in values.items():
                    accumulator["map_at_200"][name].append(value.cpu())
                for k in accumulator["precision"]:
                    accumulator["precision"][k].append(relevant[:, :k].float().mean(dim=1).cpu())
                accumulator["top1_correct"].append(relevant[:, 0].float().cpu())
                if mode == "gate_barycenter":
                    ranked_scores = scores.gather(dim=1, index=ranked)
                    primary_ap[start:stop] = ap.cpu()
                    primary_positive_similarity[start:stop] = (
                        (ranked_scores * relevant.float()).sum(dim=1)
                        / positives.clamp_min(1)
                    ).cpu()
                    semantic_margin[start:stop] = (
                        primary_positive_similarity[start:stop]
                        - ranked_scores.masked_fill(relevant, -torch.inf)
                        .max(dim=1)
                        .values.cpu()
                    )
                    top1_correct[start:stop] = relevant[:, 0].float().cpu()

            prior = p
            prior_entropy = -(prior * prior.clamp_min(1e-12).log()).sum(dim=-1)
            prior_entropy_per_query[start:stop] = prior_entropy.cpu()
            component_pairwise = _component_pair_cosines(d)
            component_disagreement_per_query[start:stop] = (
                1.0 - component_pairwise.mean(dim=-1)
            ).cpu()
            sketch_similarity_sum += (
                (d * sketch_embeddings[start:stop].to(device)[:, None, :]).sum(dim=-1)
                .double()
                .sum(dim=0)
                .cpu()
            )
            pairwise_sum += component_pairwise.double().sum(dim=0).cpu()
            pairwise_count += component_pairwise.shape[0]

            # Label-based positive diagnostics.  For deterministic controls the
            # same angular softmax is reported explicitly as post-hoc routing;
            # it is not part of the no-routing training objective.
            query_posterior_entropy = torch.empty(stop - start, device=device)
            query_posterior_effective = torch.empty(stop - start, device=device)
            query_usage_max = torch.empty(stop - start, device=device)
            for label in torch.unique(labels):
                local = torch.nonzero(labels.eq(label)).flatten()
                class_gallery = gallery[gallery_labels.eq(label)]
                class_cosines = (
                    d[local, :, None, :] * class_gallery[None, None, :, :]
                ).sum(dim=-1)
                if is_movmf:
                    concentration = prediction.concentrations[start:stop].to(device)[local]
                    log_weights = torch.log(p[local].clamp_min(1e-12))
                    log_norm = log_vmf_normalizer(
                        concentration,
                        dimension=d.shape[-1],
                        relative_to_uniform=True,
                    )
                    responsibility_logits = (
                        log_weights[:, :, None]
                        + log_norm[:, :, None]
                        + concentration[:, :, None] * class_cosines
                    )
                else:
                    responsibility_logits = (
                        torch.log(p[local].clamp_min(1e-12))[:, :, None]
                        + class_cosines / args.assignment_temperature
                    )
                responsibilities = responsibility_logits.softmax(dim=1).permute(0, 2, 1)
                entropy = -(
                    responsibilities * responsibilities.clamp_min(1e-12).log()
                ).sum(dim=-1)
                query_mean = responsibilities.mean(dim=1)
                query_entropy = -(
                    query_mean * query_mean.clamp_min(1e-12).log()
                ).sum(dim=-1)
                query_posterior_entropy[local] = entropy.mean(dim=1)
                query_posterior_effective[local] = query_entropy.exp()
                query_usage_max[local] = query_mean.max(dim=-1).values
                class_size_per_query[start + local.cpu()] = class_gallery.shape[0]
                positive_responsibility_sum += responsibilities.sum(dim=(0, 1)).double().cpu()
                positive_hard_counts += torch.bincount(
                    responsibilities.argmax(dim=-1).flatten(),
                    minlength=num_components,
                ).double().cpu()
                positive_entropy_sum += entropy.double().sum().item()
                positive_effective_sum += entropy.exp().double().sum().item()
                positive_max_sum += responsibilities.max(dim=-1).values.double().sum().item()
                positive_pair_count += entropy.numel()
                query_usage_effective_sum += query_entropy.exp().double().sum().item()
                query_usage_max_sum += query_mean.max(dim=-1).values.double().sum().item()
                positive_similarity_sum += class_cosines.double().sum(dim=(0, 2)).cpu()
                centroid = F.normalize(class_gallery.mean(dim=0), dim=-1)
                positive_centroid_similarity_sum += (
                    (d[local] * centroid[None, :]).sum(dim=-1).double().sum(dim=0).cpu()
                )
                centroid_query_count += local.numel()
            posterior_entropy_per_query[start:stop] = query_posterior_entropy.cpu()
            posterior_effective_per_query[start:stop] = query_posterior_effective.cpu()
            # The mean concentration is the uncertainty feature for Mo-vMF;
            # deterministic controls have no concentration head.
            if is_movmf:
                kappa_per_query[start:stop] = prediction.concentrations[start:stop].mean(dim=-1)
            else:
                kappa_per_query[start:stop] = float("nan")

    metrics = {
        mode: _finish_metrics(
            accumulator,
            num_queries=query_count,
            num_gallery_items=gallery_count,
        )
        for mode, accumulator in accumulators.items()
    }
    mean_prior = probabilities.double().mean(dim=0)
    prior_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    pairwise_mean = pairwise_sum / pairwise_count
    responsibility_stats = {
        "source": "positive_gallery_vmf_posterior"
        if is_movmf
        else "posthoc_positive_gallery_angular_softmax",
        "assignment_temperature": None if is_movmf else args.assignment_temperature,
        "num_positive_pairs": positive_pair_count,
        "mean_responsibilities": (positive_responsibility_sum / positive_pair_count).tolist(),
        "hard_assignment_fractions": (positive_hard_counts / positive_pair_count).tolist(),
        "mean_entropy": positive_entropy_sum / positive_pair_count,
        "normalized_mean_entropy": (
            0.0
            if num_components == 1
            else positive_entropy_sum / positive_pair_count / math.log(num_components)
        ),
        "mean_effective_components": positive_effective_sum / positive_pair_count,
        "mean_max_responsibility": positive_max_sum / positive_pair_count,
        "mean_query_usage_effective_components": query_usage_effective_sum / query_count,
        "mean_query_max_component_usage": query_usage_max_sum / query_count,
    }
    correlation_features = {
        "query_ap": primary_ap,
        "positive_cosine": primary_positive_similarity,
        "semantic_margin": semantic_margin,
        "prior_entropy": prior_entropy_per_query,
        "posterior_entropy": posterior_entropy_per_query,
        "posterior_effective_components": posterior_effective_per_query,
        "component_disagreement": component_disagreement_per_query,
        "top1_correct": top1_correct,
        "class_size": class_size_per_query,
    }
    if is_movmf:
        finite_kappa = torch.isfinite(kappa_per_query)
        correlations = {
            f"kappa_vs_{name}": _safe_spearman(
                kappa_per_query[finite_kappa], values[finite_kappa]
            )
            for name, values in correlation_features.items()
        }
    else:
        correlations = {f"kappa_vs_{name}": None for name in correlation_features}
    correlations.update(
        {
            f"{name}_vs_query_ap": _safe_spearman(values, primary_ap)
            for name, values in correlation_features.items()
            if name != "query_ap"
        }
    )
    component_roles = []
    for component in range(num_components):
        component_roles.append(
            {
                "component": component,
                "mean_gate_weight": mean_prior[component].item(),
                "mean_positive_responsibility": responsibility_stats["mean_responsibilities"][component],
                "retrieval": metrics[f"mu{component}"],
                "mean_cosine_to_raw_sketch": (sketch_similarity_sum[component] / query_count).item(),
                "mean_cosine_to_individual_positive": (
                    positive_similarity_sum[component] / positive_pair_count
                ).item(),
                "mean_cosine_to_positive_centroid": (
                    positive_centroid_similarity_sum[component] / centroid_query_count
                ).item(),
            }
        )
    thresholds = {}
    if is_movmf:
        for fixed_kappa in (128, 256, 512, 1024):
            thresholds[str(fixed_kappa)] = [
                (
                    math.log(mean_prior[0].item() / mean_prior[component].item())
                    / fixed_kappa
                )
                if component != 0
                else 0.0
                for component in range(num_components)
            ]
    elif model_type == "deterministic_k3_photo_predictor":
        # The deterministic analyzer uses the same angular softmax only as a
        # post-hoc positive-gallery diagnostic, so its equivalent boundary is
        # tau * log(pi_0 / pi_s), not log(pi_0 / pi_s) / kappa.
        thresholds[f"angular_tau_{args.assignment_temperature:g}"] = [
            (
                args.assignment_temperature
                * math.log(mean_prior[0].item() / mean_prior[component].item())
            )
            if component != 0
            else 0.0
            for component in range(num_components)
        ]

    result = {
        "protocol": "standard_zs_sbir_sketch_only_inference",
        "diagnostic_test_evaluation": True,
        "uses_test_labels_for_scoring": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "split_identities": split_identities,
        "model_type": model_type,
        "is_movmf": is_movmf,
        "num_components": num_components,
        "num_positive_photos": checkpoint.get("metadata", {}).get(
            "positives_per_anchor_per_step"
        ),
        "trainable_parameters": sum(p.numel() for p in predictor.parameters()),
        "model_config": model_config,
        "training_metadata": checkpoint.get("metadata", {}),
        "metric_denominator_note": {
            "full_mAP": "full-gallery AP divided by all category-relevant gallery items",
            "mAP@200_prefix_positive": "top-200 AP divided by positives found in top 200",
            "mAP@200_all_relevant": "top-200 AP divided by all category-relevant gallery items",
            "mAP@200_min_relevant_k": "top-200 AP divided by min(total relevant, 200)",
        },
        "metrics": metrics,
        "component_roles": component_roles,
        "mean_prior": mean_prior.tolist(),
        "mean_prior_entropy": prior_entropy.double().mean().item(),
        "mean_prior_effective_components": prior_entropy.exp().double().mean().item(),
        "mean_pairwise_component_cosine": pairwise_mean.tolist(),
        "mean_component_disagreement": (1.0 - pairwise_mean.mean()).item(),
        "responsibility_statistics": responsibility_stats,
        "correlations_spearman": correlations,
        "kappa_statistics": (
            {
                "mean": prediction.concentrations.double().mean().item(),
                "std": prediction.concentrations.double().std(unbiased=False).item(),
                "min": prediction.concentrations.min().item(),
                "max": prediction.concentrations.max().item(),
                "upper_bound": max_concentration,
                "mean_by_component": prediction.concentrations.double()
                .mean(dim=0)
                .tolist(),
                "fraction_near_upper_bound": prediction.concentrations.ge(
                    max_concentration - 0.01 * max_concentration
                ).double().mean().item(),
            }
            if is_movmf
            else None
        ),
        "cosine_advantage_threshold_log_prior_ratio_over_kappa": thresholds,
        "assignment_temperature": (
            None if is_movmf else args.assignment_temperature
        ),
        "angular_logsumexp_temperature": args.angular_logsumexp_temperature,
    }
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
