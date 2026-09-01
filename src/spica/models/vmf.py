import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .retrieval import MoVmfPrediction, VmfPrediction

LOG_NORMALIZER_VERSION = "uniform_asymptotic_order2_v1"
SCORE_NORMALIZATION_VERSION = "query_baseline_effective_kappa_v1"


@dataclass(frozen=True, slots=True)
class K1VmfLoss:
    total: Tensor
    positive_nll: Tensor
    cosine_ranking: Tensor
    positive_cosine: Tensor
    negative_cosine: Tensor
    relative_log_normalizer: Tensor


def log_vmf_normalizer(
    concentration: Tensor,
    *,
    dimension: int,
    relative_to_uniform: bool = False,
) -> Tensor:
    """Approximate log C_D(kappa) stably for high-dimensional vMF densities."""
    if dimension < 32:
        raise ValueError(
            "The high-dimensional vMF approximation requires dimension >= 32, "
            f"got {dimension}"
        )
    if not concentration.is_floating_point():
        raise TypeError(
            f"concentration must be floating-point, got {concentration.dtype}"
        )
    if not torch.isfinite(concentration).all().item():
        raise ValueError("concentration must contain only finite values")
    if torch.any(concentration < 0).item():
        raise ValueError("concentration must be non-negative")

    output_dtype = concentration.dtype
    kappa = concentration.to(dtype=torch.float64)
    nu = dimension / 2.0 - 1.0

    # DLMF 10.41 large-order expansion of I_nu(nu*z), through u_2/nu^2.
    # This algebraic form computes log C_D(kappa) - log C_D(0) directly,
    # avoiding cancellation between nu*log(kappa) and nu*eta(kappa/nu).
    z = kappa / nu
    radius = torch.sqrt(1.0 + z.square())
    delta = z.square() / (radius + 1.0)
    t = radius.reciprocal()

    u1 = (3.0 * t - 5.0 * t.pow(3)) / 24.0
    u2 = (81.0 * t.square() - 462.0 * t.pow(4) + 385.0 * t.pow(6)) / 1152.0
    correction = u1 / nu + u2 / (nu * nu)

    u1_at_zero = -1.0 / 12.0
    u2_at_zero = 1.0 / 288.0
    correction_at_zero = u1_at_zero / nu + u2_at_zero / (nu * nu)

    relative_log_normalizer = (
        nu * (torch.log1p(delta / 2.0) - delta)
        + 0.5 * torch.log1p(delta)
        - torch.log1p((correction - correction_at_zero) / (1.0 + correction_at_zero))
    )

    if relative_to_uniform:
        return relative_log_normalizer.to(dtype=output_dtype)

    log_uniform_normalizer = (
        math.lgamma(dimension / 2.0)
        - math.log(2.0)
        - (dimension / 2.0) * math.log(math.pi)
    )
    return (relative_log_normalizer + log_uniform_normalizer).to(dtype=output_dtype)


def k1_vmf_retrieval_loss(
    prediction: VmfPrediction,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    *,
    margin: float = 0.2,
    nll_weight: float = 1.0,
    ranking_weight: float = 1.0,
) -> K1VmfLoss:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    for name, weight in (
        ("nll_weight", nll_weight),
        ("ranking_weight", ranking_weight),
    ):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {weight}")
    if nll_weight == 0 and ranking_weight == 0:
        raise ValueError("At least one loss weight must be positive")

    direction = prediction.mean_direction
    concentration = prediction.concentration
    named_embeddings = (
        ("mean_direction", direction),
        ("positive_embeddings", positive_embeddings),
        ("negative_embeddings", negative_embeddings),
    )
    for name, embeddings in named_embeddings:
        if embeddings.ndim != 2:
            raise ValueError(
                f"{name} must have shape [batch_size, embedding_dim], "
                f"got {tuple(embeddings.shape)}"
            )
        if not embeddings.is_floating_point():
            raise TypeError(f"{name} must be floating-point, got {embeddings.dtype}")
        if not torch.isfinite(embeddings).all().item():
            raise ValueError(f"{name} must contain only finite values")

    if positive_embeddings.shape != direction.shape:
        raise ValueError(
            "positive_embeddings must match mean_direction shape "
            f"{tuple(direction.shape)}, got {tuple(positive_embeddings.shape)}"
        )
    if negative_embeddings.shape != direction.shape:
        raise ValueError(
            "negative_embeddings must match mean_direction shape "
            f"{tuple(direction.shape)}, got {tuple(negative_embeddings.shape)}"
        )
    if concentration.shape != direction.shape[:1]:
        raise ValueError(
            "concentration must have shape [batch_size], "
            f"got {tuple(concentration.shape)}"
        )
    if not concentration.is_floating_point():
        raise TypeError(
            f"concentration must be floating-point, got {concentration.dtype}"
        )
    if not torch.isfinite(concentration).all().item():
        raise ValueError("concentration must contain only finite values")
    if torch.any(concentration <= 0).item():
        raise ValueError("concentration must be strictly positive")

    positive_cosine = (direction * positive_embeddings).sum(dim=-1)
    negative_cosine = (direction * negative_embeddings).sum(dim=-1)
    relative_log_normalizer = log_vmf_normalizer(
        concentration,
        dimension=direction.shape[1],
        relative_to_uniform=True,
    )
    positive_log_likelihood = relative_log_normalizer + concentration * positive_cosine
    positive_nll = -positive_log_likelihood.mean()

    # Keep the deterministic cosine-margin contract. Letting learned kappa scale
    # this term would make concentration an escape variable on misranked pairs.
    cosine_ranking = F.softplus(margin - positive_cosine + negative_cosine).mean()
    total = nll_weight * positive_nll + ranking_weight * cosine_ranking

    return K1VmfLoss(
        total=total,
        positive_nll=positive_nll,
        cosine_ranking=cosine_ranking,
        positive_cosine=positive_cosine,
        negative_cosine=negative_cosine,
        relative_log_normalizer=relative_log_normalizer,
    )


@dataclass(frozen=True, slots=True)
class MoVmfLoss:
    total: Tensor
    positive_nll: Tensor
    density_ranking: Tensor
    normalized_positive_score: Tensor
    normalized_negative_score: Tensor
    positive_log_likelihood: Tensor
    posterior_responsibilities: Tensor
    mixture_probabilities: Tensor
    effective_concentration: Tensor


def _validate_mo_vmf_prediction(prediction: MoVmfPrediction) -> None:
    directions = prediction.mean_directions
    concentrations = prediction.concentrations
    mixture_logits = prediction.mixture_logits
    if directions.ndim != 3:
        raise ValueError(
            "mean_directions must have shape [batch_size, num_components, "
            f"embedding_dim], got {tuple(directions.shape)}"
        )
    if directions.shape[1] == 0:
        raise ValueError("mean_directions must contain at least one component")
    expected_parameter_shape = directions.shape[:2]
    if concentrations.shape != expected_parameter_shape:
        raise ValueError(
            "concentrations must match the first two mean_directions dimensions, "
            f"got {tuple(concentrations.shape)} and {expected_parameter_shape}"
        )
    if mixture_logits.shape != expected_parameter_shape:
        raise ValueError(
            "mixture_logits must match the first two mean_directions dimensions, "
            f"got {tuple(mixture_logits.shape)} and {expected_parameter_shape}"
        )

    for name, values in (
        ("mean_directions", directions),
        ("concentrations", concentrations),
        ("mixture_logits", mixture_logits),
    ):
        if not values.is_floating_point():
            raise TypeError(f"{name} must be floating-point, got {values.dtype}")
        if not torch.isfinite(values).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if torch.any(concentrations <= 0).item():
        raise ValueError("concentrations must be strictly positive")


def _validate_mo_vmf_targets(
    prediction: MoVmfPrediction,
    targets: Tensor,
    *,
    name: str,
    paired: bool,
) -> None:
    if targets.ndim != 2:
        raise ValueError(
            f"{name} must have shape [num_items, embedding_dim], "
            f"got {tuple(targets.shape)}"
        )
    if targets.shape[1] != prediction.mean_directions.shape[2]:
        raise ValueError(
            f"{name} embedding dimension must match mean_directions, got "
            f"{targets.shape[1]} and {prediction.mean_directions.shape[2]}"
        )
    if paired and targets.shape[0] != prediction.mean_directions.shape[0]:
        raise ValueError(
            f"Paired {name} batch size must match mean_directions, got "
            f"{targets.shape[0]} and {prediction.mean_directions.shape[0]}"
        )
    if not targets.is_floating_point():
        raise TypeError(f"{name} must be floating-point, got {targets.dtype}")
    if not torch.isfinite(targets).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _mo_vmf_base_terms(
    prediction: MoVmfPrediction,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    log_mixture_weights = F.log_softmax(prediction.mixture_logits, dim=-1)
    relative_log_normalizers = log_vmf_normalizer(
        prediction.concentrations,
        dimension=prediction.mean_directions.shape[2],
        relative_to_uniform=True,
    )
    base_terms = log_mixture_weights + relative_log_normalizers
    base_log_partition = torch.logsumexp(base_terms, dim=-1)
    base_probabilities = F.softmax(base_terms, dim=-1)
    effective_concentration = (base_probabilities * prediction.concentrations).sum(
        dim=-1
    )
    return (
        base_terms,
        base_log_partition,
        effective_concentration,
        log_mixture_weights,
    )


def mo_vmf_gallery_scores(
    prediction: MoVmfPrediction,
    gallery_embeddings: Tensor,
    *,
    normalized: bool = True,
) -> Tensor:
    """Score a gallery under a per-query Mo-vMF density.

    Normalization subtracts and divides by positive query-only constants, so it
    preserves gallery ordering. For one component it reduces exactly to cosine.
    """
    _validate_mo_vmf_prediction(prediction)
    _validate_mo_vmf_targets(
        prediction,
        gallery_embeddings,
        name="gallery_embeddings",
        paired=False,
    )
    base_terms, base_log_partition, effective_concentration, _ = _mo_vmf_base_terms(
        prediction
    )
    component_cosines = torch.einsum(
        "bkd,gd->bkg",
        prediction.mean_directions,
        gallery_embeddings,
    )
    component_log_likelihoods = (
        base_terms[:, :, None]
        + prediction.concentrations[:, :, None] * component_cosines
    )
    log_likelihoods = torch.logsumexp(component_log_likelihoods, dim=1)
    if not normalized:
        return log_likelihoods
    return (log_likelihoods - base_log_partition[:, None]) / (
        effective_concentration[:, None]
    )


def mo_vmf_retrieval_loss(
    prediction: MoVmfPrediction,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    *,
    margin: float = 0.2,
    nll_weight: float = 1.0,
    ranking_weight: float = 1.0,
) -> MoVmfLoss:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    for name, weight in (
        ("nll_weight", nll_weight),
        ("ranking_weight", ranking_weight),
    ):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {weight}")
    if nll_weight == 0 and ranking_weight == 0:
        raise ValueError("At least one loss weight must be positive")

    _validate_mo_vmf_prediction(prediction)
    _validate_mo_vmf_targets(
        prediction,
        positive_embeddings,
        name="positive_embeddings",
        paired=True,
    )
    _validate_mo_vmf_targets(
        prediction,
        negative_embeddings,
        name="negative_embeddings",
        paired=True,
    )

    base_terms, base_log_partition, effective_concentration, log_mixture_weights = (
        _mo_vmf_base_terms(prediction)
    )
    # Elementwise paired products avoid a batched outer-product backward kernel,
    # while computing the same per-component cosine values.
    positive_component_cosines = (
        prediction.mean_directions * positive_embeddings[:, None, :]
    ).sum(dim=-1)
    negative_component_cosines = (
        prediction.mean_directions * negative_embeddings[:, None, :]
    ).sum(dim=-1)
    positive_component_log_likelihoods = (
        base_terms + prediction.concentrations * positive_component_cosines
    )
    negative_component_log_likelihoods = (
        base_terms + prediction.concentrations * negative_component_cosines
    )
    positive_log_likelihood = torch.logsumexp(
        positive_component_log_likelihoods,
        dim=-1,
    )
    negative_log_likelihood = torch.logsumexp(
        negative_component_log_likelihoods,
        dim=-1,
    )

    # Subtracting a query-only baseline and dividing by a positive query-only
    # scale preserves the exact Mo-vMF gallery ordering. At K=1 this is exactly
    # cosine, matching the K=1 objective without letting raw kappa scale the
    # margin arbitrarily.
    normalized_positive_score = (
        positive_log_likelihood - base_log_partition
    ) / effective_concentration
    normalized_negative_score = (
        negative_log_likelihood - base_log_partition
    ) / effective_concentration

    positive_nll = -positive_log_likelihood.mean()
    density_ranking = F.softplus(
        margin - normalized_positive_score + normalized_negative_score
    ).mean()
    total = nll_weight * positive_nll + ranking_weight * density_ranking

    return MoVmfLoss(
        total=total,
        positive_nll=positive_nll,
        density_ranking=density_ranking,
        normalized_positive_score=normalized_positive_score,
        normalized_negative_score=normalized_negative_score,
        positive_log_likelihood=positive_log_likelihood,
        posterior_responsibilities=F.softmax(
            positive_component_log_likelihoods,
            dim=-1,
        ),
        mixture_probabilities=log_mixture_weights.exp(),
        effective_concentration=effective_concentration,
    )


@dataclass(frozen=True, slots=True)
class MoVmfMultiPositiveLoss:
    total: Tensor
    positive_nll: Tensor
    density_ranking: Tensor
    posterior_balance: Tensor
    posterior_sharpness: Tensor
    balanced_assignment: Tensor
    direction_diversity: Tensor
    normalized_positive_scores: Tensor
    normalized_negative_score: Tensor
    ranking_positive_scores: Tensor
    ranking_negative_score: Tensor
    positive_log_likelihoods: Tensor
    posterior_responsibilities: Tensor
    mean_query_responsibilities: Tensor
    mixture_probabilities: Tensor
    effective_concentration: Tensor


def mo_vmf_multi_positive_retrieval_loss(
    prediction: MoVmfPrediction,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    *,
    margin: float = 0.2,
    nll_weight: float = 1.0,
    ranking_weight: float = 1.0,
    balance_weight: float = 0.0,
    sharpness_weight: float = 0.0,
    diversity_weight: float = 0.0,
    assignment_weight: float = 0.0,
    diversity_cosine_threshold: float = 0.9,
    ranking_score_transform: str = "identity",
) -> MoVmfMultiPositiveLoss:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    for name, weight in (
        ("nll_weight", nll_weight),
        ("ranking_weight", ranking_weight),
        ("balance_weight", balance_weight),
        ("sharpness_weight", sharpness_weight),
        ("diversity_weight", diversity_weight),
        ("assignment_weight", assignment_weight),
    ):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {weight}")
    if nll_weight == 0 and ranking_weight == 0:
        raise ValueError("At least one retrieval loss weight must be positive")
    if not math.isfinite(diversity_cosine_threshold) or not (
        -1 <= diversity_cosine_threshold <= 1
    ):
        raise ValueError(
            "diversity_cosine_threshold must be finite and in [-1, 1], "
            f"got {diversity_cosine_threshold}"
        )
    if ranking_score_transform not in {
        "identity",
        "tanh",
        "semantic_barycenter",
    }:
        raise ValueError(
            "ranking_score_transform must be 'identity', 'tanh', or "
            f"'semantic_barycenter', got {ranking_score_transform!r}"
        )

    _validate_mo_vmf_prediction(prediction)
    if positive_embeddings.ndim != 3:
        raise ValueError(
            "positive_embeddings must have shape [batch_size, num_positives, "
            f"embedding_dim], got {tuple(positive_embeddings.shape)}"
        )
    if positive_embeddings.shape[0] != prediction.mean_directions.shape[0]:
        raise ValueError("Positive and prediction batch sizes must match")
    if positive_embeddings.shape[1] == 0:
        raise ValueError("positive_embeddings must contain at least one positive")
    if positive_embeddings.shape[2] != prediction.mean_directions.shape[2]:
        raise ValueError("Positive and prediction embedding dimensions must match")
    if not positive_embeddings.is_floating_point():
        raise TypeError("positive_embeddings must be floating-point")
    if not torch.isfinite(positive_embeddings).all().item():
        raise ValueError("positive_embeddings must contain only finite values")
    _validate_mo_vmf_targets(
        prediction,
        negative_embeddings,
        name="negative_embeddings",
        paired=True,
    )

    base_terms, base_log_partition, effective_concentration, log_mixture_weights = (
        _mo_vmf_base_terms(prediction)
    )
    positive_component_cosines = (
        prediction.mean_directions[:, :, None, :] * positive_embeddings[:, None, :, :]
    ).sum(dim=-1)
    negative_component_cosines = (
        prediction.mean_directions * negative_embeddings[:, None, :]
    ).sum(dim=-1)
    positive_component_log_likelihoods = (
        base_terms[:, :, None]
        + prediction.concentrations[:, :, None] * positive_component_cosines
    )
    negative_component_log_likelihoods = (
        base_terms + prediction.concentrations * negative_component_cosines
    )
    positive_log_likelihoods = torch.logsumexp(
        positive_component_log_likelihoods,
        dim=1,
    )
    negative_log_likelihood = torch.logsumexp(
        negative_component_log_likelihoods,
        dim=-1,
    )
    normalized_positive_scores = (
        positive_log_likelihoods - base_log_partition[:, None]
    ) / effective_concentration[:, None]
    normalized_negative_score = (
        negative_log_likelihood - base_log_partition
    ) / effective_concentration
    if ranking_score_transform == "tanh":
        ranking_positive_scores = normalized_positive_scores.tanh()
        ranking_negative_score = normalized_negative_score.tanh()
    elif ranking_score_transform == "semantic_barycenter":
        semantic_direction = F.normalize(
            (log_mixture_weights.exp()[:, :, None] * prediction.mean_directions).sum(
                dim=1
            ),
            dim=-1,
        )
        ranking_positive_scores = (
            semantic_direction[:, None, :] * positive_embeddings
        ).sum(dim=-1)
        ranking_negative_score = (semantic_direction * negative_embeddings).sum(dim=-1)
    else:
        ranking_positive_scores = normalized_positive_scores
        ranking_negative_score = normalized_negative_score

    positive_nll = -positive_log_likelihoods.mean()
    density_ranking = F.softplus(
        margin - ranking_positive_scores + ranking_negative_score[:, None]
    ).mean()

    responsibilities = F.softmax(
        positive_component_log_likelihoods,
        dim=1,
    ).permute(0, 2, 1)
    mean_query_responsibilities = responsibilities.mean(dim=1)
    num_components = prediction.mean_directions.shape[1]
    posterior_balance = (
        (
            mean_query_responsibilities
            * (mean_query_responsibilities.clamp_min(1e-12) * num_components).log()
        )
        .sum(dim=-1)
        .mean()
    )
    posterior_sharpness = (
        -(responsibilities * responsibilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
    )

    if assignment_weight > 0:
        num_positives = positive_embeddings.shape[1]
        if num_positives != num_components:
            raise ValueError(
                "Balanced assignment requires num_positives == num_components, "
                f"got {num_positives} and {num_components}"
            )
        if num_components > 6:
            raise ValueError(
                "Exact balanced assignment supports at most 6 components, "
                f"got {num_components}"
            )
        with torch.no_grad():
            permutations = torch.tensor(
                tuple(itertools.permutations(range(num_components))),
                dtype=torch.long,
                device=positive_embeddings.device,
            )
            cosines_by_target = positive_component_cosines.permute(0, 2, 1)
            expanded_cosines = cosines_by_target[:, None].expand(
                -1,
                permutations.shape[0],
                -1,
                -1,
            )
            gather_indices = permutations[None, :, :, None].expand(
                cosines_by_target.shape[0],
                -1,
                -1,
                -1,
            )
            assignment_scores = (
                expanded_cosines.gather(
                    dim=-1,
                    index=gather_indices,
                )
                .squeeze(-1)
                .sum(dim=-1)
            )
            best_permutations = permutations[assignment_scores.argmax(dim=-1)]
            assignment_targets = F.one_hot(
                best_permutations,
                num_classes=num_components,
            ).to(dtype=responsibilities.dtype)
        balanced_assignment = (
            -(assignment_targets * responsibilities.clamp_min(1e-12).log())
            .sum(dim=-1)
            .mean()
        )
    else:
        balanced_assignment = prediction.mean_directions.new_zeros(())

    if num_components > 1:
        component_pairs = torch.triu_indices(
            num_components,
            num_components,
            offset=1,
            device=prediction.mean_directions.device,
        )
        component_cosines = (
            prediction.mean_directions[:, component_pairs[0], :]
            * prediction.mean_directions[:, component_pairs[1], :]
        ).sum(dim=-1)
        direction_diversity = (
            F.relu(component_cosines - diversity_cosine_threshold).square().mean()
        )
    else:
        direction_diversity = prediction.mean_directions.new_zeros(())

    total = (
        nll_weight * positive_nll
        + ranking_weight * density_ranking
        + balance_weight * posterior_balance
        + sharpness_weight * posterior_sharpness
        + assignment_weight * balanced_assignment
        + diversity_weight * direction_diversity
    )
    return MoVmfMultiPositiveLoss(
        total=total,
        positive_nll=positive_nll,
        density_ranking=density_ranking,
        posterior_balance=posterior_balance,
        posterior_sharpness=posterior_sharpness,
        balanced_assignment=balanced_assignment,
        direction_diversity=direction_diversity,
        normalized_positive_scores=normalized_positive_scores,
        normalized_negative_score=normalized_negative_score,
        ranking_positive_scores=ranking_positive_scores,
        ranking_negative_score=ranking_negative_score,
        positive_log_likelihoods=positive_log_likelihoods,
        posterior_responsibilities=responsibilities,
        mean_query_responsibilities=mean_query_responsibilities,
        mixture_probabilities=log_mixture_weights.exp(),
        effective_concentration=effective_concentration,
    )


@dataclass(frozen=True, slots=True)
class DominantSatelliteRegularization:
    gate_prior: Tensor
    dominant_sketch_anchor: Tensor
    dominant_photo_anchor: Tensor
    semantic_consistency: Tensor
    satellite_coverage: Tensor
    spread_matching: Tensor
    satellite_concentration_floor: Tensor
    semantic_center: Tensor


def dominant_satellite_regularization(
    prediction: MoVmfPrediction,
    sketch_embeddings: Tensor,
    positive_embeddings: Tensor,
    *,
    target_dominant_weight: float = 0.8,
    consistency_temperature: float = 0.07,
    satellite_concentration_floor: float = 16.0,
) -> DominantSatelliteRegularization:
    _validate_mo_vmf_prediction(prediction)
    num_components = prediction.mean_directions.shape[1]
    if num_components < 2:
        raise ValueError("Dominant-satellite regularization requires K >= 2")
    if not math.isfinite(target_dominant_weight) or not (
        1.0 / num_components < target_dominant_weight < 1.0
    ):
        raise ValueError(
            "target_dominant_weight must be between 1 / K and 1, "
            f"got {target_dominant_weight}"
        )
    if not math.isfinite(consistency_temperature) or consistency_temperature <= 0:
        raise ValueError("consistency_temperature must be finite and positive")
    if (
        not math.isfinite(satellite_concentration_floor)
        or satellite_concentration_floor <= 0
    ):
        raise ValueError("satellite_concentration_floor must be finite and positive")
    if sketch_embeddings.ndim != 2 or sketch_embeddings.shape != (
        prediction.mean_directions.shape[0],
        prediction.mean_directions.shape[2],
    ):
        raise ValueError("sketch_embeddings shape must match the prediction batch")
    if (
        positive_embeddings.ndim != 3
        or positive_embeddings.shape[0] != prediction.mean_directions.shape[0]
        or positive_embeddings.shape[1] < 2
        or positive_embeddings.shape[2] != prediction.mean_directions.shape[2]
    ):
        raise ValueError(
            "positive_embeddings must have shape [batch_size, M, embedding_dim]"
        )
    for name, embeddings in (
        ("sketch_embeddings", sketch_embeddings),
        ("positive_embeddings", positive_embeddings),
    ):
        if not embeddings.is_floating_point():
            raise TypeError(f"{name} must be floating-point")
        if not torch.isfinite(embeddings).all().item():
            raise ValueError(f"{name} must contain only finite values")

    sketch_directions = F.normalize(sketch_embeddings, dim=-1)
    positive_directions = F.normalize(positive_embeddings, dim=-1)
    log_probabilities = F.log_softmax(prediction.mixture_logits, dim=-1)
    probabilities = log_probabilities.exp()
    satellite_weight = (1.0 - target_dominant_weight) / (num_components - 1)
    target_probabilities = probabilities.new_full(
        (num_components,),
        satellite_weight,
    )
    target_probabilities[0] = target_dominant_weight
    gate_prior = (
        (target_probabilities * (target_probabilities.log() - log_probabilities))
        .sum(dim=-1)
        .mean()
    )

    dominant_direction = prediction.mean_directions[:, 0]
    dominant_sketch_anchor = (
        1.0 - (dominant_direction * sketch_directions).sum(dim=-1)
    ).mean()
    positive_centroid = F.normalize(positive_directions.mean(dim=1), dim=-1)
    dominant_photo_anchor = (
        1.0 - (dominant_direction * positive_centroid).sum(dim=-1)
    ).mean()

    semantic_center = F.normalize(
        (probabilities[:, :, None] * prediction.mean_directions).sum(dim=1),
        dim=-1,
    )
    consistency_logits = semantic_center @ sketch_directions.T / consistency_temperature
    labels = torch.arange(
        consistency_logits.shape[0],
        device=consistency_logits.device,
    )
    semantic_consistency = 0.5 * (
        F.cross_entropy(consistency_logits, labels)
        + F.cross_entropy(consistency_logits.T, labels)
    )

    satellite_cosines = (
        prediction.mean_directions[:, 1:, None, :] * positive_directions[:, None, :, :]
    ).sum(dim=-1)
    satellite_coverage = (1.0 - satellite_cosines.max(dim=-1).values).mean()

    positive_pairs = torch.triu_indices(
        positive_directions.shape[1],
        positive_directions.shape[1],
        offset=1,
        device=positive_directions.device,
    )
    positive_pairwise_cosines = (
        positive_directions[:, positive_pairs[0], :]
        * positive_directions[:, positive_pairs[1], :]
    ).sum(dim=-1)
    target_spread = positive_pairwise_cosines.mean(dim=-1)
    component_pairs = torch.triu_indices(
        num_components,
        num_components,
        offset=1,
        device=prediction.mean_directions.device,
    )
    component_pairwise_cosines = (
        prediction.mean_directions[:, component_pairs[0], :]
        * prediction.mean_directions[:, component_pairs[1], :]
    ).sum(dim=-1)
    spread_matching = (
        (component_pairwise_cosines - target_spread[:, None]).square().mean()
    )

    satellite_concentrations = prediction.concentrations[:, 1:]
    satellite_concentration_floor_loss = (
        (
            F.relu(satellite_concentration_floor - satellite_concentrations)
            / satellite_concentration_floor
        )
        .square()
        .mean()
    )
    return DominantSatelliteRegularization(
        gate_prior=gate_prior,
        dominant_sketch_anchor=dominant_sketch_anchor,
        dominant_photo_anchor=dominant_photo_anchor,
        semantic_consistency=semantic_consistency,
        satellite_coverage=satellite_coverage,
        spread_matching=spread_matching,
        satellite_concentration_floor=satellite_concentration_floor_loss,
        semantic_center=semantic_center,
    )
