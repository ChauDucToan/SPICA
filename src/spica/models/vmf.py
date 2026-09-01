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
