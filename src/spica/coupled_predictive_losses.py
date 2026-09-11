"""Versioned loss graphs for coupled predictive V1 and contextual fusion V2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .models.coupled_predictive import CoupledPredictiveModel, CoupledPredictiveOutput

_MARGIN = 0.2
_TEMPERATURE = 0.07


def _rank_live(query: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
    """Softplus ranking against the live prompted photo bank."""
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    positive_score = (query * positive).sum(dim=-1)
    negative_score = (query[:, None] * negative).sum(dim=-1)
    return F.softplus(_MARGIN + negative_score - positive_score[:, None]).mean()


def _main_photo_multi_positive(
    query: Tensor, live_photos: Tensor, photo_labels: Tensor, labels: Tensor
) -> Tensor:
    """Average supervised contrastive loss over every positive photo."""
    logits = F.normalize(query, dim=-1) @ F.normalize(live_photos, dim=-1).T
    log_probability = F.log_softmax(logits / _TEMPERATURE, dim=-1)
    positive = photo_labels[None, :] == labels[:, None]
    positive_count = positive.sum(dim=-1)
    if not bool(positive_count.gt(0).all()):
        raise ValueError("multi-positive objective requires at least one live photo per label")
    return (-(log_probability * positive).sum(dim=-1) / positive_count).mean()


def _ce(query: Tensor, text: Tensor, classids: Tensor, labels: Tensor) -> Tensor:
    positions = torch.searchsorted(classids, labels)
    logits = F.normalize(query, dim=-1) @ F.normalize(text, dim=-1).T
    return F.cross_entropy(logits / _TEMPERATURE, positions)


def _align(query: Tensor, target: Tensor) -> Tensor:
    return (1.0 - F.cosine_similarity(query, target.detach(), dim=-1)).mean()


def _view_terms(
    output: CoupledPredictiveOutput,
    positive: Tensor,
    negative: Tensor,
    text: Tensor,
    classids: Tensor,
    labels: Tensor,
    architecture: str,
    *,
    live_photos: Tensor | None = None,
    photo_labels: Tensor | None = None,
    main_photo_objective: str = "paired_softplus",
) -> dict[str, Tensor]:
    terms = {
        "rank_pool": _rank_live(output.q, positive, negative),
        "ce_pool": _ce(output.q, text, classids, labels),
    }
    if architecture in {"predictive", "predictive_fusion_v2"}:
        if main_photo_objective == "paired_softplus":
            rank_key, rank_value = "rank_i", _rank_live(output.mu_i, positive, negative)
        elif main_photo_objective == "multi_positive_supervised_contrastive":
            if live_photos is None or photo_labels is None:
                raise ValueError("multi-positive objective requires live photos and labels")
            rank_key, rank_value = "rank_i", _main_photo_multi_positive(output.mu_i, live_photos, photo_labels, labels)
        elif main_photo_objective == "multi_positive_pooled_contrastive":
            if live_photos is None or photo_labels is None:
                raise ValueError("multi-positive objective requires live photos and labels")
            rank_key, rank_value = "rank_q_mp", _main_photo_multi_positive(output.q, live_photos, photo_labels, labels)
        elif main_photo_objective == "multi_positive_three_head_contrastive":
            if live_photos is None or photo_labels is None:
                raise ValueError("multi-positive objective requires live photos and labels")
            terms.update(
                rank_i=_main_photo_multi_positive(output.mu_i, live_photos, photo_labels, labels),
                rank_t_mp=_main_photo_multi_positive(output.mu_t, live_photos, photo_labels, labels),
                rank_q_mp=_main_photo_multi_positive(output.q, live_photos, photo_labels, labels),
            )
            rank_key = rank_value = None
        else:
            raise ValueError(f"unknown main_photo_objective: {main_photo_objective}")
        if rank_key is not None:
            terms[rank_key] = rank_value
        terms.update(
            ce_t=_ce(output.mu_t, text, classids, labels),
            align_i=_align(output.mu_i, positive),
            align_t=_align(output.mu_t, text[torch.searchsorted(classids, labels)]),
        )
    if architecture == "predictive_fusion_v2":
        terms["ce_i"] = _ce(output.mu_i, text, classids, labels)
    return terms


def task_loss(
    terms: dict[str, Tensor], architecture: str,
    main_photo_objective: str = "paired_softplus",
    *,
    lambda_mp_t: float = 0.0,
    lambda_mp_q: float = 0.0,
) -> Tensor:
    """Return the two-view main ranking/classification objective."""
    if main_photo_objective == "multi_positive_three_head_contrastive":
        _validate_three_head_coefficients(lambda_mp_t, lambda_mp_q)
    if architecture == "pooled":
        return 0.5 * (
            terms["clean_rank_pool"]
            + terms["clean_ce_pool"]
            + terms["masked_rank_pool"]
            + terms["masked_ce_pool"]
        )
    rank = "rank_q_mp" if main_photo_objective == "multi_positive_pooled_contrastive" else "rank_i"
    ce = "ce_i" if architecture == "predictive_fusion_v2" else "ce_t"
    task = 0.5 * (
        terms[f"clean_{rank}"]
        + terms[f"clean_{ce}"]
        + terms[f"masked_{rank}"]
        + terms[f"masked_{ce}"]
    )
    if main_photo_objective == "multi_positive_three_head_contrastive":
        task = task + 0.5 * (
            lambda_mp_t * (terms["clean_rank_t_mp"] + terms["masked_rank_t_mp"])
            + lambda_mp_q * (terms["clean_rank_q_mp"] + terms["masked_rank_q_mp"])
        )
    return task


def _validate_three_head_coefficients(lambda_mp_t: float, lambda_mp_q: float) -> None:
    for name, value in (("lambda_mp_t", lambda_mp_t), ("lambda_mp_q", lambda_mp_q)):
        try:
            valid = not isinstance(value, bool) and math.isfinite(value) and value >= 0.0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite non-negative scalar")


def _validate_photo_ce_coefficient(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lambda_photo_ce must be a finite non-negative scalar")
    try:
        valid = math.isfinite(value) and value >= 0.0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("lambda_photo_ce must be a finite non-negative scalar")


def _validate_sketch_ref_coefficient(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lambda_sketch_ref must be a finite non-negative scalar")
    try:
        valid = math.isfinite(value) and value >= 0.0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("lambda_sketch_ref must be a finite non-negative scalar")


def _sketch_reference_ce(teacher: Tensor, student: Tensor) -> Tensor:
    """Match masked original-CLIP teacher rows to clean student-q columns."""
    logits = F.normalize(teacher.detach(), dim=-1) @ F.normalize(student, dim=-1).T
    labels = torch.arange(teacher.shape[0], device=teacher.device)
    return F.cross_entropy(logits / _TEMPERATURE, labels)


def coupled_region_loss(
    model: CoupledPredictiveModel,
    clean: Tensor,
    corrupted: Tensor,
    photos: Tensor,
    positive_indices: Tensor,
    negative_indices: Tensor,
    labels: Tensor,
    photo_labels: Tensor,
    photo_ids: object,
    *,
    lambda_sig,
    sigreg=None,
    main_photo_objective: str = "paired_softplus",
    lambda_mp_t: float = 0.0,
    lambda_mp_q: float = 0.0,
    lambda_photo_ce: float | None = None,
    lambda_sketch_ref: float | None = None,
) -> dict[str, Tensor]:
    """Build clean/corrupted loss terms from one shared photo/text bank."""
    del photo_ids
    if lambda_sketch_ref is not None:
        _validate_sketch_ref_coefficient(lambda_sketch_ref)
        if (
            model.architecture != "predictive_fusion_v2"
            or main_photo_objective != "multi_positive_supervised_contrastive"
            or lambda_sig != 0
            or sigreg is not None
            or lambda_photo_ce is not None
        ):
            raise ValueError(
                "sketch reference requires predictive_fusion_v2, supervised multi-positive "
                "photo objective, lambda_sig=0 without SIGReg, and no photo CE"
            )
    if lambda_photo_ce is not None:
        _validate_photo_ce_coefficient(lambda_photo_ce)
        if (
            model.architecture != "predictive_fusion_v2"
            or main_photo_objective != "multi_positive_supervised_contrastive"
            or lambda_sig != 0
            or sigreg is not None
        ):
            raise ValueError(
                "photo CE requires predictive_fusion_v2, supervised multi-positive "
                "photo objective, and lambda_sig=0 without SIGReg"
            )
    if main_photo_objective == "multi_positive_three_head_contrastive":
        _validate_three_head_coefficients(lambda_mp_t, lambda_mp_q)
    outputs = model(torch.cat((clean, corrupted), dim=0))
    batch = clean.shape[0]
    clean_output = CoupledPredictiveOutput(
        outputs.g[:batch], outputs.q[:batch],
        None if outputs.mu_i is None else outputs.mu_i[:batch],
        None if outputs.mu_t is None else outputs.mu_t[:batch],
    )
    masked_output = CoupledPredictiveOutput(
        outputs.g[batch:], outputs.q[batch:],
        None if outputs.mu_i is None else outputs.mu_i[batch:],
        None if outputs.mu_t is None else outputs.mu_t[batch:],
    )

    live_photos = model.encode_photo(photos)
    photo_reference = model.photo_reference(photos)
    text = model.text_bank()
    classids = model.classids
    positive = live_photos[positive_indices]
    negative = live_photos[negative_indices]
    clean_terms = _view_terms(
        clean_output, positive, negative, text, classids, labels, model.architecture,
        live_photos=live_photos, photo_labels=photo_labels,
        main_photo_objective=main_photo_objective,
    )
    masked_terms = _view_terms(
        masked_output, positive, negative, text, classids, labels, model.architecture,
        live_photos=live_photos, photo_labels=photo_labels,
        main_photo_objective=main_photo_objective,
    )

    result = {
        f"clean_{name}": value for name, value in clean_terms.items()
    }
    result.update({f"masked_{name}": value for name, value in masked_terms.items()})
    result["anchor_i"] = _align(live_photos, photo_reference)
    result["anchor_t"] = _align(text, model.T0)

    if lambda_sig:
        sigreg_loss = 0.5 * (sigreg(clean_output.g) + sigreg(masked_output.g))
    else:
        sigreg_loss = clean_output.g.new_zeros(())
    result["sigreg"] = sigreg_loss

    task = task_loss(
        result,
        model.architecture,
        main_photo_objective,
        lambda_mp_t=lambda_mp_t,
        lambda_mp_q=lambda_mp_q,
    )
    if model.architecture in {"predictive", "predictive_fusion_v2"}:
        task = task + 0.125 * (
            result["clean_rank_pool"] + result["clean_ce_pool"]
            + result["masked_rank_pool"] + result["masked_ce_pool"]
        )
        task = task + 0.025 * (
            result["clean_align_i"] + result["clean_align_t"]
            + result["masked_align_i"] + result["masked_align_t"]
        )
    if model.architecture == "predictive_fusion_v2":
        task = task + 0.125 * (result["clean_ce_t"] + result["masked_ce_t"])
    result["total"] = task + 0.5 * result["anchor_i"] + 0.5 * result["anchor_t"] + lambda_sig * sigreg_loss
    if lambda_photo_ce is not None:
        result["photo_ce"] = _ce(live_photos, text, classids, photo_labels)
        if lambda_photo_ce != 0:
            result["total"] = result["total"] + lambda_photo_ce * result["photo_ce"]
    if lambda_sketch_ref is not None:
        # Original CLIP encodes either modality; preserve the old photo batch exactly.
        sketch_reference = model.photo_reference(corrupted)
        result["sketch_ref"] = _sketch_reference_ce(sketch_reference, clean_output.q)
        if lambda_sketch_ref != 0:
            result["total"] = result["total"] + lambda_sketch_ref * result["sketch_ref"]
    return result


__all__ = ["coupled_region_loss", "task_loss"]
