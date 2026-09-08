"""Loss graph for coupled-predictive V1."""

from __future__ import annotations

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
) -> dict[str, Tensor]:
    terms = {
        "rank_pool": _rank_live(output.q, positive, negative),
        "ce_pool": _ce(output.q, text, classids, labels),
    }
    if architecture == "predictive":
        terms.update(
            rank_i=_rank_live(output.mu_i, positive, negative),
            ce_t=_ce(output.mu_t, text, classids, labels),
            align_i=_align(output.mu_i, positive),
            align_t=_align(output.mu_t, text[torch.searchsorted(classids, labels)]),
        )
    return terms


def task_loss(terms: dict[str, Tensor], architecture: str) -> Tensor:
    """Return the two-view main ranking/classification objective."""
    if architecture == "pooled":
        return 0.5 * (
            terms["clean_rank_pool"]
            + terms["clean_ce_pool"]
            + terms["masked_rank_pool"]
            + terms["masked_ce_pool"]
        )
    return 0.5 * (
        terms["clean_rank_i"]
        + terms["clean_ce_t"]
        + terms["masked_rank_i"]
        + terms["masked_ce_t"]
    )


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
) -> dict[str, Tensor]:
    """Build clean/corrupted V1 loss terms from one shared photo/text bank."""
    del photo_labels, photo_ids
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
        clean_output, positive, negative, text, classids, labels, model.architecture
    )
    masked_terms = _view_terms(
        masked_output, positive, negative, text, classids, labels, model.architecture
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

    task = task_loss(result, model.architecture)
    if model.architecture == "predictive":
        task = task + 0.125 * (
            result["clean_rank_pool"] + result["clean_ce_pool"]
            + result["masked_rank_pool"] + result["masked_ce_pool"]
        )
        task = task + 0.025 * (
            result["clean_align_i"] + result["clean_align_t"]
            + result["masked_align_i"] + result["masked_align_t"]
        )
    result["total"] = task + 0.5 * result["anchor_i"] + 0.5 * result["anchor_t"] + lambda_sig * sigreg_loss
    return result


__all__ = ["coupled_region_loss", "task_loss"]
