"""Loss graph for the approved coupled-predictive region objective.

This module deliberately contains no trainer, sampler, or configuration layer.
It wires one update: two sketch views, one live photo bank, one live text bank,
and detached original-CLIP references.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor

from .models.coupled_predictive import CoupledPredictiveModel, CoupledPredictiveOutput
from .models.jepa import jepa_text_classification_loss
from .semantic_text import text_anchor_loss

_MARGIN = 0.2
_TEMPERATURE = 0.07


def _finite_scalar(value: object, *, name: str, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must return a scalar tensor")
    if value.ndim != 0 or not value.is_floating_point():
        raise ValueError(f"{name} must return a finite floating-point scalar")
    if value.device != device or not torch.isfinite(value).item():
        raise ValueError(f"{name} must return a finite scalar on the query device")
    return value


def _rank_live(query: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
    """Softplus category ranking, retaining gradients through live photos."""
    if query.ndim != 2 or positive.shape != query.shape:
        raise ValueError("query and positive must have shape [B, D]")
    if negative.ndim != 3 or negative.shape[0] != query.shape[0] or negative.shape[2] != query.shape[1]:
        raise ValueError("negative must have shape [B, K, D]")
    if negative.shape[1] == 0:
        raise ValueError("each query needs at least one negative photo")
    for name, value in (("query", query), ("positive", positive), ("negative", negative)):
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must be finite floating-point values")
        if value.device != query.device or value.dtype != query.dtype:
            raise ValueError("ranking embeddings must share dtype and device")
        if (value.norm(dim=-1) == 0).any().item():
            raise ValueError(f"{name} must have nonzero rows")
    q = F.normalize(query, dim=-1)
    pos = F.normalize(positive, dim=-1)
    neg = F.normalize(negative, dim=-1)
    positive_score = (q * pos).sum(dim=-1)
    negative_score = (q[:, None] * neg).sum(dim=-1)
    return F.softplus(_MARGIN + negative_score - positive_score[:, None]).mean()


def _check_images(clean: Tensor, corrupted: Tensor, photos: Tensor, device: torch.device, dtype: torch.dtype) -> None:
    values = (("clean", clean), ("corrupted", corrupted), ("photos", photos))
    for name, value in values:
        if not isinstance(value, Tensor) or value.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, 3, height, width]")
        if value.shape[1] != 3 or any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must have a nonempty RGB batch")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating-point")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
        if value.device != device or value.dtype != dtype:
            raise ValueError(f"{name} must use the model device and common image dtype")
    if clean.shape != corrupted.shape:
        raise ValueError("clean and corrupted views must have the same shape")


def _check_ids(photo_ids: object, count: int) -> None:
    if isinstance(photo_ids, Tensor):
        if photo_ids.ndim != 1 or photo_ids.shape[0] != count:
            raise ValueError("photo_ids must contain one ID per photo")
        if photo_ids.is_floating_point() or photo_ids.is_complex() or photo_ids.dtype == torch.bool:
            raise TypeError("tensor photo_ids must be integer IDs")
        values = photo_ids.detach().cpu().tolist()
    else:
        if isinstance(photo_ids, (str, bytes)) or not isinstance(photo_ids, Sequence):
            raise TypeError("photo_ids must be a sequence of unique IDs")
        values = list(photo_ids)
        if len(values) != count:
            raise ValueError("photo_ids must contain one ID per photo")
    try:
        unique = len(set(values))
    except TypeError as error:
        raise TypeError("photo_ids must contain hashable IDs") from error
    if unique != count:
        raise ValueError("photo_ids must be unique: encode each photo identity once")


def _check_labels(
    model: object,
    clean: Tensor,
    photos: Tensor,
    positive_indices: Tensor,
    negative_indices: Tensor,
    labels: Tensor,
    photo_labels: Tensor,
) -> None:
    b, n = clean.shape[0], photos.shape[0]
    expected_device = clean.device
    for name, value, shape in (
        ("positive_indices", positive_indices, (b,)),
        ("negative_indices", negative_indices, (b, -1)),
        ("labels", labels, (b,)),
        ("photo_labels", photo_labels, (n,)),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != torch.long:
            raise TypeError(f"{name} must have dtype torch.long")
        if value.device != expected_device:
            raise ValueError(f"{name} must be on the image device")
        if name == "negative_indices":
            if value.ndim != 2 or value.shape[0] != b or value.shape[1] == 0:
                raise ValueError("negative_indices must have shape [B, K>0]")
        elif tuple(value.shape) != shape:
            raise ValueError(f"{name} has the wrong shape")
    for name, value, upper in (
        ("positive_indices", positive_indices, n),
        ("negative_indices", negative_indices, n),
    ):
        if (value < 0).any().item() or (value >= upper).any().item():
            raise ValueError(f"{name} contains an out-of-range photo index")

    classids = getattr(model, "classids", None)
    if not isinstance(classids, Tensor) or classids.ndim != 1 or classids.dtype != torch.long:
        raise RuntimeError("model.classids must be a one-dimensional long tensor")
    if classids.device != expected_device:
        raise ValueError("model.classids must be on the image device")
    if classids.numel() == 0 or not torch.equal(classids, classids.sort().values):
        raise RuntimeError("model.classids must be sorted and nonempty")
    if torch.unique(classids).numel() != classids.numel():
        raise RuntimeError("model.classids must be unique")
    def known(value: Tensor) -> bool:
        return torch.isin(value, classids).all().item()

    if not known(labels) or not known(photo_labels):
        raise ValueError("labels must all be present in model.classids")
    positive_labels = photo_labels[positive_indices]
    negative_labels = photo_labels[negative_indices]
    if not torch.equal(positive_labels, labels):
        raise ValueError("each positive photo must have the query category")
    if (negative_labels == labels[:, None]).any().item():
        raise ValueError("negative photos must have a different category, not merely a different instance")


def _view_terms(
    output: CoupledPredictiveOutput,
    positive: Tensor,
    negative: Tensor,
    text: Tensor,
    classids: Tensor,
    labels: Tensor,
) -> dict[str, Tensor]:
    ce_t, _ = jepa_text_classification_loss(
        output.mu_t, text, classids, labels, temperature=_TEMPERATURE, detach_text=False
    )
    ce_pool, _ = jepa_text_classification_loss(
        output.q, text, classids, labels, temperature=_TEMPERATURE, detach_text=False
    )
    positions = torch.searchsorted(classids, labels)
    return {
        "rank_i": _rank_live(output.mu_i, positive, negative),
        "ce_t": ce_t,
        "rank_pool": _rank_live(output.q, positive, negative),
        "ce_pool": ce_pool,
        "align_i": text_anchor_loss(output.mu_i, positive),
        "align_t": text_anchor_loss(output.mu_t, text[positions]),
    }


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
    lambda_sig: Real,
    sigreg=None,
) -> dict[str, Tensor]:
    """Build the complete two-view loss graph for one update.

    The two query views are concatenated for exactly one model forward.  Live
    photo/text targets are each encoded once and reused by both views. Positive
    alignment targets and original-CLIP references are detached; rank/CE banks
    remain live.  ``lambda_sig=0`` is a real control:
    it does not call ``sigreg``.
    """
    if not isinstance(model, CoupledPredictiveModel):
        raise TypeError("model must be a CoupledPredictiveModel")
    device = next(model.parameters(), None)
    if device is None:
        raise ValueError("model must have parameters")
    model_device = device.device
    model_dtype = device.dtype
    if not isinstance(lambda_sig, Real) or isinstance(lambda_sig, bool):
        raise TypeError("lambda_sig must be a finite nonnegative real number")
    if not math.isfinite(float(lambda_sig)) or float(lambda_sig) < 0:
        raise ValueError("lambda_sig must be finite and nonnegative")
    _check_images(clean, corrupted, photos, model_device, model_dtype)
    _check_labels(model, clean, photos, positive_indices, negative_indices, labels, photo_labels)
    _check_ids(photo_ids, photos.shape[0])
    if float(lambda_sig) > 0:
        if not callable(sigreg):
            raise ValueError("positive lambda_sig requires a callable SIGReg")
        if clean.shape[0] < 2:
            raise ValueError("SIGReg requires at least two independent query rows")

    outputs = model(torch.cat((clean, corrupted), dim=0))
    if not isinstance(outputs, CoupledPredictiveOutput):
        raise TypeError("model forward must return CoupledPredictiveOutput")
    b = clean.shape[0]
    clean_output = CoupledPredictiveOutput(
        outputs.g[:b], outputs.q[:b], outputs.mu_i[:b], outputs.mu_t[:b]
    )
    masked_output = CoupledPredictiveOutput(
        outputs.g[b:], outputs.q[b:], outputs.mu_i[b:], outputs.mu_t[b:]
    )

    live_photos = model.encode_photo(photos)
    photo_reference = model.photo_reference(photos)
    text = model.text_bank()
    fixed_text = model.fixed_text_bank
    classids = model.classids
    positive = live_photos[positive_indices]
    negative = live_photos[negative_indices]
    clean_terms = _view_terms(clean_output, positive, negative, text, classids, labels)
    masked_terms = _view_terms(masked_output, positive, negative, text, classids, labels)

    result: dict[str, Tensor] = {}
    for prefix, terms in (("clean", clean_terms), ("masked", masked_terms)):
        result.update({f"{prefix}_{name}": value for name, value in terms.items()})
    result["anchor_i"] = text_anchor_loss(live_photos, photo_reference)
    result["anchor_t"] = text_anchor_loss(text, fixed_text)

    if float(lambda_sig) == 0:
        sigreg_loss = clean_output.g.new_zeros(())
    else:
        clean_sig = _finite_scalar(sigreg(clean_output.g), name="sigreg(clean)", device=clean_output.g.device)
        masked_sig = _finite_scalar(sigreg(masked_output.g), name="sigreg(masked)", device=masked_output.g.device)
        sigreg_loss = 0.5 * (clean_sig + masked_sig)
    result["sigreg"] = sigreg_loss

    clean_weighted = (
        clean_terms["rank_i"] + clean_terms["ce_t"]
        + 0.25 * (clean_terms["rank_pool"] + clean_terms["ce_pool"])
        + 0.05 * (clean_terms["align_i"] + clean_terms["align_t"])
    )
    masked_weighted = (
        masked_terms["rank_i"] + masked_terms["ce_t"]
        + 0.25 * (masked_terms["rank_pool"] + masked_terms["ce_pool"])
        + 0.05 * (masked_terms["align_i"] + masked_terms["align_t"])
    )
    result["total"] = (
        0.5 * (clean_weighted + masked_weighted)
        + 0.5 * result["anchor_i"]
        + 0.5 * result["anchor_t"]
        + float(lambda_sig) * result["sigreg"]
    )
    return result


__all__ = ["coupled_region_loss"]
