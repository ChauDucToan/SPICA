"""Small, auditable primitives for the semantic-text campaign.

The trainer owns orchestration; this module only defines the fixed-bank identity
and the class-mean anchor so the loss cannot accidentally be scaled by views or
batch frequency.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def clone_fixed_text_bank(values: Tensor, *, device: torch.device | None = None) -> Tensor:
    """Return a normal, detached tensor safe to save for backward as a target.

    ``encode_class_text_bank`` runs under ``inference_mode``.  Cloning here,
    outside that context, is intentional: a detached inference tensor cannot be
    used by every autograd backward implementation as a saved target.
    """
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise ValueError("fixed text bank must have shape [classes, dimension]")
    source = values.detach()
    if source.is_inference():
        result = torch.empty(source.shape, dtype=source.dtype, device=source.device)
        result.copy_(source)
    else:
        result = source.clone()
    if device is not None:
        result = result.to(device)
    if not torch.isfinite(result).all().item():
        raise ValueError("fixed text bank must be finite")
    return result


def text_anchor_loss(learned: Tensor, fixed: Tensor) -> Tensor:
    """Mean ``1-cosine`` over the complete, ordered class bank."""
    if learned.ndim != 2 or fixed.ndim != 2 or learned.shape != fixed.shape:
        raise ValueError("learned and fixed text banks must have identical 2-D shapes")
    if learned.shape[0] == 0:
        raise ValueError("text bank must contain at least one class")
    if not torch.isfinite(learned).all().item() or not torch.isfinite(fixed).all().item():
        raise ValueError("text banks must be finite")
    if (learned.norm(dim=-1) == 0).any().item() or (fixed.norm(dim=-1) == 0).any().item():
        raise ValueError("text bank rows must have non-zero norm")
    # T0 is a target, never a gradient-bearing input, even when a caller passes
    # a tensor that was accidentally created with requires_grad=True.
    return (1.0 - F.cosine_similarity(learned, fixed.detach(), dim=-1)).mean()


def text_drift(learned: Tensor, fixed: Tensor) -> float:
    """Detached scalar form of :func:`text_anchor_loss` for diagnostics."""
    with torch.no_grad():
        return float(text_anchor_loss(learned, fixed).item())


def tensor_sha256(values: Tensor) -> str:
    """Hash tensor bytes plus shape and dtype, independently of device."""
    if not isinstance(values, Tensor):
        raise TypeError("values must be a tensor")
    cpu = values.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(b"\0")
    digest.update(str(cpu.dtype).encode())
    digest.update(b"\0")
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def semantic_text_identity(
    *,
    class_ids: list[int] | tuple[int, ...],
    class_names: list[str] | tuple[str, ...],
    token_ids: Tensor,
    eot_positions: list[int] | tuple[int, ...],
    model_name: str,
    pretrained: str | None,
    fixed_bank: Tensor,
    initial_context: Tensor | None = None,
    initial_learned_bank: Tensor | None = None,
    anchor_formula: str = "mean_c(1-cosine(T_learned[c], stopgrad(T0[c])))",
    lambda_anchor: float = 0.0,
) -> dict[str, Any]:
    """Build checkpoint metadata for immutable T0 and mutable soft context."""
    if len(class_ids) != len(class_names) or len(class_ids) != fixed_bank.shape[0]:
        raise ValueError("semantic text classes and fixed bank are misaligned")
    if tuple(class_ids) != tuple(sorted(class_ids)):
        raise ValueError("semantic text class IDs must be sorted")
    return {
        "schema_version": 1,
        "class_ids": [int(value) for value in class_ids],
        "class_names": [str(value) for value in class_names],
        "class_order_sha256": hashlib.sha256(
            json.dumps(
                [[int(cid), str(name)] for cid, name in zip(class_ids, class_names, strict=True)],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "token_ids": token_ids.detach().cpu().tolist(),
        "token_ids_sha256": tensor_sha256(token_ids),
        "eot_positions": [int(value) for value in eot_positions],
        "model_name": str(model_name),
        "pretrained": pretrained,
        "fixed_bank_sha256": tensor_sha256(fixed_bank),
        "initial_context_sha256": None if initial_context is None else tensor_sha256(initial_context),
        "initial_learned_bank_sha256": None if initial_learned_bank is None else tensor_sha256(initial_learned_bank),
        "anchor_formula": anchor_formula,
        "anchor_reduction": "mean_over_all_train_classes_once_per_update",
        "lambda_anchor": float(lambda_anchor),
    }
