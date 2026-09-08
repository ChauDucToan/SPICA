"""DESIGN ILLUSTRATION ONLY — not a trainer or a certified CLIP implementation.

CPU tensor self-check: PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python <this file>
No dataset, pretrained model, optimizer step, W&B, or real SIGReg is run here.
Shared prompts are passed by reference, not copied into predictor-owned parameters.
"""
from copy import deepcopy
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from spica.models.jepa import jepa_text_classification_loss
from spica.semantic_text import text_anchor_loss


def unit(x: Tensor) -> Tensor:
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("features must be finite floating-point tensors")
    if x.numel() == 0 or (x.norm(dim=-1) == 0).any():
        raise ValueError("features must have nonempty, nonzero rows")
    return F.normalize(x, dim=-1)


# A. Separate trainable student: no shared weight storage with original CLIP.
def make_student_visual(original_clip: nn.Module) -> nn.Module:
    student = deepcopy(original_clip.visual)
    student.requires_grad_(True)
    # These are unused by context_tokens; do not pretend they are being trained.
    student.proj.requires_grad_(False)
    student.ln_post.requires_grad_(False)
    return student


def context_tokens(student: nn.Module, masked_sketch: Tensor) -> Tensor:
    """Proposed OpenCLIP ViT path: raster corruption MUST already be applied.

    Patch tokens retain their original positional information and CLS-mediated
    attention, but the returned array excludes CLS. No text/photo hint enters.
    This private-API adapter still requires real-CLIP validation before integration.
    """
    tokens = student.transformer(student._embeds(masked_sketch))
    return tokens[:, 1:]  # [B, 49, 768] for the proposed 224px ViT-B/32.


# B. One narrow shared cross-attention block, two modality queries.
class SmallPromptPredictor(nn.Module):
    def __init__(self, context_dim=768, text_dim=512, width=256, output_dim=512, heads=4):
        super().__init__()
        self.context_in = nn.Linear(context_dim, width)
        self.photo_in = nn.Linear(context_dim, width)
        self.text_in = nn.Linear(text_dim, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.norm2 = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, h: Tensor, photo_prompt: Tensor, text_context: Tensor):
        if h.ndim != 3 or h.shape[0] == 0 or h.shape[1] == 0:
            raise ValueError("context must be [B>0, N>0, context_dim]")
        if photo_prompt.ndim != 2 or text_context.ndim != 2:
            raise ValueError("shared prompts must be global [tokens, width], not per-query targets")
        if photo_prompt.shape[0] == 0 or text_context.shape[0] == 0:
            raise ValueError("both modality prompts must be nonempty")
        memory = self.context_in(h)
        n_photo = photo_prompt.shape[0]
        queries = torch.cat((self.photo_in(photo_prompt), self.text_in(text_context)), dim=0)
        queries = queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        attended, _ = self.attention(queries, memory, memory, need_weights=False)
        values = self.norm1(queries + attended)
        values = self.norm2(values + self.ffn(values))
        mu_i = unit(self.output(values[:, :n_photo].mean(dim=1)))
        mu_t = unit(self.output(values[:, n_photo:].mean(dim=1)))
        return mu_i, mu_t


# C. Three branches: unnormalized pooled latent, direct head, predictor.
def sketch_branches(h: Tensor, photo_prompt: Tensor, text_context: Tensor,
                    predictor: SmallPromptPredictor, pooled_head: nn.Module):
    g = h.mean(dim=1)  # No L2 normalization or batch standardization before SIGReg.
    q = unit(pooled_head(g))
    mu_i, mu_t = predictor(h, photo_prompt, text_context)
    return {"g": g, "q": q, "mu_i": mu_i, "mu_t": mu_t}


# D. Unlike historical jepa_ranking_loss, this preserves photo-prompt gradients.
def softplus_rank(q: Tensor, positive: Tensor, negative: Tensor, *, margin=0.2):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if q.ndim != 2 or positive.shape != q.shape:
        raise ValueError("query/positive must be matching [B, D]")
    if negative.ndim != 3 or negative.shape[0] != q.shape[0] or negative.shape[2] != q.shape[1]:
        raise ValueError("negatives must be [B, K>0, D]")
    if negative.shape[1] == 0:
        raise ValueError("a labeled rank row needs at least one valid negative")
    q, positive, negative = unit(q), unit(positive), unit(negative)
    s_pos = (q * positive).sum(-1)
    s_neg = (q[:, None] * negative).sum(-1)
    return F.softplus(margin + s_neg - s_pos[:, None]).mean()


# E. Trusted-label illustration only; no implicit pseudo-labeling of missing labels.
def supervised_terms(branches, positive, negative, text, text_class_ids,
                     labels, positive_labels, negative_labels, *, margin=0.2, tau=0.07):
    b, k = negative.shape[:2]
    tensors = (positive, negative, text_class_ids, labels, positive_labels, negative_labels,
               *branches.values())
    if any(value.device != text.device for value in tensors):
        raise ValueError("features and label tensors must be on the same device")
    for ids, shape in ((labels, (b,)), (positive_labels, (b,)),
                       (negative_labels, (b, k)), (text_class_ids, (text.shape[0],))):
        if ids.dtype != torch.long or tuple(ids.shape) != shape:
            raise ValueError("labels must have the expected shape and torch.long dtype")
        if (ids < 0).any():
            raise ValueError("missing labels are not supported by this supervised illustration")
    if not torch.equal(labels, positive_labels) or (negative_labels == labels[:, None]).any():
        raise ValueError("category retrieval requires same-class positives and different-class negatives")
    if not torch.equal(text_class_ids, text_class_ids.sort().values) or text_class_ids.unique().numel() != text_class_ids.numel():
        raise ValueError("text class bank must be sorted and unique")
    text = unit(text)
    # Existing helper validates that all query labels actually exist in the bank.
    ce_t, _ = jepa_text_classification_loss(branches["mu_t"], text, text_class_ids, labels,
                                          temperature=tau, detach_text=False)
    ce_q, _ = jepa_text_classification_loss(branches["q"], text, text_class_ids, labels,
                                          temperature=tau, detach_text=False)
    positions = torch.searchsorted(text_class_ids, labels)
    return {
        "rank_i": softplus_rank(branches["mu_i"], positive, negative, margin=margin),
        "ce_t": ce_t,
        "rank_pool": softplus_rank(branches["q"], positive, negative, margin=margin),
        "ce_pool": ce_q,
        # Fixed-kappa vMF, up to a constant/scale: cosine against detached targets.
        "align_i": text_anchor_loss(branches["mu_i"], positive),
        "align_t": text_anchor_loss(branches["mu_t"], text[positions]),
    }


# F. References are loss-only and must be original, unprompted CLIP outputs.
def preservation_terms(photo_live, photo_reference, text_live, text_reference):
    """Photo rows: unique sampled identities; text rows: complete ordered train bank."""
    return {"anchor_i": text_anchor_loss(photo_live, photo_reference),
            "anchor_t": text_anchor_loss(text_live, text_reference)}


def sigreg_term(g: Tensor, verified_sigreg=None):
    if verified_sigreg is None:
        raise NotImplementedError("Exact SIGReg is a design port, not the repository's standardized SignatureRegularizer")
    if g.ndim != 2 or g.shape[0] < 2 or not torch.isfinite(g).all():
        raise ValueError("SIGReg needs unnormalized [independent sketches >=2, dimension]")
    loss = verified_sigreg(g)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("SIGReg must return a finite scalar")
    return loss


def weighted_total(terms, weights):
    """No default training lambdas: caller must explicitly supply every coefficient."""
    if not terms or set(terms) != set(weights):
        raise ValueError("every loss term needs an explicit weight")
    for name, weight in weights.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid weight for {name}")
        if terms[name].ndim != 0 or not torch.isfinite(terms[name]):
            raise ValueError(f"invalid scalar loss for {name}")
    return sum(weights[name] * loss for name, loss in terms.items())


def region_objective(clean, corrupted, anchors, *, lambda_sig,
                     sigreg_clean=None, sigreg_corrupted=None):
    """Approved group weights, equal views, reference anchors ONCE/update.

    lambda_sig=0 explicitly names the no-SIGReg control. A positive lambda
    requires BOTH real per-view SIGReg values; there is no silent fallback.
    """
    expected = {"rank_i", "ce_t", "rank_pool", "ce_pool", "align_i", "align_t"}
    if set(clean) != expected or set(corrupted) != expected:
        raise ValueError("both views must provide exactly the six supervised losses")
    if set(anchors) != {"anchor_i", "anchor_t"}:
        raise ValueError("provide photo/text reference anchors once, outside view losses")
    weights = {"rank_i": 1.0, "ce_t": 1.0, "rank_pool": 0.25, "ce_pool": 0.25,
               "align_i": 0.05, "align_t": 0.05, "anchor_i": 0.5, "anchor_t": 0.5}
    # Validate each view independently, not merely the averaged values.
    for view in (clean, corrupted):
        weighted_total(view, {name: weights[name] for name in expected})
    terms = {name: 0.5 * (clean[name] + corrupted[name]) for name in expected}
    terms.update(anchors)
    if not math.isfinite(lambda_sig) or lambda_sig < 0:
        raise ValueError("SIGReg coefficient must be finite and nonnegative")
    if lambda_sig > 0:
        if sigreg_clean is None or sigreg_corrupted is None:
            raise ValueError("enabled SIGReg needs both verified per-view losses")
        terms["sigreg"] = weighted_total(
            {"clean": sigreg_clean, "corrupted": sigreg_corrupted},
            {"clean": 0.5, "corrupted": 0.5},
        )
        weights["sigreg"] = lambda_sig
    elif sigreg_clean is not None or sigreg_corrupted is not None:
        raise ValueError("no-SIGReg control must not compute/pass SIGReg losses")
    return weighted_total(terms, weights)


def demo():
    """ONE CPU tensor graph check; not a CLIP/masking/SIGReg or retrieval gate."""
    torch.manual_seed(7)
    h = torch.randn(4, 5, 8, requires_grad=True)
    c_i = nn.Parameter(torch.randn(3, 8))
    c_t = nn.Parameter(torch.randn(4, 6))
    predictor = SmallPromptPredictor(8, 6, 8, 6, 2)
    pooled_head = nn.Linear(8, 6)
    branches = sketch_branches(h, c_i, c_t, predictor, pooled_head)
    assert all(branches[key].shape == (4, 6) for key in ("q", "mu_i", "mu_t"))
    assert all(torch.allclose(branches[key].norm(dim=-1), torch.ones(4), atol=1e-6)
               for key in ("q", "mu_i", "mu_t"))
    assert branches["g"].shape == (4, 8)
    p = torch.randn(4, 6, requires_grad=True)
    n = torch.randn(4, 2, 6, requires_grad=True)
    t = torch.randn(3, 6, requires_grad=True)
    ids, labels = torch.tensor([2, 5, 9]), torch.tensor([2, 5, 9, 2])
    neg_labels = torch.tensor([[5, 9], [2, 9], [2, 5], [5, 9]])
    terms = supervised_terms(branches, p, n, t, ids, labels, labels, neg_labels)
    assert torch.autograd.grad(terms["align_i"], p, allow_unused=True, retain_graph=True)[0] is None
    assert torch.autograd.grad(terms["align_t"], t, allow_unused=True, retain_graph=True)[0] is None
    refs_i, refs_t = torch.randn(4, 6, requires_grad=True), torch.randn(3, 6, requires_grad=True)
    anchors = preservation_terms(p, refs_i, t, refs_t)
    ref_grads = torch.autograd.grad(sum(anchors.values()), (refs_i, refs_t), allow_unused=True, retain_graph=True)
    assert ref_grads == (None, None)
    for name, target in (("rank_i", p), ("rank_i", n), ("ce_t", t)):
        grad = torch.autograd.grad(terms[name], target, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.norm() > 0
    terms.update(anchors)
    # Unit coefficients exercise wiring ONLY; they are not proposed training lambdas.
    weighted_total(terms, dict.fromkeys(terms, 1.0)).backward()
    for value in (h, c_i, c_t, pooled_head.weight, predictor.attention.in_proj_weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.norm() > 0
    bad_negatives = neg_labels.clone()
    bad_negatives[0, 0] = labels[0]
    try:
        supervised_terms(branches, p, n, t, ids, labels, labels, bad_negatives)
    except ValueError:
        pass
    else:
        raise AssertionError("same-class negative was accepted")
    try:
        sigreg_term(branches["g"])
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unimplemented SIGReg silently ran")
    with torch.no_grad():
        shuffled = predictor(h.flip(0), c_i, c_t)[0]
        assert not torch.allclose(shuffled, branches["mu_i"])
    names = ("rank_i", "ce_t", "rank_pool", "ce_pool", "align_i", "align_t")
    clean = {name: torch.tensor(1.0, requires_grad=True) for name in names}
    masked = {name: torch.tensor(1.0, requires_grad=True) for name in names}
    reference = {name: torch.tensor(1.0, requires_grad=True) for name in ("anchor_i", "anchor_t")}
    total = region_objective(clean, masked, reference, lambda_sig=0.0)
    torch.testing.assert_close(total, torch.tensor(3.6))
    total.backward()
    expected = {"rank_i": 0.5, "ce_t": 0.5, "rank_pool": 0.125, "ce_pool": 0.125,
                "align_i": 0.025, "align_t": 0.025}
    for view in (clean, masked):
        for name, coefficient in expected.items():
            torch.testing.assert_close(view[name].grad, torch.tensor(coefficient))
    for loss in reference.values():
        torch.testing.assert_close(loss.grad, torch.tensor(0.5))
    # Scalar arithmetic ONLY: these are not synthetic stand-ins for a SIGReg algorithm.
    with_sig = region_objective(clean, masked, reference, lambda_sig=0.2,
                               sigreg_clean=torch.tensor(2.0), sigreg_corrupted=torch.tensor(4.0))
    torch.testing.assert_close(with_sig, torch.tensor(4.2))
    try:
        region_objective(clean, masked, reference, lambda_sig=0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("positive SIGReg lambda silently omitted its loss")
    print("CPU_TENSOR_ILLUSTRATION_PASS (NO CLIP/SIGReg/mask/training certification)")
    print("proposed_predictor_parameters:", sum(p.numel() for p in SmallPromptPredictor().parameters()))


if __name__ == "__main__":
    demo()
