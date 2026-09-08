import pytest
import torch

from spica.coupled_predictive_losses import coupled_region_loss
from spica.models.sigreg import SIGReg

from test_coupled_predictive_model import _model


def _batch():
    clean = torch.randn(2, 3, 8, 8)
    corrupted = torch.randn(2, 3, 8, 8)
    photos = torch.randn(3, 3, 8, 8)
    positive = torch.tensor([0, 1], dtype=torch.long)
    negative = torch.tensor([[1], [0]], dtype=torch.long)
    labels = torch.tensor([2, 7], dtype=torch.long)
    photo_labels = torch.tensor([2, 7, 2], dtype=torch.long)
    return clean, corrupted, photos, positive, negative, labels, photo_labels


def _loss(model, *, lambda_sig=0.0, sigreg=None):
    return coupled_region_loss(
        model,
        *_batch(),
        ["photo-a", "photo-b", "photo-c"],
        lambda_sig=lambda_sig,
        sigreg=sigreg,
    )


def test_actual_tiny_openclip_single_forward_reuse_and_gradients():
    model = _model()
    clean, corrupted, photos, positive, negative, labels, photo_labels = _batch()
    before = {name: value.detach().clone() for name, value in model.original_clip.state_dict().items()}
    forward_calls = 0
    encode_calls = 0
    reference_calls = 0
    text_calls = 0
    original_forward = model.forward
    original_encode = model.encode_photo
    original_reference = model.photo_reference
    original_text = model.text_bank.forward

    def forward(value):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(value)

    def encode(value):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(value)

    def reference(value):
        nonlocal reference_calls
        reference_calls += 1
        return original_reference(value)

    def text():
        nonlocal text_calls
        text_calls += 1
        return original_text()

    model.forward = forward
    model.encode_photo = encode
    model.photo_reference = reference
    model.text_bank.forward = text
    result = coupled_region_loss(
        model,
        clean,
        corrupted,
        photos,
        positive,
        negative,
        labels,
        photo_labels,
        ["photo-a", "photo-b", "photo-c"],
        lambda_sig=0.0,
    )
    assert forward_calls == encode_calls == reference_calls == text_calls == 1
    assert set(result) == {
        "total", "sigreg", "anchor_i", "anchor_t",
        *(f"{view}_{name}" for view in ("clean", "masked") for name in (
            "rank_i", "ce_t", "rank_pool", "ce_pool", "align_i", "align_t"
        )),
    }
    assert result["total"].ndim == 0 and torch.isfinite(result["total"])
    result["total"].backward()
    for parameter in (
        model.photo_model.photo_prompt, model.text_bank.context,
        model.student_visual.transformer.resblocks[0].mlp.c_fc.weight,
        model.predictor.output.weight, model.pooled_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    assert all(parameter.grad is None for parameter in model.original_clip.parameters())
    for name, value in model.original_clip.state_dict().items():
        assert torch.equal(value, before[name])


def test_exact_view_and_reference_coefficients_and_sigreg_gradients():
    # A scalar wiring check makes the approved 0.5/0.125/0.025/0.5 values
    # observable without replacing the integrated model graph.
    model = _model()
    result = _loss(model)
    terms = [
        result[f"{view}_{name}"]
        for view in ("clean", "masked")
        for name in ("rank_i", "ce_t", "rank_pool", "ce_pool", "align_i", "align_t")
    ]
    total = result["total"]
    expected = (
        0.5 * (result["clean_rank_i"] + result["masked_rank_i"])
        + 0.5 * (result["clean_ce_t"] + result["masked_ce_t"])
        + 0.125 * (result["clean_rank_pool"] + result["masked_rank_pool"])
        + 0.125 * (result["clean_ce_pool"] + result["masked_ce_pool"])
        + 0.025 * (result["clean_align_i"] + result["masked_align_i"])
        + 0.025 * (result["clean_align_t"] + result["masked_align_t"])
        + 0.5 * result["anchor_i"]
        + 0.5 * result["anchor_t"]
    )
    torch.testing.assert_close(total, expected)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in terms)

    model = _model()
    calls = []
    regularizer = SIGReg(knots=5, num_projections=4, seed=3)
    original_regularizer = regularizer.forward

    def spy(value):
        calls.append(value)
        return original_regularizer(value)

    regularizer.forward = spy
    result = _loss(model, lambda_sig=0.2, sigreg=regularizer)
    assert len(calls) == 2
    assert all(value.ndim == 2 and value.shape[0] == 2 for value in calls)
    assert result["sigreg"].ndim == 0 and torch.isfinite(result["sigreg"])
    result["total"].backward()
    assert model.student_visual.transformer.resblocks[0].mlp.c_fc.weight.grad is not None
    assert model.student_visual.transformer.resblocks[0].mlp.c_fc.weight.grad.norm() > 0


def test_literal_coefficients_and_detached_positive_targets(monkeypatch):
    import spica.coupled_predictive_losses as losses

    leaves = [torch.tensor(1.0, requires_grad=True) for _ in range(14)]
    names = ("rank_i", "ce_t", "rank_pool", "ce_pool", "align_i", "align_t")
    views = iter((dict(zip(names, leaves[:6])), dict(zip(names, leaves[6:12]))))
    anchors = iter(leaves[12:])
    monkeypatch.setattr(losses, "_view_terms", lambda *args: next(views))
    monkeypatch.setattr(losses, "text_anchor_loss", lambda *args: next(anchors))
    total = _loss(_model())["total"]
    assert total.item() == pytest.approx(3.6)
    gradients = torch.autograd.grad(total, leaves)
    assert [value.item() for value in gradients] == pytest.approx(
        [0.5, 0.5, 0.125, 0.125, 0.025, 0.025] * 2 + [0.5, 0.5]
    )


def test_live_photo_rank_and_alignment_target_gradient_edges():
    from spica.coupled_predictive_losses import _rank_live
    from spica.semantic_text import text_anchor_loss

    query = torch.randn(2, 6, requires_grad=True)
    positive = torch.randn(2, 6, requires_grad=True)
    negative = torch.randn(2, 3, 6, requires_grad=True)
    grads = torch.autograd.grad(_rank_live(query, positive, negative), (query, positive, negative))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    q_grad, p_grad = torch.autograd.grad(text_anchor_loss(query, positive), (query, positive), allow_unused=True)
    assert q_grad.norm() > 0 and p_grad is None
    with pytest.raises(ValueError, match="nonzero"):
        _rank_live(query, torch.zeros_like(positive), negative)


def test_control_skips_sigreg_and_validation_is_before_forward():
    model = _model()
    spy_calls = []

    def spy(value):
        spy_calls.append(value)
        return value.sum()

    result = _loss(model, lambda_sig=0.0, sigreg=spy)
    assert not spy_calls
    assert result["sigreg"].item() == 0

    clean, corrupted, photos, positive, negative, labels, photo_labels = _batch()
    with pytest.raises(ValueError, match="out-of-range"):
        coupled_region_loss(
            model, clean, corrupted, photos,
            torch.tensor([0, 3]), negative, labels, photo_labels,
            ["a", "b", "c"], lambda_sig=0,
        )
    clean[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        coupled_region_loss(
            model, clean, corrupted, photos, positive, negative, labels, photo_labels,
            ["a", "b", "c"], lambda_sig=0,
        )
    clean, corrupted, photos, positive, negative, labels, photo_labels = _batch()
    labels[0] = 999
    with pytest.raises(ValueError, match="classids"):
        coupled_region_loss(
            model, clean, corrupted, photos, positive, negative, labels, photo_labels,
            ["a", "b", "c"], lambda_sig=0,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached the model")

    model.forward = forbidden
    for bad_lambda in (-1, float("nan"), float("inf"), True):
        with pytest.raises((ValueError, TypeError), match="lambda_sig"):
            _loss(model, lambda_sig=bad_lambda)
    with pytest.raises(ValueError, match="callable"):
        _loss(model, lambda_sig=0.1)
    batch = _batch()
    with pytest.raises(ValueError, match="unique"):
        coupled_region_loss(model, *batch, ["a", "a", "b"], lambda_sig=0)
    batch[4][0, 0] = 2  # Different photo instance, SAME query category.
    with pytest.raises(ValueError, match="different category"):
        coupled_region_loss(model, *batch, ["a", "b", "c"], lambda_sig=0)
    assert all(not parameter.requires_grad for parameter in model.original_clip.parameters())
