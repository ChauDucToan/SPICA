import inspect

import pytest
import torch
import open_clip
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg

from spica.evaluation.text_bank import SoftPromptTextBank
from spica.models.clip import FrozenClipEncoder
from spica.models.coupled_predictive import CoupledPredictiveModel


def _fixture() -> tuple[FrozenClipEncoder, object, dict[int, str]]:
    torch.manual_seed(19)
    clip = CLIP(
        embed_dim=6,
        vision_cfg=CLIPVisionCfg(
            layers=1, width=8, head_width=4, patch_size=4, image_size=8
        ),
        text_cfg=CLIPTextCfg(
            context_length=77, vocab_size=49408, width=8, heads=2, layers=1
        ),
    ).float().cpu()
    return FrozenClipEncoder(clip), open_clip.get_tokenizer("ViT-B-32"), {2: "cat", 7: "dog"}


def _model() -> CoupledPredictiveModel:
    encoder, tokenizer, classes = _fixture()
    return CoupledPredictiveModel(encoder, tokenizer, classes)


def test_cpu_shapes_gradients_and_query_api() -> None:
    model = _model()
    assert isinstance(model.text_bank, SoftPromptTextBank)
    assert model.photo_model.sketch_prompt.requires_grad is False
    assert model.photo_model.photo_prompt.shape == (3, 8)
    assert model.text_bank.context.shape == (4, 8)
    sketches = torch.randn(2, 3, 8, 8)
    output = model(sketches)
    assert output.keys() == ("g", "q", "mu_i", "mu_t")
    assert output.g.shape == (2, 8)
    for name in ("q", "mu_i", "mu_t"):
        value = output[name]
        assert value.shape == (2, 6)
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=0)
    output.q.sum().backward()
    assert model.student_visual.transformer.resblocks[0].mlp.c_fc.weight.grad is not None
    assert model.photo_model.photo_prompt.grad is None
    assert model.text_bank.context.grad is None
    model.zero_grad(set_to_none=True)
    output = model(sketches)
    (output.mu_i.sum() + output.mu_t.sum()).backward()
    assert model.photo_model.photo_prompt.grad is not None
    assert model.text_bank.context.grad is not None


def test_photo_reference_is_detached_and_prompted_photo_is_differentiable() -> None:
    model = _model()
    photos = torch.randn(2, 3, 8, 8)
    live = model.encode_photo(photos)
    assert live.requires_grad
    assert model.photo_reference(photos).requires_grad is False
    live.sum().backward()
    assert model.photo_model.photo_prompt.grad is not None
    assert all(parameter.grad is None for parameter in model.original_clip.parameters())


def test_teacher_is_immutable_independently_stored_and_not_in_query_forward(monkeypatch) -> None:
    model = _model()
    teacher_before = {name: value.detach().clone() for name, value in model.original_clip.state_dict().items()}
    student = list(model.student_visual.parameters())
    teacher = list(model.original_clip.visual.parameters())
    assert student and teacher
    assert all(a.data_ptr() != b.data_ptr() for a in student for b in teacher)
    assert all(not parameter.requires_grad for parameter in model.original_clip.parameters())
    assert model.original_clip.training is False
    assert set(name.split(".")[0] for name, _ in model.named_parameters()) >= {
        "original_clip", "student_visual", "predictor", "pooled_head", "photo_model", "text_bank"
    }
    model.train()
    assert model.training and not model.original_clip.training and not model.photo_model.visual.training
    def forbidden(*args, **kwargs):
        raise AssertionError("query forward invoked the original teacher")

    monkeypatch.setattr(model.original_clip.visual, "forward", forbidden)
    monkeypatch.setattr(model.original_clip, "encode_text", forbidden)
    model(torch.randn(2, 3, 8, 8)).mu_i.sum().backward()
    assert all(torch.equal(value, teacher_before[name]) for name, value in model.original_clip.state_dict().items())
    assert all(parameter.grad is None for parameter in model.original_clip.parameters())


def test_state_roundtrip_preserves_output_and_prompt_ownership() -> None:
    model = _model()
    sketches = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
        expected = model(sketches)
    restored = _model()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        actual = restored(sketches)
    for name in expected.keys():
        torch.testing.assert_close(expected[name], actual[name])
    assert restored.photo_model.photo_prompt.data_ptr() != restored.text_bank.context.data_ptr()
    assert restored.T0.requires_grad is False
    assert restored.classids is restored.text_bank.class_labels
    assert restored.initial_text_parity_max_error <= 1e-6
    for key in ("text_bank.class_labels", "text_bank.token_ids"):
        state = model.state_dict()
        state[key] = state[key].clone() + 1
        with pytest.raises(RuntimeError, match="constructed class bank"):
            restored.load_state_dict(state)


def test_optimizer_groups_are_unique_and_only_trainable() -> None:
    model = _model()
    groups = model.optimizer_parameter_groups()
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    assert not any(parameter is next(model.original_clip.parameters()) for group in groups for parameter in group["params"])
    by_name = {group["name"]: group for group in groups}
    assert by_name["student_visual"]["lr"] == pytest.approx(1e-5)
    assert by_name["predictor"]["lr"] == pytest.approx(1e-4)
    assert by_name["pooled_head"]["lr"] == pytest.approx(1e-4)
    named = dict(model.named_parameters())
    for group in groups:
        for name, parameter in zip(group["parameter_names"], group["params"], strict=True):
            assert named[name] is parameter
            if parameter.ndim < 2 or name.endswith("bias"):
                assert group["weight_decay"] == 0
    assert by_name["prompts"]["weight_decay"] == pytest.approx(1e-4)
    assert any(group["weight_decay"] == 0 for group in groups if group["name"].endswith("no_decay"))


def test_query_views_do_not_share_context_and_context_affects_prediction() -> None:
    model = _model().eval()
    sketches = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        together = model(sketches).mu_i
        separate = torch.cat([model(row[None]).mu_i for row in sketches])
        swapped = model(sketches.flip(0)).mu_i
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(swapped, together.flip(0), atol=1e-6, rtol=1e-5)
    assert (together[0] - together[1]).norm() > 1e-6
    # Only an initialization graph check, not evidence against trained shortcuts.


def test_forward_rejects_labels_and_bad_rgb() -> None:
    model = _model()
    assert tuple(inspect.signature(model.forward).parameters) == ("sketches",)
    with pytest.raises(ValueError):
        model(torch.randn(0, 3, 8, 8))
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 8, 8))
    with pytest.raises(ValueError):
        model(torch.full((2, 3, 8, 8), float("nan")))
    with pytest.raises(TypeError):
        model(torch.zeros(2, 3, 8, 8, dtype=torch.int64))
