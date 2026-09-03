import inspect

import pytest
import torch
import torch.nn.functional as F
from open_clip.model import VisionTransformer
from torch import nn

from spica.models.frozen_prompt import FrozenPromptModel


def _tiny_vit() -> VisionTransformer:
    torch.manual_seed(7)
    return VisionTransformer(
        image_size=8,
        patch_size=4,
        width=8,
        layers=2,
        heads=2,
        mlp_ratio=2,
        output_dim=6,
    )


def test_prompts_are_distinct_and_inserted_after_positioning_and_ln_pre() -> None:
    visual = _tiny_vit()
    positional_before = visual.positional_embedding.detach().clone()
    model = FrozenPromptModel(visual, prompt_length=3)
    images = torch.randn(2, 3, 8, 8)
    expected_base = visual._embeds(images)
    first_block_inputs: list[torch.Tensor] = []
    handle = visual.transformer.resblocks[0].register_forward_pre_hook(
        lambda _module, args: first_block_inputs.append(args[0].detach().clone())
    )

    sketch_output = model(images)
    photo_output = model.encode_photo(images)
    handle.remove()

    assert model.sketch_prompt is not model.photo_prompt
    assert model.sketch_prompt.shape == model.photo_prompt.shape == (3, 8)
    assert first_block_inputs[0].shape == (2, 8, 8)
    assert torch.equal(first_block_inputs[0][:, :1], expected_base[:, :1])
    assert torch.equal(
        first_block_inputs[0][:, 1:4],
        visual.ln_pre(model.sketch_prompt.detach()).unsqueeze(0).expand(2, -1, -1),
    )
    assert torch.equal(first_block_inputs[0][:, 4:], expected_base[:, 1:])
    assert torch.equal(
        first_block_inputs[1][:, 1:4],
        visual.ln_pre(model.photo_prompt.detach()).unsqueeze(0).expand(2, -1, -1),
    )
    assert torch.equal(visual.positional_embedding, positional_before)
    assert sketch_output.shape == photo_output.shape == (2, 6)
    assert torch.allclose(sketch_output.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(photo_output.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_only_prompts_receive_gradients_and_clip_stays_byte_identical() -> None:
    visual = _tiny_vit()
    model = FrozenPromptModel(visual, prompt_length=2)
    clip_before = {
        name: value.detach().clone() for name, value in visual.state_dict().items()
    }
    prompts_before = (
        model.sketch_prompt.detach().clone(),
        model.photo_prompt.detach().clone(),
    )
    optimizer = torch.optim.SGD(
        (value for value in model.parameters() if value.requires_grad), lr=0.2
    )
    images = torch.randn(3, 3, 8, 8)

    loss = (model(images) * torch.randn(3, 6)).sum()
    loss = loss + (model.encode_photo(images) * torch.randn(3, 6)).sum()
    loss.backward()

    assert model.trainable_parameter_names == ("sketch_prompt", "photo_prompt")
    assert model.trainable_parameter_count == 2 * 2 * 8
    assert set(model.clip_parameter_names) == {
        f"visual.{name}" for name, _ in visual.named_parameters()
    }
    assert {
        name for name, value in model.named_parameters() if value.grad is not None
    } == {
        "sketch_prompt",
        "photo_prompt",
    }
    optimizer.step()
    assert not torch.equal(model.sketch_prompt, prompts_before[0])
    assert not torch.equal(model.photo_prompt, prompts_before[1])
    assert all(
        torch.equal(value, clip_before[name])
        for name, value in visual.state_dict().items()
    )


def test_fp_ln_trains_only_prompts_and_all_visual_layernorm_affines() -> None:
    visual = _tiny_vit()
    model = FrozenPromptModel(
        visual,
        prompt_length=1,
        train_visual_layernorm=True,
    )
    layernorm_names = {
        f"visual.{module_name}.{parameter_name}"
        for module_name, module in visual.named_modules()
        if isinstance(module, nn.LayerNorm)
        for parameter_name, _ in module.named_parameters(recurse=False)
    }

    assert set(model.trainable_parameter_names) == {
        "sketch_prompt",
        "photo_prompt",
        *layernorm_names,
    }
    assert all(
        not value.requires_grad
        for name, value in model.named_parameters()
        if name.startswith("visual.") and name not in layernorm_names
    )
    model.train()
    assert model.training
    assert not visual.training
    assert all(not module.training for module in visual.modules())


def test_zero_prompt_matches_vanilla_visual_output() -> None:
    visual = _tiny_vit()
    model = FrozenPromptModel(visual, prompt_length=0)
    images = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        expected = F.normalize(visual(images), dim=-1)
        actual = model(images)

    assert model.sketch_prompt.shape == model.photo_prompt.shape == (0, 8)
    assert model.trainable_parameter_names == ()
    assert torch.equal(actual, expected)


def test_forward_is_sketch_only_and_unsupported_towers_fail_closed() -> None:
    assert tuple(inspect.signature(FrozenPromptModel.forward).parameters) == (
        "self",
        "sketch_images",
    )
    with pytest.raises(TypeError, match="OpenCLIP ViT"):
        FrozenPromptModel(nn.Linear(3, 2), prompt_length=1)
    with pytest.raises(ValueError, match="non-negative"):
        FrozenPromptModel(_tiny_vit(), prompt_length=-1)


def test_attention_diagnostics_match_openclip_attention_weights() -> None:
    visual = _tiny_vit()
    model = FrozenPromptModel(visual, prompt_length=2)
    images = torch.randn(2, 3, 8, 8)
    block = visual.transformer.resblocks[0]
    original_forward = nn.MultiheadAttention.forward

    with torch.no_grad():
        base = visual._embeds(images)
        prompts = (
            visual.ln_pre(model.sketch_prompt)
            .unsqueeze(0)
            .expand(images.shape[0], -1, -1)
        )
        tokens = torch.cat((base[:, :1], prompts, base[:, 1:]), dim=1)
        normalized = block.ln_1(tokens)
        _, weights = block.attn(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        prompt_weights = weights[..., 1:3]
        expected_cls = prompt_weights[..., 0, :].sum(dim=-1).mean().item()
        expected_patch = prompt_weights[..., 3:, :].sum(dim=-1).mean().item()

    diagnostics = model.attention_diagnostics(images)
    assert diagnostics["cls_to_prompt_mass"] == pytest.approx(expected_cls)
    assert diagnostics["patch_to_prompt_mass"] == pytest.approx(expected_patch)
    assert nn.MultiheadAttention.forward is original_forward
