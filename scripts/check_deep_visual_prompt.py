"""Standalone CPU check for deep visual prompts plus exact-white key masking."""

import importlib.util
import io
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ruff: noqa: E402
import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch import nn

from spica.models.sketch_teacher import SketchTeacher, cosine_distillation

ATOL = 4e-5
RTOL = 4e-5
MEAN = (0.48, 0.46, 0.42)
STD = (0.27, 0.26, 0.25)


def trainable(teacher):
    return [(name, parameter) for name, parameter in teacher.named_parameters()
            if parameter.requires_grad]


def adapter_parameters(teacher):
    return [(name, parameter) for name, parameter in trainable(teacher)
            if ".parametrizations.in_proj_weight.0." in name]


def assert_finite(values):
    assert all(value is not None and torch.isfinite(value).all() for value in values)


def assert_nested_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            assert_nested_equal(left[key], right[key])
        elif torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def retrieval_loss(queries, photos, labels, photo_labels):
    positives = photo_labels[None, :] == labels[:, None]
    return -(queries @ photos.T).log_softmax(dim=1)[positives].view(labels.numel(), -1).mean()


def archived_teacher_class():
    path = (
        ROOT
        / "outputs/student_whitepatch_teacher_10epochs_20260928/source/src/spica/models/sketch_teacher.py"
    )
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("archived_whitepatch_teacher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SketchTeacher


def finalize(teacher, tokens):
    pooled, _ = teacher.visual._pool(tokens)
    if teacher.visual.proj is not None:
        pooled = pooled @ teacher.visual.proj
    return F.normalize(pooled, dim=-1)


def capture_layers(teacher, sketches):
    outputs = [None] * len(teacher.visual.transformer.resblocks)
    handles = []
    for index, block in enumerate(teacher.visual.transformer.resblocks):
        handles.append(
            block.register_forward_hook(
                lambda _module, _args, output, index=index: outputs.__setitem__(
                    index, output.detach().clone()
                )
            )
        )
    try:
        with torch.no_grad():
            result = teacher(sketches)
    finally:
        for handle in handles:
            handle.remove()
    return result, outputs


def explicit_deep_stack(teacher, sketches, native_blocked):
    """Stack blocks one sample at a time, trimming blocked native keys each layer."""
    results = []
    layer_outputs = [[] for _ in teacher.visual.transformer.resblocks]
    for index in range(sketches.shape[0]):
        normalized = (sketches[index : index + 1] - teacher.mean) / teacher.std
        tokens = teacher.visual._embeds(normalized)
        tokens = tokens[:, ~native_blocked[index]]
        for layer, (block, prompt) in enumerate(
            zip(teacher.visual.transformer.resblocks, teacher.deep_prompts)
        ):
            learned = teacher.visual.ln_pre(prompt.tokens).to(tokens.dtype)
            augmented = torch.cat(
                (
                    tokens[:, :1],
                    learned.unsqueeze(0),
                    tokens[:, 1:],
                ),
                dim=1,
            )
            augmented = block(augmented)
            layer_outputs[layer].append(augmented.detach())
            tokens = torch.cat((augmented[:, :1], augmented[:, 5:]), dim=1)
        results.append(finalize(teacher, tokens))
    return torch.cat(results), layer_outputs


def expected_deep_mask(native_blocked, dtype, heads):
    blocked = torch.cat(
        (
            native_blocked[:, :1],
            torch.zeros(native_blocked.shape[0], 4, dtype=torch.bool),
            native_blocked[:, 1:],
        ),
        dim=1,
    )
    batch, length = blocked.shape
    mask = torch.zeros(batch, heads, length, length, dtype=dtype)
    mask.masked_fill_(blocked[:, None, None, :], float("-inf"))
    return blocked, mask.reshape(batch * heads, length, length)


def validate_deep_flags(flags):
    """Check exact deep experiment flags before data loading or device use."""
    if not flags.lora_teacher or not flags.visual_prompt or not flags.mask_white_patches:
        raise ValueError("Deep prompts require LoRA teacher, visual prompts and white mask")
    if not flags.deep_visual_prompt:
        raise ValueError("Deep visual prompts are required for this experiment")
    if flags.lambda_kd != 1.2:
        raise ValueError("White-patch experiment requires --visual-prompt --lambda-kd 1.2")


def check_train_boundary():
    """Reject incomplete deep experiment flags before data loading or device use."""
    required = {
        "lora_teacher": True,
        "visual_prompt": True,
        "mask_white_patches": True,
        "deep_visual_prompt": True,
        "lambda_kd": 1.2,
    }
    with tempfile.TemporaryDirectory() as directory:
        for missing in required:
            flags = SimpleNamespace(**required)
            setattr(flags, missing, False if missing != "lambda_kd" else None)
            flags.output = Path(directory) / f"missing_{missing}"
            try:
                validate_deep_flags(flags)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted incomplete deep flags: {missing}")
            assert not flags.output.exists()

        # Complete flags pass this boundary; no data config is needed for this check.
        validate_deep_flags(SimpleNamespace(**required))


def main():
    check_train_boundary()

    torch.set_num_threads(1)
    torch.manual_seed(17)
    visual = VisionTransformer(
        image_size=64,
        patch_size=32,
        width=32,
        layers=12,
        heads=4,
        mlp_ratio=2,
        output_dim=16,
    ).cpu().eval()
    raw = torch.rand(4, 3, 64, 64) * 0.8 + 0.1

    # Existing non-deep white-mask output remains byte-identical to archived source.
    archived = archived_teacher_class()
    old = archived(visual, MEAN, STD, visual_prompt=True, mask_white_patches=True).cpu().eval()
    current = SketchTeacher(
        visual, MEAN, STD, visual_prompt=True, mask_white_patches=True
    ).cpu().eval()
    assert old.state_dict().keys() == current.state_dict().keys()
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key], current.state_dict()[key]), key
    with torch.no_grad():
        assert torch.equal(old(raw), current(raw))

    teacher = SketchTeacher(
        visual,
        MEAN,
        STD,
        visual_prompt=True,
        mask_white_patches=True,
        deep_visual_prompt=True,
    ).cpu().eval()
    assert len(teacher.deep_prompts) == 12
    assert len({id(prompt.tokens) for prompt in teacher.deep_prompts}) == 12
    prompt_parameters = [
        parameter for name, parameter in trainable(teacher) if name.endswith(".tokens")
    ]
    adapters = adapter_parameters(teacher)
    assert len(prompt_parameters) == 12
    assert sum(parameter.numel() for parameter in prompt_parameters) == 12 * 4 * 32
    assert len(adapters) == 12 * 4
    assert sum(parameter.numel() for _, parameter in trainable(teacher)) == (
        12 * 4 * 32 + sum(parameter.numel() for _, parameter in adapters)
    )
    assert all(parameter.device.type == "cpu" for _, parameter in trainable(teacher))

    # Exact RGB patch masks: native CLS and four inserted prompts are always valid.
    batch = torch.full((4, 3, 64, 64), 0.2)
    batch[0, :, :32, :32] = 1
    batch[1] = 1
    batch[1, 0, 0, 0] = 0.5
    batch[2, :, :32, :32] = 1
    batch[2, :, 32:, 32:] = 1
    expected_native = torch.tensor(
        [
            [False, True, False, False, False],
            [False, False, True, True, True],
            [False, True, False, False, True],
            [False, False, False, False, False],
        ]
    )
    native_blocked = teacher.white_patch_mask(batch)
    assert torch.equal(native_blocked, expected_native)
    deep_blocked, expected_mask = expected_deep_mask(native_blocked, torch.float32, heads=4)
    assert not deep_blocked[:, :5].any()

    # Every block sees nine tokens (CLS, four prompts, four original patch positions)
    # and the same additive key mask.
    seen_lengths = []
    seen_masks = []
    handles = []
    def observe(_module, args, kwargs):
        seen_lengths.append(tuple(args[0].shape))
        seen_masks.append(kwargs["attn_mask"].detach().clone())

    for block in teacher.visual.transformer.resblocks:
        handles.append(block.register_forward_pre_hook(observe, with_kwargs=True))
    with torch.no_grad():
        masked_output = teacher(batch)
    for handle in handles:
        handle.remove()
    assert seen_lengths == [(4, 9, 32)] * 12
    assert len(seen_masks) == 12
    for seen in seen_masks:
        assert torch.equal(seen, expected_mask)
        assert torch.equal(seen[expected_mask == 0], torch.zeros_like(seen[expected_mask == 0]))
        assert torch.isneginf(seen[expected_mask == float("-inf")]).all()
    assert torch.isfinite(masked_output).all()

    # Dense masked blocks equal explicitly stacked blocks with blank native tokens trimmed.
    _, full_layers = capture_layers(teacher, batch)
    trimmed_output, trimmed_layers = explicit_deep_stack(teacher, batch, native_blocked)
    assert torch.allclose(masked_output, trimmed_output, atol=ATOL, rtol=RTOL)
    for layer, (full, trimmed_samples) in enumerate(zip(full_layers, trimmed_layers)):
        for index, trimmed in enumerate(trimmed_samples):
            valid_full = ~torch.cat(
                (native_blocked[index, :1], torch.zeros(4, dtype=torch.bool),
                 native_blocked[index, 1:]), dim=0
            )
            assert torch.allclose(
                full[index : index + 1, valid_full], trimmed, atol=ATOL, rtol=RTOL
            ), layer

    # No blank patches: masked path agrees with explicit no-mask manual stacking.
    no_blank = torch.rand(4, 3, 64, 64) * 0.8 + 0.1
    no_blank_native = teacher.white_patch_mask(no_blank)
    assert not no_blank_native.any()
    with torch.no_grad():
        masked_no_blank = teacher(no_blank)
    explicit_no_mask, _ = explicit_deep_stack(teacher, no_blank, no_blank_native)
    assert torch.allclose(masked_no_blank, explicit_no_mask, atol=ATOL, rtol=RTOL)
    # Changing a later prompt cannot alter any earlier block output: prompts are replaced,
    # not propagated through native token state.
    baseline_output, baseline_layers = capture_layers(teacher, no_blank[:1])
    saved_prompt = teacher.deep_prompts[7].tokens.detach().clone()
    with torch.no_grad():
        teacher.deep_prompts[7].tokens.add_(0.7)
    changed_output, changed_layers = capture_layers(teacher, no_blank[:1])
    with torch.no_grad():
        teacher.deep_prompts[7].tokens.copy_(saved_prompt)
    assert torch.equal(baseline_layers[6], changed_layers[6])
    assert baseline_output.shape == changed_output.shape == (1, 16)

    # Native CLS pooling and projection are used after final prompt discard.
    with torch.no_grad():
        _, final_layers = capture_layers(teacher, no_blank[:1])
        final_native = torch.cat((final_layers[-1][:, :1], final_layers[-1][:, 5:]), dim=1)
        assert torch.equal(teacher(no_blank[:1]), finalize(teacher, final_native))

    # All-white input keeps valid CLS/prompts and has finite backward values.
    all_white = torch.ones(2, 3, 64, 64)
    all_white_blocked = teacher.white_patch_mask(all_white)
    assert not all_white_blocked[:, :1].any()
    assert all_white_blocked[:, 1:].all()
    teacher.train()
    teacher.zero_grad(set_to_none=True)
    white_output = teacher(all_white)
    assert torch.isfinite(white_output).all()
    white_output[:, 0].sum().backward()
    assert_finite([parameter.grad for _, parameter in trainable(teacher)])

    # KD target is detached; two updates train all independent prompts and LoRA only.
    student = nn.Linear(16, 16, bias=False)
    student_input = torch.randn(4, 16)
    photos = F.normalize(torch.randn(4, 16), dim=-1)
    labels = torch.tensor([0, 1, 0, 1])
    photo_labels = torch.tensor([0, 0, 1, 1])
    teacher_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable(teacher)], lr=1e-2, weight_decay=1e-2
    )
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2, weight_decay=1e-2)

    teacher.zero_grad(set_to_none=True)
    student.zero_grad(set_to_none=True)
    detached = cosine_distillation(
        F.normalize(student(student_input), dim=-1), teacher(batch)
    )
    detached.backward()
    assert all(parameter.grad is None for _, parameter in trainable(teacher))

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in teacher.named_parameters()
        if not parameter.requires_grad
    }
    losses = []
    prompt_gradient_sums = []
    for _ in range(2):
        teacher_optimizer.zero_grad(set_to_none=True)
        student_optimizer.zero_grad(set_to_none=True)
        teacher_queries = teacher(batch)
        student_queries = F.normalize(student(student_input), dim=-1)
        loss = (
            retrieval_loss(teacher_queries, photos, labels, photo_labels)
            + retrieval_loss(student_queries, photos, labels, photo_labels)
            + 1.2 * cosine_distillation(student_queries, teacher_queries)
        )
        loss.backward()
        prompt_grads = [parameter.grad for parameter in prompt_parameters]
        assert_finite(prompt_grads)
        prompt_sum = sum(gradient.abs().sum().item() for gradient in prompt_grads)
        assert prompt_sum > 0
        assert all(gradient.abs().sum().item() > 0 for gradient in prompt_grads)
        assert_finite([parameter.grad for _, parameter in adapters])
        assert_finite([parameter.grad for parameter in student.parameters()])
        prompt_gradient_sums.append(prompt_sum)
        losses.append(loss.item())
        teacher_optimizer.step()
        student_optimizer.step()

    for name, before in frozen_before.items():
        assert torch.equal(before, dict(teacher.named_parameters())[name]), name

    # Adapter-only weights_only restore reproduces exact output and AdamW state.
    teacher.eval()
    with torch.no_grad():
        output_before_save = teacher(batch)
    payload = {
        "adapter": teacher.adapter_state_dict(),
        "optimizer": deepcopy(teacher_optimizer.state_dict()),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = SketchTeacher(
        visual,
        MEAN,
        STD,
        visual_prompt=True,
        mask_white_patches=True,
        deep_visual_prompt=True,
    ).cpu().eval()
    missing, unexpected = restored.load_state_dict(loaded["adapter"], strict=False)
    assert not unexpected
    restored_trainable = {name for name, _ in trainable(restored)}
    assert not (restored_trainable & set(missing))
    restored_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable(restored)], lr=1e-2, weight_decay=1e-2
    )
    restored_optimizer.load_state_dict(loaded["optimizer"])
    with torch.no_grad():
        assert torch.equal(output_before_save, restored(batch))
    assert_nested_equal(teacher_optimizer.state_dict(), restored_optimizer.state_dict())

    print(
        {
            "status": "PASS",
            "scope": "CPU_tiny_OpenCLIP_ViT_12_block_deep_visual_prompt_white_mask_QVLoRA",
            "image_size": 64,
            "patch_size": 32,
            "native_patches": 4,
            "width": 32,
            "heads": 4,
            "blocks": 12,
            "prompt_tokens_per_block": 4,
            "prompt_parameters": 12 * 4 * 32,
            "lora_parameter_tensors": 48,
            "updates": 2,
            "losses": losses,
            "prompt_gradient_sums": prompt_gradient_sums,
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "pretrained_or_real_data": False,
            "network_or_gpu": False,
        }
    )


if __name__ == "__main__":
    main()
