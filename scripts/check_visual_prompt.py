"""Standalone CPU checks for shallow visual prompts plus Q/V-LoRA."""

import importlib.util
import io
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch import nn

from spica.models.sketch_teacher import SketchTeacher, cosine_distillation


def adapter_parameters(teacher):
    return [
        (name, parameter)
        for name, parameter in teacher.named_parameters()
        if ".parametrizations.in_proj_weight.0." in name
    ]


def trainable_parameters(teacher):
    return [(name, p) for name, p in teacher.named_parameters() if p.requires_grad]


def optimizer_for(teacher):
    return torch.optim.AdamW(
        [p for _, p in trainable_parameters(teacher)], lr=1e-2, weight_decay=1e-2
    )


def pre_ln_tokens(visual, images):
    patches = visual.conv1(images)
    patches = patches.reshape(patches.shape[0], patches.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(patches.dtype).view(1, 1, -1).expand(images.shape[0], -1, -1)
    return visual.patch_dropout(
        torch.cat((cls, patches), dim=1) + visual.positional_embedding.to(patches.dtype)
    )


def retrieval_loss(queries, photos, labels, photo_labels):
    logits = queries @ photos.T
    positives = photo_labels[None, :] == labels[:, None]
    return -(logits.log_softmax(dim=1)[positives]).view(labels.shape[0], -1).mean()


def assert_finite(grads):
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in grads)


def assert_nested_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            assert_nested_equal(left[key], right[key])
        elif torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def archived_teacher_class():
    path = (
        Path(__file__).resolve().parents[1]
        / "outputs/student_lora_teacher_10epochs_20260928/source/src/spica/models/sketch_teacher.py"
    )
    spec = importlib.util.spec_from_file_location("archived_sketch_teacher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SketchTeacher


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    mean = (0.48, 0.46, 0.42)
    std = (0.27, 0.26, 0.25)
    visual = VisionTransformer(
        image_size=32,
        patch_size=16,
        width=32,
        layers=12,
        heads=4,
        mlp_ratio=2,
        output_dim=16,
    ).cpu().eval()
    raw = torch.rand(4, 3, 32, 32)

    # Default current module remains byte-identical to archived pre-prompt behavior.
    archived = archived_teacher_class()
    old = archived(visual, mean, std).cpu().eval()
    rng_before = torch.get_rng_state()
    current_default = SketchTeacher(visual, mean, std).cpu().eval()
    assert torch.equal(rng_before, torch.get_rng_state())
    assert old.state_dict().keys() == current_default.state_dict().keys()
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key], current_default.state_dict()[key]), key
    with torch.no_grad():
        assert torch.equal(old(raw), current_default(raw))

    # Prompt initialization is isolated and does not alter LoRA initialization.
    rng_before = torch.get_rng_state()
    no_prompt = SketchTeacher(visual, mean, std, visual_prompt=False).cpu().eval()
    assert torch.equal(rng_before, torch.get_rng_state())
    rng_before = torch.get_rng_state()
    prompted = SketchTeacher(visual, mean, std, visual_prompt=True).cpu().eval()
    assert torch.equal(rng_before, torch.get_rng_state())
    no_prompt_adapters = dict(adapter_parameters(no_prompt))
    prompted_adapters = dict(adapter_parameters(prompted))
    assert no_prompt_adapters.keys() == prompted_adapters.keys()
    for name in no_prompt_adapters:
        assert torch.equal(no_prompt_adapters[name], prompted_adapters[name]), name

    assert len(prompted.visual.transformer.resblocks) == 12
    assert prompted.visual.ln_pre[0].tokens.shape == (4, 32)
    original_tokens = pre_ln_tokens(visual, raw)
    inserted = prompted.visual.ln_pre[0](original_tokens)
    assert inserted.shape == (4, 9, 32)
    assert torch.equal(inserted[:, :1], original_tokens[:, :1])
    assert torch.equal(inserted[:, 5:], original_tokens[:, 1:])
    assert torch.equal(
        prompted.visual._embeds(raw), prompted.visual.ln_pre[1](inserted)
    ), "explicit concat + native ln_pre mismatch"

    block_shapes = []
    hooks = [
        block.register_forward_hook(
            lambda _module, _inputs, output: block_shapes.append(tuple(output.shape))
        )
        for block in prompted.visual.transformer.resblocks
    ]
    with torch.no_grad():
        prompted_output = prompted(raw)
    for hook in hooks:
        hook.remove()
    assert block_shapes == [(4, 9, 32)] * 12

    # Forward uses native OpenCLIP CLS pooling and projection.
    with torch.no_grad():
        normalized_raw = (raw - prompted.mean) / prompted.std
        encoded = prompted.visual.transformer(prompted.visual._embeds(normalized_raw))
        native_pooled, _ = prompted.visual._pool(encoded)
        assert prompted.visual.pool_type == "tok"
        assert torch.equal(native_pooled, prompted.visual.ln_post(encoded)[:, 0])
        projected = native_pooled @ prompted.visual.proj
        assert torch.equal(prompted_output, F.normalize(projected, dim=-1))

    # Base parameters stay frozen; teacher retrieval and student KD have separate paths.
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in prompted.named_parameters()
        if not parameter.requires_grad
    }
    adapters = adapter_parameters(prompted)
    assert len(adapters) == 48
    trainable = trainable_parameters(prompted)
    assert len(trainable) == 49
    adapter_state = prompted.adapter_state_dict()
    assert len(adapter_state) == 49
    assert "visual.ln_pre.0.tokens" in adapter_state
    assert set(adapter_state) == {name for name, _ in trainable}
    assert all(parameter.requires_grad for _, parameter in adapters)
    assert all(
        not parameter.requires_grad
        for name, parameter in prompted.named_parameters()
        if name not in {adapter_name for adapter_name, _ in trainable}
    )
    with torch.no_grad():
        _ = prompted(raw)
    for name, before in base_before.items():
        assert torch.equal(before, dict(prompted.named_parameters())[name]), name

    student = nn.Linear(16, 16, bias=False)
    student_input = torch.randn(4, 16)
    fixed_photos = F.normalize(torch.randn(4, 16), dim=-1)
    labels = torch.tensor([0, 1, 0, 1])
    photo_labels = torch.tensor([0, 0, 1, 1])
    teacher_optimizer = optimizer_for(prompted)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2, weight_decay=1e-2)
    losses = []
    prompt_grad_sums = []
    b_grad_sums = []
    a_grad_sums = []

    for update in range(2):
        teacher_optimizer.zero_grad(set_to_none=True)
        student_optimizer.zero_grad(set_to_none=True)
        teacher_queries = prompted(raw)
        student_queries = F.normalize(student(student_input), dim=-1)
        teacher_retrieval = retrieval_loss(
            teacher_queries, fixed_photos, labels, photo_labels
        )
        student_retrieval = retrieval_loss(
            student_queries, fixed_photos, labels, photo_labels
        )
        kd = cosine_distillation(student_queries, teacher_queries)
        teacher_retrieval.backward(retain_graph=True)
        retrieval_grads = [parameter.grad.detach().clone() for _, parameter in trainable]
        prompted.zero_grad(set_to_none=True)
        student.zero_grad(set_to_none=True)
        total = teacher_retrieval + student_retrieval + kd  # fixed lambda_KD = 1
        total.backward()
        total_teacher_grads = [parameter.grad for _, parameter in trainable]
        assert all(torch.equal(left, right) for left, right in zip(retrieval_grads, total_teacher_grads))
        assert_finite(total_teacher_grads)
        assert_finite([parameter.grad for parameter in student.parameters()])

        prompt_grads = [parameter.grad for name, parameter in trainable if name.endswith("ln_pre.0.tokens")]
        b_grads = [
            parameter.grad
            for name, parameter in adapters
            if name.endswith(("q_b", "v_b"))
        ]
        a_grads = [
            parameter.grad
            for name, parameter in adapters
            if name.endswith(("q_a", "v_a"))
        ]
        assert_finite(prompt_grads + b_grads + a_grads)
        prompt_grad_sums.append(sum(gradient.abs().sum().item() for gradient in prompt_grads))
        b_grad_sums.append(sum(gradient.abs().sum().item() for gradient in b_grads))
        a_grad_sums.append(sum(gradient.abs().sum().item() for gradient in a_grads))
        assert prompt_grad_sums[-1] > 0
        assert b_grad_sums[-1] > 0
        if update == 0:
            assert a_grad_sums[-1] == 0
        else:
            assert a_grad_sums[-1] > 0
        losses.append(total.item())
        teacher_optimizer.step()
        student_optimizer.step()

    for name, before in base_before.items():
        assert torch.equal(before, dict(prompted.named_parameters())[name]), name
    assert not fixed_photos.requires_grad

    # Adapter-only, weights_only reload restores exact output and AdamW state.
    with torch.no_grad():
        output_before_save = prompted(raw)
    payload = {
        "adapter": prompted.adapter_state_dict(),
        "optimizer": deepcopy(teacher_optimizer.state_dict()),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = SketchTeacher(visual, mean, std, visual_prompt=True).cpu().eval()
    missing, unexpected = restored.load_state_dict(loaded["adapter"], strict=False)
    assert not unexpected
    restored_trainable = {name for name, _ in trainable_parameters(restored)}
    assert not (restored_trainable & set(missing))
    restored_optimizer = optimizer_for(restored)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    with torch.no_grad():
        assert torch.equal(output_before_save, restored(raw))
    assert_nested_equal(teacher_optimizer.state_dict(), restored_optimizer.state_dict())

    print(
        {
            "status": "PASS",
            "scope": "CPU_tiny_OpenCLIP_ViT_12_block_visual_prompt_QVLoRA",
            "blocks": 12,
            "prompt_tokens": 4,
            "prompt_shape": [4, 32],
            "lora_parameter_tensors": 48,
            "adapter_state_tensors": 49,
            "updates": 2,
            "losses": losses,
            "prompt_grad_sums": prompt_grad_sums,
            "lora_b_grad_sums": b_grad_sums,
            "lora_a_grad_sums": a_grad_sums,
            "pretrained_or_real_data": False,
            "network_or_gpu": False,
        }
    )


if __name__ == "__main__":
    main()
