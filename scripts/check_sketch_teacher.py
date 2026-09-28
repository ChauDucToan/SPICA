"""Standalone CPU check for SketchTeacher Q/V LoRA."""

import io
from copy import deepcopy

import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch import nn

from spica.models.sketch_teacher import SketchTeacher, cosine_distillation
from spica.train_student import multi_positive_loss


def adapter_parameters(teacher):
    return [
        (name, parameter)
        for name, parameter in teacher.named_parameters()
        if ".parametrizations.in_proj_weight.0." in name
    ]


def optimizer_for(teacher):
    return torch.optim.AdamW(
        [parameter for parameter in teacher.parameters() if parameter.requires_grad],
        lr=1e-2,
        weight_decay=1e-2,
    )


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

    rng_before = torch.get_rng_state()
    teacher = SketchTeacher(visual, mean, std).cpu().eval()
    assert torch.equal(rng_before, torch.get_rng_state()), "teacher changed CPU RNG"
    assert len(teacher.visual.transformer.resblocks) == 12
    assert all(block.attn.dropout == 0 for block in teacher.visual.transformer.resblocks)

    adapters = adapter_parameters(teacher)
    assert len(adapters) == 12 * 4
    assert all(parameter.requires_grad for _, parameter in adapters)
    assert all(
        not parameter.requires_grad
        for name, parameter in teacher.named_parameters()
        if name not in {adapter_name for adapter_name, _ in adapters}
    )
    for block in teacher.visual.transformer.resblocks:
        adapter = block.attn.parametrizations.in_proj_weight[0]
        assert adapter.q_a.shape == (8, 32)
        assert adapter.q_b.shape == (32, 8)
        assert adapter.v_a.shape == (8, 32)
        assert adapter.v_b.shape == (32, 8)
        assert torch.count_nonzero(adapter.q_b) == 0
        assert torch.count_nonzero(adapter.v_b) == 0

    normalized = (raw - teacher.mean) / teacher.std
    with torch.no_grad():
        expected_base = F.normalize(visual(normalized), dim=-1)
        zero_adapter = teacher(raw)
    assert torch.allclose(zero_adapter, expected_base, atol=1e-6, rtol=1e-6)

    # Check packed Q/V algebra against an explicitly unparametrized visual.
    bases = [
        block.attn.in_proj_weight.detach().clone()
        for block in visual.transformer.resblocks
    ]
    expected_weights = []
    with torch.no_grad():
        for block, base in zip(teacher.visual.transformer.resblocks, bases):
            adapter = block.attn.parametrizations.in_proj_weight[0]
            adapter.q_b.normal_()
            adapter.v_b.normal_()
            q_delta = adapter.q_b @ adapter.q_a
            v_delta = adapter.v_b @ adapter.v_a
            expected = base + torch.cat((q_delta, torch.zeros_like(q_delta), v_delta))
            effective = block.attn.in_proj_weight.detach()
            assert torch.equal(effective[32:64], base[32:64]), "packed K changed"
            assert torch.allclose(effective, expected, atol=0, rtol=0)
            expected_weights.append(expected)

        explicit_visual = deepcopy(visual)
        for block, expected in zip(explicit_visual.transformer.resblocks, expected_weights):
            block.attn.in_proj_weight.copy_(expected)
        explicit_output = F.normalize(explicit_visual(normalized), dim=-1)
        parametrized_output = teacher(raw)
    assert torch.allclose(parametrized_output, explicit_output, atol=1e-6, rtol=1e-6)

    # Restore zero B for the expected first-step LoRA gradient condition.
    with torch.no_grad():
        for block in teacher.visual.transformer.resblocks:
            adapter = block.attn.parametrizations.in_proj_weight[0]
            adapter.q_b.zero_()
            adapter.v_b.zero_()

    base_after_init = [base.clone() for base in bases]
    labels = torch.tensor([0, 1, 0, 1])
    photo_labels = torch.tensor([0, 0, 1, 1])
    student = nn.Linear(16, 16, bias=False)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2, weight_decay=1e-2)
    teacher_optimizer = optimizer_for(teacher)

    # Detached teacher in cosine_distillation must receive no gradient.
    teacher.zero_grad(set_to_none=True)
    student.zero_grad(set_to_none=True)
    detached_teacher = teacher(raw[:2])
    student_features = F.normalize(student(torch.randn(2, 16)), dim=-1)
    cosine_distillation(student_features, detached_teacher).backward()
    assert all(parameter.grad is None for _, parameter in adapters)

    losses = []
    fixed_photos = F.normalize(torch.randn(4, 16), dim=-1)
    for update in range(2):
        teacher_optimizer.zero_grad(set_to_none=True)
        student_optimizer.zero_grad(set_to_none=True)
        teacher_queries = teacher(raw)
        queries = F.normalize(student(torch.randn(4, 16)), dim=-1)
        retrieval = multi_positive_loss(teacher_queries, fixed_photos, labels, photo_labels)
        retrieval = retrieval + multi_positive_loss(queries, fixed_photos, labels, photo_labels)
        retrieval.backward()
        assert sum(parameter.grad.abs().sum().item() for _, parameter in adapters) > 0
        assert sum(parameter.grad.abs().sum().item() for parameter in student.parameters()) > 0

        teacher_optimizer.zero_grad(set_to_none=True)
        student_optimizer.zero_grad(set_to_none=True)
        teacher_queries = teacher(raw)
        queries = F.normalize(student(torch.randn(4, 16)), dim=-1)
        retrieval = multi_positive_loss(teacher_queries, fixed_photos, labels, photo_labels)
        retrieval = retrieval + multi_positive_loss(queries, fixed_photos, labels, photo_labels)
        loss = retrieval + cosine_distillation(queries, teacher_queries)
        loss.backward()
        losses.append(loss.item())

        a_grad = [parameter.grad for name, parameter in adapters if ".q_a" in name or ".v_a" in name]
        b_grad = [parameter.grad for name, parameter in adapters if ".q_b" in name or ".v_b" in name]
        assert all(gradient is not None for gradient in a_grad + b_grad)
        if update == 0:
            assert all(torch.count_nonzero(gradient) == 0 for gradient in a_grad)
            assert sum(gradient.abs().sum().item() for gradient in b_grad) > 0
        else:
            assert sum(gradient.abs().sum().item() for gradient in a_grad) > 0
        assert sum(parameter.grad.abs().sum().item() for parameter in student.parameters()) > 0
        assert sum(gradient.abs().sum().item() for gradient in b_grad) > 0
        teacher_optimizer.step()
        student_optimizer.step()

    assert all(
        torch.equal(base, block.attn.parametrizations.in_proj_weight.original.detach())
        for base, block in zip(base_after_init, teacher.visual.transformer.resblocks)
    )

    teacher.eval()
    with torch.no_grad():
        eval_output = teacher(raw)
    teacher.train()
    with torch.no_grad():
        train_output = teacher(raw)
    assert torch.allclose(train_output, eval_output, atol=2e-6, rtol=2e-5), (
        "train/eval parity exceeded atol=2e-6, rtol=2e-5"
    )

    # Save only adapter weights plus optimizer state, then restore into fresh base weights.
    output_before_save = eval_output.clone()
    payload = {
        "adapter": teacher.adapter_state_dict(),
        "optimizer": deepcopy(teacher_optimizer.state_dict()),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = SketchTeacher(visual, mean, std).cpu().eval()
    missing, unexpected = restored.load_state_dict(loaded["adapter"], strict=False)
    assert not unexpected
    assert not any(name in missing for name, _ in adapter_parameters(restored))
    restored_optimizer = optimizer_for(restored)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    with torch.no_grad():
        assert torch.equal(output_before_save, restored(raw))
    for left, right in zip(
        teacher_optimizer.state_dict()["state"].values(),
        restored_optimizer.state_dict()["state"].values(),
    ):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(left[key], right[key]), f"optimizer state mismatch: {key}"

    print(
        {
            "status": "PASS",
            "scope": "CPU_tiny_OpenCLIP_ViT_12_block_QVLoRA",
            "updates": 2,
            "losses": losses,
            "train_eval_atol": 2e-6,
            "train_eval_rtol": 2e-5,
            "pretrained_or_real_data": False,
            "network_or_gpu": False,
        }
    )


if __name__ == "__main__":
    main()
