"""Standalone CPU checks for SketchTeacher exact-white patch key masking."""

import importlib.util
import io
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch import nn

from spica.models.sketch_teacher import SketchTeacher, cosine_distillation

ATOL = 4e-5
RTOL = 4e-5
MEAN = (0.48, 0.46, 0.42)
STD = (0.27, 0.26, 0.25)


def assert_finite(values):
    assert all(value is not None and torch.isfinite(value).all() for value in values)


def trainable(teacher):
    return [(name, p) for name, p in teacher.named_parameters() if p.requires_grad]


def retrieval_loss(queries, photos, labels, photo_labels):
    positives = photo_labels[None, :] == labels[:, None]
    return -(queries @ photos.T).log_softmax(dim=1)[positives].view(labels.numel(), -1).mean()


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            assert_state_equal(left[key], right[key])
        elif torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def archived_teacher_class():
    path = (
        Path(__file__).resolve().parents[1]
        / "outputs/student_lora_vpt_kd12_teacher_10epochs_20260928/source/src/spica/models/sketch_teacher.py"
    )
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("archived_sketch_teacher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SketchTeacher


def project(visual, tokens):
    pooled, _ = visual._pool(tokens)
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled, dim=-1)


def expected_mask(blocked, heads):
    batch, length = blocked.shape
    mask = torch.zeros(batch, heads, length, length)
    return mask.masked_fill(blocked[:, None, None, :], float("-inf")).reshape(
        batch * heads, length, length
    )


def main():
    from spica.train_student import train
    with tempfile.TemporaryDirectory() as directory:
        for joint, vpt, coefficient in [(False, False, 1.2), (True, False, 1.2),
                                         (True, True, 1.), (True, True, None)]:
            args = SimpleNamespace(output=Path(directory) / 'never_created',
                                   lora_teacher=joint, visual_prompt=vpt,
                                   lambda_kd=coefficient, mask_white_patches=True)
            try:
                train(args)
            except ValueError:
                pass
            else:
                raise AssertionError('wrong mask experiment accepted')
            assert not args.output.exists()
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

    # Default false remains byte-identical to the archived historical teacher.
    archived = archived_teacher_class()
    old = archived(visual, MEAN, STD, visual_prompt=True).cpu().eval()
    current = SketchTeacher(visual, MEAN, STD, visual_prompt=True).cpu().eval()
    assert old.state_dict().keys() == current.state_dict().keys()
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key], current.state_dict()[key]), key
    with torch.no_grad():
        assert torch.equal(old(raw), current(raw))

    masked = SketchTeacher(
        visual, MEAN, STD, visual_prompt=True, mask_white_patches=True
    ).cpu().eval()
    assert masked.prompt_count == 4
    assert len(masked.visual.transformer.resblocks) == 12
    assert masked.visual.conv1.kernel_size == (32, 32)

    # Exact RGB equality, one-channel gray pixel, and per-image batch masks.
    batch = torch.full((4, 3, 64, 64), 0.2)
    batch[0, :, :32, :32] = 1
    batch[1] = 1
    batch[1, 0, 0, 0] = 0.5  # one channel differs: patch is not white
    batch[2, :, :32, :32] = 1
    batch[2, :, 32:, 32:] = 1
    expected_patches = torch.tensor(
        [[True, False, False, False], [False, True, True, True], [True, False, False, True], [False] * 4]
    )
    blocked = masked.white_patch_mask(batch)
    assert torch.equal(blocked[:, :5], torch.zeros(4, 5, dtype=torch.bool))
    assert torch.equal(blocked[:, 5:], expected_patches)

    # Every native OpenCLIP attention block receives the same B*heads x L x L key mask.
    seen_masks = []
    hooks = [
        block.attn.register_forward_hook(
            lambda _module, _args, kwargs, _output: seen_masks.append(
                kwargs["attn_mask"].detach().clone()
            ),
            with_kwargs=True,
        )
        for block in masked.visual.transformer.resblocks
    ]
    with torch.no_grad():
        masked_batch_output = masked(batch)
    for hook in hooks:
        hook.remove()
    assert len(seen_masks) == 12
    expected = expected_mask(blocked, heads=4)
    for seen in seen_masks:
        assert torch.equal(seen, expected)
    assert torch.isfinite(masked_batch_output).all()

    # No blank patches: masked and unmasked paths agree within predeclared CPU tolerance.
    unmasked = SketchTeacher(visual, MEAN, STD, visual_prompt=True).cpu().eval()

    # Masked full sequence equals physically trimmed valid-token sequence at final CLS.
    with torch.no_grad():
        normalized = (batch - masked.mean) / masked.std
        tokens = masked.visual._embeds(normalized)
        full_tokens = masked.visual.transformer(tokens, attn_mask=expected)
        for index in range(batch.shape[0]):
            keep = ~blocked[index]
            trimmed_tokens = masked.visual.transformer(tokens[index : index + 1, keep])
            assert torch.allclose(
                full_tokens[index : index + 1, keep], trimmed_tokens, atol=ATOL, rtol=RTOL
            )
            assert torch.allclose(
                project(masked.visual, full_tokens[index : index + 1, :1]),
                project(masked.visual, trimmed_tokens[:, :1]),
                atol=ATOL,
                rtol=RTOL,
            )

        # Arbitrary blank-token values cannot affect valid outputs when used as keys.
        perturbed = tokens.clone()
        perturbed[blocked] = torch.randn_like(perturbed[blocked]) * 37 + 11
        perturbed_tokens = masked.visual.transformer(perturbed, attn_mask=expected)
        assert torch.allclose(
            full_tokens[~blocked], perturbed_tokens[~blocked], atol=ATOL, rtol=RTOL
        )

    with torch.no_grad():
        no_blank_masked = masked(raw)
        no_blank_unmasked = unmasked(raw)
        assert torch.allclose(
            no_blank_masked, no_blank_unmasked, atol=ATOL, rtol=RTOL
        ), (no_blank_masked - no_blank_unmasked).abs().max().item()

    # All-white input stays finite; CLS and four prompts remain valid and gradients finite.
    all_white = torch.ones(2, 3, 64, 64)
    all_white_blocked = masked.white_patch_mask(all_white)
    assert not all_white_blocked[:, :5].any()
    assert all_white_blocked[:, 5:].all()
    masked.zero_grad(set_to_none=True)
    all_white_output = masked(all_white)
    assert torch.isfinite(all_white_output).all()
    all_white_output.square().mean().backward()
    assert_finite([p.grad for _, p in trainable(masked)])

    # Train/eval parity for masked path, including the native attention-mask route.
    parity = SketchTeacher(
        visual, MEAN, STD, visual_prompt=True, mask_white_patches=True
    ).cpu()
    parity.eval()
    with torch.no_grad():
        eval_output = parity(batch)
    parity.train()
    with torch.no_grad():
        train_output = parity(batch)
    assert torch.allclose(train_output, eval_output, atol=ATOL, rtol=RTOL)

    # KD is detached from teacher; two joint optimizer updates train prompt + LoRA only.
    teacher = SketchTeacher(
        visual, MEAN, STD, visual_prompt=True, mask_white_patches=True
    ).cpu().train()
    student = nn.Linear(16, 16, bias=False)
    student_input = torch.randn(4, 16)
    photos = F.normalize(torch.randn(4, 16), dim=-1)
    labels = torch.tensor([0, 1, 0, 1])
    photo_labels = torch.tensor([0, 0, 1, 1])
    teacher_optimizer = torch.optim.AdamW([p for _, p in trainable(teacher)], lr=1e-2)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)

    teacher.zero_grad(set_to_none=True)
    student.zero_grad(set_to_none=True)
    kd_only = cosine_distillation(
        F.normalize(student(student_input), dim=-1), teacher(batch)
    )
    kd_only.backward()
    assert all(p.grad is None for _, p in trainable(teacher))

    frozen_before = {
        name: p.detach().clone() for name, p in teacher.named_parameters() if not p.requires_grad
    }
    update_losses = []
    for _ in range(2):
        teacher_optimizer.zero_grad(set_to_none=True)
        student_optimizer.zero_grad(set_to_none=True)
        teacher_queries = teacher(batch)
        student_queries = F.normalize(student(student_input), dim=-1)
        teacher_retrieval = retrieval_loss(teacher_queries, photos, labels, photo_labels)
        student_retrieval = retrieval_loss(student_queries, photos, labels, photo_labels)
        loss = teacher_retrieval + student_retrieval + cosine_distillation(
            student_queries, teacher_queries
        )
        loss.backward()
        assert_finite([p.grad for _, p in trainable(teacher)])
        assert_finite([p.grad for p in student.parameters()])
        update_losses.append(loss.item())
        teacher_optimizer.step()
        student_optimizer.step()

    assert all(
        torch.equal(before, dict(teacher.named_parameters())[name])
        for name, before in frozen_before.items()
    )

    # weights_only adapter + optimizer restore reproduces masked output and optimizer state.
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
        visual, MEAN, STD, visual_prompt=True, mask_white_patches=True
    ).cpu().eval()
    missing, unexpected = restored.load_state_dict(loaded["adapter"], strict=False)
    assert not unexpected
    assert not ({name for name, _ in trainable(restored)} & set(missing))
    restored_optimizer = torch.optim.AdamW([p for _, p in trainable(restored)], lr=1e-2)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    with torch.no_grad():
        assert torch.equal(output_before_save, restored(batch))
    assert_state_equal(teacher_optimizer.state_dict(), restored_optimizer.state_dict())

    print(
        {
            "status": "PASS",
            "scope": "CPU_tiny_OpenCLIP_ViT_12_block_mask_white_patches",
            "image_size": 64,
            "patch_size": 32,
            "width": 32,
            "output_dim": 16,
            "heads": 4,
            "prompt_tokens": 4,
            "attention_blocks_checked": 12,
            "updates": 2,
            "losses": update_losses,
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "pretrained_or_real_data": False,
            "network_or_gpu": False,
        }
    )


if __name__ == "__main__":
    main()
