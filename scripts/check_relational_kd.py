"""Standalone check for dynamic visual prompt and relational distillation."""

import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch import nn

from spica.models.sketch_student import SketchStudent
from spica.models.sketch_teacher import (
    SketchTeacher,
    cross_modal_relational_kd,
    intra_modal_affinity_kd,
    relational_distillation,
)


def main():
    torch.manual_seed(42)
    # 1. Test relational distillation losses
    b, d = 4, 16
    student_q = F.normalize(torch.randn(b, d), dim=-1)
    teacher_q = F.normalize(torch.randn(b, d), dim=-1)
    photos = F.normalize(torch.randn(8, d), dim=-1)

    student_patches = F.normalize(torch.randn(b, 49, 16), dim=-1)
    teacher_patches = F.normalize(torch.randn(b, 49, 16), dim=-1)
    pos_photos = photos[:b]
    neg_photos = photos[b:]

    total_kd, rel_kd, aff_kd, fd_kd, patch_kd, margin_kd = relational_distillation(
        student_q,
        teacher_q,
        photos,
        student_patches=student_patches,
        teacher_patches=teacher_patches,
        pos_photos=pos_photos,
        neg_photos=neg_photos,
        temperature=0.2,
    )
    assert total_kd.ndim == 0 and torch.isfinite(total_kd)
    assert rel_kd.ndim == 0 and torch.isfinite(rel_kd) and rel_kd >= 0
    assert aff_kd.ndim == 0 and torch.isfinite(aff_kd) and aff_kd >= 0
    assert fd_kd.ndim == 0 and torch.isfinite(fd_kd) and fd_kd >= 0
    assert patch_kd.ndim == 0 and torch.isfinite(patch_kd) and patch_kd >= 0
    assert margin_kd.ndim == 0 and torch.isfinite(margin_kd) and margin_kd >= 0
    assert torch.allclose(total_kd, rel_kd + aff_kd + fd_kd + 0.5 * patch_kd + margin_kd)

    # 2. Test Dynamic Visual Prompt in SketchTeacher
    mean = (0.48, 0.46, 0.42)
    std = (0.27, 0.26, 0.25)
    visual = VisionTransformer(
        image_size=32,
        patch_size=16,
        width=32,
        layers=2,
        heads=4,
        mlp_ratio=2,
        output_dim=16,
    ).cpu().eval()

    teacher = SketchTeacher(visual, mean, std, visual_prompt=True).cpu().eval()
    raw_sketch = torch.rand(2, 3, 32, 32)
    out = teacher(raw_sketch)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()

    print({"status": "PASS", "scope": "CPU_visual_prompt_and_relational_kd"})


if __name__ == "__main__":
    main()
