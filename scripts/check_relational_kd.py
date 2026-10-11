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

    total_kd, rel_kd, aff_kd, fd_kd = relational_distillation(student_q, teacher_q, photos, temperature=0.2)
    assert total_kd.ndim == 0 and torch.isfinite(total_kd)
    assert rel_kd.ndim == 0 and torch.isfinite(rel_kd) and rel_kd >= 0
    assert aff_kd.ndim == 0 and torch.isfinite(aff_kd) and aff_kd >= 0
    assert fd_kd.ndim == 0 and torch.isfinite(fd_kd) and fd_kd >= 0
    assert torch.allclose(total_kd, rel_kd + aff_kd + fd_kd)

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

    teacher = SketchTeacher(visual, mean, std, dynamic_prompt=True).cpu().eval()
    assert teacher.prompt is not None and teacher.prompt.dynamic
    raw_sketch = torch.rand(2, 3, 32, 32)
    out_no_student = teacher(raw_sketch)
    assert out_no_student.shape == (2, 16)
    assert torch.isfinite(out_no_student).all()

    student_mock = nn.Module()
    student_mock.stem = nn.Sequential(
        nn.Conv2d(1, 192, 3, padding=1),
        nn.AdaptiveAvgPool2d((28, 28)),
    )
    teacher.student = student_mock
    out_with_student = teacher(raw_sketch)
    assert out_with_student.shape == (2, 16)
    assert torch.isfinite(out_with_student).all()

    print({"status": "PASS", "scope": "CPU_dynamic_prompt_and_relational_kd"})


if __name__ == "__main__":
    main()
