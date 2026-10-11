# VMamba-Tiny SS2D: fresh 60 epochs và Dependency Injection

**Ngày:** 2026-10-04. **Trạng thái tài liệu:** protocol trước launch; không phải kết quả hoàn thành.

## 1. Quyết định được người dùng phê duyệt

- Thay toàn bộ backbone CNN–Mamba bằng **VMamba-Tiny chính thức**, không chỉ thay scan trong CNN.
- **Khởi tạo mới, không dùng ImageNet-pretrained VMamba**.
- **Student-only:** CLIP photo encoder/gallery frozen; giữ multi-positive retrieval loss, không sketch teacher, LoRA, visual prompt hay KD.
- Một run **60 epochs**, TU-Berlin official 220/30, seed42, B32, `drop_last=False`; không resume, chọn best checkpoint, thử seed/hyperparameter hoặc tự retry production.
- Đo thời gian trên GPU/batch train thật trước khi ước lượng. Không thay scan chính thức bằng Python fallback để có số đo.

## 2. Backbone và ranh giới “chính thức”

Nguồn: [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba/tree/2ed52ead062a51a64521ed3871d52914bf532876), commit **`2ed52ead062a51a64521ed3871d52914bf532876`**, MIT.

Variant: `vmamba_tiny_s1l8`, YAML `classification/configs/vssm/vmambav2v_tiny_224.yaml`: depths `[2,2,8,2]`, dims `[96,192,384,768]`, state1, SSM ratio1, SS2Dv2 `v05_noz`, drop-path0.2, LN2d, downsamplev3, patchembedv2. Đây là **variant SS2D được upstream cung cấp**, không phải SS2Dv0 đầu tiên hay Mamba-2.

[Adapter](../src/spica/models/vmamba_student.py) giữ nguyên stage, scan bốn hướng, initialization, pooling và drop-path upstream; đổi native classifier head từ 1000 thành **512** và L2-normalize bằng FP32. Tổng **29,873,792** tham số trainable. Các convolution thuộc patch embedding/SS2D nguyên bản vẫn còn; không thêm các CNN/hybrid block/foreground pooling của Student cũ. Mã upstream được copy byte-for-byte vào [`src/spica/vendor/vmamba/`](../src/spica/vendor/vmamba/README.md), giữ license và loại khỏi Ruff formatting.

RGB `[0,1]`, resize/crop224 như Student cũ, không ImageNet normalization hay augmentation mới. Photo transform theo frozen OpenAI CLIP ViT-B/32 QuickGELU. Vì vậy đây là **adaptation cho sketch retrieval**, không phải tái lập training ImageNet hoặc một cấu hình ZS-SBIR của tác giả VMamba.

Kernel: `selective_scan_cuda_oflex` build từ chính commit trên cho Python3.12/Torch2.13+cu130/SM120. Factory kiểm tra backend đã import và từ chối nếu thiếu; không dùng reference Python scan. Cross-scan/merge dùng Triton upstream. Build riêng trong `outputs/vmamba_build_20261004/`: CUDA nvcc13.0.88, CCCL12.8.90 headers tương thích API upstream, GCC13/glibc2.40 từ nixpkgs `50ab793786d9de88ee30ec4e4c24fb4236fc2674`. Không sửa Python/CUDA/C++ của upstream. Giữ log build thất bại trước đó: thiếu Nix libcuda path; CUDA13 CCCL mới/glibc2.42 không tương thích các API header cũ. Đây không phải production training failures.

Thêm optional extra `vmamba` với fvcore và các dependency của nó; native wheel cục bộ được cài `--no-deps`. Giữ nguyên phiên bản các package có sẵn, đặc biệt Torch/torchvision/Triton/mamba-ssm; không sửa môi trường sibling. Receipt bind SHA của binary, wheel, source kernel và environment. Wheel phụ thuộc ABI/GPU build, không được xem như dependency portable cho máy khác.

## 3. Loss, optimizer và lịch

- Positive: **mọi photo cùng category trong batch**, không suy diễn instance pairing.
- Loss: `multi_positive_loss`, temperature0.07; không thêm objective.
- AdamW LR3e-4, weight decay0.01; giữ grouping cũ: vector/bias và parameter có `_no_weight_decay` không decay.
- BF16 autocast, Student gradient clipping1.
- Warmup5% + cosine theo **28,920 updates**, warmup **1,446**.
- 15,400 sketches / epoch → **482 batches**, last batch **8**, **924,000 observations** / 60 epochs.

Final checkpoint duy nhất; final clean test 2,400 queries/27,989 gallery, full mAP, AP200 `min(relevant,200)` denominator, P200. Không intermediate test, masking hoặc checkpoint selection. Các class label chỉ phục vụ training loss/evaluator, không vào encoder ở inference; không unseen-class text bank. W&B `sbir`: final ba retrieval scalar; losses/epoch và runtime lưu local. MP-Q step1189 là control cố định lịch sử, khác hệ thống/budget; không phải matched backbone-only ablation. Official test đã từng được xem, không phải blind holdout. Không nâng các historical raw FAIL/UNVERIFIED thành PASS.

## 4. DI nhỏ, không framework

[`train_student.py`](../src/spica/train_student.py) nhận hai dependency qua keyword:

```python
from spica.models.vmamba_student import build_vmamba_tiny
from spica.train_student import multi_positive_loss, train

train(args, student_factory=build_vmamba_tiny, retrieval_loss=multi_positive_loss)
```

Factory không đối số, trả về `nn.Module` có `.device`, nhận `B×3×224×224` RGB và trả `B×512` embedding normalized. Loss nhận `(query, photos, labels, photo_labels)` và trả scalar differentiable. Có thể dùng constructor, callable hoặc `functools.partial`; với preset khác, model/factory nên cung cấp `experiment_metadata` để ghi rõ các lựa chọn. Trainer archive source local; không bảo đảm provenance của code/package ngoài repository chỉ vì được inject.

CLI tương đương, không cần sửa nhánh `if architecture` trong trainer:

```bash
PYTHONPATH=src \
TRITON_LIBCUDA_PATH=/run/opengl-driver/lib \
LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH \
.venv/bin/python -m spica.train_student \
  --epochs 60 \
  --student-factory spica.models.vmamba_student:build_vmamba_tiny \
  --data-config configs/data/tuberlin_220_30.yaml \
  --clip-weights "$CLIP_WEIGHTS" \
  --mpq-summary "$MPQ_SUMMARY" \
  --output outputs/vmamba_tiny_tu_60epochs_20261004 \
  --run-name VMamba-Tiny-SS2D-Scratch-TU-60epochs \
  --wandb-mode online
```

`--retrieval-loss MODULE:CALLABLE` dành cho thí nghiệm tương lai được phê duyệt riêng; **không đổi loss trong run này**. Chỉ import code local được tin cậy, không chạy spec từ nguồn không tin cậy. Nếu vừa truyền dependency API vừa truyền flag tương ứng, fail trước launch. Bỏ flag/injection giữ defaults CNN–Mamba100 / joint10 và loss cũ. DI ở encoder/loss đủ cho yêu cầu hiện tại: không DI container, registry framework hay refactor trainer sang nhiều lớp.

## 5. Checks, timing và launch fail-closed

Standalone check: [`scripts/check_vmamba_student.py`](../scripts/check_vmamba_student.py), synthetic CUDA: recurrence/gradient FP64 với 65/257/3136 positions; full224 forward/backward, 14 SS2D blocks/bốn nhóm direction, unit embeddings, finite gradients, safe weights-only round-trip; chủ động chặn Python scan fallback. Không encode real/official data, không xác nhận toàn bộ mọi input/kernel.

Planned gate root **`outputs/vmamba_tiny_gate_20261004/`**:

1. Exercise trainer thật qua DI, giữ horizon60/B32; gate chỉ 3 updates32/32/prefix8 train-only, checkpoint epoch và planned stop trước epoch2/final test. Check FP64 loss/gradient, frozen photo state, all-block gradients, model/Adam moments/scheduler/RNG restore. Prefix8 là coverage last-batch shape, không phải batch cuối nguyên bản.
2. Encode/evaluate một subset train-only nhỏ bằng evaluator thật; check saved scores/ranks/AP/P200 bằng CPU FP64 trong tolerance cố định, không tolerance relaxation. Timing final inference chỉ suy ra từ train subset; không mở official-test images trước production final.
3. Benchmark tiếp một prefix train với loader/photo forward/loss/backward/AdamW thật, có warmup và CUDA synchronize; đây là throwaway measurement state, **không checkpoint/resume cho production**. Ghi thời gian median/quantiles, wall-clock end-to-end, peak VRAM và dự báo60 trong `timing.json`. Không khẳng định thời gian CPU/setup/test bằng thời gian GPU compute.
4. Source/manifest/CLIP/native-kernel/environment SHA bind gate và launch; GPU/PID/root inspection, không duplicate. **Freeze mọi nonignored source/config/docs suốt actual gates + production**. Gate scripts/receipts/tiến độ mới chỉ nằm trong ignored `outputs/`.

Planned production **`outputs/vmamba_tiny_tu_60epochs_20261004/`**; adjacent launch/execution receipt + runtime + live PID là authority. Một supervisor chờ child, ghi exit code; không autoretry/resume. Nếu gate fail hoặc runtime fail, giữ nguyên evidence và dừng để quyết định, không tự thay architecture/seed/precision/budget.

Tài liệu này không chứng minh đã launch/hoàn thành, không hứa mAP tăng hoặc thời gian cố định. Đọc timing receipt và runtime thực tế để biết số đo/trạng thái. Training snapshot có source dirty/untracked đã archive; HEAD riêng không đại diện đầy đủ source. Không commit/push hay sửa sibling/historical artifacts trong campaign.
