# V1 — real pretrained CLIP, batch32 no-update GPU preflight

**2026-09-08 · PASS trong scope preflight · CHƯA TRAIN.**

User đã duyệt hai bước: kiểm pretrained encoder/prompts/gradient thật, rồi đo bộ nhớ batch32 × clean/corrupted2views, không optimizer update hay diagnostic λ. Chạy đúng một graph với SIGReg không được gọi. Không W&B, không official-test/gallery evaluation, không đổi dependencies hay historical artifacts.

## Identity và evidence

- Root: **`outputs/coupled_predictive_preflight_20260908T101310Z/`**; [tracked raw-precision summary](coupled_predictive_v1_preflight_2026-09-08.json) chứa selected diagnostics và raw receipt hashes.
- Script: `scripts/preflight_coupled_predictive.py`; CPU helper tests: `tests/test_coupled_predictive_preflight.py`.
- CUDA execution: **10:13:20–10:13:42 UTC**, 2026-09-08. `preflight_receipt.json` và7 `phase_*.json`: PASS.
- Source HEAD tại execution: `d53dfb42457d922bb87e6bf3fe3273b859241a3d` **cộng script/test preflight chưa commit**, đã archive/hash đầy đủ; không gọi HEAD này là toàn bộ executed snapshot.
- Executed source SHA256 **`38b0cd249cb8e0581585c11bfffd1e00e71165ce8c1930a8576de0eec37a3bcb`**,550 source/config files. Trước/sau execution không đổi. `source_snapshot/files/` và `source_snapshot/index.json` chứa actual files/hashes, kể cả source-like files của `outputsnewgate/` theo inventory hiện hữu. Chỉ đọc/copy, không sửa chúng.
- CLIP `ViT-B-32-quickgelu`, direct local safetensors path;605,143,284bytes, SHA256 **`e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`**. Không dùng random weights, không tải pretrained từ network.
- GPU RTX5070Ti, logical device0; Torch2.13.0+cu130/CUDA13.0; float32. Không AMP, accumulation, activation checkpointing hay tự giảm batch khi đo.
- Báo cáo này và commit sau execution là postprocessing, **không phải executed source snapshot**, càng không phải historical S0–S2 training source.

## Data và graph thật

Reuse data/split/loader của semantic S0 **chỉ để lấy train batch**, không reuse S0 model/loss/optimizer. Pseudo3407:84 train classes,46,624 sketches,58,950 train photos, canonical positive pool8,400; pairing manifest SHA đã kiểm. Positive sampling `same_class` trong canonical pool, không gọi đây là exact-pair experiment.

-32 unique original sketches;32 positives +32 negatives tạo64 unique photo IDs. Labels/path membership kiểm với train manifest; negative khác category. Tất cả ảnh chỉ đọc.
-32 region records `status=ok`, severities25/50/75% có17/7/8 samples. Clean clone, seed4242+update0/view1; không tạo mask view0, không thinning, không drop/retry blank samples.
-Query forward **1 lần cho64 views**; live photo encoder, original photo reference và complete84-class text bank mỗi loại1 lần. Shared prompts kiểm cùng storage với parameter owner.
-Student output `[64,50,768]`; predictor context đúng slice bỏ CLS `[64,49,768]`; `g` đúng mean patch tokens `[64,768]`, không normalize. `q/mu_i/mu_t` `[64,512]`, unit norm; photo bank `[64,512]`, text bank `[84,512]`.
-Photo prompt `[3,768]`, text context `[4,512]`, T0 `[84,512]`; initial text parity max error **0.0**. Tất cả registered parameters/buffers trên CUDA0.

## Freeze, gradient và no-update

-152 student visual state entries byte-equal teacher tại initialization, không shared storage.
-All original CLIP weights frozen/eval; teacher constructor before/after parameter/state hashes khớp. Không original gradients; unused student ln_post/proj cũng không gradient.
-**482/482 model state entries không đổi** sau graph; parameter flags/bytes khớp trước/sau. Không tạo optimizer, không optimizer/scheduler step.
-**173/173 trainable parameter tensors** có gradient hữu hạn, khác0; tổng89,036,288 trainable elements. Group L2 và từng parameter norm nằm trong raw receipt.
-Gradient norm C_I **1.669823884964**, C_T **1.093824625015**. Rank→live photo bank norm **0.271372765303**; CE→live text bank **4.167008399963**. Các edge probes chỉ chứng minh autograd route, không calibration hoặc conflict analysis.
-Total loss **6.703705787659**. SIGReg scalar0 là explicit no-compute control, không chọn hệ số arm SIGReg. Initial text anchor **−4.75418e−8** do cosine floating-point roundoff; không sửa/clamp raw evidence.

## Bộ nhớ — empirical, không training guarantee

Scope: **full forward + intermediate rank/CE edge probes + total backward**, peak reset trước forward. Không gồm optimizer states/step; không phải pure trainer-step benchmark. Hooks/diagnostic overhead hiện diện.

| Metric | Bytes | GiB |
|---|---:|---:|
| Baseline allocated |1,078,708,224|1.005|
| Peak allocated |**7,876,152,320**|**7.335**|
| Peak reserved |**9,007,267,840**|**8.389**|
| CUDA-reported device total |16,602,497,024|15.462|

Full graph + edge checks0.500886s, total backward0.189955s (synchronized single observation, **không throughput benchmark**). Whole preflight≈21.46s gồm source/data/weight verification, initialization và state hashing. GPU trở về10MiB sau process exit.

**Kết luận giới hạn:** float32 batch32/64views no-SIG graph vừa VRAM trên batch này; chưa chứng minh full AdamW training, SIGReg-on, nhiều batch/probes, gallery caching hoặc checkpoint lifecycle đều vừa bộ nhớ.

## Independent checks và reproduce

Parent full suite **293 tests PASS**, Ruff/diff PASS trước launch. Independent reviewer sau execution: PASS, hash lại550 archived files/aggregate và CPU reconstruct32 views. Parent còn lưu runnable postprocessor `verify_parent.py`, `parent_verification.json/.log`:96 raw-image SHA checks,96 normalized tensor checks,32 full region metadata + clean/corrupted hashes khớp; objective scalar recomputation delta2.56229e−7. Đây là CPU reconstruction/receipt audit, **không second GPU gradient replay**. Hash proof chứng nhận những tensors đã serialize thành hashes; không tạo gradient arrays vốn chưa serialize.

```bash
# Chỉ chạy lại preflight khi được yêu cầu; output phải mới, không có retry/fallback tự động.
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
  scripts/preflight_coupled_predictive.py --output-dir outputs/<NEW_ROOT> --device cuda

# Postprocessing CPU, không model update/GPU:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  outputs/coupled_predictive_preflight_20260908T101310Z/verify_parent.py
```

**Tiếp theo chưa thực hiện:** khóa diagnosticρ/batches/parameter scope và scheduler/promotion; xin phép riêng cho diagnostic λ, trainer implementation và training R0/R1. Preflight PASS không tự cấp phép các bước đó. [CPU implementation milestone](coupled_predictive_v1_cpu_readiness_2026-09-08.md), [region protocol](designs/coupled_predictive_region_v1/region_first_protocol.md), [SIGReg identity](sigreg_source_identity_2026-09-08.json).
