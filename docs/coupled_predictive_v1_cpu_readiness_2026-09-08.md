# Coupled predictive V1 — model/loss CPU milestone

**2026-09-08 · CPU GATES PASS · CHƯA TRAIN.**

**Cập nhật sau milestone này:** user đã duyệt và hoàn tất [real pretrained/real-data GPU preflight](coupled_predictive_v1_preflight_2026-09-08.md), B32/64views/no-update/noSIG PASS. Những giới hạn “chưa GPU/pretrained gate” bên dưới là trạng thái tại CPU milestone; λ diagnostic/trainer/training vẫn pending.

Sau thống kê S0–S2, user đã chọn implement model/loss V1 và CPU gates trước, **không train**. Báo cáo này cập nhật trạng thái implementation; các bản [architecture](designs/coupled_predictive_region_v1/architecture.md) và [region protocol](designs/coupled_predictive_region_v1/region_first_protocol.md) vẫn là design specification, không phải evidence train. Semantic baseline giữ **S0** theo [kết quả đã verified](semantic_text_step1_results_2026-09-08.md).

## Code đã có

- `src/spica/models/coupled_predictive.py`: `CoupledPredictiveModel` nhận `FrozenClipEncoder` đã tạo, tokenizer và class map; không tự tải weights. Student là deep copy độc lập, lấy patch tokens trước ln_post/projection, bỏ CLS. Original CLIP frozen/eval, unused student ln_post/proj frozen.
- Query `model(sketches)` trả `g`, `q`, `mu_i`, `mu_t`; không nhận label/photo/teacher hint. Predictor một block cross-attention, width256/4heads; main retrieval là **mu_i**, gallery dùng `encode_photo()` và cùng joint checkpoint. Không tự chọn/fuse q.
- C_I (3 visual tokens) và C_T (4 text tokens) chỉ có một parameter owner; predictor nhận đúng các parameter đó. Bank khởi tạo bằng prefix `a photo of a`, kiểm tra T0 parity atol/rtol1e-6. Một class-ID buffer; checkpoint khác class IDs/token IDs bị từ chối trước load.
- `optimizer_parameter_groups()` chỉ trả groups, không tạo optimizer: student LR1e-5, predictor/pool1e-4; matrix WD1e-2, bias/LN và vector WD0; prompts LR/WD1e-4. Mỗi trainable parameter xuất hiện đúng một lần; original CLIP không nằm trong groups. Hiện chỉ nhận CLIP float32; chưa chứng nhận mixed precision.
- `src/spica/data/coupled_views.py:region_pair`: promote đúng region helper đã thiết kế, seed4242+update/view1, local RNG, clean clone + một mask, giữ metadata/blank status. Không import/call thinning.
- `src/spica/coupled_predictive_losses.py:coupled_region_loss`: concatenate hai query views một forward; live unique photo bank, original photo reference và complete train text bank mỗi loại encode một lần. Bản tiny test dùng2 classes; production phải cấp đúng complete84-class train map, không suy tiny gate đã kiểm dataset.

API loss:

```python
terms = coupled_region_loss(
    model, clean, corrupted, unique_photos,
    positive_indices, negative_indices, query_labels, photo_labels, photo_ids,
    lambda_sig=0.0,  # control: tuyệt đối không gọi regularizer
)
loss = terms["total"]  # vẫn giữ autograd graph
```

Photos được truyền dưới dạng bank unique IDs, positive indices `[B]`, negatives `[B,K]`; labels/indices long và cùng device. Kiểm tra unknown classes, positive khác class, negative cùng class (dù khác instance), index ngoài range, duplicate photo IDs và nonfinite inputs. Không phát hiện được annotation sai nhưng tự-consistent; không giải quyết noisy/missing labels.

## Loss và SIGReg

Hệ số mỗi view: rank_i/CE_t **0.5**, rank_pool/CE_pool **0.125**, align_i/align_t **0.025**. Photo/text anchors **0.5 mỗi loại, một lần** ngoài view loop. Scalar-one test tổng3.6 và autograd từng hệ số đúng. Rank/CE giữ live photo/text gradients; positive cosine targets và original references detached.

`src/spica/models/sigreg.py:SIGReg` là local implementation **formula-equivalent với official MINIMAL**, pin `c293d291ca87cd4fddee9d3fffe4e914c7272052`; [source identity](sigreg_source_identity_2026-09-08.json) ghi URLs/SHA và license inconsistency. Không substitute legacy standardized `SignatureRegularizer`.

- Defaults17 nodes trên[0,3],256 Gaussian unit projection columns, symmetry-weighted trapezoid quadrature và Epps–Pulley statistic nhân B. Không center/standardize/normalize latent; float32 objective, autocast disabled.
- Private CPU projection RNG được serialize, không dùng global RNG. **Không bit-identical official CUDA-random sequence**. Fixed-direction objective/gradient parity tolerance atol2e-6, rtol0.
- Loss gọi SIGReg riêng trên `g_clean[B,D]` và `g_corrupted[B,D]`, resample directions mỗi call, rồi mean50–50; không flatten64 views/tokens thành independent samples. Module riêng cũng nhận `[V,B,D]` và shared directions giữa views trong một call; loss không dùng route đó.
- `lambda_sig` bắt buộc explicit. Zero không gọi callable (kể cả nếu được truyền); positive cần supplied implementation và finite scalar từng view. Tiny test λ0.2 chỉ là kiểm wiring, **không chọn hệ số experiment**. Callable injection không tự chứng minh implementation là faithful.
- Giữ frequency buffers float32; `.half()` module bị từ chối. Chưa distributed/trainer integration, chưa diagnostic λ.

## Kiểm chứng và giới hạn

Parent: **23 targeted tests PASS**, **290 full-suite tests PASS**, Ruff PASS. Independent reviewer chạy lại23 tests PASS, không còn blocker trong scope CPU. Evidence local: `outputs/coupled_predictive_cpu_gate_20260908T080000Z/` (`targeted.log`, `pytest.log`, `ruff.log`, `verification.json`).

Gates dùng **tiny real OpenCLIP architecture, random weights**, synthetic8×8 images; không mock toàn bộ CLIP forward, nhưng cũng không pretrained ViT-B/32. Bao gồm region→model→loss backward, frozen teacher/storage separation, nonzero finite gradients, photo live/detached-target edges, view isolation, single-encode counters, unique optimizer ownership và nontrivial state roundtrip. Query-context effect chỉ tại initialization, không phải trained anti-shortcut evidence. Không optimizer step trong các V1 tests; full regression suite có các synthetic smoke checks lịch sử.

Đã sửa trước final gate: test group name cũ; heads theo device teacher; class/token checkpoint guard; scalar coefficient và live-target gradient checks; SIGReg empty-view/frequency dtype guards. Source receipt test không còn phụ thuộc bắt buộc vào ignored outputs: tracked pin luôn kiểm, raw-source hash test **skip rõ ràng nếu local bundle vắng**, không tự download. Tại máy này bundle có đủ và hashes PASS.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
git diff --check
```

**Chưa có:** full pretrained adapter/memory/no-update gate, actual-data batching, trainer/checkpoint lifecycle/retrieval integration, optimizer update parity, calibrated SIGReg λ, locked diagnostic/scheduler/promotion details. Cần authorization tiếp theo trước các bước execution. Không launch V1/S0–S2, không W&B run mới trong milestone này, không nâng dependencies, không push; historical artifacts và `outputsnewgate/` giữ nguyên. Thinning tiếp tục parked.
