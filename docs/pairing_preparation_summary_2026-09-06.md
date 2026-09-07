# Pairing A/B — bàn giao chuẩn bị CPU (2026-09-06)

## Kết luận

**Hoàn tất ba việc chuẩn bị: mapping, protocol/config A/B và CPU integration gate. Chưa chạy pilot GPU/1800 steps; chưa có kết quả mAP A/B.**

Ba worker `cx/gpt-5.6-luna`, thinking `max` làm song song theo phạm vi file độc lập, sau đó reviewer độc lập và parent kiểm tra/tích hợp lại. HEAD ban đầu `efa4bd1d489d059d2a0dd4086a181ec96a784f2f`; không commit/push, nâng dependency, thay ảnh hoặc sửa historical artifacts.

| Việc | Kết quả kiểm chứng cuối |
|---|---|
| Mapping | 46,624 cặp pseudo-train, 8,400 photo gốc; loader kiểm tra scope, class, quy ước tên file, đường dẫn và SHA256 |
| Đối chiếu nguồn | 55,024/55,024 file ảnh được dùng khớp SHA256 với nguồn chỉ đọc trong Downloads; không copy ảnh |
| Config A/B | Hydra compose thật và `_validate()` đạt; campaign riêng, cùng seed42/split3407, primary bắt buộc step1800 |
| Sampler dữ liệu thật | Batch 32 query/negative giống nhau giữa A/B; 32/32 positive khác nhau; B đúng mapping |
| CPU integration | Hai bước mỗi arm qua trainer thực, model CLIP nhỏ khởi tạo ngẫu nhiên đầy đủ visual/text; `CPU_SYNTHETIC_SMOKE_PASS` |
| Tests | **204 passed in 3.32s**; Ruff PASS; `git diff --check` PASS |

## 1. Mapping và dữ liệu

Manifest được hai config sử dụng:

- `outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.json`
- SHA256: `545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c`
- Source audit: `outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.source_audit.json`
- SHA256 source audit: `4b219609ec722603557271ed9cf63c02113a04363eea5c8dfbe1d760834abf84`

Photo target chỉ lấy từ `256x256/photo/tx_000000000000_ready/<class>/<id>.jpg`, không tìm tùy ý trên gallery đã gộp. Bỏ hậu tố `-integer` cuối sketch stem để khớp photo stem; loader độc lập kiểm tra lại quy tắc này, không chỉ tin builder.

Pseudo-train: **46,624 sketches / 58,950 photos**, trong đó **8,400 photo gốc** vừa bằng tập photo của mapping. Pseudo-validation: **10,963 sketches / 13,999 photos**; không đưa vào training mapping. Official unseen không được đánh giá.

Nguồn `/home/oslamelon/Downloads/Research/ZS BIR dataset/Sketchy` chỉ được đọc để kiểm tra SHA; không có ảnh hoặc manifest dataset nào bị sửa/copy. Metadata mới nằm trong `outputs/`, không trong dataset gốc.

Parent đã xem 6 cặp khác lớp trong `outputs/pairing_preparation_validation_20260906_222204/canonical_pair_samples.png` (table, teapot, harp, wading_bird, rocket, chicken): không thấy mismatch hiển nhiên về đối tượng ở các mẫu này. Đây chỉ là spot-check, **không phải xác minh annotation instance độc lập trên toàn bộ dataset**. Trạng thái mapping vẫn là `filename_convention_canonical`.

## 2. Protocol A/B đã chuẩn bị

| Thành phần | A | B |
|---|---|---|
| Config | `configs/experiments/pairing_pilot_A.yaml` | `configs/experiments/pairing_pilot_B.yaml` |
| Positive | Ngẫu nhiên cùng lớp trong pool photo gốc | Photo gốc được mapping tới sketch |
| Pool positive | Cùng 8,400 photo gốc | Cùng 8,400 photo gốc |
| Negative | Cùng full pseudo-train photo pool | Như A |
| Query/gallery đánh giá | Cùng pseudo-validation, relevance theo class | Như A |

Campaign: `frozen_prompt_pairing_pilot_2026-09-06`. Toàn bộ weights CLIP đóng băng, gồm LayerNorm, visual/text projections, text tower và logit scale. **Chỉ sketch/photo prompts được cập nhật**. Hard text CE chỉ là supervision của query, không phải text input khi inference. Giữ softplus rank loss hiện hữu; không thay bằng helper JEPA vốn detach photo targets. Không patch-shuffle loss; giữ DataLoader shuffle.

Đây là **same-class canonical versus paired canonical**, không phải so với control historical lấy positives từ merged gallery. Cùng pool giúp tránh trộn tác động của correspondence với khác biệt xử lý ảnh giữa nhánh gốc và extended.

Primary bắt buộc from-scratch, seed42, pseudo split3407, step1800; không resume smoke thành primary. Chọn đúng checkpoint fixed-step, không peak candidate so với fixed control. Trạng thái một run primary là `PRIMARY_FIXED_STEP_UNCOMPARED`, không tự gắn nhãn MATCHED trước khi đối chiếu hai arm. Pairing SHA/treatment nằm trong lineage/cache/checkpoint.

## 3. Bằng chứng CPU cuối cùng

Thư mục: `outputs/pairing_preparation_validation_20260906_222204/`.

- `pytest_final.log`: 204 tests PASS.
- `ruff_final.log`: PASS.
- `real_data_preflight.json`, `resolved_A.yaml`, `resolved_B.yaml`: compose/validator thật, strict loader 46,624 records, batch sampling thật. Preflight sampler dùng `num_workers=0` và transform CPU nhỏ; **không phải đo production workers hoặc preprocessing CLIP pretrained**.
- `check_real_data.py`: lệnh tái lập preflight chỉ đọc, không nạp model.
- `integration_final/gate_summary.json`: `CPU_SYNTHETIC_SMOKE_PASS`, 2 updates/arm.
- `integration_final/pairing_pilot_A/`, `integration_final/pairing_pilot_B/`: raw histories, checkpoints, configs và provenance của synthetic smoke.

Gate thực thi production trainer/loaders/loss/backward/optimizer/evaluation/checkpoints. Nó xác nhận riêng từng prompt có gradient hữu hạn, khác zero và cập nhật; toàn bộ tiny CLIP state giữ nguyên; B dùng đúng pair, A có positive khác B trong khi query/label/negative/khởi tạo/text bank/gallery identity khớp. SHA256 checkpoint được tính lại từ file, không chỉ đọc field đã ghi.

Model trong gate là **CLIP nhỏ ngẫu nhiên**, không phải pretrained ViT-B/32. Production artifact ghi `CPU_SMOKE`; chỉ gate summary biết fixture là synthetic. Không suy mAP hay khả năng vượt SketchLVM từ metric fixture.

### Các lỗi chuẩn bị đã sửa trước khi chốt

- Gate ngắn không còn giả cấu hình 1800 cho validator: dùng smoke riêng với actual horizon2.
- Config trỏ đúng manifest thực, thay đường dẫn kế hoạch chưa tồn tại.
- Loader chặn thay target bằng ảnh khác cùng lớp dù hash hợp lệ.
- Không chỉ kiểm tra visual weights rồi tuyên bố toàn CLIP bất biến.
- Fixture có nhiều canonical positives/class để A/B thực sự khác nhau.
- Hydra package directive được đưa lên dòng đầu: trước đó merge YAML thủ công che mất lỗi `+experiments=...` không áp dụng vào root. Unit tests và gate hiện dùng **Hydra compose thật**.

Các log/thư mục lần kiểm tra trước vẫn được giữ; dùng hậu tố `final` ở trên để xác định bằng chứng kết thúc. Source hash của smoke ghi snapshot lúc chạy; tài liệu bàn giao bổ sung sau đó không được gán ngược vào run.

## Tái lập và bước tiếp theo

CPU gate, chọn output chưa tồn tại:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/check_pairing_pilot_integration_cpu.py \
  --output-root /tmp/spica-pairing-gate-NEW
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
```

**Chỉ sau khi user duyệt GPU pilot**, chạy tuần tự hai lệnh dưới đây; chúng chưa được thực thi:

```bash
PYTHONPATH=src .venv/bin/python -m spica.train_frozen_prompt \
  --config-name train_frozen_prompt +experiments=pairing_pilot_A device=cuda
PYTHONPATH=src .venv/bin/python -m spica.train_frozen_prompt \
  --config-name train_frozen_prompt +experiments=pairing_pilot_B device=cuda
```

Chưa kiểm chứng pretrained-backbone/GPU preflight, runtime hoặc mAP thực. Chưa chạy masking, relational KD, text-context matrix, official unseen, multi-seed hay lambda search.

Chi tiết từng nhánh: [mapping](pairing_manifest_preparation.md), [protocol](pairing_pilot_protocol.md), [CPU gate](pairing_cpu_gate.md).
