# SPICA — bắt đầu ở đây

## Tiến độ mới nhất — semantic baseline / predictive design (2026-09-08)

- C/M1800 và3600 đã hoàn tất, online W&B/checkpoint verification đã có: [kết quả3600](docs/masked_view_3600_results_2026-09-07.md). Kết quả có trade-off; không khẳng định baseline tối ưu hoặc masking thắng mọi metric.
- Progress checkpoint: `4cd4a15`; **không phải source snapshot đã dùng để train**. Raw outputs/tensors và `outputsnewgate/` giữ local, không reset/xóa chúng.
- Đọc [Bước 0 + thiết kế Bước 1](docs/semantic_baseline_step0_step1_protocol.md) và [audit identities](docs/semantic_baseline_step0_audit.json) trước khi tiếp tục. Protocol nội bộ đã chốt; SeCo exact reproduction còn BLOCKED. S0/S1/S2 đã implement và qua CPU/real-CLIP no-update gates; xem [readiness](docs/semantic_text_step1_readiness_2026-09-07.md).
- User đã duyệt: S2 cosine text-anchor classmean, lambda=1 cố định; promotion tại best_clean phải tăng clean prefix AP200, P200/full mAP không giảm so S0. Không tự đổi metric selection hoặc gọi lambda đã calibrated.
- User đã duyệt restart từ đầu sau reboot làm ngắt S0@3000 ở root `outputs/semantic_text_execution_20260907T160000Z/` (giữ nguyên, runtime LAUNCHING cũ bị stale). Root mới **`outputs/semantic_text_execution_20260908T011006Z/` đã VERIFIED S0/S1/S2,3600 updates/arm**: [kết quả](docs/semantic_text_step1_results_2026-09-08.md), [raw-precision summary](docs/semantic_text_step1_results_2026-09-08.json), [W&B tổng hợp](https://wandb.ai/a-cctest05187-erd/spica/runs/1p29dtr0). S1/S2 FAIL promotion so S0; giữ S0. 9 CUDA selected replays delta0, 361historyrows/arm,9selectedartifactaliases SHA-verified. P200 scalar replay verified; không serialize per-query P200 arrays/independent second evaluator. Không launch lại. Training source HEAD `6a4d8fc`, SHA256 `ca4671cc50010a654f217ddb46714a71063ba4e63a25d6cdd634c9298873d9d1`; docs/commit sau khi hoàn tất không phải training snapshot.
- User đã duyệt **implement V1 model/loss + CPU gates, chưa train**: [CPU readiness](docs/coupled_predictive_v1_cpu_readiness_2026-09-08.md), [region-first protocol](docs/designs/coupled_predictive_region_v1/region_first_protocol.md), [architecture](docs/designs/coupled_predictive_region_v1/architecture.md). Separate fine-tuned sketch encoder, shared CLIP–predictor prompts, region deletion + clean/corrupted50–50; accepted loss/optimizer groups đã implement. Parent290tests/23targeted + independent23tests PASS. SIGReg formula/gradient parity với pinned official MINIMAL; private CPU RNG không bit-identical official CUDA sequence. **Real pretrained/real-data GPU preflight đã được user duyệt và PASS**: [kết quả](docs/coupled_predictive_v1_preflight_2026-09-08.md), root `outputs/coupled_predictive_preflight_20260908T101310Z/`. B32/64views no-update/noSIG,173trainable gradient tensors hữu hạn/nonzero;482state entries unchanged; peak allocated7.335GiB/reserved8.389GiB (gồm edge checks, chưa optimizer states). Source archive550files SHA `38b0cd249cb8e0581585c11bfffd1e00e71165ce8c1930a8576de0eec37a3bcb`; parent293tests + independent archive/mask reconstruction PASS. Lambda diagnostic, scheduler/promotion lock và trainer/training còn pending; không production-training readiness claim. Thinning parked, không activate. Không tự launch thêm GPU/train V1.
- Local commit sau mỗi milestone; không push/official unseen/multi-seed/lambda search tự động. Postprocessing verifier có thể giả định current source == training source hoặc yêu cầu per-query P200 arrays không có trong historical serialization: kiểm tra giả định, dùng archived source/replay, không retrain hay sửa raw evidence để ép PASS.

## Corrected alignment campaign đã được kiểm chứng

Đọc [bản bàn giao ngắn](docs/corrected_alignment_handoff.md) trước. Bản này dẫn tới kết quả, raw evidence và review bundle của lần kiểm chứng **2026-09-06**; không nhầm với báo cáo historical 2026-09-05.

- Pilot: seed42, pseudo split3407, R/MD/MS, 1800 steps/arm, covariance=0.
- Kết quả: **R > MD > MS**; negative result cho cấu hình này, không promote mainline.
- Calibration CLIP thật VALID; cả R–MD/R–MS/MD–MS MATCHED theo **corrected_v2**.
- Snapshot được tiếp quản: `82c6461`; training mới dùng snapshot đó cộng bản sửa ba chỗ đọc mAP trong `src/spica/train_alignment.py`. Source archive/hash nằm trong evidence; commit bàn giao hiện tại không phải training snapshot.

## Quy tắc cho agent tiếp theo

- Đọc HEAD và `git status` trước khi sửa. Không ghi đè/xóa/di chuyển historical artifacts.
- Kết luận số liệu phải truy được run → config/source → checkpoint/step → raw metric. Không lấy peak candidate so với fixed-step control.
- Không coi thiếu checkpoints trên GitHub là chưa chạy: các tensor lớn chỉ lưu local; bundle chứa raw histories, metrics và đường dẫn/SHA256 checkpoint.
- Không đổi calibration của run đã train, không gọi historical/INCOMPLETE/UNVERIFIED comparison là MATCHED.
- Không tự chạy lại GPU campaign, official unseen, covariance, 5400 steps, multi-seed hoặc lambda search nếu chưa được yêu cầu mới. Không tự push, thuê GPU hay kill process.
- Dùng môi trường dự án `.envrc`/`.venv` và `uv.lock` hiện hữu; không nâng dependencies. Lệnh tests/replay có trong báo cáo kiểm chứng.
- Source hash training bao gồm source/docs không bị Git ignore: thêm tài liệu hoặc commit mới có thể đổi hash, không tự suy ra run cũ phải retrain. Dùng source snapshot đã lưu để phân biệt thay đổi training và postprocessing.
