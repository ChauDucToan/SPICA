# SPICA — bắt đầu ở đây

## Tiến độ mới nhất — semantic baseline (2026-09-07)

- C/M1800 và3600 đã hoàn tất, online W&B/checkpoint verification đã có: [kết quả3600](docs/masked_view_3600_results_2026-09-07.md). Kết quả có trade-off; không khẳng định baseline tối ưu hoặc masking thắng mọi metric.
- Progress checkpoint: `4cd4a15`; **không phải source snapshot đã dùng để train**. Raw outputs/tensors và `outputsnewgate/` giữ local, không reset/xóa chúng.
- Đọc [Bước 0 + thiết kế Bước 1](docs/semantic_baseline_step0_step1_protocol.md) và [audit identities](docs/semantic_baseline_step0_audit.json) trước khi tiếp tục. Protocol nội bộ đã chốt; SeCo exact reproduction còn BLOCKED. S0/S1/S2 chưa implement/train.
- User đã duyệt: S2 cosine text-anchor classmean, lambda=1 cố định; promotion tại best_clean phải tăng clean prefix AP200, P200/full mAP không giảm so S0. Không tự đổi metric selection hoặc gọi lambda đã calibrated.
- User yêu cầu local commit sau mỗi phần công việc hoàn tất. Không tự push; commit/design không tự cấp phép training/GPU diagnostic/official unseen/multi-seed. Bước tiếp theo là xin quyền implementation S0/S1/S2 + CPU gates, không launch campaign.

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
