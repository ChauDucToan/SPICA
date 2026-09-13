# 5. Tái lập, bằng chứng và checklist

## 5.1 Nguồn kết quả

| Dataset | Campaign | Run W&B | Fixed step |
|---|---|---|---:|
| Sketchy104/21 | `outputs/percent20_official_execution_20260912T133000Z/` | d1d32lfm |4446|
| TU220/30 | `outputs/percent20_remaining_execution_20260912/` | yifk5qi3 |1189|
| QuickDraw80/30 | cùng root TU | aevluutb |18229|

`tables/provenance.csv` chứa HEAD/source/config/checkpoint/summary SHA đầy đủ. Mỗi run có `resolved_config.json`, `checkpoint_latest.pt/.json`, `source_snapshot/files/`, `test_metrics.jsonl`, `test_evaluations.jsonl`, `test/step_N/summary.json`. Dùng archived source để mô tả source training, không lấy HEAD hiện tại sau viết tài liệu.

## 5.2 Audit đã làm trong lượt chuẩn bị

`build_assets.py` kiểm tra:
- Step trong resolved config/checkpoint/summary, source hash config/summary và hash checkpoint cuối khớp nhau. Không đối chiếu toàn bộ cấu hình từng trường với summary; SHA config được ghi riêng để truy xuất.
- Từng file archive source và aggregate khớp source hash training.
- Saved model-state-before/after evaluation bằng nhau và khớp checkpoint metadata; predictor forward count0. Đây là kiểm tra **record đã lưu**, không một forward mới.
- Đúng5mốcceil20%, cả6scalars cuối khớp local history.
-30condition summaries khớp hash khai báo trong summary tổng, và NPZ khớp hash trong condition metadata đã lưu;5mảng metric mỗi condition được lấy trung bình FP64 rồi so summary, tolerance2e−6.
- Masked macro bằng trung bình9condition với tolerance1e−12.

`audit.json` ghi số query-condition và sai số. Không thay đổi tolerance do thấy lỗi; nếu assert fail phải giữ log, không gọi full PASS. Không gọi việc lấy mean của saved AP là independent recomputation của AP/rankings.

`online_check.json`: read-only API cuối run xác nhận5historyrows/dataset,90metric scalar comparisons tổng, tolerance1e−12. Đây là kiểm tra nhất quán scalar-history, không phải kết quả đánh giá độc lập. Không tải model artifacts, không resume run, không ghi summary/state từ ngoài. QuickDraw và TU đã `finished`; Sketchy vẫn `failed` đúng raw incident.

## 5.3 Môi trường và lưu trữ

- Repo dùng `.envrc`, `.venv`, `uv.lock` có sẵn; không nâng dependency khi tái kiểm tra.
- GPU training: RTX5070Ti khoảng16GB, NixOS. Đây không phải latency benchmark được chuẩn hóa.
- Dataset/cache CLIP và các campaign/gate mới giữ NVMe.
-34root lịch sử được chuyển sang `sda/DoAn/spica_archive_20260912/outputs/` và giữ đường dẫn cũ qua symlink; một root giữ102tracked files dạng thường trên NVMe để không phá Git layout.
- Cần mount `/dev/sda1` tại `sda/` trước khi đọc symlink lịch sử. Không coi thư mục symlink không truy cập được là bằng chứng chưa từng train.
- Đã xóa88checkpoint NVMe và14checkpoint SDA được user duyệt. Receipts: `outputs/checkpoint_cleanup_review_20260912/deletion_receipt.json`, `outputs/storage_migration_review_20260912/final_receipt.json`.
- Không đóng gói hàng chục GB weights/datasets vào Git/Word; chỉ trích đường dẫn và SHA. Sao lưu ngoài máy phải xem quyền truy cập/dung lượng/giấy phép dữ liệu riêng.

## 5.4 Tái dựng báo cáo, không tái train

```bash
# Từ repo root; chỉ CPU, đọc artifacts và tạo lại bảng/hình.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python docs/thesis/build_assets.py
```

Code này ghi lại files generated dưới `docs/thesis/`; không chạy khi có campaign đang freeze nonignored source/docs. Không chạy lệnh launch cũ vào root đã tồn tại. Gate cũ ràng buộc source, nên code logging mới hoặc commit tài liệu không làm kết quả cũ vô hiệu, nhưng **không tự cho phép dùng gate cũ để train source mới**. Muốn train mới cần authorization, fresh root và gate phù hợp.

Compact checkpoint gồm trainable tensors/optimizer/scheduler/RNG/cache identity; không phải full standalone CLIP weight bundle. Reconstruction cần đúng CLIP cache và seed trước khi dựng backbone, kể cả frozen random prompt. Không có exact worker-stream resume.

## 5.5 Checklist trước khi nộp

- [ ] Điền trường/khoa/tác giả/MSSV/GVHD, bố cục và quy định trình bày của trường.
- [ ] GVHD duyệt tên đề tài, phạm vi và mức tuyên bố đóng góp/tính mới.
- [ ] Đọc toàn văn paper nền tảng; thêm nguồn gốc dataset và giấy phép, kiểm tra đúng spelling tên tác giả/venue theo template.
- [ ] Viết tổng quan liên quan thành lập luận riêng, không dùng abstract như thể đã tái hiện paper.
- [ ] Mọi bảng ghi split, final step, seed, cutoff/mẫu số metric, clean/masked, đơn vị%.
- [ ] Không dùng peak candidate so fixed control; không trộn pseudo84/20 với official104/21.
- [ ] Hình có caption và nhắc Sketchy source-guard incident nếu dùng chung ba bộ.
- [ ] Appendix nêu raw failure/UNVERIFIED, phạm vi audit, checkpoint đã xóa, test đã được quan sát.
- [ ] Kiểm tra số liệu cuối từ CSV, không chép số làm tròn trên tooltip W&B.
- [ ] Kiểm tra link local/online và backup những artifacts cần cho buổi bảo vệ; không công bố credentials/path riêng không cần thiết.
- [ ] Nếu cần qualitative retrieval panel hoặc demo trực tiếp, chọn quy tắc chọn mẫu trước; chưa tạo thêm evaluation trong bộ này.
- [ ] Rà soát chính sách sử dụng AI của trường và khai báo hỗ trợ soạn thảo nếu được yêu cầu.

## 5.6 Sườn trình bày bảo vệ (10 slide)

1.Bài toán/phạm vi →2.domain gap/mất mực →3.related work →4.train/inference MP-Q →5.loss/sampling →6.protocol/metrics →7.final clean/masked →8.ablation có giới hạn →9.failures/limitations →10.kết luận/hướng tiếp theo.

Câu hỏi dự kiến: Vì sao q inference nhưng MP ở μ_I? Vì sao QuickDraw thấp? mAP200 dùng mẫu số nào? Có leakage không? Vì sao Sketchy failed? Tái lập được đến đâu sau dọn checkpoint? Trả lời bằng bằng chứng và giới hạn ở các mục trên, không đoán cơ chế như kết luận chắc chắn.
