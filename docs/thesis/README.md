# Bộ tài liệu viết đồ án SPICA — 13/09/2026

Đây là **bộ nguyên liệu đã tổ chức và bản nháp kỹ thuật**, không phải luận văn đã hoàn thiện theo mẫu trường. Không điền thay tên trường, tác giả, GVHD, tuyên bố đóng góp hay yêu cầu đạo đức của cơ sở đào tạo.

## Bắt đầu ở đâu?

1. [Đề cương và thông điệp](01_outline.md): bố cục chương, câu hỏi nghiên cứu, tóm tắt nháp.
2. [Phương pháp và giao thức](02_method.md): kiến trúc MP-Q, loss, dữ liệu, metric và sampling.
3. [Kết quả chính](03_results.md): bảng số cuối official ba bộ, hình train-progress theo %, không chọn peak.
4. [Thực nghiệm bổ sung và hạn chế](04_discussion.md): diễn giải đúng bằng chứng, lịch sử ablation và điểm cần tránh.
5. [Tái lập và bàn giao](05_reproducibility.md): artifacts, kiểm tra, caveats, checklist trước nộp.
6. [Tài liệu tham khảo](06_references.md) + [BibTeX](references.bib): ba nguồn nền tảng đã đối chiếu metadata nguồn chính.

## Tài sản dùng ngay

- `tables/final_metrics.csv`: clean, masked macro và ba mức xóa; metric **0–1**.
- `tables/conditions.csv`: 10 điều kiện/dataset, gồm mask-status; không loại hàng thất bại đạt mức xóa.
- `tables/progress.csv`: 15 điểm test, có `progress_percent` tính từ bước thực tế. Đây là file cục bộ, **không sửa history W&B cũ**.
- `tables/datasets.csv`: số lớp/ảnh và ngân sách train.
- `tables/provenance.csv`: run → HEAD/source → config → checkpoint → summary SHA256.
- `training_configs.json`: cấu hình kỹ thuật trích từ các run thực tế, không phải defaults mới.
- `figures/*.png`: hình 200dpi cho Word; `*.svg`: vector cho in ấn/LaTeX.
- `audit.json`: kiểm tra archived source/checkpoint/NPZ và trung bình metric đã lưu; không replay model/full-gallery sort.
- `online_check.json`: kiểm tra read-only W&B cuối run:15rows/90scalars khớp local. Có thể chạy lại `.venv/bin/python docs/thesis/check_online.py` với credentials W&B hiện có; không ghi remote.

Hình `progress` dùng trục X %, các hình metric dùng trục Y %. Dữ liệu CSV giữ số gốc để tự định dạng trong Excel/Word. Ví dụ0.1547865894 =15.47865894%, không phải0.1548%.

## Trạng thái phải giữ khi viết

- **Sketchy104/21:**4446updates và5test đã có; `FAILED_NO_RETRY` tại guard source cuối run do script audit của assistant được đặt nhầm vào source. Không gọi toàn campaign VERIFIED.
- **TU220/30, QuickDraw80/30:**1189/18229updates,5test mỗi bộ, runner exit0; raw `TRAIN_AND_PERIODIC_OFFICIAL_EVAL_FINISHED_UNVERIFIED` giữ nguyên.
- Các bằng chứng kiểm tra trong bộ tài liệu là phạm vi hẹp, không tự nâng raw status. Một seed training42; không có kết luận ý nghĩa thống kê đa seed.
- Code thêm `progress_percent` ở commit `5c8bfd5` được sửa **sau** các run; không phải source đã tạo kết quả.

## Dựng lại bảng/hình

Từ repo root, dùng môi trường có sẵn, không cài thêm dependency:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python docs/thesis/build_assets.py
```

Lệnh đọc artifacts local, kiểm tra và ghi lại **chỉ các tài sản sinh tự động của bộ tài liệu**, không train, không GPU, không sửa raw results. Ổ SDA cần mount vì các báo cáo lịch sử có symlink. Không chạy trình sinh này trong lúc source đang được đóng băng cho một campaign mới.

**Chưa bao gồm:** bản Word/LaTeX theo mẫu trường, toàn văn tổng quan tài liệu, tái hiện SOTA, thống kê đa seed, hoặc phụ lục qualitative truy xuất mới. Các phần cần người viết bổ sung được đánh dấu trong checklist, không ngụy tạo kết quả cho đủ chương.
