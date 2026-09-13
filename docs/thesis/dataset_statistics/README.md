# Thống kê dataset cho đồ án

Đã đo **416.452 sketch** của official train + test, không lấy mẫu. Photo chỉ được đếm theo manifest, không dùng để đo nét. Ảnh nguồn không bị sửa; không dùng GPU, model hay W&B.

## Ba bảng kết quả

**[Mở 3 bảng: Sketchy, TU-Berlin, QuickDraw](tables/dataset_statistics.md)**

Mỗi bảng tách train/test và có trung bình, trung vị, độ lệch chuẩn tổng thể cho:
- Số pixel mực ở224×224.
- Tỷ lệ pixel mực trên50.176pixel (%).
- **Proxy độ dày raster**: trung bình đường kính EDT ở các điểm cực đại cục bộ của từng sketch.

**Không diễn giải proxy như độ dày nét chính xác:** nét đơn rộng1pixel cho proxy2px; nét3pixel cho proxy4px. Junction, lưới pixel và ngưỡng mực ảnh hưởng phép đo. TU/QuickDraw có proxy sát2px không có nghĩa mọi nét gốc đều rộng2px. Xem [phương pháp](../dataset_statistics_method.md).

[CSV thống kê](tables/summary.csv) · [JSON đầy đủ](tables/summary.json) · [Tất cả category](tables/class_counts_all.csv) · [Category được vẽ](tables/class_counts_selected.csv)

## 12 Log-scale Bar Charts

Trục X: category. Trục Y: **số ảnh trên thang log**, không phải giá trị log(count) đã thay cho count. Cột xanh: top10; cam: bottom5; số lượng thực ghi trên cột. Tất cả15category được sắp giảm dần theo count.

Nếu bằng count, thứ tự tên category quyết định lựa chọn. Bottom5 loại những category đã chọn ở top10 để không trùng. Các tập đồng đều như TU sketch có thể không có chênh lệch thực giữa top và bottom; màu chỉ thể hiện nhóm được chọn, không hàm ý mất cân bằng.

| Dataset | Train sketch | Train photo | Test sketch | Test photo |
|---|---|---|---|---|
| Sketchy | [PNG](figures/sketchy_104_21_train_sketch_counts.png) | [PNG](figures/sketchy_104_21_train_photo_counts.png) | [PNG](figures/sketchy_104_21_test_sketch_counts.png) | [PNG](figures/sketchy_104_21_test_photo_counts.png) |
| TU-Berlin | [PNG](figures/tuberlin_220_30_train_sketch_counts.png) | [PNG](figures/tuberlin_220_30_train_photo_counts.png) | [PNG](figures/tuberlin_220_30_test_sketch_counts.png) | [PNG](figures/tuberlin_220_30_test_photo_counts.png) |
| QuickDraw | [PNG](figures/quickdraw_80_30_train_sketch_counts.png) | [PNG](figures/quickdraw_80_30_train_photo_counts.png) | [PNG](figures/quickdraw_80_30_test_sketch_counts.png) | [PNG](figures/quickdraw_80_30_test_photo_counts.png) |

Mỗi hình có bản SVG cùng tên trong `figures/` để dùng vector trong đồ án.

## Kiểm tra và phạm vi

- Counts/SHA manifest được loader official kiểm tra trước scan.
- Mỗi ảnh lưu SHA256 file nguồn và RGB sau preprocessing; không loại ảnh trắng âm thầm (lượt này không có sketch trắng theo ngưỡng).
- Parent đọc lại toàn bộ CSV, dùng `statistics` chuẩn Python để đối chiếu mean/median/population-SD: sai số lớn nhất **2,28e−13**.
- Parent đếm độc lập cả12manifest và kiểm tra12lựa chọn top10/bottom5/sort.
- Replay12ảnh (đầu/cuối mỗi split/dataset) xác nhận hash RGB và mask threshold khớp transform normalized của dự án. Không replay độc lập toàn bộ416.452phép EDT.
- [Receipt producer](scan_receipt.json) · [Receipt parent](parent_receipt.json).

Raw CSV khoảng112MiB và bằng chứng thực thi giữ ở `outputs/dataset_statistics_20260913/`, không đưa vào Git. Script: `docs/thesis/dataset_stats.py`; synthetic checks chạy bằng `--self-check`. Đây là thống kê hình học dataset, không phải bằng chứng giải thích nhân quả về điểm retrieval hay chứng minh nét vector thật.
