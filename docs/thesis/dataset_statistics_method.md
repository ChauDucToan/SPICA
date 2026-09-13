# Phương pháp thống kê dataset

## Phạm vi

Đo toàn bộ **sketch** của official train và test: Sketchy104/21, TU-Berlin220/30, QuickDraw80/30; tổng416.452ảnh. **Photo chỉ đếm category theo manifest**, không đo “mực” trên ảnh tự nhiên. Không train, không GPU, không đọc checkpoint/pretrained, không W&B/network, không sửa ảnh nguồn, không cài dependency.

`load_benchmark_protocol` kiểm tra config/manifest/class-map SHA, counts, tên lớp, đường dẫn tồn tại và overlap giữa các split trước khi mở ảnh. Receipt lưu protocol identity SHA. Mỗi ảnh có path, class/split, kích thước gốc, SHA256 file nguồn và SHA256 RGB sau transform. CSV per-image nằm ở ignored root `outputs/dataset_statistics_20260913/`.

## Preprocessing và pixel mực

Dùng RGB Pillow và hai phép hình học của `build_clip_eval_transform(224)`: resize cạnh ngắn224 bằng bicubic, sau đó center-crop224×224. Không thêm xóa nét, crop ngẫu nhiên hay augmentation. Kích thước gốc chỉ là metadata; **các số đo px đều trên đầu vào224×224**, không phải ảnh gốc.

Một pixel là mực khi mean(R,G,B)/255 <0.9, cùng quy tắc threshold của masking trong dự án. Với RGB uint8, tương đương R+G+B <688.5 (tức ≤688). Không có pixel nằm chính xác trên ngưỡng688.5. Worker kiểm tra mẫu đầu của nó với đường transform normalized rồi denormalize; parent kiểm tra thêm đầu/cuối mỗi dataset/split.

- `ink_pixels`: số pixel thỏa điều kiện.
- `ink_fraction_percent`: 100×ink_pixels/50.176.
- Ảnh trắng được giữ và nhận0ở hai đại lượng này. Lượt scan không có sketch trắng theo ngưỡng.

## Proxy độ dày — không phải độ dày stroke chính xác

Đại lượng `raster_line_width_estimate_px` được đo như sau:

1. Nhị phân hóa mực và thêm viền0rộng1pixel để foreground chạm biên vẫn có khoảng cách hữu hạn.
2. Tính Euclidean distance transform bằng OpenCV `DIST_L2`, `DIST_MASK_PRECISE`.
3. Chọn pixel foreground có distance không nhỏ hơn cực đại lân cận3×3 (dilation). Bao gồm plateau; **không phải thuật toán skeletonization chính xác**.
4. Với mỗi ảnh, lấy trung bình2×distance trên các pixel được chọn.

Distance ở đây đến **tâm pixel nền** theo lưới raster. Vì vậy nét thẳng1pixel cho proxy2px; nét3pixel cho proxy4px. Chẵn/lẻ, góc nghiêng, đầu nét, junction và ngưỡng mực có thể gây bias. **Không ghi “độ dày nét thật trung bình2px” chỉ vì proxy bằng2px.** Đây là chỉ số raster nhất quán để mô tả dữ liệu, không đo độ dày bút hay stroke vector và không phải phép đo vật lý.

Ảnh trắng nhậnNaNwidth và chỉ bị loại khỏi thống kê width, không khỏi số ảnh/mật độ. Ảnh đặc foreground vẫn hữu hạn nhờ zero-padding. Synthetic checks gồm nét1px/3px, mask trắng/đặc và hình tròn; parent không tính lại độc lập EDT của toàn bộ dataset.

## Trung bình, trung vị, độ lệch chuẩn

Với mỗi dataset/split, thống kê **giữa các ảnh** của từng đại lượng (width dùng giá trị mean-ridge-diameter của mỗi ảnh):

- Trung bình: Σxᵢ/N.
- Trung vị: giá trị giữa sau sắp xếp; nếu Nchẵn lấy trung bình hai giá trị giữa.
- Độ lệch chuẩn tổng thể: sqrt(Σ(xᵢ−mean)²/N), `ddof=0` vì đã đo toàn bộ tập manifest.

Không trung bình category trước, không gộp mọi ridge pixel của dataset thành một phân phối để tính width. Pixel-density và fraction-density chỉ khác một hệ số tuyến tính, không phải hai thuộc tính độc lập. Ba bảng, mỗi dataset một bảng, chứa đủ train/test×3đại lượng×3thống kê; CSV giữ precision đầy đủ.

## Category Log-scale Bar Chart

Đếm trực tiếp nhãn trong manifest, riêng từng dataset×split×modality, tổng12biểu đồ. Chọn top10 theo count giảm dần, bottom5 theo count tăng dần sau khi loại top10. Khi bằng count, chọn theo tên category tăng dần. Các official splits đều có≥15lớp; có15category không trùng mỗi biểu đồ. Danh sách kết hợp được xếp count giảm dần; ties theo tên.

X là tên category; Y là số lượng trên thanglog; số thực được chú thích trên cột. Xanh top10, cam bottom5. Khi counts đồng đều (ví dụ TU sketch), top/bottom chỉ là tie-break, không thể suy ra imbalance. Xuất12PNG và12SVG, `class_counts_all.csv` và `class_counts_selected.csv`.

## Chạy và tái kiểm tra

```bash
# Chỉ synthetic checks, không đọc dataset hoặc ghi report:
CUDA_VISIBLE_DEVICES='' .venv/bin/python docs/thesis/dataset_stats.py --self-check

# Full scan đã thực thi; không chạy lại vào root/report hiện hữu:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python docs/thesis/dataset_stats.py \
  --workers 8 --output outputs/dataset_statistics_20260913
```

Script yêu cầu raw output mới và không ghi đè report `docs/thesis/dataset_statistics/`. Muốn chạy một nghiên cứu khác phải chuẩn bị đích report riêng trước; không xóa raw cũ để ép chạy. Process pool tối đa8worker, OpenCV1thread/worker, OpenCL off. Decode lỗi làm scan fail-closed, giữ partial CSV và `FAILED_PARTIAL`; không bỏ qua ảnh lỗi để đủ bảng.

`PASS_FULL` chỉ nói scan sketch/count/report hoàn tất theo định nghĩa trên, không chứng nhận nhãn ngữ nghĩa hoặc độ dày stroke thật. Receipt parent xác nhận aggregation saved rows và counts độc lập, với12real-input mask replays; không phải all-image independent geometry replay. Source thực thi đã lưu tại `outputs/dataset_statistics_20260913/producer_source.py`.
