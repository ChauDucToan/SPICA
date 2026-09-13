# 1. Đề cương và thông điệp

## Tên đề tài gợi ý

**Nghiên cứu truy xuất ảnh từ phác thảo thuộc lớp chưa thấy với mô hình thị giác–ngôn ngữ dưới nhiễu xóa vùng raster**.

Tên SPICA được dùng như tên dự án; không tự tạo cách viết đầy đủ của chữ viết tắt nếu chưa được tác giả chốt.

## Tóm tắt nháp

Đồ án khảo sát bài toán truy xuất ảnh theo lớp từ phác thảo, trong đó tập lớp đánh giá không giao với tập lớp dùng để thích nghi mô hình. Trên nền CLIP, hệ thống sử dụng một bộ mã hóa phác thảo được tinh chỉnh, các prompt ảnh/văn bản và nhánh dự đoán kết hợp thông tin. Huấn luyện kết hợp hai góc nhìn phác thảo sạch và bị xóa vùng raster, với mục tiêu nhiều ảnh dương cùng lớp trên ngân hàng ảnh của mini-batch. Truy xuất sử dụng biểu diễn q từ các patch token, trong khi mục tiêu ảnh chính huấn luyện qua nhánh μ_I. Hệ thống được đánh giá trên các official split Sketchy104/21, TU-Berlin220/30 và QuickDraw80/30, với ngân sách khoảng2,47 lượt quan sát phác thảo, seed42 và batch32. Kết quả cuối clean full mAP lần lượt khoảng51,58%,43,97%,15,48%; trung bình chín điều kiện xóa là40,05%,32,64%,12,75%. Những kết quả này cho thấy cấu hình hoạt động được trên ba bộ, đồng thời còn khoảng trống lớn về hiệu quả trên QuickDraw. Nghiên cứu không tuyên bố vượt SOTA, hiệu quả đa seed hoặc ưu thế nhân quả của từng thành phần. Một sự cố kiểm tra source sau khi Sketchy hoàn tất được giữ nguyên trong hồ sơ, tách biệt với số đo truy xuất đã lưu.

Đây là đoạn nháp để người viết biên tập, không phải tuyên bố tác quyền hay bản tóm tắt đã được GVHD duyệt.

## Câu hỏi nghiên cứu

1. Cấu hình MP-Q đạt hiệu quả truy xuất cuối như thế nào trên từng official split?
2. Metric thay đổi ra sao khi phác thảo mất25/50/75% thông tin mực theo chính sách xóa raster?
3. Readout q, mục tiêu nhiều-positive và các thử nghiệm bổ sung đem lại bằng chứng gì trong các giao thức đã chạy?
4. Hạn chế nào của dữ liệu, kiểm chứng, sampling và quá trình lựa chọn thiết kế ảnh hưởng đến diễn giải?

Không biến câu hỏi3 thành kết luận rằng từng thành phần đều hiệu quả trên cả ba bộ: nhiều ablation chỉ chạy pseudo Sketchy hoặc một bộ TU.

## Bố cục chương đề xuất

| Chương | Nội dung | Tài sản có sẵn |
|---|---|---|
| 1. Mở đầu | Động cơ SBIR, domain gap, mất thông tin, mục tiêu/phạm vi | Tóm tắt và câu hỏi ở trên |
| 2. Cơ sở và liên quan | CLIP, prompt learning, ZS-SBIR, contrastive loss, retrieval metrics | 06_references + đọc toàn văn bổ sung |
| 3. Phương pháp | Phân biệt train/inference, shared prompts, MP loss, xóa raster | 02_method + training_configs |
| 4. Thiết lập thực nghiệm | Splits, counts, seed/budget, masking, không selection | datasets.csv + provenance.csv |
| 5. Kết quả và thảo luận | Fixed-final, severity, progress, ablation theo đúng split | 03_results +04_discussion + figures |
| 6. Kết luận | Đóng góp kỹ thuật thực tế, giới hạn và hướng tiếp theo | 04_discussion + checklist |
| Phụ lục | Hash, run IDs, lỗi kiểm chứng, storage, công thức AP | audit/provenance +05_reproducibility |

## Đóng góp có thể mô tả một cách trung thực

- Xây dựng và vận hành một pipeline nghiên cứu CLIP-based retrieval có phân biệt ảnh/phác thảo, nhiều-positive cùng lớp và đánh giá mất mực raster.
- Thiết lập đánh giá fixed-step trên ba official split, với báo cáo clean và masked cùng định nghĩa metric tường minh.
- Tổ chức bằng chứng thực nghiệm, kết quả âm, lịch sử lựa chọn thiết kế và provenance để giảm nhầm lẫn giữa các biến thể.

Đây là đóng góp triển khai/khảo sát; **chưa đủ để tuyên bố thuật toán mới đầu tiên, tốt nhất, hay chứng minh một cơ chế lý thuyết**. Đối chiếu tổng quan tài liệu trước khi xác định tính mới trong bản nộp.
