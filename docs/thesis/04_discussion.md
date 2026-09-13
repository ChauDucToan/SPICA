# 4. Thảo luận, ablation và giới hạn

## 4.1 Diễn giải kết quả official

- MP-Q fixed-final clean full mAP: Sketchy51,5844%, TU43,9661%, QuickDraw15,4787%. Masked macro tương ứng40,0467%,32,6385%,12,7538%.
- Mất mực làm giảm hiệu quả tổng hợp trong cấu hình này. Không từ đó suy ra từng query đều giảm hoặc severity đo được luôn đúng bằng mức yêu cầu.
- Giá trị QuickDraw thấp trong saved local results khớp scalar W&B; không có bằng chứng mất metric ở các hàng đã kiểm tra. Điều này không thay cho independent ranking/AP recomputation. Số cuối trùng các scalar clean/masked của historical MP-Q official QuickDraw; TU cũng trùng historical scalar. Đây là sự nhất quán kết quả đã lưu, không chứng minh toàn bộ trajectory/encoder bit-exact giữa các campaign trong lượt audit này.
- Khoảng cách giữa ba bộ không chỉ thể hiện robustness: số query/gallery, số lớp, phân bố, phong cách vẽ và độ khó khác nhau. Không xếp hạng độ bền cross-dataset chỉ từ tỷ lệ masked/clean.
- QuickDraw có1query ở `mask_f75_s303` mang `target_unreachable`; hàng đó được giữ trong đánh giá. Xem `tables/conditions.csv`.

## 4.2 Những thí nghiệm lịch sử có thể đưa vào đồ án

Không trộn số dưới đây với bảng official104/21. Mỗi hàng chỉ dẫn tới báo cáo đã có; khi trích số chi tiết phải giữ checkpoint/split/metric của báo cáo đó.

| Câu hỏi/nhánh | Phạm vi | Kết luận giới hạn | Báo cáo gốc |
|---|---|---|---|
| q so với μ_I trên cùng F2_MP checkpoint | Sketchy pseudo84/20, fixed3600 | q tốt hơn μ_I trên clean và masked tổng hợp; không chứng minh predictor không cần train | [MP-Q readout](../fusion_mp_q_readout_results_2026-09-09.md) |
| Chuyển main MP từ μ_I sang q (QMP) | pseudo84/20, fixed3600 | QMP kém MP-Q trong phép thử này; không tự suy ra nguyên nhân là collapse | [QMP analysis](../fusion_qmp_decline_analysis_2026-09-09.md) |
| Thêm text MP và q MP (TQMP) | pseudo84/20, fixed3600 | Không vượt MP-Q; thay nhiều thành phần nên không phải causal text-only ablation | [TQMP](../fusion_tqmp_results_and_benchmark_gap_2026-09-10.md) |
| Photo-CE bổ sung | pseudo84/20, fixed3600 | Clean giảm nhẹ, masked fullAP tăng rất nhỏ; không promote | [PCE](../fusion_photo_ce_results_2026-09-10.md) |
| Masked teacher → clean student sketch-reference | official TU220/30, fixed1189 | Clean fullAP44,8616% so43,9661%; masked33,3941% so32,6385%. Một TU ablation sau khi thấy baseline | [TU-SREF](../coupled_sketch_ref_tuberlin_results_2026-09-11.md) |
| SIGReg trên fusion | pseudo84/20 | Không đạt promotion của cặp F2/F2_SIG | [F2/SIG](../fusion_f2_sig_results_2026-09-09.md) |
| Xóa cục bộ vs rải rác |100queries, không update | Có đáp ứng cục bộ, không chứng minh meanpool là bottleneck | [Local](../local_information_diagnostic_2026-09-10.md) |
| Center/zoom/jitter |100queries, không update | Không tự promote preprocessing; independent fullAP gate có FAIL | [Scale/local](../scale_local_reliability_report_2026-09-10.md) |

SREF dùng λ đo ở initialization, không phải λ tối ưu đã search. Các seed101/202/303 của mask không cung cấp variance theo training seed. Các ablation lịch sử đã được lựa chọn sau khi thấy kết quả trước đó; mô tả là quá trình phát triển có quan sát, không một nghiên cứu blind pre-registered toàn bộ.

## 4.3 Vì sao chưa thể tuyên bố ngang/vượt paper?

SketchLVM và SeCo-SBIR là nguồn liên quan, không phải control được tái hiện đầy đủ trong repo. Cần đối chiếu native CLIP readout, prompt/adaptation recipe, train duration, splits, gallery và công thức AP200. TU paper có thể dùng P@100; bảng này hiện P@200. Không đổi tên cột để làm số trông tương đương.

Các giả thuyết như train ngắn, không dùng native CLS readout, objective μ_I nhưng retrieval q, hoặc mức adaptation mạnh là **hướng kiểm tra**, chưa phải nguyên nhân đã được can thiệp và chứng minh. Điểm yếu QuickDraw cần được báo cáo, không che đi bằng chỉ dùng AP200 prefix-positive vốn có mẫu số khác.

## 4.4 Các giới hạn phải đưa vào bản nộp

1. Một training seed42, không multiseed confidence interval hay kiểm định significance.
2. Test official được theo dõi5lần; không blind holdout. Fixed-final giúp tránh chọn peak, nhưng không xóa được việc đã quan sát test trong quá trình nghiên cứu.
3. Xóa raster không tương đương phác thảo tự nhiên thiếu nét, occlusion vật lý hay semantic part deletion.
4. Audit mới xác minh hash và trung bình metric per-query đã lưu. Không chạy lại encoder, independent full-gallery sort hoặc kiểm tra lại toàn bộ công thức AP từng query trong lượt chuẩn bị này.
5. Sketchy raw failure là lỗi source guard sau khi có kết quả; phải ghi trong provenance. Không sửa evidence để gọi cả ba VERIFIED.
6. Một số historical checkpoints đã xóa theo user approval; replay tại các bước đó không còn khả dụng. Hash/log không thay thế weights.
7. W&B từng hiển thị crashed trong khoảng log thưa; kết thúc TU/Q hiện `finished` theo API. Trạng thái UI không phải chứng cứ duy nhất cho tiến độ hoặc chất lượng mô hình.

## 4.5 Hướng tiếp theo — đề xuất, chưa thực hiện

- Đối chiếu evaluator paper và bổ sung đúng cutoff TU P@100 từ saved rankings sau một audit riêng.
- Multiseed và budget/convergence study với protocol khóa trước khi xem test mới.
- So sánh readout/adaptation/objective bằng một thay đổi mỗi arm, cùng exposure và sampling.
- Kiểm tra robustness trên dữ liệu mất nét tự nhiên, không chỉ synthetic raster deletion.

Không diễn đạt các đề xuất này như kết quả đã có; không chạy train bổ sung chỉ để lấp phần báo cáo.
