# Vì sao VMamba Student học kém CNN–Mamba trong run này?

## Kết luận trực tiếp: output representation collapse

Phân tích **CPU trên embeddings/ranks/checkpoints đã lưu**, không mở ảnh, không encoder replay, không gradient measurement hay train mới. Run VMamba60 giữ raw `FINISHED_UNVERIFIED`; các kiểm tra dưới đây không chứng nhận toàn bộ evaluator/encoder.

| Chỉ dấu | VMamba-Tiny60 | CNN–Mamba100 |
|---|---:|---:|
| Final full mAP đã báo cáo | 3.9237% | 26.0709% |
| Train loss tại epoch60 | 4.158637 | 0.943022 |
| Mean cosine giữa các query khác nhau | ≈1 | 0.042616 |
| Tổng variance của unit embeddings quanh mean | 8.513e-16 | 0.956985 |
| Số danh sách top200 khác nhau / 2400 queries | **1** | **2400** |

CNN–Mamba10 cũng đạt final mAP **25.5322%**, với loss cuối **2.543511**. Không có checkpoint/evaluation CNN60 được dùng ở đây: dòng loss epoch60 lấy từ history của run100, có schedule100. Budgets và warmup horizons khác nhau nên không phải matched60 ablation.

**2400 VMamba query vectors vẫn khác nhau ở mức bit**, không phải cùng một tensor chính xác. Tuy nhiên max L2 giữa query đầu và các query khác chỉ **1.289e-7**, max component difference **1.192e-7**. Sự khác biệt gần mức precision FP32 không tạo ra top200 khác nhau. Effective rank trên variance nhỏ như vậy chủ yếu phản ánh nhiễu số, không nên diễn giải như thông tin ngữ nghĩa.

Về chức năng:

```text
z(sketch) ≈ z_common
score(sketch, photo) ≈ z_common · CLIP(photo)
```

Retrieval gần như mất sự phụ thuộc vào query. Với 30 classes, mỗi class80 queries, một top200 cố định có mean P200 = `sum_c(count_c/200)/30 = 1/30 ≈ 3.3333%`, đúng với raw P200 của run. Điều này không đòi hỏi gallery có class sizes bằng nhau. Full ranks chưa được replay; full AP trong cùng class chỉ khác tối đa 1.181e-6 trong arrays đã lưu.

## TRAIN loss cũng gần predictor không phân biệt positive/negative

Bank một batch đủ có64 photos. Nếu softmax đều, multi-positive loss vẫn bằng `log(64)` bất kể số positive (loss average theo positive). Tính cả last batch8/bank16, loss epoch của predictor đều là:

```text
(15392 × log(64) + 8 × log(16)) / 15400 = 4.15816293
```

VMamba epoch60 **4.15863693**, chỉ hơn baseline khoảng0.000474. Loss quanh4.16 gần suốt run, trong khi CNN100 giảm ngay từ những epoch đầu. Vì vậy dấu hiệu không chỉ là train tốt rồi tổng quát hóa TEST kém; objective TRAIN cũng không học được phân biệt hữu ích. Không có saved final TRAIN features để khẳng định mức collapse trên mọi ảnh TRAIN.

## Head học một hướng chung rất mạnh

SVD của saved linear head VMamba `512×768`:

- Một singular component chiếm **80.108%** tổng squared singular-value energy; sau3 gate updates chỉ **0.632%**.
- Mean query direction gần trùng dominant left singular vector: absolute cosine **0.999705**.
- CNN100 projection có component lớn nhất chỉ **1.671%** energy.

Đây là bằng chứng readout cuối bị một hướng chung chi phối, phù hợp với collapsed outputs. **Không chứng minh head là nơi đầu tiên gây lỗi**: chưa có activations trước pooling/head; backbone có thể đã mất thông tin trước đó hoặc cùng thích nghi với head. Matrix head cũng không phải rank1 chính xác.

## Những giải thích kiến trúc có cơ sở, nhưng chưa được chứng minh nhân quả

1. **Mất sketch-specific foreground prior.** CNN tính `ink = 1 - mean(RGB)`, học convolution trên ink, rồi pooling bằng `0.1 + 0.9 × adaptive_max_pool(ink)`. VMamba nhận RGB trắng nền trực tiếp và average-pool learned features không dùng ink mask. Thành phần chung của canvas có thể lấn tín hiệu nét vẽ; không được đồng nhất average learned features với average RGB hoặc kết luận pooling luôn làm mất hình học.
2. **Resolution/readout khác.** CNN học convolution đầu ở224, giữ grid28×28 trước foreground pooling. VMamba patch stem v2 stride2/2 tạo56×56 trước SS2D, rồi56→28→14→7 và global average pooling. Thin-stroke information có thể khó giữ hơn, nhưng7×7 không tự chứng minh thông tin đã bị mất.
3. **Optimization recipe chưa kiểm chứng cho backbone mới.** VMamba có29,873,792 params/14 SS2D blocks/drop-path tới0.2; CNN có2,251,782 params/hai hybrid blocks với local CNN và global residual gate ban đầu0.1/không drop-path. Giữ LR3e-4, AdamW và contrastive loss không bảo đảm hai model học tốt như nhau. Tham số nhiều không phải bằng chứng overfit; ở đây TRAIN loss chưa giảm tốt.

Gate step3 trên32 TRAIN queries đã có mean pair cosine **0.999612**. Đó là tín hiệu direction ban đầu ít đa dạng, nhưng tập32 TRAIN và2400 TEST khác nhau; không dùng số này như longitudinal variance ablation hoặc final TRAIN diagnosis.

## Những nguyên nhân không được dữ liệu hiện có ủng hộ

- **Gallery/split khác:** ba run dùng cùng manifest/config/CLIP SHAs; toàn bộ gallery embeddings, labels, paths **giống hệt**, max embedding difference0.
- **Student bị frozen hoặc quên optimizer:** 276 Adam states đều có step28920, weights ở stem/các stages/head khác checkpoint step3; model và moments finite. Điều này không chứng minh gradient hữu ích hoặc loại trừ mọi coding bug.
- **Chỉ thiếu epochs:** CNN10 đã có retrieval hữu ích; VMamba60 loss vẫn gần đều. Không có cơ sở tự train100 để sửa failure này.
- **DI framework:** được thêm sau khi run kết thúc, không thể gây collapse của snapshot đã train.
- **BF16 production:** actual trainer của cả các controls và VMamba run là FP32. BF16 chỉ thuộc prefix timing riêng; không dùng timing đó để giải thích chất lượng.
- **VMamba không pretrained còn CNN pretrained:** cả hai sketch encoders đều random-init; pretrained CLIP photo gallery được chia sẻ.

Read-only source review và đối chiếu parent chưa thấy lỗi chọn factory/loss hoặc readout dimensions rõ ràng. Scoped native scan/gradient/restore gates trước run không loại trừ mọi lỗi runtime, và **nonzero gradients/weights updated không đồng nghĩa learning thành công**. Gate không theo dõi được sự thoát collapse; PASS của nó chỉ là numerical/plumbing scope, không phải chứng nhận model học tốt.

## Bước tiếp theo phù hợp

Không tự tăng epochs hoặc thay hàng loạt hyperparameters. Nếu được phê duyệt, đo TRAIN-only variance/gradient tại stem, từng stage, trước pooling, trước/ sau head để xác định nơi signal biến mất; sau đó thử một thay đổi có mục tiêu mỗi lượt (chẳng hạn foreground readout hoặc input ink). Không dùng official test để chọn các sửa đổi, không tự thêm pretraining/KD hay production arm mới.

Evidence: `outputs/vmamba_cnn_saved_analysis_20261004/analysis.json`, `head_spectral_diagnosis.json`, script `analyze.py`. Input artifact SHAs được lưu; computed stats chỉ dựa vào saved tensors. Source references: archived `models/sketch_student.py:99–104`, `models/vmamba_student.py:24–59`, `vendor/vmamba/vmamba.py:1357–1364,1391–1410` trong training roots. Reviewer read-only dùng required `cx/gpt-6-luna/max`; không phải delegation của implementation trước đó.
