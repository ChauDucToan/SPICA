# Hướng 1 — Shared encoder full/masked: GPU pilot 2026-09-07

## Kết luận

**Đã hoàn tất triển khai, CPU gate, GPU preflight, training C → M tuần tự và đánh giá clean + masked tại step1800. Kết quả âm cho cấu hình này: M thấp hơn C về full mAP ở clean và cả ba mức masking.** Không tự chuyển sang JEPA/teacher/student/LoRA hoặc chạy thêm training.

| Query khi đánh giá | C: train full + full | M: train full + masked | M − C |
|---|---:|---:|---:|
| Clean | 0.265979747 | 0.172586816 | −0.093392931 |
| Mục tiêu xóa 25% mực | 0.210402560 | 0.146166818 | −0.064235742 |
| Mục tiêu xóa 50% mực | 0.137763993 | 0.113990776 | −0.023773217 |
| Mục tiêu xóa 75% mực | 0.080134612 | 0.079089390 | −0.001045222 |
| **Macro 9 điều kiện masked — primary** | **0.142767055** | **0.113082328** | **−0.029684727** |

Các dòng masked là trung bình full mAP của ba mask seeds `101,202,303`; macro là trung bình của 3 severities × 3 seeds. **Đây không phải ba training seeds.** Mỗi condition có 10,963 queries và 13,999 gallery photos. Mẫu số full AP là toàn bộ category positives. mAP@200 dùng `prefix_positive` được lưu riêng, không dùng để thay primary metric hoặc so trực tiếp với SketchLVM official.

Chênh lệch macro là **−2.968473 điểm phần trăm**; clean là **−9.339293 điểm phần trăm**. M giữ được tỷ lệ điểm so với clean của chính nó tốt hơn, nhưng clean của M thấp hơn đáng kể: tỷ lệ retention không đảo ngược kết quả mAP tuyệt đối thấp hơn. Mức75% gần nhau không phải chiến thắng của M. Chưa có khoảng tin cậy hoặc bằng chứng nhiều training seeds để kết luận statistical significance; kết quả không bác bỏ mọi cách masked training.

## Kiến trúc và protocol đã chạy

- Campaign: `frozen_prompt_masked_view_pilot_2026-09-07`.
- Configs: `configs/experiments/masked_view_C.yaml`, `masked_view_M.yaml`.
- C: full + full; M: full + masked. Cả hai dùng **cùng sketch encoder và sketch prompts cho hai views**; không teacher, KD, consistency loss, LoRA, SIGReg hay JEPA.
- Mỗi step: batch32 sketch gốc, query forward batch64 chứa hai views; photo positive/negative chỉ encode một lần rồi dùng cho hai views. Rank và hard-text query CE được lấy trung bình với trọng số0.5 mỗi view. Giữ softplus rank hiện hữu.
- CLIP gốc frozen toàn bộ visual/text towers, LayerNorm, projections và logit scale. Chỉ `sketch_prompt`, `photo_prompt` trainable. Vì photo prompts học riêng ở C/M, gallery embedding values có thể khác giữa arms; paths/labels và danh sách gallery giữ nguyên.
- Seed42, pseudo split3407, từ đầu1800steps/arm, không resume/chọn peak. C/M bắt đầu cùng model state và training source.
- Positives cùng lớp từ canonical pool8,400 ảnh; negative pool58,950 ảnh pseudo-train. Training46,624 sketches; validation10,963 sketches/13,999 photos. Pairing manifest chỉ dùng để xác định canonical pool trong hướng1, không thay bằng paired sampling.

## Masking chính xác là gì?

`ink_centered_square_v1` chọn tâm trên pixel có mực rồi xóa một vùng vuông liên tục bằng màu trắng. Ink được định nghĩa bởi grayscale dưới0.9 sau khôi phục chuẩn hóa CLIP. Mask xảy ra trên raster đã resize/crop, **trước encoder**, không phải xóa feature sau attention. Seed từ path tương đối/step; không tiêu thụ global RNG của sampler.

Train fractions được lấy trong `[0.25,0.5,0.75]`, mask train seed4242. C cũng tạo mask và metadata để cân bằng quy trình, nhưng không đưa masked tensor vào model. Model không nhận seed, bbox hoặc tỷ lệ bị xóa. Full view không truyền features vào masked view.

Tỷ lệ xóa được đo theo **số pixel có mực**, không phải diện tích vuông hay số stroke. Realized train fractions:

| Target | Trung bình thực tế | Min–max |
|---:|---:|---:|
| 0.25 | 0.257442 | 0.250000–0.385714 |
| 0.50 | 0.509428 | 0.500000–0.630542 |
| 0.75 | 0.757018 | 0.750000–0.900000 |

Source/dataset chỉ được đọc; không copy/sửa ảnh hoặc manifest dataset. Raster PNG hiện tại chưa cung cấp stroke sequence, nên không gọi kết quả này là true-stroke deletion hoặc sketch prefix retrieval.

## Bằng chứng thực thi và kiểm chứng độc lập

Training root: `outputs/masked_view_gpu_pilot_20260907T011358Z/`.

- GPU preflight thật PASS; C/M training đều exit0, 1800steps, chạy tuần tự trên RTX5070Ti.
- Training runtime khoảng616.4s/arm; peak GPU allocated memory4,409,082,880bytes. Đây không bao gồm toàn bộ thời gian hậu xử lý masked evaluation.
- Traces `C/train_observations.jsonl`, `M/train_observations.jsonl`: **57,600 rows/arm**. Query/label/positive/negative, photo hashes, mask seeds/bboxes và input/output hashes được đối chiếu, khớp giữa hai arms.
- Full CLIP byte identity và finite/nonzero gradients của cả hai prompts được ghi trong run/probe artifacts.
- Training source SHA256 chung: `5d635b4605d0e954bb1089c1e9995efe49b48c7d070790482aaf1e963b2cc206`.
- Pairing manifest SHA256: `545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c`.
- Pretrained local safetensors SHA256: `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`; offline load, không download.

| Checkpoint1800 | SHA256 |
|---|---|
| `C/checkpoints/frozen_prompt_step1800.pt` | `0af31d564f4068a5dbd510acbee533c73d9a1d44c67213da07d26e4c702a5d4a` |
| `M/checkpoints/frozen_prompt_step1800.pt` | `ba66c367f006edc560ce980d35d61fb220e23e571386f860db52f0f94c1d3bad` |

Final evaluation root: `outputs/masked_view_gpu_pilot_20260907T011358Z/postprocess_20260907T015350Z/`.

- `eval_C/masked_view_evaluation.json`, `eval_M/masked_view_evaluation.json`: cả hai `COMPLETE`, 9conditions mỗi arm.
- Parent và reviewer độc lập đã kiểm tra checkpoint hashes và trung bình raw per-query AP, tái tạo đúng clean/per-condition/macro mAP. Clean replay delta bằng0 ở cả hai.
- Mỗi arm đánh giá98,667 masked queries (9 ×10,963). Tất cả mask status `ok`; không blank/unreachable/zero-fraction.
- `eval_C/query_mask_manifest.jsonl` và `eval_M/query_mask_manifest.jsonl` **byte-identical**: SHA256 `d9f8215d4faf9163cb75ff4aa4d6e29dd335444d4911940ec73afb05d1415552`.
- Chi tiết/raw commands/exit codes: `final_summary.json`, `eval_C.log`, `eval_M.log`, các file `*_exit_code.txt` trong final evaluation root; training logs/preflight/source archive ở training root.
- CPU suite sau sửa hậu xử lý: **224 passed**; parent chạy lại `224 passed in3.32s`, `git diff --check` PASS; Ruff PASS đã ghi trong `final_ruff.log`.

## Lỗi hậu xử lý được sửa, không retrain

Lần evaluation C đầu tiên bị chặn do raw run ghi pairing path tương đối nhưng evaluator tái tạo path tuyệt đối rồi so chuỗi. File và SHA không khác nhau. Fix giữ nguyên representation đường dẫn trong provenance để đối chiếu, đồng thời resolve đường dẫn độc lập để kiểm tra file/hash. Regression tests xác nhận absolute/relative cases hợp lệ và SHA/path thay đổi bị chặn.

Failed artifact `eval_C/masked_view_evaluation_failed.json` ở training root được giữ nguyên. Evaluation sau sửa ghi vào **postprocess directory mới**; không thay checkpoints, configs, masks, training loss hoặc training source. Patch/source hậu xử lý nằm trong `postprocess_20260907T015350Z/source_postprocess/`. Báo cáo này được viết sau cả training và evaluation, không gán hash tài liệu mới ngược vào run.

Lưu ý presentation: bảng trong raw `final_summary_vi.md` ghi nhầm nhãn “CLEAN target25/50/75%”; đó là các điều kiện **masked**, không phải clean. Bảng ở đầu tài liệu này lấy từ raw JSON/AP, đã tách clean thành dòng riêng. Không sửa hồi tố raw artifact.

## Phạm vi kết luận

Hướng1 **chưa cải thiện mục tiêu absolute masked retrieval trong pilot này**. Chưa chạy official unseen, multi-seed training, bootstrap, teacher/student, JEPA/LeJEPA, LoRA, covariance, lambda search hoặc extension5400steps. Không tự gắn nhãn `MATCHED`/`corrected_v2`; raw runs giữ `PRIMARY_FIXED_STEP_UNCOMPARED` và checklist so sánh ở trên là evidence riêng của C/M.

Không cần retrain để hoàn thành báo cáo. Muốn nghiên cứu tiếp cần một yêu cầu thí nghiệm mới; không tự chuyển sang hướng2 từ negative result này.
