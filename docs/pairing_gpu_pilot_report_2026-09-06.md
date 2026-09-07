# Pairing GPU pilot A → B — 2026-09-06

## Kết quả

**Đã chạy thành công A rồi B tuần tự trên RTX 5070 Ti**, cùng seed42, pseudo split3407, 1800 steps/arm, from-scratch. GPU preflight và cả hai trainer đều exit `0`.

| Arm @ step1800 | Treatment | Full pseudo mAP (primary) | mAP@200 (secondary) | P@200 |
|---|---|---:|---:|---:|
| A | Positive ngẫu nhiên cùng lớp, canonical pool | 0.265972276973 | 0.736403760806 | 0.670289141717 |
| B | Positive canonical được pairing tới sketch | 0.266959102909 | 0.733241190936 | 0.663145111564 |
| B − A | | **+0.000986825937** | **−0.003162569870** | −0.007144030153 |

Full mAP tăng **0.098683 điểm phần trăm**, nhưng mAP@200 giảm **0.316257 điểm phần trăm**. Đây là chênh lệch mô tả của một seed/split, **chưa chứng minh paired supervision cải thiện đáng tin cậy**. Không promote; chưa có official-unseen, bootstrap hay multi-seed evidence.

Full AP dùng toàn bộ relevant photos làm mẫu số. mAP@200 ở đây dùng `prefix_positive`; không so trực tiếp với số SketchLVM trên official benchmark.

## Execution và ràng buộc

- A: `2026-09-06 15:41:07Z → 15:48:00Z`, exit0.
- B: `2026-09-06 15:48:36Z → 15:55:30Z`, exit0; bắt đầu sau khi A kết thúc.
- Thời gian training theo trainer: A `362.834027 s`, B `364.693305 s`; không đồng nhất với toàn bộ wall time tiến trình.
- Peak GPU allocated memory mỗi arm: `3,519,603,200` bytes.
- Backbone thật: `ViT-B-32-quickgelu/openai`, tải offline từ safetensors local; không download hoặc đổi dependency.
- Train `46,624 sketches / 58,950 photos`, canonical-positive pool `8,400`; validation `10,963 queries / 13,999 photos`.
- Toàn bộ CLIP-owned parameters giữ byte-identical, bao gồm text tower, LayerNorm, projections và logit scale. Chỉ `sketch_prompt` và `photo_prompt` trainable; gradient cuối của cả hai finite và nonzero.
- Không thay ảnh nguồn/dataset, không chạy official unseen, masking/KD, multi-seed, covariance, lambda search, 5400 steps, commit hoặc push.

## Provenance

Evidence root: [`outputs/pairing_gpu_pilot_20260906_154017_actual/`](../outputs/pairing_gpu_pilot_20260906_154017_actual/).

- Commands/timestamps/exit: `commands.log`.
- Raw metrics: `A/probe_step1800.json`, `B/probe_step1800.json`; primary key `val.full_mAP`, per-query evidence `val.average_precision_per_query`.
- Full histories/configs/checkpoints: `A/run_result.json`, `B/run_result.json`, mỗi arm có `.hydra/config.yaml` và `checkpoints/`.
- Summary: `evidence/pairing_gpu_pilot_summary.json`, `evidence/pairing_gpu_pilot_summary_vi.md`.
- Source archive: `evidence/source_files.tar.gz`, inventory `evidence/source_snapshot.json`.

| Identity | Giá trị |
|---|---|
| Reviewed HEAD | `efa4bd1d489d059d2a0dd4086a181ec96a784f2f` + prepared working-tree changes |
| Source snapshot A/B | `f76598361e8c1d432ce3958a40d039ff7be8277930e9b3651eaf94e5a02e0ab2` |
| Initial model A/B | `a3dd00440ebcce9e3755d86b2706b6d93a3afc6101b0fe6931375884dbbca692` |
| Pairing manifest | `545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c` |
| Pretrained safetensors | `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31` |
| A checkpoint1800 | `26c33edc1fce1f8add47f6b67961bcb667065f4d6f681594c5562a664be5aa96` |
| B checkpoint1800 | `21d09e05c686cb5dfa931c6da503a8d4fa7d323c5bc53775e6318c3427117d60` |

Parent tính lại SHA256 của cả hai checkpoint và đối chiếu raw metrics. Reviewer độc lập xác nhận trung bình 10,963 AP hữu hạn của từng arm tái tạo chính xác full mAP; configs/source/initial state/pairing/query-gallery identities và các training settings chung khớp, ngoại trừ treatment và định danh đầu ra. Không re-encode gallery hoặc chạy training bổ sung để review.

## Giới hạn của bằng chứng

1. **Không dùng nhãn `MATCHED` hoặc `corrected_v2`.** Corrected-v2 thuộc campaign alignment trước. Các run này giữ `PRIMARY_FIXED_STEP_UNCOMPARED`; báo cáo chỉ mô tả fixed-step A/B contrast, không coi single seed là kết luận nhân quả/robustness.
2. Chưa lưu per-step sample IDs/trace. Sampler đã được kiểm tra cùng RNG và các controls/configs khớp, nhưng không có raw trace để xác nhận từng draw đã xảy ra trong GPU training.
3. Arm manifests chứa field `positive_sampling` gây nhầm ở **entry của arm còn lại**: A manifest ghi same_class cho companion B, và ngược lại. Config thực, selected-arm entry, resolved treatment và probe của mỗi run đúng. Không sửa hồi tố artifact; khi truy kết quả dùng **selected-arm config/probe**. Cần sửa schema sinh manifest trước campaign tương lai, không phải thay calibration hay chạy lại pilot này.
4. Một lần inventory preflight trước đó bỏ sót cached safetensors và báo thiếu model: `outputs/pairing_gpu_pilot_20260906_223457_13379/`. Parent xác nhận normal offline loader hoạt động rồi mới khởi chạy campaign actual. Attempt đó không có training/checkpoint, không tính là thêm một run.
5. Tài liệu này được thêm sau training; source hash tài liệu hiện tại thay đổi không đồng nghĩa run phải retrain. Không so full mAP này với historical ~0.67 như một đối chứng causal vì treatment/protocol khác.

**Kết luận thực dụng:** chỉ đổi positive từ same-class canonical sang paired canonical chưa tạo lợi ích retrieval nhất quán trong pilot này. Chưa có cơ sở tuyên bố vượt SketchLVM.
