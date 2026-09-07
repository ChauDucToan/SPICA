# Hướng1 — C/M 3600 steps: kết quả đã thực thi

## Trạng thái

**C và M đã train từ đầu đủ3600 updates, tuần tự, exit0; W&B online đã upload và verify checkpoint artifacts.** Không early stopping; không chạy KD/teacher hoặc variant ngoài C/M.

- Campaign: `frozen_prompt_masked_view_3600_2026-09-07`.
- Evidence root: `outputs/masked_view_3600_execution_20260907T073119Z/`.
- C: [W&B qkbhbvw3](https://wandb.ai/a-cctest05187-erd/spica/runs/qkbhbvw3).
- M: [W&B f90sf97b](https://wandb.ai/a-cctest05187-erd/spica/runs/f90sf97b).
- Source snapshot SHA256 chung: `e60c74aa84fe09ee2216248408f2e145460a7b421294a991f97a5a00291309af`.
- Configs/source/Hydra overrides/optimizer/scheduler/RNG lưu trong run outputs và checkpoints. Không sửa source/config/docs trong lúc chạy C/M.
- GPU preflight thật PASS; toàn bộ original CLIP visual/text/LayerNorm/projections/logit scale frozen và byte-identical. Chỉ sketch/photo prompts học.

## Latest và hai best của từng run

Tất cả mAP@200 trong bảng là **prefix_positive**, theo tiêu chí user đã chốt, chưa được xác minh equivalent với con số SketchLVM paper. Masked là macro9 điều kiện3 fractions ×3 mask seeds, không phải3 training seeds. Tất cả dùng pseudo-validation split3407, không official test.

| Arm | Selection | Step | Clean P@200 | Clean mAP@200 | Masked P@200 | Masked mAP@200 |
|---|---|---:|---:|---:|---:|---:|
| C | latest |3600|0.449898|0.699090|0.225162|0.329966|
| C | best_clean |1800|0.670306|0.736407|0.305038|0.339448|
| C | best_masked |1200|0.658371|0.735521|0.301417|0.343001|
| M | latest |3600|0.388535|0.703248|0.243886|0.390180|
| M | best_clean |600|0.625806|0.729790|0.327563|0.383216|
| M | best_masked |2400|0.443107|0.715157|0.270547|0.402876|

Best chọn riêng theo clean hoặc macro masked mAP@200_prefix_positive; chỉ xét600–3600, tie giữ step sớm. P@200 luôn lấy tại cùng checkpoint, không ghép với peak P@200 ở step khác. Latest là3600, không được thay bằng best.

### Full mAP đối chiếu

| Arm / selection | Clean full mAP | Masked macro full mAP |
|---|---:|---:|
| C / latest |0.186984|0.106077|
| C / best_clean |0.265980|0.142767|
| C / best_masked |0.261934|0.141752|
| M / latest |0.178823|0.116687|
| M / best_clean |0.244144|0.144984|
| M / best_masked |0.179677|0.117118|

Full histories với P@200 và cả3 AP@200 denominators/full mAP nằm trong W&B và raw probe JSON. Không đổi metric convention historical.

## Diễn giải đúng phạm vi

1. **Không lặp lại kết luận “M kém C ở mọi masked metric” của fixed-step1800.** Ở latest3600, M cao hơn C ở masked P@200, full mAP và prefix mAP@200. Nhưng C cũng suy giảm mạnh về cuối, nên không coi thắng latest là bằng chứng M đạt chất lượng tốt hơn mọi checkpoint C.
2. So best_clean–best_clean, C vẫn cao hơn M về clean P@200 và prefix mAP@200. M giữ masked scores cao hơn tại checkpoint clean-best của chính nó; đây là so sánh theo cùng selection rule, không cùng step.
3. So best_masked–best_masked, M cao hơn về prefix mAP@200 (**0.402876 vs0.343001**), nhưng thấp hơn về masked P@200 (**0.270547 vs0.301417**) và full mAP (**0.117118 vs0.141752**). Đây là trade-off giữa các metric, không universal dominance.
4. C3600/best_clean rơi vào1800 trong các mốc đã đánh giá; M best_clean600, best_masked2400. **Không chứng minh1800 là nghiệm tối ưu của baseline.** Lịch train hiện tại và budget3600 chỉ cung cấp các checkpoint đã quan sát; không khảo sát mọi schedule/hyperparameter hoặc bước ở giữa probes.
5. Stage1 campaign đã hoàn tất. Có thể khép **vòng xác nhận3600 này**, không tự bác bỏ toàn bộ masked training hoặc mở search/seeds/variants để tìm kết quả tốt.

## Tái lập, hashes và kiểm chứng độc lập

- C/M có đủ checkpoints `0,600,1200,1800,2400,3000,3600`; mỗi arm115,200 observations =32×3600.
- Query/label/positive/negative paths và mask metadata khớp row-by-row toàn bộ traces. Historical negative-image SHA có thể null; không nâng path identity thành negative-byte identity khi thiếu hash.
- Parent và reviewer tính lại raw per-query full AP/AP@200 và macro9 conditions, khớp scalar reports; lựa chọn best và aliases đúng.
- Online W&B aliases `latest/best_clean/best_masked` đã được tải xuống qua API và kiểm tra SHA256 khớp local, không chỉ dùng fake backend của CPU gate.
- So actual prompt tensors với historical pilot tại0 và1800: `torch.equal` cho cả sketch/photo prompts ở C và M, và các metric1800 khớp. File checkpoint SHA khác vì provenance mới; không nhầm file-hash khác thành weights khác. Điều này kiểm chứng schedule probes mới không đổi trajectory qua1800 trong hai run này, không phải cam kết GPU bitwise trên mọi máy hoặc mọi run.
- CPU gate trước chạy:244 tests PASS; source archive/reference đầy đủ. Không dùng nhãn corrected_v2/MATCHED của campaign alignment lịch sử.

| Arm/alias | Step | Checkpoint SHA256 |
|---|---:|---|
| C/latest |3600|`c142f47dbb29c7b0689b7dd6d83491f12137bd076a58d4be38202af685f95e44`|
| C/best_clean |1800|`6ff8d13936116379d892b62a07d00eb077f49d1ce2b3fb73d56a6afafec75b53`|
| C/best_masked |1200|`39764f4b74bf9056ef3a105bddd6d13c53995210892c3ddabe2b30525e33ec20`|
| M/latest |3600|`f2fb5290d6b4ea5d5512d51aa6c90fc6d8b3a6917d055d84ecc528042b08cab3`|
| M/best_clean |600|`d330733f42789c26df41524da983bb823d4897b7d48b2c947c39d979e0d6f220`|
| M/best_masked |2400|`d27d9aab1eef240f8940b0a368292f9083342b822ad1c76e2d6aadad186aad32`|

### Correction cho worker summary giữ nguyên

`execution_summary.md/json` ghi `mask statuses {'ok':9}` theo số **conditions**, không phải số query-mask records. Mỗi probe thực tế có **98,667 records ok =9×10,963**, không blank/unreachable/zero-fraction. Cả7 probes mỗi arm là690,669 masked evaluation records; không coi chúng là independent samples vì query/masks lặp qua checkpoints.

Worker artifacts được giữ nguyên; correction nằm trong báo cáo này và `parent_verified_selection_table.json` với fields riêng `condition_count=9`, `successful_mask_records=98667`.

## Evidence index

Trong execution root:

- `gpu_preflight_actual/preflight_result.json`: CLIP thật + GPU/autograd/byte identity preflight.
- `provenance_before_training/`: nguồn trước launch và configs đã chốt.
- `C/run_result.json`, `M/run_result.json`: histories, selections, configs/source/run IDs.
- `<arm>/masked_view_probe_step<step>.json`: raw AP arrays, masks và metrics các conditions.
- `<arm>/.hydra/`, `<arm>/resolved_config.yaml`, `<arm>/source_snapshot/`.
- `<arm>/checkpoints/`: step files và named aliases, không chứa teacher mới.
- `wandb_verification/summary.json`: verification artifact aliases/hashes/downloads.
- `parent_verified_selection_table.json`: parent-verified sáu checkpoint rows.
- `C_exit_code.txt`, `M_exit_code.txt`, execution logs.

Runtime trong trainer (bao gồm các probe trong timed loop): C2637.9s, M2619.5s, không gọi đây là training-forward-only throughput. GPU peak allocated trong loop4,409,082,880bytes/arm; GPU đã idle sau hoàn tất.

## Hướng2 tiếp theo — vẫn NOTRUN

Đề xuất reference teacher là **C3600/best_clean@1800 của campaign mới**, được chọn theo rule đã chốt trước khi train, không tái gắn nhãn checkpoint historical thành optimal teacher. Trước khi thêm loss/train stage2 cần đo KD gradients với masked rank/CE và full task gradient, rồi duyệt calibration/initialization/budget stage2 theo [proposal](masked_view_3600_protocol_and_direction2_proposal.md).

Chưa chạy teacher/KD diagnostic hoặc stage2 training trong campaign này. Một seed/pseudo split, chưa official benchmark hoặc multi-seed confirmation.
