# Semantic text S0/S1/S2 — verified results, 2026-09-08

## Kết luận

**Giữ S0 làm semantic baseline hiện tại. S1/S2 đều FAIL promotion gate so S0.** S2 text anchor giúp khôi phục một phần chất lượng so S1 ở best_clean, nhưng không đủ để giữ P200/full mAP của S0. Không kết luận soft prompts/anchoring thất bại trong mọi cấu hình hoặc dùng kết quả này để chứng minh predictor/SIGReg tương lai sẽ tốt hơn.

| Arm | Thiết kế | W&B run |
|---|---|---|
| S0 | Hard text bank; independent sketch/photo visual prompts | [ltg40zc4](https://wandb.ai/a-cctest05187-erd/spica/runs/ltg40zc4) |
| S1 | Shared soft text context4, không anchor | [m6h296ii](https://wandb.ai/a-cctest05187-erd/spica/runs/m6h296ii) |
| S2 | Như S1 + fixed lambda1 classmean cosine text anchor | [qjr6ukiu](https://wandb.ai/a-cctest05187-erd/spica/runs/qjr6ukiu) |

Cả ba train từ đầu3600 updates, seed42/pseudo3407;84 train classes,20 pseudo-validation classes,21 official classes không dùng. Original CLIP image/text/LN/embeddings/projections/logit scale frozen; chỉ visual prompts4608params và S1/S2 thêm text context2048params. Full/full query views, **không training masks**, có clean +9masked evaluation probes mỗi600steps. Protocol: [Step0/1](semantic_baseline_step0_step1_protocol.md); quyền design-only trong tài liệu cũ đã được supersede bằng user training request như [readiness](semantic_text_step1_readiness_2026-09-07.md).

## 1. Promotion: cùng best_clean, không peak mixing

Chọn clean/masked prefix-positive AP200 trên steps600–3600, earliest tie; step0 không đủ điều kiện. P200/full AP lấy tại chính checkpoint được chọn. Candidate chỉ pass nếu clean prefix AP200 tăng nghiêm ngặt và clean P200/full mAP không giảm so S0. Masked metrics report-only cho chọn semantic baseline.

Cả ba `best_clean` ở step1800:

| Arm | Clean prefix AP200 | Clean P200 | Clean full mAP | Promotion vs S0 |
|---|---:|---:|---:|---|
| S0 |0.736406897|0.670306017|0.265979747|Control retained|
| S1 |0.730069785|0.517013579|0.204940348|FAIL|
| S2 |0.736756643|0.589515174|0.232639437|FAIL|

Deltas tính từ raw precision, **điểm phần trăm**:

| So sánh | Δ clean prefix AP200 | Δ clean P200 | Δ clean full mAP |
|---|---:|---:|---:|
| S1 − S0 |−0.633711|−15.329244|−6.103940|
| S2 − S0 |+0.034975|−8.079084|−3.334031|
| S2 − S1 |+0.668686|+7.250159|+2.769909|

S2−S1 là **anchor attribution comparison**, không phải promotion so S0. Tại latest3600, S2 cũng không thắng S1 trên mọi metric; không gọi anchor cải thiện đơn điệu xuyên training. Prefix-positive AP200 có thể tăng dù ít positives được retrieved hơn; P200/full mAP guards thực sự thay đổi quyết định ở đây.

## 2. Toàn bộ selections và denominators

Mọi giá trị0–1. `all-relevant` chia tổng relevant gallery R; `min(R,200)` chia min(R,200); prefix-positive chia max(1,R200). Full mAP dùng toàn gallery. Bảng JSON [compact raw-precision evidence](semantic_text_step1_results_2026-09-08.json) chứa cả per-fraction masked metrics,7-step trajectories, checkpoint hashes và raw evidence receipts.

### Clean

|Arm|Selection|Step|Prefix AP200|P200|Full mAP|AP200 all-relevant|AP200 min(R,200)|
|---|---|---:|---:|---:|---:|---:|---:|
|S0|latest|3600|0.699090|0.449898|0.186984|0.103979|0.363889|
|S0|best_clean|1800|0.736407|0.670306|0.265980|0.168805|0.590770|
|S0|best_masked|1200|0.735521|0.658371|0.261934|0.165573|0.579458|
|S1|latest|3600|0.717928|0.409471|0.176353|0.094630|0.331177|
|S1|best_clean|1800|0.730070|0.517014|0.204940|0.124540|0.435849|
|S1|best_masked|1800|0.730070|0.517014|0.204940|0.124540|0.435849|
|S2|latest|3600|0.714648|0.385911|0.179909|0.088313|0.309069|
|S2|best_clean|1800|0.736757|0.589515|0.232639|0.145735|0.510026|
|S2|best_masked|1800|0.736757|0.589515|0.232639|0.145735|0.510026|

### Masked macro9

25/50/75% requested ink deletion × seeds101/202/303, query-macro then condition mean; clean gallery.

|Arm|Selection|Step|Prefix AP200|P200|Full mAP|AP200 all-relevant|AP200 min(R,200)|
|---|---|---:|---:|---:|---:|---:|---:|
|S0|latest|3600|0.329966|0.225162|0.106077|0.040148|0.140508|
|S0|best_clean|1800|0.339448|0.305038|0.142767|0.062875|0.220047|
|S0|best_masked|1200|0.343001|0.301417|0.141752|0.062238|0.217817|
|S1|latest|3600|0.336423|0.218067|0.105679|0.038440|0.134531|
|S1|best_clean|1800|0.344183|0.257874|0.117871|0.049201|0.172188|
|S1|best_masked|1800|0.344183|0.257874|0.117871|0.049201|0.172188|
|S2|latest|3600|0.337342|0.208638|0.105765|0.035939|0.125774|
|S2|best_clean|1800|0.343300|0.277876|0.127794|0.055359|0.193742|
|S2|best_masked|1800|0.343300|0.277876|0.127794|0.055359|0.193742|

Masking robustness không universally improved: best_masked prefix có chênh lệch nhỏ, nhưng P200/full mAP của S1/S2 thấp hơn S0. Không dùng retention từ một clean baseline yếu hơn làm bằng chứng thắng tuyệt đối.

## 3. Text drift và compute

Drift = mean84(1−cos(T_learned,T0)):

| Arm | Drift@1800 | Drift@3600 | Timed loop seconds | Peak allocated bytes | Trainable params |
|---|---:|---:|---:|---:|---:|
| S0 |≈0|≈0|2302.963496|4,409,255,424|4608|
| S1 |0.640151381|0.624543965|2623.932980|6,677,188,096|6656|
| S2 |0.234240085|0.207136005|2632.050249|6,677,188,096|6656|

Loop timing bao gồm scheduled probes trong loop; loại setup/initial probe và final postprocessing/replay/W&B reporting ngoài loop. `seconds_per_update` trong raw report là loop amortized, không kernel-only optimizer timing; không dùng các estimated500/5400 fields làm actual training evidence. S0≈38.38min, S1≈43.73min, S2≈43.87min. Peak allocation khoảng4.41GB/6.68GB/6.68GB (decimal GB).

Anchor làm drift nhỏ hơn trong các probes quan sát, nhưng preservation không đảm bảo retrieval tốt hơn S0. Initial/S0 cosine penalty khoảng−4.75e−8 là float32 rounding gần0, không negative divergence có ý nghĩa. Không clamp/rewrite raw numbers.

115,200 original observations và230,400 query-view forwards mỗi arm,3600 updates; tổng10,800 updates trong campaign hoàn chỉnh. Run S0 bị reboot ngắt ở campaign cũ là wasted/interrupted compute riêng, không cộng thành matched budget hoặc lấy checkpoint của nó vào comparison này. Không có extra seed/search/official unseen hay5400steps.

## 4. Provenance, restart và kiểm chứng

Execution root: `outputs/semantic_text_execution_20260908T011006Z/`.

- Training HEAD: `6a4d8fc09c1ba15d8c68bbd67ade9f0a2f344ab1`.
- Shared training source SHA256: **`ca4671cc50010a654f217ddb46714a71063ba4e63a25d6cdd634c9298873d9d1`**.
- Source archive534files/arm; parent và independent reviewer recompute aggregate SHA từ actual ordered path/content bytes, không chỉ tin index field. `outputsnewgate/` source-like files có trong actual inventory, không xóa để làm đẹp hash.
- Commit thiết kế V1 `94c741f` đến sau khi cả ba arm hoàn tất; không phải source dùng train. Không yêu cầu retrain chỉ vì HEAD/docs thay đổi.
- Cũ `outputs/semantic_text_execution_20260907T160000Z/`, W&B `buktk5xa`, bị reboot khi S0 đang probe3000; user chọn restart S0→S1→S2 từ đầu. Old checkpoints/logs vẫn giữ nguyên. Fresh preflight và first600 model/optimizer/scheduler/19,200 trace-row parity PASS trong restart receipts.
- Three traces115,200rows **byte-identical**; no train-mask fields. Initial visual prompts/T0 torch.equal across arms; S1/S2 context initialization torch.equal. T0 shape84×512, shared context4×512. Whole original CLIP byte-invariance producer checks PASS.
- Seven immutable checkpoints/arm có config/source/split/text-state/optimizer/scheduler/RNG lineage; current context hashes validated từng step, latest/best aliases matched.
- Per probe98,667 mask records; per arm690,669, all3arms2,072,007 repeated evaluation records. All status `ok`, no blank/unreachable/zero; matching condition hashes acrossarms/steps. Repeated masks/probes không là independent observations/seeds.

### Local + online + real replay

1. `local_verification_20260908/verification_report.json`: full-AP và3AP200 per-query means recomputed; all9condition/fraction/macro arithmetic checked, selections reselected từ raw precision.
2. `wandb_verification/wandb_verification.json`:3 finished runs; **361 unsampled history rows/arm**, step_train0..3600 mỗi10,7retrieval probes;70 scalar keys/probe,490/arm,**1470 scalar comparisons total**. No dropped history steps. Actual sparse diagnostic cadence recorded, không giả mọi diagnostic có ở mọi training step.
3. Nine logical selected W&B artifacts downloaded và SHA-verified, named/step aliases matched actual checkpoint. S1/S2 best_clean/best_masked cùngstep1800, không phải9unique checkpoints. Actual artifact inventory retained; không claim toàn source archive/uv.lock đã upload nếu API không cho thấy.
4. `selected_replay_20260908/replay_summary.json`: **9 standalone CUDA replays,7unique checkpoints, all five clean/condition/macro metrics delta0.0**;81condition-hash checks PASS,10,963queries×13,999gallery. No optimizer/model updates.
5. `parent_summary/summary.json`: aggregate-source recompute và direct selection/alias/promotion checks, links/SHA của raw evidence, full-precision compact tables.

**P200 scope:** producer/evaluator không serialize per-query P200 arrays. P200 scalar được recompute từ images/checkpoint trong standalone GPU replay và khớp chính xác; không có independent second evaluator/sort hoặc stored per-query P200 array. Không đánh đồng scalar replay với độc lập xác minh mọi query của một implementation thứ hai. Raw full-AP/3AP200 arrays được giữ local. Text initial-bank hash không có tensor gốc trong mọi nonzero checkpoint để tự recompute; initial bank parity/T0/step0 context evidence được giữ riêng.

## 5. Scientific limits và bước tiếp theo

Một seed/pseudo split, validation đã dùng cho development nhiều lần; không statistical significance, SeCo/SketchLVM paper-equivalence hoặc official unseen superiority. Budget3600 là observed pilot, không convergence proof. Soft text labels vẫn là hard-label supervision, không label smoothing. Kết quả negative của S1/S2 không tự chứng minh original CLIP prior tối ưu cho một fine-tuned predictor workstream.

Model V1 đã có [region-first design](designs/coupled_predictive_region_v1/region_first_protocol.md); starting architecture/trainability khác S0–S2. Tiếp theo cần xác nhận phạm vi implementation/CPU gates và các diagnostic/schedule/selection details trước launch; không tự chạy GPU campaign mới từ yêu cầu thống kê.

### Reproduce reporting/replay (fresh output directory)

```bash
# Data-read-only local verification; writes its own derived reports.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python outputs/semantic_text_execution_20260908T011006Z/local_verification_20260908/verify_local.py
.venv/bin/python outputs/semantic_text_execution_20260908T011006Z/parent_summary.py
# Example selected replay, no optimizer update; never reuse an output directory.
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_masked_view.py --run-result outputs/semantic_text_execution_20260908T011006Z/runs/semantic_text_S2/run_result.json --selection best_clean --output-dir <NEW_DIR> --device cuda
```

Raw verifier/replay bundles và tensors giữ local. Source/metrics receipts trong compact JSON được commit, không push hay upload dataset images. Báo cáo tổng hợp W&B là một reporting run riêng, không resume/ghi đè training histories.
