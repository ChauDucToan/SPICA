# V1 region-first — quyết định thiết kế mới nhất

**2026-09-08 · DESIGN SPECIFICATION · Không cấp phép launch model mới.**

**Cập nhật sau design:** User đã duyệt implement model/loss + CPU gates; [CPU milestone](../../coupled_predictive_v1_cpu_readiness_2026-09-08.md) đã PASS, gồm SIGReg formula-equivalent pinned MINIMAL. Các đoạn “chưa implement” bên dưới mô tả draft minh họa, không phải trạng thái source mới. Lambda diagnostic và GPU/training vẫn chưa thực hiện/được cấp phép.

Bản này bổ sung [architecture.md](architecture.md) và supersede các mục còn ghi “chưa chốt” về region-first, view mixing, loss starting weights, optimizer starting groups và budget. Bản nháp V1 trước đó được giữ nguyên tại `outputs/coupled_predictive_design_20260908T021035Z/` cùng primary TC-JEPA receipts. Code minh họa được soạn/kiểm tra trong output root trong lúc train. Bản tracked này được lưu sau khi runner báo cả S0–S2 COMPLETED; không thay archived source S0–S2.

## 1. User đã duyệt gì?

- Sketch encoder riêng, CLIP-initialized và fine-tuned; không hint token vào encoder. Shared photo/text CLIP prompts đi vào predictor nhỏ; original CLIP/reference frozen.
- **Region deletion trước**. Thinning viết riêng, để thử nghiệm sau, không nằm trong train/eval campaign region-first.
- Mỗi original sketch có clean và corrupted views; sketch-side losses trung bình **0.5/0.5**.
- Starting group weights: task1, pooled auxiliary0.5, positive alignment0.1, reference preservation1.
- Optimizer starting groups và budget đã được user đồng ý ở lượt tiếp theo: AdamW; student1e-5, predictor/head1e-4, prompts1e-4; effective batch32,3600 optimizer updates.
- SIGReg: chấp nhận hướng **control không SIGReg → diagnostic scale → cố định hệ số trước arm bật SIGReg**. Không có lambda đã calibrated, không mặc định lambda1. Exact implementation còn cần pin/review.
- Trình tự nghiên cứu: region-first/clean labels → predictor controls → preservation/SIGReg controls → thinning → noisy/missing labels.

**User chưa yêu cầu implement production trainer hoặc train V1.** CPU code minh họa không thay thế real tokenizer/CLIP/memory/gradient gates. LR/weights là starting values đã duyệt, không gọi calibrated/optimal hoặc faithful TC-JEPA/SeCo.

## 2. Region policy và clean/corrupted budget

Reuse **`ink_centered_square_v1`** hiện có; không viết masker thứ hai.

| Thuộc tính | Thiết kế region-first |
|---|---|
| Tensor | CPU normalized CLIP RGB224×224; mask trước encoder attention |
| Ink threshold | Mean RGB sau denormalization <0.9 |
| Requested train severities | 0.25/0.50/0.75, uniform choice bằng local RNG |
| Train base seed | 4242 |
| Sample key | Canonical relative POSIX path trong dataset; không absolute/.. |
| Mask seed | `mask_seed(4242 + zero_based_update, relative_path, view=1)` |
| Train views | Clean clone + một region-deleted clone; không tạo rồi bỏ mask cho clean view0 |
| Eval | Clean + fractions0.25/0.50/0.75 × seeds101/202/303, unchanged evaluator ordering/hash convention |
| Blank/unreachable | Giữ status/counts và output của helper; không âm thầm drop sample/retry cho mask dễ hơn |

Requested fractions/seeds reuse contract cũ để so sánh, không phải severity search/calibration mới. `masking_sketch.region_pair()` khớp masked view1 của historical full_masked route cho cùng step/path/image. Việc bỏ mask view0 không đụng global RNG; demo so trực tiếp helper và kiểm tra Python/Torch RNG.

Mỗi update dự kiến B32 original observations,64 sketch-view forwards, cùng positive/negative photo identities và text bank cho cả hai views. Có thể concatenate `[clean, corrupted]` theo batch axis rồi chunk outputs; attention chỉ trong từng sample, không cho clean H_s làm key/value của masked predictor. Student forward không nhận target photo, text label, teacher feature hoặc per-query CLIP hint.

Điểm giống B32/2views không đủ chứng nhận VRAM: student fine-tuning khác frozen-prompt runs. Microbatch/accumulation chỉ chốt sau preflight; không đổi effective batch hoặc sample counts âm thầm.

## 3. Loss đã chốt: nhóm, views và hệ số thực tế

Cho v thuộc {clean, corrupted}:

```
T_v = rank_i,v + ce_t,v
P_v = (rank_pool,v + ce_pool,v) / 2
A_v = (align_i,v + align_t,v) / 2
R   = (anchor_i + anchor_t) / 2
L_noSIG = (T_clean + T_corrupted)/2
        + 0.5 * (P_clean + P_corrupted)/2
        + 0.1 * (A_clean + A_corrupted)/2
        + 1.0 * R
```

Hệ số trên từng scalar đã mean batch/negative axis:

| Term | Mỗi view clean/corrupted | Ngoài view loop |
|---|---:|---:|
| rank_i |0.5|—|
| ce_t |0.5|—|
| rank_pool |0.125|—|
| ce_pool |0.125|—|
| align_i |0.025|—|
| align_t |0.025|—|
| anchor_i |—|0.5|
| anchor_t |—|0.5|

**Không nhân đôi reference anchors theo số views.** Text anchor là classmean toàn84 pseudo-train classes một lần/update; photo anchor mean trên unique sampled photo identities đã encode trong update. Live photo/text encode once/update, reuse graph cho cả views; reference clone ngoài inference_mode.

`model_sketch.region_objective()` thực hiện đúng bảng trên. Demo đặt mỗi scalar bằng1 thì tổng=3.6; backward kiểm tra từng hệ số gradient. Đây là kiểm tra arithmetic, không calibration.

Loss semantics giữ như V1:
- Main inference μ_I nhận softplus photo ranking; cả live photo positives/negatives có gradient tới C_I.
- μ_T và pooled q nhận hard-label CE trên learned train text bank, `detach_text=False`.
- Positive cosine alignment là fixed-kappa vMF tới hằng số/scale, targets detached; không learned uncertainty.
- Original CLIP weights/reference không update; shared C_I/C_T vẫn dịch chuyển qua live rank/CE/predictor/anchor.
- Margin0.2 và CE tau0.07 giữ làm inherited starting values, không search ở lượt thiết kế này.

## 4. SIGReg: cách chốt scale, chưa có hệ số đã đo

Arm control: `lambda_sig=0`, **không tính SIGReg**. Đây là contrast rõ ràng, không gọi full model đã chạy đủ SIGReg.

Arm bật SIGReg:

`L = L_noSIG + lambda_sig * (SIGReg(g_clean) + SIGReg(g_corrupted))/2`.

Mỗi SIGReg nhận `[B,D]` unnormalized pooled latent, mỗi row từ một original sketch trong view đó. Không flatten patch tokens thành samples; không pool64 correlated views rồi gọi là64 independent samples. Không batch-standardize để fake Gaussian conformity. Hai view losses có thể trung bình mà không giả định độc lập giữa views.

### Diagnostic proposal để review trước implementation

1. Pin faithful official SIGReg source, số projections/frequencies và randomness/view-axis contract. Legacy `SignatureRegularizer` của repo không được substitute.
2. Freeze một tập nhỏ train-only diagnostic batches và masks, dùng initial matched model snapshot; no optimizer/model update. Khôi phục RNG/state, giữ raw gradients/norms và source identity.
3. Đo trên **cùng tập parameters của last student transformer block**:
   - `g_task = grad[(T_clean+T_corrupted)/2]`;
   - `g_sig = grad[(SIGReg(g_clean_latent)+SIGReg(g_corrupted_latent))/2]`.
4. Đề xuất design value `rho=0.1` và `lambda_sig = rho * median_b(||g_task,b||_2 / ||g_sig,b||_2)` trên4 batches B32. **Rho0.1,4batches và parameter scope là chi tiết đề xuất mới, chưa phải kết quả calibrated hoặc quyết định user đã xác nhận từng số.** Cần khóa trước diagnostic.
5. Nếu norm0/nonfinite hoặc ratios không ổn định: dừng gate, báo evidence; không epsilon-divide/clip hệ số âm thầm. Anchor gradient≈0 ở khởi tạo là bình thường, không dùng làm mẫu số calibration SIGReg.
6. Cố định lambda đã đo xuyên arm; không online adaptive weighting/selection bằng validation score. Log gradient norms/cosines/drift nhưng không suy AdamW update từ raw gradient đơn lẻ.

`region_objective()` yêu cầu explicit `lambda_sig`; lambda>0 mà thiếu một trong hai per-view losses sẽ fail. Lambda0 mà vẫn đưa SIGReg values cũng fail để control không âm thầm có compute/side effects. `sigreg_term()` vẫn fail-closed khi implementation chưa được cung cấp. Scalar2/4 trong demo chỉ kiểm tra công thức trung bình, **không giả lập SIGReg**.

## 5. Optimizer/budget đã được đồng ý và chi tiết schedule còn phải khóa

| Group | AdamW LR | Weight decay |
|---|---:|---:|
| Student visual context parameters |1e-5|1e-2 cho weight matrices|
| Predictor + pooled head |1e-4|1e-2 cho weight matrices|
| C_I + C_T |1e-4|1e-4|
| Bias/LayerNorm trainable trong student/predictor |Theo group chủ|0|
| Original CLIP/reference |Không trong optimizer|Frozen|

Native student ln_post/proj không được dùng bởi patch-token path nên frozen/unused, không kể là trained. Original image/text towers, original LN/embeddings/projections/logit scale đều frozen; học LN của **bản student riêng** không thay chính sách teacher. Prompts là token parameters có WD1e-4 explicit, không tự coi là LN/no-decay.

Proposed optimizer defaults để ghi resolved config: AdamW betas(.9,.999),eps1e-8,amsgradfalse; không thêm optimizer framework. Parameter ownership phải unique bằng identity; shared prompts không bị đăng ký/update hai lần.

- **Budget agreed:**3600 optimizer updates, effective batch32, clean/corrupted64 query views/update.
- Original observations115,200 và query-view forwards230,400/arm, chưa tính probes; khoảng2.47 dataset passes trên46,624 sketches, không chứng minh convergence.
- Probes/checkpoints0/600/1200/1800/2400/3000/3600. Không early stop/extend theo curve.
- Starting schedule direction: warmup ngắn rồi cosine. **Đề xuất chi tiết chưa được user chốt từng số:** warmup180updates (=5%), cosine về0 ở cuối3600, same schedule mọi matched arms. Không implement scheduler ở draft.
- Kế thừa seed42/pseudo3407 và84/20train/validation như development proposal, official21 classes vẫn excluded; không tự train/eval official unseen.
- Real effective-batch32 memory gate còn pending. Gradient accumulation không tự tăng statistical batch của SIGReg; không detach microbatch features để giảm memory rồi gọi regularizer còn tác động student.

LR/budget acceptance không tự authorize new GPU diagnostic/train. Warmup steps/floor, rho/batch scope, faithful SIGReg pin và final comparator selection phải khóa trước execution.

## 6. Thinning có code nhưng cố ý không tham gia region-first

`masking_sketch.thinning_preview(image, iterations)`:
- Standalone binary3×3 ink erosion trên CPU, padded outside image as background.
- Chỉ thay ink pixels bị xóa bằng normalized white, giữ nguyên RGB/antialias các pixels khác.
- Có counts, realized fraction, SHA, iterations_applied, blank status.
- Nếu một vòng sẽ xóa sạch toàn bộ ink thì bỏ vòng đó/dừng; không đảm bảo giữ mọi connected component hoặc semantic identity.
- Không topology-preserving skeletonization, không true-stroke masking, không calibrated severity. Không random-pixel dropout.

**`region_pair()` không gọi thinning_preview; không có runtime policy selector bật nhầm thinning.** CPU spy check xác minh điều này. Không preview dataset thật/training thinning ở lượt này. Cần actual image previews, severity contract và matched controls trước experiment sau.

## 7. Trình tự thí nghiệm — chỉ thiết kế, chưa launch

### Giai đoạn A: region-first trên clean annotations

**R0: fine-tuned pooled control**
- Cùng student init/trainability, prompted photo/text CLIP, data order, masks, optimizer/budget.
- Query làq_s, objective hai views của `rank_q + CE_q` và reference nhóm1; không predictor/SIGReg.

**R1: coupled predictor package, no SIGReg**
- Main queryμ_I, retained pooled auxiliary, shared prompts, đầy đủ `L_noSIG` đã chốt.
- Cùng sample/negative/mask traces vớiR0, photo/text prompt initialization identical; predictor init không được làm đổi sampler hoặc student initialization RNG.

So R1−R0 đánh giá **predictor package** gồm architecture + auxiliary/regression objectives; không gọi là hiệu ứng thuần số layer/predictor. Cần thêm head-matched control nếu muốn attribution hẹp hơn. Không so riêng với prompt-onlyS0 rồi gán gain cho JEPA.

### Giai đoạn B: kiểm tra preservation và SIGReg

- Starting weights đã duyệt có preservation ON trongR0/R1. Để cô lập nó, so R1 với bản cùng cấu hình `lambda_ref=0`; không giả vờ anchor đã được chứng minh từ contrastR0–R1.
- Sau faithful diagnostic, so R1 với bản thêm fixed-lambda SIGReg, mọi thứ khác giữ nguyên.
- Tất cả là proposed contrasts, không tự launch4arms hoặc lambda search.

### Giai đoạn C/D: thinning rồi noisy/missing labels

Thinning giữ file/code riêng tới khi policy/severity được chốt. Noise/missing-label trials không nằm trong region-first:
- Bắt đầu bằng clean reference; sau đó thử noise/missing riêng, không cùng lúc.
- Phân biệt text-only noisy labels có pairing sạch với noise lan vào positive/negative sampling.
- Missing sketch label nhưng paired photo còn label sạch phải có baseline label transfer.
- Hidden gold labels không được dùng để xây positives/filter negatives cho learner. Validation ground truth giữ sạch.

## 8. Reporting/selection proposal

Mainμ_I cố định cho R1; q_s báo auxiliary, không chọn q/μ/fusion post hoc. R0 dùngq theo architecture định trước. Same-selection comparisons, không lấy candidate peak so control fixedstep.

Reuse reports clean + macro9/fraction/seed, P200 + three AP200 denominators + full mAP. Retain latest, best_clean, best_masked; earliest tie, exclude0. **Final V1 promotion gate cần chốt trước launch**, không tự kế thừa quyền selection của S0–S2. Đề xuất: giữ clean guard như trước và thêm masked absolute-improvement requirement cho robustness claim, không dùng retention từ clean baseline yếu để chứng minh thắng.

Evidence phải có run/config/source → step/checkpoint → raw metrics, optimizer/scheduler/RNG, shared prompt state, original CLIP invariance, sample/photo/mask identity, context-ablation/replay và compute. Một seeddevelopment không là statistical significance, noisy-label robustness hay paper superiority.

## 9. Code mapping và self-check scope

- `masking_sketch.py:region_pair`: clean clone + historical region view1, local RNG, metadata.
- `masking_sketch.py:thinning_preview`: parked standalone erosion, never invoked byregion_pair.
- `model_sketch.py:region_objective`: equal view losses, approved group coefficients, anchors once, explicit SIGReg control/gates.
- `model_sketch.py:SmallPromptPredictor/supervised_terms`: kiến trúc/gradient rules từV1.

Schematic update (pseudocode; photo gathers/encoders/trainer chưa integrate):

```python
clean, corrupted, mask_meta = region_pair(image, relative_path, update_index)
# Batched clean/corrupted forward is permitted, but never context cross-feeding.
clean_terms = supervised_terms(clean_branches, positive_live, negative_live,
                               text_live, class_ids, labels, positive_labels, negative_labels)
masked_terms = supervised_terms(masked_branches, positive_live, negative_live,
                                text_live, class_ids, labels, positive_labels, negative_labels)
anchors = preservation_terms(unique_photo_live, unique_photo_reference, text_live, text_reference)
# Explicit initial no-SIGReg control:
loss = region_objective(clean_terms, masked_terms, anchors, lambda_sig=0.0)
# Later SIGReg contrast must pass verified SIGReg(g_clean), SIGReg(g_corrupted)
# and the coefficient fixed by a reviewed diagnostic; never silently substitute.
```

Chỉ chạy CPU synthetic tensor/selfchecks, không optimizer step, không pretrained CLIP, không data manipulation hay new W&B. Source/hash kiểm tra giữ nguyên trong verification receipt. Đã chuyển sang tracked docs sau khi S0–S2 kết thúc. `design_verification.json` là receipt của draft trước khi chuyển, các file hashes trong đó tham chiếu bản gốc tại `outputs/coupled_predictive_region_design_20260908T025016Z/`, không phải hash các tài liệu tracked đã bổ sung trạng thái này. Không push.
