# Coupled predictive campaign — protocol và architecture comparison

**2026-09-08 · User đã duyệt diagnostic + trainer + chạy ngầm tuần tự R0 → R1 → R1+SIG, online W&B.**

Execution root đã đặt trước launch: `outputs/coupled_predictive_execution_20260908T153000Z/`; xem `runtime.json` để biết trạng thái thực tế, không suy trạng thái từ tài liệu prelaunch.

Đây là protocol khóa trước launch, **chưa phải báo cáo kết quả3600**. Thinning/noisy labels/official unseen/multi-seed/lambda search không nằm trong campaign. R1+SIG dùng tên máy `R1_SIG`; không thêm arm thứ tư cho “kiến trúc ban đầu”.

## 1. Phạm vi đơn giản hóa code

User xác nhận chỉ dọn **V1**, không toàn repo. Đã xóa5 pytest files V1 (`test_coupled_predictive_{model,losses,preflight}.py`, `test_coupled_views.py`, `test_sigreg.py`), bỏ shape/dtype/norm/class/config guards khỏi pure V1 model/loss/SIGReg và evaluator wrapper. Input validation tập trung ở `src/spica/data/coupled_training.py` và `coupled_views.py`. Historical tests/modules/artifacts giữ nguyên.

Các script diagnostic/preflight/verifier vẫn là executable evidence gates; launcher/trainer giữ no-overwrite, failure receipts, finite-update/IO safety. Đây không phải bỏ kiểm chứng hoặc coi NaN/ghi file lỗi là thành công. Không pytest suite mới; không nâng dependencies.

## 2. Architecture: R0, R1, R1+SIG và ý tưởng ban đầu

### Phần chung của ba arms

```text
Original sketch → clean + region-corrupted → student visual (fine-tuned riêng)
                                               │
                                    H:49 patch tokens ×768
                                               │ mean
                                               g:768 (không normalize)
                                               │ W_pool
                                               q:512 (normalize)

Photo → original frozen image tower + C_I(3×768) → photo bank live
Class names → original frozen text tower + C_T(4×512) → text bank live
Photo / hard class template → original CLIP không prompts → references detached
```

Original/reference CLIP hoàn toàn frozen; student là bản copy độc lập. Native student ln_post/proj không dùng bởi patch-token route nên frozen. Text bank gồm toàn84 train classes; C_T khởi tạo từ `a photo of a`. Shared prompts là global parameters, không chứa true label/target embedding của query.

### R1 và R1+SIG có thêm predictor

```text
C_I → Linear(768→256) ─┐
C_T → Linear(512→256) ─┴→ global prompt queries
H   → Linear(768→256) ───→ visual keys/values
                           │ 1 cross-attention block,4heads
                           │ residual / LN / MLP / output512
                           ├→ μ_I: photo prediction
                           └→ μ_T: text prediction
```

Predictor có1,183,744 parameters. `μ_I` là retrieval query đã chốt; `q` chỉ auxiliary trong R1/R1+SIG. Gallery luôn dùng C_I tại cùng checkpoint. Không chọn q/μ/fusion post hoc; không target photo/text features, true query label hay clean-sketch latent truyền sang masked predictor.

| Thành phần | R0 | R1 | R1+SIG | Ý tưởng ban đầu của user |
|---|---|---|---|---|
| Student riêng, fine-tuned | Có | Có | Có | Có, sau khi chốt option2 |
| Corruption | Region deletion | Region deletion | Region deletion | Ban đầu thinning/deletion; đã chọn region-first |
| Pooled branch q | Main retrieval | Auxiliary | Auxiliary | Có pooled auxiliary |
| Shared CLIP–predictor prompts | CLIP prompts, không predictor | Có | Có | Có |
| Predictor photo/text | Không | Có | Có | Có |
| Main retrieval | q | μ_I | μ_I | Đã cụ thể hóa thành μ_I |
| SIGReg trên g | Không compute | Không compute | Fixed-λ | Có trong ý tưởng |
| Frozen original references | Photo + text | Photo + text | Photo + text | Được bổ sung/chốt trong thiết kế |
| Positive prediction | Không | Cosine, fixed-κ equivalent | Như R1 | Đề xuất vMF; chưa là learned uncertainty |
| Negatives | Photo rank + text CE | Main rank/CE + pooled aux | Như R1 | Muốn positive prediction lẫn discrimination |

**R1+SIG là hiện thực region-first gần nhất với kiến trúc đã cùng chốt**, không phải tuyên bố exact TC-JEPA/LeJEPA/SeCo reproduction. SIGReg chỉ thêm objective, không thêm trainable predictor/head hay thay inference architecture. R0 có87,852,544 trainable elements/153 tensors; R1 và R1+SIG có89,036,288/173. R0 vẫn tiêu thụ cùng predictor initialization RNG draws rồi bỏ module trước sử dụng, để student/pool/prompts khởi tạo byte-identical giữa arms.

Không EMA teacher, không hint token vào student, không learned κ/uncertainty, không label-noise/missing-label mechanism. Same-class different-instance photo vẫn là positive, không hard negative.

## 3. Loss khóa

Mỗi view v có `rank_pool(q)` và `CE_pool(q)`; R1 thêm `rank_i(μ_I)`, `CE_t(μ_T)`, positive `align_i/align_t`.

- Rank: mean `softplus(0.2 + cosine_negative − cosine_positive)`, live photo gradients giữ nguyên.
- CE: hard-label train-class CE, temperature0.07; learned bank không detach.
- Align/reference: mean `1−cosine`, target detached.
- References: photo anchor trên unique sampled photo IDs; text anchor trên toàn84 classes, mỗi loại1 lần/update.

**R0:** mỗi view rank_pool và CE_pool hệ số0.5; photo/text anchors0.5 mỗi loại.

**R1:** mỗi view rank_i/CE_t0.5, rank_pool/CE_pool0.125, align_i/align_t0.025; photo/text anchors0.5 mỗi loại.

**R1+SIG:** loss R1 cộng `λ × [SIGReg(g_clean)+SIGReg(g_corrupted)]/2`. Mỗi view là `[32,768]`; không flatten64 correlated views hoặc patch tokens thành independent samples.

R1−R0 đo **predictor package**, không tách riêng lợi ích số layers/parameters. R1+SIG−R1 cô lập SIGReg. S0 prompt-only là external reference, không phải matched R0.

## 4. Diagnostic SIGReg đã đo và verified

- Root `outputs/coupled_sigreg_diagnostic_20260908T135446Z/`.
-4 batchB32 đầu của train loader seed42; cùng initial model, mask steps0/1/2/3.128 original sketches; không optimizer/model update, không validation-driven tuning.
- Last student transformer block:12 parameter tensors,7,087,872 flattened coordinates.
- Numerator: gradient của main task `(rank_i+CE_t)` averaged hai views, **không** toàn bộ auxiliary/reference loss.
- Denominator: gradient của mean per-view SIGReg. Raw FP32 gradients lưu `.pt`; audit norm/ratio/cosine bằng float64.
- SIGReg pinned MINIMAL:17 nodes[0,3],256 normalized Gaussian directions, private CPU RNG seed42, resample mỗi view/call. Training R1+SIG khởi tạo lại seed42, không tiếp tục RNG đã tiêu thụ bởi diagnostic.

| Batch | Norm task | Norm SIG | Ratio | Cosine |
|---:|---:|---:|---:|---:|
|0|2.313816406|34.833792149|0.066424476|−0.026768924|
|1|2.632752605|34.661743829|0.075955573|−0.103469011|
|2|3.069557101|34.519401346|0.088922663|0.003740791|
|3|2.931363838|36.770925543|0.079719610|−0.009273069|

**λ = 0.1 × median(ratios) = `0.007783759129112511`**, giữ nguyên raw precision xuyên arm. CV0.103726; max/min1.338703. Parent và independent reviewer recompute raw gradients/552-file source archive PASS. Spread vừa phải trong4 batches; không có numeric stability threshold được đặt sau khi xem kết quả.3/4 cosines âm nhẹ: không coi scale matching là loại bỏ gradient conflict.

Raw `diagnostic_result.json` giữ `MEASURED_PENDING_REVIEW`; quyết định dùng λ nằm riêng ở `diagnostic_verified.json`, không sửa raw evidence. Diagnostic source SHA `8f7c7687c0d6c0ba6af20c3ada1b5075e550570def3ec25deb1931f10714b050`. Training source sau docs/gates có thể khác; bind bằng exact component hashes + initialization + immutable diagnostic receipt, không ép full-source equality giả.

Đây là **gradient-scale initialization**, không λ tối ưu hoặc đảm bảo10% AdamW update/khả năng generalize. Không lambda search.

## 5. Data, budget và scheduler

- Seed42, pseudo split3407:84 train/20 validation; official21 classes không load/evaluate.46,624 train sketches,58,950 train photos,8,400 canonical positive pool; positives sampled same-class, không gọi exact-pair experiment.
- Batch32, two views50/50;3600 updates/arm,115,200 original observations và230,400 sketch-view forwards. Probes0/600/1200/1800/2400/3000/3600.
- Region helper `ink_centered_square_v1`, threshold0.9; fractions.25/.5/.75 bằng local uniform RNG; seed `mask_seed(4242+zero_based_update, relative_path, view=1)`; không generate clean mask0.
- Validation clean + macro9 (fractions.25/.5/.75 × seeds101/202/303), ordered identical queries/gallery/masks; RNG/mode preserved quanh toàn probe.
- AdamW betas(.9,.999),eps1e−8, default amsgrad false. Student LR1e−5, predictor/pool1e−4; matrix WD1e−2, biases/LN/vectorsWD0; C_I/C_T LR/WD1e−4.
- Warmup180: optimizer update1 dùng baseLR/180, update180 đạt baseLR. Sau đó cosine floor0 tại scheduler state sau update3600; actual LR được log trước mỗi optimizer step. Không silent early-stop/extend/change batch/AMP.

## 6. Gates, launch và reporting

Prelaunch runtime checks (không pytest): tiny CPU common-init/gradient/SIGReg formula/scheduler PASS; actual2-update AdamW smoke từng R0/R1/R1+SIG PASS, checkpoint/model-hash/optimizer/SIGReg state roundtrip PASS.64 observation/mask rows của smoke byte-identical ba arms. Real full R1@2 probe10,963 queries/13,999 gallery,9 conditions/98,667 masks PASS; không xem metric smoke là kết quả campaign. Peak allocated có optimizer≈7.96/7.99/7.99GiB; đây vẫn là2 updates, không convergence/training-memory guarantee mọi tình huống.

Online W&B preflight run `vfyyww28` verified3 unsampled steps và3 downloaded artifact aliases; dữ liệu scalar của gate được gắn nhãn synthetic, không phải retrieval results. Campaign runs ở `a-cctest05187-erd/spica`, mỗi arm run riêng;361 scalar-history rows dự kiến, merged probe logs tránh drop trùng step.

Runner `scripts/run_coupled_campaign.py` inert nếu không `--launch`; background process tuần tựR0→R1→R1_SIG, stop-on-error/no-retry/no-resume. Không sửa source/config/docs khi arms đang chạy. Checkpoint7 mốc immutable, selected alias JSON `latest/best_clean/best_masked`; large teacher weights tái dựng từ verified CLIP cache, không serialize lại600MB mỗi checkpoint. Optimizer/scheduler/Python/NumPy/Torch/CUDA/loader/SIGReg RNG được lưu; **chưa claim exact resume**, không tự resume sau reboot.

`best_clean`: clean prefix AP200 cao nhất, positive steps only, earliest tie. `best_masked`: masked macro9 prefix AP200 tương tự; `latest`3600. Gate tại từng arm's best_clean: candidate clean prefix strictly tăng, P200/full mAP không giảm, masked macro9 prefix strictly tăng. Report R1 vsR0, R1+SIG vsR1 và vsR0; không peak-vs-fixed control. Negative promotion không phải verification failure.

Sau ba arms, runner tự gọi verifier: source/config/checkpoint/data/LR/trace/selection checks; per-query AP/P200 và top-index consistency; selected CUDA replays; online unsampled histories; selected artifact downloads/SHA; summary JSON/Markdown và W&B report tables. Đủ evidence mới gọi VERIFIED. Báo cáo cuối cần bảng mọi selections/5metrics, diagnostic, architecture comparison ở trên, runtime/VRAM, promotion và one-seed limitations. Không báo training hoàn tất chỉ vì runner đã được launch.
