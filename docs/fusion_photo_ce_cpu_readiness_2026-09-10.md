# B — Photo→text CE trên nền F2_MP: code + CPU readiness

Ngày 2026-09-10. Parent HEAD trước thay đổi: `4cd947019a94494a36cfaa387b0cad65a8d4a6cb`.

## Phạm vi user đã duyệt

- Chọn **B trước**, chỉ implementation + CPU; chưa GPU, pretrained/data preflight, calibration hoặc training campaign.
- User chọn **λ tường minh, chưa chọn giá trị production**. Không mặc định λ=1, không tái sử dụng λ của MP/SIG/TQMP.
- A1/A2 có thể kiểm tra ở nhánh riêng sau; chưa triển khai. C/D và deep coupling không thuộc milestone này.
- Giữ F2_MP-Q@3600 làm fixed q reference; không đổi historical F2_MP μ_I readout/metrics, không promote candidate chưa train.

## Công thức và routing đã implement

Arm `F2_MP_PCE`, method/campaign `coupled_predictive_fusion_mp_photo_ce_v1`, architecture `predictive_fusion_v2`.

\[
L=L_{F2\_MP}+\lambda_{photoCE}\frac1{N_p}\sum_{j=1}^{N_p}
-\log\frac{\exp(\hat p_j^\top\hat t_{y_j}/.07)}{\sum_{c\in C_{train}}\exp(\hat p_j^\top\hat t_c/.07)}.
\]

- `p`: actual **unique live photo bank**, không phải sketch-derived μ_I. `t`: existing learned train-class text bank, không phải T0; cả hai giữ autograd.
- Mean một lần trên toàn bank mỗi update, bao gồm ảnh xuất hiện dưới vai trò positive hoặc negative của triplets; dùng **photo_labels** của từng ảnh. Không chỉ CE trên positives, không nhân đôi theo clean/masked views, không tạo thêm encoder forward.
- Giữ main MP(μ_I)1, CE_i1, CE_t.25, q paired rank/CE .25 mỗi term, align_i/t .05 mỗi term, anchors photo/text .5 mỗi term; mean hai sketch views như F2_MP. Không thêm MP(q)/MP(μ_T), không SIG.
- Sampler vẫn B32, một positive + một negative/sketch, bank≤64 unique photos từ full58,950 pool. Không sửa sampler/masks/preprocessing/model/optimizer/schedule.
- **Training main head μ_I; evaluation q** bằng QOnlyAdapter, predictor bypass. Không đưa labels/text targets/teacher vào query inference.
- Primary metadata: clean/full_mAP@3600 so fixed F2_MP-Q q `.49166918150172645`; masked reference `.3557525980363653`. Best aliases vẫn auxiliary prefix AP200, không peak-vs-fixed comparison.

Photo-CE riêng có gradient vào `photo_model.photo_prompt` và `text_bank.context`, **không trực tiếp vào student/predictor/pooled_head**. Toàn loss vẫn cập nhật các thành phần cũ. Đây là ràng buộc bổ sung cho prompts đang thích nghi, không phải khẳng định original CLIP thiếu quan hệ image–text hay rằng CE sẽ tăng mAP.

## API và compatibility

- `coupled_region_loss(..., lambda_photo_ce=None)`: mặc định giữ nguyên historical keys/math/gradients.
- `lambda_photo_ce=0`: tính/log raw `photo_ce`, nhưng total và gradients giữ exact F2_MP.
- CLI `--arm F2_MP_PCE --campaign-id coupled_predictive_fusion_mp_photo_ce_v1 --lambda-photo-ce <explicit>`; **không có giá trị production mặc định**. Đây là mô tả API, không lệnh launch đã được duyệt.
- Thiếu λ, bool/nonnumeric/negative/NaN/inf và integer không biểu diễn được hữu hạn bị reject; arm cũ từ chối cờ này, kể cả giá trị0. Sai campaign/diagnostic bị reject trước device/data.
- Low-level enabled photo-CE chỉ chấp nhận fusion_v2 + existing supervised MP objective + noSIG. CLI dùng `argparse.SUPPRESS`, không chèn key mới vào config mặc định của arm cũ.
- λ, source `explicit_argument_no_calibration_claim`, temperature/reduction/target, training/inference heads được truyền tới resolved config, checkpoint, run_result và W&B config allowlist. Scalar loss logging dùng tên `photo_ce`, không đổi nghĩa `ce_i`.
- **Không mở rộng launcher** `scripts/run_fusion_multipositive.py` trong milestone CPU này. Production lambda, real gates, launch integration và training cần quyết định/ủy quyền tiếp theo.

Files:
- `src/spica/coupled_predictive_losses.py`
- `src/spica/data/coupled_training.py` — chỉ arm routing, không đổi loader/batch.
- `src/spica/train_coupled_predictive.py`
- `scripts/check_fusion_photo_ce_cpu.py`

## CPU evidence

Tất cả dùng existing `.venv`, CPU hidden-CUDA/offline/W&B disabled, không dependency upgrade.

| Gate | Evidence | Kết quả/phạm vi |
|---|---|---|
| New parent checker | `outputs/fusion_photo_ce_cpu_parent_20260910T092000Z/receipt.json` | **4 PASS /0 FAIL/0 SKIP** |
| Historical MP/QMP/TQMP regression | `outputs/fusion_photo_ce_regression_20260910T092500Z/{MP,QMP,TQMP}/receipt.json` | **5+6+5 PASS /0 FAIL/0 SKIP** |
| Independent mocked trainer smoke | `outputs/fusion_photo_ce_independent_20260910T092000Z/train_smoke/` | Actual `_train_impl`,2 tiny CPU updates; real data/cache/CLIP/provenance mocked; W&B forbidden |
| Corrected independent CPU review | `outputs/fusion_photo_ce_independent_20260910T092000Z/safe_review/receipt.json` | Raw status **PASS_WITH_UNVERIFIED_PROVENANCE**; formula/gradient/safe-load/metadata scoped PASS; mock source archive not authenticated |
| Parent final binding/restore | `outputs/fusion_photo_ce_parent_review_20260910T093500Z/receipt.json` | **SCOPED_CPU_READINESS_PASS**,20 parent+regression checks bound; actual saved optimizer restore exact |

New checker verifies:
- Archived MP scalar/gradient parity, unchanged auxiliary terms, explicit-zero total/gradient exact.
- Independent CE expression and weighted gradient composition; synthetic λ **.37 is a fixture, not a production choice**.
- Noncontiguous class IDs `[2,7]`,3 queries sharing a positive,4 unique bank photos with every row used and negatives from other classes.
- One existing model/photo/reference/text call per total; no added photo forward.
- Photo-only gradient route and47 tiny active full-loss gradient tensors finite/nonzero; one tiny AdamW step and model/optimizer restore; original teacher unchanged.
- Invalid CLI values/routes rejected before `_device`; q full-vs-bypass exact, predictor0calls.

Independent safe review uses a separately written `logsumexp-minus-correct-logit` oracle, not just the parent CE helper. Mock trainer has84 protocol class IDs but a **tiny2-class model/B3**, explicitly not a B32/realCLIP certification. Config/checkpoint/run_result/W&B allowlist carry identical new metadata. Parent separately loads saved step0/2 with `weights_only=True` + scoped NumPy whitelist and calls optimizer `load_state_dict`:47 states/94 moment tensors at step2, exact restore,0 further updates. Loader/batch/pool functions AST-equal archived MP; model unchanged.

### Preserved review defects and limits

- Original independent smoke script contained a prohibited `weights_only=False` fallback. Its receipt records safe loading succeeded, so fallback was not taken. **Old script is preserved but not approved for reuse**; the new `safe_review/` child fails closed without unsafe fallback/global safe-list mutation.
- Original review overstated optimizer reload; safe review checked state structure/moments but did not itself load optimizer state. Parent final script explicitly performs and verifies that restore. Repeated checkpoint loads do not establish RNG replay.
- Mock provenance has empty manifest and `synthetic-source`; its UNVERIFIED status is retained. Current source component hashes are separately bound, not called a real training source archive.
- Worker initial fixture receipt remains at `/tmp/fusion_photo_ce_cpu_check_20260910T091500Z/`; accepted parent checker improves fixture validity, weighted-gradient/optimizer/routing checks. No production failure or raw historical artifact was overwritten to force PASS.
- No real pretrained encoder/data/optimizer-memory gate, retrieval evaluation, online W&B, calibrated coefficient, performance gain or deployment-latency claim. Historical local-diagnostic full-AP FAIL and campaign statuses remain unchanged.

Re-run the committed CPU checker with a **fresh** output path:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python scripts/check_fusion_photo_ce_cpu.py --output outputs/<fresh_cpu_gate>
```

Ruff on the four changed code files and `git diff --check`: PASS. Existing MP/QMP/TQMP standalone checks ran; no claim that the entire historical pytest suite was rerun.

## Next boundary

B code/CPU milestone complete. Await explicit choice/calibration of λ and separate GPU/training authorization. A remains separate future work. Preserve pre-existing `.gitignore`, unfinished `scripts/verify_coupled_campaign.py`, and `outputsnewgate/`; local scoped commit only, no push.
