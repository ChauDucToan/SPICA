# MP(μ_T) rồi MP(q) — sequential initialization diagnostic (2026-09-10)

## User choices and implementation scope

User explicitly requested adding MP(μ_T), then measuring the coefficient on MP(q). Clarifications accepted:
1. **Measure λ_T first**, rather than fix MP(μ_T)=1.
2. Measure at **common fresh initialization**, not F2_MPcheckpoint3600.
3. **ρ_T=ρ_q=.1**; four B32 training batches. λ_T calibrated on last student block; λ_q additionally constrained by pooled-head norm, taking the smaller candidate.

Implemented standalone **`scripts/diagnose_fusion_mp_tq.py`**. Existing model/loss/data/trainer/default heads **not changed**; no training arm registered. Diagnostic constructs the real differentiable extended loss using existing production F2_MP base graph and same live bank. No SIG/optimizer/update/W&B/network/validation-metric search.

Define `B = L_F2_MP` (full total, not main rank alone). B retains MP(μ_I)1, CE_i1, CE_t.25, q paired-rank.25/CE.25, align_i/t.05, anchorsI/T.5, clean/masked50–50.

`T = .5[MP(μ_T_clean)+MP(μ_T_masked)]`, `Q = .5[MP(q_clean)+MP(q_masked)]`.

Candidate **`B + λ_T T + λ_q Q`**, not the failed QMP transfer `B−MP(μ_I)+MP(q)`. `MP` is existing average-per-positive live-bank contrastive loss, tau.07, no detach of query/live gallery. Same full58950photo pool,1positive+1negative/sketch,≤64unique photos/batch. One model forward of64views plus one live bank forward/batch; new losses reuse captured tensors. No labels/targets enter model forward. μ_T is the predicted sketch-side head, not the global class text bank.

## Actual measurement

Root **`outputs/fusion_mp_tq_diagnostic_20260910T001200Z/`** (directory label reserved; actual result finished00:11:37.905021Z).

- Diagnostic source573files SHA **`42485ce4bfc86dc296cba91d997a3d9a810c1859a82d87eb9566bc7d1705b53f`**; base HEAD `b685377` plus uncommitted diagnostic source at measurement. Later documentation/commit is not this archive identity.
- Script SHA **`d7aeb62b8ab7715731789a0e440491b30263f95bd937377bd4b91191f8cbe9e8`**.
- Cached ViT-B-32-quickgelu, seed42, float32,84trainclasses. Full initialization hashes exact historical F2_MP initialization.
-128observations and mask rows equal first128historical F2_MP records (strip only trainer-added step/LR annotations for trace comparison, preserve zero_based_step). Same clean/corrupted/photobank input hashes captured; no model update between batches.
- Four disjoint gradient scopes: last student block12parameters/7,087,872coordinates, predictor, pooled head, shared prompts. Four components: B,T,Q and weighted text CE `CE_i+.25CE_t+.25CE_q`, mean views. Text CE is diagnostic only, not added twice.
-16`autograd.grad` calls; all raw gradients saved CPUFP32. Missing Q→predictor/T→pooled gradients explicitly recorded and flattened as exact zero, not hidden failed derivatives. Q also has no direct text-context gradient. Parameter `.grad` buffers remainNone; global RNG and model state guards unchanged; original frozen CLIP unchanged. No optimizer constructed.
- Peak CUDAallocated7,887,155,712bytes/reserved8,220,835,840. Not deployment memory/latency or full-optimizer certification.

## Sequential selection

For four fixed-state batches, gradient norms and arithmetic evaluated in FP64:

\[
\lambda_T=.1\operatorname{median}_b\frac{\|g_B^{last}\|}{\|g_T^{last}\|},\qquad
 g_{B1}=g_B+\lambda_Tg_T.
\]

\[
\lambda_q=.1\min_{s\in\{last,pool\}}\operatorname{median}_b
\frac{\|g_{B1}^{s}\|}{\|g_Q^{s}\|}.
\]

λ_T is fixed before computing λ_q. Stage2 is exact gradient linear combination at the same unchanged initialization, **not** an intervening training update or new sample set.

|Coefficient|Value|
|---|---:|
|**λ_T**|**0.17877235601108843**|
|λ_q from student block only|0.2146793430374309|
|λ_q from pooled head|0.023774345199536452|
|**Selected λ_q**|**0.023774345199536452**|

Pooled-head constraint binds. Ignoring pooled head would select **9.029874×** more MP(q) weight. The added Q gradient would then be roughly72–98% of pooled-head baseline norm on these batches, not10%.

Measured contribution ranges with selected coefficients:

|Scope|λ_T·T / B gradient norm|λ_q·Q / B1 gradient norm|
|---|---:|---:|
|Last student block|7.11–11.66%|1.004–1.285%|
|Pooled head|0%|7.95–10.89%|
|Predictor|7.15–11.69%|0%|
|Shared prompts|10.72–16.50%|0.943–3.207%|

**10% is a median-ratio calibration target, not a strict cap per batch or every parameter group.** In particular λ_T was not separately capped on prompts. CV of calibration ratios: T/last19.30%, Q/last10.91%, Q/pool12.52%; descriptive only, no posthoc threshold/search.

## Gradient-direction findings

- T vs B cosine on last student block **.614–.802**, positive4/4; predictor .549–.767. MP(μ_T) generally reinforces the baseline direction at initialization, not necessarily complementary new information.
- Q vs B1 last-block cosine **−.0133 to .0667**: weak directional alignment, two small negative values. Not strong whole-student anti-alignment.
- Q vs text-CE last-block cosine **.0179–.0613**, pooled **.1207–.2138**, positive4/4. These initialization measurements **do not establish the strong q–text conflict hypothesized from late-training QMP logs**.
- Q vs B1 shared-prompt cosine negative4/4, **−.6793 to −.0356**. These are aggregate prompt-space gradients, not evidence all prompts or the whole model conflict. Selected Q contribution there is only .943–3.207% in these four batches.
- Final-vs-B gradient cosine stays .99793–.99851 on last block, .99474–.99702 pooled, .99764–.99835 predictor, .99159–.99878 prompts. No claim this remains true through training.

Interpretation: keeping main MP(μ_I), adding a modest MP(μ_T), and making MP(q) substantially weaker than1 is a **measured initialization-scale candidate**. It has not been shown to improve retrieval. Norm/cosine balancing does not optimize mAP, equalize AdamW parameter updates, or prove absence of later conflict/collapse/overfit.

## Verification and artifacts

- Parent implementation review replaced oversized worker prototype with shared-helper standalone diagnostic; worker prototypes/failure receipts preserved at `outputs/fusion_mp_tq_worker_intermediates_20260909T235752Z/` (new work products relocated from accidentally nonignored worker directory, no historical training artifacts touched). Not used on GPU.
- Final CPU `outputs/fusion_mp_tq_cpu_final_20260910T001800Z/receipt.json`: **4PASS/0FAIL/0SKIP**. Production base scalar/gradient exact, real new-head MP oracle/routing, toy coefficient selection/zero rejection, actual tiny combined-loss autograd gradient linearity relativeL2<1e-5, model unchanged.
- Independent pre-GPU CPU/static review `outputs/fusion_mp_tq_independent_20260910T000500Z/`: same4PASS plus distinct four-batch non-proportional toy reference. Only subsequent source edit removed one unused import; parent SHA-reconstruction checked exact sole difference and reran finalCPU/Ruff.
- Real raw result **MEASURED_PENDING_REVIEW** retained, SHA `153b47573c0893b794562526dd723ed82f8abbb3c69b94ee5b3e5fe1c6ff633b`.
- `independent_cpu/verify_cpu.py` recomputes all64raw gradient vectors' statistics in NumPyFP64, no model/encoder replay: **2041checks PASS**, maxall-scalar delta6.173e−13. λ_T independently .17877235601109107, λ_q .023774345199536473. All8artifact hashes,573archivefiles,128trace/mask comparisons, source/layout/unused-zero contracts verified. Model-before/after evidence is producer state guards, **not independent model reconstruction**.
- Independent receiptSHA **`195419b095b6dcf7dcfa0347e50cf693f71feee4db6a552b5392bddcfadf7578`**.
- Parent `verify_parent.py` + separate **`diagnostic_verified.json` PASS,14boundfiles**. Verification scope **RAW_GRADIENT_CALIBRATION_ONLY**, not training launch gate. Raw result not rewritten toPASS.

Raw full precision: `diagnostic_result.json`, `lambda_selection.json`, `measured_rows.json`, four `raw_gradients_batchXX.pt`, `data_records.json`, `gradient_layout.json`, independent/parent receipts. No GPU failure/retry in accepted measurement.

**No new training launched.** No production objective/default/inference/head policy changed, no MPμ_T training run or λ search by validation results. Future training requires explicit authorization and separately implemented/versioned trainer route, actual optimizer/checkpoint gates, provenance binding and q-readout control. Preserve unrelated `.gitignore`, unfinished V1 verifier diff and outputsnewgate/.
