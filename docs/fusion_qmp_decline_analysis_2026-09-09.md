# F2_QMP3600 — completed, regression versus fixed F2_MP-Q

## Identity and scope

User asked why QMP looks worse. Read-only source/log/online/saved-ranking analysis; **no new GPU/model forward/backward/optimizer, no retraining**.

- Training root `outputs/fusion_qmp_execution_20260909T184000Z/`; HEAD `78e21eef2eead32f9fe3b0cb4a240e0be283fce4`.
- Source571files SHA `f8337d8aeeabb5d713d3e268528a00ff1d6b7833ccc542f4dce9da6627df7eef`.
- COMPLETE3600/exit0, finished2026-09-09T19:19:27Z. Raw runtime **ARM_FINISHED_UNVERIFIED** retained.
- W&B finished [vb9j58vw](https://wandb.ai/a-cctest05187-erd/spica/runs/vb9j58vw).
- Checkpoint3600 SHA `82de3718b593d72014c53a92ed2d1d05c3867c90f0885e355632c253558750ff`.
- Baseline fixed **F2_MP-Q@3600**, q-only evaluation of MPcheckpoint SHA `58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e`, `outputs/fusion_mp_q_evaluation_20260909T145000Z/`. Its training W&B y40hu06b records μ_I, not q. No old history rewritten.

## Fixed-step results

|Metric|F2_MP-Q q|F2_QMP q|ΔQMP−baseline|
|---|---:|---:|---:|
|Clean full mAP|.49166918150172645|.46069669041452666|−.03097249108719979|
|Clean P200|.5203552750202357|.47140152220255166|−.04895375281768404|
|Clean prefix AP200|.5354335989771134|.48932735265986904|−.04610624631724436|
|Masked full mAP|.3557525980363653|.33943117642133797|−.01632142161502731|
|Masked P200|.3585222947277403|.3335210278903649|−.0250012668373754|
|Masked prefix AP200|.37329428466850845|.35255720667670604|−.02073707799180241|

All5metrics decline in **clean and each9masked conditions**, not merely a macro anomaly. Clean4658queries improve/6305decline/0ties at1e-7 tolerance;57.51%decline,13/20classes decline. Not universal across every query/class.

Clean fullAP loss splits into top200 contribution−.015709713353411092 (50.72%) and tail−.015262777733788702. Same full-gallery-positive denominator used for both contributions; not subtracting prefixAP from fullAP. Thus no improvement-in-tail/top200 trade-off explains this regression.

QMP clean full bystep0/600/1200/1800/2400/3000/3600: `.088526/.443326/.457005/.442355/.460215/.461163/.460697`. Bestclean alias1200 selects **prefixAP200**; bestmasked/latest3600. No candidate peak vsfixed control claim; no earlier F2_MP q trajectory encoded. Different step0 metric vs historicalMPμ_I is readout difference, not different initialization.

## What the controls exclude

Independent online/local audit `outputs/fusion_qmp_execution_20260909T184000Z/analysis_readonly_20260909/online_audit/receipt.json`:

-361unsampled online rows,490retrieval scalar comparisons at7probes exact local-vs-online; config q/MPq/tau.07/noSIG correct.
-Initialization JSON/model-init hashes exactMP;115200observations/masks and3601LRrows byte-identical. Parent separately hashes these files and agrees.
-Frozen-original state before/after equal; identical model/sampler/optimizer structure. Archived model implementation unchanged, objective/evaluation/config dispatch are expected differences.
-Source snapshots independently streamed and verified; QMP571files hash above, historicalMP565files `ab064ced3b11c95f142ac94b336072d88312f5e14d776a386416bbf3cf70f29e`.

No evidence in these checks for wrong inference head, sampling/masking/init/LR drift, NaN failure, or wrong baseline scalar. These are not all-query encoder-replay proofs.

## Mechanistic evidence, not complete causal identification

The implemented transfer is correct: MP(μ_I)→MP(q), not additional MP(μ_T). Keeping coefficient1 does **not** preserve the parameter-wise gradients:

-Previously the main MP gradient traversed the query predictor and shared student, plus the live prompted photo bank.
-Now it traverses pooled_head and shared student, plus the same live photo-bank path; the query-side predictor no longer receives that MP gradient. Predictor still receives CE_i/CE_t/alignment. Shared prompts still have other gradient paths.
-Consequently q gains direct photo-MP supervision while μ_I loses it. A good q **readout** under predictor-mediated training need not improve when made the main MP training head.

Existing training logs support a changed balance. Means over **60logged batches, steps3010..3600**, matched observations but separately trained model states:

|Quantity|MP|QMP|
|---|---:|---:|
|pooled_head matrix total raw-gradient L2 norm (group4)|.286342|1.379019|
|predictor matrix total raw-gradient L2 norm (group2)|3.843025|2.404575|
|student matrix total raw-gradient L2 norm (group0)|71.236060|72.650378|
|clean q paired ranking loss|.602627|.581042|
|clean q text CE|.140866|.202346|
|clean μ_I photo alignment loss|.573099|.714747|
|clean μ_T text alignment loss|.384467|.336388|

Pooled-head matrix gradient norm is **4.816×**, while predictor gradient is smaller. These are total raw gradients before AdamW, **not** isolated MP norms, gradient-angle/conflict measurements, or4.816× effective parameter updates.

q paired photo ranking improves on observed train batches while q text CE worsens; μ_I photo alignment worsens after its MP loss was removed, while μ_T text alignment improves. This is consistent with a photo/text supervision balance shifting, not proof that any one changed scalar causes the retrieval decline. Raw MP losses on different heads (.902104→.862446 clean) and total objectives (2.307568→2.246094) are descriptive, not equal-objective comparisons.

**Supported conclusion:** this exact MP-routing intervention reduces held-out pseudo-class retrieval in this seed, despite fitting its in-batch training objectives. The initial inference hypothesis “q reads out better, so stronger MP(q) should train better” was not supported. More specifically blaming gradient conflict, collapse, overfitting, semantic loss or gallery drift requires additional no-update measurements; existing logs cannot distinguish them conclusively. No claim that more steps, MPμ_T or a new weight automatically fixes it.

## Verification boundaries and follow-up

Parent reproducible CPU script `analysis_readonly_20260909/analyze_saved.py`, report `saved_analysis.json`: IDs/labels and98667maskrecords exact baseline; all-query saved top200 relevance/P200/all3AP200 semantics independently recomputed for10conditions. FullAP scalar checked against saved per-query means only; **no independent full-gallery re-sort or QMP CUDA replay**. Every inputfile bound by SHA. Initial analysis TypeError treating per-group gradient dict as scalar retained in `saved_analysis.log`/`analyze_saved_before_gradient_schema_fix.py`; corrected only analysis, new outputlog, no training/data changes.

W&B metadata listing currently exposes **zero logged artifacts** for this run, independently confirmed by parent `parent_artifact_metadata.json`, despite train stderr claiming artifact upload/sync. Histories and local selected checkpoint hashes are verified; **no online selected-alias/download verification claim**. This artifact visibility issue does not explain local retrieval decline and has not been repaired or used to trigger training.

Retain F2_MP-Q as the stronger fixed3600candidate within this comparison. Next diagnostic, if authorized: same checkpoint/train batches evaluate all three heads, isolated MP-vs-text gradient norms/cosines on student/pool/prompts, and gallery/query feature geometry. No updates; not MPμ_T training yet. Existing controls reused. Do not switch policy or launch this automatically.
