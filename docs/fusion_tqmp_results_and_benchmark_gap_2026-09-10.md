# F2_TQMP results and SketchLVM/SeCo comparison audit (2026-09-10)

## Correct the premise: improved vs QMP, not vs the strongest q control

User asks why adding text improves clean full mAP and what SPICA lacks versus SketchLVM/SeCo. TQMP has just completed3600; new analysis is read-only, no GPU/model replay or training.

Training root `outputs/fusion_tqmp_execution_20260910T014500Z/`, HEAD `670856b020489312e88b734ed422a304f009bc47`, source576files SHA `b8eb37c7446aae4227f0560f7bc70c53970d17aab46681227b8fb8b13b8ba4ca`. W&B [wwje6auk](https://wandb.ai/a-cctest05187-erd/spica/runs/wwje6auk), COMPLETE3600/exit0, finished02:23:58Z. All3selected aliases latest/best_clean/best_masked point3600, checkpointSHA `e914f6d38bed502a0357c221dc79131cb9b2f17e05860ec371a77b03f1965dfc`. Raw runtime **ARM_FINISHED_UNVERIFIED**, not rewritten.

|Run/readout, fixed3600|Clean full mAP|Clean P200|Clean prefix AP200|Masked full mAP|
|---|---:|---:|---:|---:|
|F2_MP-Q q|**.49166918150172645**|**.5203552750202357**|**.5354335989771134**|**.3557525980363653**|
|F2_QMP q|.46069669041452666|.47140152220255166|.48932735265986904|.33943117642133797|
|F2_TQMP q|.4853425143738697|.5051732998072705|.5198776717818485|.35426667650254323|

TQMP−QMP cleanfull **+.02464582395934306**. TQMP−MP-Q **−.00632666712785673** cleanfull, masked **−.00148592153382204**. No promotion over the fixed q control. Across clean+9conditions+maskedmacro (55metric-condition cells),54decline vsMP-Q, only75%mask/seed202 fullmAP improves (+.000560303). No claim every condition/all5metrics decline.

Why this is not a causal text ablation:
- All prior F2/MP/QMP arms already had text CE and text alignment.
- MP(μ_T) trains a **sketch-derived predictor head** against live photos. It is not new captions/text data or injecting true class names into query inference.
- Relative to QMP, TQMP simultaneously restores main MP(μ_I), reduces MP(q) coefficient1→.0237743452, and adds MP(μ_T)×.1787723560.
- Relative to MP-Q, TQMP adds two terms together and remains slightly worse. Their individual effects are not identified.

Mechanistic hypothesis: adding photo-MP to a text-supervised μ_T can connect its semantic discrimination to photo retrieval while retaining the μ_I path. Smaller q-MP avoids routing the main gradient directly into pooled head. But the experiment only tests the package, not that explanation separately.

Observed last60logged batches3010..3600: q text CE MP/QMP/TQMP `.140866/.202346/.146456`; pooled matrix raw gradient norm `.286342/1.379019/.2983`. TQMP resembles MP more than QMP on these quantities. Not isolated component gradients or proof of effective AdamW-update behavior. The initialization calibration alone cannot establish improved mAP.

## Verification tiers

`analysis_readonly_20260910/verify_readonly.py`, `summary.json`, `receipt.json`:361unsampled W&B rows,490retrieval scalar checks initially with1e−6 tolerance; localselectedcheckpoint hash checks, summary/log means. Worker summary described controls/archive as verified more strongly than its script established; parent explicitly completed those checks rather than accepting that wording.

`analysis_readonly_20260910/verify_parent.py`, `parent_receipt.json`:
-All490local-vs-online retrieval scalars **exact delta0**.
-Streamed all576archivedfiles and aggregate sourcehash, actualcheckpoint3600 hash.
-Initialization JSON,115200observation/mask records and3601LRrows **byteidentical to MP and QMP**.
-Logged frozen-original before/after equal; **no independent full model-state replay**, no selected artifact download/online alias certification, no full-gallery independent sort or per-query ranking audit in this turn.
-Parent receipt SHA references worker script/history/inputs/summary; raw run/evidence left untouched.

## External comparison: gap has not been measured on a matched benchmark

Primary sources reread2026-09-10:
1. [SketchLVM paper2303.13440v3](https://arxiv.org/html/2303.13440v3), §4/§6/Table1.
2. [Official Sketch_LVM repository](https://github.com/aneeshan95/Sketch_LVM), describes release as a workable basic version.
3. Pinned [model_LN_prompt.py](https://raw.githubusercontent.com/aneeshan95/Sketch_LVM/ca33be98cb9f986813f611c997ceaa474b0bc96e/src/model_LN_prompt.py) and [dataset_retrieval.py](https://raw.githubusercontent.com/aneeshan95/Sketch_LVM/ca33be98cb9f986813f611c997ceaa474b0bc96e/src/dataset_retrieval.py).
4. [SeCo-SBIR2608.03120v1](https://arxiv.org/html/2608.03120v1), §4/§5/Table1. §5 implementation details also read from previously SHA-recorded same-v1 HTML `outputs/semantic_baseline_step0_20260907T124707Z/seco.html` because current web output was truncated.
5. [SeCo public repository contents](https://api.github.com/repos/huutuan1705/SeCoSBIR/contents): only index.html visible at this check; no claim authors have not run experiments.

**SPICA here:**84pseudo-train/20pseudo-validation classes,10963queries/13999full-gallery photos, pseudo split3407, original official21classes unused. PrimaryfullAP over allgallery/categorypositives; prefixAP200/allrelevant/min(R,200) reported separately.

**Paper numbers:** SketchLVM Sketchy `.723` is labelled **mAP@200**, P200 `.725`, official104/21 split. SeCo Sketchy-Ext-2 `.800` labelled **mAP@200**; Sketchy-Ext-1 `.806` labelled mAP@all on a different table column/split. Cannot subtract SPICA `.485343 full_mAP` from these and call it a measured architecture gap. Exact paper AP200 denominator remains unresolved. Even SPICA `.519878 prefixAP200` is not automatically equivalent.

**Released code is not a certified paper evaluator:** pinned Sketch_LVM computes retrieval_average_precision without explicit top-k (full AP on its constructed gallery), concatenates one sampled same-class photo per query, and dataset __getitem__ randomly chooses photos even for validation. It does not enumerate SPICA's unique full photo gallery. Released training_step has triplet loss only, whereas paper describes sketch+photo text CE. Repo preprocessing also differs from SPICA. These are reproduction caveats, not evidence that published results are invalid.

## Concrete differences worth testing, not established causes

### 1. Preserve/use CLIP's pretrained retrieval readout

SPICA `src/spica/models/coupled_predictive.py`: `_context_tokens()` takes transformer patch tokens excludingCLS; q uses mean-patch pooling and a **new randomly initialized Linear768→512**. Native student `ln_post`/`proj` stay frozen but are not used in this q path. The copied student transformer is fine-tuned broadly; original CLIP weights remain untouched.

Thus SPICA inherits pretrained patch processing but **not the native pretrained global CLS→LN→projection readout for q**. It must learn a new sketch→CLIP-gallery mapping from downstream data. This is an intentional region-first design choice, not an implementation bug. It is a high-priority candidate explanation for weaker semantic transfer, **not proven causal by current results**.

SketchLVM primarily adapts visual prompts and LN rather than broadly fine-tuning the visual transformer; SeCo uses prompts, coupling, adapters and LN. Neither paper's statement of mostly frozenCLIP implies exact compliance with SPICA's original-weights-frozen constraint. Do not silently unfreeze original LN or call a constrained adaptation an exact reproduction.

### 2. Text supervision currently is not symmetric across actual modalities

SPICA CE_i and CE_t refer to **sketch predictor heads μ_I/μ_T**, not to actual gallery-photo embeddings. q also has textCE; live photos receive ranking gradients and frozen-photo cosine anchor, but **no explicit photo-to-text classCE**.

Both papers describe `CE_sketch + CE_photo`. SeCo additionally couples contextualized text-encoder prompt representations into visual layers. SPICA's sharing of global prompt parameters with a downstream predictor is not the same deep text-guided adaptation. Adding MP(μ_T) alone does not implement either mechanism. These are concrete missing mechanisms, but their usefulness under our constraints must be tested individually.

### 3. Query-side protection of pretrained semantics

SPICA anchors live photos to original photo features and text prototypes toT0. It has **no direct native-CLIP sketch-reference consistency loss** for q/student. SeCo explicitly describes frozen-reference InfoNCE for sketch ANDphoto, with augmented reference input and clean trainable input. Scalar cosine anchors on photo/text are not equivalent to that relation-preserving regularization. Broad student adaptation may generalize less well without such protection, but no current measurement proves catastrophic forgetting.

### 4. Training objective and budget differ

SPICA3600×B32 /46624 ≈ **2.47083 passes of original sketch observations** (two views do not double dataset epochs), clean/masked50–50. SketchLVM paper reports60epochs/B64; SeCo10epochs/B32. Optimizers and schedules differ as well. No convergence or compute equivalence has been established. More training could help or hurt, so this is not authorization to extend runs or proof that budget alone explains the gap.

Masking is SPICA's explicit research objective, not an error. But there is no matched clean-only F2_MP control isolating its clean cost; S0 and other historical controls have additional architecture/sampler differences.

## Recommendation

Stop adding MP terms solely because a relative curve improves. Keep MP-Q as current fixed3600reference. First align benchmark interpretation on the same pseudo split/evaluator without touching officialtest. Then prioritize a no-update comparison of native CLIP-style global readout versus mean-patch/new-head readout, and audit query-side semantic preservation. A subsequent minimal constrained CLIP-adaptation baseline / photoCE / consistency ablation would require separate approval and one change at a time. No automatic new GPU, baseline training, official-unseen, seed or search launched in this turn.
