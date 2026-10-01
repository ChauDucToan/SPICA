# SketchLVM public inputs — clean gap and raster masking (2026-10-01)

## Fixed run and authorization

User asked why local clean performance is below the original and requested masking; explicitly selected **the newest52.27% checkpoint**, not another training run. Original fresh60 authorization: [protocol](sketchlvm_paper_clean_protocol_2026-09-30.md). Training root `outputs/sketchlvm_tu_60epochs_public_inputs_clean_20260930/`, training source HEAD `1019e400b0b27273bcc30013814cbd11fbb469c2`,54archived files, seed42/B64/60epochs/14460updates/924000observations. Trainer/final clean exit0, finished2026-10-01T00:59:48Z. Checkpoint SHA **`5491c1aa93cc0df610c6dfe9c992af399a7d5bb80d14af61141ba3553c22a375`**, config SHA `465c2cbf544aa3f3fbca71bf7c33fc6fef71dd8e462cbc0eabed8dcdb0466eed`. [W&B clean run](https://wandb.ai/a-cctest05187-erd/sbir/runs/v197ddsx); this report did not perform an independent online-history audit.

Masking root **`outputs/sketchlvm_public_inputs_masking_20261001/`**, adjacent launch JSON/log; evaluator `outputs/sketchlvm_public_inputs_masking_gate_20261001/evaluate_masked.py`. One fixed-checkpoint CUDA evaluation, exit0,12:00:41.683→12:01:09.713Z. **No training, optimizer updates, clean/gallery re-encoding, W&B writes, retry, seed search, or model change.** Clean features/metrics and trained-photo gallery were reused from SHA-bound original clean artifacts; restored model-state hash matches original clean producer's before/after hash. Model/RNG/modes unchanged during masking. Original producer and new evaluation both remain **FINISHED_UNVERIFIED**.

## 1. What explains the clean gap?

**The main cause is not established.** New run improves on local old run; “decrease” here refers to the published result, not new minus old.

| Metric | Previous native-input run | New public-input run | Paper Table1 |
|---|---:|---:|---:|
| mAP@all |51.609379%|52.272251%|65.10%|
| P@100 |66.073750%|66.762500%|73.20%|

New−old mAP **+0.662872percentage points**; new−paper **−12.827749pp** mAP and **−6.437500pp** P100. Same2400query paths/labels and27989gallery path/label order.1461queries improve,939decline (threshold1e−6);19/30class means improve,11decline. Interpolated median query AP gain **+0.229612pp**. Laptop+5.411574pp vs rollerblades−3.808305pp illustrates non-uniform effects; one seed, no significance claim.

### Established facts, not a causal diagnosis

- Both runs have identical manifest/CLIP/prototype SHAs, optimizer/loss/budget/architecture settings and core model source. Saved CUDA-gate initialization vectors are **106/106 exact-equal**. This is gate initialization evidence, not a complete production sampler/RNG replay.
- Changing the input profile (native CLIP crop/normalization→public padding/ImageNet normalization) yielded only+0.66pp locally; it did not close the paper gap. All official sketch source dimensions are224×224, and old/new **raw/preprocessed/erased pixel and bitmap hashes match for all9600condition rows**. Thus a changed test-sketch crop is not the observed difference between these runs; normalization, learned states, and potentially photo preprocessing still differ. Gallery pixel preprocessing was not independently replayed.
- Train loss still decreases: new epoch1–5 mean1.646623→epoch56–60 mean1.094532; old late mean1.091918. No missing epochs or obvious training abort. Training-only losses cannot prove convergence, optimality, overfit or the retrieval gap's cause.
-70training sketches/class is the verified public ZSE-SBIR manifest: **220classes×70**, not70classes or local subsampling. [Manifest/FP64 audit](sketchlvm_manifest_fp64_recheck_2026-09-30.md). Exact SketchLVM-author sample pools are nevertheless not established.

### Highest-priority unresolved correspondences

1. **Complete paper code/evaluator is unavailable in the audited public basic implementation.** Public `model_LN_prompt.py` uses shared CLIP/LNs, cosine triplet only, margin.2 and separate LN/prompt LRs. The paper describes separate branches/LNs, classification, margin.3 and Adam1e−5/B64/60. Our implementation retains those paper-category additions rather than silently switching to the basic trainer. Public validation forms its gallery from sampled positive-photo outputs; that is not a verified TU-Berlin final full-gallery evaluator. It must not be assumed to be how the published table was generated.
2. Public preprocessing source implements **Sketchy**, not TU-Berlin. Paper does not fully pin category CE temperature/runtime/init/sampler. Local CE uses frozen native scale100.0 (tau.01); no evidence establishes this as the exact author setting. These are fidelity questions, **not demonstrated explanations** of12.83pp.
3. Paper states204489TU photos; local official manifests total204070 (419fewer). Could be a release/filtering/reporting difference; its location or metric impact has not been identified.
4. The published table labels TU mAP@all/P100, while the prose also discusses top200retrieval evaluation. Metric/gallery correspondence requires the actual author evaluator, not substituting local AP200/P200 for paper mAP@all/P100.
5. The historical one-row FP64 gate discrepancy2.26603e−5 is small in the observed gate. It provides **no evidence** explaining the12.83pp gap; neither does this report establish an all-official numerical-error bound. Raw FP64 FAIL is preserved.

No evidence proves patch downsampling, average-shape reliance, whole-encoder collapse, insufficient epochs, or a particular loss coefficient as the main cause. No new hyperparameter attempt is authorized by these results.

## 2. Masking results

Same fixed trained checkpoint;2400queries against cached27989adapted-photo embeddings. After **that run's exact public224RGB preprocessing**, before ImageNet normalization: ink iff meanRGB<.9; whiten all3channels at the first `ceil(f*N)` row-major ink pixels for f=.25/.50/.75; preserve every other pixel and coordinate. Not semantic-part removal, random masking, patch deletion or white-key attention masking. Inputs are square224×224, so artificial black padding contributes no new border ink in these query sources.

| Condition | mAP@all | AP@200 (minR200) | P@100 | P@200 | FullAP retention |
|---|---:|---:|---:|---:|---:|
| Clean |52.272251%|58.581075%|66.762500%|64.786875%|100%|
| Erase25% |33.555140%|34.662260%|44.110000%|42.501250%|64.193026%|
| Erase50% |15.787745%|12.481672%|20.158333%|19.415000%|30.202918%|
| Erase75% |5.660312%|1.749516%|5.359167%|5.310833%|10.828521%|

FullAP drops **18.717111/36.484506/46.611939pp**. Query W/L/T at1e−6:25%477/1923/0;50%213/2187/0;75%112/2288/0. All30class mean APs decline at all3levels;75% erasure reduces AP for95.33%queries. Mean inkN1153.01625, range245–4261, no blankN queries. Removal is counted relative to ink, not the whole white canvas.

For context only, existing previous-run masked mAP was33.835453/16.495535/5.839237%; new−old **−0.280314/−0.707789/−0.178925pp**. No old checkpoint was re-encoded. Same saved pixel/mask hashes support comparable views, but differing learned checkpoints/input normalization prevent attributing these differences to one isolated mechanism. **The small clean gain did not transfer to improved masked performance.**

Interpretation: strong sensitivity to this deterministic removal rule. Training had no such masking augmentation/objective; severe ink loss also removes potentially category-relevant structure. This test does not identify a geometry bottleneck, prove forgetting or establish that the author's published SketchLVM behaves similarly. No published matching raster25/50/75benchmark is claimed.

## 3. Review scope and provenance

- Independent read-only review found no masking metric/binding blocker:7200savedtop200rows AP200/P200/P100,120class means and paired/retention statistics;18artifact hashes; cached clean chain/checkpoint/config/54archive/6manifests;9600trace consistency. No image opening, encoder, pretrained load, GPU or full-gallery second sort in that review.
- Parent separately ran `outputs/sketchlvm_public_inputs_masking_review_20261001/{parent_verify.py,parent_verify.log,parent_receipt.json}`: **SCOPED_SAVED_RANK_BINDINGS_PASS_MEDIAN_CONVENTION_CORRECTED**. Parent tensor oracle maxAP200/P200/P100 error0; independent review's NumPy arithmetic max3.33e−16. FullAP checked as saved means/paired values only, **not independently recomputed full-sort AP**. Photo gallery bytes were not re-opened; cache identity is producer SHA evidence.
- Clean-gap worker analysis `outputs/sketchlvm_clean_gap_analysis_20261001/` used a lower-order median for even-sized vectors. Originals remain intact; corrected interpolated global/per-class medians are recorded in `outputs/sketchlvm_public_inputs_masking_review_20261001/corrected_clean_statistics.json`. Mean mAP and W/L/T were unaffected. Do not cite lower median as the interpolated0.5quantile.
- Mask summary SHA **`edcc05fa6c53deed1ad38ab380af1ea58026e8b63e8a5c91b7431cae5cb92f8e`**. Historical FP64FAIL SHA remains **`14c8ca068bac0cb549d132d65d9761bb75e5d19c7f6a30eec022f2c90b3b4b40`**. These scoped checks do not replace raw statuses or certify exact paper reproduction.
- Source/data/checkpoint/cache frozen during evaluation. No active campaign, no automatic training/retry/replay/search/cleanup/push. Post-report commit is not the training snapshot.

Primary source: [SketchLVM paper](https://arxiv.org/pdf/2303.13440v3), pp3–4(category objective),6(settings/dataset/evaluation),7(Table1),12(separate encoders); public [Sketch_LVM](https://github.com/aneeshan95/Sketch_LVM) pinned `ca33be98cb9f986813f611c997ceaa474b0bc96e`, retained source excerpts under `outputs/sketchlvm_paper_clean_readiness_20260930/`.
