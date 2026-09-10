# Local-information diagnostic: MP-Q and TQMP @3600

## Scope / identities

User requested a clearer explanation of suspected CLIP-readout/text-supervision/reference-consistency/budget gaps and an actual local-information check. User explicitly selected **both fixed MP-Q and TQMP** for a no-update diagnostic on100pseudo-validation sketches,5/class. No training, new checkpoint selection, official-unseen evaluation, new dependency, or inference-default change.

Root: `outputs/local_information_diagnostic_20260910T034000Z/`.

- MP-Q: F2_MP training checkpoint3600 SHA `58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e`, q-only readout.
- TQMP: checkpoint3600 SHA `e914f6d38bed502a0357c221dc79131cb9b2f17e05860ec371a77b03f1965dfc`, q-only readout.
- Working HEAD before diagnostic `e3caf57e7d4100530dd9713b8de4910112407e6a`; training source identities remain those in the previous reports, not this documentation commit.
- Production model source matches both archived model implementations. Safe archived loader validates checkpoint/step/arm/model state and originalCLIP hash with `weights_only=True` and explicit NumPy safe globals. Unfinished worktree V1 verifier not executed.
- `measurement/protocol_lock.json`, `completion/source_lock.json` bind source/input evidence. They are component/source-file locks, not a claim to archive the entire dependency environment.

## Key findings

1. **All100 actual source files are already224×224.** Their path includes `256x256`, but that is not their decoded size. Current production preprocessing is bit-exact with prepared inputs and does not spatially resize/crop these images. No loss caused by the current resize/crop stage was observed in this sample. Earlier processing before these dataset files existed is not audited here.
2. Sketch occupancy is sparse: mean **66.5102% of49patches have zero thresholded ink**; median occupiedpatchcount15.5, range3–37. Seven queries have both ink bounding-box dimensions≤64pixels. Some objects are already tiny inside the224canvas.
3. Local perturbation is **not simply erased at patch embedding or throughout the trunk**. At patch-conv, the16×16deletion changes only its one32×32patch; through the transformer the response spreads. At block12,≈81% of patch-token squared-change energy is outside the directly altered input patch. This is contextual propagation, not proof of semantic recognition or preservation of every detail.
4. Both q readouts respond to small-region changes. Mean local q cosine distance is.028529/.029092. There are large retrieval changes in both directions; **low mean AP degradation does not imply local-insensitivity or robust understanding**.
5. TQMP has no demonstrated local-understanding improvement over MP-Q. Responses are similar; no part annotations, significance test, or isolated-loss attribution.

## Input / perturbation protocol

`prepare_inputs.py`, finalaccepted `inputs_final/`:

- Select5queries/class by SHA256(`local_information_v1|`+dataset-relative-path), independent of model outputs.100queries/20pseudo-validation classes, no training/official-unseen query selected.
- Thresholded ink: RGBmean<.9 after production preprocessing. Class labels are used only to balance samples and evaluate relevance, never passed to query forward or perturbation selection.
- Up to8SHA-ranked16×16occupiedcells perquery, in distinct32×32patches. No model-based region selection. Queries with fewer eligiblepatches retain3–7regions, not replaced.
- **772local deletions**, each white16×16cell; retain at least1ink pixel.
- **772matched-count scattered deletions**: erase exactly the same number of thresholdedink pixels elsewhere in the same query, selected with deterministic local RNG.
-100clean +100white-background controls. All actualbackground cells are purewhite, so they are no-op tests, not ink-matched deletions.
- Total **1744rows/arm**, identical input tensors/masks/IDs across arms.13,999galleryphotoIDs/labels are identical across arms; **each arm uses its own checkpoint's prompted gallery embeddings**, not a shared embedding bank or cross-checkpoint swap.

Each16pxsquare covers.5102% of image area, but ink removal varies substantially: median2.1523%,5th/95thpercentiles.2171%/8.1737%, maximum30.0781%.25/772erase>10%ofink. Therefore “small spatial window” is not uniformly a mild semantic deletion, especially for tiny sketches.

Scattered controls match thresholdedink COUNT only, not erased darkness/antialiaspixels, spatial support, connectedness, or semantic content. Local-vs-scattered is descriptive evidence about these perturbation packages, not a perfectly isolated causal location effect.

## Pixel/feature/retrieval separation

Architecture inspected: image224→patch32→49patchtokens(+CLS inside attention),12transformerblocks, q=`normalize(Linear(mean(final_patch_tokens)))`. Predictor is bypassed duringqprobe. Existing forward graph used with temporary observation hooks; no alternative attention, pooling, crop-inference, or teacher objective was evaluated.

### Layer response to local deletion

Equal-query means: average each query's available localvariants, then average100queries.

|Quantity|MP-Q|TQMP|
|---|---:|---:|
|Patch-conv response outside directly alteredpatch|0|0|
|Block4 response energy outside alteredpatch|.211343|.209641|
|Block8 response energy outside alteredpatch|.582711|.583037|
|Block12 response energy outside alteredpatch|.813982|.812375|
|Block12 spatial change-energy participation ratio|19.5986|19.2970|
|Block12 relative Frobenius response|.210067|.211938|
|Block12 mean-delta / RMS-token-delta|.542334|.541891|
|q cosine distance after localdeletion|.028529|.029092|

For patch-token changesΔh_j, mean-delta/RMS-token-delta is

`||mean_j Δh_j|| / sqrt(mean_j ||Δh_j||²)`.

At patch embedding, a single affected token among49 gives1/7=.142857 mechanically; at the final layer the response has spread and this ratio is≈.542. **It is not “54% information retained”, nor proof meanpool is optimal.** Cross-layer response norms also are not directly comparable information units.

Initially blank spatial positions can hold contextual object information at later layers. The66.5%blankinputpatches observation does not justify calling66.5%finaltokens useless or dropping them without testing.

### Retrieval on the100-query sample only

Full AP uses category relevance over the full13,999photo gallery. Groupmeans first averagevariants perquery, then equally average100queries. These are **not the full10,963-query validation mAP**, and the sample was not reweighted to the historical query distribution.

|Group|MP-Q fullAP samplemacro|TQMP fullAP samplemacro|
|---|---:|---:|
|Clean|.437275044|.429714112|
|Localdeletion|.434538571|.427326479|
|Matched-count scattereddeletion|.440929446|.432602177|
|Local−clean|−.002736473|−.002387633|
|Scattered−clean|+.003654402|+.002888064|

- LocalmeanAPdeclines for57/100queries MP-Q,59/100TQMP (threshold1e−6).
- Perlocalvariant |ΔAP|>.05 in129/772MP-Q,122/772TQMP.
- Median |ΔAP|=.009026/.010337; large positive and negative effects partially cancel in the signedmean.
- White-background controls produce exactsame savedq/finalpatchtensors. GPUAP reduction differs at most1.1920929e−7; independentFP64AP is exactlysame. Do not call the original serializedGPUAP bit-exact.

### Illustrations, explicitly posthoc

`parent/local_deletion_examples.png` selects minimum and maximum MP-Q localΔAP among772already-measuredregions **only to illustrate failure/sensitivity**, not to select queries/checkpoints for evaluation:

- q61, pistol: local16pxregion removes2.2001%ink; AP **.7872→.2367** (Δ−.55052). Matched-count scattered deletionΔ−.03334.
- q54, hedgehog: localregion removes5.8758%ink; AP **.3082→.8106** (Δ+.50243). ScatteredΔ−.03941.

These examples show that local structure can substantially affect retrieval. They do not establish that the affected region corresponds to a correctly understood semanticpart. An AP increase after deleting information may reflect removal of misleading cues, spurious sensitivity, or ranking effects; current evidence does not distinguish them.

Original/preprocessed sample grid: `inputs_final/contact_sheets/page_01.png`. Source and transformed images are displayed at reduced visual sizes in the sheet; statistical assertions use native224arrays, not screenshots.

## Verification and preserved failures

### CPU preparation

Worker prototype `inputs/` and `prepare_inputs_before_parent_fix.py` preserved. Parent corrected odd-size center-crop coordinate rounding, required zero-ink backgroundfallback, added actualsourceimageSHA, fixed displaygrid scaling, and added nonsquare/maskoutside selfchecks. Fresh `inputs_final/`; actualclean/masks/indices/kinds **exactly unchanged** because realfiles224square and everybackgroundpurewhite. No GPU used before correction.

`independent_gate/verify_inputs_and_measure_review.py`, `receipt.json`: independent selection/imageSHA/production-equivalent preprocessing/maskgeometry/inkcount/source224/legacyarray equality and preGPU architecture/source checks PASS. An earlier read-only reviewer could not persist a receipt; this durable worker audit supersedes that reporting limitation. Hashing checkpoints during preparation was provenance-only, not deserialization or GPU use.

### First MP measurement / historical batch-size issue

The first MP measurement saved all1744rawfeatures/retrieval, then failed a historicalcleanAP threshold2e−5: maxdelta **9.876489639282227e−5**. Failure retained at `measurement/failure.json` and log; originalsource preserved under `measurement/source_files/`.

Investigation:
- Re-encoded MPgallery equals priorgalleryfeatures exactly.
- Same currentcleanimages with a256-sized batch recover prior q **maxdelta0**.
- Small perquery batches used for perturbations differ from historical q by up to9.954907e−5.
- Thus batch shape is sufficient to reproduce/remove the observed q discrepancy in this check; no claim identifying the responsible kernel.

No historicaltolerance was increased and no historicalexactreplayPASS was claimed. Historicalcomparison is now descriptive, separate from within-current-batch clean/perturbed comparisons.

`complete_measurement.py` reconstructed MP and verified **all1744savedq/finalpatchrows exact**, with before/afterstatehash equality and gradientsNone. This is a **new verification forward**, not a claim the originalabortedprocess saved its afterstateguard. MP verification stateSHA `6419b0b94a9d20c17a6b0e7e781d949672899d929bfcefc2dc321e437672e4bd`.

TQMP then measured for the firsttime; stateguard was moved before downstreammetricreporting. Its historicalcleanAPmaxdelta2.922490239e−5 remains descriptive/nonexact. Both use correctqroute and source/checkpointbindings; no updates/backward.

### Independent raw/metric audit

`independent_results/verify.py`, `receipt.json`, `report.md` PASS:
- Reconstructed all1744g/z/q from savedfinalpatches and savedheadweights perarm.
- Independently recomputed all14stage formulas on80rawrows/arm (20queries×clean/local01/scattered01/background). Intermediate activations for otherrows were not all serialized, so no allrow/allstage independentclaim.
- Independently sorted fullgallery for all1744rows/arm using FP64reconstructedq; maxfullAPdelta **1.69995e−5 MP /1.46565e−5 TQMP**, under predeclared2e−5 threshold. Not bit-exact:29/42top200rankrows and5/5relevancerows differ. No thresholdadjustment or secondmodelencoder.
- Querymacro/local-scatter summaries independently reconstructed; no sameCPU-script claim of validating pixelpipeline (that is the separate gate).

Parent `build_report.py`, `parent/receipt.json` PASS:
- Source/input/rawfilehash bindings; both independentreceipts.
- ActualimageSHA/dimensions/inkbbox and patchoccupancy.
- All1744final-stage response formulas replay exactly perarm; allsavedtop200 P200/threeAP200denominators reconstructed, maxerror1.344e−7.
- Own FP64perqueryaggregation verifies producerfloat32summary within2e−7.
- Background rawtensors exact; GPUAProunding explicitly retained.

Rawtrainingruntime/statuses and oldW&Bruns remain unchanged. Diagnosticfeatures/reports are local, not retrospective additions to historical μ_I metrics.

## What this changes about the A–D hypotheses

**A — Pretrained readout:** Still a valid untested architecture difference, but this diagnostic does **not** support the blanket explanation “patch→mean loses all local signals”. The actual q demonstrably responds to local changes. It does not compare controlled nativeCLS vsmean training, and plugging oldprojection into an alreadymean-trainedstudent would only be a readoutdiagnostic, not such a controlledablation. CLS still participates inside currentattention; it is discarded only from the finalqpool.

**B — Actualphoto textCE:** Still absent; CE_i/CE_t refer to sketchheads. This is an alignment difference, not establishedpixel/localinformationloss. No photoCE intervention was tested here.

**C — Sketch-reference consistency:** Still absent as a direct teacheranchor; frozenphoto/textanchors do not constrain every aspect of the trainablesketchmapping. Neither localresponse magnitude nor student–teacher displacement alone proves forgetting. No newteacherconsistency was tested here.

**D — Budget/objective:**3600×32/46624≈2.47083originalsketchobservationpasses, not2.47guaranteedvisits perindividualsketch and not4.94epochs because oftwoviews. This audit does not identify undertraining, the isolatedcleanprice ofmasking, or benefitfromlongertraining.

**Updated priority:** investigate input objectscale/occupancy and stable semantic use of localstructure before adding moreMPheads. Whole-sketch ink-bbox normalization or a context-preserving global/local view comparison is a possible nextcontrolled diagnostic, **not executed or automaticallyauthorized**. Need annotated/validatedsemanticparts or a carefullyspecifiedtask to establish localunderstanding sufficiency. Do not train/changebackbone/pooling/loss merely from these correlations.
