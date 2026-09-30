# SketchLVM paper-category reimplementation — TU-Berlin raster erasure

**Completed:** [results](sketchlvm_raster_results_2026-09-30.md), 60epochs/14460updates + all4finalconditions exit0. User explicitly accepted retained prelaunch FP64 FAIL before launch; no tolerance change. Raw FINISHED_UNVERIFIED, scoped final saved-statistics review only. The planned operations below are historical, not permission to rerun.

## User-approved scope

Fresh seed 42, 60 epochs, batch 64, plain Adam learning rate 1e-5 (betas .9/.999, epsilon 1e-8, weight decay 0). No schedule, gradient clipping, augmentation, resume, retry, best-checkpoint selection, or intermediate official test. Official TU-Berlin 220/30: 15,400 training sketches, 241 updates/epoch (last batch 40), 14,460 updates / 924,000 sketch observations. Final test: 2,400 queries and all 27,989 gallery photos.

This implements the **user-selected paper-described category additions**, not the public basic triplet-only/shared-LayerNorm version and not a certified exact paper reproduction. Separate sketch/photo visual prompts and LayerNorm affine parameters; original CLIP visual/text parameters otherwise frozen. OpenAI ViT-B/32 QuickGELU weights SHA-256 `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`.

Three independent shallow 768-dimensional prompt tokens per modality, appended after the native `[CLS, 49 positioned patches]` sequence and before native `ln_pre`, propagated through all 12 layers; native CLS/LN/projection readout. Unit-normal initialization follows released code, not a paper-specified initialization. Train all visual LayerNorm affines in both branches: **84,480 parameters / 106 tensors**. Photo gallery uses the **trained photo prompts/LayerNorms**, not original unadapted CLIP.

Loss on normalized embeddings:

```
mean(relu(0.3 + cosine(sketch, negative_photo) - cosine(sketch, positive_photo)))
+ 0.5 * (CE(sketch, seen_text_prototypes) + CE(positive_photo, seen_text_prototypes))
```

No negative-photo CE. Frozen original CLIP seen-class text prototypes use `a photo of a {category}` (underscores replaced by spaces) and frozen pretrained `exp(logit_scale)`; this choice was explicitly approved, not inferred as an exact paper temperature. No unseen text bank or labels enter the encoder at inference. Existing sampler selects one same-class positive and one other-class negative; resized/center-cropped 224 RGB, native normalization inside each branch.

## Separate final evaluator

`src/spica/evaluate_sketchlvm_raster.py`, CLI `--run RUN --output FRESH_OUTPUT`, restores trainable-only checkpoint from archived model code plus pinned external CLIP. Safe CPU `weights_only=True` loading; exact config/source/manifest/step bindings. Encode the adapted photo gallery once. Conditions: clean and 25%, 50%, 75% raster erasure.

After bicubic resize and center crop to 224, before normalization, define ink pixels as mean RGB < .9. Erase the first `ceil(fraction * N)` ink pixels in **row-major (top-to-bottom, left-to-right) order**, setting all three channels to 1. Leave every unselected pixel and position unchanged. All-white input removes zero pixels. Count fraction differs slightly from requested fraction because of ceiling. Recorded ink mass means `sum(1 - mean_RGB)` **over threshold-qualified ink pixels**, not total-image darkness or semantic stroke importance.

Save source/pixel/mask hashes, N/k/actual count fraction/removed ink mass, CPU features, top-200 rankings, per-query/per-class full AP, AP@200 (denominator min(R,200)), and P@200. Paired deltas use the same query/gallery; median uses interpolated 0.5 quantile, win/loss/tie tolerance 1e-6, retention is masked macro full AP / clean macro full AP. Erasure fraction is not recall, a semantic part mask, or a universal robustness measure.

## Operations and verification

Implementation: `src/spica/models/sketchlvm.py`, `src/spica/train_sketchlvm.py`, and separate evaluator. Only final three clean retrieval metrics and progress_percent in W&B project `sbir`; local epoch loss components stay local. Compact atomic rolling checkpoint each epoch; final checkpoint only for evaluation. No artifacts/model upload.

Readiness drafts under `outputs/sketchlvm_readiness_20260928/` had worker integration defects (Euclidean triplet, omitted sketch CE, incompatible config/archive/runtime keys, CPU-only evaluator, repeated gallery encoding). Corrected before pretrained/CUDA gates or production. Initial full-size CPU backward attempt was killed for memory; not a model/CUDA certification. Independent static review identified physical-path alias checks and runtime/class geometry checks, corrected before gates. Source/docs freeze during active training/evaluation is deliberate; standalone replay binds execution-critical dependencies and only requires the current tree to remain unchanged during that replay, not historical docs to equal current docs.

Gate: **`outputs/sketchlvm_gate_20260930/`**, actual trainer three CUDA updates (64/64/prefix40), train-only 32-query/256-photo four-condition evaluator, compact whole-model/Adam/RNG restore and independent CPU saved-gradient/metrics/mask review. Gate-only loader uses zero workers rather than production four; prefix40 tests batch-size handling, not the actual last-epoch sampler stream. Final horizon/data validators are explicitly bypassed only for the miniature evaluator and rejection of the non-final checkpoint is checked. Do not call this an official-image or full production certification. Worker gate draft mistakenly claimed after-CLS insertion and claimed unimplemented audits; unexecuted drafts retained separately, not accepted evidence.

Planned production root: **`outputs/sketchlvm_tu_60epochs_raster_20260930/`**. Adjacent launch receipt/runtime/live PID are authoritative; inspect before launch, no duplicates. Gate receipts must pass scoped review before launch. Freeze all nonignored source/config/docs through finish; avoid manual sleep/reboot. Preserve user-dirty AGENTS.md line, existing models/runs, sibling environment/source, failures, and artifacts. No automatic retry, resume, new arm/seed/search, cleanup, or push.

Raw finish stays **FINISHED_UNVERIFIED** pending scoped independent analysis. Official test has already been observed repeatedly; exploratory comparison, not blind holdout, significance, geometry-preservation proof, or guaranteed unseen CLIP-pretraining categories. Public references: https://arxiv.org/html/2303.13440v3 and https://github.com/aneeshan95/Sketch_LVM (pinned code `ca33be98cb9f986813f611c997ceaa474b0bc96e`).
