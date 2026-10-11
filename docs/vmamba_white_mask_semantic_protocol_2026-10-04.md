# VMamba white masking → frozen CLIP semantic supervision

## User-authorized scope (2026-10-04)

User separately selected frozen CLIP teacher (not GAN), full14SS2D masking+masked pooling, TWO fresh60 arms, fixed text/photo coefficients1. Implementation/gates precede production; this protocol is not a completion receipt.

| Arm | Encoder | Objective |
|---|---|---|
| Mask-only | Random official Tiny s1l8 adapted with white masking | Existing multi-positive retrieval |
| Mask+semantic | Same initialization/mask/data stream | Retrieval + text CE + photo-prototype cosine |

Each: TU-Berlin220seen/30test, seed42, B32, FP32, AdamW3e-4/wd.01/native no-decay exclusions/clip1, 60epochs/28,920updates/924,000observations, lastbatch8, warmup1,446/cosine. No pretrained VMamba, LoRA/prompts/KD/GAN/EMA, other arms/seeds, selection/resume/retry. Freeze both recipes before first launch, including second-arm losses; do not modify it after observing first-arm official results. Prior official test results already seen: not blind holdout.

## White masking contract

- Read raw RGB[0,1] after unchanged resize/crop224; a pixel is white iff all channels exactly1. No tolerance, threshold, morphology, raster erasure or input-ink conversion.
- Derive token validity by any nonwhite contributor in the actual overlapping stem convolutions. Propagate occupancy conservatively through native downsampling receptive fields, on original56/28/14/7 grids. All four scan directions receive the corresponding mask ordering; never sort/compact patches.
- For invalid SS2D locations, effective delta=0 preserves SSM state; u=0 and C=0 suppress both state readout and D*u. Compute finite softplus(delta+learned bias) before applying mask; native CUDA oflex receives delta_softplus=False/no extra bias. No custom selective-scan kernel or Python scan fallback.
- Re-zero invalid intermediate features/branch outputs/residual/MLP locations so affine biases and later convolutions cannot reuse blank-token features. Pool final normalized features using valid locations only.
- Entirely white samples use the unchanged native forward separately and are restored to original batch order, avoiding empty readout. Merely setting their mask to all-valid inside a mixed masked scan does not guarantee the native FP32 arithmetic path. Nonwhite sample token grids remain unmodified; no token compaction. This is an explicit no-object fallback, not CLS attention masking.
- The model is a **masked adaptation** of the pinned official Tiny, not unchanged upstream forward or author ImageNet/SBIR reproduction. Original vendor Python and native wheel remain unchanged. Spatial grids stay fixed; skipped state evolution deliberately changes propagation through blank intervals. Dense execution provides no speedup guarantee.

## Frozen teacher and loss

CLIP ViT-B/32 QuickGELU photo/text weights pinned to the existing SHA. Photo gallery uses the same frozen encoder/transforms, never the Student or a trained photo branch.

```text
z = normalize(Student(sketch))
t_c = normalize(CLIP_text("a photo of a {seen class name}"))
mu_c = normalize(mean_j normalize(CLIP_photo(train_photo_cj)))
L = L_existing_retrieval
  + CE(z @ T.T / 0.07, seen_class_index)
  + mean(1 - dot(z, mu_ground_truth_seen_class))
```

All three coefficients1. Both banks detached/frozen; encoder inference takes only sketches, never labels/text bank. Photo/text banks use seen TRAIN only, not the30 unseen names or any official gallery photo. Text negatives include all220seen classes. Select up-to20 distinct TRAIN manifest paths/class with isolated seed42, no replacement: squirrel4/fork13/allother20, total4377. Different paths do not independently certify no duplicate image content. Cache selected paths/counts, manifest/source/CLIP/bank hashes and embeddings before training; reuse unchanged.

This strengthens category-semantic supervision; the mean target is not an instance/part correspondence and can still encourage within-class collapse. It does not guarantee learning fine sketch details or escaping the observed global collapse.

## Gates and execution

Original gate root: `outputs/vmamba_white_mask_gate_20261004/`, retained with raw checker-dtype FAIL and mixed all-white fallback parity FAIL. User explicitly authorized dtype-only overlay, then synthetic localization/wrapper/checker correction and re-freeze (no tolerance increase). Fresh corrected gate root: `outputs/vmamba_white_mask_gate_20261004_v2/`. Original89file source snapshot and logs preserved before fixes. No real images/production were opened/launched by those failed synthetic gates. Planned production roots:

1. `outputs/vmamba_white_mask_fresh60_20261004/`
2. `outputs/vmamba_white_mask_semantic_fresh60_20261004/`

Before any encoding/gate/production: inspect live processes/GPU/adjacent receipts, require fresh paths, freeze all nonignored source/config/docs and this environment. Never touch sibling environment/source or historical evidence.

Required checks: finite native zero-delta identity-state forward/backward against FP64 reference including irregular/reverse/allblank masks; native/all-valid legacy parity; correct14blocks/mask propagation/no branch leakage; frozen banks/photo isolation, text negative and prototype loss gradients; actual framework trainer startup, short TRAIN-only updates/optimizer/scheduler/RNG checkpoint restoration and batch/data-stream binding. Add TRAIN-only learning/variance diagnostics rather than treating nonzero gradients as learning proof: after actual3updates, use a disposable fixed first TRAIN batch for256updates at the same full60 schedule. Predeclared learning criterion: final eval loss ≤99% of initial fixed-batch loss and total centered query variance >1e-6. Failure blocks both production arms; no automatic threshold relaxation or retry. This tiny seen-batch check is not generalization evidence. Measure actual FP32 fullB32 memory/timing; no BF16 forecast substitution. Planned early-stop gate statuses remain raw and are described separately from production.

Final only: all2400queries/27989gallery, fullmAP/AP200/P200, three scalars per run in W&B `sbir`; component/epoch losses and progress local. No intermediate official test, prototype nearest-class inference, hard-text routing or best checkpoint. Launch second arm only after the first trainer child exits and its adjacent execution receipt/source checks pass; the coordinating pair supervisor remains live, with no source changes between arms; do not tune based on first result. No silent tolerance increase/production retry/resume. Raw successful finish stays `FINISHED_UNVERIFIED` pending scoped review.
