# SketchLVM — fresh60 public inputs, final clean only

## Authorization and scope

User explicitly chose **“Rà khớp cấu hình gốc rồi train fresh60 clean-only”** after asking for clean performance near the paper. Exactly one fresh seed42 run; no guarantee of65.10% mAP, no hyperparameter search, official-test selection, intermediate test, resume, retry, raster test, extra seed/arm, or push. Preserve previous run and its FP64 FAIL unchanged. This authorization supersedes the earlier decision not to duplicate training, but does not authorize repeated attempts to reach a target metric.

Planned production: `outputs/sketchlvm_tu_60epochs_public_inputs_clean_20260930/`; adjacent launch/execution receipts, runtime and actual PIDs are authoritative. Planned gate: `outputs/sketchlvm_paper_clean_gate_20260930/`. Read-only CPU readiness/review: `outputs/sketchlvm_paper_clean_readiness_20260930/`. No launch/completion claim in this pre-gate document.

## Primary-source audit

- Paper: [CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not](https://arxiv.org/pdf/2303.13440v3), pp3–4,6–7, supplementary p12.
- Public source: [Sketch_LVM](https://github.com/aneeshan95/Sketch_LVM), pinned `ca33be98cb9f986813f611c997ceaa474b0bc96e`.
- `src/dataset_retrieval.py` uses `ImageOps.pad(size=(224,224))` before `Resize((224,224))`, ToTensor, ImageNet normalization. Local audit checkout is `/tmp/Sketch_LVM-audit/`; operations must bind retained source copies/hashes rather than rely on this temporary path.
- Public `src/model_LN_prompt.py` uses **one shared CLIP/LN branch**, separate prompts, cosine triplet only, margin.2 and separate LN/prompt learning rates. It is not the complete paper-category training implementation.
- Paper describes separate sketch/photo encoders/prompts/LNs, classification using frozen text, cosine margin.3, Adam1e-5/B64/60epochs. Retain these paper-category choices; do not silently replace them with the basic public trainer.
- The public dataset code implements **Sketchy**, not a TU-Berlin loader. Applying its preprocessing to TU-Berlin is source-supported alignment, **not verified author TU-Berlin preprocessing**. Category CE temperature/native frozen scale and initialization are not completely specified by the paper. FP32/library/RNG/evaluator correspondence remains unverified.

## Only changed production profile

Explicit CLI `--paper-clean`, method `sketchlvm_paper_category_public_inputs_v2`. Old default trainer/evaluator remains native-CLIP-input + four-condition raster behavior.

| Item | Previous run | New authorized run |
|---|---|---|
| Spatial inputs | Bicubic short-edge224 + center crop | PIL ImageOps.pad224square, default BICUBIC/default black padding; square Resize default bilinear (no-op after pad) |
| Normalization | Native CLIP mean/std | ImageNet mean(.485,.456,.406), std(.229,.224,.225) |
| Final official conditions | Clean + raster25/50/75 | **Clean only** |
| Metrics | FullAP, AP200, P200 | Same + P100 from saved top200 indices |

Both sketch and photo inputs use the same selected spatial transform/normalization. No white padding substitution, augmentation, learned normalization or Student/teacher changes. Frozen native text prototypes/logit scale remain original CLIP; ImageNet normalization applies only to the visual inputs.

Unchanged: official ZSE-SBIR manifests220seen/30unseen,15400train sketches (70/class),176081train photos,2400test queries/27989test photos. All6manifest/map SHAs already match the pinned public release; this does not prove identical SketchLVM-author training pools. Separate three shallow std1 prompts/branch appended after positioned nativeCLS+49patches before preLN; separate visualLNaffines;84480trainable parameters/106tensors; frozen OpenAI B32QuickGELU base/nativeCLS. Loss `mean(relu(.3+cos(sk,neg)-cos(sk,pos))) + .5*(sketchCE+positivePhotoCE)`, no negativeCE. Frozen220text prototypes `a photo of a {category}`, frozen native exp(logit_scale). Seed42/B64/drop_lastFalse, Adam1e-5/betas.9,.999/eps1e-8/wd0/no schedule or clipping;60epochs×241updates, finalbatch40,14460updates/924000observations.

## Operational checks and reporting

- Standalone ignored CPU input smoke, Ruff/compile/static review. No new pytest suite or model-embedded numerical validators. Production IO/config/horizon/finiteness safety stays intact.
- Train-only actual3CUDAupdates64/64/prefix40, original loss/backward/Adam/checkpoint; whole model,106optimizer states/212moments and RNG restore; nonfinal checkpoint rejection;32train queries/256train photos clean-only evaluator. Gate-only zero workers/prefix/horizon/count bypasses disclosed; no official image before production final.
- Independent CPU saved loss/input-gradient and Adam replay, source/manifest/CLIP bindings,32train image public-input pixel/hash replay, savedrankAP200/P200/P100 and means. **No independent FP64 full-gallery sort or encoder certification in this new gate.** Original gate1/128FP64 FAIL and producerUNVERIFIED remain unchanged; scoped checks must not relabel them.
- Freeze all nonignored source/config/docs once gates start through training/evaluation. Fresh paths/PIDs/GPU/disk preflight; one detached idle-inhibited process. No sleep/reboot/cleanup/push.
- Final trained photo gallery encoded once; final clean query embeddings; compact adaptation+optimizer/RNG checkpoint, config/source/data/CLIP bindings and saved query ranks/metrics. W&B `sbir`, four final clean metrics + progress100 only; epoch losses local. No evaluator-created W&B run.
- Raw finish **FINISHED_UNVERIFIED**; report mAP@all and P@100 against paper65.10%/73.20% without claiming exact reproduction, significance/SOTA or guaranteed unseen CLIP-pretraining classes. Official test previously observed; metric gap is not a proven cause diagnosis.
