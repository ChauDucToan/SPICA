# Repository Guidelines

## Agent-Specific Instructions

## CNN–Mamba Student branch — user-authorized implementation (2026-09-28)

**Latest user-authorized whitepatch mask ablation:** λ1.2 joint10 finished rawUNVERIFIED Studentfull .2662939949 teacher .5672210936. User now requested mask completely white patches all layers. Fresh10/4820/seed42B32, **retain λKD1.2, LoRA+4VPT and all other losses/schedules/photo/Student**. New `--mask-white-patches` teacher-only train+eval: RGB raw after224resizecrop, patch32 blank iff EVERY channel/pixel exactly1, no tolerance/morphology/no sorting; original positions preserved, CLS+4prompts alwaysvalid even allwhite. Additive−inf key mask B*heads,L,L to all12nativeattentionlayers; padded query outputs never read by validkeys; nativeCLS/LN/proj preserved. Dense mask no speedup promise. Legacy false exactprevious; reject oldKDcalibration with mask because changedforward. Gate **`outputs/student_whitepatch_gate_20260928/`**, plannedproduction **`outputs/student_whitepatch_teacher_10epochs_20260928/`** inspectruntime+receipts+PIDs/no duplicate/retry/resume/push/extraarm. Must gate12maskpaths, realtrimmed-tokenparity/blankcase/KDisolation/frozenstate+jointrestore before launch. Freeze source/config/docs during gates+production. Officialtestpreviouslyseen; no bestselection, sixsbirfinalscalars only; preserve historical/sibling env/artifacts.

**Latest user authorization fixed λKD=1.2:** Fresh10epochs/4820updates, retain LoRA+4VPT/Student/photoCLIP/data/seed42/B32/alllosses and finalonly sbir evaluation, change λ ONLY via new `--lambda-kd 1.2` (notcalibrated, mutuallyexclusive --kd-calibration). Prior calibratedλ .163140544 run finished rawUNVERIFIED Studentfull .259736746, teacher .567221094; teacher finaladapters/features/gallery byteexact priorλ1, so no teacher improvement claim. New gate **`outputs/student_lora_vpt_kd12_gate_20260928/`**, planned production **`outputs/student_lora_vpt_kd12_teacher_10epochs_20260928/`**, runtime/receipt/PIDs authoritative/no duplicate/retry/resume/extraarms/push. Freeze source/config/docs while gates/productionactive. User also requested SeCo/SketchLVM research and proposed whitepatch→PAD; analysis only in ignored `outputs/clip_sketch_research_20260928/`, **NO patch sorting/masking in this λ1.2 run**. Keep original positions with patchtokens; reordering alone does not focus CLSattention; real blankkeymask must affect allblocks, no zero/PADembedding shortcut or speedup claim. Any patchmask training requires separate explicit decision/gates. Preserve all prior runs/siblingfiles/env.

**Latest authorization KD calibration then fresh10:** User selected measure4trainB32 at fixed initialization, zero optimizerupdates/no officialimages, set λKD so median4(λ||gKD||/||gretrieval||)=.1 over all Student parameters; specifically λ=.1/median4(KDnorm/retrievalnorm), not median inverse. Retain LoRA+4shallowVPT teacher, same Student/seed42/data/4820updates/241warmup; teacher learns frombatch1 as before. λ fixed after scopedCPUreview, not optimum/perbatchcap/Adamratio; no lambda search or officialtest selection. New `scripts/calibrate_student_kd.py`, trainer explicit `--kd-calibration REVIEWED_RECEIPT`; absent receipt exact oldλ1. Planned diagnostic **`outputs/student_lora_vpt_kd_diagnostic_20260928/`**, gate **`outputs/student_lora_vpt_kd_gate_20260928/`**, production **`outputs/student_lora_vpt_kd_teacher_10epochs_20260928/`**. Validate raw gradients/selection/source and actual weighted trainer CUDA/checkpoint before launch; inspect runtime+receipts+PIDs/no duplicates. Freeze source/config/docs once measurement starts through production. Prior VPTλ1 finished rawUNVERIFIED Studentfull .2637826476/teacherfull .5672210936, so weaker than noVPTλ1 .2669895063/.5733337582; this experiment changes λ ONLY, no removal of VPT. Preserve all historical evidence/sibling env/source, no retry/resume/newseed/extraarm/push. Future rawfinish UNVERIFIED until reviewed.

**Latest VPT authorization:** User chose **keep λKD=1** (no calibration) to isolate adding teacher visual prompts. Fresh10epochs/4820steps, same LoRAQ/V r8alpha8/LR1e-4, StudentLR3e-4, B32seed42, photoCLIPfrozen, simultaneous teacher/Student frombatch1. New `--lora-teacher --visual-prompt`: 4 shallow learned768D tokens afterCLS/afteroriginalpositions/beforepreLN, normalstd.02, no new position embeddings, no deep reinsertion; native CLSpooling. PromptLR1e-4/wd.01 same teacher group, total teacher297984trainable (294912LoRA+3072prompt), λKD1unchanged. No-prompt mode remains exact historical teacher. Prior joint10 atHEAD136b917 finished rawUNVERIFIED: Studentfull .2669895063, teacherfull .5733337582; same frozen gallery bytes as Student-only10. Do not call VPT gain before result or LoRA gain over unmeasured originalCLIP. Planned root **`outputs/student_lora_vpt_teacher_10epochs_20260928/`**, fresh gate **`outputs/student_lora_vpt_gate_20260928/`**, inspect receipts/runtime/PIDs/no duplicates. FreshactualCUDAgate required, preserve all old gates/artifacts; finalonly Student+teacher in sbir, no selection/resume/retry/extra arms/push. Freeze source/config/docs whileactive. Official test previously seen, notblindholdout. No lambda change authorized for this run.

**Latest authorization: fresh joint teacher/Student10epochs**, user approved fixed λKD=1 (not calibrated), CLIP sketch teacher only LoRA Q/V all12 blocks r8/alpha8/dropout0/LR1e-4; Student unchanged LR3e-4, photoCLIP frozen. Teacher retrieval and Student retrieval+cosineKD with detached teacher target, both learn from first batch, no separate teacher warm-up. Shared 5%LRwarmup/cosine over4820steps, AdamWwd.01, clip1 independently Student/teacher, seed42/B32; same geometry, no augmentation/text/EMA. New `--lora-teacher` mode preserves old Student-only100 default. Fresh init (not resume100), finalonly both readouts, Student primary + teacher diagnostic, sbir six separated scalars; no selection/retry/search/extra arms. Old100 completed rawUNVERIFIED fullmAP .2607088325; old10 .2553224267; MPQ .4396614906. Previously seen official test exploratory. Planned root **`outputs/student_lora_teacher_10epochs_20260928/`**, gate `outputs/student_lora_gate_20260928/`, inspect runtime+receipt+PIDs before launch; freeze source/config/docs during gates/production. Teacher adapter checkpoint requires pinned original CLIP and normalization from archived config/code; no base pretrained upload. Gates required actual joint trainer CUDA, frozen base/teacher gradient isolation + adapter/optimizer restore. Preserve historical artifacts/sibling source/env; no automatic push. Subagents must use cx/gpt-5.6-luna/max, no silent fallback.

**New authorization after the completed 10-epoch run:** Fresh **100 epochs from initialization** on official TU220/30; keep model/seed42/B32/loss/data, replan cosine/5% warmup for 48,200 updates (482/epoch,1,540,000 observations). User explicitly chose fresh100, not resume90. 10-epoch source `a09f097`, run `outputs/student_tu_10epochs_20260928_v2/` rawFINISHED_UNVERIFIED, final clean fullmAP .2553224267 vs fixed MP-Q .4396614906; seen official result prompted this follow-up, **not blind holdout**. New `src/spica/train_student.py` `EPOCHS=100`, W&B `sbir`, final three clean metrics only, no test/selection midrun. Planned root `outputs/student_tu_100epochs_20260928/`, inspect runtime+adjacent launch receipt/PIDs, no duplicates. Preserve all previous failed/successful outputs, no resume/retries/new arm/seed, no push unless separately requested. **Freeze all nonignored source/config/docs throughout active production**. Previous 10-epoch status below is historical and does not describe the new 100-epoch run.

This worktree is `feature/cnn-mamba-student`, branched directly from `main` at `b21154e`, NOT from `experiments`. The user explicitly authorized implementation here, superseding the historical read-only restriction and historical Mo-vMF implementation order below for this branch only. Never edit/check out the sibling `spica/` worktree: its existing campaign/source freeze and unrelated dirty files must remain intact.

- Student: `src/spica/models/sketch_student.py`, 2,251,782 randomly initialized parameters. CNN, two shared-weight four-direction Mamba-2 hybrid blocks, residual gates, foreground pooling, normalized512-D output. No custom validators or tests inside the model, no custom production selective-scan kernel. This is not an exact VMamba implementation.
- User approved official `mamba-ssm` in an isolated `.venv`; pinned Git source and compatible `apache-tvm-ffi==0.1.9` in `pyproject.toml`/`uv.lock`. Existing main dependency versions, including Torch2.13/CUDA13, unchanged. Do not alter the sibling environment. No subagent used for this implementation; do not misreport it as cx/gpt-6-luna work.
- `src/spica/train_student.py`: 10 full officialTU220/30 epochs, seed42/B32/drop_last=False, frozen CLIP photo targets, multi-positive contrastive only; no text KD, sketch-teacher loss, masking or consistency. 15400sketches/482batches per epoch →4820updates/154000observations. Final clean evaluation only, no validation/checkpoint selection/retry/resume.
- User-selected primary control: official MP-Q1189 clean fullmAP `.4396614905633032`. Supplied summary SHA `afb927f7142aafe98bf57c742a034d8478bdb2586880a53f6a3c5c15b2dfba42`; all six train/test manifest/class-map SHAs checked against control. Different photo branch, loss package and budget: not a matched backbone-only ablation, not significance, not a blind holdout.
- Minimal standalone check `scripts/check_student.py` is outside the model, not a test suite. CPU full224 forward/backward/scan layout/loss/weights round-trip passed using FP64 reference SSD with stable chunk carry; initial upstream reference tail/overflow failures retained under `outputs/implementation_20260928/`. This does NOT certify production CUDA kernels. No pretrained/data encoding or optimizer updates in this check.
- **User authorized train10epochs + final test + W&B project `sbir` (2026-09-28)**. Sibling triplet campaign finished09:37UTC; no concurrent GPU work. Synthetic actual CUDA forward/backward/state roundtrip PASS; initial Triton `/sbin/ldconfig` failure on NixOS preserved, solved with `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` (no dependency/model change). Fresh real-train smoke/restore evidence at `outputs/student_gate_20260928/` required before launch. Initial production `outputs/student_tu_10epochs_20260928/` failed at CLIP loading before optimizer/W&B/updates: trainer `.resolve()` stripped `.safetensors` cache symlink suffix; old external smoke did not resolve and missed the bug. Failure preserved. **User explicitly authorized fix + fresh launch**, new planned root **`outputs/student_tu_10epochs_20260928_v2/`**. Preserve suffix with expanduser/absolute, no unsafe pickle fallback. Gate must exercise actual trainer startup. Adjacent launch receipt + runtime + live PIDs authoritative; inspect before launch, no duplicates/further retries. All source/docs frozen during production; final clean three scalars only in `sbir`, no epoch loss curves. Local runtime updates step1/every50/epoch. Raw finish stays UNVERIFIED, not automatic promotion. No extra seed/arm/search/push.

Treat this file as the durable record of repository conventions, current architectural decisions, and implementation order. Sections below describe the historical main branch unless explicitly retained above.

## Primary Research Protocol and Architecture

### Primary protocol: standard ZS-SBIR

The primary SPICA result must follow standard zero-shot sketch-based image retrieval:

- Train on seen classes and evaluate on disjoint unseen classes.
- At inference, the model receives only a query sketch, a photo gallery, and learned model parameters.
- Rank gallery photos from sketch-conditioned scores. Ground-truth query labels are evaluator-only metadata and must never enter the retrieval model.
- The primary method must not require the unseen-class text bank at inference. This keeps comparisons with SketchLVM/CLIP-AT, SpLIP, SeCo-SBIR, and visual ZS-SBIR baselines protocol-compatible.
- Learned class-agnostic prompts are allowed at inference because they are model parameters rather than query-label information.
- Match dataset split, gallery construction, and metric definitions (`mAP@all`, `mAP@200`, `P@K`) before comparing reported numbers across papers.

### Primary architecture: sketch-conditioned Mo-vMF retrieval

Keep conditional mixture-of-von-Mises-Fisher prediction on the frozen CLIP photo hypersphere as the primary research contribution. For a sketch `s`, the model predicts:

```text
{pi_k(s), mu_k(s), kappa_k(s)} for k = 1, ..., K
```

with mixture weights summing to one, unit component directions, and positive concentrations. Frozen or consistently prompted CLIP photo embeddings `z_p` are unit vectors. Rank a gallery photo with mixture log-likelihood:

```text
score(s, p) = log sum_k [
    pi_k(s) C_D(kappa_k(s)) exp(kappa_k(s) mu_k(s)^T z_p)
]
```

The standard inference path is therefore:

```text
query sketch
-> visual/context encoder
-> sketch-conditioned Mo-vMF head
-> mixture likelihood against frozen photo embeddings
-> gallery ranking
```

This path must not consume true class text or a selected unseen-class prompt. Text may be used during training as semantic supervision, but the trained primary model must be executable from the sketch alone.

### Role of text

For the primary protocol, valid text uses include:

- Seen-class CLIP text embeddings as training-only semantic teachers or auxiliary classification targets.
- Distillation from seen-class text into a sketch-conditioned latent that no longer needs text at inference.
- Class-agnostic learned textual or visual prompt tokens shared across queries.
- Training-only regularization of global semantic components or mixture parameters.

Do not make ground-truth unseen class text part of standard inference. If an experiment uses all unseen class names to classify a sketch and condition retrieval, label it explicitly as **unseen-vocabulary-assisted ZS-SBIR** and report it separately from the primary standard protocol.

### Status of text-fusion diagnostics

The existing hard, soft-posterior, and oracle text-fusion experiments are diagnostics rather than the primary architecture:

- Oracle text uses the true query category and is a label-leaking upper bound, never a valid primary result.
- Hard predicted text requires an unseen-class vocabulary and reduces category-level retrieval largely to zero-shot classification.
- Soft-posterior fusion forms one deterministic mean text vector and therefore collapses distinct semantic hypotheses; it is not a mixture model.
- On Sketchy 104/21 with frozen OpenAI CLIP ViT-B/32 QuickGELU, observed exploratory test results were: sketch-only `mAP=0.247028`, hard predicted text `mAP=0.698833`, and oracle text `mAP=0.919814`.
- The soft-posterior sweep converged to hard argmax as temperature approached zero (`mAP=0.697817` at `temperature=0.001`, `alpha=1`). Increasing temperature reduced mAP, and every tested curve preferred `alpha=1`. Static global sketch/text averaging therefore did not improve the hard predicted-text baseline.
- Per-query branch-complementarity diagnostic (Sketchy 104/21, same setup): text classification accuracy 0.727036. On correct-text queries (9229) sketch mAP=0.291116 vs text mAP=0.924725; on incorrect-text queries (3465) sketch mAP=0.129601 vs text mAP=0.097170. Sketch rescue rate on incorrect queries was 0.594805 with mean AP gain 0.032431. Oracle per-query routing (max of the two branch APs) reached mAP=0.714054, a maximum possible gain of 0.015221 over predicted text.
- Score-margin bucketing (quartiles of top-1/top-2 class-score margin): text accuracy is monotone in margin (0.387/0.620/0.909/0.992 from low to high) and text mAP beats sketch mAP in every bucket (0.4076 vs 0.1421 in the lowest-margin quartile; 0.9389 vs 0.4164 in the highest). A sketch-route rule breaks even only when it isolates failed predictions at roughly 95% precision; no margin bucket reaches that, and entropy adds no new information because it is monotone in margin through softmax.

These findings reject deterministic global soft averaging for the primary path. They do not reject conditional Mo-vMF: averaging class text embeddings produces one direction, whereas Mo-vMF retains separate photo-space components and scores them with mixture likelihood. Do not conflate uncertainty over class names with one-to-many variation among valid photos.

### Agreement Gate status

Agreement Gate between the sketch-only branch and the predicted-text branch is **rejected** based on the diagnostics above. Complementarity is real but too small and too diluted: the expected gain on failed predictions (+0.032 AP) is dominated by the cost of mis-routing successful ones (-0.634 AP), and the per-query oracle ceiling of +0.0152 mAP leaves no room for a practical router. Do not build a cross-attention or learned gate for this branch pair.

If gating is reconsidered later, it must satisfy all of:

- Its inference inputs are available from the sketch or from class-agnostic learned parameters (a gate consuming the unseen text bank belongs to the separately named vocabulary-assisted protocol).
- A per-query diagnostic shows the intended branches are complementary in the relevant operating region, not only in aggregate.
- The expected gain justifies the added model complexity relative to the conditional Mo-vMF head.

### Implementation order from the current state

1. Preserve the existing sketch-only standard evaluation as the protocol anchor.
2. Treat the text-fusion diagnostics (soft/hard/oracle, branch complementarity, margin buckets) as closed; do not extend them unless a new signal appears.
3. Agreement Gate is rejected; revisit only under the conditions listed above.
4. Stabilize the training dataset, transforms, and positive/negative sampling contracts.
5. Implement a sketch-only conditional Mo-vMF head and mixture-likelihood retrieval before adding text supervision.
6. Add seen-class text as training-only auxiliary supervision and verify that standard inference remains sketch-only.
7. Evaluate category-level ZS-SBIR, GZS-SBIR, and a fine-grained or instance-sensitive protocol so that one-to-many modeling is tested meaningfully.
8. Treat unseen-vocabulary-assisted fusion/gating as an optional secondary experiment.
9. Compare against deterministic point prediction and relevant probabilistic embedding baselines such as PCME/PCME++; do not claim first use without a fresh literature verification.

## Data Architecture

Keep the data package compact for the current project scope:

```text
src/spica/data/
├── __init__.py
├── manifest.py
├── datasets.py
└── transforms.py
```

- `manifest.py` reads image manifests and class maps into typed records. It does not open images or depend on PyTorch datasets.
- `datasets.py` contains both `RetrievalEvalDataset` and `RetrievalTrainDataset`. Do not create separate `eval_dataset.py` or `train_dataset.py` files unless this module becomes substantially large.
- `transforms.py` contains sketch/photo preprocessing for training and evaluation. Dataset classes receive transforms through their constructors rather than constructing transforms internally.
- Use ZSE-SBIR's conceptual separation between stochastic training samples and deterministic query/gallery evaluation, but do not copy its implementation structure.
- Keep the YAML files and their manifests as the source of truth for dataset protocols; do not hard-code unseen classes in Python.

Follow this implementation order:

1. Implement and verify `manifest.py`.
2. Add `RetrievalEvalDataset` to `datasets.py`.
3. Add minimal evaluation transforms to `transforms.py`.
4. Verify separate sketch-query and photo-gallery datasets for all four protocols.
5. Add `RetrievalTrainDataset` with class-aware positive and negative sampling.
6. Add the DataLoader or data-builder layer only after the dataset contracts are stable.

## Project Structure & Module Organization

Spica is a Python 3.12 project using a `src` layout. Put reusable package code in `src/spica/`; `src/spica/__init__.py` defines the package boundary. Dataset-specific utilities currently live under `data/Sketchy-std/`, including argument parsing in `args/`. Large local datasets belong in `datasets/`, which is intentionally ignored by Git. Store generated models as `.pt`, `.pth`, or `.ckpt` files and generated artifacts under `outputs/`; these are also ignored. Add automated tests under `tests/`, mirroring package paths where practical (for example, `tests/test_dataset.py`).

## Build, Test, and Development Commands

- `uv sync --dev` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run ruff check .` runs the configured Python linter.
- `uv run ruff format .` formats Python files consistently.
- `uv run --with pytest pytest` runs tests discovered under `tests/`; pytest is configured but is not yet a persistent development dependency.
- `uv build` creates source and wheel distributions using `uv_build`.

There is no application entry point yet. Run an individual module or script with `uv run python path/to/script.py` and pass dataset paths explicitly rather than relying on machine-specific defaults.

## Coding Style & Naming Conventions

Use four spaces for indentation and follow Ruff's default formatting. Prefer type hints for public functions and concise docstrings for non-obvious behavior. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes such as `Sketchy`; use `UPPER_SNAKE_CASE` for constants. Use `pathlib.Path` for filesystem operations and keep reusable logic out of command-line parsing modules.

## Testing Guidelines

Write pytest tests named `test_*.py` with functions named `test_*`. Cover dataset splitting, path discovery, transforms, and returned tuple shapes. Use temporary directories and small generated images instead of checking datasets or model checkpoints into Git. No coverage threshold is currently enforced; new behavior and bug fixes should include focused regression tests.

## Commit & Pull Request Guidelines

The repository has no commit history yet, so no established convention exists. Use short, imperative subjects such as `Add Sketchy dataset split tests`, and keep each commit focused. Pull requests should explain the change, list validation commands, link relevant issues, and call out data or configuration assumptions. Include screenshots or sample outputs only when visual behavior changes; never commit credentials, local dataset paths, generated outputs, or large model files.
