# CNN–Mamba Sketch Student

Isolated branch `feature/cnn-mamba-student`, based on **main `b21154e`**. No code imported from the sibling experiments branch. Its datasets and the fixed MP-Q report are read-only inputs.

## Model

`src/spica/models/sketch_student.py` contains the complete Student:

```text
RGB sketch [0,1], 224×224
  → grayscale ink (1 − mean RGB)
  → CNN 224×224×32 →112×112×64 →56×56×96 →28×28×192
  → hybrid block ×2: F + local_CNN(F) + sigmoid(gate)·four_way_Mamba2(F)
  → foreground-weighted pooling (10% background floor)
  → Linear 192→512 → L2 normalization
```

- **2,251,782 parameters**, random initialization, no CLIP weights in Student.
- Four row/column forward/reverse scans share the same Mamba-2 weights within each block. Coordinate order is restored before merging. This is **not** the original VMamba SS2D architecture.
- Dense grid throughout; ink weighting only at pooling. No token deletion or claimed sparse-compute savings.
- Official Mamba-2 Triton SSD scan, native PyTorch causal convolution (`use_mem_eff_path=False`). No handwritten production Mamba, no fallback silently replacing Mamba with CNN.
- No tests/validation methods embedded in the model. IO checks remain in the training entrypoint and existing data loaders.

## Isolated environment

Run in this worktree, not `../spica`. The sibling environment must not be changed.

```bash
uv sync --frozen --extra cuda --no-dev
uv pip install --python .venv/bin/python ninja==1.13.2 wheel
MAMBA_FORCE_BUILD=TRUE uv sync --frozen --extra cuda --extra student \
  --no-dev --inexact --no-build-isolation-package mamba-ssm
```

Official source is pinned to `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`. This revision permits a core-package build without `nvcc`/`selective_scan_cuda`; Mamba-2 uses Triton at runtime. Mamba-3 dependencies are pulled by upstream even though this model does not use Mamba-3. `apache-tvm-ffi==0.1.9` is pinned because the upstream-allowed0.1.12 failed to import with tilelang0.1.8 here. Torch2.13.0+cu130, torchvision0.28.0+cu130 and every pre-existing main lockfile version are unchanged.

## Small standalone check

No test suite. The runnable check is outside the model:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/check_student.py
# Explicitly run only on an available GPU, not alongside an active campaign:
PYTHONPATH=src .venv/bin/python scripts/check_student.py --device cuda
```

CPU checks a non-square scan-layout round-trip, full224×224 forward/backward including a blank image, unit-norm output, multi-positive loss oracle, finite gradients and state-dict restoration. CPU uses upstream reference SSD/RMSNorm, with FP64 scan arithmetic and stable recursive inter-chunk state propagation. Those checker-only choices avoid upstream reference tail/overflow issues; they do not change the production model or certify its CUDA numerics. Initial reference failures and the scoped PASS log are retained in `outputs/implementation_20260928/`.

**Historical 10-epoch run, completed 2026-09-28 in W&B project `sbir`.** Actual synthetic CUDA forward/backward/round-trip check passed. On NixOS set `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`; the initial missing `/sbin/ldconfig` failure is retained. Real-train smoke/restore receipt must pass before launch. The initial `outputs/student_tu_10epochs_20260928/` launch failed before any training update/W&B initialization: resolving a Hugging Face symlink stripped the `.safetensors` suffix and OpenCLIP selected the wrong reader. That failure is retained. The user authorized a fix and fresh launch; the trainer now preserves the filename (no unsafe pickle fallback). The corrected actual-trainer startup gate is separate from the earlier external smoke. **`outputs/student_tu_10epochs_20260928_v2/runtime.json`** and its adjacent launch receipt record the historical run; the raw result is `FINISHED_UNVERIFIED`.

## Completed historical 10-epoch experiment (code at `a09f097`)

At commit `a09f097`, `src/spica/train_student.py` trained official TU-Berlin220/30:

- 15,400 train sketches, B32, **drop_last=False**: 482 batches/epoch; **4,820 updates and154,000 sketch observations** after10 complete epochs.
- Seed42; AdamW LR3e-4, weight decay.01; 5% warmup then cosine. SSM no-weight-decay markers respected. FP32 parameters; global gradient clipping at1.
- Frozen OpenAI CLIP ViT-B/32 QuickGELU photo encoder. No photo prompts, text targets, sketch-teacher distillation, image augmentation/masking, or consistency term in this V1.
- One sampled positive and negative photo/sketch from the existing main loader. Multi-positive contrastive temperature.07, averaged over **all same-class entries in the batch photo bank**. Repeated sampled photos are retained, not deduplicated.
- No intermediate test/validation or checkpoint selection. One rolling checkpoint after each epoch; final clean official evaluation after epoch10. No resume/automatic retry.
- Minimal optional W&B project **`sbir`**: config plus final three clean retrieval scalars; no system/code/model/artifact capture. Default disabled. Training loss and epoch progress are local; no retrieval curves before the final test.

The user selected **current official MP-Q** as primary reference, not a CNN-only control:

|Metric|MP-Q fixed1189|
|---|---:|
|Clean full mAP (primary)|0.4396614905633032|
|Clean mAP@200, denominator min(R,200)|0.4218314166392649|
|Clean P@200|0.5028541551022014|

The supplied control summary must have SHA256 `afb927f7142aafe98bf57c742a034d8478bdb2586880a53f6a3c5c15b2dfba42`; its six manifest/class-map hashes bind dataset identity. The completed 10-epoch run at `outputs/student_tu_10epochs_20260928_v2/` used 4,820 updates and reported clean mAP@all 0.2553224267 versus MP-Q 0.4396614906, raw `FINISHED_UNVERIFIED`. It took ~10m34s of training plus ~22s final evaluation. This compares different systems and budgets, not an isolated Mamba/backbone gain.

## Fresh 100-epoch experiment (user-authorized after seeing epoch-10 result)

The current `src/spica/train_student.py` uses `EPOCHS=100`: 48,200 updates, 1,540,000 sketch observations, same architecture/seed/B32/loss/data as the 10-epoch run. Schedule has 5% warmup across the **new full horizon**, not a continuation of the 10-epoch LR schedule. Fresh initialization; no resume. One rolling checkpoint per epoch; **only** epoch-100 official clean test and W&B project `sbir` metrics, no checkpoint selection. The official test was previously observed, so this is an exploratory follow-up, not a blind holdout or proof of significance. Planned output root `outputs/student_tu_100epochs_20260928/`; its runtime and adjacent launch receipt are authoritative. Freeze all source/config/docs during production and preserve the 10-epoch outputs unchanged.

## Joint CLIP–LoRA teacher + Student, fresh 10 epochs

User-approved follow-up after Student100 fullmAP 0.2607088325 (raw UNVERIFIED).
Use **`--lora-teacher`** to select a fresh10epoch/4820update run; default Student-only100 remains unchanged. `src/spica/models/sketch_teacher.py` copies the frozen CLIP visual backbone and trains only Q/V LoRA across12blocks, rank8/alpha8/dropout0,294912 adapter parameters, LR1e-4. Student remains random-init2251782 parameters/LR3e-4; photoCLIP frozen. Both receive multi-positive photo retrieval; Student also receives cosine KD from detached teacher, **λKD=1 fixed, not calibrated**. Teacher targets are from the same pre-update state; teacher backward precedes Student forward to release activations, one optimizer step updates both. Shared5%warmup/cosine, wd.01, gradclip1 separately, same seed42/B32/data geometry. No teacher-only warmup, EMA, text or augmentation. Photo and teacher see CLIP-normalized pixels; Student sees rawRGB. LoRA initializes as exact zero perturbation and its initialization preserves the Student/data RNG stream.

Only finalepoch10 tests: primary Student under `test/cleaned/*`, teacher diagnostic under `teacher/cleaned/*`, W&B `sbir`. No checkpoint selection; official test already observed. One rolling checkpoint contains Student, LoRA adapters, joint optimizer/scheduler/RNG, config/baseCLIP SHA; original CLIP weights stay external. No resume/exact DataLoader-stream guarantee. Local losses separate teacher retrieval, Student retrieval, KD. Planned root **`outputs/student_lora_teacher_10epochs_20260928/`**, gates `outputs/student_lora_gate_20260928/`. Preserve previous runs; freeze source/config/docs while active. Run standalone CPU check `PYTHONPATH=src .venv/bin/python scripts/check_sketch_teacher.py` before real CUDA gate.

## Add shallow visual prompts to the teacher (fresh10, λKD unchanged)

User explicitly chose **keep λKD=1** to isolate adding VPT after observing joint10 StudentfullmAP0.2669895063/teacher0.5733337582. Add `--visual-prompt` to `--lora-teacher` for a fresh10epoch/4820update run. Four learned768D tokens are inserted afterCLS **after original positional embeddings and before the native pre-LN**; original patch positions remain unchanged, prompt tokens have no separate positional embeddings. Tokens persist through all12blocks, no deep reinsertion; keep nativeCLSpooling/projection. Prompt initialization normalstd.02, LR1e-4/wd.01 in the same teacher group and shared gradient clip1;297984 teacher trainable parameters (294912LoRA+3072prompt). LoRA initialization and Student/data RNG unchanged by prompt initialization. Extra tokens do change initial attention even if zero; no zero-prompt parity claim. No-prompt modes preserve historical numerical behavior.

No change to Student/loss/λKD/seed/photoencoder/data/budget; teacher and Student still train frombatch1. Evaluate both only at epoch10, W&B `sbir` with separate Student/teacher metrics, no selection. New planned root **`outputs/student_lora_vpt_teacher_10epochs_20260928/`**, gate `outputs/student_lora_vpt_gate_20260928/`. The rolling `teacher_adapter` checkpoint now includes the prompt tensor; restore with the same explicit VPT flag and pinned CLIP base. No resume/retry, preserve earlier artifacts. CPU check: `PYTHONPATH=src .venv/bin/python scripts/check_visual_prompt.py` (includes local archived previous-teacher parity); actual CUDA gate required before production. This is a previously-seen-test follow-up, not a blind evaluation.

## Calibrate KD weight, retaining LoRA + VPT (fresh10)

User authorized changing **only λKD** after VPTλ1 finished (StudentfullmAP .2637826476, teacher .5672210936). `scripts/calibrate_student_kd.py` measures four trainB32 batches at the same fixed initialization, with zero optimizer updates. Compute raw L2 gradients of retrieval and cosineKD on **all Student parameters** and select `lambda = 0.1 / median4(norm(gKD)/norm(gRetrieval))`. Thus median weighted norm ratio at initialization is10%; this is not an optimum, perbatch cap, or Adam-update ratio. Teacher will change during training; the measured initialization ratio need not persist. No official images/test results enter selection; no search.

Raw gradient/input-feature artifacts, source/manifests, and unchanged-state evidence stay `MEASURED_PENDING_REVIEW` under `outputs/student_lora_vpt_kd_diagnostic_20260928/`. An independent scopedCPUreview produces a separate receipt. Pass that reviewed receipt to `--kd-calibration PATH` together with `--lora-teacher --visual-prompt`; trainer rejects incompatible source/data/seed/CLIP and invalid coefficient, binds receiptSHA in config. No receipt keeps historicalλ1. Gates `outputs/student_lora_vpt_kd_gate_20260928/`, fresh production root **`outputs/student_lora_vpt_kd_teacher_10epochs_20260928/`**. Same10epochs/4820steps/241LRwarmup, teacher learns frombatch1, finalonly bothreadouts/sbir, no selection/resume/retry. Local `kd_loss` remains **unweighted**, `loss=student_retrieval_loss+lambda*kd_loss`; teacher loss is separate. Freeze source/docs/config through calibration and production, preserve prior runs. Standalone selection check: `PYTHONPATH=src .venv/bin/python scripts/calibrate_student_kd.py --self-check`.

## User-fixed KD=1.2 follow-up (fresh10, same LoRA+VPT)

After measuredλ≈.16314 underperformedλ1 (Studentfull .259736746 vs .263782648; teacher identical .567221094), user requested **`--lambda-kd 1.2`**, not another calibration. This flag requires teacher mode, is mutually exclusive with `--kd-calibration`, and rejects nonfinite/nonpositive values before device/data access. No flag/receipt still usesλ1. New fresh10epochs/4820updates with **`--lora-teacher --visual-prompt --lambda-kd 1.2`**, planned root **`outputs/student_lora_vpt_kd12_teacher_10epochs_20260928/`**, gate `outputs/student_lora_vpt_kd12_gate_20260928/`. Only KD coefficient changes; no masks/patchsorting/model/photoencoder/seed/schedule changes. Student final primary and teacher diagnostic, sbir, no selection/retry/resume. Previously seen official test, notblindholdout. Source/config/docs freeze whileactive.

Parallel paper research and a synthetic CPU attention-permutation/PAD demonstration live under ignored `outputs/clip_sketch_research_20260928/`. Neither SeCo nor SketchLVM describes white-patch packing. Sorting positionedtokens alone preserves CLS output for native bidirectional attention; masking is a distinct intervention. **No patch modification is included in this training run.**

## Exactly-white sketch patch key masking (fresh10, λKD1.2)

User-authorized follow-up to unmaskedλ1.2 (Studentfull .2662939949, teacher .5672210936, rawUNVERIFIED). Add **`--mask-white-patches`** to `--lora-teacher --visual-prompt --lambda-kd 1.2`. Mask only the sketch teacher, during train and final evaluation; Student/photo remain unchanged. After normal224resize/crop but BEFORE normalization, a32×32patch is blank iff every RGB channel/pixel equals1 exactly. No threshold tuning, faint/antialiased nonwhite pixels remain valid. Keep originalpositions and nativeCLS pooling; **no sorting**. CLS+4prompts always valid, including allwhite input. Use additive−inf mask on blankkeys in ALL12attentionlayers; blankquery outputs cannot influence validtokens in laterlayers because their keys remain blocked. Dense sequences remain54tokens; no FLOP/speedup promise. Defaultfalse is historical behavior; oldcalibration receipt is rejected with maskmode.

Only patchmask changes versus λ1.2 baseline: same initialweights/seed42/B32/loss/schedule/4820updates. Run finalbothreadouts/sbir/no selection/resume/retry. LocalCPUcheck `PYTHONPATH=src .venv/bin/python scripts/check_white_patch_mask.py` uses tinyOpenCLIP plus localarchivedpreviousmode; not realCLIP certification. FreshCUDAgate **`outputs/student_whitepatch_gate_20260928/`**, production **`outputs/student_whitepatch_teacher_10epochs_20260928/`**; inspect receipts/runtime/PIDs before launch, freeze source/docs/config whileactive. Prior test alreadyobserved, notblindholdout. Mask rule and scope recorded in checkpointconfig; restoring requires same maskflag.

## Teacher deep visual prompt, all 12 layers (fresh10)

User authorized a fresh10epoch single-arm ablation after whitepatchmaskλ1.2 StudentfullmAP .2688652228 / teacher .5693351723. Use `--lora-teacher --visual-prompt --deep-visual-prompt --mask-white-patches --lambda-kd 1.2`. Replace the4 persistent shallow teacher tokens with **4 independent learned tokens per layer ×12**, trainable36864 prompt params+294912LoRA=331776 teacher params. For each layer, insert afterCLS, before native49patchtokens; apply originalpreLN to learned prompt, preserve original patch positions, drop the four outputprompttokens before the next layer. Same exactwhite-keymask (all12), same nativeCLS readout; no image patch sorting, no text-guided coupling/SeCo reproduction. Student, frozenphoto, λKD1.2, all losses/LRs/seed42/B32/4820updates/official finalonly Student+teacher W&Bsbir unchanged. Source stays clean old modes; defaultno-deep retains historical behavior. CPU standalone check `PYTHONPATH=src .venv/bin/python scripts/check_deep_visual_prompt.py`; real CUDA gate `outputs/student_deepvpt_gate_20260928/` before planned production `outputs/student_deepvpt_teacher_10epochs_20260928/`. No selection, resume, automatic retry, official-test tuning claim or push; final raw UNVERIFIED. Freeze source/docs/config during gates/production.

## Frozen seen-training semantic diagnostic (not a test suite)

Completed report: [Student vs teacher](docs/student_seen_semantics_2026-09-28.md). Centroid top1 Student52.69%/teacher79.68%; sampled-galleryfullAP27.87%/58.14%. Use correctedoverlayCSV linked in report (raw perqueryPearson/overlap columns swapped; originalfailure retained), no encoder rerun. Seen-training only, not independentholdout/geometryqualityproof.

`src/spica/diagnose_student_semantics.py` restores the archived deep-prompt checkpoint without updating encoders. User-approved scope: all15400seen-train sketches, localRandom42 split56centroid/14query per220class, up to8randomtrainphotos/class=1756gallery (squirrel has only4; user approved no duplication). Compare normalized nearestsketchcentroid top1 (same-modality class separation) with sketch→photo fullAP/AP200/P200 on the SAME3080queries. Gallery identical for Student/teacher; matchedcosine, scorePearson, top200overlap and stable ordinal rank correlation (ties broken bygalleryorder, not tie-corrected Spearman), perquery/classCSV and feature caches. **Every sketch was already seen during encoder training: this is not a validation holdout, puregeometrytest, or independent generalization claim.** No classifier optimizer/encoderupdates/backward/newofficialtest/W&B. Sampled gallery not comparable to official27989gallery; **P@200 max4% for219classes, 2% for squirrel**; AP denominators use actual positive counts. With14queries/class, query-macro and class-macro AP coincide. AP is more informative than P200 here.

```bash
TRITON_LIBCUDA_PATH=/run/opengl-driver/lib PYTHONPATH=src HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m spica.diagnose_student_semantics \
  --run outputs/student_deepvpt_teacher_10epochs_20260928 \
  --output outputs/student_semantics_diagnostic_20260928 --device cuda
```

Output must be fresh. SHA-binds checkpoint/config, originalCLIP and archivedencoderfiles, trainmanifest/classmap/split/sourceimages, diagnostic sourcearchive. Safeweights_only restore and exacttrainableteacherkeys; actualdecodeallowlist; fixedprivateDataLoaderRNG, no state/grad changes. Fixedseed does not certify CUDAkernel/near-tie bitreproducibility; savedfeatureaudit is separate from encoderreplay. InspectGPU/PIDs before encode, freeze source/docs/config duringexecution. **jaxtyping0.3.11** installed in isolatedStudent .venv and lock, with Tensor shape hints on models/loss/diagnostic (`Float[Tensor,"batch 512"]`, etc.). No runtimechecker or newtestframework, no numericalmodelchange. Dependencies of existinglock remain unchanged.

## Explicit launch command

Make the existing TU data available via the ignored path `datasets/TUBerlin` (a local symlink is sufficient). The local worktree already has this link; no images/manifests were edited.

```bash
TRITON_LIBCUDA_PATH=/run/opengl-driver/lib \
PYTHONPATH=src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m spica.train_student \
  --clip-weights /path/to/open_clip_model.safetensors \
  --mpq-summary ../spica/outputs/percent20_remaining_execution_20260912/runs/tuberlin_220_30/test/step_1189/summary.json \
  --output outputs/student_tu_100epochs_RUN_ID \
  --wandb-mode online
```

For the authorized joint10 run, add `--lora-teacher` and use the fresh output `outputs/student_lora_teacher_10epochs_20260928` instead. Do not launch twice; inspect its receipt/runtime and live PIDs first.

Output must not exist. It contains source/config/manifest hashes, a source archive, local epoch history, the rolling checkpoint, final query/gallery embeddings and per-query metrics, and the fixed-control comparison. Final status is `FINISHED_UNVERIFIED`, not independent encoder-replay certification. Do not modify source/data while a run is active.
