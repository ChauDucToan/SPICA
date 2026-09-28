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
