# Pairing pilot CPU integration gate

The runnable gate is `scripts/check_pairing_pilot_integration_cpu.py`. It calls the production `spica.train_frozen_prompt.run` loop for both new pairing arms; it does not reimplement the rank loss, text CE, optimizer, checkpoint writer, evaluation, or invariant validators. The test seams are limited to a generated tiny image fixture, a tiny randomly initialized OpenCLIP-compatible CLIP (both visual and text towers), and the trainer's CLIP loader. No dataset download or source dataset access is performed.

## Command

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/check_pairing_pilot_integration_cpu.py
```

The default output is a unique `outputs/pairing_cpu_gate_<id>/` directory. Use `--output-root /tmp/spica-pairing-cpu-gate` for a named disposable directory; it must not already exist. Exit status `0` means PASS. Exit status `2` means PENDING because worker-2 has not yet supplied `configs/experiments/pairing_pilot_A.yaml` and `pairing_pilot_B.yaml`. Other nonzero status is a failed gate.

The gate checks, for both arms (with the production validator and no config mutation):

- actual production loader, forward path, inline `F.softplus` rank loss (including non-detached photo path), hard-text CE, backward, AdamW step, evaluation, checkpoints, and report;
- finite nonzero sketch/photo prompt gradients and changed prompt state;
- byte-identical CLIP-owned weights, no CLIP optimizer membership, and the production freeze policy;
- no text, photo, or oracle class required for query inference and no official unseen selection;
- identical traced queries, labels, and negatives across A/B; three deterministic mapped canonical originals per class make A's positive pool non-vacuous, while B is exactly the assigned pair;
- both arms change sketch and photo prompts separately from their step-0 checkpoints, keep the same initialization, hard-text outputs, and validation order, and preserve an all-state tiny-CLIP before/after snapshot;
- identical query/gallery evaluation identities and order;
- exact resolved-config pairing path, pairing SHA, data-manifest identity, source/config consistency, campaign/run-kind/actual-step bindings in reports and checkpoints; actual checkpoint-file SHA256 for every history row, checkpoint index, and fixed-step selection;
- intended A/B configs match before fixture overrides, except role, positive-sampling treatment, and output identity.

The fixture uses 220 synthetic classes (200 pseudo-train plus 20 pseudo-unseen), 11 photos per class (2,420 total; 220 in the 20-class validation gallery, above P@200), three mapped canonical sketch/photo originals per class, seed `42`, and pseudo split seed `3407`, with two finite updates per arm. It is deliberately not a real pretrained-backbone or throughput test: the tiny CLIP is random and CPU-only, and all evaluation is synthetic/pseudo fixture data. Each run is an explicit `run_kind=smoke`, `device=cpu` artifact selected at its actual step 2 and reported by production as `CPU_SMOKE`; the gate summary remains `CPU_SYNTHETIC_SMOKE_PASS` with `fixture_only: true`. It is not a primary/matched step-1800 experiment. A pretrained CPU forward/backward may be run separately when a local checkpoint exists, but the gate never downloads one and never claims pretrained or GPU evidence.
