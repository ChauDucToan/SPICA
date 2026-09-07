# Semantic text S0–S2 — implementation and launch readiness

The user has now authorized implementing and training S0/S1/S2 and reporting on W&B. This supersedes the **design-only** permission boundary in the Step 0/1 protocol, not its scientific protocol. No official unseen, additional seeds, lambda search, later architectures, or push is authorized.

## Fixed experiment

[Protocol](semantic_baseline_step0_step1_protocol.md): S0 hard text; S1 shared four-token soft context; S2 the same with class-mean cosine T0 anchor, lambda=1 once/update. Independent three-token sketch/photo visual prompts; all original CLIP parameters frozen. Full/full concatenated query batch64, original batch32; no training masks. 3600 updates per arm, seed42/pseudo3407; seven clean/nine-condition masked probes. Select latest, best_clean and best_masked independently; positive-step earliest ties. Promote only if best_clean prefix AP200 improves over S0 and its P200/full AP do not decrease. Masked metrics report-only for this decision.

## Gates and corrections

- Parent rerun: **267 tests PASS**, Ruff PASS, `git diff --check` PASS. Evidence: `outputs/semantic_text_parent_gate_20260907T155000Z/pytest_after_fix.log`.
- Production CPU gate: `outputs/semantic_text_parent_gate_20260907T155000Z/production/gate_summary.json`; all three arms, two updates each, nine selected replays, historical C/S0 tiny-fixture prompt/loss parity. Uses production entrypoint with fixture CLIP/data and fake disabled W&B, not real training.
- Real offline CLIP no-update GPU gate: `outputs/semantic_text_preflight_20260907T150300Z/preflight_result.json`, PASS. Safetensors SHA256 `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`; hard/soft initial-bank max error0; original CLIP unchanged; trainable counts4608/6656/6656; finite nonzero task gradients, identical visual initialization/first batch. Peak allocation about4.44GB/6.71GB/6.71GB. This checks construction and gradients without updates, **not exact production iterator/probe lifecycle**. Final source-bound repetition is recorded separately under a fresh preflight root before launch.
- W&B online preflight: [n63i5big](https://wandb.ai/a-cctest05187-erd/spica/runs/n63i5big), receipt `outputs/semantic_text_execution_20260907T160000Z/preflight/wandb_online_verification_receipt.json`. API history verified at0/10/600 with retrieval/text diagnostics together; tiny artifact aliases downloaded and SHA-verified. No dataset upload.
- Independent final read-only review: PASS, no launch blockers.

Prelaunch bugs were repaired before production: duplicate context-hash key; semantic replay falling through historical1800 guards; incomplete primary protocol locks; historical1800 mask metadata regression; missing semantic probe RNG preservation; and same-step W&B logging collision. Most importantly, the CPU smoke test previously accepted failure status1: it now requires status0. Earlier broad pytest counts alone did not certify production readiness. Parent logging change briefly failed on initial `last_train=None`; failure log remains alongside the passing rerun.

Current checkpoint validation checks fixed bank/class/token/EOT identities, all-step current-context hashes, primary dimensions and anchor formula/reduction. Initial learned-bank hash is recorded, not independently recomputable from a nonzero checkpoint alone; real initial parity evidence and step0 context are retained. No claim of independently verifying that absent tensor from its hash alone.

## Source and execution boundary

The evaluator worker created commit `7c671cd17b90e27dad9489bce8098db431fa522d` before parent approval despite no-commit instructions. It is preserved, not reset. The completed implementation is committed separately after parent gates. Historical artifacts and `outputsnewgate/` remain untouched; the latter's nonignored source-like files remain in source inventories.

Fresh execution root: `outputs/semantic_text_execution_20260907T160000Z/`. Runner is inert without `--launch`, checks source before each arm, and stops on failure/source drift without retry. Train S0→S1→S2 sequentially with online tracking in `a-cctest05187-erd/spica`. Freeze source/config/docs during all arms. Trainer captures source archive/config/checkpoint/optimizer/RNG and raw probes/observation trace. Training statistics and artifact verification are pending at this readiness milestone; no S0–S2 retrieval result is asserted here.

Reproducible checks:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
git diff --check
# Each invocation uses a fresh output directory.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/check_semantic_text_integration_cpu.py --allow-smoke --output-root <NEW_DIR>
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/preflight_semantic_text.py
```
