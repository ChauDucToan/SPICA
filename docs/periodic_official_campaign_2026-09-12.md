# Official test curves and fresh restart — 2026-09-12

## User decision (supersedes final-only monitoring for this restart)

User explicitly requested official test curves in the same training run, selected
**every1000 optimizer updates plus final**, named `test/cleaned/*` and
`test/masked/*`, requested interruption, and selected restart of **all three**:
Sketchy104/21 → TU220/30 → QuickDraw80/30. No seen-train probe in this mode.
No separate dashboard/report run. No change to architecture, loss, train pool,
seed42/B32, initialization, or matched exposure budget.

| Dataset | Updates/warmup | Official test steps |
|---|---|---|
| Sketchy | 4446/222 | 1000,2000,3000,4000,4446 |
| TU-Berlin | 1189/59 | 1000,1189 |
| QuickDraw | 18229/911 | 1000 through18000, then18229 |

Each evaluation uses **all official queries and all official gallery photos**,
clean plus9 deterministic masks. Six history metrics: mAP@200 (min(R,200)),
mAP@all, P@200 under each requested namespace, x-axis `step_train`. No history
point before step1000, no probe/loss/LR/gradient curves. P@all and definitions
remain summary-only. Checkpoint is saved before evaluation. Training pauses,
then continues with restored RNG/module flags. Full gallery features are freshly
encoded at every test, not reused from an earlier checkpoint.

Test is now **repeatedly observed monitoring**, not a blind holdout. No test-based
checkpoint/lambda/seed selection; fixed final remains primary. This is not exact
SketchLVM/SeCo recipe reproduction or confirmation of their AP200 convention.
Sketchy official remains incomparable to historical pseudo84/20.

## User-requested interruption

Prior root `outputs/minimal_official_execution_20260912T063000Z/` is preserved.
At 07:22:20UTC it was training TU (Sketchy already finished). Parent checked
runner5637 and training process group6459 identities, then sent SIGINT to those
only. No remaining campaign process at subsequent check. Original runtime/failure
logs and `user_stop_receipt.json` retain the interruption; no artifact cleanup,
checkpoint reuse, automatic resume, or old metric rewriting.

## Implementation and disk policy

Explicit `--periodic-test` on existing trainer/runner; legacy minimal mode stays
available and unchanged by default. New `configs/runtime/official_test.yaml`
contains only test_every1000/checkpoint_every100. Hydra entry skips separate
final evaluator in periodic mode, so final test is not repeated.

Common evaluator accepts `save_details=False` for intermediate tests: same full
rankings/metric math, but only small summary/provenance/condition counts are
written. **No repeated large features/NPZ/mask JSONL**, and no write-then-delete
artifact cleanup. Final test retains full existing detailed artifacts once.
One rolling trainable-only checkpoint100/final remains; original CLIP is SHA-bound
and reconstructed, no model/table/artifact uploads. W&B system/code/git/requirements
and console capture remain disabled. Memory peak is cumulative train+test in this
mode; no train-only peak or isolated latency claim.

## Fresh gates and limits

Root `outputs/periodic_official_gate_20260912/`:

- 45 targeted CPU tests, six archived loss regression groups, Ruff/diff/Hydra checks.
- New actual2CUDAupdates per dataset, with **256 distinct train queries/256 train
  photos ×10 conditions inserted after update1 and before update2 loss**. The
  same periodic wrapper/evaluator is used with explicitly train-only gate status;
  zero official images opened. Actual batch256 exercises evaluation while optimizer
  moments remain resident. No extra optimizer update during evaluation.
- Producer records model/optimizer/RNG/modes unchanged across evaluation. Independent
  CPU compares **all saved step0 andstep2 model/optimizer/scheduler/RNG values**
  exactly to previous no-intervening-eval smoke; all64traces/masks3LR byte-exact.
  This certifies the bounded two-update continuation, not arbitrary worker-stream
  resume or every future training step.
- Actual current rolling saver plus CUDA compact whole-model reconstruction all3PASS.
  Independent CPU original checkpoint restore and compact trainable/optimizer/scheduler
  restore exact:179trainable tensors,179optimizerstates,358finite/nonzero moments.
  CPU compact reconstruction does not claim an independently regenerated whole-model
  frozen-buffer hash; the whole compact reconstruction claim is actual CUDA.
- Independent NumPy FP64 full-sort checks for7680 saved train-only query-condition
  rows, all five retrieval scalars: maxAP delta Sketchy1.0584e-7/TU1.1141e-9/Q9.2817e-10,
  tolerances AP2e-5/P2002e-6 unchanged.
- Original worker verifier FAIL retained: wrongly constrained new runtime/tracking
  to old source; also had an unexecuted fullAP formula missing denominator. Parent
  review corrected both in a **separate** verifier with hand-computed oracle tests;
  no repeated GPU measurement, tolerance change, or raw evidence modification.
- Runner worker initially bypassed common checkpoint/init/stream checks in periodic
  branch; parent removed early return before gates. Regression fixtures now exercise
  common checks plus every periodic checkpoint/summary/source/state/count binding.
- Installed W&B offline journal: two synthetic rows with exact six test names and
  step axis, zero stats/artifacts, no machine/requirements files. Existing online
  authentication is checked read-only before launch; no official test pre-evaluation.

GPU smoke source614files hash
`1cfcb53c39405e40f53eaee6926b5b80ad919264d582a6edca5dde379429fee3`.
Later protocol/documentation commit is not this measurement snapshot. Fresh gate
binds production components to smoke archive and final CPU evidence.

## Launch boundary

Fresh planned root **`outputs/periodic_official_execution_20260912T093000Z/`**;
adjacent launch receipt/runtime/PIDs authoritative. Check before launch; no duplicate.
After launch freeze tracked source/config/docs until campaign terminates. Only
idle inhibition is available: avoid manual sleep/reboot. No retry/resume/push.

Completion raw status `TRAIN_AND_PERIODIC_OFFICIAL_EVAL_FINISHED_UNVERIFIED`;
local per-step consistency checks do not mean independent full official encoder
replay. Existing interrupted/historical results remain unchanged.
