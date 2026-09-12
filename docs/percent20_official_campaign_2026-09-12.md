# Official test every20% — authorized restart ALL (2026-09-12)

User requested stop/change/rerun and explicitly selected all three datasets again.
Prior periodic1000 campaign was stopped by SIGINT to verified runner19102 and
QuickDraw process group20474 at12:16UTC. Sketchy/TU had completed; QuickDraw was
interrupted. Preserve that entire root and `user_stop_20percent.json`:
`outputs/periodic_official_execution_20260912T093000Z/`. No resume/reuse of weights.

## Locked behavior

Same fresh seed42/B32, MP-Q (mainMPmu_I/inferenceq), architecture/loss/trainpool,
matched exposure and order. Only cadence and immediate W&B commit change.

| Dataset | Horizon/warmup | Full official clean+9 test steps |
|---|---|---|
| Sketchy104/21 | 4446/222 | 890,1779,2668,3557,4446 |
| TU220/30 | 1189/59 | 238,476,714,952,1189 |
| QuickDraw80/30 | 18229/911 | 3646,7292,10938,14584,18229 |

Shared `official_test_steps` computes ceil(total*i/5), i=1..5; duplicates removed
for tiny test horizons. New profile `test_every_percent:20, checkpoint_every:100`.
Explicit `--periodic-test` remains opt-in; legacy minimal default unchanged.
Full official gallery/queries each time; no probe/stale embedding cache. Same-run
six curves `test/cleaned/*`, `test/masked/*` (mAP@200 min(R,200), mAP@all, P@200),
x-axis step_train. P@all/definitions summary-only. Intermediate summaries-only
storage; full features/ranking/masks only final. Rolling checkpoint saved before
any evaluation; final primary, no best selection. Repeated observed test is not a
blind holdout or exact paper reproduction. No model/loss/calibration change.

## W&B correction

Shared complete metric calls now pass `commit=True`. Previously explicit step
implied commit=False; sparse test points stayed buffered. Actual online synthetic
gate checks **both points visible before finish**, exact six test values and state
running. No model/data upload. This certifies immediate history commit, **not the
server's long-idle Crashed classification**. Do not claim a long-idle heartbeat fix.
Historical histories/statuses remain unchanged; no fabricated/missing-value fills.

## Fresh readiness

`outputs/percent20_gate_20260912/`:
- 51 CPU tests; six archived loss regression groups; Ruff/Hydra/diff checks.
- Fresh2CUDA updates/dataset, production horizons and20% config, with batch256
  train-only clean+9 evaluation interposed after update1 and before update2 loss.
  Model, RNG, optimizer, module flags preserved; zero official images opened.
- Independent CPU all step0/2 model+optimizer+scheduler+RNG values exact to prior
  periodic smoke;64traces/masks3LR byte-exact. Original CPU restores and179states/
  358finite nonzero moments direct payload-equal. Bounded continuation evidence,
  not exact arbitrary dataloader resume certification.
- 7680 independent FP64 saved query-condition full sorts, all5 metrics and masked
  means PASS; maxAP1.0584e-7, declared AP2e-5/P2002e-6 unchanged.
- Current compact saver exercised in **test-owned TemporaryDirectory** and actual
  CUDA whole-model reconstruction PASS for all3. Temporary compact serialization
  removed by its context manager; original step0/step2 artifacts retained. No old
  checkpoint/dataset/archive cleanup. CPU worker initially loaded full state while
  labeling a subset check; parent separately loaded only179 trainable keys and
  checked exact state. Original evidence retained. No claim temporary files are
  available for later SHA re-reading or independent fresh CPU compact reconstruction.
- Source archive616 files; production source may include this later documentation.
  Fresh gate binds current critical components, data, archived smoke and live logging.

## Disk and launch boundary

Measured historical final run/eval footprint approximately7GiB combined, plus1GiB
rolling temporary replacement and3GiB reserve: prelaunch free-space floor **11GiB**
for this periodic mode. About11.6GiB remains after fresh gates. Intermediate tests
save no repeated large arrays. No historical artifacts deleted. This is a storage
budget, not a guarantee against unrelated disk consumption.

Fresh planned root `outputs/percent20_official_execution_20260912T133000Z/`;
adjacent launch receipt and runtime/PIDs authoritative; never duplicate/retry.
Once active freeze source/config/docs through all3datasets. Idle inhibition only,
avoid manual sleep/reboot. Finish raw
`TRAIN_AND_PERIODIC_OFFICIAL_EVAL_FINISHED_UNVERIFIED`, not independent full official
encoder replay certification. No automatic retries/resume/newseeds/search/push.
