# Official MP-Q sequential campaign — 2026-09-12

## Authorization and locked protocol

User explicitly selected MP-Q for all three official datasets, fresh initialization,
seed42/B32, approximately 2.47 sketch-observation passes, in this order:

| Dataset | Train sketches/photos/classes | Test queries/photos/classes | Updates / warmup |
|---|---|---|---|
| Sketchy104/21 | 57,587 / 72,949 / 104 | 12,694 / 12,553 / 21 | 4,446 / 222 |
| TU-Berlin220/30 | 15,400 / 176,081 / 220 | 2,400 / 27,989 / 30 | 1,189 / 59 |
| QuickDraw80/30 | 236,080 / 149,428 / 80 | 92,291 / 54,151 / 30 | 18,229 / 911 |

Updates = round(3600 × train sketch count / 46624); warmup = round(.05 × updates).
Reference exposure is 3600×32/46624 = 2.4708304735758406. Sketchy official is
**not comparable to historical pseudo84/20**. Labels restart at zero: checked
semantic class-name and path disjointness, not numeric-label disjointness.

Unchanged main MP(mu_I), inference q; not QMP or SREF. Same architecture, loss
coefficients, full seen photo pool, one positive/one negative per sketch and at
most 64 unique live photos per update. No new augmentation, selection or loss.

Reuse [minimal runtime policy](minimal_runtime_policy.md): fixed seen-train32/256
probe clean+9 every5 updates, six curves only; AP200 denominator min(R,200).
P@all is summary-only (.03125 on probe); probe P200 ceiling .04. This is monitoring,
not validation. One atomic rolling checkpoint every100/final, no production
step0/best artifacts or uploads. No exact data-stream resume claim.
Each dataset trains then evaluates its final checkpoint on official clean+9;
`final/...` W&B summary stays separate from probe history. No retries/resume.

## Fresh gates and preserved failures

Evidence root: `outputs/minimal_official_gate_20260912T050000Z/`.

- Fresh metadata-only all-three audit, no image decoding, SHA-bound official
  manifests/configs; Sketchy adapter reuses the existing common sampler.
- 29 CPU tests after restore fix, plus six archived SREF/legacy loss regression
  groups (no new SREF training). Source/manifest/probe/checkpoint/runner contracts.
- Actual two CUDA optimizer updates per dataset, preserving production horizons;
  179 optimizer states/358 finite nonzero moments. Existing trainer unchanged by
  subsequent evaluator/runner fixes. Independent CPU replay of all64 initial
  traces/masks and3LR rows per dataset; original frozen state preserved.
- Actual current runtime probe at smoke step2:32 queries/256 photos, ten conditions,
  zero predictor calls, unchanged model state. Independent FP64 NumPy full sorts
  for960 saved rows, AP tolerance2e-5/P2002e-6, no tolerance changes. Single probe
  calls measured .339/.323/.323 seconds; not a sustained throughput benchmark.
- Real compact CUDA restore initially **failed**: omitted frozen random
  `photo_model.sketch_prompt` was reconstructed without the training seed.
  `compact_failure_diagnostic.json` isolates this sole mismatch; seed-before-cache
  construction restores full state exactly. Fixed evaluator only, no model/loss
  change, no repeat optimizer runs. Original failure log retained. All three
  compact checkpoints then passed actual whole-model CUDA reconstruction hashes,
  including original CLIP, without image evaluation or optimizer updates.
- Original CPU worker compared optimizer states after scheduler construction had
  reset LR in both copies. Parent `verify_parent_restore.py` closes this gap by
  constructing scheduler first, restoring both, and comparing every optimizer
  and scheduler value directly to saved payloads. Original worker evidence retained.
- Runner worker directory/checkpoint-record/final-probe schema bugs caught before
  launch and fixed; original files retained in planning `worker_originals/`.
- Fresh CPU tests bind current runner/evaluator, fresh CUDA receipts bind evaluator;
  **all other covered training/probe components remain exact to smoke archives**.
  Smoke full-source hash `d7cda7843d75d090f9904e3bb06dea8ec450cafb48fbe05b4d3e908abf9494d4`
  (608 files) is not the later production source snapshot. No old launch gate reused.
- One explicitly synthetic online backend run verified six metrics plus axis and
  post-finish API summary writes. No official metric/model upload in the gate.

## Execution boundary

Runner: `scripts/run_coupled_benchmarks.py`, inert without `--launch`.
Fresh planned root: `outputs/minimal_official_execution_20260912T063000Z/`;
adjacent launch receipt, runtime status and live PIDs are authoritative, not the
root timestamp. Inspect before launch; never duplicate an existing campaign.
Freeze tracked source/config/docs once launched through all three final evaluations.
Keep historical artifacts and unrelated dirty files; no cleanup/push.

Completion writes `TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED` and local results;
this is not independent full evaluator/encoder replay certification. No automatic
additional experiments, test-driven tuning or checkpoint selection.
