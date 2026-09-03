# Corrected SPICA transport summary (2026-09-03)

All selection uses pseudo-unseen validation and structured `experiment_role`; official unseen is excluded.

## Integrity status
- Causal decomposition — NOT_RUN: matched structured roles endpoint0_factorial_B/D have not both run
- Endpoint=0 factorial — NOT_RUN: structured endpoint=0 factorial roles not run: A, B, C, D
- P1 freeze × optimizer — COMPLETED: raw artifact validated
- P2 two-stage transport — COMPLETED: raw artifact validated
- P3 projection refit — COMPLETED: raw artifact validated
- P4 direction ablation — NOT_RUN: P4 dedicated frozen-origin direction runs have not run; P2 S1-S4 are reported separately
- P5 statistical confirmation — NOT_RUN: P5 three-seed confirmation has not run
- P6 deterministic K — NOT_RUN: P6 is deferred until P1-P5 finalists
- Split manifest-entry provenance — LEGACY: historical runs record class/count split identity but not manifest-entry identity
- Mo-vMF — DEFER

## Scientific result
P1 raw branch trajectories are available in the JSON and plot sidecar; interpretation uses matched late metrics.

### P1 step-5400 matched values
| branch | z0 mAP | q mAP |
|---|---:|---:|
| A | 0.435665 | 0.468001 |
| B | 0.629244 | 0.634188 |
| C | 0.426860 | 0.462219 |
| D | 0.629244 | 0.633961 |
Encoder-trainable A/C are 0.1936/0.2024 mAP below frozen B/D on z0 at step 5400; optimizer reset changes the trainable branch by only 0.0088 mAP and the frozen branch by 0.0000.

P2 selected Stage-1 origin: step 73 (pseudo-unseen mAP 0.640898); all S1-S4 resume from that hashed checkpoint, freeze before the first update, reset optimizer state, and reinitialize the head.

### P2 step-5400 direction ablation
| variant | direction target | mAP | P@200 | mAP@200 |
|---|---|---:|---:|---:|
| S1 | none | 0.629499 | 0.692591 | 0.711301 |
| S2 | class_centroid | 0.626434 | 0.693944 | 0.713679 |
| S3 | moving | 0.626032 | 0.693869 | 0.713701 |
| S4 | fixed_reference | 0.624831 | 0.686769 | 0.704769 |
S1 is the best step-5400 P2 variant; adding class-centroid, moving, or fixed-reference direction supervision does not improve pseudo-unseen mAP over no direction.

### P3 projection-refit pseudo-unseen mAP
| role | step | frozen W | orthogonal | ridge |
|---|---:|---:|---:|---:|
| freeze_optimizer_source | 73 | 0.629244 | 0.394821 | 0.237716 |
| freeze_optimizer_A | 5400 | 0.435665 | 0.206707 | 0.173007 |
| freeze_optimizer_C | 5400 | 0.426860 | 0.201351 | 0.174793 |
| freeze_optimizer_B | 5400 | 0.629244 | 0.394821 | 0.237716 |
| freeze_optimizer_D | 5400 | 0.629244 | 0.394821 | 0.237716 |
P3 refit metrics are descriptive controls, not a selected model; all fits use pseudo-train rows and evaluation uses pseudo-unseen rows only.
At step 5400, ridge closes about 67–69% of the trainable-vs-frozen control deficit relative to its matched refit control, while orthogonal refit closes only about 3–4%; neither restores the frozen control's absolute mAP, so this is partial compatibility evidence rather than semantic recovery.

Cell-wise peaks, when present, are labeled best-achievable/early-stopped comparisons and are not pointwise causal effects.
Official-unseen/test metrics are retained only as diagnostics and never select a checkpoint or variant.
Missing values are `null` with an explicit status/reason; no historical headline was carried forward.
Mo-vMF and K>1 remain deferred until this deterministic K=1 baseline is stable.

## Raw artifacts
- `freeze_optimizer_A`: `outputs/experiments/corrected_p1_A_trainable_restored/2026-09-02_21-34-08/run_result.json` (valid)
- `freeze_optimizer_B`: `outputs/experiments/corrected_p1_B_frozen_restored/2026-09-02_21-48-56/run_result.json` (valid)
- `freeze_optimizer_C`: `outputs/experiments/corrected_p1_C_trainable_reset/2026-09-02_22-02-25/run_result.json` (valid)
- `freeze_optimizer_D`: `outputs/experiments/corrected_p1_D_frozen_reset/2026-09-02_22-17-38/run_result.json` (valid)
- `freeze_optimizer_source`: `outputs/experiments/corrected_p1_source73/2026-09-02_21-26-39/run_result.json` (valid)
- `two_stage_S1`: `outputs/experiments/corrected_p2_S1_no_direction/2026-09-03_12-21-52/run_result.json` (valid)
- `two_stage_S2`: `outputs/experiments/corrected_p2_S2_class_centroid/2026-09-03_12-42-27/run_result.json` (valid)
- `two_stage_S3`: `outputs/experiments/corrected_p2_S3_moving/2026-09-03_12-59-04/run_result.json` (valid)
- `two_stage_S4`: `outputs/experiments/corrected_p2_S4_fixed_reference/2026-09-03_13-16-14/run_result.json` (valid)
- `two_stage_stage1`: `outputs/experiments/corrected_p2_stage1_semantic_origin/2026-09-03_12-15-17/run_result.json` (valid)
