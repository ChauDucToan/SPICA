# Corrected SPICA transport summary (2026-09-03)

All selection uses pseudo-unseen validation and structured `experiment_role`; official unseen is excluded.

## Integrity status
- Causal decomposition — NOT_RUN: matched structured roles endpoint0_factorial_B/D have not both run
- Endpoint=0 factorial — NOT_RUN: structured endpoint=0 factorial roles not run: C, D
- P1 freeze × optimizer — COMPLETED_LEGACY: historical P1 artifacts lack recorded checkpoint hashes or complete resume semantics
- P2 two-stage transport — COMPLETED_LEGACY: historical P2 artifacts lack manifest-entry identity
- P3 projection refit — COMPLETED_LEGACY: historical P3 artifact lacks manifest-entry identity
- P4 direction ablation — NOT_RUN: P4 dedicated frozen-origin direction runs have not run; P2 S1-S4 are reported separately
- P5 statistical confirmation — NOT_RUN: P5 three-seed confirmation has not run
- P6 deterministic K — NOT_RUN: P6 is deferred until P1-P5 finalists
- Split manifest-entry provenance — MIXED: some runs record manifest-entry identity and some are legacy
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
At step 5400, trainable A/C trail matched frozen B/D by 0.1936/0.2024 z0 mAP; reset changes A→C by -0.0088 and B→D by +0.0000.
P1 measurements are retained as legacy evidence: the historical artifacts do not record checkpoint hashes and complete continuation semantics required by the current validator.

P2 selected Stage-1 origin: step 73 (pseudo-unseen mAP 0.640898); all S1-S4 resume from that hashed checkpoint, freeze before the first update, reset optimizer state, and reinitialize the head.

### P2 step-5400 direction ablation
| variant | direction target | mAP | P@200 | mAP@200 |
|---|---|---:|---:|---:|
| S1 | none | 0.629499 | 0.692591 | 0.711301 |
| S2 | class_centroid | 0.626434 | 0.693944 | 0.713679 |
| S3 | moving | 0.626032 | 0.693869 | 0.713701 |
| S4 | fixed_reference | 0.624831 | 0.686769 | 0.704769 |
The best step-5400 P2 variant is S1; direction-target differences are descriptive and remain pseudo-unseen-only.

### P3 projection-refit pseudo-unseen mAP
| role | step | frozen W | orthogonal | ridge |
|---|---:|---:|---:|---:|
| freeze_optimizer_source | 73 | 0.629244 | 0.394821 | 0.237716 |
| freeze_optimizer_A | 5400 | 0.435665 | 0.206707 | 0.173007 |
| freeze_optimizer_C | 5400 | 0.426860 | 0.201351 | 0.174793 |
| freeze_optimizer_B | 5400 | 0.629244 | 0.394821 | 0.237716 |
| freeze_optimizer_D | 5400 | 0.629244 | 0.394821 | 0.237716 |
P3 refit metrics are descriptive controls, not a selected model; all fits use pseudo-train rows and evaluation uses pseudo-unseen rows only.
Matched-control recovery fractions, absolute mAP, and alignment/rank diagnostics are recorded per method; neither refit is a selected model or evidence of semantic recovery by itself.

Cell-wise peaks, when present, are labeled best-achievable/early-stopped comparisons and are not pointwise causal effects.
Official-unseen/test metrics are retained only as diagnostics and never select a checkpoint or variant.
Missing values are `null` with an explicit status/reason; no historical headline was carried forward.
Mo-vMF and K>1 remain deferred until this deterministic K=1 baseline is stable.

## Raw artifacts
- `freeze_optimizer_A`: `outputs/experiments/corrected_p1_A_trainable_restored/2026-09-02_21-34-08/run_result.json` (legacy_split_provenance)
- `freeze_optimizer_B`: `outputs/experiments/corrected_p1_B_frozen_restored/2026-09-02_21-48-56/run_result.json` (legacy_split_provenance)
- `freeze_optimizer_C`: `outputs/experiments/corrected_p1_C_trainable_reset/2026-09-02_22-02-25/run_result.json` (legacy_split_provenance)
- `freeze_optimizer_D`: `outputs/experiments/corrected_p1_D_frozen_reset/2026-09-02_22-17-38/run_result.json` (legacy_split_provenance)
- `freeze_optimizer_source`: `outputs/experiments/corrected_p1_source73/2026-09-02_21-26-39/run_result.json` (legacy_split_provenance)
- `two_stage_S1`: `outputs/experiments/corrected_p2_S1_no_direction/2026-09-03_12-21-52/run_result.json` (legacy_split_provenance)
- `two_stage_S2`: `outputs/experiments/corrected_p2_S2_class_centroid/2026-09-03_12-42-27/run_result.json` (legacy_split_provenance)
- `two_stage_S3`: `outputs/experiments/corrected_p2_S3_moving/2026-09-03_12-59-04/run_result.json` (legacy_split_provenance)
- `two_stage_S4`: `outputs/experiments/corrected_p2_S4_fixed_reference/2026-09-03_13-16-14/run_result.json` (legacy_split_provenance)
- `two_stage_stage1`: `outputs/experiments/corrected_p2_stage1_semantic_origin/2026-09-03_12-15-17/run_result.json` (legacy_split_provenance)
- `endpoint0_factorial_A`: `outputs/experiments/transport_factorial_base_no_text/2026-09-03_17-23-09/run_result.json` (valid)
- `endpoint0_factorial_B`: `outputs/experiments/transport_factorial_base_text/2026-09-03_17-46-43/run_result.json` (valid)
