# Corrected SPICA transport summary (2026-09-02)

All selection uses pseudo-unseen validation and structured `experiment_role`; official unseen is excluded.

## Integrity status
- Causal decomposition — NOT_RUN: matched structured roles endpoint0_factorial_B/D have not both run
- Endpoint=0 factorial — NOT_RUN: structured endpoint=0 factorial roles not run: A, B, C, D
- P1 freeze × optimizer — COMPLETED: raw artifact validated
- P2 two-stage transport — NOT_RUN: P2 stage 1 and all S1-S4 structured runs are not complete; S0 is stage-1 z0
- P3 projection refit — NOT_RUN: P3 adapter/projection refit has not run
- P4 direction ablation — NOT_RUN: P4 frozen-origin direction runs have not completed
- P5 statistical confirmation — NOT_RUN: P5 three-seed confirmation has not run
- P6 deterministic K — NOT_RUN: P6 is deferred until P1-P5 finalists
- Mo-vMF — DEFER

## Scientific result
P1 raw branch trajectories are available in the JSON and plot sidecar; interpretation must use their matched late metrics.

Cell-wise peaks, when present, are labeled best-achievable/early-stopped comparisons and are not pointwise causal effects.
Missing values are `null` with an explicit status/reason; no historical headline was carried forward.

## Raw artifacts
- `freeze_optimizer_A`: `outputs/experiments/corrected_p1_A_trainable_restored/2026-09-02_21-34-08/run_result.json` (valid)
- `freeze_optimizer_B`: `outputs/experiments/corrected_p1_B_frozen_restored/2026-09-02_21-48-56/run_result.json` (valid)
- `freeze_optimizer_C`: `outputs/experiments/corrected_p1_C_trainable_reset/2026-09-02_22-02-25/run_result.json` (valid)
- `freeze_optimizer_D`: `outputs/experiments/corrected_p1_D_frozen_reset/2026-09-02_22-17-38/run_result.json` (valid)
- `freeze_optimizer_source`: `outputs/experiments/corrected_p1_source73/2026-09-02_21-26-39/run_result.json` (valid)
