# SPICA alignment objective results (2026-09-05)

## Decision

The implementation and protocol checks passed; the machine-readable Phase A
record is `outputs/alignment_protocol_audit_2026-09-05.json`. However, the proposed full
text-anchor + spherical-Log-map + covariance objective did **not** improve the
matched FP3 control in this campaign. It should not be promoted as an
improvement on this evidence.

All selection used `full_pseudo_unseen_mAP` on the seed-3407 pseudo-unseen
validation split. Official unseen metrics below are diagnostic only.

## Phase B geometry

Baseline checkpoint: `outputs/experiments/frozen_prompt_v2/frozen_prompt_v2_FP3_continuation/checkpoints/frozen_prompt_step5400.pt`

| Split | Classes | Sketches | Photos | Mean tangent distance | Covariance distance | Sketch log angle | Photo log angle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pseudo-train | 84 | 1,344 | 1,344 | 0.917677 | 0.256956 | 1.244508 | 1.263200 |
| Pseudo-unseen diagnostic | 20 | 320 | 320 | 0.915840 | 0.283732 | 1.250202 | 1.265937 |

The tangent orthogonality diagnostic was `4.49e-08` on pseudo-train and
`5.34e-08` on pseudo-unseen. Semantic class-name overlap was empty. Full JSON:
`outputs/alignment_geometry_baseline_2026-09-05.json`.

At the selected primary checkpoints, the full objective reduced its
pseudo-unseen tangent mean/covariance distances to `0.871839/0.245907` versus
`0.900047/0.272759` for the matched control, yet retrieval mAP was lower. This
is a useful negative diagnostic: moment agreement alone was not a sufficient
selection criterion.

## Phase C–D pilot

All arms used the same seed (`42`), 100 updates, 16 classes × 2 sketches per
batch, four positive photos per sketch, and the same FP3 rank/classification
losses. Deltas are versus the matched control's best pseudo-unseen mAP
(`0.596864`). Short one-update smoke artifacts, where present, were excluded.

| Arm | Best mAP | Best step | Δ vs control |
|---|---:|---:|---:|
| `alignment_control` | 0.596864 | 100 | 0.000000 |
| `alignment_mean_text_log` | 0.587313 | 100 | -0.009552 |
| `alignment_cov_text_log` | 0.593557 | 100 | -0.003307 |
| `alignment_full_text_log` | 0.585004 | 100 | -0.011860 |
| `alignment_full_chordal` | 0.594359 | 100 | -0.002505 |
| `alignment_full_photo_anchor` | 0.592522 | 100 | -0.004342 |

Pilot report: `outputs/objective_alignment_pilot_report_2026-09-05.md`.

## Phase E–F primary and replications

For context, the historical FP3 pseudo-unseen reference reached `0.672752`
at step 5,400 under its original one-positive protocol. The new matched control
reached `0.671575` at its selected step 1,800 in the primary run, so the
multi-positive/matched-sampler control did not create a large headline gain by
itself.

The primary campaign used seed 42. The separate replication campaign used
seeds 123 and 3407. Each run trained from scratch through 5,400 updates and
selected its own best pseudo-unseen checkpoint.

| Run | Arm | Seed | Selected step | Best pseudo-unseen mAP |
|---|---|---:|---:|---:|
| Primary | matched control | 42 | 1,800 | 0.671575 |
| Primary | full text + Log + covariance | 42 | 5,400 | 0.656565 |
| Replication | matched control | 123 | 5,400 | 0.673802 |
| Replication | full text + Log + covariance | 123 | 5,400 | 0.657605 |
| Replication | matched control | 3407 | 5,400 | 0.675764 |
| Replication | full text + Log + covariance | 3407 | 1,800 | 0.639631 |

Paired best-mAP deltas for the full objective are:

- seed 42: `-0.015010`
- seed 123: `-0.016198`
- seed 3407: `-0.036133`
- replication mean (seeds 123 and 3407): `-0.026165`
- all three paired deltas: mean `-0.022447`, sample SD `0.011867` (descriptive only; `n=3`)

Primary report: `outputs/objective_alignment_report_seed42_2026-09-05.md`.
Replication report: `outputs/objective_alignment_replication_report_2026-09-05.md`.
The primary/pilot runs record source snapshot
`e6f8647cf2aeaddfb289c2740349487c4808f494c7341a36dbbe5847e75d2db7`;
the replication runs record
`79873244e63adfc71614061bfba3390cb6833dd8feecbdbedbc4677f532fbdea`.

## Official unseen diagnostics

These were run only after selection and were not used to choose checkpoints.

| Arm | Selected checkpoint | Full mAP | P@200 | mAP@200 |
|---|---|---:|---:|---:|
| Matched control | primary seed-42 step 1800 | 0.666674 | 0.717647 | 0.741837 |
| Full alignment | primary seed-42 step 5400 | 0.652874 | 0.698269 | 0.722145 |

JSON artifacts: `outputs/alignment_control_official_diagnostic_2026-09-05.json`
and `outputs/alignment_full_text_log_official_diagnostic_2026-09-05.json`.

Campaign manifests: `outputs/experiment_manifest_objective_alignment_pilot_2026-09-05.json`,
`outputs/experiment_manifest_objective_alignment_2026-09-05.json`, and
`outputs/experiment_manifest_objective_alignment_replication_2026-09-05.json`.

## Integrity and inference contract

- `PYTHONPATH=src python -m pytest -q`: **110 passed**.
- `ruff check src tests scripts`: **all checks passed**.
- All alignment runs used train-only positive-photo targets; validation/test
  examples were not used to fit moments.
- Hard text embeddings were detached anchors used only by the training loss.
- Positive-photo moments were detached targets; photo prompts still received
  ranking gradients.
- The predictor takes only a raw sketch at inference; text and photos are not
  predictor inputs.
- The CLIP visual/text towers stayed byte-identical; only visual prompts and
  soft text context were trainable.

The historical FP3 re-evaluation and historical pseudo-unseen curve remain
reference artifacts only; they were not merged into these new campaign
statistics. Literature comparison and novelty boundaries are in
`docs/objective_alignment_protocol.md`.
