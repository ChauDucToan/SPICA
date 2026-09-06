# Corrected alignment historical results

All values below are recomputed from raw `run_result.json` histories; no Markdown value was copied into recomputed fields.
Fixed-step and peak tables are separate. Matching fails closed on campaign, seed, split, initialization, horizon, protocol, source, and non-treatment config differences.

## Campaign `objective_alignment_corrected_pilot_2026-09-05` — requested horizon `50` — status `VALID`

### Fixed-step per run

| Role | Seed | Status | mAP@horizon | Paired delta |
|---|---:|---|---:|---:|
| `alignment_mean_text_log` | 42 | VALID | 0.536322 | -0.003466 |
| `alignment_mean_text_log_symmetric` | 42 | VALID | 0.544296 | 0.004507 |
| `alignment_control` | 42 | VALID | 0.539788 | — |

### Peak per run

| Role | Seed | Peak mAP | Peak step | Retention | Absolute decay | Peak delta |
|---|---:|---:|---:|---:|---:|---:|
| `alignment_mean_text_log` | 42 | 0.536322 | 50 | 1.000000 | 0.000000 | -0.003466 |
| `alignment_mean_text_log_symmetric` | 42 | 0.544296 | 50 | 1.000000 | 0.000000 | 0.004507 |
| `alignment_control` | 42 | 0.539788 | 50 | 1.000000 | 0.000000 | — |

### Aggregate

| Role | Unique seeds | Paired delta mean | Sample std | n |
|---|---:|---:|---:|---:|
| `alignment_control` | [42] | — | — | 0 |
| `alignment_mean_text_log` | [42] | -0.003466 | — | 1 |
| `alignment_mean_text_log_symmetric` | [42] | 0.004507 | — | 1 |

### Matching/provenance notes

