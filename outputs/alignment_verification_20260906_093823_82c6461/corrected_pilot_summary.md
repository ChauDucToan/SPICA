# Corrected pilot: verified raw-metric replay

Seed42; pseudo split3407; 84 train / 20 pseudo-validation classes; covariance=0. Fixed horizon1800.

| Arm | Seed | mAP@1800 | Delta vs R | Matching status |
|---|---:|---:|---:|---|
| R | 42 | 0.671575253 | +0.000000000 | VALID_CONTROL |
| MD | 42 | 0.667110088 | -0.004465165 | MATCHED |
| MS | 42 | 0.654153227 | -0.017422026 | MATCHED |

| Arm | Peak mAP | Peak step | mAP@1800 | Absolute decay |
|---|---:|---:|---:|---:|
| R | 0.671575253 | 1800 | 0.671575253 | 0.000000000 |
| MD | 0.667110088 | 1800 | 0.667110088 | 0.000000000 |
| MS | 0.654153227 | 1800 | 0.654153227 | 0.000000000 |

Peak is only the maximum over the five scheduled evaluations, not a continuous-training maximum. Candidate peaks are not compared to a fixed-step control.

## Query uncertainty

Exact query IDs/order and gallery identity match before bootstrap. 10,000 paired resamples; bootstrap seed3407.

| Difference | Mean AP delta | 95% query-bootstrap CI |
|---|---:|---|
| MD − R | -0.004465165 | [-0.006198697, -0.002797231] |
| MS − R | -0.017422026 | [-0.019023598, -0.015863415] |
| MS − MD | -0.012956861 | [-0.014351134, -0.011567708] |

These intervals quantify queries of one seed/split, NOT robustness across seeds/classes/splits. Bootstrap and checkpoint mAP both use float64 means of the stored AP values (independent recomputation agrees within 1e-12).

## Geometry and efficiency

| Arm | Step | mAP | Semantic margin | Corrected mean gap | Sketch effective rank | Sketch mean feature variance |
|---|---:|---:|---:|---:|---:|---:|
| R | 0 | 0.174470 | 0.148904 | 0.936955 | 27.254 | 0.00025949 |
| R | 500 | 0.657262 | 0.442685 | 0.849792 | 24.013 | 0.00138274 |
| R | 1800 | 0.671575 | 0.475544 | 0.796686 | 23.204 | 0.00146824 |
| MD | 0 | 0.174470 | 0.148904 | 0.936955 | 27.254 | 0.00025949 |
| MD | 500 | 0.659084 | 0.457037 | 0.775479 | 23.489 | 0.00139524 |
| MD | 1800 | 0.667110 | 0.489653 | 0.733997 | 21.973 | 0.00145863 |
| MS | 0 | 0.174470 | 0.148904 | 0.936955 | 27.254 | 0.00025949 |
| MS | 500 | 0.651381 | 0.407546 | 0.649998 | 23.237 | 0.00130041 |
| MS | 1800 | 0.654153 | 0.459623 | 0.625390 | 22.204 | 0.00140990 |

Mean gaps use fixed first16 images/class, correctly routed photo prompts and valid log-map samples; no fitting on validation. Spread/rank and semantic margin use the existing training-probe subset/protocol; these are different summaries, not identical subsets.

| Arm | Training seconds | Amortized seconds/update | Peak CUDA allocation (bytes) |
|---|---:|---:|---:|
| R | 1065.089 | 0.591716 | 8425837568 |
| MD | 1112.815 | 0.618231 | 8425838592 |
| MS | 1062.814 | 0.590452 | 8425838592 |

Timing includes scheduled probes after step0; it is not isolated optimizer/kernel timing. Peak allocation excludes the initial probe reset. Raw and weighted losses and per-objective gradient norms/ratios/cosines at0/500/1800 are in `raw_metrics/offline_gradients_*.json`; full per-update raw losses are in each raw history. Offline diagnostics never perturb training.

## Conclusion

Negative result for this seed42/pseudo-split pilot: MD and MS both lose to R at step1800. No mainline promotion.
No novelty claim follows from these numbers. Mean agreement alone does not establish retrieval utility. No extra lambda, covariance, horizon, seed or official-unseen campaign was tried.

## Traceability

`corrected_pilot_summary.json` links every row to run_result/config/source, exact checkpoint step/path/SHA256 and raw history. `resolved_configs/`, `raw_metrics/`, `matching_validation.json`, `checkpoint_state_validation.json` and `training_source_fixed.tar.gz` retain the evidence. Historical results are inventory-only, not numbers substituted into these tables. This bundle is local, not published.
