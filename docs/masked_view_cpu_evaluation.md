# WORKSTREAM3 masked-view CPU evaluation

This workstream evaluates a **query-only** ink mask against the final frozen-prompt
checkpoint. It does not retrain, use official unseen data, alter the gallery, or
interpret the training mAP as a masked score.

## CPU gate

The gate is intentionally explicit and smoke-only:

```bash
python scripts/check_masked_view_integration_cpu.py --allow-smoke
```

Use a fresh output path for every run; existing output directories are refused.
The gate is incomplete (not PASS) when the shared worker-1 mask API or the
worker-2 C/M campaign is absent. It runs the production two-arm CPU trainer gate
for two updates, then exercises all nine fraction/seed conditions on a complete
toy retrieval fixture. Toy retrieval numbers are synthetic and are not an mAP
claim. No GPU, download, dependency installation, or official-unseen access is
performed.

## Final evaluation

```bash
python scripts/evaluate_masked_view.py \
  --run-result <run>/run_result.json \
  --output-dir <new-output-directory> \
  [--device cpu|cuda]
```

The default is fixed step 1800 for `run_kind=primary`; `run_kind=smoke` requires
an explicit step <=15. The evaluator refuses an absent or mismatched
checkpoint, config, source snapshot, role/campaign/run kind, pairing identity,
source split, manifest, cache identity, gallery path/label order, or output
directory. It requires the approved mask plan recorded in the resolved config
(version, fractions, seeds, view=0, and threshold), rather than silently
hard-coding a replacement. It reconstructs pseudo-validation seed 3407 and
requires the corrected 10,963-query/13,999-gallery validation sizes. It loads
the full OpenCLIP visual backbone using the existing `frozen_prompt_v2`
checkpoint format and only loads trainable state from the checkpoint. It reuses
the checkpoint-bound final photo cache; every mask condition uses the same
gallery embedding and no photo re-encoding is performed.

Masks are applied to normalized query tensors before the encoder. Stable seeds
are derived from `(base_seed, dataset-root-relative-path, view=0)` and never
from Python's process hash or global RNG. The three requested fractions and
base seeds are evaluated with the same per-query realization across conditions.
`query_mask_manifest.jsonl` records path, label, mask metadata, hashes, bbox,
`ink_pixels_before/erased/after`, and status for every condition.
`masked_view_evaluation.json` records the nine-condition macro full mAP,
per-condition query AP arrays, P@200 and prefix-positive mAP@200,
finite-validated clean-forward sanity mAP, realized-fraction summaries, and
separate `blank_input`, `target_unreachable`, `zero_fraction`, and `ok` counts
per condition plus aggregate counts. Clean replay is fail-closed at an
explicit absolute tolerance of 1e-6 on CPU and 1e-5 on CUDA; no tolerance is
implicit.

The report deliberately does not provide confidence intervals across masks or
claim monotonic per-query behavior. A completed report is evidence for this
run/checkpoint/split/cache identity only; it does not promote the masked method
or replace the fixed-step raw training result. Any failure exits nonzero and
writes a `FAILED` artifact rather than a misleading `COMPLETE` report.
