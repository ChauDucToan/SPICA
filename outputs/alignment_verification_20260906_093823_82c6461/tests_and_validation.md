# Executed tests and validation

## Repository/code checks

- Start HEAD: `82c6461df7931c2c9cc4d8815aab9fe16517e399` (exact reviewed snapshot); working tree clean. No `AGENTS.md` in the project or ancestor directories.
- One execution-proven defect fixed: `src/spica/train_alignment.py:1673,1713–1714` treated `evaluate_prompted()`'s `CategoryRetrievalEvaluation` as a dictionary. Three mAP accesses now use `.metrics.mean_average_precision`. All `evaluate_prompted` callers and the returned dataclass/helper were checked before this change. No model, loss, optimizer or dependency change.
- Original reproducer: `bash run_smoke.sh` (from this evidence directory, via repository-root script); R failed at `probe(0)` before any optimizer update. Exact invocation/traceback: `logs/smoke_R.log`; sequence exit1: `logs/smoke_sequence.log`.
- Recheck: `bash run_fixed_campaign.sh`; **R/MD/MS smoke50 completed, each exit0**, with real checkpoint save/cache comparison and retrieval evaluation at0/50. Paths: `fixed_attempt/logs/smoke_{R,MD,MS}.log`. The script remains a runnable end-to-end regression check; failed attempt was not overwritten.
- Patch and exact training source: `execution_fix.patch`, `source_snapshot_fixed.json`, `training_source_fixed.tar.gz`.

## Targeted pytest

Executed before AND after the fix:

```bash
CUDA_VISIBLE_DEVICES="" direnv exec . uv run --frozen --no-sync pytest -q \
  tests/test_spherical_log_map.py tests/test_alignment.py \
  tests/test_diagnose_photo_routing.py tests/test_alignment_sampler.py \
  tests/test_alignment_calibration.py tests/test_checkpoint_loading.py \
  tests/test_alignment_reporting.py tests/test_alignment_corrected_reporting.py \
  tests/test_frozen_prompt_integrity.py
```

| Execution | Exit | Passed | Failed | Raw log |
|---|---:|---:|---:|---|
| Snapshot before fix | 0 | 82 | 0 | `logs/targeted_tests.log` |
| After execution fix | 0 | 82 | 0 | `logs/targeted_tests_after_fix.log` |

Coverage includes small-angle gradients/antipodal masking, covariance n−1, real diagnostic-caller photo routing, detached/symmetric calibration, model/buffer/gradient/train-eval/RNG/sampler restoration, checkpoint loading, and corrected pairing/reporting. These are executed results, not source-reading PASS claims. The smoke failure demonstrates these tests alone did not cover the real trainer’s probe return type.

## Pairing risk checks beyond existing tests

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. direnv exec . uv run --frozen --no-sync \
  python outputs/alignment_verification_20260906_093823_82c6461/check_pairing.py
```

**13 mutation checks passed, exit0**: batch size, positive count, rank/CE weights, learning rate, weight decay, absent/different optimizer identity, MD–MS lambda, calibration canonical path/hash, different-seed-only control, duplicate candidate with higher mAP. Evidence: `pairing_negative_checks.json`, `logs/pairing_negative_checks_correct_field.log`.

An initial version of this supplemental check mutated the raw-field name `optimizer_groups` on a validated dictionary (which exposes `optimizer_identity`). Its assertion failed due to the check targeting an unused field; the check was corrected to the production validated field, not relaxed. Original failure is preserved in `logs/pairing_negative_checks.log`. No product change resulted.

Existing executed tests additionally cover corrupted calibration bytes/identity, exact checkpoint step/metric binding, missing final horizon, INCOMPLETE/UNVERIFIED suppression, no other-seed fallback, and duplicate-arm rejection.

Actual smoke artifacts were passed through the production report command with `--horizon 50 --include-smoke`; all three runs are VALID and R–MD/R–MS/MD–MS are MATCHED in **corrected_v2**, not historical mode. Evidence: `smoke_corrected_report.json`, `logs/smoke_corrected_report.log` (exit0).

## Ruff

```bash
direnv exec . uv run --frozen --no-sync ruff check \
  src/spica/train_alignment.py src/spica/alignment_artifacts.py \
  src/spica/models/alignment.py src/spica/models/checkpoint.py \
  src/spica/data/samplers.py scripts/summarize_alignment.py \
  scripts/build_corrected_manifest.py scripts/diagnose_alignment_geometry.py \
  scripts/bootstrap_alignment.py tests/test_spherical_log_map.py \
  tests/test_alignment*.py tests/test_diagnose_photo_routing.py \
  tests/test_checkpoint_loading.py tests/test_frozen_prompt_integrity.py
```

**All checks passed, exit0** before/after fix: `logs/ruff_correct_paths.log`, `logs/ruff_after_fix.log`. An initial invocation listed two nonexistent paths (`models/spherical.py`, `checkpoint_loading.py`), yielding two E902 path errors and exit1 (`logs/ruff.log`); corrected paths were then checked without changing code to silence lint.

The three one-shot evidence scripts also passed Ruff, exit0 (`logs/ruff_evidence_scripts.log`). Exact expanded commands are recorded at the start of every log.

## Real execution evidence

CUDA kernel and environment checks: `gpu_preflight.md`. Real four-batch calibration: `calibration_diagnostic_corrected.json` (source-bound artifact used by MD/MS). Final pilot matching, checkpoint states, offline diagnostics and raw-number reproduction are reported in `verification_summary.md` and `matching_validation.json`; smoke is not effectiveness evidence.
