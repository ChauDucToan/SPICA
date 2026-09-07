# Masked-view C/M CPU gate

The gate is an explicit fixture smoke test, not a GPU or scientific result.
It runs the production Hydra-composed `frozen_prompt_masked_view_pilot_2026-09-07`
trainer for roles `frozen_prompt_masked_view_C` (`full_full`) and
`frozen_prompt_masked_view_M` (`full_masked`) for two updates each, then calls
the production masked evaluator at step 2 for all nine fraction/seed
conditions.

```bash
PYTHONPATH=src .venv/bin/python scripts/check_masked_view_integration_cpu.py \
  --allow-smoke \
  --output-root outputs/masked_view_cpu_gate_<fresh-id>
```

The output directory must be new. The gate fails with its traceback and a
non-zero exit status; it never turns trainer or evaluator errors into
`PENDING`.

## Checks

- C and M are composed from their actual experiment configs and validated by
  the production trainer validator. The fixture records `pretrained: openai`
  in the resolved config, while the patched `load_frozen_clip` seam supplies a
  tiny CPU CLIP, so no pretrained download occurs.
- The fixture uses the existing complete 220-class / three-mapped-photo
  pairing fixture builder and its convention-valid pairing manifest. The
  fixture manifest is explicitly allowed only by the smoke overrides; it is
  never presented as the production pairing SHA.
- Both views are materialized before the encoder. C forwards two identical
  full tensors; M forwards the same first full tensor and a changed masked
  second tensor. Photo positives and negatives are one concatenated `2B`
  encode per step. The two losses are logged and checked as equal-weight
  averages; C's two view losses are numerically equal.
- The observation JSONL is production output. C/M rows match by root-relative
  path, step, label, positive/negative sample, mask seed/box/input/output
  hashes, and per-view metadata. Masked pixels are non-empty and never a
  no-op. Checkpoints bind exact SHA256, step, resolved config, split/manifest,
  source snapshot, and code commit. Prompt parameters update from step 0 to 2;
  only the two visual prompts are optimizer parameters; CLIP parameters,
  including projections, remain byte-identical.
- The production evaluator writes nine conditions, finite per-query AP,
  macro mAP, a clean replay mAP matching the raw step-2 metric, an unchanged
  gallery identity, and a hashed query mask manifest. C/M mask identities and
  hashes are compared.

This document records gate procedure and result format only. It does not edit
or replace `docs/masked_view_cpu_evaluation.md`, and it makes no GPU, download,
official-unseen, or performance claim.
