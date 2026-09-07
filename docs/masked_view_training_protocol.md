# Frozen-prompt masked-view pilot protocol

Status: **NEW / CPU implementation review only**. No GPU training was run by this workstream.

Campaign: `frozen_prompt_masked_view_pilot_2026-09-07`

- Control: `masked_view_C`, actual role `frozen_prompt_masked_view_C`, `full_full`.
- Candidate: `masked_view_M`, actual role `frozen_prompt_masked_view_M`, `full_masked`.
- Both use one `FrozenPromptModel` visual tower with shared sketch prompt parameters across the two views and a trainable photo prompt. All original CLIP parameters, including LayerNorm, text tower, projections, and logit scale, remain frozen. There is no teacher, KD, consistency, LoRA, or SIGReg loss.
- Primary is from scratch, seed 42, pseudo split 3407, batch 32, workers 4, 1800 updates, probes `0,15,44,73,100,250,500,1000,1800`. The CPU smoke gate is 1–15 updates with batch 2.
- The existing pairing manifest is bound by SHA256 `545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c`; positive sampling is `same_class`.

Mask policy: `ink_centered_square_v1`, threshold `0.9`, train fractions `[0.25,0.5,0.75]`, evaluation fractions `[0.25,0.5,0.75]`, train seed `4242`, evaluation seeds `[101,202,303]`. The implementation computes both full and masked CPU views for every sample in both arms; C discards the masked tensor. This keeps sampling, metadata, and compute budgets aligned. Photo positives and negatives are encoded once per batch and reused by both query views. Rank and text CE are averaged across the two views with equal weight, using the trainer's inline `F.softplus` rank loss.

The append-only `train_observations.jsonl` trace records root-relative sketch identifiers, step, labels, positive/negative paths, and per-view mask metadata. Output directories must be fresh and resume is rejected. Reports/checkpoints include view mode, mask policy/hash, two-view budget, pairing identity, and fixed-step status (`PRIMARY_FIXED_STEP_UNCOMPARED` or `CPU_SMOKE`). Masked checkpoints retain the compatible `model_type: frozen_prompt_v2`; the campaign/view metadata is carried separately, so existing prompt-checkpoint loaders need no subtype exception. The campaign manifest keeps `sketch_view_mode` only in each role entry (not as an ambiguous campaign-level scalar).

Planned later command (intentionally **NOTRUN**):

```text
PYTHONPATH=src .venv/bin/python -m spica.train_frozen_prompt \
  --config-name train_frozen_prompt +experiments=masked_view_C device=cuda
PYTHONPATH=src .venv/bin/python -m spica.train_frozen_prompt \
  --config-name train_frozen_prompt +experiments=masked_view_M device=cuda
```

Evaluation is a separate worker concern: at final step 1800, independently evaluate masked queries over three fractions and three evaluation seeds, report macro-average full mAP, and do not use those results for model selection. Existing clean probes remain the fixed-step selection record.
