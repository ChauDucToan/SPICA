# SPICA frozen-prompt final campaign

All primary cells were trained under one clean experiment-code commit. Selection used only pseudo-unseen validation; the official unseen split was never read.

## Primary fixed-step evidence

FP3 versus FP3S at step 5400: mean delta=0.015427; direction consistent=True; retain photo prompt=True.
FP-LN versus matched FP2 at step 5400: mean delta=0.003479; direction consistent=True; retain LayerNorm=False.
Selected finalist: `frozen_prompt_final_FP3`; prompt-versus-FP5 mainline gate=True.
All paired comparisons include 2000 query-bootstrap draws. Query uncertainty, training-seed variance, and split variance are separate JSON fields.

## Extended training

- frozen_prompt_final_FP3: peak 0.675976 at step 9000; decay 0.001874; retention 0.997227; **plateaued**.
- frozen_prompt_final_FP3S: peak 0.661919 at step 9000; decay 0.000944; retention 0.998574; **plateaued**.

A peak at step 10800 is a boundary peak, not convergence.

## True split robustness

Splits [101, 202, 303] use independently retrained prompt and FP5 models. Training/validation class lists, hashes, checkpoint locations, and query AP vectors are raw fields in the JSON report.

## Provenance

Repository commit: `da14bc1f07575c74a23ee9afd31deb92c70f26e1`; experiment-code commit: `da14bc1f07575c74a23ee9afd31deb92c70f26e1`; tracked tree clean: `True`.
Legacy v2 artifacts are excluded from every primary conclusion; no automatic legacy substitution is performed.

## Plots

- outputs/frozen_prompt_final_campaign_2026-09-04_da14bc1f0757/frozen_prompt_final_photo_ablation.png
- outputs/frozen_prompt_final_campaign_2026-09-04_da14bc1f0757/frozen_prompt_final_layernorm.png
- outputs/frozen_prompt_final_campaign_2026-09-04_da14bc1f0757/frozen_prompt_final_extended_training.png
- outputs/frozen_prompt_final_campaign_2026-09-04_da14bc1f0757/frozen_prompt_final_seed_confirmation.png
- outputs/frozen_prompt_final_campaign_2026-09-04_da14bc1f0757/frozen_prompt_final_split_robustness.png
- outputs/frozen_prompt_final_campaign_2026-09-04_da14bc1f0757/frozen_prompt_final_geometry.png
- outputs/frozen_prompt_final_campaign_2026-09-04_da14bc1f0757/frozen_prompt_final_attention.png

FINAL SPICA FROZEN-PROMPT VERDICT

Repository commit: da14bc1f07575c74a23ee9afd31deb92c70f26e1
Experiment-code commit: da14bc1f07575c74a23ee9afd31deb92c70f26e1
Results commit: pending artifact commit
Working tree tracked files clean: YES
Historical artifacts preserved: YES
Artifact provenance valid: YES
Official unseen used for selection: NO

FP3 mean ± std: 0.674310 ± 0.001720
FP3S mean ± std: 0.658883 ± 0.004220
Photo-prompt mean effect: 0.015427
Photo-prompt direction consistent: YES
Should photo prompt remain: YES

FP2 mean ± std: 0.656985 ± 0.008871
FP-LN mean ± std: 0.660464 ± 0.008314
Matched LayerNorm mean effect: 0.003479
LayerNorm direction consistent: YES
Should LayerNorm remain: NO

Best mAP@5400: 0.676268
Best mAP@10800: 0.674102
Extended peak: 0.675976
Extended peak step: 9000
Converged or boundary peak: plateaued

Selected prompt mean ± std: 0.674310 ± 0.001720
FP5 mean ± std: 0.641848 ± 0.002541
Prompt-minus-FP5 mean delta: 0.032462 ± 0.003735

Split-101 prompt / FP5: 0.640284 / 0.639516
Split-202 prompt / FP5: 0.763964 / 0.729794
Split-303 prompt / FP5: 0.740669 / 0.715034
Across-split prompt mean ± std: 0.714972 ± 0.065722
Across-split FP5 mean ± std: 0.694782 ± 0.048427
Across-split mean delta: 0.020191

Best prompt configuration: frozen_prompt_final_FP3
Trainable parameter count: 6656
Frozen CLIP byte-identical: YES
Text enters predictor: NO
Text required at inference: NO
Photo required for query inference: NO

Should transport return: NO
Should direction supervision return: NO
Should distance prediction return: NO
Should Mo-vMF/K>1 run now: NO

Strongest supported mechanism: frozen visual prompting trained by retrieval ranking plus loss-only soft-text classification
Largest remaining confound: training-seed and pseudo-class-split variance despite independent split retraining
Recommended mainline architecture: frozen_prompt_final_FP3
Recommended mainline loss: rank loss plus soft-text CE at the query
Recommended inference query: raw sketch image
