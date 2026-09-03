# Frozen-prompt SPICA probe

> Status: incomplete primary campaign. Missing cells are `not_run`; no historical transport or JEPA result is substituted.

## Repository and artifact audit

- Starting commit: `494769741dacc8f27a04c4761637ca78e61432d0`
- Branch: `experiments`; working tree: dirty before implementation and remains dirty because historical outputs were already removed.
- Python 3.12.13; PyTorch 2.13.0+cu130; OpenCLIP 3.3.0; NVIDIA GeForce RTX 5070 Ti, 16 GB.
- Dataset: `sketchy_104_21`; pseudo-validation seed 3407; training seed 42; pseudo-train/pseudo-unseen classes are disjoint.
- `uv run --frozen ruff check .`: passed. `uv run --frozen pytest -q`: 88 passed.
- Provenance is valid for the current FP0 run; the full campaign is not yet valid because six roles are not_run.

## Exact architecture

- FP0–FP4 freeze CLIP-owned visual attention, MLP, LayerNorm, class/positional embeddings, projection, text tower/token embeddings/projection/logit scale. FP1–FP3 add separate class-independent 3-token sketch/photo prompts. FP4 adds no visual prompts and only trains the shared loss-side soft text context.
- FP-LN is explicitly **not fully frozen**: visual prompt parameters and visual LayerNorm affine parameters are trainable.
- FP5 is the existing partial depth-4 sketch-encoder adaptation reference, frozen at its selected semantic origin.
- Text is used only by the training classifier and is not required by sketch inference.
- Prompted photo galleries are re-encoded and cache identities include checkpoint hash, length, mode, modality, CLIP identity and manifest identity.

Official unseen used for selection: **NO**

## Audit

- This report does not borrow missing cells from historical transport or JEPA artifacts.
- Missing cells are represented as `not_run`.

## Pointwise results

### frozen_prompt_FP0
step | full pseudo-unseen mAP | P@200 | mAP@200
---: | ---: | ---: | ---:
0 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
15 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
44 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
73 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
100 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
250 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
500 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
1000 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
1800 | 0.269618 | 0.36702407256817415 | 0.4177867534718976
5400 | 0.269618 | 0.36702407256817415 | 0.4177867534718976

### frozen_prompt_FP1
**not_run** — primary campaign cell was not completed.

### frozen_prompt_FP2
**not_run** — primary campaign cell was not completed.

### frozen_prompt_FP3
**not_run** — primary campaign cell was not completed.

### frozen_prompt_FP4
**not_run** — primary campaign cell was not completed.

### frozen_prompt_FP5
**not_run** — primary campaign cell was not completed.

### frozen_prompt_FP_LN
**not_run** — primary campaign cell was not completed.

## Verdict

The requested campaign is incomplete, so no architecture selection or statistical confirmation is justified.

FINAL FROZEN-PROMPT SPICA VERDICT

Repository commit: 494769741dacc8f27a04c4761637ca78e61432d0
Working tree clean: NO
Artifact provenance valid: PARTIAL (FP0 current; other cells not_run)
Official unseen used for selection: NO

Vanilla frozen CLIP peak mAP: 0.269618 (FP0 current run)
Visual-prompt-only peak mAP: not_run
Visual + frozen-text peak mAP: not_run
Visual + soft-text peak mAP: not_run
Text-soft-prompt-only peak mAP: not_run
Prompt + LayerNorm peak mAP: not_run
Early-adapt-then-freeze peak mAP: not_run

Best frozen-prompt configuration: not_run
Best frozen-prompt checkpoint: not_run
Best frozen-prompt late mAP: not_run
Frozen-prompt retention ratio: not_run

Does text-only soft prompting change visual retrieval: not_run
Does trainable text prompting improve over fixed text: not_run
Do visual prompts improve over vanilla frozen CLIP: not_run
Do visual prompts match early encoder adaptation: not_run
Does LayerNorm fine-tuning help: not_run
Does prompt tuning prevent encoder forgetting: not_run

CLIP visual backbone frozen: YES for FP0–FP4; FP-LN/FP5 not_run
CLIP text backbone frozen: YES by design
Text required at inference: NO

Recommended semantic-origin architecture: defer until all primary cells run
Recommended trainable parameters: visual prompts only (candidate; not selected)
Recommended loss: rank plus optional loss-side CE (candidate; not selected)
Recommended inference query: prompted sketch image only

Should historical fixed-15 transport return: NO
Should direction supervision return: NO
Should distance prediction return: NO
Should Mo-vMF/K>1 be tested now: NO

Strongest supported mechanism: none yet for this campaign
Largest remaining confound: incomplete primary prompt study
Most important next experiment: complete FP1–FP5 and FP-LN sequentially

