# Repository Guidelines

## Agent-Specific Instructions

Agents must not create or delete project files, modify non-Markdown project files, or execute commands that change repository state. Only inspect the repository, analyze issues, edit existing Markdown files, and provide proposed changes for the user to implement.

Treat this file as the durable record of repository conventions, current architectural decisions, and implementation order.

## Primary Research Protocol and Architecture

### Primary protocol: standard ZS-SBIR

The primary SPICA result must follow standard zero-shot sketch-based image retrieval:

- Train on seen classes and evaluate on disjoint unseen classes.
- At inference, the model receives only a query sketch, a photo gallery, and learned model parameters.
- Rank gallery photos from sketch-conditioned scores. Ground-truth query labels are evaluator-only metadata and must never enter the retrieval model.
- The primary method must not require the unseen-class text bank at inference. This keeps comparisons with SketchLVM/CLIP-AT, SpLIP, SeCo-SBIR, and visual ZS-SBIR baselines protocol-compatible.
- Learned class-agnostic prompts are allowed at inference because they are model parameters rather than query-label information.
- Match dataset split, gallery construction, and metric definitions (`mAP@all`, `mAP@200`, `P@K`) before comparing reported numbers across papers.

### Primary architecture: sketch-conditioned Mo-vMF retrieval

Keep conditional mixture-of-von-Mises-Fisher prediction on the frozen CLIP photo hypersphere as the primary research contribution. For a sketch `s`, the model predicts:

```text
{pi_k(s), mu_k(s), kappa_k(s)} for k = 1, ..., K
```

with mixture weights summing to one, unit component directions, and positive concentrations. Frozen or consistently prompted CLIP photo embeddings `z_p` are unit vectors. Rank a gallery photo with mixture log-likelihood:

```text
score(s, p) = log sum_k [
    pi_k(s) C_D(kappa_k(s)) exp(kappa_k(s) mu_k(s)^T z_p)
]
```

The standard inference path is therefore:

```text
query sketch
-> visual/context encoder
-> sketch-conditioned Mo-vMF head
-> mixture likelihood against frozen photo embeddings
-> gallery ranking
```

This path must not consume true class text or a selected unseen-class prompt. Text may be used during training as semantic supervision, but the trained primary model must be executable from the sketch alone.

### Role of text

For the primary protocol, valid text uses include:

- Seen-class CLIP text embeddings as training-only semantic teachers or auxiliary classification targets.
- Distillation from seen-class text into a sketch-conditioned latent that no longer needs text at inference.
- Class-agnostic learned textual or visual prompt tokens shared across queries.
- Training-only regularization of global semantic components or mixture parameters.

Do not make ground-truth unseen class text part of standard inference. If an experiment uses all unseen class names to classify a sketch and condition retrieval, label it explicitly as **unseen-vocabulary-assisted ZS-SBIR** and report it separately from the primary standard protocol.

### Status of text-fusion diagnostics

The existing hard, soft-posterior, and oracle text-fusion experiments are diagnostics rather than the primary architecture:

- Oracle text uses the true query category and is a label-leaking upper bound, never a valid primary result.
- Hard predicted text requires an unseen-class vocabulary and reduces category-level retrieval largely to zero-shot classification.
- Soft-posterior fusion forms one deterministic mean text vector and therefore collapses distinct semantic hypotheses; it is not a mixture model.
- On Sketchy 104/21 with frozen OpenAI CLIP ViT-B/32 QuickGELU, observed exploratory test results were: sketch-only `mAP=0.247028`, hard predicted text `mAP=0.698833`, and oracle text `mAP=0.919814`.
- The soft-posterior sweep converged to hard argmax as temperature approached zero (`mAP=0.697817` at `temperature=0.001`, `alpha=1`). Increasing temperature reduced mAP, and every tested curve preferred `alpha=1`. Static global sketch/text averaging therefore did not improve the hard predicted-text baseline.
- Per-query branch-complementarity diagnostic (Sketchy 104/21, same setup): text classification accuracy 0.727036. On correct-text queries (9229) sketch mAP=0.291116 vs text mAP=0.924725; on incorrect-text queries (3465) sketch mAP=0.129601 vs text mAP=0.097170. Sketch rescue rate on incorrect queries was 0.594805 with mean AP gain 0.032431. Oracle per-query routing (max of the two branch APs) reached mAP=0.714054, a maximum possible gain of 0.015221 over predicted text.
- Score-margin bucketing (quartiles of top-1/top-2 class-score margin): text accuracy is monotone in margin (0.387/0.620/0.909/0.992 from low to high) and text mAP beats sketch mAP in every bucket (0.4076 vs 0.1421 in the lowest-margin quartile; 0.9389 vs 0.4164 in the highest). A sketch-route rule breaks even only when it isolates failed predictions at roughly 95% precision; no margin bucket reaches that, and entropy adds no new information because it is monotone in margin through softmax.

These findings reject deterministic global soft averaging for the primary path. They do not reject conditional Mo-vMF: averaging class text embeddings produces one direction, whereas Mo-vMF retains separate photo-space components and scores them with mixture likelihood. Do not conflate uncertainty over class names with one-to-many variation among valid photos.

### Agreement Gate status

Agreement Gate between the sketch-only branch and the predicted-text branch is **rejected** based on the diagnostics above. Complementarity is real but too small and too diluted: the expected gain on failed predictions (+0.032 AP) is dominated by the cost of mis-routing successful ones (-0.634 AP), and the per-query oracle ceiling of +0.0152 mAP leaves no room for a practical router. Do not build a cross-attention or learned gate for this branch pair.

If gating is reconsidered later, it must satisfy all of:

- Its inference inputs are available from the sketch or from class-agnostic learned parameters (a gate consuming the unseen text bank belongs to the separately named vocabulary-assisted protocol).
- A per-query diagnostic shows the intended branches are complementary in the relevant operating region, not only in aggregate.
- The expected gain justifies the added model complexity relative to the conditional Mo-vMF head.

### Implementation order from the current state

1. Preserve the existing sketch-only standard evaluation as the protocol anchor.
2. Treat the text-fusion diagnostics (soft/hard/oracle, branch complementarity, margin buckets) as closed; do not extend them unless a new signal appears.
3. Agreement Gate is rejected; revisit only under the conditions listed above.
4. Stabilize the training dataset, transforms, and positive/negative sampling contracts.
5. Implement a sketch-only conditional Mo-vMF head and mixture-likelihood retrieval before adding text supervision.
6. Add seen-class text as training-only auxiliary supervision and verify that standard inference remains sketch-only.
7. Evaluate category-level ZS-SBIR, GZS-SBIR, and a fine-grained or instance-sensitive protocol so that one-to-many modeling is tested meaningfully.
8. Treat unseen-vocabulary-assisted fusion/gating as an optional secondary experiment.
9. Compare against deterministic point prediction and relevant probabilistic embedding baselines such as PCME/PCME++; do not claim first use without a fresh literature verification.

## Data Architecture

Keep the data package compact for the current project scope:

```text
src/spica/data/
├── __init__.py
├── manifest.py
├── datasets.py
└── transforms.py
```

- `manifest.py` reads image manifests and class maps into typed records. It does not open images or depend on PyTorch datasets.
- `datasets.py` contains both `RetrievalEvalDataset` and `RetrievalTrainDataset`. Do not create separate `eval_dataset.py` or `train_dataset.py` files unless this module becomes substantially large.
- `transforms.py` contains sketch/photo preprocessing for training and evaluation. Dataset classes receive transforms through their constructors rather than constructing transforms internally.
- Use ZSE-SBIR's conceptual separation between stochastic training samples and deterministic query/gallery evaluation, but do not copy its implementation structure.
- Keep the YAML files and their manifests as the source of truth for dataset protocols; do not hard-code unseen classes in Python.

Follow this implementation order:

1. Implement and verify `manifest.py`.
2. Add `RetrievalEvalDataset` to `datasets.py`.
3. Add minimal evaluation transforms to `transforms.py`.
4. Verify separate sketch-query and photo-gallery datasets for all four protocols.
5. Add `RetrievalTrainDataset` with class-aware positive and negative sampling.
6. Add the DataLoader or data-builder layer only after the dataset contracts are stable.

## Project Structure & Module Organization

Spica is a Python 3.12 project using a `src` layout. Put reusable package code in `src/spica/`; `src/spica/__init__.py` defines the package boundary. Dataset-specific utilities currently live under `data/Sketchy-std/`, including argument parsing in `args/`. Large local datasets belong in `datasets/`, which is intentionally ignored by Git. Store generated models as `.pt`, `.pth`, or `.ckpt` files and generated artifacts under `outputs/`; these are also ignored. Add automated tests under `tests/`, mirroring package paths where practical (for example, `tests/test_dataset.py`).

## Build, Test, and Development Commands

- `uv sync --dev` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run ruff check .` runs the configured Python linter.
- `uv run ruff format .` formats Python files consistently.
- `uv run --with pytest pytest` runs tests discovered under `tests/`; pytest is configured but is not yet a persistent development dependency.
- `uv build` creates source and wheel distributions using `uv_build`.

There is no application entry point yet. Run an individual module or script with `uv run python path/to/script.py` and pass dataset paths explicitly rather than relying on machine-specific defaults.

## Coding Style & Naming Conventions

Use four spaces for indentation and follow Ruff's default formatting. Prefer type hints for public functions and concise docstrings for non-obvious behavior. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes such as `Sketchy`; use `UPPER_SNAKE_CASE` for constants. Use `pathlib.Path` for filesystem operations and keep reusable logic out of command-line parsing modules.

## Testing Guidelines

Write pytest tests named `test_*.py` with functions named `test_*`. Cover dataset splitting, path discovery, transforms, and returned tuple shapes. Use temporary directories and small generated images instead of checking datasets or model checkpoints into Git. No coverage threshold is currently enforced; new behavior and bug fixes should include focused regression tests.

## Commit & Pull Request Guidelines

The repository has no commit history yet, so no established convention exists. Use short, imperative subjects such as `Add Sketchy dataset split tests`, and keep each commit focused. Pull requests should explain the change, list validation commands, link relevant issues, and call out data or configuration assumptions. Include screenshots or sample outputs only when visual behavior changes; never commit credentials, local dataset paths, generated outputs, or large model files.
