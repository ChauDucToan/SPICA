# Ink-region mask protocol

## Scope

`src/spica/data/masking.py` implements `ink_centered_square_v1`: a deterministic,
CPU-only corruption applied to a normalized CLIP image `[3, H, W]` before the
encoder. It erases an axis-aligned square of the observed raster ink by
replacing pixels with normalized CLIP white. This is **raster ink-region
erasing**, not stroke tracing or a claim about true semantic strokes.

The full and masked views use the same encoder in the planned shared-encoder
experiment. Mask selection uses the full observed raster only; its coordinates
and metadata are not passed to the model.

## API

```python
from spica.data.masking import apply_ink_mask, mask_seed

seed = mask_seed(base_seed, root_relative_sample_key, view=0)
masked, metadata = apply_ink_mask(
    normalized_cpu_image,
    fraction=0.5,
    seed=seed,
)
```

`mask_seed` uses SHA-256 over `(base_seed, sample_key, view)` and does not use
Python hash randomization or mutate global RNG state. The caller owns the seed
meaning: training uses base seed `4242`, with the global training step as
`view`, and chooses uniformly from `{0.25, 0.5, 0.75}` using its local RNG;
evaluation repetitions use base seeds `101`, `202`, and `303`. The sample key is
the root-relative path.

`apply_ink_mask` validates a finite CPU floating tensor with shape `[3,H,W]`,
finite `mean`/positive finite `std` sequences of length three, finite threshold,
and `0 <= fraction < 1`. It returns a clone and never mutates its input. The
current default CLIP mean/std are used to unnormalize only for ink selection.
Ink is `mean(R,G,B) < ink_threshold` (default `0.9`).

## Algorithm

1. Unnormalize the tensor and threshold the per-pixel RGB mean.
2. If there is ink, select one ink pixel uniformly with a local
   `random.Random(seed)`.
3. Expand a centered, axis-aligned square around that pixel, clipped to image
   borders, until the smallest radius reaching the requested fraction of ink
   pixels. The target counts thresholded ink pixels, not area or stroke length.
4. Never erase the final ink pixel. If the requested target cannot be reached
   under that constraint, use the best attainable nonblank square and report
   `target_unreachable`; a one-pixel sketch can therefore be a no-op.
5. Replace the selected box with normalized white.

For a fixed image and seed, the center is shared across fractions, so squares
are nested as severity increases. Integer raster geometry can overshoot the
target; there is no hidden guarantee of exact fraction or equal area. Blank
inputs are returned byte-for-byte equivalent by value with `blank_input`.

## Stable metadata

Every result contains JSON-native:

- `policy_version`, `seed`, `requested_fraction`, `realized_fraction`
- `ink_pixels_before`, `ink_pixels_erased`, `ink_pixels_after`
- `bbox` as `[left, top, right, bottom]` (exclusive), or `null`
- `status`: `ok`, `blank_input`, `zero_fraction`, or `target_unreachable`
- `input_sha256`, `output_sha256`

`fraction` means erased thresholded-ink count divided by ink count. Metadata is
for audit/review and is not model input.

## Review and limits

Run the CPU tests with:

```bash
.venv/bin/pytest -q tests/test_ink_region_mask.py
.venv/bin/ruff check src/spica/data/masking.py tests/test_ink_region_mask.py scripts/inspect_ink_region_masks.py
```

The review script reads the project Sketchy manifests and deterministic
pseudo-train/pseudo-validation split (seed `3407`), never official unseen
classes, and writes contact sheets plus per-query JSON under a unique
`outputs/masking_preparation_*` directory. It records selected sample counts,
blank/no-op/full-zero prevention outcomes through the per-query statuses. It is
a visual protocol check, not inference evidence and not a claim that masks
represent true pen strokes. The source dataset at
`/home/oslamelon/Downloads/Research/ZS BIR dataset` is read-only.

No hyperparameter search is part of this protocol. GPU execution, training,
downloads, dependency changes, and integration into trainer/evaluator are
outside this workstream.
