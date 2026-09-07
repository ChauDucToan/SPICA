# Sketchy pairing manifest preparation

Generated CPU-only metadata (new output; the prior manifest was not overwritten):

- Manifest: `outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.json`
- Manifest SHA256: `545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c`
- Source audit: `outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.source_audit.json`
- Source audit SHA256: `4b219609ec722603557271ed9cf63c02113a04363eea5c8dfbe1d760834abf84`
- Records: 46,624 pseudo-train sketch/photo pairs

The builder uses the `seed=3407`, 20-class pseudo validation split from
`configs/data/sketchy_104_21.yaml`. Records contain relative sketch/photo
paths, labels, and SHA256 identities. Photos are restricted to
`256x256/photo/tx_000000000000_ready`; the extended branch is rejected.
The loader additionally requires the normalized sketch stem (removing a final
`-integer`) to equal the canonical photo stem and requires the same class
folder. The held-out pseudo classes are recorded separately and are not
included in training records. The label is an operational class annotation
used for the filename convention, not independently verified fine-grained pairing annotation.

Pool and source evidence from the generated manifest:

- Train sketches: 46,624; train photos: 58,950.
- Canonical photo pool: 8,400; mapped photo pool: 8,400.
- `canonical_photo_pool_equals_mapping: true` (no pool broadening was applied).
- Source audit: 55,024 unique used files, all 55,024 hash-matched against the
  read-only source root `/home/oslamelon/Downloads/Research/ZS BIR dataset/Sketchy`.
  No images were copied.

Reproduction and validation commands:

```text
.venv/bin/python scripts/build_sketchy_pairing_manifest.py \
  --data-config configs/data/sketchy_104_21.yaml \
  --pseudo-val-seed 3407 --pseudo-val-num-classes 20 \
  --output <NEW_OUTPUT_DIRECTORY>/sketchy_pseudo_train_pairing.json
.venv/bin/pytest -q tests/test_sketchy_pairing.py
.venv/bin/ruff check src/spica/data/pairing.py src/spica/data/datasets.py scripts/build_sketchy_pairing_manifest.py tests/test_sketchy_pairing.py
sha256sum outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.json \
  outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.source_audit.json
```

Choose a new output path when reproducing: the builder refuses to overwrite existing metadata. The build completed with 46,624 records. The CPU test result was `5 passed`
and Ruff passed. This work changes no source images or dataset manifests and
does not claim official evaluation or independent ground truth.
