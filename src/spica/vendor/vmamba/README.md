# Pinned upstream VMamba source

Source: https://github.com/MzeroMiko/VMamba
Commit: `2ed52ead062a51a64521ed3871d52914bf532876`.

`vmamba.py`, `csm_triton.py`, `csms6s.py` and `mamba2/*.py` are byte-for-byte copies of `classification/models/` at this commit. The accompanying Tiny YAML is copied from `classification/configs/vssm/vmambav2v_tiny_224.yaml`. The top-level package initializer is intentionally empty: it does not import unrelated classification model registries.

The authorized experiment uses **VMamba-Tiny s1l8 / SS2Dv2 / v05_noz**, not the optional Mamba-2 implementation. Those upstream Mamba-2 helper files remain because the unmodified upstream module imports them eagerly. No upstream Python or selective-scan kernel math is modified. Do not Ruff-format these files; their exact upstream hashes are part of the experiment provenance.

Build the author's `selective_scan_cuda_oflex` extension from `kernels/selective_scan/` at the same commit, against the active Torch/CUDA/Python ABI. The experiment factory requires this extension and does not permit the upstream sequential Python scan fallback. Kernel wheel/build receipts and SHA bindings live in ignored `outputs/`; no compiled binaries are committed. Install optional pure-Python dependencies with the `vmamba` extra. Keep existing Torch, torchvision, Triton and Mamba package versions unchanged.

The project adapter changes only the native classifier output size from 1000 to 512 and L2-normalizes its output. Backbone stages, initialization, SS2D, native pooling and drop-path schedule are retained. This is retrieval adaptation, not an ImageNet training reproduction; the fresh experiment uses RGB [0,1] with the historical Student geometry, no pretrained VMamba weights, no ImageNet normalization and no new augmentation.

MIT license: see `LICENSE`. Retain upstream copyright notices in the copied helper files.
