# GPU preflight

- Environment: existing project `.envrc` / `.venv` on NixOS, Nix-provided `uv`/driver. This repository has no devenv/flake configuration; no environment or dependency was rebuilt/upgraded. All execution uses `direnv exec . … uv run --frozen --no-sync` (installed project environment, no lockfile resolution).
- Python: `/home/oslamelon/Desktop/Projects/spica/.venv/bin/python3`; underlying uv CPython 3.12.13.
- PyTorch: **2.13.0+cu130**; `torch.version.cuda`: **13.0**.
- Driver: **595.91.07**; GPU: **NVIDIA GeForce RTX 5070 Ti**, 16,303 MiB as reported by NVIDIA-SMI.
- At preflight: NVIDIA-SMI 10 MiB used; CUDA `mem_get_info()` free **16,351,625,216 bytes**, total **16,602,497,024 bytes**. These are preflight values, not a reservation.
- `/run/opengl-driver/lib/libcuda.so.1` resolves to `/nix/store/5pfbkh57zv5znqhd1n0zc2qwliawvgq8-nvidia-x11-595.91.07/lib/libcuda.so.595.91.07`.
- `torch.cuda.is_available()` **True**. Actual CUDA matrix forward, backward, finite/nonzero gradient assertions and `torch.cuda.synchronize()` **PASS**, exit **0**.
- No driver-lookup failure, CPU-only build, CUDA incompatibility, permissions failure or insufficient-VRAM failure occurred at preflight. No stub driver, system changes, reboot or process termination used.
- Every training/diagnostic job explicitly prefixes `/run/opengl-driver/lib` while retaining the inherited `LD_LIBRARY_PATH`; the override is applied after `direnv` so direnv cannot discard the inherited GCC/library entries. The initial preflight used the project's default direnv library path; both exact effective paths are recorded.
- GPU jobs are sequential, with no official-unseen evaluation.

Raw evidence: [`logs/cuda_preflight.log`](logs/cuda_preflight.log), [`logs/environment.log`](logs/environment.log). Exact training/diagnostic commands and process-scoped library paths are echoed in each job log.

CUDA check executed:

```python
import torch
assert torch.cuda.is_available()
x = torch.randn(128, 128, device='cuda', requires_grad=True)
y = (x @ x.T).square().mean()
y.backward()
torch.cuda.synchronize()
assert torch.isfinite(y) and torch.isfinite(x.grad).all() and x.grad.norm() > 0
```
