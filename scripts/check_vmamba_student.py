"""Standalone CUDA check, not a test suite or official-data evaluation.

PYTHONPATH=src TRITON_LIBCUDA_PATH=/run/opengl-driver/lib \
LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH \
.venv/bin/python scripts/check_vmamba_student.py
"""

import io
import json
from functools import partial

import torch
import torch.nn.functional as F

from spica.models.vmamba_student import build_vmamba_tiny
from spica.train_student import callable_name, import_callable, multi_positive_loss
from spica.vendor.vmamba import csms6s


def check_scan():
    """Independent FP64 recurrence and gradients, including a 3136-position scan."""
    errors = []
    for length in (65, 257, 3136):
        batch, groups, channels, states = 1, 4, 2, 1
        shapes = [(batch, groups * channels, length)] * 2 + [
            (groups * channels, states),
            (batch, groups, states, length),
            (batch, groups, states, length),
            (groups * channels,),
            (groups * channels,),
        ]
        values = [torch.randn(s, device="cuda") * 0.2 for s in shapes]
        values[1] -= 2
        values[2] = -values[2].abs() - 0.5
        actual = [v.detach().requires_grad_() for v in values]
        expected = [v.detach().double().requires_grad_() for v in values]
        output = csms6s.selective_scan_fn(*actual, delta_softplus=True, backend="oflex")
        u, delta, a, b, c, d, bias = expected
        b = b.repeat_interleave(channels, dim=1)
        c = c.repeat_interleave(channels, dim=1)
        dt = F.softplus(delta + bias[None, :, None])
        state = torch.zeros(
            batch, groups * channels, states, device="cuda", dtype=torch.float64
        )
        rows = []
        for i in range(length):
            step = dt[:, :, i, None]
            state = (step * a).exp() * state + step * u[:, :, i, None] * b[:, :, :, i]
            rows.append((state * c[:, :, :, i]).sum(-1) + d * u[:, :, i])
        oracle = torch.stack(rows, dim=-1)
        weights = torch.randn_like(output) / length
        actual_grads = torch.autograd.grad((output * weights).sum(), actual)
        expected_grads = torch.autograd.grad(
            (oracle * weights.double()).sum(), expected
        )
        torch.testing.assert_close(output.double(), oracle, atol=2e-5, rtol=2e-4)
        for x, y in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(x.double(), y, atol=2e-5, rtol=2e-4)
        errors.append(
            {
                "length": length,
                "output_max_abs": (output.double() - oracle).abs().max().item(),
                "gradient_max_abs": max(
                    (x.double() - y).abs().max().item()
                    for x, y in zip(actual_grads, expected_grads, strict=True)
                ),
            }
        )
    return errors


def main():
    torch.manual_seed(42)
    assert csms6s.WITH_SELECTIVESCAN_OFLEX
    assert (
        import_callable("spica.models.vmamba_student:build_vmamba_tiny")
        is build_vmamba_tiny
    )
    assert callable_name(partial(build_vmamba_tiny)) == callable_name(build_vmamba_tiny)
    for invalid in ("", "no_colon", ":name", "spica.train_student:", "a:b:c"):
        try:
            import_callable(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)
    scan_errors = check_scan()
    model = build_vmamba_tiny().cuda().train()
    blocks = [block.op for layer in model.backbone.layers for block in layer.blocks]
    assert len(blocks) == 14 and sum(p.numel() for p in model.parameters()) == 29873792
    assert all(op.x_proj_weight.shape[0] == 4 and op.d_state == 1 for op in blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    images = torch.ones(2, 3, 224, 224, device="cuda")
    images[1, :, 50:150, 80:85] = 0
    photos = F.normalize(torch.randn(4, 512, device="cuda"), dim=-1)
    labels = torch.tensor([0, 1], device="cuda")
    photo_labels = torch.tensor([0, 1, 0, 1], device="cuda")
    # Any accidental sequential Python fallback must fail this check.
    old_reference = csms6s.selective_scan_torch
    csms6s.selective_scan_torch = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("Python scan fallback")
    )
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = model(images)
            loss = multi_positive_loss(embeddings, photos, labels, photo_labels)
        assert embeddings.shape == (2, 512) and embeddings.dtype == torch.float32
        torch.testing.assert_close(
            embeddings.norm(dim=-1), torch.ones(2, device="cuda"), atol=2e-6, rtol=0
        )
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        assert all(
            (op.x_proj_weight.grad.flatten(1).abs().sum(1) > 0).all() for op in blocks
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            before = model(images)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_vmamba_tiny().cuda().eval()
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(images), before, atol=0, rtol=0)
    finally:
        csms6s.selective_scan_torch = old_reference
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "synthetic_native_CUDA_scan_FP64_oracle_and_VMamba14blocks",
                "parameters": 29873792,
                "loss": loss.item(),
                "scan_errors": scan_errors,
                "real_training_updates": 0,
                "official_images": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
