"""One synthetic forward/backward/round-trip check, outside the model.

CPU uses upstream PyTorch SSD (FP64)/RMSNorm references with a stable
chunk-state recurrence, avoiding the upstream masked-future exponential overflow.
--device cuda uses the actual production kernels; neither mode loads data/CLIP.
"""

import argparse
import io
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from spica.models.sketch_student import FourWayMamba, SketchStudent
from spica.train_student import multi_positive_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)

    # Isolate scan packing/unpacking: an identity mixer must restore a non-square grid.
    scan = FourWayMamba()
    scan.norm = nn.Identity()
    scan.mamba = nn.Identity()
    grid = torch.arange(192 * 3 * 5).float().reshape(1, 192, 3, 5)
    assert torch.equal(scan(grid), grid)

    with ExitStack() as stack:
        if args.device == "cpu":
            import mamba_ssm.modules.mamba2 as upstream
            from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, rms_norm_ref
            import mamba_ssm.ops.triton.ssd_combined as ssd

            def stable_state_passing(states, decay, initial_states=None):
                carry = (
                    torch.zeros_like(states[:, 0])
                    if initial_states is None
                    else initial_states
                )
                previous = []
                for chunk, log_decay in zip(states.unbind(1), decay.unbind(-1)):
                    previous.append(carry)
                    carry = carry * log_decay.exp().unsqueeze(-1) + chunk
                return torch.stack(previous, dim=1), carry

            stack.enter_context(
                patch.object(ssd, "state_passing_ref", stable_state_passing)
            )

            def reference_scan(x, dt, a, b, c, chunk_size, **kwargs):
                allowed = {
                    k: kwargs[k]
                    for k in ("D", "z", "dt_bias", "dt_softplus")
                    if k in kwargs
                }
                length = x.shape[1]
                pad = (-length) % chunk_size
                # Upstream reference needs whole chunks; causal tail padding
                # cannot change the original prefix. Production handles tails.
                x, b, c = (F.pad(v, (0, 0, 0, 0, 0, pad)) for v in (x, b, c))
                dt = F.pad(dt, (0, 0, 0, pad))
                # The upstream state-passing reference is unstable in FP32 on
                # long sequences. Use FP64 here only, not in the CUDA model.
                allowed = {
                    k: v.double() if isinstance(v, torch.Tensor) else v
                    for k, v in allowed.items()
                }
                result = ssd.ssd_chunk_scan_combined_ref(
                    x.double(),
                    dt.double(),
                    a.double(),
                    b.double(),
                    c.double(),
                    chunk_size,
                    **allowed,
                )
                return result[:, :length].to(x.dtype)

            def reference_norm(self, x, z=None):
                return rms_norm_ref(
                    x,
                    self.weight,
                    None,
                    z=z,
                    eps=self.eps,
                    group_size=self.group_size,
                    norm_before_gate=self.norm_before_gate,
                )

            stack.enter_context(
                patch.object(upstream, "mamba_chunk_scan_combined", reference_scan)
            )
            stack.enter_context(patch.object(RMSNorm, "forward", reference_norm))

        model = SketchStudent().to(args.device)
        sketches = torch.ones(2, 3, 224, 224, device=args.device)
        sketches[0, :, 30:180, 81:83] = 0
        # Include an entirely white image: pooling must remain finite.
        query = model(sketches)
        assert query.shape == (2, 512) and torch.isfinite(query).all()
        assert torch.allclose(
            query.norm(dim=-1), torch.ones(2, device=args.device), atol=1e-5
        )
        photos = F.normalize(torch.randn(4, 512, device=args.device), dim=-1)
        labels = torch.tensor([0, 1], device=args.device)
        photo_labels = torch.tensor([0, 1, 0, 2], device=args.device)
        loss = multi_positive_loss(query, photos, labels, photo_labels)
        logits = query @ photos.T / 0.07
        oracle = torch.stack(
            [
                torch.logsumexp(logits[i], dim=0)
                - logits[i, photo_labels == labels[i]].mean()
                for i in range(2)
            ]
        ).mean()
        assert torch.allclose(loss, oracle, atol=1e-6)
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, missing
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(block.gate.grad.abs() > 0 for block in model.blocks)
        assert all(
            block.global_context.mamba.A_log.grad.abs().sum() > 0
            for block in model.blocks
        )
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = SketchStudent().to(args.device)
        restored.load_state_dict(
            torch.load(buffer, map_location=args.device, weights_only=True)
        )
        with torch.no_grad():
            assert torch.equal(model(sketches), restored(sketches))
        print(
            {
                "status": "PASS",
                "scope": "CPU_FP64_SSD_reference_stable_chunk_carry"
                if args.device == "cpu"
                else "production_CUDA_kernels",
                "parameters": sum(p.numel() for p in model.parameters()),
                "loss": loss.item(),
                "training_updates": 0,
                "pretrained_or_real_data": False,
            }
        )


if __name__ == "__main__":
    main()
