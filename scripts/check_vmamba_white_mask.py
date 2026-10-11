"""Synthetic CUDA checks for adapted white-mask VMamba; never touches real images."""

import io

import torch
import torch.nn.functional as F

from spica.models.vmamba_student import (
    build_vmamba_tiny,
    build_vmamba_tiny_white_masked,
)
from spica.models.vmamba_white_mask import MaskedSS2D, MaskedVSSBlock, white_mask
from spica.vendor.vmamba import csms6s


def reference(u, dt, a, b, c, d, valid):
    valid = valid.repeat_interleave(u.shape[1] // valid.shape[1], dim=1)
    state = torch.zeros(
        (*u.shape[:2], a.shape[-1]), dtype=torch.float64, device=u.device
    )
    ys = []
    for i in range(u.shape[-1]):
        keep = valid[..., i, None]
        step = dt[..., i, None] * keep
        bi = b[..., i].repeat_interleave(u.shape[1] // b.shape[1], dim=1)
        ci = c[..., i].repeat_interleave(u.shape[1] // c.shape[1], dim=1)
        state = torch.exp(step * a) * state + step * u[..., i, None] * bi
        ys.append(((state * ci).sum(-1) + d * u[..., i]) * keep.squeeze(-1))
    return torch.stack(ys, -1)


def check_masked_scan():
    for length in (17, 3136):
        bsz, groups, channels, states = 2, 4, 2, 2
        valid = torch.rand(bsz, groups, length, device="cuda") > 0.35
        valid[0, :, length // 3 : length // 2] = False
        valid[1] = False
        valid[0, 1::2] = valid[0, 1::2].flip(-1)
        shapes = [(bsz, groups * channels, length)] * 2 + [
            (groups * channels, states),
            (bsz, groups, states, length),
            (bsz, groups, states, length),
            (groups * channels,),
        ]
        vals = [torch.randn(s, device="cuda") * 0.1 for s in shapes]
        u, delta, a, b, c, d = [v.detach().requires_grad_() for v in vals]
        expanded = (
            valid[:, :, None]
            .expand(bsz, groups, channels, length)
            .reshape(bsz, -1, length)
        )
        dt = F.softplus(delta.float()) * expanded
        u_masked = u.float() * expanded
        b_masked = b * valid[:, :, None]
        c_masked = c * valid[:, :, None]
        actual = csms6s.selective_scan_fn(
            u_masked,
            dt,
            -a.float().abs(),
            b_masked.contiguous(),
            c_masked.contiguous(),
            d.float(),
            None,
            False,
            backend="oflex",
        )
        expected = reference(
            u.double(),
            F.softplus(delta.double()),
            -a.double().abs(),
            b.double(),
            c.double(),
            d.double(),
            valid,
        )
        weight = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * weight).sum(), (u, delta, a, b, c, d))
        ge = torch.autograd.grad(
            (expected * weight.double()).sum(), (u, delta, a, b, c, d)
        )
        torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-4)
        for x, y in zip(ga, ge, strict=True):
            torch.testing.assert_close(x.double(), y.double(), atol=3e-5, rtol=3e-4)
        assert actual[1].count_nonzero() == 0
        assert ga[0][~expanded].count_nonzero() == 0
        assert ga[1][~expanded].count_nonzero() == 0
        assert (
            actual.view(bsz, groups, channels, length)[
                ~valid[:, :, None].expand(-1, -1, channels, -1)
            ].count_nonzero()
            == 0
        )
        assert ga[4][~valid[:, :, None].expand(-1, -1, states, -1)].count_nonzero() == 0


def main():
    assert csms6s.WITH_SELECTIVESCAN_OFLEX
    check_masked_scan()
    torch.manual_seed(21)
    native = build_vmamba_tiny().cuda().eval()
    torch.manual_seed(21)
    model = build_vmamba_tiny_white_masked().cuda().eval()
    assert (
        len([m for m in model.modules() if m.__class__.__name__ == "MaskedSS2D"]) == 14
    )
    assert (
        sum(p.numel() for p in model.parameters())
        == sum(p.numel() for p in native.parameters())
        == 29873792
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            native.state_dict().values(), model.state_dict().values(), strict=True
        )
    )
    images = torch.ones(2, 3, 224, 224, device="cuda")
    images[0, :, 48:152, 80:88] = torch.rand(3, 104, 8, device="cuda")
    # Sample 1 is all-white; its mask disables itself without changing batch shape.
    assert white_mask(images)[1].all()
    with torch.no_grad():
        # Per-sample all-white fallback equals native path; mixed batch keeps native grid and order.
        y = model(images)
        torch.testing.assert_close(y[1], native(images[1:])[0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            model(torch.ones_like(images)),
            native(torch.ones_like(images)),
            atol=0,
            rtol=0,
        )
        dense = torch.rand_like(images)
        torch.testing.assert_close(model(dense), native(dense), atol=0, rtol=0)
    expected = []
    occupancy = white_mask(images[:1])
    for _ in range(2):
        occupancy = F.max_pool2d(occupancy.float(), 3, 2, 1).bool()
    for depth in (2, 2, 8, 2):
        expected.extend([occupancy] * depth)
        occupancy = F.max_pool2d(occupancy.float(), 3, 2, 1).bool()
    seen_ops, seen_blocks = [], []

    def observe_op(module, inputs, output):
        if len(inputs) == 1:
            return
        valid = inputs[1]
        torch.testing.assert_close(valid, expected[len(seen_ops)])
        assert output.masked_select(~valid.expand_as(output)).count_nonzero() == 0
        inputs[0].retain_grad()
        seen_ops.append((inputs[0], valid))

    def observe_block(module, inputs, output):
        if len(inputs) == 1:
            return
        valid = inputs[1]
        assert output.masked_select(~valid.expand_as(output)).count_nonzero() == 0
        seen_blocks.append(valid)

    with torch.no_grad():
        for layer in model.backbone.layers:
            for block in layer.blocks:
                block.norm.bias.fill_(0.2)
                block.norm2.bias.fill_(-0.3)
                block.op.out_norm.bias.fill_(0.37)
                block.mlp.fc2.bias.fill_(0.21)
    hooks = []
    for module in model.modules():
        if isinstance(module, MaskedSS2D):
            hooks.append(module.register_forward_hook(observe_op))
        elif isinstance(module, MaskedVSSBlock):
            hooks.append(module.register_forward_hook(observe_block))
    model.eval()  # Deterministic synthetic gradients; real B32 trainer gate tests drop-path.
    x = images.detach().clone().requires_grad_()
    out = model(x)
    assert out.shape == (2, 512) and torch.isfinite(out).all()
    (out[:, :4] * torch.randn_like(out[:, :4])).sum().backward()
    assert x.grad is not None and x.grad[0].abs().sum() > 0
    assert len(seen_ops) == len(seen_blocks) == 14
    for feature, valid in seen_ops:
        assert (
            feature.grad.masked_select(~valid.expand_as(feature)).count_nonzero() == 0
        )
    for hook in hooks:
        hook.remove()
    for layer in model.backbone.layers:
        for block in layer.blocks:
            op = block.op
            assert (
                op.x_proj_weight.grad is not None
                and op.x_proj_weight.grad.abs().sum() > 0
            )
            assert (
                op.dt_projs_bias.grad is not None
                and op.dt_projs_bias.grad.abs().sum() > 0
            )
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    model.eval()
    restored = build_vmamba_tiny_white_masked().cuda().eval()
    restored.load_state_dict(
        torch.load(stream, map_location="cuda", weights_only=True), strict=True
    )
    torch.testing.assert_close(restored(x.detach()), model(x.detach()), atol=0, rtol=0)
    print(
        "PASS: synthetic native oflex masked recurrence/oracle, 14-block model, gradients, state restore; no official images or updates"
    )


if __name__ == "__main__":
    main()
