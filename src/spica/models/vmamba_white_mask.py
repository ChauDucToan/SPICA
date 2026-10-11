"""Project-owned white-occupancy propagation and native-oflex SS2D masking."""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..vendor.vmamba import csms6s
from ..vendor.vmamba.csm_triton import cross_merge_fn, cross_scan_fn
from ..vendor.vmamba.vmamba import SS2D, VSSM, VSSBlock


def masked(x: Tensor, valid: Tensor) -> Tensor:
    return x * valid.to(dtype=x.dtype)


def propagate(valid: Tensor, kernel: int, stride: int, padding: int = 0) -> Tensor:
    return (
        F.conv2d(
            valid.float(),
            torch.ones((1, 1, kernel, kernel), device=valid.device),
            stride=stride,
            padding=padding,
        )
        > 0
    )


def white_mask(images: Tensor) -> Tensor:
    valid = (images != 1).any(dim=1, keepdim=True)
    return torch.where(
        valid.flatten(1).any(1)[:, None, None, None], valid, torch.ones_like(valid)
    )


class MaskedSS2D(SS2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward = self.forward_masked

    def forward_masked(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        if valid is None or valid.all():
            return super().forwardv2(x)
        x = masked(self.in_proj(x), valid)
        x = masked(self.act(self.conv2d(x)), valid)
        y = self.forward_masked_core(x, valid)
        return masked(self.out_proj(y), valid)

    def forward_masked_core(self, x: Tensor, valid: Tensor) -> Tensor:
        b, d, h, w = x.shape
        k, n, length = 4, self.d_state, h * w
        validity = (
            cross_scan_fn(valid.float(), in_channel_first=True, out_channel_first=True)
            .view(b, k, length)
            .bool()
        )
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True).view(
            b, k, d, length
        )
        projected = F.conv1d(
            xs.reshape(b, k * d, length), self.x_proj_weight.reshape(-1, d, 1), groups=k
        )
        dt, bs, cs = projected.view(b, k, -1, length).split((self.dt_rank, n, n), dim=2)
        dt = F.conv1d(
            dt.contiguous().view(b, -1, length),
            self.dt_projs_weight.reshape(-1, self.dt_rank, 1),
            groups=k,
        )
        u = xs.reshape(b, k * d, length).float()
        dt = F.softplus(dt.float() + self.dt_projs_bias.float().view(1, -1, 1))
        valid_channels = (
            validity[:, :, None, :].expand(b, k, d, length).reshape(b, k * d, length)
        )
        dt = dt * valid_channels
        u = u * valid_channels
        bs = bs.float() * validity[:, :, None, :]
        cs = cs.float() * validity[:, :, None, :]
        y = csms6s.selective_scan_fn(
            u,
            dt,
            -self.A_logs.float().exp(),
            bs.contiguous(),
            cs.contiguous(),
            self.Ds.float(),
            None,
            False,
            backend="oflex",
        ).view(b, k, d, h, w)
        y = cross_merge_fn(y, in_channel_first=True, out_channel_first=True).view(
            b, d, h, w
        )
        y = self.out_norm(y)
        return masked(y, valid).to(x.dtype)


class MaskedVSSBlock(VSSBlock):
    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        if valid is None:
            return super().forward(x)
        x = masked(x, valid)
        x = masked(x + self.drop_path(self.op(self.norm(x), valid)), valid)
        return masked(x + self.drop_path(self.mlp(self.norm2(x))), valid)


class MaskedVSSM(VSSM):
    """Same upstream builder/parameter tree; adapts forward only."""

    @staticmethod
    def _make_layer(
        dim,
        drop_path,
        use_checkpoint=False,
        norm_layer=None,
        downsample=None,
        channel_first=False,
        _SS2D=MaskedSS2D,
        **kwargs,
    ):
        from collections import OrderedDict

        blocks = [
            MaskedVSSBlock(
                hidden_dim=dim,
                drop_path=p,
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                channel_first=channel_first,
                _SS2D=_SS2D,
                **kwargs,
            )
            for p in drop_path
        ]
        return torch.nn.Sequential(
            OrderedDict(blocks=torch.nn.Sequential(*blocks), downsample=downsample)
        )

    def forward(self, images: Tensor) -> Tensor:
        blank = images.eq(1).flatten(1).all(1)
        if blank.any() and not blank.all():
            # True native fallback: mask=1 alone still uses adapted softplus/scan math.
            white = super().forward(images[blank])
            result = white.new_empty((len(images), white.shape[-1]))
            result[blank] = white
            result[~blank] = self.forward(images[~blank])
            return result
        valid = white_mask(images)
        if valid.all():
            return super().forward(images)
        stem = self.patch_embed
        valid = propagate(valid, 3, 2, 1)
        x = masked(stem[0](images), valid)
        x = masked(stem[2](stem[1](x)), valid)
        x = masked(stem[4](x), valid)
        valid = propagate(valid, 3, 2, 1)
        x = masked(stem[5](x), valid)
        x = masked(stem[7](stem[6](x)), valid)
        for i, layer in enumerate(self.layers):
            for block in layer.blocks:
                x = block(x, valid)
            if i < len(self.layers) - 1:
                downsample = layer.downsample
                valid = propagate(valid, 3, 2, 1)
                x = masked(downsample[1](downsample[0](x)), valid)
                x = masked(downsample[3](downsample[2](x)), valid)
        x = masked(self.classifier.norm(x), valid)
        pooled = x.sum((2, 3)) / valid.sum((2, 3)).clamp_min(1)
        return self.classifier.head(pooled)
