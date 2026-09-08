"""Pinned minimal LeJEPA SIGReg."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """Random-sliced Epps--Pulley statistic from the pinned MINIMAL formula."""

    def __init__(
        self,
        knots: int = 17,
        num_projections: int = 256,
        t_max: float = 3.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        quadrature = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        quadrature[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.num_projections = num_projections
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", quadrature * phi)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self._generator_state = generator.get_state()

    def get_extra_state(self) -> dict[str, Tensor]:
        return {"generator_state": self._generator_state.clone()}

    def set_extra_state(self, state: object) -> None:
        self._generator_state = state["generator_state"].detach().cpu().clone()

    def _sample_directions(self, dimension: int, generator: Optional[torch.Generator]) -> Tensor:
        if generator is None:
            private = torch.Generator(device="cpu")
            private.set_state(self._generator_state)
            directions = torch.randn(
                (dimension, self.num_projections),
                dtype=torch.float32,
                device="cpu",
                generator=private,
            )
            self._generator_state = private.get_state()
        else:
            device = getattr(generator, "device", torch.device("cpu"))
            directions = torch.randn(
                (dimension, self.num_projections),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        return directions / directions.norm(p=2, dim=0)

    def forward(
        self,
        values: Tensor,
        *,
        directions: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        batch, dimension = values.shape[-2:]
        if directions is None:
            projection = self._sample_directions(dimension, generator)
        else:
            projection = directions.float()
            projection = projection / projection.norm(p=2, dim=0)

        # The pinned statistic stays FP32 even when the caller uses autocast.
        with torch.autocast(device_type=values.device.type, enabled=False):
            projected = values.float() @ projection.to(device=values.device)
            x_t = projected.unsqueeze(-1) * self.t.to(device=values.device)
            cos_mean = torch.cos(x_t).mean(dim=-3)
            sin_mean = torch.sin(x_t).mean(dim=-3)
            phi = self.phi.to(device=values.device)
            weights = self.weights.to(device=values.device)
            error = (cos_mean - phi).square() + sin_mean.square()
            return ((error @ weights) * batch).mean()


__all__ = ["SIGReg"]
