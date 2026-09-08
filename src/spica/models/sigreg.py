"""Minimal, pinned SIGReg implementation.

This is an independent PyTorch implementation of the SIGReg code shown in
LeJEPA's pinned ``MINIMAL.md``.  It intentionally implements only the
random-sliced Epps--Pulley statistic: no batch standardization, distributed
reduction, clipping, or learned parameters.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (minimal LeJEPA form).

    ``values`` is ``[B, D]`` or ``[V, B, D]``.  For multiple views, the
    statistic is evaluated independently over each ``[B, D]`` view and then
    averaged over views and random projections.  Inputs are *not* centered,
    standardized, normalized, or otherwise preprocessed.

    The default values match the pinned official minimal example: 17
    frequencies on ``[0, 3]`` and 256 random unit projections.  Random
    directions come from a private CPU generator whose state is included in
    ``state_dict``; the process-global PyTorch RNG is not consumed.

    Args:
        knots: Number of linearly spaced non-negative frequency nodes.
        num_projections: Number of Gaussian random projection directions.
        t_max: Largest frequency node.
        seed: Initial seed for this module's private projection generator.
    """

    def __init__(
        self,
        knots: int = 17,
        num_projections: int = 256,
        t_max: float = 3.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be at least 2")
        if num_projections < 1:
            raise ValueError("num_projections must be positive")
        if not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("t_max must be finite and positive")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")

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
        """Persist the private CPU generator without moving it with the module."""
        return {"generator_state": self._generator_state.clone()}

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict) or "generator_state" not in state:
            raise RuntimeError("SIGReg state is missing generator_state")
        generator_state = state["generator_state"]
        if not isinstance(generator_state, Tensor) or generator_state.dtype != torch.uint8:
            raise RuntimeError("SIGReg generator_state must be a uint8 tensor")
        self._generator_state = generator_state.detach().cpu().clone()

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
            # A CPU generator is useful for replay on every device; otherwise
            # PyTorch requires the generator device to match the random tensor.
            device = getattr(generator, "device", torch.device("cpu"))
            directions = torch.randn(
                (dimension, self.num_projections),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        norms = directions.norm(p=2, dim=0)
        if not torch.isfinite(norms).all().item() or (norms == 0).any().item():
            raise RuntimeError("projection sampling produced an invalid direction")
        return directions / norms

    def forward(
        self,
        values: Tensor,
        *,
        directions: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Return the averaged Epps--Pulley statistic.

        ``directions`` can be supplied as ``[D, K]`` for deterministic parity
        checks or replay.  Supplied directions are normalized exactly as the
        official random projections are.  ``generator`` is an optional
        caller-owned generator and is mutually exclusive with ``directions``.
        """
        if values.ndim not in (2, 3):
            raise ValueError("values must have shape [B, D] or [V, B, D]")
        if any(size < 1 for size in values.shape):
            raise ValueError("values must have positive view, batch and feature dimensions")
        if any(buffer.dtype != torch.float32 for buffer in (self.t, self.phi, self.weights)):
            raise ValueError("SIGReg frequency buffers must remain float32; do not cast the module")
        if not values.is_floating_point():
            raise TypeError("values must be floating-point")
        if not torch.isfinite(values).all().item():
            raise ValueError("values must contain only finite values")
        if directions is not None and generator is not None:
            raise ValueError("directions and generator are mutually exclusive")

        batch, dimension = values.shape[-2:]
        if directions is None:
            projection = self._sample_directions(dimension, generator)
        else:
            if directions.ndim != 2 or directions.shape != (dimension, self.num_projections):
                raise ValueError(
                    "directions must have shape "
                    f"[{dimension}, {self.num_projections}]"
                )
            if not directions.is_floating_point():
                raise TypeError("directions must be floating-point")
            if not torch.isfinite(directions).all().item():
                raise ValueError("directions must contain only finite values")
            norms = directions.float().norm(p=2, dim=0)
            if not torch.isfinite(norms).all().item() or (norms == 0).any().item():
                raise ValueError("directions must have finite nonzero float32 norms")
            projection = directions.float() / norms

        # The official statistic is float32.  Disabling autocast here also
        # keeps CPU/GPU and mixed-precision calls on the same numerical path.
        with torch.autocast(device_type=values.device.type, enabled=False):
            projected = values.float() @ projection.to(device=values.device)
            x_t = projected.unsqueeze(-1) * self.t.to(device=values.device)
            # For [B, D], -3 is the sample axis; for [V, B, D], it remains B.
            cos_mean = torch.cos(x_t).mean(dim=-3)
            sin_mean = torch.sin(x_t).mean(dim=-3)
            phi = self.phi.to(device=values.device)
            weights = self.weights.to(device=values.device)
            error = (cos_mean - phi).square() + sin_mean.square()
            statistic = (error @ weights) * batch
            return statistic.mean()


__all__ = ["SIGReg"]
