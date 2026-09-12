"""Periodic official test evaluation for paused coupled-benchmark training."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .coupled_benchmark import (
    _preserve_runtime,
    evaluate_coupled_benchmark,
)
from ..data import coupled_benchmark as benchmark_data
from ..data.coupled_benchmark import _plain
from .masked_view import _masked_loader, _transform_stats


def evaluate_periodic_test(
    model: Any,
    protocol: Any,
    transform: Any,
    output_dir: Path,
    *,
    device: Any,
    source_hash: str,
    checkpoint_selection: Mapping[str, Any],
    final: bool = False,
) -> dict[str, Any]:
    """Evaluate the full official test gallery and clean plus nine masked views.

    This function only consumes the supplied model.  It does not load a
    checkpoint, alter training state, select a checkpoint, or contact W&B.
    Non-final calls retain metrics and provenance only; the final call retains
    the detailed embedding, per-query, and mask metadata artifacts.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    with _preserve_runtime(model):
        loaders = benchmark_data.build_test_loaders(
            protocol, transform, batch_size=256, num_workers=0
        )
        mean, std = _transform_stats(transform)

        def masked_loader(fraction: float, seed: int) -> Any:
            return _masked_loader(
                protocol.test.sketch_entries,
                transform,
                root=protocol.root,
                fraction=fraction,
                seed=seed,
                batch_size=256,
                num_workers=0,
                mean=mean,
                std=std,
                ink_threshold=0.9,
            )

        return evaluate_coupled_benchmark(
            model,
            loaders["sketch"],
            loaders["photo"],
            masked_loader,
            output_dir,
            device=device,
            benchmark=str(protocol.name),
            protocol_identity=_plain(protocol.identity),
            class_names={int(k): str(v) for k, v in protocol.train.class_names.items()},
            source_hash=source_hash,
            checkpoint_selection=dict(checkpoint_selection),
            data_identity=_plain(protocol.identity),
            status="COMPLETE",
            save_details=final,
            reset_peak_memory=False,
            evaluation_scope=(
                "official_unseen_final"
                if final
                else "official_unseen_periodic_monitoring_no_selection"
            ),
        )


__all__ = ["evaluate_periodic_test"]
