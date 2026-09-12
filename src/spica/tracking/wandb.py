from collections.abc import Mapping, Sequence
import math
from numbers import Real
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import wandb

WandbMode = Literal["online", "offline", "disabled"]
Scalar = int | float
_RETRIEVAL_METRICS = frozenset({
    "full_mAP", "P@200", "mAP@200_prefix_positive",
    "mAP@200_all_relevant", "mAP@200_min_relevant_k",
})
_LOGGED_RETRIEVAL_METRICS = {
    "full_mAP": "mAP@all",
    "mAP@200_min_relevant_k": "mAP@200",
    "P@200": "P@200",
}
_ALLOWED_LOG_METRICS = frozenset({
    "step_train",
    "cleaned/mAP@200", "cleaned/mAP@all", "cleaned/P@200",
    "masked/mAP@200", "masked/mAP@all", "masked/P@200",
    "test/cleaned/mAP@200", "test/cleaned/mAP@all", "test/cleaned/P@200",
    "test/masked/mAP@200", "test/masked/mAP@all", "test/masked/P@200",
})
_ALLOWED_METRIC_SCOPES = frozenset({
    "step_train", "cleaned/*", "masked/*", "test/cleaned/*", "test/masked/*",
})


class WandbExperiment:
    """Small W&B adapter; it logs supplied metrics but never schedules evaluation."""

    def __init__(
        self,
        *,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: Sequence[str] = (),
        mode: WandbMode = "disabled",
        job_type: str | None = None,
        directory: Path | None = None,
        allow_artifacts: bool = False,
    ) -> None:
        self._allow_artifacts = allow_artifacts
        settings = wandb.Settings(
            x_disable_meta=True,
            x_disable_machine_info=True,
            x_disable_stats=True,
            disable_code=True,
            disable_git=True,
            x_save_requirements=False,
            console="off",
        )
        self._run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            group=group,
            config=dict(config) if config is not None else None,
            tags=tuple(tags),
            mode=mode,
            job_type=job_type,
            dir=str(directory) if directory is not None else None,
            reinit="create_new",
            settings=settings,
        )
        self._finished = False

    @property
    def run_id(self) -> str:
        return self._run.id

    @property
    def run_url(self) -> str | None:
        return self._run.url

    def log_metrics(
        self,
        metrics: Mapping[str, Scalar],
        *,
        step: int | None = None,
    ) -> None:
        self._ensure_active()
        unknown = {
            name for name in metrics
            if name.startswith(("cleaned/", "masked/", "test/"))
        } - _ALLOWED_LOG_METRICS
        if unknown:
            raise ValueError(f"unknown minimal retrieval metrics: {sorted(unknown)}")
        logged = {
            name: value for name, value in metrics.items()
            if name in _ALLOWED_LOG_METRICS
        }
        for name, value in logged.items():
            self._validate_scalar(name, value)
        if logged.keys() - {"step_train"}:
            self._run.log(logged, step=step)

    def define_metric(
        self,
        name: str,
        *,
        step_metric: str | None = None,
        summary: object = None,
    ) -> None:
        self._ensure_active()
        if name not in _ALLOWED_METRIC_SCOPES:
            return
        self._run.define_metric(name, step_metric=step_metric, summary=summary)

    def set_summary(self, values: Mapping[str, Any]) -> None:
        self._ensure_active()
        self._run.summary.update(dict(values))

    def log_test_retrieval(self, step: int, report: Mapping[str, Any]) -> None:
        self._ensure_active()
        self._validate_scalar("step_train", step)
        if not {"clean", "masked_macro"} <= set(report):
            raise ValueError("periodic test report must contain clean and masked_macro")
        logged = {"step_train": step}
        for scope, source in (("test/cleaned", "clean"), ("test/masked", "masked_macro")):
            metrics = report[source]
            if not isinstance(metrics, Mapping):
                raise TypeError(f"{source} metrics must be a mapping")
            allowed = _RETRIEVAL_METRICS
            unknown = set(metrics) - allowed
            if unknown or set(metrics) != allowed:
                raise ValueError(f"periodic {source} metrics must be the standard five metrics")
            logged.update(self._prefixed_metrics(scope, metrics))
        self.log_metrics(logged, step=step)

    def log_retrieval_probe(
        self,
        step: int,
        clean: Mapping[str, Any],
        masked_macro: Mapping[str, Any],
        masked_by_fraction: Mapping[object, Mapping[str, Any]],
        conditions: Sequence[Mapping[str, Any]] = (),
        *,
        diagnostics: Mapping[str, Scalar] | None = None,
    ) -> None:
        self._ensure_active()
        self._validate_scalar("step_train", step)
        logged: dict[str, Scalar] = {"step_train": step}
        logged.update(self._prefixed_metrics("cleaned", clean))
        logged.update(self._prefixed_metrics("masked", masked_macro))
        # Validate legacy inputs for callers, but do not create fraction, seed,
        # or diagnostic curves under the shared tracking policy.
        for fraction, metrics in masked_by_fraction.items():
            self._fraction_label(fraction)
            self._prefixed_metrics("masked", metrics)
        for condition in conditions:
            if "fraction" in condition:
                self._fraction_label(condition["fraction"])
            self._prefixed_metrics("masked", {
                key: value for key, value in condition.items()
                if key not in {"fraction", "seed"}
            })
        for name, value in (diagnostics or {}).items():
            if not name.startswith(("text/", "train/")):
                raise ValueError("probe diagnostics must use text/ or train/ namespaces")
            self._validate_scalar(name, value)
        self._run.log(logged, step=step)

    @staticmethod
    def _prefixed_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Scalar]:
        unknown = set(metrics) - _RETRIEVAL_METRICS
        if unknown:
            raise ValueError(f"unknown retrieval metric(s): {sorted(unknown)}")
        result: dict[str, Scalar] = {}
        for name, value in metrics.items():
            WandbExperiment._validate_scalar(f"{prefix}/{name}", value)
            if name in _LOGGED_RETRIEVAL_METRICS:
                result[f"{prefix}/{_LOGGED_RETRIEVAL_METRICS[name]}"] = value
        return result

    @staticmethod
    def _validate_scalar(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name!r} must be a finite scalar")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name!r} must be finite")

    @staticmethod
    def _fraction_label(fraction: object) -> str:
        if isinstance(fraction, bool) or not isinstance(fraction, Real):
            raise TypeError("retrieval fraction must be a finite scalar")
        if not math.isfinite(float(fraction)):
            raise ValueError("retrieval fraction must be finite")
        return f"{float(fraction):.2f}".replace(".", "")

    def log_table(
        self,
        name: str,
        *,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        step: int | None = None,
    ) -> None:
        self._ensure_active()
        if not self._allow_artifacts:
            return

        column_list = list(columns)
        row_list = [list(row) for row in rows]
        invalid_rows = [
            index for index, row in enumerate(row_list) if len(row) != len(column_list)
        ]
        if invalid_rows:
            raise ValueError(
                "Every table row must have the same number of values as columns; "
                f"invalid row indices: {invalid_rows}"
            )

        table = wandb.Table(columns=column_list, data=row_list)
        self._run.log({name: table}, step=step)

    def log_artifact(
        self,
        path: Path,
        *,
        name: str,
        artifact_type: str,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        aliases: Sequence[str] = ("latest",),
    ) -> None:
        self._ensure_active()
        if not self._allow_artifacts:
            return
        if not path.exists():
            raise FileNotFoundError(f"Artifact path not found: {path}")

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            description=description,
            metadata=dict(metadata) if metadata is not None else None,
        )
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))

        self._run.log_artifact(artifact, aliases=list(aliases))

    def finish(self, exit_code: int = 0) -> None:
        if self._finished:
            return
        self._run.finish(exit_code=exit_code)
        self._finished = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish(exit_code=0 if exception_type is None else 1)

    def _ensure_active(self) -> None:
        if self._finished:
            raise RuntimeError("Cannot log to a finished W&B run")


def publish_final_retrieval(result: Mapping, report: Mapping, *, p_all: float) -> dict:
    """Publish final scalars to this run's summary, never into probe curves."""
    run = wandb.Api().run(f"a-cctest05187-erd/spica/{result['wandb_run_id']}")
    expected = {"source_snapshot_hash": result["source_snapshot_hash"],
                "arm": result["arm"], "dataset": result["dataset"], "total_steps": result["step"]}
    if any(run.config.get(key) != value for key, value in expected.items()):
        raise ValueError("W&B final summary source/run identity mismatch")
    WandbExperiment._validate_scalar("P@all", p_all)
    values = {"final/step": result["step"], "final/scope": "official_test_final_only",
              "final/checkpoint_sha256": result["selections"]["latest"]["sha256"]}
    for scope, source in [("cleaned", "clean"), ("masked", "masked_macro")]:
        values.update(WandbExperiment._prefixed_metrics(f"final/{scope}", report[source]))
        values[f"final/{scope}/P@all"] = p_all
    run.summary.update(values)
    return {"status": "PASS", "run_id": result["wandb_run_id"], "summary": values}
