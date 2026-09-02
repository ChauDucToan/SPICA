from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import wandb

WandbMode = Literal["online", "offline", "disabled"]
Scalar = int | float


class WandbExperiment:
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
    ) -> None:
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
        self._run.log(dict(metrics), step=step)

    def log_table(
        self,
        name: str,
        *,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        step: int | None = None,
    ) -> None:
        self._ensure_active()

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
