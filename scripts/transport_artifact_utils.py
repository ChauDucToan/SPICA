"""Shared, conservative selection helpers for transport research artifacts.

Artifacts written before provenance/causal controls existed are intentionally
not upgraded by inference.  In particular, a missing ``transport_enabled``
field is *unknown*, not ``True``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]


class ArtifactIntegrityError(ValueError):
    """Raised when artifacts cannot support the requested causal contrast."""


MATCHED_FIELDS = (
    "seed",
    "train_class_scope",
    "pseudo_validation_seed",
    "model_name",
    "pretrained",
    "encoder_mode",
    "encoder_unfreeze_depth",
    "use_z0",
    "lambda_cls",
    "text_loss_location",
    "tau_cls",
    "prompt_template",
    "lambda_rank",
    "margin",
    "predictor_learning_rate",
    "encoder_learning_rate",
    "weight_decay",
    "scheduler",
    "steps",
    "batch_size",
    "num_positive_photos",
    "probe_steps",
    "inference_score_mode",
    "score_temperature",
    "map_at_k_denominator",
)

FREEZE_BRANCHES = {
    "freeze_optimizer_A": (False, False),
    "freeze_optimizer_B": (True, False),
    "freeze_optimizer_C": (False, True),
    "freeze_optimizer_D": (True, True),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def config_of(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("config", {})
    return value if isinstance(value, dict) else {}


def explicit_transport_enabled(result: dict[str, Any]) -> bool | None:
    """Return the recorded transport flag, preserving unknown historical data."""
    config = config_of(result)
    if isinstance(config.get("transport_enabled"), bool):
        return bool(config["transport_enabled"])
    metadata = result.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("transport_enabled"), bool):
        return bool(metadata["transport_enabled"])
    checkpoint_metadata = result.get("checkpoint_metadata", {})
    if isinstance(checkpoint_metadata, dict) and isinstance(checkpoint_metadata.get("transport_enabled"), bool):
        return bool(checkpoint_metadata["transport_enabled"])
    return None


def is_transport_run(result: dict[str, Any], *, require_explicit: bool = True) -> bool:
    value = explicit_transport_enabled(result)
    return value is True if require_explicit else value is not False


def is_base_run(result: dict[str, Any]) -> bool:
    return explicit_transport_enabled(result) is False


def points(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("probe_history", [])
    if not isinstance(raw, list):
        return []
    return [point for point in raw if isinstance(point, dict) and "step" in point]


def point_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    answer: dict[int, dict[str, Any]] = {}
    for point in points(result):
        try:
            answer[int(point["step"])] = point
        except (TypeError, ValueError):
            continue
    return answer


def number_at(point: dict[str, Any] | None, *path: str) -> float | None:
    current: Any = point
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def best_point(result: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [point for point in points(result) if number_at(point, "val", "mAP") is not None]
    return max(candidates, key=lambda point: number_at(point, "val", "mAP") or float("-inf"), default=None)


def run_dir_name(path: Path) -> str:
    return path.parts[-2] if len(path.parts) >= 2 else path.name


def collect_runs(run_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    answer: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(run_root.glob("**/run_result.json")):
        try:
            result = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if config_of(result).get("model_family") == "predictive_semantic_transport":
            answer.append((path.parent, result))
    return answer


def select_best_run(
    records: Iterable[tuple[Path, dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
) -> tuple[Path, dict[str, Any]] | None:
    candidates = [record for record in records if predicate(record[1]) and best_point(record[1]) is not None]

    def key(record: tuple[Path, dict[str, Any]]) -> tuple[float, int, int]:
        result = record[1]
        steps = [int(point["step"]) for point in points(result)]
        return (
            number_at(best_point(result), "val", "mAP") or float("-inf"),
            len(steps),
            max(steps, default=-1),
        )

    return max(candidates, key=key, default=None)


def source_run_provenance(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Read provenance without attributing the current report commit to a run."""
    for candidate in (
        result.get("provenance"),
        result.get("metadata", {}).get("provenance") if isinstance(result.get("metadata"), dict) else None,
        result.get("checkpoint_metadata", {}).get("provenance") if isinstance(result.get("checkpoint_metadata"), dict) else None,
    ):
        if isinstance(candidate, dict):
            return {
                "commit": candidate.get("commit"),
                "working_tree_state": candidate.get("working_tree_state"),
                "dirty_files": candidate.get("dirty_files", []),
                "source": "artifact",
            }
    return {
        "commit": None,
        "working_tree_state": "unavailable (artifact predates provenance instrumentation)",
        "dirty_files": [],
        "source": "missing",
        "run": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
    }


def repository_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"current_commit": None, "working_tree_state": "unavailable", "dirty_files": [], "error": str(error)}
    return {
        "current_commit": commit,
        "working_tree_state": "clean" if not status else "dirty",
        "dirty_files": status,
    }


def matched_transport_predicate(
    *,
    text: bool | None = None,
    k: int | None = None,
    endpoint: float | None = None,
    use_vmf: bool | None = None,
    rho_strategy: str | None = None,
    direction_target: str | None = None,
    num_positive_photos: int | None = None,
) -> Callable[[dict[str, Any]], bool]:
    """Build a strict pseudo-unseen transport selector.

    Missing flags do not pass.  This prevents the historical base-only run
    from becoming the K=1 transport control merely because it has K=1.
    """
    def predicate(result: dict[str, Any]) -> bool:
        config = config_of(result)
        if not is_transport_run(result, require_explicit=True):
            return False
        if config.get("transport_mode") != "tangent":
            return False
        if k is not None and config.get("K") != k:
            return False
        if text is not None and bool(config.get("use_text_cls")) is not text:
            return False
        if endpoint is not None:
            try:
                if float(config.get("lambda_endpoint")) != endpoint:
                    return False
            except (TypeError, ValueError):
                return False
        if use_vmf is not None and bool(config.get("use_vmf")) is not use_vmf:
            return False
        if rho_strategy is not None and config.get("rho_strategy") != rho_strategy:
            return False
        if direction_target is not None and config.get("direction_target") != direction_target:
            return False
        if num_positive_photos is not None and config.get("num_positive_photos") != num_positive_photos:
            return False
        return True
    return predicate


def matched_base_predicate(*, text: bool) -> Callable[[dict[str, Any]], bool]:
    def predicate(result: dict[str, Any]) -> bool:
        config = config_of(result)
        return (
            is_base_run(result)
            and config.get("transport_mode") == "tangent"
            and config.get("K") == 1
            and bool(config.get("use_text_cls")) is text
            and float(config.get("lambda_endpoint", 0.0)) == 0.0
            and config.get("num_positive_photos", 1) == 1
        )
    return predicate


def select_unique_role(
    records: Iterable[tuple[Path, dict[str, Any]]], role: str
) -> tuple[Path, dict[str, Any]]:
    """Select exactly one run by structured role metadata, never by its name."""
    candidates = [
        record
        for record in records
        if config_of(record[1]).get("experiment_role") == role
    ]
    if not candidates:
        raise ArtifactIntegrityError(f"missing raw run with experiment_role={role!r}")
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path, _ in candidates)
        raise ArtifactIntegrityError(
            f"ambiguous experiment_role={role!r}: {len(candidates)} raw runs ({paths})"
        )
    return candidates[0]


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def assert_matched_runs(base: dict[str, Any], transport: dict[str, Any]) -> None:
    """Validate the matched base+text versus transport+text causal control."""
    base_config = config_of(base)
    transport_config = config_of(transport)
    if base_config.get("transport_enabled") is not False:
        raise ArtifactIntegrityError("z0_B must have transport_enabled=false")
    if transport_config.get("transport_enabled") is not True:
        raise ArtifactIntegrityError("z0_T/q_T must have transport_enabled=true")
    for label, config in (("base", base_config), ("transport", transport_config)):
        if not config.get("use_text_cls") or float(config.get("lambda_cls", 0.0)) <= 0:
            raise ArtifactIntegrityError(f"{label} run is not a text-supervised run")
        if float(config.get("lambda_endpoint", 1.0)) != 0.0:
            raise ArtifactIntegrityError(f"{label} run does not use endpoint=0")
    mismatches = []
    for field in MATCHED_FIELDS:
        if field not in base_config or field not in transport_config:
            mismatches.append(f"{field}: missing")
        elif not _same(base_config[field], transport_config[field]):
            mismatches.append(
                f"{field}: {base_config[field]!r} != {transport_config[field]!r}"
            )
    base_split = base.get("data_split_identity") or base.get("pseudo_split")
    transport_split = transport.get("data_split_identity") or transport.get(
        "pseudo_split"
    )
    if not isinstance(base_split, dict) or not isinstance(transport_split, dict):
        mismatches.append("data split identity: missing")
    elif base_split.get("sha256") and transport_split.get("sha256"):
        if base_split["sha256"] != transport_split["sha256"]:
            mismatches.append("data split identity hash differs")
    elif base_split != transport_split:
        mismatches.append("data split identity differs")
    if mismatches:
        raise ArtifactIntegrityError(
            "causal runs are not matched: " + "; ".join(mismatches)
        )


def factorial_effects(cells: dict[str, float]) -> dict[str, float]:
    missing = {"A", "B", "C", "D"} - cells.keys()
    if missing:
        raise ArtifactIntegrityError(f"factorial cells missing: {sorted(missing)}")
    if not all(math.isfinite(float(cells[name])) for name in ("A", "B", "C", "D")):
        raise ArtifactIntegrityError("factorial values must be finite")
    a, b, c, d = (float(cells[name]) for name in ("A", "B", "C", "D"))
    return {
        "text_effect_without_transport": b - a,
        "text_effect_with_transport": d - c,
        "transport_effect_without_text": c - a,
        "transport_effect_with_text": d - b,
        "interaction": d - c - b + a,
    }


def validate_freeze_optimizer_role(config: dict[str, Any]) -> dict[str, bool]:
    """Resolve and validate one branch of the step-73 freeze × reset factorial."""
    role = config.get("experiment_role")
    if role == "optimizer_reset_only":
        if config.get("freeze_encoder_at_step") is not None:
            raise ArtifactIntegrityError(
                "optimizer_reset_only must keep the encoder trainable; observed freeze_encoder_at_step"
            )
        role = "freeze_optimizer_C"
    if role not in FREEZE_BRANCHES:
        raise ArtifactIntegrityError(f"unknown freeze/optimizer branch role: {role!r}")
    expected_freeze, expected_reset = FREEZE_BRANCHES[role]
    observed_freeze = config.get("freeze_encoder_at_step") == 73
    observed_reset = bool(config.get("reset_optimizer_on_resume"))
    if observed_freeze != expected_freeze or observed_reset != expected_reset:
        raise ArtifactIntegrityError(
            f"{role} requires freeze={expected_freeze}, reset={expected_reset}; "
            f"observed freeze={observed_freeze}, reset={observed_reset}"
        )
    if not config.get("resume_checkpoint_path"):
        raise ArtifactIntegrityError(
            f"{role} requires the common step-73 resume checkpoint"
        )
    return {"freeze": expected_freeze, "reset": expected_reset}


def missing_result(reason: str) -> dict[str, str]:
    return {"status": "not_run", "reason": reason}


def write_new(path: Path, content: str | bytes) -> None:
    """Create a generated artifact without overwriting historical evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if isinstance(content, bytes) else "x"
    with path.open(mode) as handle:
        handle.write(content)
