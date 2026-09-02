from pathlib import Path

from scripts.transport_artifact_utils import (
    best_point,
    explicit_transport_enabled,
    matched_transport_predicate,
    select_best_run,
)


def _run(name: str, *, transport: bool | None, value: float, k: int = 1, endpoint: float = 0.0):
    return (
        Path("outputs/experiments") / name / "run_result.json",
        {
            "config": {
                "model_family": "predictive_semantic_transport",
                "transport_enabled": transport,
                "transport_mode": "tangent",
                "K": k,
                "use_vmf": False,
                "lambda_endpoint": endpoint,
            },
            "probe_history": [{"step": 73, "val": {"mAP": value}}],
        },
    )


def test_missing_transport_flag_is_unknown_not_transport() -> None:
    _, result = _run("missing", transport=None, value=0.99)
    assert explicit_transport_enabled(result) is None
    assert not matched_transport_predicate(k=1, endpoint=0.0, use_vmf=False)(result)


def test_k1_selection_cannot_select_base_only_control() -> None:
    records = [
        _run("base", transport=False, value=0.99),
        _run("transport", transport=True, value=0.50),
    ]
    selected = select_best_run(
        records,
        matched_transport_predicate(k=1, endpoint=0.0, use_vmf=False),
    )
    assert selected is not None
    assert selected[0].parts[-2] == "transport"
    assert best_point(selected[1])["val"]["mAP"] == 0.50


def test_endpoint0_best_eligible_run_is_not_historical_endpoint1() -> None:
    records = [
        _run("endpoint1", transport=True, value=0.90, endpoint=1.0),
        _run("endpoint0", transport=True, value=0.64, endpoint=0.0),
    ]
    selected = select_best_run(
        records,
        matched_transport_predicate(k=1, endpoint=0.0, use_vmf=False),
    )
    assert selected is not None
    assert selected[0].parts[-2] == "endpoint0"
