from __future__ import annotations

import pytest

from spica.evaluation.masked_view import _selected_record


def _result() -> dict[str, object]:
    history = [
        {"training_global_step": step, "checkpoint": f"step{step}.pt", "checkpoint_sha256": f"sha{step}"}
        for step in (0, 600, 1200, 1800, 2400, 3000, 3600)
    ]
    return {
        "history": history,
        "selection": None,
        "selections": {
            name: {
                "training_global_step": step,
                "checkpoint": f"step{step}.pt",
                "checkpoint_sha256": f"sha{step}",
            }
            for name, step in (("latest", 3600), ("best_clean", 1800), ("best_masked", 2400))
        },
    }


def test_3600_named_selection_resolves_exact_history_lineage() -> None:
    result = _result()
    row, checkpoint = _selected_record(result, 1800, selection="best_clean")
    assert row["training_global_step"] == 1800
    assert checkpoint.name == "step1800.pt"


def test_3600_named_selection_rejects_history_mismatch() -> None:
    result = _result()
    result["selections"]["best_clean"]["checkpoint_sha256"] = "wrong"  # type: ignore[index]
    with pytest.raises(ValueError, match="SHA256"):
        _selected_record(result, 1800, selection="best_clean")


def test_fixed_step_selection_remains_the_legacy_path() -> None:
    result = _result()
    result["selection"] = result["history"][3]
    row, checkpoint = _selected_record(result, 1800)
    assert row["checkpoint"] == "step1800.pt"
    assert checkpoint.name == "step1800.pt"
