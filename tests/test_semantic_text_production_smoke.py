"""CPU production smoke tests for semantic S0/S1/S2.

This intentionally imports the gate rather than rebuilding a second trainer
fixture.  A failing production path remains a failing test and is preserved in
the gate's JSON failure bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_semantic_text_integration_cpu import ARMS, run_gate


def test_semantic_gate_requires_explicit_fixture_permission(tmp_path: Path) -> None:
    assert run_gate(tmp_path / "refused", allow_smoke=False) == 2
    assert not (tmp_path / "refused").exists()


def test_semantic_production_smoke_runs_all_arms_and_selected_replays(
    tmp_path: Path,
) -> None:
    output = tmp_path / "semantic_cpu_gate"
    status = run_gate(output, allow_smoke=True)
    summary = json.loads((output / "gate_summary.json").read_text(encoding="utf-8"))

    assert status == 0, json.dumps(summary, indent=2)
    assert summary["status"] == "CPU_SEMANTIC_TEXT_SMOKE_PASS"
    assert tuple(summary["arms"]) == ARMS
    assert summary["updates_per_arm"] == 2
    assert summary["probe_steps"] == [0, 1, 2]
    assert summary["mask_conditions_per_probe"] == 9
    assert summary["pretrained_downloaded"] is False
    assert summary["official_unseen_used"] is False
    assert summary["failures"] == {}
    for arm in ARMS:
        report_path = Path(summary["training_reports"][arm])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["experiment_role"] == arm
        assert report["artifact_identity"]["actual_final_step"] == 2
        assert set(summary["selected_replays"][arm]) == {
            "latest",
            "best_clean",
            "best_masked",
        }
        for replay_path in summary["selected_replays"][arm].values():
            replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
            assert replay["status"] == "COMPLETE"
            assert len(replay["conditions"]) == 9
