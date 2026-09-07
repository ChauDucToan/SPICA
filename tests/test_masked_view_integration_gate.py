from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).parents[1]
GATE = ROOT / "scripts" / "check_masked_view_integration_cpu.py"


def test_gate_uses_masked_campaign_configs_and_not_pairing_gate() -> None:
    source = GATE.read_text(encoding="utf-8")
    assert "run_pairing_gate" not in source
    assert "check_pairing_pilot_integration_cpu import" in source
    assert "run_gate as" not in source
    for arm, view_mode in (("masked_view_C", "full_full"), ("masked_view_M", "full_masked")):
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            args = compose(
                config_name="train_frozen_prompt",
                overrides=[f"+experiments={arm}"],
            )
        assert args.experiment_campaign == "frozen_prompt_masked_view_pilot_2026-09-07"
        assert args.experiment_role == f"frozen_prompt_{arm}"
        assert args.sketch_view_mode == view_mode
        assert args.mask_policy.version == "ink_centered_square_v1"


def test_gate_is_explicit_and_fail_closed() -> None:
    source = GATE.read_text(encoding="utf-8")
    assert "--allow-smoke" in source
    assert "raise FileExistsError" in source
    assert "except Exception" not in source
    assert '"CPU_MASKED_VIEW_SMOKE_PASS"' in source
    assert '"PENDING"' not in source
