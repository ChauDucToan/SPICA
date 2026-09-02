"""Dry-run semantic validation for corrected transport experiment configs."""

from __future__ import annotations

from omegaconf import OmegaConf

from spica.train_transport import PROJECT_ROOT, _validate_options

CONFIGS = (
    "transport_factorial_base_no_text",
    "transport_factorial_base_text",
    "transport_factorial_transport_no_text",
    "transport_factorial_transport_text",
    "transport_corrected_p1_source73",
    "transport_corrected_p1_A_trainable_restored",
    "transport_corrected_p1_B_frozen_restored",
    "transport_corrected_p1_C_trainable_reset",
    "transport_corrected_p1_D_frozen_reset",
    "transport_corrected_p2_stage1",
    "transport_corrected_p2_S1_no_direction",
    "transport_corrected_p2_S2_class_centroid",
    "transport_corrected_p2_S3_moving",
    "transport_corrected_p2_S4_fixed_reference",
)


def main() -> None:
    base = OmegaConf.load(PROJECT_ROOT / "configs" / "train_transport.yaml")
    for name in CONFIGS:
        override = OmegaConf.load(
            PROJECT_ROOT / "configs" / "experiments" / f"{name}.yaml"
        )
        config = OmegaConf.merge(base, override)
        if config.experiment_role in {
            "freeze_optimizer_A",
            "freeze_optimizer_B",
            "freeze_optimizer_C",
            "freeze_optimizer_D",
            "two_stage_S1",
            "two_stage_S2",
            "two_stage_S3",
            "two_stage_S4",
        }:
            config.resume_checkpoint_path = "/REQUIRED/step73.pt"
        _validate_options(config)
        print(f"VALID {name}")


if __name__ == "__main__":
    main()
