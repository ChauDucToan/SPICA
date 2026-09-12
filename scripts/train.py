"""Small Hydra entry point: existing coupled trainer, then final-only evaluation."""
import argparse
import gc
from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spica.train_coupled_benchmark import train  # noqa: E402
from evaluate_coupled_benchmark import main as evaluate  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_coupled_benchmark")
def main(config: DictConfig) -> None:
    output = Path(HydraConfig.get().runtime.output_dir)
    args = argparse.Namespace(**OmegaConf.to_container(config, resolve=True),
                              output_dir=str(output / "train"), campaign_root=str(output))
    # Selecting the explicit official_test runtime is itself an opt-in; the
    # default minimal runtime keeps the historical train-then-evaluate flow.
    args.periodic_test = bool(args.periodic_test or "test_every_percent" in args.runtime)
    result = train(args)
    if not args.smoke and not args.periodic_test:
        gc.collect()
        evaluate(["--run-dir", args.output_dir, "--output-dir", str(output / "evaluation"),
                  "--device", args.device, "--arm", result["arm"]])


if __name__ == "__main__":
    main()
