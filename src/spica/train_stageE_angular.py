"""Explicit entrypoint for the matched Stage-E angular-routing control."""

import hydra
from omegaconf import DictConfig

from .train_deterministic import HYDRA_CONFIG_DIR
from .train_stageE_no_vmf import run


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_stageE_angular",
)
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
