import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    name: str
    root: Path
    sketch_dir: str
    photo_dir: str
    ignore_dirs: frozenset[str]
    unseen_classes: tuple[str, ...]


def load_data_config(config_path: Path) -> DataConfig:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise TypeError("Data config must be a YAML mapping")

    data = raw.get("data")
    if not isinstance(data, dict):
        raise TypeError("Missing 'data' section")

    required = {"root", "sketch_dir", "photo_dir", "unseen_classes"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    project_root = Path(__file__).resolve().parents[3]

    root = Path(data["root"]).expanduser()
    if not root.is_absolute():
        root = project_root / root

    ignore_dirs = data.get("ignore_dirs", [])

    normalized_ignored = frozenset(
        Path(str(name).rstrip("/")).name for name in ignore_dirs
    )

    unseen_classes = tuple(data["unseen_classes"])

    if len(unseen_classes) != len(set(unseen_classes)):
        raise ValueError("Duplicate entries in 'unseen_classes'")

    return DataConfig(
        name=str(raw["name"]),
        root=root.resolve(),
        sketch_dir=str(data["sketch_dir"]).rstrip("/"),
        photo_dir=str(data["photo_dir"]).rstrip("/"),
        ignore_dirs=normalized_ignored,
        unseen_classes=unseen_classes,
    )


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description="Sketch-based retrieval")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/sketchy_ext_raw.yaml"),
    )
    parser.add_argument("--max-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)

    # Giữ các training option thực sự cần thiết ở đây.
    parser.add_argument("--exp-name", default="LN_prompt")
    parser.add_argument("--clip-ln-lr", type=float, default=1e-6)
    parser.add_argument("--prompt-lr", type=float, default=1e-4)
    parser.add_argument("--prompt-dim", type=int, default=768)
    parser.add_argument("--n-prompts", type=int, default=3)

    opts = parser.parse_args(argv)
    opts.data = load_data_config(opts.data_config)
    return opts


opts = parse_options()
