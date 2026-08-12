from dataclasses import dataclass
from pathlib import Path

from .yaml_reader import read_yaml


@dataclass(frozen=True)
class SplitConfig:
    sketch_manifest: Path
    photo_manifest: Path
    class_map: Path


@dataclass(frozen=True)
class DataConfig:
    version: int
    name: str
    root: Path
    train: SplitConfig
    test: SplitConfig


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise FileNotFoundError(f"Could not find project root from: {start}")


def load_data_config(config_path: Path) -> DataConfig:
    raw = read_yaml(config_path)

    required = {"version", "name", "root", "train", "test"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    project_root = _find_project_root(Path(__file__).resolve().parent)

    root = Path(raw["root"]).expanduser()
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    def load_split(name: str) -> SplitConfig:
        section = raw[name]

        required_files = {"sketch_manifest", "photo_manifest", "class_map"}

        missing_files = required_files - section.keys()
        if missing_files:
            raise ValueError(f"Missing '{name}' keys: {sorted(missing_files)}")

        split = SplitConfig(
            sketch_manifest=root / section["sketch_manifest"],
            photo_manifest=root / section["photo_manifest"],
            class_map=root / section["class_map"],
        )

        for path in (
            split.sketch_manifest,
            split.photo_manifest,
            split.class_map,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Manifest not found: {path}")

        return split

    return DataConfig(
        version=int(raw["version"]),
        name=str(raw["name"]),
        root=root,
        train=load_split("train"),
        test=load_split("test"),
    )
