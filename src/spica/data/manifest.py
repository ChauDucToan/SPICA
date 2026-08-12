from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    path: Path
    label: int


def _parse_labeled_line(
    line: str,
    *,
    source: Path,
    line_number: int,
) -> tuple[str, int]:
    try:
        value, raw_label = line.rsplit(maxsplit=1)
    except ValueError as err:
        raise ValueError(
            f"Invalid line {line_number} in {source}: expected '<value> <label>'"
        ) from err

    try:
        label = int(raw_label)
    except ValueError as err:
        raise ValueError(
            f"Invalid label on line {line_number} in {source}: {raw_label!r}"
        ) from err

    if label < 0:
        raise ValueError(f"Negative label on line {line_number} in {source}: {label}")

    return value, label


def read_manifest(
    manifest_path: Path,
    dataset_root: Path,
) -> tuple[ManifestEntry, ...]:
    entries: list[ManifestEntry] = []

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            raw_path, label = _parse_labeled_line(
                stripped,
                source=manifest_path,
                line_number=line_number,
            )
            relative_path = Path(raw_path)

            if relative_path.is_absolute():
                raise ValueError(
                    f"Absolute path on line {line_number} in {manifest_path}: {relative_path}"
                )

            if ".." in relative_path.parts:
                raise ValueError(
                    f"Path escapes dataset root on line {line_number} in {manifest_path}: {relative_path}"
                )

            entries.append(
                ManifestEntry(path=dataset_root / relative_path, label=label)
            )
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    return tuple(entries)


def read_class_map(class_map_path: Path) -> dict[int, str]:
    id_to_name: dict[int, str] = {}
    seen_names: set[str] = set()

    with class_map_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            class_name, class_id = _parse_labeled_line(
                stripped,
                source=class_map_path,
                line_number=line_number,
            )

            if class_id in id_to_name:
                raise ValueError(f"Duplicate class ID {class_id} in {class_map_path}")

            if class_name in seen_names:
                raise ValueError(
                    f"Duplicate class name {class_name!r} in {class_map_path}"
                )

            id_to_name[class_id] = class_name
            seen_names.add(class_name)

    if not id_to_name:
        raise ValueError(f"Class map is empty: {class_map_path}")

    return id_to_name
