"""Audit filename-derived Sketchy pairing candidates without declaring truth."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

SUFFIX = re.compile(r"-\d+$")


def read_manifest(path: Path) -> list[tuple[str, int]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            value, label = line.rsplit(maxsplit=1)
            rows.append((value, int(label)))
    return rows


def audit(root: Path, split: str) -> dict[str, object]:
    sketch_rows = read_manifest(
        root / "zeroshot0" / f"sketch_tx_000000000000_ready_filelist_{split}.txt"
    )
    photo_rows = read_manifest(root / "zeroshot0" / f"all_photo_filelist_{split}.txt")
    photos: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, label in photo_rows:
        photos[Path(path).stem].append((path, label))
    candidates = [SUFFIX.sub("", Path(path).stem) for path, _ in sketch_rows]
    counts = Counter(candidates)
    ambiguous_stems = {stem for stem, rows in photos.items() if len(rows) > 1}
    unique_matches = [
        photos[candidate][0] if len(photos.get(candidate, ())) == 1 else None
        for candidate in candidates
    ]
    label_mismatches = sum(
        match is not None and match[1] != sketch_label
        for match, (_, sketch_label) in zip(unique_matches, sketch_rows, strict=True)
    )
    existing_paths = sum(
        match is not None and (root / match[0]).is_file() for match in unique_matches
    )
    candidate_matches = sum(candidate in photos for candidate in candidates)
    return {
        "split": split,
        "status": "CANDIDATE_ONLY_NOT_VERIFIED",
        "sketch_rows": len(sketch_rows),
        "photo_rows": len(photo_rows),
        "exact_stem_matches": sum(
            Path(path).stem in photos for path, _ in sketch_rows
        ),
        "candidate_rows_matching_photo_stem": candidate_matches,
        "candidate_rows_with_ambiguous_photo_stem": sum(
            candidate in ambiguous_stems for candidate in candidates
        ),
        "photo_stems_with_multiple_rows": len(ambiguous_stems),
        "candidate_unique_keys": len(counts),
        "candidate_keys_with_multiple_sketches": sum(
            value > 1 for value in counts.values()
        ),
        "candidate_max_sketches_per_photo_candidate": max(counts.values(), default=0),
        "candidate_unique_photo_paths_exist": existing_paths,
        "candidate_label_mismatches": label_mismatches,
        "examples": [
            {
                "sketch": sketch_path,
                "candidate_photo_stem": candidate,
                "photo": match[0] if match is not None else None,
                "photo_mapping_status": (
                    "UNIQUE"
                    if len(photos.get(candidate, ())) == 1
                    else "AMBIGUOUS_PHOTO_STEM"
                    if len(photos.get(candidate, ())) > 1
                    else "MISSING"
                ),
                "sketch_label": sketch_label,
                "photo_label": match[1] if match is not None else None,
            }
            for (sketch_path, sketch_label), candidate, match in list(
                zip(sketch_rows, candidates, unique_matches, strict=True)
            )[:10]
        ],
    }


def _self_check() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "zeroshot0"
        data.mkdir()
        (data / "sketch_tx_000000000000_ready_filelist_train.txt").write_text(
            "a-1.png 0\na-2.png 0\nb-1.png 1\nc-1.png 2\n"
        )
        (data / "all_photo_filelist_train.txt").write_text(
            "a.jpg 0\na.jpg 0\nb.jpg 9\n"
        )
        (data / "sketch_tx_000000000000_ready_filelist_zero.txt").write_text("")
        (data / "all_photo_filelist_zero.txt").write_text("")
        (root / "a.jpg").write_bytes(b"photo")
        result = audit(root, "train")
        assert result["status"] == "CANDIDATE_ONLY_NOT_VERIFIED"
        assert result["candidate_rows_matching_photo_stem"] == 3
        assert result["candidate_rows_with_ambiguous_photo_stem"] == 2
        assert result["candidate_keys_with_multiple_sketches"] == 1
        assert result["candidate_label_mismatches"] == 1
        assert result["candidate_unique_photo_paths_exist"] == 0
        assert {row["photo_mapping_status"] for row in result["examples"]} == {
            "AMBIGUOUS_PHOTO_STEM",
            "UNIQUE",
            "MISSING",
        }
        (data / "all_photo_filelist_train.txt").write_text("a.jpg 0\nb.jpg 9\n")
        unique = audit(root, "train")
        assert unique["candidate_unique_photo_paths_exist"] == 2
        assert unique["candidate_rows_with_ambiguous_photo_stem"] == 0
        assert unique["candidate_label_mismatches"] == 1


def main() -> None:
    _self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/Sketchy"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {split: audit(args.root, split) for split in ("train", "zero")}
    train_keys = {
        SUFFIX.sub("", Path(path).stem)
        for path, _ in read_manifest(
            args.root
            / "zeroshot0"
            / "sketch_tx_000000000000_ready_filelist_train.txt"
        )
    }
    zero_keys = {
        SUFFIX.sub("", Path(path).stem)
        for path, _ in read_manifest(
            args.root
            / "zeroshot0"
            / "sketch_tx_000000000000_ready_filelist_zero.txt"
        )
    }
    result["cross_split_candidate_key_overlap"] = len(train_keys & zero_keys)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
