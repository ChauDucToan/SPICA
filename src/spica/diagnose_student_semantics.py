"""Scoped semantic diagnostic for archived deep-prompt Student and Teacher.

This module deliberately reads only the official seen-training manifests.  It
measures same-modality sketch label separation and cross-modal retrieval in one
shared photo gallery; it is not an independent holdout or a shape-quality
proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple
from unittest.mock import patch

from PIL import Image
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from .data.datasets import RetrievalEvalDataset
from .data.manifest import ManifestEntry, read_class_map, read_manifest
from .evaluation.embeddings import (
    EncodedRetrievalSet,
    encode_retrieval_loader,
    save_encoded_retrieval_set,
)
from .evaluation.metrics import CategoryRetrievalEvaluation, evaluate_category_retrieval
from .models.clip import load_frozen_clip

SEED = 42
SKETCHES_PER_CLASS = 70
CENTROID_TRAIN_PER_CLASS = 56
QUERY_PER_CLASS = 14
PHOTOS_PER_CLASS = 8
EMBEDDING_DIM = 512
GALLERY_SIZE = 219 * PHOTOS_PER_CLASS + 4  # squirrel has only four train photos
QUERY_SIZE = 220 * QUERY_PER_CLASS

FeatureMatrix = Float[Tensor, "items 512"]
LabelVector = Int[Tensor, "items"]
MaskVector = Bool[Tensor, "items"]


class SplitSelection(NamedTuple):
    all_sketches: tuple[ManifestEntry, ...]
    centroid_train: tuple[ManifestEntry, ...]
    queries: tuple[ManifestEntry, ...]
    gallery: tuple[ManifestEntry, ...]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_archived_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import archived module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_hash() -> str:
    digest = hashlib.sha256()
    digest.update(pickle.dumps(random.getstate()))
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_input_shapes(
    features: FeatureMatrix,
    labels: LabelVector,
    query_mask: MaskVector,
) -> None:
    """Validate diagnostic tensor contracts without runtime shape decorators."""
    if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected features [items, 512], got {tuple(features.shape)}")
    if labels.ndim != 1 or query_mask.ndim != 1:
        raise ValueError("Labels and query mask must be one-dimensional")
    if len(features) != len(labels) or len(features) != len(query_mask):
        raise ValueError("Features, labels, and query mask must have equal length")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("Features must be finite floating-point embeddings")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"Labels must be integer, got {labels.dtype}")
    if query_mask.dtype != torch.bool:
        raise TypeError(f"Query mask must be bool, got {query_mask.dtype}")


def feature_statistics(
    features: FeatureMatrix,
    labels: LabelVector,
    train_mask: MaskVector,
    query_mask: MaskVector,
) -> tuple[dict[str, float | int], dict[int, dict[str, int | float]]]:
    """Nearest normalized class-centroid classification, split-safe."""
    validate_input_shapes(features, labels, query_mask)
    if (
        train_mask.shape != query_mask.shape
        or train_mask.shape != labels.shape
        or train_mask.dtype != torch.bool
        or (train_mask & query_mask).any()
        or not query_mask.any()
    ):
        raise ValueError("Expected disjoint nonempty centroid/query masks")
    class_ids = sorted(labels.unique().tolist())
    normalized = F.normalize(features.float(), dim=1)
    centroids = []
    for class_id in class_ids:
        selected = normalized[train_mask & labels.eq(class_id)]
        if not len(selected):
            raise ValueError(f"No centroid-training sketches for class {class_id}")
        centroids.append(F.normalize(selected.mean(dim=0), dim=0))
    centroid_matrix = torch.stack(centroids)
    query_indices = torch.nonzero(query_mask).flatten()
    query_scores = normalized[query_indices] @ centroid_matrix.T
    predicted_positions = query_scores.argmax(dim=1)
    predicted = torch.tensor(class_ids, dtype=labels.dtype)[predicted_positions]
    actual = labels[query_indices]

    rows: dict[int, dict[str, int | float]] = {}
    for class_id in class_ids:
        local = actual.eq(class_id)
        rows[class_id] = {
            "query_count": int(local.sum()),
            "correct": int((predicted[local] == class_id).sum()),
            "accuracy": float((predicted[local] == class_id).float().mean()),
        }
    accuracy = float((predicted == actual).float().mean())
    return {
        "query_count": len(actual),
        "class_count": len(class_ids),
        "top1_accuracy": accuracy,
        "macro_accuracy": sum(float(row["accuracy"]) for row in rows.values())
        / len(rows),
    }, rows


def retrieval_stats(
    query_features: FeatureMatrix,
    query_labels: LabelVector,
    gallery_features: FeatureMatrix,
    gallery_labels: LabelVector,
    device: torch.device,
) -> tuple[CategoryRetrievalEvaluation, Tensor]:
    """Full AP, P@200 and AP@200 use actual positive count per class (4 or 8)."""
    queries = EncodedRetrievalSet(
        embeddings=query_features.float().cpu(),
        labels=query_labels.long().cpu(),
        paths=tuple(str(i) for i in range(len(query_features))),
    )
    gallery = EncodedRetrievalSet(
        embeddings=gallery_features.float().cpu(),
        labels=gallery_labels.long().cpu(),
        paths=tuple(str(i) for i in range(len(gallery_features))),
    )
    evaluation = evaluate_category_retrieval(
        queries,
        gallery,
        precision_at_k=(200,),
        top_k=200,
        query_chunk_size=256,
        device=device,
    )
    relevant = gallery.labels[evaluation.top_indices].eq(query_labels[:, None])
    precision = relevant.double().cumsum(dim=1) / torch.arange(
        1, 201, dtype=torch.float64
    )
    positives = torch.bincount(gallery.labels)[queries.labels].clamp(max=200)
    ap200 = (precision * relevant).sum(dim=1) / positives
    return evaluation, ap200


def rank_agreement(
    student_queries: FeatureMatrix,
    teacher_queries: FeatureMatrix,
    gallery: FeatureMatrix,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cosine, Pearson, top-200 overlap, ordinal rank correlation (stable tie-break).

    Ordinal ranks are NOT tie-corrected Spearman; exact ties follow gallery order.
    """
    student = F.normalize(student_queries.float(), dim=1)
    teacher = F.normalize(teacher_queries.float(), dim=1)
    gallery = F.normalize(gallery.float(), dim=1)
    cosine = (student * teacher).sum(dim=1)
    pearsons: list[Tensor] = []
    overlaps: list[Tensor] = []
    spearmans: list[Tensor] = []
    for start in range(0, len(student), 256):
        stop = min(start + 256, len(student))
        student_scores = student[start:stop] @ gallery.T
        teacher_scores = teacher[start:stop] @ gallery.T
        student_order = torch.argsort(
            student_scores, dim=1, descending=True, stable=True
        )
        teacher_order = torch.argsort(
            teacher_scores, dim=1, descending=True, stable=True
        )
        student_centered = student_scores - student_scores.mean(dim=1, keepdim=True)
        teacher_centered = teacher_scores - teacher_scores.mean(dim=1, keepdim=True)
        pearsons.append(
            F.cosine_similarity(student_centered, teacher_centered, dim=1).double()
        )
        selected = torch.zeros((stop - start, gallery.shape[0]), dtype=torch.bool)
        selected.scatter_(1, student_order[:, :200], True)
        overlaps.append(
            selected.gather(1, teacher_order[:, :200]).sum(dim=1).float() / 200.0
        )
        ranks = torch.arange(gallery.shape[0], dtype=torch.float64).expand(
            stop - start, -1
        )
        student_ranks = torch.empty_like(ranks)
        teacher_ranks = torch.empty_like(ranks)
        student_ranks.scatter_(1, student_order, ranks)
        teacher_ranks.scatter_(1, teacher_order, ranks)
        difference = student_ranks - teacher_ranks
        n = float(gallery.shape[0])
        spearmans.append(
            1.0 - 6.0 * difference.square().sum(dim=1) / (n * (n * n - 1.0))
        )
    return cosine, torch.cat(pearsons), torch.cat(overlaps), torch.cat(spearmans)


def select_entries(
    sketch_entries: tuple[ManifestEntry, ...],
    photo_entries: tuple[ManifestEntry, ...],
) -> SplitSelection:
    """Sort groups, then independently shuffle each group using seed 42."""
    sketch_groups: dict[int, list[ManifestEntry]] = defaultdict(list)
    photo_groups: dict[int, list[ManifestEntry]] = defaultdict(list)
    if len({entry.path for entry in sketch_entries}) != len(sketch_entries) or len(
        {entry.path for entry in photo_entries}
    ) != len(photo_entries):
        raise ValueError("Duplicate paths in train manifests")
    for entry in sketch_entries:
        sketch_groups[entry.label].append(entry)
    for entry in photo_entries:
        photo_groups[entry.label].append(entry)
    if set(sketch_groups) != set(photo_groups) or len(sketch_groups) != 220:
        raise ValueError("Expected equal 220-class seen sketch/photo manifests")

    sketch_rng = random.Random(SEED)
    photo_rng = random.Random(SEED)
    all_sketches: list[ManifestEntry] = []
    centroid_train: list[ManifestEntry] = []
    queries: list[ManifestEntry] = []
    gallery: list[ManifestEntry] = []
    for class_id in sorted(sketch_groups):
        sketches = sorted(sketch_groups[class_id], key=lambda item: str(item.path))
        photos = sorted(photo_groups[class_id], key=lambda item: str(item.path))
        if len(sketches) != SKETCHES_PER_CLASS or not photos:
            raise ValueError(
                f"Class {class_id} has {len(sketches)} sketches and {len(photos)} photos; "
                "expected 70 sketches and nonempty train photos"
            )
        sketch_rng.shuffle(sketches)
        photo_rng.shuffle(photos)
        all_sketches.extend(sketches)
        centroid_train.extend(sketches[:CENTROID_TRAIN_PER_CLASS])
        queries.extend(sketches[CENTROID_TRAIN_PER_CLASS:])
        gallery.extend(photos[:PHOTOS_PER_CLASS])  # no duplicate padding for squirrel
    return SplitSelection(
        tuple(all_sketches), tuple(centroid_train), tuple(queries), tuple(gallery)
    )


def image_hash(path: Path, allowed_paths: set[Path]) -> str:
    resolved = path.resolve()
    if resolved not in allowed_paths:
        raise ValueError(f"Image path is outside allowlisted train manifests: {path}")
    # Hash bytes only after manifest allowlisting; PIL reads happen in RetrievalEvalDataset.
    return sha256(resolved)


def write_split_audit(
    output: Path, selection: SplitSelection, hashes: dict[Path, str]
) -> None:
    rows: list[tuple[str, ManifestEntry, str]] = []
    centroid_paths = {entry.path for entry in selection.centroid_train}
    rows.extend(
        ("sketch", entry, "centroid_train" if entry.path in centroid_paths else "query")
        for entry in selection.all_sketches
    )
    rows.extend(("photo", entry, "gallery") for entry in selection.gallery)
    with (output / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("kind", "split", "label", "path", "source_image_sha256"))
        for kind, entry, split in rows:
            writer.writerow(
                (kind, split, entry.label, str(entry.path), hashes[entry.path])
            )


def encode_entries(
    encoder: torch.nn.Module,
    entries: tuple[ManifestEntry, ...],
    transform: Any,
    allowed_paths: set[Path],
    batch_size: int = 64,
) -> EncodedRetrievalSet:
    dataset = RetrievalEvalDataset(entries, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(SEED),
    )
    original_open = Image.open

    def train_only_open(path, *args, **kwargs):
        if Path(path).resolve() not in allowed_paths:
            raise ValueError(f"Image decode outside train allowlist: {path}")
        return original_open(path, *args, **kwargs)

    with patch.object(Image, "open", train_only_open):
        return encode_retrieval_loader(encoder, loader)


def validate_archive(run: Path, config: dict[str, Any]) -> dict[str, str]:
    archive_root = run / "source"
    required = {
        "src/spica/models/sketch_student.py",
        "src/spica/models/sketch_teacher.py",
    }
    if not required.issubset(config["source_sha256"]):
        raise ValueError("Checkpoint source map must bind both archived encoder files")
    bindings: dict[str, str] = {}
    for relative, expected in config["source_sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe source archive path: {relative}")
        source_path = archive_root / relative
        if not source_path.resolve().is_relative_to(archive_root.resolve()):
            raise ValueError(f"Source archive path escapes root: {relative}")
        if not source_path.is_file() or sha256(source_path) != expected:
            raise ValueError(f"Archived source hash mismatch: {relative}")
        bindings[relative] = str(source_path)
    return bindings


def validate_run(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config_path = run / "config.json"
    result_path = run / "result.json"
    checkpoint_path = run / "student_latest.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if sha256(config_path) != result["config_sha256"]:
        raise ValueError("Run config SHA disagrees with result.json")
    if sha256(checkpoint_path) != result["checkpoint_sha256"]:
        raise ValueError("Run checkpoint SHA disagrees with result.json")
    if config["method"] != "cnn_mamba_lora_vpt_deep_white_mask_teacher_v1":
        raise ValueError("Run is not the approved deep visual prompt checkpoint")
    teacher_config = config["sketch_teacher"]
    prompt = teacher_config["visual_prompt"]
    mask = teacher_config["white_patch_mask"]
    if (
        prompt["layers"] != 12
        or prompt["tokens_per_layer"] != 4
        or prompt["readout"] != "native_CLS"
        or prompt["position"] != "after_CLS_before_49_original_patches"
        or mask["patch_size"] != 32
        or mask["scope"] != "sketch_teacher_keys_all12layers"
        or mask["rule"]
        != "all_RGB_values_exactly_1_after_resize_crop_before_normalization"
        or teacher_config["lambda_kd"] != 1.2
        or config["dataset"] != "tuberlin_220_30"
        or result["epoch"] != 10
        or result["step"] != 4820
    ):
        raise ValueError("Checkpoint differs from approved deep-prompt configuration")
    bindings = validate_archive(run, config)
    return config, result, bindings


def load_models(
    run: Path,
    config: dict[str, Any],
    clip_weights: Path,
    device: torch.device,
    archive_bindings: dict[str, str],
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Any, dict[str, Any]]:
    archived_student = load_archived_module(
        Path(archive_bindings["src/spica/models/sketch_student.py"]),
        "archived_student_diagnostic",
    )
    archived_teacher = load_archived_module(
        Path(archive_bindings["src/spica/models/sketch_teacher.py"]),
        "archived_teacher_diagnostic",
    )
    checkpoint = torch.load(
        run / "student_latest.pt", map_location="cpu", weights_only=True
    )
    if (
        checkpoint["config"] != config
        or checkpoint["epoch"] != 10
        or checkpoint["step"] != 4820
    ):
        raise ValueError("Checkpoint/config/final step binding differs")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    bundle = load_frozen_clip(
        pretrained=str(clip_weights.expanduser().absolute()),
        device=device,
    )
    normalize = bundle.transform.transforms[-1]
    if (
        list(normalize.mean) != config["sketch_teacher"]["normalize_mean"]
        or list(normalize.std) != config["sketch_teacher"]["normalize_std"]
    ):
        raise ValueError("Pinned CLIP normalization differs from checkpoint")
    student = archived_student.SketchStudent().to(device)
    teacher_cfg = config["sketch_teacher"]
    teacher = archived_teacher.SketchTeacher(
        bundle.encoder.model.visual,
        teacher_cfg["normalize_mean"],
        teacher_cfg["normalize_std"],
        visual_prompt=True,
        mask_white_patches=True,
        deep_visual_prompt=True,
    ).to(device)
    student_result = student.load_state_dict(checkpoint["model"], strict=True)
    if student_result.missing_keys or student_result.unexpected_keys:
        raise ValueError("Student checkpoint was not restored with strict=True")

    expected_adapter = teacher.adapter_state_dict()
    actual_adapter = checkpoint["teacher_adapter"]
    if len(actual_adapter) != 60 or set(actual_adapter) != set(expected_adapter):
        raise ValueError(
            "Teacher adapter checkpoint must contain exact 60 trainable keys"
        )
    restored = teacher.load_state_dict(actual_adapter, strict=False)
    missing = set(restored.missing_keys)
    allowed_frozen = set(teacher.state_dict()) - set(expected_adapter)
    if restored.unexpected_keys or missing != allowed_frozen:
        raise ValueError("Teacher restoration missing keys outside frozen base state")

    if sum(parameter.numel() for parameter in expected_adapter.values()) != 331776:
        raise ValueError(
            "Archived teacher trainable parameter count differs from 331776"
        )
    student.eval()
    teacher.eval()
    bundle.encoder.eval()
    return student, teacher, bundle.encoder, bundle.transform, checkpoint


def per_query_rows(
    selection: SplitSelection,
    hashes: dict[Path, str],
    labels: LabelVector,
    student_eval: CategoryRetrievalEvaluation,
    teacher_eval: CategoryRetrievalEvaluation,
    student_ap200: Tensor,
    teacher_ap200: Tensor,
    cosine: Tensor,
    overlap: Tensor,
    pearson: Tensor,
    spearman: Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for index, entry in enumerate(selection.queries):
        rows.append(
            {
                "query_index": index,
                "path": str(entry.path),
                "label": int(labels[index]),
                "source_image_sha256": hashes[entry.path],
                "student_full_ap": float(
                    student_eval.average_precision_per_query[index]
                ),
                "teacher_full_ap": float(
                    teacher_eval.average_precision_per_query[index]
                ),
                "delta_teacher_minus_student_full_ap": float(
                    teacher_eval.average_precision_per_query[index]
                    - student_eval.average_precision_per_query[index]
                ),
                "student_ap200_min_relevant_k": float(student_ap200[index]),
                "teacher_ap200_min_relevant_k": float(teacher_ap200[index]),
                "delta_teacher_minus_student_ap200": float(
                    teacher_ap200[index] - student_ap200[index]
                ),
                "matched_query_cosine": float(cosine[index]),
                "gallery_score_pearson": float(pearson[index]),
                "top200_overlap_fraction": float(overlap[index]),
                "stable_ordinal_gallery_rank_correlation": float(spearman[index]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(args: argparse.Namespace) -> None:
    run = args.run.expanduser().absolute()
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Diagnostic output already exists: {output}")
    from .train_student import source_files

    diagnostic_source = source_files()
    output.mkdir(parents=True)
    json_write(output / "runtime.json", {"status": "VALIDATING"})
    for name in diagnostic_source:
        dest = output / "diagnostic_source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[2] / name).read_bytes())

    config, result, archive_bindings = validate_run(run)
    clip_weights = args.clip_weights or Path(config["clip_weights"])
    if not clip_weights.is_file():
        raise FileNotFoundError(f"Pinned CLIP weights not found: {clip_weights}")
    if not clip_weights.name.endswith(".safetensors"):
        raise ValueError("Pinned CLIP path must retain .safetensors suffix")
    if sha256(clip_weights) != config["clip_sha256"]:
        raise ValueError("Pinned CLIP SHA differs from checkpoint config")

    manifest_hashes = config["manifest_sha256"]
    train_paths = {
        Path(path): digest
        for path, digest in manifest_hashes.items()
        if Path(path).name
        in {
            "png_ready_filelist_train.txt",
            "ImageResized_ready_filelist_train.txt",
            "cname_cid.txt",
        }
    }

    def manifest_named(name: str) -> Path:
        matches = [path for path in train_paths if path.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name} manifest binding")
        return matches[0]

    train_sketch_manifest = manifest_named("png_ready_filelist_train.txt")
    train_photo_manifest = manifest_named("ImageResized_ready_filelist_train.txt")
    class_map_path = manifest_named("cname_cid.txt")
    for path, expected in manifest_hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Manifest/class-map identity mismatch: {path}")
    dataset_root = train_sketch_manifest.parent.parent
    sketch_entries = read_manifest(train_sketch_manifest, dataset_root)
    photo_entries = read_manifest(train_photo_manifest, dataset_root)
    class_map = read_class_map(class_map_path)
    if len(class_map) != 220 or len(sketch_entries) != 15400:
        raise ValueError("Expected 220 seen classes and 15400 training sketches")
    if set(class_map) != {entry.label for entry in sketch_entries}:
        raise ValueError("Seen class-map IDs differ from sketch labels")
    selection = select_entries(sketch_entries, photo_entries)
    allowed_paths = {entry.path.resolve() for entry in sketch_entries + photo_entries}
    selected_entries = selection.all_sketches + selection.gallery
    hashes = {
        entry.path: image_hash(entry.path, allowed_paths) for entry in selected_entries
    }
    write_split_audit(output, selection, hashes)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; pass --device cpu explicitly"
        )
    student, teacher, photo_encoder, photo_transform, checkpoint = load_models(
        run, config, clip_weights, device, archive_bindings
    )
    # Same resize/crop geometry used by training, without CLIP normalization,
    # feeds Student and Teacher. Photo gallery uses pinned CLIP transform.
    from .data.transforms import build_clip_eval_transform

    sketch_transform = transforms.Compose(build_clip_eval_transform().transforms[:-1])
    model_state_before = ":".join(
        (state_hash(student), state_hash(teacher), state_hash(photo_encoder))
    )
    mode_before = (student.training, teacher.training, photo_encoder.training)
    rng_before = rng_hash()

    with torch.inference_mode():
        json_write(
            output / "runtime.json", {"status": "ENCODING_STUDENT", "count": 15400}
        )
        all_student = encode_entries(
            student, selection.all_sketches, sketch_transform, allowed_paths
        )
        print("Student: encoded 15400 train sketches", flush=True)
        json_write(
            output / "runtime.json", {"status": "ENCODING_TEACHER", "count": 15400}
        )
        all_teacher = encode_entries(
            teacher, selection.all_sketches, sketch_transform, allowed_paths
        )
        print("Teacher: encoded 15400 train sketches", flush=True)
        shared_gallery = encode_entries(
            photo_encoder, selection.gallery, photo_transform, allowed_paths
        )
        print(f"Gallery: encoded {len(shared_gallery.labels)} train photos", flush=True)
    json_write(output / "runtime.json", {"status": "CPU_FEATURE_ANALYSIS"})
    if all_student.paths != all_teacher.paths or not torch.equal(
        all_student.labels, all_teacher.labels
    ):
        raise ValueError("Student and Teacher sketch path/label IDs differ")
    if shared_gallery.paths != tuple(str(entry.path) for entry in selection.gallery):
        raise ValueError("Gallery path order changed")
    if (
        all_student.embeddings.shape != (15400, EMBEDDING_DIM)
        or len(shared_gallery.labels) != GALLERY_SIZE
    ):
        raise ValueError("Unexpected encoded feature sizes")

    all_labels = all_student.labels
    centroid_paths = {entry.path for entry in selection.centroid_train}
    query_paths = {entry.path for entry in selection.queries}
    if (
        centroid_paths & query_paths
        or len(centroid_paths) != 12320
        or len(query_paths) != 3080
    ):
        raise ValueError("Centroid/query split overlaps or has wrong counts")
    centroid_mask = torch.tensor(
        [entry.path in centroid_paths for entry in selection.all_sketches],
        dtype=torch.bool,
    )
    query_mask = torch.tensor(
        [entry.path in query_paths for entry in selection.all_sketches],
        dtype=torch.bool,
    )
    for encoded in (all_student, all_teacher, shared_gallery):
        validate_input_shapes(
            encoded.embeddings,
            encoded.labels,
            torch.ones(len(encoded.labels), dtype=torch.bool),
        )
    student_centroid, student_class = feature_statistics(
        all_student.embeddings, all_labels, centroid_mask, query_mask
    )
    teacher_centroid, teacher_class = feature_statistics(
        all_teacher.embeddings, all_labels, centroid_mask, query_mask
    )
    query_indices = torch.nonzero(query_mask).flatten()
    student_queries = all_student.embeddings[query_indices]
    teacher_queries = all_teacher.embeddings[query_indices]
    query_labels = all_labels[query_indices]
    student_eval, student_ap200 = retrieval_stats(
        student_queries,
        query_labels,
        shared_gallery.embeddings,
        shared_gallery.labels,
        device,
    )
    teacher_eval, teacher_ap200 = retrieval_stats(
        teacher_queries,
        query_labels,
        shared_gallery.embeddings,
        shared_gallery.labels,
        device,
    )
    cosine, pearson, overlap, spearman = rank_agreement(
        student_queries, teacher_queries, shared_gallery.embeddings
    )

    model_state_after = ":".join(
        (state_hash(student), state_hash(teacher), state_hash(photo_encoder))
    )
    mode_after = (student.training, teacher.training, photo_encoder.training)
    rng_after = rng_hash()
    if (
        model_state_before != model_state_after
        or mode_before != mode_after
        or rng_before != rng_after
    ):
        raise RuntimeError("Diagnostic changed model state, mode, or RNG state")
    if any(
        parameter.grad is not None
        for module in (student, teacher, photo_encoder)
        for parameter in module.parameters()
    ):
        raise RuntimeError("Diagnostic left gradients on an evaluated model")

    metadata_common = {
        "checkpoint_sha256": result["checkpoint_sha256"],
        "config_sha256": result["config_sha256"],
        "archive_bindings": archive_bindings,
        "sample_paths": [str(entry.path) for entry in selection.all_sketches],
        "source_image_sha256": [hashes[entry.path] for entry in selection.all_sketches],
        "scope": "seen_train_only;not_independent_holdout;not_shape_quality_proof",
    }
    save_encoded_retrieval_set(
        all_student,
        output / "student_sketches.pt",
        metadata=metadata_common | {"arm": "student"},
    )
    save_encoded_retrieval_set(
        all_teacher,
        output / "teacher_sketches.pt",
        metadata=metadata_common | {"arm": "teacher"},
    )
    gallery_metadata = metadata_common | {
        "arm": "shared_frozen_photo_gallery",
        "sample_paths": [str(entry.path) for entry in selection.gallery],
        "source_image_sha256": [hashes[entry.path] for entry in selection.gallery],
    }
    save_encoded_retrieval_set(
        shared_gallery, output / "gallery.pt", metadata=gallery_metadata
    )

    rows = per_query_rows(
        selection,
        hashes,
        query_labels,
        student_eval,
        teacher_eval,
        student_ap200,
        teacher_ap200,
        cosine,
        pearson,
        overlap,
        spearman,
    )
    write_csv(output / "per_query.csv", rows)
    per_class_rows = []
    for class_id in sorted(student_class):
        query_class = query_labels.eq(class_id)
        indices = torch.nonzero(query_class).flatten()
        per_class_rows.append(
            {
                "label": class_id,
                "query_count": len(indices),
                "student_centroid_accuracy": student_class[class_id]["accuracy"],
                "teacher_centroid_accuracy": teacher_class[class_id]["accuracy"],
                "delta_teacher_minus_student_centroid_accuracy": teacher_class[
                    class_id
                ]["accuracy"]
                - student_class[class_id]["accuracy"],
                "student_full_ap": float(
                    student_eval.average_precision_per_query[indices].mean()
                ),
                "teacher_full_ap": float(
                    teacher_eval.average_precision_per_query[indices].mean()
                ),
                "delta_teacher_minus_student_full_ap": float(
                    (
                        teacher_eval.average_precision_per_query[indices]
                        - student_eval.average_precision_per_query[indices]
                    ).mean()
                ),
                "gallery_positives": int(shared_gallery.labels.eq(class_id).sum()),
                "student_ap200_min_relevant_k": float(student_ap200[indices].mean()),
                "teacher_ap200_min_relevant_k": float(teacher_ap200[indices].mean()),
            }
        )
    write_csv(output / "per_class.csv", per_class_rows)

    if source_files() != diagnostic_source:
        raise RuntimeError("Diagnostic source changed during execution")
    results = {
        "status": "FINISHED_SCOPED_SEMANTICS_DIAGNOSTIC",
        "diagnostic_source_sha256": diagnostic_source,
        "scope": "15400_seen_train_sketches;centroid_56_and_query_14_per_class;up_to_8_train_photos_per_class_squirrel4;no_test_images;no_training;no_classifier_optimizer",
        "interpretation": "Same-modality centroid accuracy measures sketch-label separation. Cross-modal retrieval measures sketch-to-photo retrieval. Images were used by encoder training, so neither is an independent holdout or shape-quality proof.",
        "checkpoint_sha256": result["checkpoint_sha256"],
        "config_sha256": result["config_sha256"],
        "archive_bindings": archive_bindings,
        "clip_weights": str(clip_weights),
        "clip_sha256": sha256(clip_weights),
        "data_identity": {
            "dataset": config["dataset"],
            "class_count": len(class_map),
            "sketch_count": len(sketch_entries),
            "gallery_count": len(shared_gallery.labels),
            "manifest_sha256": manifest_hashes,
            "data_config_sha256": config["data_config_sha256"],
        },
        "selection": {
            "seed": SEED,
            "centroid_train_per_class": CENTROID_TRAIN_PER_CLASS,
            "query_per_class": QUERY_PER_CLASS,
            "gallery_per_class_max": PHOTOS_PER_CLASS,
            "squirrel_gallery_count": 4,
            "ordering": "sorted manifest groups;local Random(42) shuffle per modality",
        },
        "student": {
            "centroid": student_centroid,
            "retrieval": {
                "full_mAP": student_eval.metrics.mean_average_precision,
                "P@200": student_eval.metrics.precision_at_k[200],
                "mAP@200_min_relevant_k": float(student_ap200.mean()),
                "class_macro_full_AP": sum(
                    float(row["student_full_ap"]) for row in per_class_rows
                )
                / 220,
            },
        },
        "teacher": {
            "centroid": teacher_centroid,
            "retrieval": {
                "full_mAP": teacher_eval.metrics.mean_average_precision,
                "P@200": teacher_eval.metrics.precision_at_k[200],
                "mAP@200_min_relevant_k": float(teacher_ap200.mean()),
                "class_macro_full_AP": sum(
                    float(row["teacher_full_ap"]) for row in per_class_rows
                )
                / 220,
            },
        },
        "matched_arms": {
            "query_cosine_mean": float(cosine.mean()),
            "query_cosine_min": float(cosine.min()),
            "query_cosine_max": float(cosine.max()),
            "gallery_score_pearson_mean": float(pearson.mean()),
            "top200_overlap_mean": float(overlap.mean()),
            "stable_ordinal_gallery_rank_mean": float(spearman.mean()),
            "stable_ordinal_gallery_rank_min": float(spearman.min()),
        },
        "evaluation_integrity": {
            "model_state_before": model_state_before,
            "model_state_after": model_state_after,
            "rng_before": rng_before,
            "rng_after": rng_after,
            "modes_before": mode_before,
            "modes_after": mode_after,
            "no_gradients": True,
            "input_image_allowlist": "selected entries from seen train manifests only",
        },
        "execution": {
            "command": " ".join(sys.argv),
            "device": str(device),
            "independent_encoder_replay": False,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "limits": "fixed split/RNG; CUDA kernels and near-tie rankings not certified bit-reproducible",
        },
        "artifacts_sha256": {
            name: sha256(output / name)
            for name in (
                "student_sketches.pt",
                "teacher_sketches.pt",
                "gallery.pt",
                "per_query.csv",
                "per_class.csv",
                "split_manifest.csv",
            )
        },
    }
    json_write(output / "results.json", results)
    (output / "execution_commands.txt").write_text(
        f"Command: {' '.join(sys.argv)}\nDevice: {device}; no independent encoder replay.\n",
        encoding="utf-8",
    )
    json_write(
        output / "runtime.json",
        {"status": "FINISHED_SCOPED_SEMANTICS_DIAGNOSTIC", "output": str(output)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Archived deep-prompt checkpoint run root",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh diagnostic output directory"
    )
    parser.add_argument(
        "--clip-weights",
        type=Path,
        help="Pinned .safetensors path; defaults to checkpoint config",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, default: cuda")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output_was_present = arguments.output.exists() or arguments.output.is_symlink()
    try:
        diagnose(arguments)
    except BaseException as error:
        if not output_was_present and arguments.output.is_dir():
            json_write(
                arguments.output / "runtime.json",
                {"status": "FAILED_NO_RETRY", "error": repr(error)},
            )
        raise
