"""CUDA-only final SketchLVM raster-erasure evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import data as data_package
from .config.data import DataConfig, load_data_config
from .data.manifest import ManifestEntry, read_class_map, read_manifest
from .diagnose_student_semantics import (
    encode_entries,
    load_archived_module,
    retrieval_stats,
    rng_hash,
    sha256,
    state_hash,
)
from .models.clip import load_frozen_clip
from .evaluation.embeddings import EncodedRetrievalSet, save_encoded_retrieval_set
from .train_student import source_files, write_json as _write_json
from .train_sketchlvm import (
    EXPECTED_CLIP_SHA256,
    PAPER_IMAGE_MEAN,
    PAPER_IMAGE_STD,
    PAPER_METHOD,
    PAPER_PREPROCESSING,
    _paper_clean_transform,
)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
STEPS_PER_EPOCH = 241
TOTAL_STEPS = 14_460
TRAIN_SKETCHES = 15_400
ADAPTATION_PARAMETERS = 84_480
ADAPTATION_KEYS = 106
QUERY_COUNT = 2_400
GALLERY_COUNT = 27_989
FRACTIONS = (0.25, 0.50, 0.75)
CONDITIONS = ("clean", "raster_25", "raster_50", "raster_75")


def _raw_transform(bundle_transform: Any) -> Any:
    items = list(getattr(bundle_transform, "transforms", ()))
    if not items or not isinstance(items[-1], transforms.Normalize):
        raise TypeError("Native CLIP evaluation transform must end in Normalize")
    return transforms.Compose(items[:-1])


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _bitmap_sha(bitmap: Tensor) -> str:
    return hashlib.sha256(
        bitmap.to(dtype=torch.uint8).contiguous().numpy().tobytes()
    ).hexdigest()


def _cpu_mask_oracle(clean: Tensor, fraction: float) -> np.ndarray:
    mean = clean.detach().cpu().numpy().mean(axis=0, dtype=np.float32)
    ink = mean < np.float32(0.9)
    positions = np.flatnonzero(ink.ravel(order="C"))
    count = int(math.ceil(fraction * len(positions))) if len(positions) else 0
    bitmap = np.zeros(ink.shape, dtype=np.uint8)
    if count:
        bitmap.ravel(order="C")[positions[:count]] = 1
    return bitmap


def rasterize_sketch(
    image: Image.Image, fraction: float
) -> tuple[Tensor, dict[str, Any]]:
    """Raster-erasure mask on raw RGB after the native 224 resize/crop."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    source_rgb = image.convert("RGB")
    raw = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )(source_rgb).to(dtype=torch.float32)
    bitmap_np = _cpu_mask_oracle(raw, fraction)
    bitmap = torch.from_numpy(bitmap_np.copy())
    mean = raw.mean(dim=0)
    ink = mean < 0.9
    n = int(ink.sum().item())
    k = int(bitmap.sum().item())
    if k != (math.ceil(fraction * n) if n else 0):
        raise AssertionError("Raster mask count differs from row-major oracle")
    masked = raw.clone()
    if k:
        masked[:, bitmap.bool()] = 1.0
    if not torch.equal(masked[:, ~bitmap.bool()], raw[:, ~bitmap.bool()]):
        raise AssertionError("Raster erasure changed an unselected pixel")
    total = float((1.0 - mean[ink]).sum().item())
    removed = float((1.0 - mean[bitmap.bool()]).sum().item()) if k else 0.0
    trace = {
        "N": n,
        "k": k,
        "requested_fraction": float(fraction),
        "actual_fraction": float(k / n) if n else 0.0,
        "mask_bitmap_sha256": _bitmap_sha(bitmap),
        "raw_source_rgb_decoded_sha256": hashlib.sha256(
            source_rgb.tobytes()
        ).hexdigest(),
        "pre_224_fp32_sha256": _tensor_sha(raw),
        "post_224_fp32_sha256": _tensor_sha(masked),
        "removed_ink_mass": removed,
        "total_ink_mass": total,
        "removed_ink_mass_fraction": removed / total if total else 0.0,
    }
    return masked, trace


class _RasterTransform:
    def __init__(self, fraction: float, raw_transform: Any) -> None:
        self.fraction = fraction
        self.raw_transform = raw_transform
        self.traces: list[dict[str, Any]] = []

    def __call__(self, image: Image.Image) -> Tensor:
        if self.fraction:
            value, trace = rasterize_sketch(image, self.fraction)
        else:
            source_rgb = image.convert("RGB")
            value = self.raw_transform(source_rgb).to(dtype=torch.float32)
            bitmap = torch.zeros((224, 224), dtype=torch.uint8)
            ink = value.mean(dim=0) < 0.9
            trace = {
                "N": int(ink.sum().item()),
                "k": 0,
                "requested_fraction": 0.0,
                "actual_fraction": 0.0,
                "mask_bitmap_sha256": _bitmap_sha(bitmap),
                "raw_source_rgb_decoded_sha256": hashlib.sha256(
                    source_rgb.tobytes()
                ).hexdigest(),
                "pre_224_fp32_sha256": _tensor_sha(value),
                "post_224_fp32_sha256": _tensor_sha(value),
                "removed_ink_mass": 0.0,
                "removed_ink_mass_fraction": 0.0,
                "total_ink_mass": float((1.0 - value.mean(dim=0)[ink]).sum().item()),
            }
        self.traces.append(trace)
        return value


@contextmanager
def _preserve_rng() -> Iterator[None]:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def _allowlisted_decoder(paths: set[Path]) -> Iterator[None]:
    original = data_package.datasets._load_rgb_image
    allowed = {path.resolve() for path in paths}

    def guarded(entry: ManifestEntry) -> Image.Image:
        if entry.path.resolve() not in allowed:
            raise PermissionError(f"Image decode outside allowlist: {entry.path}")
        return original(entry)

    data_package.datasets._load_rgb_image = guarded
    try:
        yield
    finally:
        data_package.datasets._load_rgb_image = original


def _path(value: Any) -> Path:
    return Path(str(value)).expanduser().absolute()


def _finite_tensor(value: Tensor, name: str) -> None:
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _validate_config(run: Path) -> tuple[dict[str, Any], Path, str]:
    config_path = run / "config.json"
    checkpoint_path = run / "sketchlvm_latest.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Run must contain config.json and sketchlvm_latest.pt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")
    required = {
        "source_sha256",
        "manifest_sha256",
        "data_config",
        "data_config_sha256",
        "dataset_root",
        "datasetname",
        "clip_weights",
        "clip_sha256",
        "normalization_mean",
        "normalization_std",
        "epochs",
        "seed",
        "batch_size",
        "train_sketches",
        "steps_per_epoch",
        "total_steps",
        "loss",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(
            f"Checkpoint config missing exact required fields: {sorted(missing)}"
        )
    disk_sha = sha256(config_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("config") != config:
        raise ValueError("Disk config.json does not exactly equal checkpoint config")
    if payload.get("epoch") != EPOCHS or payload.get("step") != TOTAL_STEPS:
        raise ValueError("Checkpoint must be epoch 60, step 14460")
    if config["epochs"] != EPOCHS or config["seed"] != SEED:
        raise ValueError("Checkpoint horizon or seed differs")
    if (
        config["batch_size"] != BATCH_SIZE
        or config["train_sketches"] != TRAIN_SKETCHES
        or config["steps_per_epoch"] != STEPS_PER_EPOCH
        or config["total_steps"] != TOTAL_STEPS
        or payload.get("examples_seen") != 924_000
        or config.get("method")
        not in {
            "sketchlvm_paper_category_v1",
            PAPER_METHOD,
        }
        or config["loss"]
        != {
            "triplet": "normalized_cosine_distance_hinge",
            "margin": 0.3,
            "ce_weight": 0.5,
            "negative_ce": False,
            "ce_targets": "sketch_and_positive_photo",
        }
        or config.get("prompt_count") != 3
        or config.get("native_cls_readout") is not True
        or (
            config.get("method") == PAPER_METHOD
            and (
                config.get("evaluation_profile") != "clean_only"
                or tuple(config.get("normalization_mean", ())) != PAPER_IMAGE_MEAN
                or tuple(config.get("normalization_std", ())) != PAPER_IMAGE_STD
                or config.get("preprocessing") != PAPER_PREPROCESSING
                or config.get("optimizer")
                != {
                    "name": "Adam",
                    "learning_rate": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "schedule": "constant",
                    "warmup": False,
                    "gradient_clipping": "none",
                }
                or any(
                    config.get(key) != value
                    for key, value in {
                        "examples_seen_budget": 924_000,
                        "dtype": "FP32",
                        "prompt_width": 768,
                        "prompt_placement": "after_CLS_and_49_patches_after_pos_before_preLN",
                        "layer_norm": "separate_sketch_photo_native_CLIP_LN_affines",
                        "photo_encoder": "trained_photo_prompts_and_LN_frozen_CLIP_base",
                        "frozen_original_text": True,
                        "learned_logit_scale": False,
                        "logit_scale_cap": None,
                        "text_template": "a photo of a {category}",
                    }.items()
                )
            )
        )
    ):
        raise ValueError("Checkpoint training contract differs")
    adaptation = payload.get("adaptation")
    if not isinstance(adaptation, dict) or len(adaptation) != ADAPTATION_KEYS:
        raise ValueError("Checkpoint adaptation must contain exactly 106 keys")
    for name, value in adaptation.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError("Checkpoint adaptation must be a tensor mapping")
        _finite_tensor(value, f"adaptation[{name}]")
    runtime = json.loads((run / "runtime.json").read_text(encoding="utf-8"))
    if runtime.get("status") not in {"FINAL_EVALUATION", "FINISHED_UNVERIFIED"}:
        raise ValueError("Trainer has not finished training")
    return config, checkpoint_path, disk_sha


def _validate_data(config: dict[str, Any]) -> tuple[DataConfig, dict[str, Any]]:
    data_path = _path(config["data_config"])
    if not data_path.is_file() or sha256(data_path) != config["data_config_sha256"]:
        raise ValueError("Data config path or SHA mismatch")
    data = load_data_config(data_path)
    if data.name != config["datasetname"] or data.name != "tuberlin_220_30":
        raise ValueError("Unexpected dataset name")
    if str(data.root) != str(config["dataset_root"]):
        raise ValueError("Dataset root differs from checkpoint")
    train_names = read_class_map(data.train.class_map)
    test_names = read_class_map(data.test.class_map)
    if len(train_names) != 220 or len(test_names) != 30:
        raise ValueError("Expected 220 seen and 30 unseen classes")
    if set(train_names.values()) & set(test_names.values()):
        raise ValueError("Seen and unseen class names overlap")
    paths = (
        data.train.sketch_manifest,
        data.train.photo_manifest,
        data.train.class_map,
        data.test.sketch_manifest,
        data.test.photo_manifest,
        data.test.class_map,
    )
    expected_hashes = config["manifest_sha256"]
    actual_hashes = {str(path): sha256(path) for path in paths}
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        actual_hashes
    ):
        raise ValueError("manifest_sha256 must bind exactly all six manifests/maps")
    if expected_hashes != actual_hashes:
        raise ValueError("Manifest SHA mismatch")
    train_sketch = read_manifest(data.train.sketch_manifest, data.root)
    train_photo = read_manifest(data.train.photo_manifest, data.root)
    queries = read_manifest(data.test.sketch_manifest, data.root)
    gallery = read_manifest(data.test.photo_manifest, data.root)
    if len(queries) != QUERY_COUNT or len(gallery) != GALLERY_COUNT:
        raise ValueError("Unexpected official query/gallery count")
    if len(train_sketch) != TRAIN_SKETCHES or any(
        sum(entry.label == label for entry in train_sketch) != 70
        for label in train_names
    ):
        raise ValueError("Expected 70 training sketches per seen class")
    if {entry.label for entry in train_photo} != set(train_names):
        raise ValueError("Training photo classes differ from seen class map")
    if set(entry.label for entry in queries) != set(test_names):
        raise ValueError("Query labels do not match unseen class map")
    if set(entry.label for entry in gallery) != set(test_names):
        raise ValueError("Gallery labels do not match unseen class map")
    if any(
        sum(entry.label == label for entry in queries) != 80 for label in test_names
    ):
        raise ValueError("Official queries must contain 80 per unseen class")
    all_entries = train_sketch + train_photo + queries + gallery
    all_paths = [entry.path.resolve() for entry in all_entries]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError(
            "Official and training manifest paths must be disjoint and unique"
        )
    if any(not path.is_file() for path in all_paths):
        raise FileNotFoundError("Manifest contains missing image")
    # Official train/test numeric IDs restart at zero; semantic names are disjoint.
    return data, {
        "queries": queries,
        "gallery": gallery,
        "test_names": test_names,
        "manifest_sha256": actual_hashes,
        "data_config_sha256": sha256(data_path),
        "data_config_path": str(data_path),
    }


def _load_model(
    run: Path,
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], Any, str]:
    source_map = config["source_sha256"]
    source_key = "src/spica/models/sketchlvm.py"
    if not isinstance(source_map, dict) or source_key not in source_map:
        raise ValueError("source_sha256 must bind the archived SketchLVM model")
    archive = (run / "source").resolve()
    for relative, digest in source_map.items():
        path = Path(relative)
        saved = archive / path
        if (
            path.is_absolute()
            or ".." in path.parts
            or not saved.resolve().is_relative_to(archive)
        ):
            raise ValueError(f"Unsafe source archive path: {relative}")
        if not saved.is_file() or sha256(saved) != digest:
            raise ValueError(f"Archived source SHA mismatch: {relative}")
    expected_source = source_map[source_key]
    if not isinstance(expected_source, str):
        raise ValueError("Archived source SHA must be a string")
    source = run / "source" / "src" / "spica" / "models" / "sketchlvm.py"
    if not source.is_file() or sha256(source) != expected_source:
        raise ValueError("Archived SketchLVM source SHA mismatch")
    archived = load_archived_module(source, "archived_sketchlvm_evaluator")
    weights = _path(config["clip_weights"])
    if weights.suffix != ".safetensors" or not weights.is_file():
        raise ValueError("clip_weights must be an existing .safetensors file")
    if (
        sha256(weights) != config["clip_sha256"]
        or config["clip_sha256"] != EXPECTED_CLIP_SHA256
    ):
        raise ValueError("CLIP weight SHA mismatch")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    bundle = load_frozen_clip(pretrained=str(weights), device=device)
    normalize = bundle.transform.transforms[-1]
    native_mean = tuple(float(x) for x in normalize.mean)
    native_std = tuple(float(x) for x in normalize.std)
    if config.get("method") == PAPER_METHOD:
        image_mean, image_std = PAPER_IMAGE_MEAN, PAPER_IMAGE_STD
    else:
        image_mean, image_std = native_mean, native_std
    if (
        tuple(config["normalization_mean"]) != image_mean
        or tuple(config["normalization_std"]) != image_std
    ):
        raise ValueError("Checkpoint normalization differs from selected preprocessing")
    model = archived.SketchLVM(bundle.encoder.model.visual, image_mean, image_std).to(
        device
    )
    model.eval()
    expected = model.adaptation_state_dict()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adaptation = payload["adaptation"]
    if (
        len(expected) != ADAPTATION_KEYS
        or sum(x.numel() for x in expected.values()) != ADAPTATION_PARAMETERS
    ):
        raise ValueError("Archived SketchLVM adaptation contract differs")
    if set(adaptation) != set(expected):
        raise ValueError("Checkpoint adaptation keys differ from model trainable keys")
    for name, value in expected.items():
        saved = adaptation[name]
        if saved.shape != value.shape or saved.dtype != value.dtype:
            raise ValueError(f"Adaptation shape/dtype mismatch: {name}")
    restored = model.load_state_dict(adaptation, strict=False)
    allowed_missing = set(model.state_dict()) - set(adaptation)
    if restored.unexpected_keys or set(restored.missing_keys) != allowed_missing:
        raise ValueError("Restore missing keys outside frozen base")
    actual = model.adaptation_state_dict()
    for name in expected:
        if not torch.equal(actual[name], adaptation[name].cpu()):
            raise ValueError(f"Adaptation restore differs from checkpoint: {name}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Restored model contains gradients")
    return (
        model,
        {"source": str(source), "source_sha256": expected_source},
        bundle,
        sha256(source),
    )


def _validate_features(encoded: EncodedRetrievalSet, count: int, name: str) -> None:
    if encoded.embeddings.shape != (count, 512):
        raise ValueError(f"{name} must contain [count, 512] CLIP embeddings")
    if len(encoded.paths) != count or encoded.labels.numel() != count:
        raise ValueError(f"{name} count mismatch")
    _finite_tensor(encoded.embeddings, name)
    norms = encoded.embeddings.float().norm(dim=1)
    if (
        not torch.isfinite(norms).all()
        or torch.any(norms <= 0)
        or torch.any((norms - 1).abs() > 2e-3)
    ):
        raise ValueError(f"{name} embeddings fail normalized boundary check")
    if len(set(encoded.paths)) != count:
        raise ValueError(f"{name} paths are not unique")


def _metrics(
    queries: EncodedRetrievalSet,
    gallery: EncodedRetrievalSet,
    device: torch.device,
    output: Path,
    *,
    include_p100: bool = False,
) -> tuple[dict[str, float], Tensor, Tensor, Tensor, Tensor | None]:
    _validate_features(queries, QUERY_COUNT, "queries")
    _validate_features(gallery, GALLERY_COUNT, "gallery")
    evaluation, ap200 = retrieval_stats(
        queries.embeddings, queries.labels, gallery.embeddings, gallery.labels, device
    )
    top = evaluation.top_indices
    relevant = gallery.labels[top].eq(queries.labels[:, None])
    p200 = relevant[:, : min(200, gallery.labels.numel())].double().mean(dim=1)
    p100 = (
        relevant[:, : min(100, gallery.labels.numel())].double().mean(dim=1)
        if include_p100
        else None
    )
    full_ap = evaluation.average_precision_per_query.double()
    ap200 = ap200.double()
    p200 = p200.double()
    for value, name in (
        (full_ap, "full AP"),
        (ap200, "AP200"),
        (p200, "P200"),
        *([(p100, "P100")] if p100 is not None else []),
    ):
        if (
            not torch.isfinite(value).all()
            or torch.any(value < 0)
            or torch.any(value > 1)
        ):
            raise ValueError(f"{name} fails metric boundary check")
    scores = evaluation.top_scores.float()
    _finite_tensor(scores, "top scores")
    payload = {
        "full_ap": full_ap.cpu(),
        "ap200": ap200.cpu(),
        "p200": p200.cpu(),
        **({"p100": p100.cpu()} if p100 is not None else {}),
        "top200_indices": top.cpu(),
        "top200_scores": scores.cpu(),
    }
    torch.save(payload, output / "per_query.pt")
    np.savez(
        output / "per_query.npz",
        **{key: value.numpy() for key, value in payload.items()},
    )
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("query_index", "path", "label", "full_AP", "AP200", "P200")
            + (("P100",) if include_p100 else ())
        )
        for index, path in enumerate(queries.paths):
            writer.writerow(
                (
                    index,
                    path,
                    int(queries.labels[index]),
                    float(full_ap[index]),
                    float(ap200[index]),
                    float(p200[index]),
                    *((float(p100[index]),) if p100 is not None else ()),
                )
            )
    return (
        {
            "full_mAP": full_ap.mean().item(),
            "mAP@200_min_relevant_k": ap200.mean().item(),
            "P@200": p200.mean().item(),
            **({"P@100": p100.mean().item()} if p100 is not None else {}),
        },
        full_ap,
        ap200,
        p200,
        p100,
    )


def _class_csv(
    output: Path,
    conditions: dict[str, dict[str, Tensor]],
    names: dict[int, str],
    labels: Tensor,
    *,
    include_p100: bool = False,
) -> None:
    clean = conditions["clean"]
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "condition",
                "class_id",
                "class_name",
                "query_count",
                "full_mAP",
                "mAP@200_min_relevant_k",
                "P@200",
                *(("P@100",) if include_p100 else ()),
                "delta_full_AP",
                "delta_AP200",
                "delta_P200",
                *(("delta_P100",) if include_p100 else ()),
            )
        )
        for condition, values in conditions.items():
            for class_id in sorted(names):
                selected = labels == class_id
                delta = {
                    key: values[key] - clean[key] for key in ("full", "ap200", "p200")
                }
                if include_p100:
                    delta["p100"] = values["p100"] - clean["p100"]
                writer.writerow(
                    (
                        condition,
                        class_id,
                        names[class_id],
                        int(selected.sum()),
                        float(values["full"][selected].mean()),
                        float(values["ap200"][selected].mean()),
                        float(values["p200"][selected].mean()),
                        *(
                            (float(values["p100"][selected].mean()),)
                            if include_p100
                            else ()
                        ),
                        float(delta["full"][selected].mean()),
                        float(delta["ap200"][selected].mean()),
                        float(delta["p200"][selected].mean()),
                        *(
                            (float(delta["p100"][selected].mean()),)
                            if include_p100
                            else ()
                        ),
                    )
                )


def _run(run: Path, output: Path) -> dict[str, Any]:
    config, checkpoint, config_digest = _validate_config(run)
    data, info = _validate_data(config)
    source_before = source_files()
    critical = (
        "src/spica/evaluate_sketchlvm_raster.py",
        "src/spica/train_sketchlvm.py",
        "src/spica/diagnose_student_semantics.py",
        "src/spica/train_student.py",
        "src/spica/models/clip.py",
        "src/spica/config/data.py",
        "src/spica/data/datasets.py",
        "src/spica/data/manifest.py",
        "src/spica/data/transforms.py",
        "src/spica/evaluation/embeddings.py",
        "src/spica/evaluation/metrics.py",
        "pyproject.toml",
        "uv.lock",
    )
    if any(
        source_before.get(name) != config["source_sha256"].get(name)
        for name in critical
    ):
        raise ValueError(
            "Current evaluation dependencies differ from training snapshot"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("SketchLVM raster evaluation requires CUDA")
    device = torch.device("cuda")
    model, source_info, bundle, source_digest = _load_model(
        run, config, checkpoint, device
    )
    paper_clean = config.get("method") == PAPER_METHOD
    raw_transform = (
        _paper_clean_transform() if paper_clean else _raw_transform(bundle.transform)
    )
    official_paths = {
        entry.path.resolve() for entry in info["queries"] + info["gallery"]
    }
    physical_hashes = {str(path): sha256(path) for path in sorted(official_paths)}
    evaluator_path = Path(__file__).absolute()
    evaluator_digest = sha256(evaluator_path)
    checkpoint_digest = sha256(checkpoint)
    before = {
        "model": state_hash(model),
        "rng": rng_hash(),
        "modes": {name: module.training for name, module in model.named_modules()},
    }
    runtime: dict[str, Any] = {"status": "RUNNING", "stages": {}}
    _write_json(output / "runtime.json", runtime)
    with torch.inference_mode(), _allowlisted_decoder(official_paths):
        gallery = encode_entries(
            model.photo, info["gallery"], raw_transform, official_paths, BATCH_SIZE
        )
        _validate_features(gallery, GALLERY_COUNT, "gallery")
        save_encoded_retrieval_set(gallery, output / "gallery.pt")
        runtime["stages"]["gallery"] = {"finished": time.time()}
        _write_json(output / "runtime.json", runtime)
        conditions = ("clean",) if paper_clean else CONDITIONS
        condition_values: dict[str, dict[str, Tensor]] = {}
        result: dict[str, Any] = {}
        trace_rows: dict[str, list[dict[str, Any]]] = {}
        query_reference: EncodedRetrievalSet | None = None
        clean_traces: list[dict[str, Any]] | None = None
        for condition in conditions:
            fraction = (
                0.0 if condition == "clean" else int(condition.rsplit("_", 1)[1]) / 100
            )
            transform = _RasterTransform(fraction, raw_transform)
            queries = encode_entries(
                model.sketch, info["queries"], transform, official_paths, BATCH_SIZE
            )
            _validate_features(queries, QUERY_COUNT, f"{condition} queries")
            if query_reference is not None and (
                queries.paths != query_reference.paths
                or not torch.equal(queries.labels, query_reference.labels)
            ):
                raise AssertionError(
                    "Query path/label ordering changed across conditions"
                )
            query_reference = queries
            condition_dir = output / condition
            condition_dir.mkdir()
            save_encoded_retrieval_set(queries, condition_dir / "queries.pt")
            metrics, full, ap200, p200, p100 = _metrics(
                queries,
                gallery,
                device,
                condition_dir,
                include_p100=paper_clean,
            )
            condition_values[condition] = {"full": full, "ap200": ap200, "p200": p200}
            if p100 is not None:
                condition_values[condition]["p100"] = p100
            result[condition] = metrics
            rows = []
            for index, (entry, trace) in enumerate(
                zip(info["queries"], transform.traces, strict=True)
            ):
                row = {
                    "condition": condition,
                    "query_index": index,
                    "path": str(entry.path),
                    "class_id": entry.label,
                    "class_name": info["test_names"][entry.label],
                    "physical_source_file_sha256": physical_hashes[
                        str(entry.path.resolve())
                    ],
                    **trace,
                }
                rows.append(row)
            trace_rows[condition] = rows
            if clean_traces is None:
                clean_traces = rows
            else:
                for left, right in zip(clean_traces, rows, strict=True):
                    for key in (
                        "path",
                        "class_id",
                        "raw_source_rgb_decoded_sha256",
                        "physical_source_file_sha256",
                        "pre_224_fp32_sha256",
                    ):
                        if left[key] != right[key]:
                            raise AssertionError(f"Shared query trace changed: {key}")
            runtime["stages"][condition] = {"finished": time.time()}
            _write_json(output / "runtime.json", runtime)
        labels = query_reference.labels
        _class_csv(
            output,
            condition_values,
            info["test_names"],
            labels,
            include_p100=paper_clean,
        )
        _write_json(output / "mask_trace.json", trace_rows)
        with (output / "mask_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = list(trace_rows["clean"][0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for condition in conditions:
                writer.writerows(trace_rows[condition])
        clean = condition_values["clean"]["full"]
        for condition in conditions[1:]:
            paired: dict[str, Any] = {}
            for metric, values in condition_values[condition].items():
                difference = values - condition_values["clean"][metric]
                suffix = {"full": "full_AP", "ap200": "AP200", "p200": "P200"}[metric]
                paired.update(
                    {
                        f"mean_query_delta_{suffix}": difference.mean().item(),
                        f"median_query_delta_{suffix}": torch.quantile(
                            difference, 0.5
                        ).item(),
                        f"wins_{suffix}": int((difference > 1e-6).sum()),
                        f"losses_{suffix}": int((difference < -1e-6).sum()),
                        f"ties_{suffix}": int((difference.abs() <= 1e-6).sum()),
                    }
                )
            paired["masked_retention"] = (
                condition_values[condition]["full"].mean().item() / clean.mean().item()
            )
            result[condition].update(paired)
    after = {
        "model": state_hash(model),
        "rng": rng_hash(),
        "modes": {name: module.training for name, module in model.named_modules()},
    }
    if before != after or any(
        parameter.grad is not None for parameter in model.parameters()
    ):
        raise AssertionError(
            "Model state, RNG, mode, or gradients changed during evaluation"
        )
    if (
        sha256(evaluator_path) != evaluator_digest
        or sha256(checkpoint.with_name("config.json")) != config_digest
        or sha256(checkpoint) != checkpoint_digest
        or sha256(Path(source_info["source"])) != source_digest
        or source_files() != source_before
        or any(
            sha256(Path(path)) != digest
            for path, digest in config["manifest_sha256"].items()
        )
        or sha256(_path(config["data_config"])) != config["data_config_sha256"]
        or sha256(_path(config["clip_weights"])) != config["clip_sha256"]
    ):
        raise AssertionError(
            "Source/config/evaluator/data/CLIP changed during evaluation"
        )
    provenance = {
        "status": "FINISHED_UNVERIFIED",
        "config": config,
        "config_sha256": config_digest,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "source": source_info,
        "source_sha256": source_digest,
        "evaluator_sha256_before_after": [evaluator_digest, sha256(evaluator_path)],
        "evaluation_source_sha256": source_before,
        "data": {
            "data_config": info["data_config_path"],
            "data_config_sha256": info["data_config_sha256"],
            "manifest_sha256": info["manifest_sha256"],
            "dataset_root": str(data.root),
        },
        "physical_source_file_sha256": physical_hashes,
        "conditions": list(conditions),
        "model_state_before_after": [before["model"], after["model"]],
        "rng_before_after": [before["rng"], after["rng"]],
    }
    _write_json(output / "provenance.json", provenance)
    artifacts = [
        "gallery.pt",
        "mask_trace.json",
        "mask_trace.csv",
        "per_class.csv",
        "provenance.json",
    ]
    artifacts += [
        f"{condition}/{name}"
        for condition in conditions
        for name in ("queries.pt", "per_query.pt", "per_query.npz", "per_query.csv")
    ]
    result = {
        "status": "FINISHED_UNVERIFIED",
        "artifacts_sha256": {name: sha256(output / name) for name in artifacts},
        **result,
    }
    _write_json(output / "summary.json", result)
    _write_json(output / "artifacts_sha256.json", result["artifacts_sha256"])
    runtime["status"] = "FINISHED_UNVERIFIED"
    _write_json(output / "runtime.json", runtime)
    return result


def evaluate(run: Path, output: Path) -> dict[str, Any]:
    """Run one fresh, non-overwriting final evaluation."""
    run = run.expanduser().absolute()
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output}")
    output.mkdir(parents=True)
    _write_json(output / "runtime.json", {"status": "INITIALIZING"})
    with _preserve_rng():
        try:
            return _run(run, output)
        except BaseException as error:
            _write_json(
                output / "failure.json",
                {"status": "FAILED_NO_RETRY", "error": repr(error)},
            )
            _write_json(
                output / "runtime.json",
                {"status": "FAILED_NO_RETRY", "error": repr(error)},
            )
            raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.run, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
