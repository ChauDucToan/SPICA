"""Train paper-category SketchLVM adaptation with final-only raster evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms

from .config.data import DataConfig, load_data_config
from .data.loaders import build_retrieval_train_loader
from .data.manifest import read_class_map, read_manifest
from .data.transforms import build_clip_eval_transform
from .models.clip import load_frozen_clip
from .models.sketchlvm import SketchLVM
from .train_student import ROOT, sha256, source_files, write_json

SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
STEPS_PER_EPOCH = 241
TOTAL_STEPS = 14_460
TRAIN_SKETCHES = 15_400
EXAMPLES_SEEN_BUDGET = 924_000
LEARNING_RATE = 1e-5
TRIPLET_MARGIN = 0.3
CE_WEIGHT = 0.5
EXPECTED_CLIP_SHA256 = (
    "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
)

EmbeddingBatch = Float[Tensor, "batch embedding"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _raw_transform() -> transforms.Compose:
    return transforms.Compose(build_clip_eval_transform().transforms[:-1])


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}")


def _class_indices(labels: Tensor, mapping: dict[int, int]) -> Tensor:
    return torch.tensor(
        [mapping[int(label)] for label in labels.cpu()],
        dtype=torch.long,
        device=labels.device,
    )


def _losses(
    sketch: EmbeddingBatch,
    positive: EmbeddingBatch,
    negative: EmbeddingBatch,
    labels: Tensor,
    prototypes: Tensor,
    label_indices: dict[int, int],
    scale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cosine hinge plus sketch/positive-photo CE; no negative-photo CE."""
    sketch = F.normalize(sketch, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    triplet = F.relu(
        TRIPLET_MARGIN
        + (sketch * negative).sum(dim=-1)
        - (sketch * positive).sum(dim=-1)
    ).mean()
    targets = _class_indices(labels, label_indices)
    text = F.normalize(prototypes, dim=-1).T
    sketch_ce = F.cross_entropy(scale * (sketch @ text), targets)
    photo_ce = F.cross_entropy(scale * (positive @ text), targets)
    loss = triplet + CE_WEIGHT * (sketch_ce + photo_ce)
    return loss, triplet, sketch_ce, photo_ce


def _frozen_state_sha(model: torch.nn.Module) -> str:
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name not in trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(
    output: Path,
    model: SketchLVM,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    step: int,
    examples_seen: int,
) -> None:
    payload: dict[str, Any] = {
        "adaptation": model.adaptation_state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "examples_seen": examples_seen,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = output / "sketchlvm_latest.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, output / "sketchlvm_latest.pt")


def _text_prototypes(
    bundle: Any, class_names: dict[int, str], output: Path
) -> tuple[Tensor, str, list[str]]:
    labels = sorted(class_names)
    names = [class_names[label].replace("_", " ") for label in labels]
    tokens = bundle.tokenizer([f"a photo of a {name}" for name in names]).to(
        bundle.encoder.device
    )
    with torch.no_grad():
        prototypes = bundle.encoder.encode_text(tokens).float()
    _finite(prototypes, "text prototypes")
    path = output / "text_prototypes.pt"
    torch.save({"labels": labels, "names": names, "embeddings": prototypes.cpu()}, path)
    return prototypes, sha256(path), names


def _validate_protocol(data: DataConfig, data_config_path: Path) -> dict[str, Any]:
    if data.name != "tuberlin_220_30":
        raise ValueError("SketchLVM run requires official TU-Berlin 220/30")
    train_names = read_class_map(data.train.class_map)
    test_names = read_class_map(data.test.class_map)
    if len(train_names) != 220 or len(test_names) != 30:
        raise ValueError("Expected 220 train classes and 30 test classes")
    semantic_train = {
        name.replace("_", " ").casefold() for name in train_names.values()
    }
    semantic_test = {name.replace("_", " ").casefold() for name in test_names.values()}
    if semantic_train & semantic_test:
        raise ValueError("Train and test class semantics overlap")

    train_sketch = read_manifest(data.train.sketch_manifest, data.root)
    train_photo = read_manifest(data.train.photo_manifest, data.root)
    test_sketch = read_manifest(data.test.sketch_manifest, data.root)
    test_photo = read_manifest(data.test.photo_manifest, data.root)
    if len(train_sketch) != TRAIN_SKETCHES:
        raise ValueError(f"Expected {TRAIN_SKETCHES} training sketches")
    if len(test_sketch) != 2_400 or len(test_photo) != 27_989:
        raise ValueError("Unexpected official test manifest counts")
    if any(
        sum(entry.label == label for entry in train_sketch) != 70
        for label in train_names
    ):
        raise ValueError("Official training requires 70 sketches per seen class")
    if {entry.label for entry in train_sketch} != set(train_names):
        raise ValueError("Training sketch labels do not match class map")
    if {entry.label for entry in train_photo} != set(train_names):
        raise ValueError("Training photo labels do not match class map")
    if {entry.label for entry in test_sketch} != set(test_names) or {
        entry.label for entry in test_photo
    } != set(test_names):
        raise ValueError("Test manifest labels do not match class map")
    if any(
        sum(entry.label == label for entry in test_sketch) != 80 for label in test_names
    ):
        raise ValueError("Official test must contain 80 sketches per class")

    train_entries = (*train_sketch, *train_photo)
    test_entries = (*test_sketch, *test_photo)
    train_paths = {entry.path.resolve() for entry in train_entries}
    test_paths = {entry.path.resolve() for entry in test_entries}
    if len(train_paths) != len(train_entries) or len(test_paths) != len(test_entries):
        raise ValueError("Manifest contains duplicate physical image paths")
    if train_paths & test_paths:
        raise ValueError("Train and test image paths overlap")
    for path in train_paths | test_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_paths = [
        data.train.sketch_manifest,
        data.train.photo_manifest,
        data.train.class_map,
        data.test.sketch_manifest,
        data.test.photo_manifest,
        data.test.class_map,
    ]
    manifests = {str(path): sha256(path) for path in manifest_paths}
    return {
        "dataset_root": str(data.root),
        "datasetname": data.name,
        "data_config": str(data_config_path),
        "data_config_sha256": sha256(data_config_path),
        "manifest_sha256": manifests,
        "train_sketches": len(train_sketch),
        "train_photos": len(train_photo),
        "test_queries": len(test_sketch),
        "test_gallery": len(test_photo),
        "train_classes": len(train_names),
        "test_classes": len(test_names),
    }


def _guard_inputs(
    source: dict[str, str], manifests: dict[str, str], clip_path: Path, clip_digest: str
) -> None:
    if source_files() != source:
        raise RuntimeError("Source files changed; refusing evaluation")
    if any(sha256(Path(path)) != digest for path, digest in manifests.items()):
        raise RuntimeError("Manifest files changed; refusing evaluation")
    if sha256(clip_path) != clip_digest:
        raise RuntimeError("CLIP weights changed; refusing evaluation")


def _final_evaluate(output: Path) -> dict[str, Any]:
    from .evaluate_sketchlvm_raster import evaluate

    return evaluate(run=output, output=output / "raster_test")


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if args.wandb_mode not in {"online", "disabled"}:
        raise ValueError("Unsupported W&B mode")

    data_config_path = args.data_config.expanduser().absolute()
    data = load_data_config(data_config_path)
    protocol = _validate_protocol(data, data_config_path)
    clip_path = args.clip_weights.expanduser().absolute()
    if not clip_path.is_file():
        raise FileNotFoundError(clip_path)
    clip_digest = sha256(clip_path)
    if clip_digest != EXPECTED_CLIP_SHA256:
        raise ValueError("CLIP weights do not match pinned OpenAI ViT-B/32 QuickGELU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production SketchLVM training")

    output.mkdir(parents=True)
    seed_everything(SEED)
    source = source_files()
    for name in source:
        archive = output / "source" / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, archive)
    class_names = read_class_map(data.train.class_map)
    manifests = protocol["manifest_sha256"]
    write_json(output / "runtime.json", {"status": "INITIALIZING"})
    run = None

    try:
        bundle = load_frozen_clip(pretrained=str(clip_path), device="cuda")
        normalize = bundle.transform.transforms[-1]
        if not isinstance(normalize, transforms.Normalize):
            raise TypeError("Expected native CLIP Normalize transform")
        model = SketchLVM(
            bundle.encoder.model.visual,
            tuple(float(value) for value in normalize.mean),
            tuple(float(value) for value in normalize.std),
        ).to(bundle.encoder.device)
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if sum(parameter.numel() for parameter in trainable) != 84_480:
            raise RuntimeError("Unexpected SketchLVM adaptation parameter count")
        frozen_before = _frozen_state_sha(model)
        optimizer = torch.optim.Adam(
            trainable,
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        loader = build_retrieval_train_loader(
            data,
            _raw_transform(),
            _raw_transform(),
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        if len(loader) != STEPS_PER_EPOCH or len(loader.dataset) != TRAIN_SKETCHES:
            raise RuntimeError("Unexpected training loader geometry")
        prototypes, prototype_sha, prototype_names = _text_prototypes(
            bundle, class_names, output
        )
        label_indices = {
            label: index for index, label in enumerate(sorted(class_names))
        }
        scale = bundle.encoder.model.logit_scale.exp().detach().float()
        _finite(scale, "pretrained logit scale")
        device = bundle.encoder.device
        epochs = EPOCHS
        total_steps = epochs * STEPS_PER_EPOCH
        examples_seen_budget = epochs * TRAIN_SKETCHES
        config: dict[str, Any] = {
            "method": "sketchlvm_paper_category_v1",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip(),
            "source_sha256": source,
            **protocol,
            "clip_weights": str(clip_path),
            "clip_sha256": clip_digest,
            "clip_model": "ViT-B-32-quickgelu",
            "normalization_mean": list(normalize.mean),
            "normalization_std": list(normalize.std),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": epochs,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "total_steps": total_steps,
            "train_sketches": TRAIN_SKETCHES,
            "examples_seen_budget": examples_seen_budget,
            "optimizer": {
                "name": "Adam",
                "learning_rate": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "schedule": "constant",
                "warmup": False,
                "gradient_clipping": "none",
            },
            "dtype": "FP32",
            "loss": {
                "triplet": "normalized_cosine_distance_hinge",
                "margin": TRIPLET_MARGIN,
                "ce_weight": CE_WEIGHT,
                "negative_ce": False,
                "ce_targets": "sketch_and_positive_photo",
            },
            "prompt_count": 3,
            "prompt_width": 768,
            "prompt_initialization": "randn_std_1_public_nonpaper_fixed",
            "prompt_placement": "after_CLS_and_49_patches_after_pos_before_preLN",
            "layer_norm": "separate_sketch_photo_native_CLIP_LN_affines",
            "native_cls_readout": True,
            "preserve_positions": True,
            "photo_encoder": "trained_photo_prompts_and_LN_frozen_CLIP_base",
            "text_template": "a photo of a {category}",
            "text_underscore_replacement": "spaces",
            "prototype_bank_labels": sorted(class_names),
            "prototype_bank_names": prototype_names,
            "prototype_bank_sha256": prototype_sha,
            "frozen_original_text": True,
            "pretrained_logit_scale_exp": float(scale),
            "learned_logit_scale": False,
            "logit_scale_cap": None,
            "final_only": True,
            "final_evaluation": True,
            "honesty": "paper-category reimplementation; not exact due sampler, preprocess, schedule, and unspecified initialization details",
            "official_test_previously_observed": True,
        }
        write_json(output / "config.json", config)
        if args.wandb_mode == "online":
            import wandb

            run = wandb.init(
                project="sbir",
                entity="a-cctest05187-erd",
                name="SketchLVM-paper-category-TU-60epochs",
                config=config,
                dir=str(output),
                settings=wandb.Settings(
                    x_disable_stats=True,
                    x_disable_meta=True,
                    disable_code=True,
                    disable_git=True,
                    x_save_requirements=False,
                    console="off",
                ),
            )
            run.define_metric("progress_percent")
            run.define_metric("test/clean/*", step_metric="progress_percent")
            write_json(output / "wandb.json", {"id": run.id, "url": run.url})

        step = 0
        examples_seen = 0
        write_json(output / "runtime.json", {"status": "TRAINING", "step": 0})
        with (output / "localtraining.jsonl").open("x", encoding="utf-8") as history:
            for epoch in range(1, epochs + 1):
                model.train()
                totals = {
                    "loss": 0.0,
                    "triplet": 0.0,
                    "sketch_ce": 0.0,
                    "photo_ce": 0.0,
                }
                for batch in loader:
                    images = batch["sketch"].to(device, non_blocking=True)
                    positive_images = batch["positive_photo"].to(
                        device, non_blocking=True
                    )
                    negative_images = batch["negative_photo"].to(
                        device, non_blocking=True
                    )
                    labels = batch["label"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    sketch_embedding = model.sketch(images)
                    positive_embedding = model.photo(positive_images)
                    negative_embedding = model.photo(negative_images)
                    loss, triplet, sketch_ce, photo_ce = _losses(
                        sketch_embedding,
                        positive_embedding,
                        negative_embedding,
                        labels,
                        prototypes,
                        label_indices,
                        scale,
                    )
                    _finite(loss, "loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float("inf"), error_if_nonfinite=True
                    )
                    for parameter in trainable:
                        if parameter.grad is None:
                            raise RuntimeError("Trainable parameter has no gradient")
                        _finite(parameter.grad, "gradient")
                    optimizer.step()
                    for parameter in trainable:
                        _finite(parameter, "parameter")
                    step += 1
                    count = len(labels)
                    examples_seen += count
                    totals["loss"] += loss.item() * count
                    totals["triplet"] += triplet.item() * count
                    totals["sketch_ce"] += sketch_ce.item() * count
                    totals["photo_ce"] += photo_ce.item() * count
                    if step == 1 or step % 50 == 0:
                        write_json(
                            output / "runtime.json",
                            {
                                "status": "TRAINING",
                                "epoch": epoch,
                                "step": step,
                                "total_steps": total_steps,
                                "examples_seen": examples_seen,
                            },
                        )
                row = {
                    "epoch": epoch,
                    "step": step,
                    "examples_seen": examples_seen,
                    **{
                        name: value / len(loader.dataset)
                        for name, value in totals.items()
                    },
                }
                history.write(json.dumps(row) + "\n")
                history.flush()
                _save_checkpoint(
                    output, model, optimizer, config, epoch, step, examples_seen
                )
                write_json(output / "runtime.json", {"status": "TRAINING", **row})

        if step != total_steps or examples_seen != examples_seen_budget:
            raise RuntimeError(
                "Training did not reach the required step/example budget"
            )
        frozen_after = _frozen_state_sha(model)
        if frozen_before != frozen_after:
            raise RuntimeError("Frozen CLIP base changed during training")
        _guard_inputs(source, manifests, clip_path, clip_digest)
        del loader, optimizer, prototypes, model, bundle
        gc.collect()
        torch.cuda.empty_cache()
        write_json(
            output / "runtime.json",
            {"status": "FINAL_EVALUATION", "epoch": EPOCHS, "step": TOTAL_STEPS},
        )
        raster = _final_evaluate(output)
        _guard_inputs(source, manifests, clip_path, clip_digest)
        raster_path = output / "raster_test" / "summary.json"
        config_path = output / "config.json"
        checkpoint_path = output / "sketchlvm_latest.pt"
        config_on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config_on_disk:
            raise RuntimeError("Checkpoint config differs from config.json")
        clean = raster["clean"]
        result = {
            "status": "FINISHED_UNVERIFIED",
            "epoch": EPOCHS,
            "step": TOTAL_STEPS,
            "examples_seen": examples_seen,
            "checkpoint_sha256": sha256(checkpoint_path),
            "config_sha256": sha256(config_path),
            "raster_result_path": str(raster_path),
            "raster_result_sha256": sha256(raster_path),
            "raster": raster,
            "scope": "clean_and_raster_final_only;paper_category_reimplementation;previously_seen_official_test;unverified",
        }
        write_json(output / "result.json", result)
        if run is not None:
            run.log(
                {
                    "progress_percent": 100.0,
                    "test/clean/full_mAP": clean["full_mAP"],
                    "test/clean/mAP@200_min_relevant_k": clean[
                        "mAP@200_min_relevant_k"
                    ],
                    "test/clean/P@200": clean["P@200"],
                },
                step=TOTAL_STEPS,
                commit=True,
            )
            run.finish()
        write_json(
            output / "runtime.json",
            {"status": "FINISHED_UNVERIFIED", "epoch": EPOCHS, "step": TOTAL_STEPS},
        )
        return result
    except BaseException as error:
        write_json(
            output / "runtime.json", {"status": "FAILED_NO_RETRY", "error": repr(error)}
        )
        if run is not None:
            run.finish(exit_code=1)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config", type=Path, default=ROOT / "configs/data/tuberlin_220_30.yaml"
    )
    parser.add_argument("--clip-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
