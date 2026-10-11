"""Train MambaSketchStem while keeping CLIP ViT body completely frozen.

No KD, no teacher LoRA adapters, no visual prompts.
Sketch features pass through trainable MambaSketchStem to replace CLIP patch projection,
then enter the frozen 12-layer CLIP ViT body for zero-shot photo retrieval.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from .config.data import load_data_config
from .data.datasets import RetrievalEvalDataset
from .data.loaders import build_retrieval_train_loader
from .data.manifest import read_class_map, read_manifest
from .data.transforms import build_clip_eval_transform
from .evaluation.embeddings import encode_retrieval_loader, save_encoded_retrieval_set
from .evaluation.metrics import evaluate_category_retrieval
from .models.clip import load_frozen_clip
from .models.sketch_stem import CLIPWithSketchStem

QueryEmbeddings = Float[Tensor, "batch embedding"]
PhotoEmbeddings = Float[Tensor, "photos embedding"]
QueryLabels = Int[Tensor, "batch"]
PhotoLabels = Int[Tensor, "photos"]
Scalar = Float[Tensor, ""]

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 20
SEED = 42
BATCH_SIZE = 32
TEMPERATURE = 0.07
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MPQ_SUMMARY_SHA256 = "afb927f7142aafe98bf57c742a034d8478bdb2586880a53f6a3c5c15b2dfba42"


def multi_positive_loss(
    query: QueryEmbeddings,
    photos: PhotoEmbeddings,
    labels: QueryLabels,
    photo_labels: PhotoLabels,
) -> Scalar:
    log_prob = F.log_softmax(query @ photos.T / TEMPERATURE, dim=-1)
    positive = labels[:, None] == photo_labels[None, :]
    return (-(log_prob * positive).sum(dim=1) / positive.sum(dim=1)).mean()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_files():
    names = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    return {
        name: sha256(ROOT / name)
        for name in sorted(set(names))
        if (
            Path(name).suffix in {".py", ".yaml", ".toml", ".lock", ".md"}
            or Path(name).name == "LICENSE"
        )
        and not name.startswith("outputs/")
    }


def learning_rate_factor(step, total_steps):
    warmup = max(1, round(0.05 * total_steps))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))


@torch.inference_mode()
def final_retrieval(model, teacher, data, sketch_transform, photo_transform, output):
    encoded = []
    for encoder, manifest, transform, name in (
        (model, data.test.sketch_manifest, sketch_transform, "queries"),
        (teacher, data.test.photo_manifest, photo_transform, "gallery"),
    ):
        dataset = RetrievalEvalDataset(read_manifest(manifest, data.root), transform)
        loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
        features = encode_retrieval_loader(encoder, loader)
        save_encoded_retrieval_set(features, output / f"{name}.pt")
        encoded.append(features)
    queries, gallery = encoded
    result = evaluate_category_retrieval(
        queries,
        gallery,
        precision_at_k=(200,),
        top_k=200,
        query_chunk_size=256,
        device=model.device,
    )
    relevant = (gallery.labels[result.top_indices] == queries.labels[:, None]).double()
    precision = relevant.cumsum(dim=1) / torch.arange(1, 201, dtype=torch.float64)
    positives = torch.bincount(gallery.labels)[queries.labels].clamp(max=200)
    ap200 = (precision * relevant).sum(dim=1) / positives
    torch.save(
        {
            "full_ap": result.average_precision_per_query,
            "ap200": ap200,
            "top200_indices": result.top_indices,
        },
        output / "per_query.pt",
    )
    return {
        "full_mAP": result.metrics.mean_average_precision,
        "P@200": result.metrics.precision_at_k[200],
        "mAP@200_min_relevant_k": ap200.mean().item(),
        "query_count": len(queries.labels),
        "gallery_count": len(gallery.labels),
    }


def train(args: argparse.Namespace):
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    epochs: int = getattr(args, "epochs", None) or EPOCHS
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    data = load_data_config(args.data_config)
    if data.name != "tuberlin_220_30":
        raise ValueError("This experiment is official TU-Berlin only")

    seen = read_class_map(data.train.class_map)
    unseen = read_class_map(data.test.class_map)
    if (
        len(seen) != 220
        or len(unseen) != 30
        or set(seen.values()) & set(unseen.values())
    ):
        raise ValueError("Expected disjoint official TU-Berlin 220/30 classes")

    baseline = json.loads(args.mpq_summary.read_text())
    clip_sha = sha256(args.clip_weights)
    source = source_files()
    manifests = {
        str(path): sha256(path)
        for split in (data.train, data.test)
        for path in (split.sketch_manifest, split.photo_manifest, split.class_map)
    }

    output.mkdir(parents=True)
    for name in source:
        archive = output / "source" / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, archive)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    stem_channels = getattr(args, "stem_channels", 192)
    config = {
        "benchmark": "tuberlin_220_30",
        "method": "mamba_sketch_stem_frozen_clip_v1",
        "dataset": data.name,
        "epochs": epochs,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "stem_channels": stem_channels,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "source_sha256": source,
        "manifest_sha256": manifests,
        "data_config_sha256": sha256(args.data_config),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "clip_weights": str(args.clip_weights.expanduser().absolute()),
        "clip_sha256": clip_sha,
        "photo_encoder": "frozen_openai_CLIP_ViT-B-32-quickgelu",
        "loss": "multi_positive_retrieval_only",
        "wandb_project": "sbir",
        "primary_metric": "clean/full_mAP",
        "mpq_summary": str(args.mpq_summary.resolve()),
        "mpq_summary_sha256": sha256(args.mpq_summary),
    }

    write_json(output / "config.json", config)
    write_json(output / "runtime.json", {"status": "INITIALIZING"})

    run = None
    try:
        bundle = load_frozen_clip(
            pretrained=str(args.clip_weights.expanduser().absolute()), device="cuda"
        )
        photo_encoder = bundle.encoder

        # Wrap frozen CLIP ViT body with trainable MambaSketchStem
        model = CLIPWithSketchStem(
            bundle.encoder.model.visual, stem_channels=stem_channels
        ).to("cuda")

        trainable_params = [p for p in model.stem.parameters() if p.requires_grad]
        config["trainable_parameters"] = sum(p.numel() for p in trainable_params)
        config["frozen_clip_parameters"] = sum(
            p.numel() for p in model.visual.parameters()
        )

        # Respect SSM no-weight-decay rules
        decay, no_decay = [], []
        for _, parameter in model.stem.named_parameters():
            target = (
                no_decay
                if parameter.ndim < 2 or getattr(parameter, "_no_weight_decay", False)
                else decay
            )
            target.append(parameter)

        groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(
            groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        sketch_transform = transforms.Compose(
            build_clip_eval_transform().transforms[:-1]
        )
        loader = build_retrieval_train_loader(
            data,
            sketch_transform,
            bundle.transform,
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        total_steps = epochs * len(loader)
        scheduler = LambdaLR(
            optimizer, lambda step: learning_rate_factor(step, total_steps)
        )

        config.update(
            {
                "steps_per_epoch": len(loader),
                "total_steps": total_steps,
                "warmup_steps": round(0.05 * total_steps),
            }
        )
        write_json(output / "config.json", config)

        if args.wandb_mode == "online":
            import wandb

            run = wandb.init(
                project="sbir",
                entity=getattr(args, "wandb_entity", None)
                or os.environ.get("WANDB_ENTITY"),
                name=getattr(args, "run_name", None)
                or f"MambaStem{stem_channels}ch-FrozenCLIP-TU-{epochs}epochs",
                config=config,
                dir=str(output),
                mode="online",
            )
            write_json(output / "wandb.json", {"id": run.id, "url": run.url})

        step = 0
        examples_seen = 0
        write_json(
            output / "runtime.json", {"status": "TRAINING", "epoch": 0, "step": 0}
        )

        with (output / "training.jsonl").open("x") as history:
            for epoch in range(1, epochs + 1):
                model.train()
                loss_sum = 0.0

                for batch in loader:
                    sketches = batch["sketch"].to("cuda", non_blocking=True)
                    labels = batch["label"].to("cuda", non_blocking=True)
                    photos = torch.cat(
                        (batch["positive_photo"], batch["negative_photo"])
                    ).to("cuda")
                    photo_labels = torch.cat(
                        (batch["label"], batch["negative_label"])
                    ).to("cuda")

                    with torch.no_grad():
                        targets = photo_encoder(photos)

                    optimizer.zero_grad(set_to_none=True)
                    query = model(sketches)
                    loss = multi_positive_loss(query, targets, labels, photo_labels)

                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step + 1}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    scheduler.step()

                    step += 1
                    count = len(labels)
                    examples_seen += count
                    loss_sum += loss.item() * count

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

                epoch_loss = loss_sum / len(loader.dataset)
                row = {
                    "epoch": epoch,
                    "step": step,
                    "examples_seen": examples_seen,
                    "loss": epoch_loss,
                }

                if run is not None:
                    run.log(
                        {
                            "epoch": epoch,
                            "loss metrics/total_loss": epoch_loss,
                        },
                        step=step,
                    )

                history.write(json.dumps(row) + "\n")
                history.flush()
                print(json.dumps(row), flush=True)

                temporary = output / "model_latest.pt.tmp"
                torch.save(
                    {
                        "stem": model.stem.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "config": config,
                    },
                    temporary,
                )
                temporary.replace(output / "model_latest.pt")
                write_json(output / "runtime.json", {"status": "TRAINING", **row})

        write_json(
            output / "runtime.json",
            {"status": "FINAL_EVALUATION", "epoch": epochs, "step": step},
        )
        model.eval()

        metrics = final_retrieval(
            model,
            photo_encoder,
            data,
            sketch_transform,
            bundle.transform,
            output,
        )

        delta = {
            key: metrics[key] - baseline["clean"][key]
            for key in ("full_mAP", "P@200", "mAP@200_min_relevant_k")
        }

        result = {
            "status": "FINISHED_UNVERIFIED",
            "epoch": epochs,
            "step": step,
            "examples_seen": examples_seen,
            "clean": metrics,
            "delta_vs_mpq": delta,
            "primary_full_map_improved": delta["full_mAP"] > 0,
            "checkpoint_sha256": sha256(output / "model_latest.pt"),
            "config_sha256": sha256(output / "config.json"),
        }
        write_json(output / "result.json", result)

        if run is not None:
            artifact = wandb.Artifact(
                name=f"mamba-stem-{run.id}",
                type="model",
                metadata={"epoch": epochs, "step": step},
            )
            artifact.add_file(str(output / "model_latest.pt"))
            run.log_artifact(artifact)
            run.log(
                {
                    "progress_percent": 100.0,
                    "test/cleaned/mAP@all": metrics["full_mAP"],
                    "test/cleaned/mAP@200": metrics["mAP@200_min_relevant_k"],
                    "test/cleaned/P@200": metrics["P@200"],
                },
                step=step,
                commit=True,
            )
            run.finish()

        write_json(
            output / "runtime.json",
            {"status": result["status"], "epoch": epochs, "step": step},
        )
        return result

    except BaseException as error:
        write_json(
            output / "runtime.json",
            {"status": "FAILED_NO_RETRY", "error": repr(error)},
        )
        if run is not None:
            run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=ROOT / "configs/data/tuberlin_220_30.yaml",
    )
    parser.add_argument(
        "--clip-weights",
        type=Path,
        required=True,
        help="Local OpenAI ViT-B/32 QuickGELU weights",
    )
    parser.add_argument(
        "--mpq-summary",
        type=Path,
        required=True,
        help="Final official TU MP-Q step1189 summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--stem-channels", type=int, default=192)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "online"), default="disabled"
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)

    train(parser.parse_args())


if __name__ == "__main__":
    main()
