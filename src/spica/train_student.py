"""Train the random-init student for 100 complete TU-Berlin epochs, then evaluate.

No validation loop, checkpoint selection, resume, or automatic retry. The final
clean retrieval comparison is against a supplied, fixed-step MP-Q summary.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
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
from .models.sketch_student import SketchStudent

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 100
SEED = 42
BATCH_SIZE = 32
TEMPERATURE = 0.07
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
# User-selected official MP-Q control, fixed step1189 (not a selected peak).
MPQ_SUMMARY_SHA256 = "afb927f7142aafe98bf57c742a034d8478bdb2586880a53f6a3c5c15b2dfba42"


def multi_positive_loss(query, photos, labels, photo_labels):
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
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    return {
        name: sha256(ROOT / name)
        for name in sorted(set(names))
        if Path(name).suffix in {".py", ".yaml", ".toml", ".lock", ".md"}
        and not name.startswith("outputs/")
    }


def learning_rate_factor(step, total_steps):
    warmup = max(1, round(0.05 * total_steps))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))


@torch.inference_mode()
def final_retrieval(student, teacher, data, sketch_transform, photo_transform, output):
    encoded = []
    for encoder, manifest, transform, name in (
        (student, data.test.sketch_manifest, sketch_transform, "queries"),
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
        device=student.device,
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


def train(args):
    # Boundary/IO checks stay outside the model. Never overwrite an earlier run.
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    data = load_data_config(args.data_config)
    if data.name != "tuberlin_220_30":
        raise ValueError("This first 10-epoch experiment is official TU-Berlin only")
    seen = read_class_map(data.train.class_map)
    unseen = read_class_map(data.test.class_map)
    if (
        len(seen) != 220
        or len(unseen) != 30
        or set(seen.values()) & set(unseen.values())
    ):
        raise ValueError("Expected disjoint official TU-Berlin 220/30 classes")
    if sha256(args.mpq_summary) != MPQ_SUMMARY_SHA256:
        raise ValueError("MP-Q summary differs from the fixed user-selected control")
    baseline = json.loads(args.mpq_summary.read_text())
    if (
        baseline["benchmark"] != data.name
        or baseline["step"] != 1189
        or baseline["status"] != "COMPLETE"
        or baseline["query_count"] != 2400
        or baseline["gallery_count"] != 27989
    ):
        raise ValueError("Expected the final official TU MP-Q step1189 summary")
    clip_sha = sha256(args.clip_weights)
    source = source_files()
    manifests = {
        str(path): sha256(path)
        for split in (data.train, data.test)
        for path in (split.sketch_manifest, split.photo_manifest, split.class_map)
    }
    for split_name in ("train", "test"):
        split = getattr(data, split_name)
        for key, digest in baseline["protocol_identity"][split_name][
            "file_sha256"
        ].items():
            if sha256(getattr(split, key)) != digest:
                raise ValueError(
                    f"Manifest/class-map differs from MP-Q: {split_name}/{key}"
                )
    output.mkdir(parents=True)
    for name in source:
        archive = output / "source" / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, archive)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = {
        "method": "cnn_fourway_mamba2_student_v1",
        "dataset": data.name,
        "epochs": EPOCHS,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
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
        "student_initialization": "random",
        "photo_encoder": "frozen_openai_CLIP_ViT-B-32-quickgelu",
        "loss": "average_per_positive_contrastive;all_same_class_bank_entries_positive",
        "photo_bank": "B_sampled_positive+B_sampled_negative;duplicate_samples_retained",
        "selection": "none;final_epoch100_only",
        "precision": "fp32",
        "wandb_project": "sbir",
        "primary_metric": "clean/full_mAP",
        "comparison": "different_systems_not_backbone_only_ablation",
        "mpq_summary": str(args.mpq_summary.resolve()),
        "mpq_summary_sha256": sha256(args.mpq_summary),
        "mpq_checkpoint": baseline["checkpoint_selection"],
        "mpq_source": baseline["source_snapshot_hash"],
        "mpq_clean": baseline["clean"],
        "official_test_previously_observed": True,
    }
    write_json(output / "config.json", config)
    write_json(output / "runtime.json", {"status": "INITIALIZING"})
    run = None
    try:
        # Keep the .safetensors suffix: HF cache symlink targets have no suffix,
        # and OpenCLIP selects its safe reader from the supplied filename.
        bundle = load_frozen_clip(
            pretrained=str(args.clip_weights.expanduser().absolute()), device="cuda"
        )
        teacher = bundle.encoder
        student = SketchStudent().to("cuda")
        # Same deterministic geometry as CLIP, without its final normalization.
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
        steps_per_epoch = len(loader)
        total_steps = EPOCHS * steps_per_epoch
        config.update(
            {
                "train_sketches": len(loader.dataset),
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "warmup_steps": round(0.05 * total_steps),
                "student_parameters": sum(p.numel() for p in student.parameters()),
            }
        )
        write_json(output / "config.json", config)
        # Respect the SSM's no-weight-decay parameters; retain its initialization.
        decay, no_decay = [], []
        for _, parameter in student.named_parameters():
            target = (
                no_decay
                if parameter.ndim < 2 or getattr(parameter, "_no_weight_decay", False)
                else decay
            )
            target.append(parameter)
        optimizer = torch.optim.AdamW(
            [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = LambdaLR(
            optimizer, lambda step: learning_rate_factor(step, total_steps)
        )
        if args.wandb_mode == "online":
            import wandb

            run = wandb.init(
                project="sbir",
                entity="a-cctest05187-erd",
                name="CNN-Mamba-TU-100epochs",
                config=config,
                dir=str(output),
                mode="online",
                settings=wandb.Settings(
                    x_disable_stats=True,
                    x_disable_meta=True,
                    disable_code=True,
                    disable_git=True,
                    x_save_requirements=False,
                    console="off",
                ),
            )
            write_json(output / "wandb.json", {"id": run.id, "url": run.url})
        step = 0
        examples_seen = 0
        write_json(
            output / "runtime.json", {"status": "TRAINING", "epoch": 0, "step": 0}
        )
        with (output / "training.jsonl").open("x") as history:
            for epoch in range(1, EPOCHS + 1):
                student.train()
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
                        targets = teacher(photos)
                    optimizer.zero_grad(set_to_none=True)
                    loss = multi_positive_loss(
                        student(sketches), targets, labels, photo_labels
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), 1.0, error_if_nonfinite=True
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
                history.write(json.dumps(row) + "\n")
                history.flush()
                print(json.dumps(row), flush=True)
                temporary = output / "student_latest.pt.tmp"
                torch.save(
                    {
                        "model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "config": config,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "python_rng": random.getstate(),
                        "exact_data_stream_resume": False,
                    },
                    temporary,
                )
                temporary.replace(output / "student_latest.pt")
                write_json(output / "runtime.json", {"status": "TRAINING", **row})
        if source_files() != source or any(
            sha256(path) != digest for path, digest in manifests.items()
        ):
            raise RuntimeError(
                "Source/manifests changed during training; no evaluation"
            )
        write_json(
            output / "runtime.json",
            {"status": "FINAL_EVALUATION", "epoch": EPOCHS, "step": step},
        )
        student.eval()
        del loss, sketches, photos, targets, batch
        metrics = final_retrieval(
            student, teacher, data, sketch_transform, bundle.transform, output
        )
        if (
            metrics["query_count"] != baseline["query_count"]
            or metrics["gallery_count"] != baseline["gallery_count"]
        ):
            raise RuntimeError("Student and MP-Q retrieval counts differ")
        delta = {
            key: metrics[key] - baseline["clean"][key]
            for key in ("full_mAP", "P@200", "mAP@200_min_relevant_k")
        }
        if source_files() != source or any(
            sha256(path) != digest for path, digest in manifests.items()
        ):
            raise RuntimeError("Source/manifests changed during final evaluation")
        result = {
            "status": "FINISHED_UNVERIFIED",
            "epoch": EPOCHS,
            "step": step,
            "examples_seen": examples_seen,
            "clean": metrics,
            "delta_vs_mpq": delta,
            "primary_full_map_improved": delta["full_mAP"] > 0,
            "checkpoint_sha256": sha256(output / "student_latest.pt"),
            "config_sha256": sha256(output / "config.json"),
            "scope": "clean_final_only;different_photo_branch_and_training_budget;one_seed;previously_seen_official_test;no_significance_claim",
        }
        write_json(output / "result.json", result)
        if run is not None:
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
            {"status": result["status"], "epoch": EPOCHS, "step": step},
        )
        return result
    except BaseException as error:
        write_json(
            output / "runtime.json", {"status": "FAILED_NO_RETRY", "error": repr(error)}
        )
        if run is not None:
            run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config", type=Path, default=ROOT / "configs/data/tuberlin_220_30.yaml"
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
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "online"), default="disabled"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
