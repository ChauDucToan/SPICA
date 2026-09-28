"""Train Student-only for 100 epochs, or joint LoRA teacher/Student for 10.

No validation loop, checkpoint selection, resume, or automatic retry. The final
clean retrieval comparison is against a supplied, fixed-step MP-Q summary.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
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
from .models.sketch_teacher import SketchTeacher, cosine_distillation

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 100
SEED = 42
BATCH_SIZE = 32
TEMPERATURE = 0.07
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
KD_CALIBRATION_RULE = "lambda=0.1/median4(norm(g_KD)/norm(g_retrieval));all_Student_parameters"
# User-selected official MP-Q control, fixed step1189 (not a selected peak).
MPQ_SUMMARY_SHA256 = "afb927f7142aafe98bf57c742a034d8478bdb2586880a53f6a3c5c15b2dfba42"


def multi_positive_loss(query, photos, labels, photo_labels):
    log_prob = F.log_softmax(query @ photos.T / TEMPERATURE, dim=-1)
    positive = labels[:, None] == photo_labels[None, :]
    return (-(log_prob * positive).sum(dim=1) / positive.sum(dim=1)).mean()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_kd_calibration(path):
    value = json.loads(path.read_text())
    if (value["status"] != "SCOPED_GRADIENT_CALIBRATION_PASS"
            or value["rule"] != KD_CALIBRATION_RULE
            or value["optimizer_updates"] != 0
            or value["test_images_opened"] != 0
            or value["state_before"] != value["state_after"]
            or len(value["batches"]) != 4):
        raise ValueError("Invalid KD calibration protocol")
    ratios = []
    for index, row in enumerate(value["batches"]):
        nr, nk = row["retrieval_grad_norm"], row["kd_grad_norm"]
        if row["batch"] != index or any(not math.isfinite(x) or x <= 0 for x in (nr, nk)):
            raise ValueError("Invalid KD gradient norms")
        ratio = nk / nr
        if not math.isclose(ratio, row["ratio_kd_to_retrieval"], rel_tol=1e-12):
            raise ValueError("KD ratio disagrees with gradient norms")
        ratios.append(ratio)
    coefficient = value["lambda_kd"]
    if (not math.isfinite(coefficient) or coefficient <= 0
            or not math.isclose(coefficient, 0.1 / statistics.median(ratios), rel_tol=1e-12)
            or len(value["weighted_ratios"]) != 4
            or not all(math.isclose(a, coefficient*b, rel_tol=1e-12)
                       for a, b in zip(value["weighted_ratios"], ratios))):
        raise ValueError("KD coefficient does not implement the approved rule")
    if set(value["artifacts_sha256"]) != {f"batch_{i}.pt" for i in range(4)}:
        raise ValueError("Expected exactly four KD raw-gradient artifacts")
    for name, digest in value["artifacts_sha256"].items():
        if sha256(path.parent / name) != digest:
            raise ValueError(f"KD artifact changed: {name}")
    return value


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
    joint = args.lora_teacher
    visual_prompt = getattr(args, "visual_prompt", False)
    if visual_prompt and not joint:
        raise ValueError("--visual-prompt requires --lora-teacher")
    calibration_path = getattr(args, "kd_calibration", None)
    if calibration_path is not None and not (joint and visual_prompt):
        raise ValueError("--kd-calibration requires --lora-teacher --visual-prompt")
    epochs = 10 if joint else EPOCHS
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
    lambda_kd = 1.0
    calibration = None
    if calibration_path is not None:
        calibration = read_kd_calibration(calibration_path)
        if (calibration["source_sha256"] != source
                or calibration["manifest_sha256"] != manifests
                or calibration["clip_sha256"] != clip_sha
                or calibration["data_config_sha256"] != sha256(args.data_config)
                or calibration["seed"] != SEED
                or calibration["batch_size"] != BATCH_SIZE
                or calibration["visual_prompt_tokens"] != 4):
            raise ValueError("KD calibration does not match this source/data/initialization protocol")
        lambda_kd = float(calibration["lambda_kd"])
        if not math.isfinite(lambda_kd) or lambda_kd <= 0:
            raise ValueError("KD calibration coefficient must be finite and positive")
    output.mkdir(parents=True)
    for name in source:
        archive = output / "source" / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, archive)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = {
        "method": "cnn_mamba_lora_teacher_v1" if joint else "cnn_fourway_mamba2_student_v1",
        "dataset": data.name,
        "epochs": epochs,
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
        "selection": f"none;final_epoch{epochs}_only",
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
    if visual_prompt:
        config["method"] = "cnn_mamba_lora_vpt_teacher_v1"
    if joint:
        config["sketch_teacher"] = {
            "base": "CLIP_ViT-B-32-quickgelu_frozen",
            "lora": "Q,V_all12blocks", "rank": 8, "alpha": 8, "dropout": 0,
            "lr": 1e-4, "lambda_kd": lambda_kd, "kd": "cosine_teacher_detached",
            "lambda_calibrated": calibration is not None, "start": "first_batch_no_teacher_warmup",
            "teacher_objective": "multi_positive_retrieval_only",
            "gradient_clip": "1_each_student_and_teacher_separately",
        }
        if calibration is not None:
            config["sketch_teacher"]["kd_calibration"] = {
                "path": str(calibration_path.absolute()), "sha256": sha256(calibration_path),
                "rule": calibration["rule"], "batches": 4,
                "scope": "initialization_gradient_norm_only_not_optimal_or_Adam_update_ratio",
            }
        if visual_prompt:
            config["sketch_teacher"]["visual_prompt"] = {
                "tokens": 4, "type": "shallow_after_CLS_before_preLN",
                "original_positional_embeddings": "unchanged_no_prompt_positions",
                "init": "normal_std0.02", "lr": 1e-4, "weight_decay": 0.01,
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
        sketch_teacher = None
        adapter_parameters = []
        if joint:
            normalize = bundle.transform.transforms[-1]
            if not isinstance(normalize, transforms.Normalize):
                raise TypeError("Expected pinned CLIP Normalize as final transform")
            config["sketch_teacher"]["normalize_mean"] = list(normalize.mean)
            config["sketch_teacher"]["normalize_std"] = list(normalize.std)
            sketch_teacher = SketchTeacher(
                teacher.model.visual, normalize.mean, normalize.std,
                visual_prompt=visual_prompt,
            ).to("cuda").eval()
            adapter_parameters = [p for p in sketch_teacher.parameters() if p.requires_grad]
            config["sketch_teacher"]["trainable_parameters"] = sum(
                p.numel() for p in adapter_parameters
            )
        if calibration is not None:
            actual = {"student": state_sha(student), "teacher": state_sha(sketch_teacher),
                      "photo": state_sha(teacher)}
            if actual != calibration["state_before"]:
                raise ValueError("KD calibration model initialization differs from actual trainer")
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
        total_steps = epochs * steps_per_epoch
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
        groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        if joint:
            groups.append({"params": adapter_parameters, "lr": 1e-4})
        optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = LambdaLR(
            optimizer, lambda step: learning_rate_factor(step, total_steps)
        )
        if args.wandb_mode == "online":
            import wandb

            run = wandb.init(
                project="sbir",
                entity="a-cctest05187-erd",
                name=("CNN-Mamba-LoRA-VPT-KDcal-TU-10epochs" if calibration is not None else
                      "CNN-Mamba-LoRA-VPT-Teacher-TU-10epochs" if visual_prompt else
                      "CNN-Mamba-LoRA-Teacher-TU-10epochs" if joint else "CNN-Mamba-TU-100epochs"),
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
            for epoch in range(1, epochs + 1):
                student.train()
                loss_sum = 0.0
                teacher_sum = retrieval_sum = kd_sum = 0.0
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
                    if joint:
                        teacher_query = sketch_teacher(sketches)
                        teacher_loss = multi_positive_loss(
                            teacher_query, targets, labels, photo_labels
                        )
                        if not torch.isfinite(teacher_loss):
                            raise FloatingPointError(f"Nonfinite teacher loss at step {step + 1}")
                        # Free teacher activations before Student forward; targets are pre-update.
                        teacher_loss.backward()
                        teacher_query = teacher_query.detach()
                    query = student(sketches)
                    retrieval_loss = multi_positive_loss(query, targets, labels, photo_labels)
                    loss = retrieval_loss
                    if joint:
                        kd_loss = cosine_distillation(query, teacher_query)
                        loss = loss + lambda_kd * kd_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), 1.0, error_if_nonfinite=True
                    )
                    if joint:
                        torch.nn.utils.clip_grad_norm_(
                            adapter_parameters, 1.0, error_if_nonfinite=True
                        )
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    count = len(labels)
                    examples_seen += count
                    loss_sum += loss.item() * count
                    if joint:
                        teacher_sum += teacher_loss.item() * count
                        retrieval_sum += retrieval_loss.item() * count
                        kd_sum += kd_loss.item() * count
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
                if joint:
                    row.update(
                        teacher_loss=teacher_sum / len(loader.dataset),
                        student_retrieval_loss=retrieval_sum / len(loader.dataset),
                        kd_loss=kd_sum / len(loader.dataset),
                    )
                history.write(json.dumps(row) + "\n")
                history.flush()
                print(json.dumps(row), flush=True)
                temporary = output / "student_latest.pt.tmp"
                torch.save(
                    {
                        "model": student.state_dict(),
                        **({"teacher_adapter": sketch_teacher.adapter_state_dict()} if joint else {}),
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
            {"status": "FINAL_EVALUATION", "epoch": epochs, "step": step},
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
        teacher_metrics = None
        if joint:
            teacher_output = output / "teacher"
            teacher_output.mkdir()
            teacher_metrics = final_retrieval(
                sketch_teacher, teacher, data, sketch_transform, bundle.transform, teacher_output
            )
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
            "epoch": epochs,
            "step": step,
            "examples_seen": examples_seen,
            "clean": metrics,
            "delta_vs_mpq": delta,
            "primary_full_map_improved": delta["full_mAP"] > 0,
            "checkpoint_sha256": sha256(output / "student_latest.pt"),
            "config_sha256": sha256(output / "config.json"),
            "scope": "clean_final_only;different_photo_branch_and_training_budget;one_seed;previously_seen_official_test;no_significance_claim",
        }
        if joint:
            result["teacher_clean"] = teacher_metrics
        write_json(output / "result.json", result)
        if run is not None:
            run.log(
                {
                    "progress_percent": 100.0,
                    "test/cleaned/mAP@all": metrics["full_mAP"],
                    "test/cleaned/mAP@200": metrics["mAP@200_min_relevant_k"],
                    "test/cleaned/P@200": metrics["P@200"],
                    **({
                        "teacher/cleaned/mAP@all": teacher_metrics["full_mAP"],
                        "teacher/cleaned/mAP@200": teacher_metrics["mAP@200_min_relevant_k"],
                        "teacher/cleaned/P@200": teacher_metrics["P@200"],
                    } if joint else {}),
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
    parser.add_argument("--lora-teacher", action="store_true",
                        help="Fresh 10 epochs joint Q/V-LoRA teacher + Student, fixed cosine KD=1")
    parser.add_argument("--visual-prompt", action="store_true",
                        help="Add four shallow visual tokens to the LoRA sketch teacher")
    parser.add_argument("--kd-calibration", type=Path,
                        help="Reviewed initialization-gradient receipt; absent keeps KD=1")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "online"), default="disabled"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
