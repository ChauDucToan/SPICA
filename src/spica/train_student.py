"""Train through a dependency-injector container with final-only retrieval evaluation.

Legacy defaults remain CNN–Mamba Student-only for 100 epochs or joint LoRA
teacher/Student for 10. Factories/losses are trusted local Python callables.
No validation loop, checkpoint selection, resume, or automatic retry.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from dependency_injector import __version__ as di_version
from dependency_injector import containers
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from .config.data import load_data_config
from .containers import active_provider, create_training_container
from .data.datasets import RetrievalEvalDataset
from .data.loaders import build_retrieval_train_loader
from .data.manifest import read_class_map, read_manifest
from .data.transforms import build_clip_eval_transform
from .evaluation.embeddings import encode_retrieval_loader, save_encoded_retrieval_set
from .evaluation.metrics import evaluate_category_retrieval
from .models.clip import load_frozen_clip
from .models.sketch_student import SketchStudent, StudentPatchProjector
from .models.sketch_teacher import (
    SketchTeacher,
    cosine_distillation,
    relational_distillation,
)

QueryEmbeddings = Float[Tensor, "batch embedding"]
PhotoEmbeddings = Float[Tensor, "photos embedding"]
QueryLabels = Int[Tensor, "batch"]
PhotoLabels = Int[Tensor, "photos"]
Scalar = Float[Tensor, ""]

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 100
SEED = 42
BATCH_SIZE = 32
TEMPERATURE = 0.07
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
KD_CALIBRATION_RULE = (
    "lambda=0.1/median4(norm(g_KD)/norm(g_retrieval));all_Student_parameters"
)
# User-selected official MP-Q control, fixed step1189 (not a selected peak).
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


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_kd_calibration(path):
    value = json.loads(path.read_text())
    if (
        value["status"] != "SCOPED_GRADIENT_CALIBRATION_PASS"
        or value["rule"] != KD_CALIBRATION_RULE
        or value["optimizer_updates"] != 0
        or value["test_images_opened"] != 0
        or value["state_before"] != value["state_after"]
        or len(value["batches"]) != 4
    ):
        raise ValueError("Invalid KD calibration protocol")
    ratios = []
    for index, row in enumerate(value["batches"]):
        nr, nk = row["retrieval_grad_norm"], row["kd_grad_norm"]
        if row["batch"] != index or any(
            not math.isfinite(x) or x <= 0 for x in (nr, nk)
        ):
            raise ValueError("Invalid KD gradient norms")
        ratio = nk / nr
        if not math.isclose(ratio, row["ratio_kd_to_retrieval"], rel_tol=1e-12):
            raise ValueError("KD ratio disagrees with gradient norms")
        ratios.append(ratio)
    coefficient = value["lambda_kd"]
    if (
        not math.isfinite(coefficient)
        or coefficient <= 0
        or not math.isclose(coefficient, 0.1 / statistics.median(ratios), rel_tol=1e-12)
        or len(value["weighted_ratios"]) != 4
        or not all(
            math.isclose(a, coefficient * b, rel_tol=1e-12)
            for a, b in zip(value["weighted_ratios"], ratios)
        )
    ):
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


def callable_name(value: Callable) -> str:
    """Name functions, constructors, partials and callable objects without addresses."""
    if isinstance(value, partial):
        value = value.func
    return f"{getattr(value, '__module__', type(value).__module__)}:{getattr(value, '__qualname__', type(value).__qualname__)}"


def import_callable(spec: str) -> Callable:
    """Resolve a trusted local MODULE:CALLABLE; never fetch code remotely."""
    module, separator, name = spec.partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError("Expected MODULE:CALLABLE")
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError(f"Not callable: {spec}")
    return value


def train(
    args: argparse.Namespace,
    *,
    container: containers.DynamicContainer | None = None,
    student_factory: Callable[[], torch.nn.Module] | None = None,
    retrieval_loss: Callable[..., Tensor] | None = None,
):
    """Consume framework providers, or compose them from compatible API/CLI callables.

    Encoders expose .device and return B×512 embeddings from B×3×224×224 RGB.
    Absent injections resolve the legacy symbols at call time, not import time.
    Do not combine a supplied container with API/CLI component overrides.
    """
    # Boundary/IO checks stay outside the model. Never overwrite an earlier run.
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    joint = args.lora_teacher
    visual_prompt = getattr(args, "visual_prompt", False)
    if visual_prompt and not joint:
        raise ValueError("--visual-prompt requires --lora-teacher")
    mask_white = getattr(args, "mask_white_patches", False)
    deep_prompt = getattr(args, "deep_visual_prompt", False)
    if deep_prompt and (not joint or not visual_prompt or not mask_white):
        raise ValueError(
            "Deep prompts require LoRA teacher, visual prompts and white mask"
        )
    if mask_white and not joint:
        raise ValueError("--mask-white-patches requires --lora-teacher")
    calibration_path = getattr(args, "kd_calibration", None)
    if calibration_path is not None and (mask_white or not (joint and visual_prompt)):
        raise ValueError("--kd-calibration requires --lora-teacher --visual-prompt")
    fixed_lambda = getattr(args, "lambda_kd", None)
    if fixed_lambda is not None:
        if not joint or calibration_path is not None:
            raise ValueError(
                "--lambda-kd requires --lora-teacher and cannot be combined with calibration"
            )
        if not math.isfinite(fixed_lambda) or fixed_lambda <= 0:
            raise ValueError("--lambda-kd must be finite and positive")
    if mask_white and (not visual_prompt or fixed_lambda != 1.2):
        raise ValueError(
            "White-patch experiment requires --visual-prompt --lambda-kd 1.2"
        )
    epochs: int | None = getattr(args, "epochs", None)
    if epochs is None:
        epochs = 10 if joint else EPOCHS
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    factory_spec = getattr(args, "student_factory", None)
    loss_spec = getattr(args, "retrieval_loss", None)
    supplied_container = container is not None
    if (
        supplied_container
        and any(
            value is not None
            for value in (student_factory, retrieval_loss, factory_spec, loss_spec)
        )
        or student_factory is not None
        and factory_spec is not None
        or retrieval_loss is not None
        and loss_spec is not None
    ):
        raise ValueError(
            "Use a container or one API/CLI override per component, not both"
        )
    injected_student = (
        supplied_container or student_factory is not None or factory_spec is not None
    )
    injected_loss = (
        supplied_container or retrieval_loss is not None or loss_spec is not None
    )
    if container is None:
        if student_factory is None:
            student_factory = (
                import_callable(factory_spec)
                if factory_spec is not None
                else SketchStudent
            )
        if retrieval_loss is None:
            retrieval_loss = (
                import_callable(loss_spec)
                if loss_spec is not None
                else multi_positive_loss
            )
        container = create_training_container(student_factory, retrieval_loss)
    retrieval_loss = container.retrieval_loss
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
    # Manifest validation bypassed per user instruction: assume data invariant.
    lambda_kd = 1.0 if fixed_lambda is None else float(fixed_lambda)
    calibration = None
    if calibration_path is not None:
        calibration = read_kd_calibration(calibration_path)
        if (
            calibration["source_sha256"] != source
            or calibration["manifest_sha256"] != manifests
            or calibration["clip_sha256"] != clip_sha
            or calibration["data_config_sha256"] != sha256(args.data_config)
            or calibration["seed"] != SEED
            or calibration["batch_size"] != BATCH_SIZE
            or calibration["visual_prompt_tokens"] != 4
        ):
            raise ValueError(
                "KD calibration does not match this source/data/initialization protocol"
            )
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
    kd_temperature = getattr(args, "kd_temperature", 0.2)
    config = {
        "method": "cnn_mamba_lora_teacher_v1"
        if joint
        else "cnn_fourway_mamba2_student_v1",
        "dependency_injection": {
            "framework": "dependency-injector",
            "version": di_version,
        },
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
            "lora": "Q,V_all12blocks",
            "rank": 8,
            "alpha": 8,
            "dropout": 0,
            "lr": 1e-4,
            "lambda_kd": lambda_kd,
            "kd": "relational_cross_modal_affinity_and_feature_anchor",
            "kd_temperature": kd_temperature,
            "lambda_calibrated": calibration is not None,
            "start": "first_batch_no_teacher_warmup",
            "teacher_objective": "multi_positive_retrieval_only",
            "gradient_clip": "1_each_student_and_teacher_separately",
        }
        if mask_white:
            config["method"] = (
                "cnn_mamba_lora_vpt_deep_white_mask_teacher_v1"
                if deep_prompt
                else "cnn_mamba_lora_vpt_white_patch_mask_teacher_v1"
            )
            config["sketch_teacher"]["white_patch_mask"] = {
                "rule": "all_RGB_values_exactly_1_after_resize_crop_before_normalization",
                "patch_size": 32,
                "scope": "sketch_teacher_keys_all12layers",
                "positions": "original_unchanged",
                "CLS_and_prompts": "always_valid",
                "photo_and_student": "unchanged",
                "train_and_eval": True,
                "compute": "dense_mask_no_speedup_claim",
            }
            if deep_prompt:
                config["sketch_teacher"]["visual_prompt"] = {
                    "tokens_per_layer": 4,
                    "layers": 12,
                    "type": "independent_shallow_tokens_replaced_each_layer",
                    "position": "after_CLS_before_49_original_patches",
                    "readout": "native_CLS",
                    "init": "normal_std0.02",
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                }
        if fixed_lambda is not None:
            config["sketch_teacher"]["lambda_source"] = (
                "explicit_user_choice_not_calibrated"
            )
        if calibration is not None:
            config["sketch_teacher"]["kd_calibration"] = {
                "path": str(calibration_path.absolute()),
                "sha256": sha256(calibration_path),
                "rule": calibration["rule"],
                "batches": 4,
                "scope": "initialization_gradient_norm_only_not_optimal_or_Adam_update_ratio",
            }
        if visual_prompt and not deep_prompt:
            config["sketch_teacher"]["visual_prompt"] = {
                "tokens": 4,
                "type": "shallow_after_CLS_before_preLN",
                "original_positional_embeddings": "unchanged_no_prompt_positions",
                "init": "normal_std0.02",
                "lr": 1e-4,
                "weight_decay": 0.01,
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
        student = container.student().to("cuda")
        if injected_student:
            provider = active_provider(container.student)
            factory_name = callable_name(getattr(provider, "provides", provider))
            config["student_class"] = callable_name(type(student))
            config["student_initial_state_sha256"] = state_sha(student)
            config["student_factory"] = factory_name
            config["student_model"] = getattr(student, "experiment_metadata", {})
            config["method"] = config["student_model"].get("method", factory_name)
            if joint:
                config["method"] += ";joint_sketch_teacher"
        loss_object = getattr(
            active_provider(container.retrieval_loss), "provides", None
        )
        if injected_loss:
            config["loss"] = callable_name(loss_object)
            config["retrieval_loss_metadata"] = getattr(
                loss_object, "experiment_metadata", {}
            )
        sketch_teacher = None
        patch_projector = None
        adapter_parameters = []
        if joint:
            normalize = bundle.transform.transforms[-1]
            config["sketch_teacher"]["normalize_mean"] = list(normalize.mean)
            config["sketch_teacher"]["normalize_std"] = list(normalize.std)
            patch_projector = StudentPatchProjector(
                in_channels=getattr(student, "channels", 384)
            ).to("cuda")
            sketch_teacher = (
                SketchTeacher(
                    teacher.model.visual,
                    normalize.mean,
                    normalize.std,
                    visual_prompt=visual_prompt,
                    mask_white_patches=mask_white,
                    deep_visual_prompt=deep_prompt,
                )
                .to("cuda")
                .eval()
            )
            adapter_parameters = [
                p for p in sketch_teacher.parameters() if p.requires_grad
            ]
            config["sketch_teacher"]["trainable_parameters"] = sum(
                p.numel() for p in adapter_parameters
            )
        if calibration is not None:
            actual = {
                "student": state_sha(student),
                "teacher": state_sha(sketch_teacher),
                "photo": state_sha(teacher),
            }
            if actual != calibration["state_before"]:
                raise ValueError(
                    "KD calibration model initialization differs from actual trainer"
                )
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
        if patch_projector is not None:
            decay.extend(patch_projector.parameters())
        groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        if joint:
            groups.append({"params": adapter_parameters, "lr": 1e-4})
        optimizer = torch.optim.AdamW(
            groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = LambdaLR(
            optimizer, lambda step: learning_rate_factor(step, total_steps)
        )
        if args.wandb_mode == "online":
            import wandb

            run = wandb.init(
                project="sbir",
                entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
                name=getattr(args, "run_name", None)
                or (
                    f"{config['method']}-TU-{epochs}epochs"
                    if injected_student or getattr(args, "epochs", None) is not None
                    else f"CNN-Mamba-LoRA-DeepVPT-WhiteMask-KD{lambda_kd:g}-TU-10epochs"
                    if deep_prompt
                    else f"CNN-Mamba-LoRA-VPT-WhiteMask-KD{lambda_kd:g}-TU-10epochs"
                    if mask_white
                    else (
                        f"CNN-Mamba-LoRA{'-VPT' if visual_prompt else ''}-KD{lambda_kd:g}-TU-10epochs"
                    )
                    if fixed_lambda is not None
                    else "CNN-Mamba-LoRA-VPT-KDcal-TU-10epochs"
                    if calibration is not None
                    else "CNN-Mamba-LoRA-VPT-Teacher-TU-10epochs"
                    if visual_prompt
                    else "CNN-Mamba-LoRA-Teacher-TU-10epochs"
                    if joint
                    else "CNN-Mamba-TU-100epochs"
                ),
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
        data_stream = (
            hashlib.sha256() if getattr(args, "trace_data_stream", False) else None
        )
        write_json(
            output / "runtime.json", {"status": "TRAINING", "epoch": 0, "step": 0}
        )
        with (output / "training.jsonl").open("x") as history:
            for epoch in range(1, epochs + 1):
                student.train()
                if patch_projector is not None:
                    patch_projector.train()
                loss_sum = 0.0
                teacher_sum = retrieval_sum = kd_sum = 0.0
                kd_rel_sum = kd_aff_sum = kd_fd_sum = kd_patch_sum = kd_margin_sum = 0.0
                component_sums = {}
                for batch in loader:
                    if data_stream is not None:
                        for paths in zip(
                            batch["sketch_path"],
                            batch["positive_photo_path"],
                            batch["negative_photo_path"],
                            strict=True,
                        ):
                            data_stream.update(("\0".join(paths) + "\0").encode())
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
                    teacher_patches = None
                    if joint:
                        teacher_query, teacher_patches = sketch_teacher(sketches, return_patches=True)
                        teacher_loss = retrieval_loss(
                            teacher_query, targets, labels, photo_labels
                        )
                        if not torch.isfinite(teacher_loss):
                            raise FloatingPointError(
                                f"Nonfinite teacher loss at step {step + 1}"
                            )
                        # Free teacher activations before Student forward; targets are pre-update.
                        teacher_loss.backward()
                        teacher_query = teacher_query.detach()
                        teacher_patches = teacher_patches.detach()
                    query = student(sketches)
                    student_patches = patch_projector(student.last_features) if joint else None
                    retrieval_value = retrieval_loss(
                        query, targets, labels, photo_labels
                    )
                    loss = retrieval_value
                    if joint:
                        pos_photos = targets[:len(labels)]
                        neg_photos = targets[len(labels):]
                        kd_loss, rel_loss, aff_loss, fd_loss, patch_loss, margin_loss = relational_distillation(
                            query,
                            teacher_query,
                            targets,
                            student_patches=student_patches,
                            teacher_patches=teacher_patches,
                            pos_photos=pos_photos,
                            neg_photos=neg_photos,
                            temperature=kd_temperature,
                        )
                        loss = loss + lambda_kd * kd_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), 1.0, error_if_nonfinite=True
                    )
                    if joint:
                        torch.nn.utils.clip_grad_norm_(
                            patch_projector.parameters(), 1.0, error_if_nonfinite=True
                        )
                        torch.nn.utils.clip_grad_norm_(
                            adapter_parameters, 1.0, error_if_nonfinite=True
                        )
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    count = len(labels)
                    examples_seen += count
                    loss_sum += loss.item() * count
                    for name, value in getattr(
                        loss_object, "last_components", {}
                    ).items():
                        component_sums[name] = (
                            component_sums.get(name, 0.0) + value.item() * count
                        )
                    if joint:
                        teacher_sum += teacher_loss.item() * count
                        retrieval_sum += retrieval_value.item() * count
                        kd_sum += kd_loss.item() * count
                        kd_rel_sum += rel_loss.item() * count
                        kd_aff_sum += aff_loss.item() * count
                        kd_fd_sum += fd_loss.item() * count
                        kd_patch_sum += patch_loss.item() * count
                        kd_margin_sum += margin_loss.item() * count
                    if step == 1 or step % 50 == 0:
                        write_json(
                            output / "runtime.json",
                            {
                                "status": "TRAINING",
                                "epoch": epoch,
                                "step": step,
                                "total_steps": total_steps,
                                "examples_seen": examples_seen,
                                "query_centered_variance": query.detach()
                                .var(dim=0, unbiased=False)
                                .sum()
                                .item(),
                                **(
                                    {"data_stream_sha256": data_stream.hexdigest()}
                                    if data_stream is not None
                                    else {}
                                ),
                            },
                        )
                epoch_loss = loss_sum / len(loader.dataset)
                row = {
                    "epoch": epoch,
                    "step": step,
                    "examples_seen": examples_seen,
                    "loss": epoch_loss,
                    **{
                        f"loss_{name}": value / len(loader.dataset)
                        for name, value in component_sums.items()
                    },
                    **(
                        {"data_stream_sha256": data_stream.hexdigest()}
                        if data_stream is not None
                        else {}
                    ),
                }
                if joint:
                    row.update(
                        teacher_loss=teacher_sum / len(loader.dataset),
                        student_retrieval_loss=retrieval_sum / len(loader.dataset),
                        kd_loss=kd_sum / len(loader.dataset),
                        kd_relational_loss=kd_rel_sum / len(loader.dataset),
                        kd_affinity_loss=kd_aff_sum / len(loader.dataset),
                        kd_feature_anchor_loss=kd_fd_sum / len(loader.dataset),
                        kd_patch_loss=kd_patch_sum / len(loader.dataset),
                        kd_dynamic_margin_loss=kd_margin_sum / len(loader.dataset),
                    )
                if run is not None:
                    loss_metrics = {
                        "epoch": epoch,
                        "loss metrics/total_loss": epoch_loss,
                        "loss metrics/student_retrieval_loss": retrieval_sum / len(loader.dataset),
                    }
                    if joint:
                        loss_metrics.update({
                            "loss metrics/teacher_loss": teacher_sum / len(loader.dataset),
                            "loss metrics/kd_total_loss": kd_sum / len(loader.dataset),
                            "loss metrics/kd_relational_loss": kd_rel_sum / len(loader.dataset),
                            "loss metrics/kd_affinity_loss": kd_aff_sum / len(loader.dataset),
                            "loss metrics/kd_feature_anchor_loss": kd_fd_sum / len(loader.dataset),
                            "loss metrics/kd_patch_loss": kd_patch_sum / len(loader.dataset),
                            "loss metrics/kd_dynamic_margin_loss": kd_margin_sum / len(loader.dataset),
                        })
                    for name, value in component_sums.items():
                        loss_metrics[f"loss metrics/{name}"] = value / len(loader.dataset)
                    run.log(loss_metrics, step=step)
                history.write(json.dumps(row) + "\n")
                history.flush()
                print(json.dumps(row), flush=True)
                temporary = output / "student_latest.pt.tmp"
                torch.save(
                    {
                        "model": student.state_dict(),
                        **(
                            {
                                "teacher_adapter": sketch_teacher.adapter_state_dict(),
                                "patch_projector": patch_projector.state_dict(),
                            }
                            if joint
                            else {}
                        ),
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
                sketch_teacher,
                teacher,
                data,
                sketch_transform,
                bundle.transform,
                teacher_output,
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
            **(
                {"train_data_stream_sha256": data_stream.hexdigest()}
                if data_stream is not None
                else {}
            ),
            "checkpoint_sha256": sha256(output / "student_latest.pt"),
            "config_sha256": sha256(output / "config.json"),
            "scope": "clean_final_only;different_photo_branch_and_training_budget;one_seed;previously_seen_official_test;no_significance_claim",
        }
        if joint:
            result["teacher_clean"] = teacher_metrics
        write_json(output / "result.json", result)
        if run is not None:
            artifact = wandb.Artifact(
                name=f"student-{run.id}",
                type="model",
                metadata={"epoch": epochs, "step": step},
            )
            artifact.add_file(str(output / "student_latest.pt"))
            run.log_artifact(artifact)
            wandb.save(str(output / "student_latest.pt"), base_path=str(output))
            run.log(
                {
                    "progress_percent": 100.0,
                    "test/cleaned/mAP@all": metrics["full_mAP"],
                    "test/cleaned/mAP@200": metrics["mAP@200_min_relevant_k"],
                    "test/cleaned/P@200": metrics["P@200"],
                    **(
                        {
                            "teacher/cleaned/mAP@all": teacher_metrics["full_mAP"],
                            "teacher/cleaned/mAP@200": teacher_metrics[
                                "mAP@200_min_relevant_k"
                            ],
                            "teacher/cleaned/P@200": teacher_metrics["P@200"],
                        }
                        if joint
                        else {}
                    ),
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
    parser.add_argument(
        "--epochs", type=int, help="Override the legacy 100/10-epoch default"
    )
    parser.add_argument(
        "--student-factory", help="Trusted local MODULE:CALLABLE returning an encoder"
    )
    parser.add_argument(
        "--retrieval-loss",
        help="Trusted local MODULE:CALLABLE accepting query/photos/labels/photo_labels",
    )
    parser.add_argument("--run-name", help="Explicit W&B experiment name")
    parser.add_argument(
        "--trace-data-stream",
        action="store_true",
        help="Fingerprint sampled TRAIN sketch/positive/negative paths",
    )
    parser.add_argument(
        "--lora-teacher",
        action="store_true",
        help="Fresh 10 epochs joint Q/V-LoRA teacher + Student, fixed cosine KD=1",
    )
    parser.add_argument(
        "--visual-prompt",
        action="store_true",
        help="Add four shallow visual tokens to the LoRA sketch teacher",
    )
    parser.add_argument(
        "--dynamic-prompt",
        action="store_true",
        help="Condition teacher visual prompt on student ink features",
    )
    parser.add_argument(
        "--kd-calibration",
        type=Path,
        help="Reviewed initialization-gradient receipt; absent keeps KD=1",
    )
    parser.add_argument(
        "--mask-white-patches",
        action="store_true",
        help="Mask exactly-white sketch patch keys in every teacher attention layer",
    )
    parser.add_argument(
        "--deep-visual-prompt",
        action="store_true",
        help="Replace shallow prompt with four independent tokens at each teacher layer",
    )
    parser.add_argument(
        "--lambda-kd",
        type=float,
        help="Explicit positive KD weight; mutually exclusive with calibration",
    )
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=0.2,
        help="Softmax temperature for cross-modal relational KD (default 0.2)",
    )
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "online"), default="disabled"
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="W&B entity/team name"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
