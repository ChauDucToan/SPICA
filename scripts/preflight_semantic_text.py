"""Real-CLIP, no-update GPU preflight for semantic text S0/S1/S2.

This is deliberately a preflight, not a trainer: it composes the production
configs, uses the production CLIP/prompt/text-bank/data-loader constructors,
and executes one real full/full loss graph per arm without an optimizer step.
It never loads official-test manifests, starts W&B, or writes old artifacts.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import traceback
from typing import Any

# Set these before importing torch/open_clip.  CUDA remains visible: this is a
# real GPU gate, unlike the CPU fixture gate.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["WANDB_MODE"] = "disabled"

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
OUT_PREFIX = "semantic_text_preflight_"
ARMS = ("semantic_text_S0", "semantic_text_S1", "semantic_text_S2")
CAMPAIGN = "frozen_prompt_semantic_text_step1_2026-09-07"
EXPECTED_PAIRING_SHA256 = "545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c"
EXPECTED_SPLIT_SHA256 = "3e02604d2ed315aa254d4264ec440a7e50233c7c9b175be224519feafda88425"
EXPECTED_CLIP_SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
EXPECTED_CLIP_BYTES = 605143284
CLIP_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai"
    / "snapshots/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors"
)
EXPECTED_TRAIN_CLASSES = tuple(
    value
    for value in range(104)
    if value not in {7, 13, 16, 27, 28, 31, 33, 34, 39, 42, 45, 51, 52, 53, 60, 65, 75, 86, 90, 99}
)
EXPECTED_VAL_CLASSES = (7, 13, 16, 27, 28, 31, 33, 34, 39, 42, 45, 51, 52, 53, 60, 65, 75, 86, 90, 99)
EXPECTED_COUNTS = {
    "train_class_count": 84,
    "validation_class_count": 20,
    "train_sketches": 46624,
    "train_photos": 58950,
    "validation_sketches": 10963,
    "validation_photos": 13999,
    "canonical_positive_photo_pool": 8400,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def tensor_hash(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode())
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def safe_git_state() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD").strip(),
        "status": _git("status", "--porcelain", "--untracked-files=all").splitlines(),
    }


def fresh_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = ROOT / "outputs" / f"{OUT_PREFIX}{stamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = ROOT / "outputs" / f"{OUT_PREFIX}{stamp}_{suffix:02d}"
    candidate.mkdir(parents=True)
    return candidate


def compose_arm(arm: str) -> Any:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="train_frozen_prompt",
            overrides=[
                f"+experiments={arm}",
                "device=cuda",
                "tracking.mode=disabled",
                "hydra.job.chdir=false",
            ],
        )


def seed42() -> None:
    import random

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def snapshot_parameters(module: torch.nn.Module) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "sha256": tensor_hash(parameter),
        }
        for name, parameter in module.named_parameters()
    }


def snapshot_state(module: torch.nn.Module) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_hash(value),
        }
        for name, value in module.state_dict().items()
        if isinstance(value, Tensor)
    }


def assert_same_snapshot(before: dict[str, Any], after: dict[str, Any], name: str) -> None:
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [key for key in changed if before.get(key) != after.get(key)]
        raise AssertionError(f"{name} changed without an optimizer step: {changed[:5]}")


def finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise AssertionError(f"{name} is nonfinite")


def memory_peak(device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _positive_paths(batch: dict[str, Any], index: int) -> tuple[str, ...]:
    values = batch["positive_photo_paths"]
    # The default PyTorch collator transposes the tuple-of-positive-paths
    # field: [num_positives][batch], while images remain [batch, num_positives].
    if isinstance(values, (list, tuple)) and len(values) == int(batch["label"].shape[0]):
        row = values[index]
    elif isinstance(values, (list, tuple)) and values:
        row = [group[index] for group in values]
    else:
        raise AssertionError("positive path trace has an invalid collated shape")
    if isinstance(row, str):
        return (row,)
    return tuple(str(value) for value in row)


def batch_trace(batch: dict[str, Any], pairing: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(batch["label"].shape[0])):
        sketch = str(batch["sketch_path"][index])
        positives = _positive_paths(batch, index)
        negative = str(batch["negative_photo_path"][index])
        label = int(batch["label"][index])
        negative_label = int(batch["negative_label"][index])
        if sketch not in pairing:
            raise AssertionError(f"batch sketch is absent from canonical pairing: {sketch}")
        if not positives or any(Path(path).resolve().parent == Path("/") for path in positives):
            raise AssertionError("invalid positive path in batch")
        if negative_label == label:
            raise AssertionError("negative sampler returned the query class")
        if not Path(sketch).is_file() or not all(Path(path).is_file() for path in positives) or not Path(negative).is_file():
            raise AssertionError("actual training batch contains a missing image")
        rows.append(
            {
                "sketch_path": sketch,
                "sketch_relative": str(Path(sketch).resolve().relative_to(data_root.resolve())),
                "label": label,
                "positive_photo_paths": list(positives),
                "positive_photo_sha256": [sha256_file(Path(path)) for path in positives],
                "negative_photo_path": negative,
                "negative_photo_sha256": sha256_file(Path(negative)),
                "negative_label": negative_label,
            }
        )
    if len(rows) != 32:
        raise AssertionError(f"actual training batch size is {len(rows)}, not 32")
    return rows


def validate_data_and_pairing(trainer: Any, args: Any) -> tuple[Any, dict[int, str], Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = trainer.load_data_config(trainer._path(args.data_config))
    split, names, split_identity, manifest_identity = trainer._load_split(data, args)
    if split_identity["sha256"] != EXPECTED_SPLIT_SHA256:
        raise AssertionError(f"pseudo split identity mismatch: {split_identity['sha256']}")
    expected_classes = set(EXPECTED_TRAIN_CLASSES) | set(EXPECTED_VAL_CLASSES)
    if tuple(split.train_class_ids) != EXPECTED_TRAIN_CLASSES or tuple(split.validation_class_ids) != EXPECTED_VAL_CLASSES:
        raise AssertionError("pseudo-train/validation class IDs differ from protocol")
    counts = {
        "train_class_count": len(split.train_class_ids),
        "validation_class_count": len(split.validation_class_ids),
        "train_sketches": len(split.train_sketch_entries),
        "train_photos": len(split.train_photo_entries),
        "validation_sketches": len(split.validation_sketch_entries),
        "validation_photos": len(split.validation_photo_entries),
    }
    if counts != {key: EXPECTED_COUNTS[key] for key in counts}:
        raise AssertionError(f"pseudo split counts mismatch: {counts}")
    if set(names) != expected_classes:
        raise AssertionError("class map does not contain exactly the 104 protocol classes")

    pairing_path = trainer._path(args.pairing_manifest_path)
    pairing_sha = sha256_file(pairing_path)
    if pairing_sha != EXPECTED_PAIRING_SHA256:
        raise AssertionError(f"pairing manifest hash mismatch: {pairing_sha}")
    pairing = trainer.load_pairing_manifest(
        pairing_path,
        dataset_root=data.root,
        sketch_entries=split.train_sketch_entries,
        photo_entries=split.train_photo_entries,
    )
    if len(pairing) != EXPECTED_COUNTS["train_sketches"]:
        raise AssertionError("canonical pairing does not cover all 84-class training sketches")
    positive_pool = {str(entry.path.resolve()) for entry in pairing.values()}
    if len(positive_pool) != EXPECTED_COUNTS["canonical_positive_photo_pool"]:
        raise AssertionError(f"canonical positive pool size is {len(positive_pool)}, not 8400")
    return data, names, split, split_identity, manifest_identity, {
        "path": str(pairing_path),
        "sha256": pairing_sha,
        "records": len(pairing),
        "unique_photo_pool": len(positive_pool),
        "pool_paths": positive_pool,
        "mapping": pairing,
    }


def assert_config(args: Any, arm: str) -> None:
    expected_mode = "hard" if arm == "semantic_text_S0" else "soft"
    expected_lambda = 1.0 if arm == "semantic_text_S2" else 0.0
    checks = {
        "role": str(args.experiment_role) == arm,
        "campaign": str(args.experiment_campaign) == CAMPAIGN,
        "run_kind": str(args.run_kind) == "primary",
        "device": str(args.device) == "cuda",
        "model_name": str(args.model_name) == "ViT-B-32-quickgelu",
        "pretrained": str(args.pretrained) == "openai",
        "view_mode": str(args.sketch_view_mode) == "full_full",
        "text_mode": str(args.text_mode) == expected_mode,
        "lambda_anchor": float(args.lambda_anchor) == expected_lambda,
        "max_steps": int(args.max_steps) == 3600,
        "probe_steps": tuple(int(value) for value in args.probe_steps) == (0, 600, 1200, 1800, 2400, 3000, 3600),
        "batch_size": int(args.batch_size) == 32,
        "num_workers": int(args.num_workers) == 4,
        "pin_memory": bool(args.pin_memory),
        "drop_last": bool(args.drop_last),
        "pseudo_val_seed": int(args.pseudo_val_seed) == 3407,
        "seed": int(args.seed) == 42,
        "tracking_disabled": str(args.tracking.mode) == "disabled",
        "official_unseen": not bool(args.official_unseen_used_for_selection),
        "no_train_masks": dict(args.mask_policy)["train_fractions"] == [] and dict(args.mask_policy)["train_seed"] is None,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{arm} resolved config failed: {failed}")


def run_arm(
    trainer: Any,
    args: Any,
    arm: str,
    shared: dict[str, Any],
    device: torch.device,
    out: Path,
) -> dict[str, Any]:
    assert_config(args, arm)
    # This is the production initialization order: seed, data/split/pairing,
    # CLIP, prompt wrapper, loader, then semantic text setup.
    seed42()
    data, names, split, split_identity, manifest_identity, pairing = validate_data_and_pairing(trainer, args)
    if shared:
        for key in ("split_identity", "manifest_identity", "pairing"):
            if key == "pairing":
                if shared[key]["sha256"] != pairing["sha256"] or shared[key]["records"] != pairing["records"]:
                    raise AssertionError(f"{arm} pairing identity differs from previous arm")
            elif shared[key] != (split_identity if key == "split_identity" else manifest_identity):
                raise AssertionError(f"{arm} data identity differs from previous arm")
    else:
        shared.update({"split_identity": split_identity, "manifest_identity": manifest_identity, "pairing": {key: pairing[key] for key in ("sha256", "records", "unique_photo_pool")}})

    torch.cuda.reset_peak_memory_stats(device)
    clip = trainer.load_frozen_clip(model_name=str(args.model_name), pretrained=args.pretrained, device=device)
    model, transform = trainer._prompt_model(args, device, clip)
    if not isinstance(model, trainer.FrozenPromptModel):
        raise AssertionError("semantic preflight did not construct FrozenPromptModel")
    clip_param_before = snapshot_parameters(clip.encoder.model)
    clip_state_before = snapshot_state(clip.encoder.model)
    clip_requires_grad = [name for name, value in clip.encoder.model.named_parameters() if value.requires_grad]
    if clip_requires_grad:
        raise AssertionError(f"original CLIP parameters are trainable: {clip_requires_grad[:5]}")
    if model.trainable_parameter_names != ("sketch_prompt", "photo_prompt"):
        raise AssertionError(f"unexpected visual trainables: {model.trainable_parameter_names}")
    if model.sketch_prompt.shape != (3, 768) or model.photo_prompt.shape != (3, 768):
        raise AssertionError("visual prompts are not 3x768")
    prompt_before = snapshot_parameters(model)

    loader = trainer._loader(
        (split.train_sketch_entries, split.train_photo_entries),
        transform,
        args,
        train=True,
        seed=42,
        positive_pairing=pairing["mapping"],
    )
    if len(loader) != 1457:
        raise AssertionError(f"unexpected 84-class DataLoader length: {len(loader)}")
    batch = next(iter(loader))
    trace = batch_trace(batch, pairing["mapping"], Path(data.root))
    if any(row["positive_photo_paths"][0] not in pairing["pool_paths"] for row in trace):
        raise AssertionError("positive sample escaped the canonical 8400-photo pool")
    del loader
    gc.collect()

    # Match production semantic text construction, including hard T0 and the
    # trainable SoftPromptTextBank's prefix initialization/parity check.
    text_bank: Any
    if str(args.text_mode) == "hard":
        text_bank = trainer.encode_class_text_bank(
            clip.encoder,
            clip.tokenizer,
            {class_id: names[class_id] for class_id in split.train_class_ids},
            prompt_template=str(args.prompt_template),
        )
    else:
        text_bank = trainer.SoftPromptTextBank(
            clip.encoder,
            clip.tokenizer,
            {class_id: names[class_id] for class_id in split.train_class_ids},
            prompt_length=int(args.soft_prompt_length),
            strict_prefix=True,
        ).to(device)
    hard_bank = text_bank if isinstance(text_bank, trainer.EncodedTextBank) else trainer.encode_class_text_bank(
        clip.encoder,
        clip.tokenizer,
        {class_id: names[class_id] for class_id in split.train_class_ids},
        prompt_template=str(args.prompt_template),
    )
    fixed_bank = trainer.clone_fixed_text_bank(hard_bank.embeddings, device=device)
    if fixed_bank.is_inference() or fixed_bank.requires_grad:
        raise AssertionError("fixed T0 is not a normal detached clone")
    if fixed_bank.shape != (84, 512) or not torch.isfinite(fixed_bank).all().item():
        raise AssertionError("fixed T0 has wrong shape or nonfinite values")
    fixed_norms = fixed_bank.norm(dim=-1)
    if (fixed_norms <= 0).any().item():
        raise AssertionError("fixed T0 contains a zero-norm class")
    class_ids = tuple(int(value) for value in hard_bank.labels.tolist())
    if class_ids != EXPECTED_TRAIN_CLASSES:
        raise AssertionError("T0 class order is not sorted protocol pseudo-train order")
    learned_initial = None
    initial_parity_max_error = None
    context_before = None
    prefix_token_ids: tuple[int, ...] | None = None
    if isinstance(text_bank, trainer.SoftPromptTextBank):
        prefix_token_ids = tuple(int(value) for value in text_bank.prefix_token_ids)
        if prefix_token_ids != (320, 1125, 539, 320):
            raise AssertionError(f"unexpected 'a photo of a' token prefix: {prefix_token_ids}")
        if text_bank.context.shape != (4, 512) or not text_bank.context.requires_grad:
            raise AssertionError("soft context is not trainable 4x512")
        context_before = snapshot_parameters(text_bank)
        learned_initial = text_bank().detach().clone()
        if learned_initial.is_inference() or not torch.isfinite(learned_initial).all().item():
            raise AssertionError("initial learned bank is invalid")
        initial_parity_max_error = float((learned_initial.float() - fixed_bank.float()).abs().max().item())
        if initial_parity_max_error > 1e-6:
            raise AssertionError(f"initial T(C) parity error is {initial_parity_max_error}")
    else:
        if text_bank.embeddings.shape != (84, 512):
            raise AssertionError("hard T0 has wrong shape")

    # This mapping is built from the same production model/text objects, but it
    # is never stepped.  It proves the active groups and expected counts.
    _, optimizer_groups = trainer.build_optimizer_parameter_groups(model, text_bank if isinstance(text_bank, trainer.SoftPromptTextBank) else None, args)
    trainable = dict(trainer._parameter_names(model, text_bank if isinstance(text_bank, trainer.SoftPromptTextBank) else None))
    trainable_names = tuple(name for name, value in trainable.items() if value.requires_grad)
    expected_count = 4608 if arm == "semantic_text_S0" else 6656
    if sum(value.numel() for value in trainable.values() if value.requires_grad) != expected_count:
        raise AssertionError(f"{arm} trainable count is not {expected_count}")
    if arm != "semantic_text_S0" and trainable_names != ("sketch_prompt", "photo_prompt", "soft_prompt.context"):
        raise AssertionError(f"unexpected {arm} trainables: {trainable_names}")
    trainable_before = snapshot_parameters(model)
    if isinstance(text_bank, trainer.SoftPromptTextBank):
        trainable_before.update({f"soft_prompt.{name}": value for name, value in snapshot_parameters(text_bank).items()})

    # One production-shaped no-update graph: 64 full query images, 64 photos,
    # and one 84-class text bank reused by both query views.
    model.train()
    query_images = torch.cat((batch["sketch"], batch["sketch"]), dim=0).to(device)
    photo_images = torch.cat((batch["positive_photos"][:, 0], batch["negative_photo"]), dim=0).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    query_values = model(query_images).chunk(2, dim=0)
    photo_values = model.encode_photo(photo_images).split((32, 32), dim=0)
    learned_bank = text_bank() if isinstance(text_bank, trainer.SoftPromptTextBank) else None
    if learned_bank is not None:
        if learned_bank.is_inference() or learned_bank.grad_fn is None:
            raise AssertionError("learned text bank is not a grad-bearing normal forward")
        bank = learned_bank
        bank_labels = text_bank.class_labels.to(device)
    else:
        bank = text_bank.embeddings.to(device)
        bank_labels = text_bank.labels.to(device)
    if bank.shape != (84, 512):
        raise AssertionError("actual loss text bank is not [84,512]")
    labels = batch["label"].long().to(device)
    positive, negative = photo_values

    def losses(current: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        query_normalized = F.normalize(current, dim=-1)
        positive_normalized = F.normalize(positive, dim=-1)
        negative_normalized = F.normalize(negative, dim=-1)
        rank = F.softplus(
            float(args.margin)
            - (query_normalized * positive_normalized).sum(-1)
            + (query_normalized * negative_normalized).sum(-1)
        ).mean()
        cls, _ = trainer.jepa_text_classification_loss(
            current,
            bank,
            bank_labels,
            labels,
            temperature=float(args.tau_cls),
            detach_text=not isinstance(text_bank, trainer.SoftPromptTextBank),
        )
        return rank, cls, rank + cls

    view_losses = [losses(current) for current in query_values]
    rank = torch.stack([value[0] for value in view_losses]).mean()
    cls = torch.stack([value[1] for value in view_losses]).mean()
    task = float(args.lambda_rank) * rank + float(args.lambda_cls) * cls
    anchor = task.new_zeros(())
    if learned_bank is not None:
        anchor = trainer.text_anchor_loss(learned_bank, fixed_bank)
    total = task + float(args.lambda_anchor) * anchor
    for value, name in ((rank, "rank"), (cls, "classification"), (anchor, "anchor"), (total, "total")):
        finite(value, name)
    for value, name in zip(query_values + photo_values + (bank,), ("query1", "query2", "positive", "negative", "bank"), strict=True):
        finite(value, name)

    parameters = [trainable[name] for name in trainable_names]
    task_grads = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    total_grads = torch.autograd.grad(
        total, parameters, retain_graph=learned_bank is not None, allow_unused=True
    )
    task_gradients: dict[str, float | None] = {}
    total_gradients: dict[str, float | None] = {}
    for name, task_grad, total_grad in zip(trainable_names, task_grads, total_grads, strict=True):
        if task_grad is None or total_grad is None:
            raise AssertionError(f"missing gradient for {name}")
        finite(task_grad, f"task gradient {name}")
        finite(total_grad, f"total gradient {name}")
        task_gradients[name] = float(task_grad.detach().float().norm().item())
        total_gradients[name] = float(total_grad.detach().float().norm().item())
        if task_gradients[name] <= 0.0 or total_gradients[name] <= 0.0:
            raise AssertionError(f"non-positive task/total gradient for {name}")
    anchor_gradients: dict[str, float | None] = {}
    if learned_bank is not None:
        anchor_only = torch.autograd.grad(anchor, [text_bank.context], allow_unused=True)[0]
        if anchor_only is None:
            raise AssertionError("anchor did not connect to soft context")
        finite(anchor_only, "anchor context gradient")
        anchor_gradients["soft_prompt.context"] = float(anchor_only.detach().float().norm().item())

    clip_param_after = snapshot_parameters(clip.encoder.model)
    clip_state_after = snapshot_state(clip.encoder.model)
    assert_same_snapshot(clip_param_before, clip_param_after, "all original CLIP parameters")
    assert_same_snapshot(clip_state_before, clip_state_after, "all original CLIP state")
    if any(parameter.grad is not None for parameter in clip.encoder.model.parameters()):
        raise AssertionError("a frozen original CLIP parameter received a .grad")
    trainable_after = snapshot_parameters(model)
    if isinstance(text_bank, trainer.SoftPromptTextBank):
        trainable_after.update({f"soft_prompt.{name}": value for name, value in snapshot_parameters(text_bank).items()})
    assert_same_snapshot(trainable_before, trainable_after, "trainable prompts/context")
    torch.cuda.synchronize(device)
    memory = memory_peak(device)
    if memory["allocated_bytes"] > 16 * 1024**3:
        raise AssertionError(f"actual batch-32 graph exceeded 16 GiB: {memory['allocated_bytes']} bytes")

    result = {
        "role": arm,
        "config": OmegaConf.to_container(args, resolve=True),
        "resolved_config_sha256": json_hash(OmegaConf.to_container(args, resolve=True)),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "index": device.index,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
        },
        "data": {
            "dataset": str(data.name),
            "root": str(data.root),
            "split_identity": split_identity,
            "manifest_identity": manifest_identity,
            "counts": {**EXPECTED_COUNTS, **{key: value for key, value in {
                "train_class_count": len(split.train_class_ids),
                "validation_class_count": len(split.validation_class_ids),
                "train_sketches": len(split.train_sketch_entries),
                "train_photos": len(split.train_photo_entries),
                "validation_sketches": len(split.validation_sketch_entries),
                "validation_photos": len(split.validation_photo_entries),
            }.items()}},
            "official_test_loaded": False,
            "official_test_used": False,
            "loader": {
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "shuffle": True,
                "drop_last": bool(args.drop_last),
                "length": 1457,
                "canonical_pairing_sha256": pairing["sha256"],
                "canonical_positive_photo_pool": pairing["unique_photo_pool"],
                "first_batch_trace": trace,
            },
        },
        "clip": {
            "weights_path": str(CLIP_PATH),
            "weights_sha256": EXPECTED_CLIP_SHA256,
            "weights_bytes": CLIP_PATH.stat().st_size,
            "weights_sha256_verified": True,
            "model_name": str(args.model_name),
            "pretrained": str(args.pretrained),
            "all_original_parameter_snapshot_before": clip_param_before,
            "all_original_parameter_snapshot_after": clip_param_after,
            "all_original_state_snapshot_before": clip_state_before,
            "all_original_state_snapshot_after": clip_state_after,
            "all_original_parameters_frozen": True,
            "all_original_parameters_byte_identical": True,
            "all_original_grads_none": True,
        },
        "initialization": {
            "visual_prompt_shapes": {"sketch_prompt": list(model.sketch_prompt.shape), "photo_prompt": list(model.photo_prompt.shape)},
            "visual_prompt_parameter_count": 4608,
            "trainable_parameter_count": sum(value.numel() for value in trainable.values() if value.requires_grad),
            "trainable_parameter_names": list(trainable_names),
            "prompt_snapshot_before": prompt_before,
            "prompt_snapshot_after_no_update": snapshot_parameters(model),
            "soft_context_snapshot_before": context_before,
            "prompt_initialization_from_seed42": True,
            "context_initialization_from_prefix": prefix_token_ids is not None,
            "prefix_token_ids": None if prefix_token_ids is None else list(prefix_token_ids),
            "initial_learned_bank_sha256": None if learned_initial is None else tensor_hash(learned_initial),
            "initial_context_sha256": None if context_before is None else context_before["context"]["sha256"],
            "initial_text_bank_max_abs_error": initial_parity_max_error,
        },
        "text_bank": {
            "class_count": int(fixed_bank.shape[0]),
            "dimension": int(fixed_bank.shape[1]),
            "class_ids": list(class_ids),
            "fixed_t0_sha256": tensor_hash(fixed_bank),
            "fixed_t0_is_normal_non_inference_detached_clone": not fixed_bank.is_inference() and not fixed_bank.requires_grad,
            "fixed_t0_zero_anchor_valid": bool((fixed_norms > 0).all().item()),
            "fixed_t0_finite": bool(torch.isfinite(fixed_bank).all().item()),
            "learned_bank_forward_once_for_loss": learned_bank is not None,
            "learned_bank_requires_grad": None if learned_bank is None else bool(learned_bank.requires_grad),
            "learned_bank_has_grad_fn": None if learned_bank is None else learned_bank.grad_fn is not None,
            "anchor_formula": "mean_c(1-cosine(T_learned[c], stopgrad(T0[c])))",
            "anchor_reduction": "84 classes once per update, not per view or batch",
            "lambda_anchor": float(args.lambda_anchor),
        },
        "loss": {
            "query_views": 2,
            "query_batch_total": 64,
            "photo_batch_total": 64,
            "rank_loss": float(rank.detach().item()),
            "classification_loss": float(cls.detach().item()),
            "anchor_loss": float(anchor.detach().item()),
            "total_loss": float(total.detach().item()),
            "all_finite": True,
            "mask_training_called": False,
        },
        "gradients": {
            "task_gradients": task_gradients,
            "total_gradients": total_gradients,
            "anchor_gradients": anchor_gradients,
            "optimizer_step_called": False,
            "scheduler_step_called": False,
            "optimizer_updates": 0,
        },
        "memory": memory,
        "status": "PASS",
    }
    return result


def main() -> int:
    out = fresh_output()
    result: dict[str, Any] = {
        "status": "FAILED",
        "preflight_kind": "REAL_CLIP_SEMANTIC_TEXT_NO_UPDATE",
        "authorization": "userTrainS0S2",
        "campaign": CAMPAIGN,
        "output_dir": str(out),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "WANDB_MODE")},
        "official_test_loaded": False,
        "online_tracking": False,
        "arms": {},
        "errors": {},
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to substitute CPU")
        device = torch.device("cuda")
        if not CLIP_PATH.is_file():
            raise FileNotFoundError(f"expected local CLIP weights are missing: {CLIP_PATH}")
        clip_bytes = CLIP_PATH.stat().st_size
        clip_sha = sha256_file(CLIP_PATH)
        if clip_bytes != EXPECTED_CLIP_BYTES or clip_sha != EXPECTED_CLIP_SHA256:
            raise RuntimeError(f"local CLIP identity mismatch: bytes={clip_bytes}, sha256={clip_sha}")
        from spica.provenance import source_snapshot
        import spica.train_frozen_prompt as trainer

        snapshot = source_snapshot(ROOT)
        result["source"] = {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "source_snapshot_hash": snapshot["sha256"],
            "source_snapshot_file_count": snapshot["file_count"],
            "git": safe_git_state(),
            "source_certification": "NOT_FINAL_PENDING_PARENT_GATE",
            "note": "This is a working-tree preflight identity, not a final parent source certificate.",
        }
        result["clip_identity"] = {"path": str(CLIP_PATH), "bytes": clip_bytes, "sha256": clip_sha}
        shared: dict[str, Any] = {}
        for arm in ARMS:
            args = compose_arm(arm)
            try:
                result["arms"][arm] = run_arm(trainer, args, arm, shared, device, out)
            except Exception as error:  # Keep per-arm evidence and continue to S1/S2.
                result["errors"][arm] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(limit=12),
                }
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        if result["errors"]:
            result["status"] = "FAIL"
        elif set(result["arms"]) != set(ARMS):
            result["status"] = "FAIL"
        else:
            traces = [result["arms"][arm]["data"]["loader"]["first_batch_trace"] for arm in ARMS]
            if not (traces[0] == traces[1] == traces[2]):
                raise AssertionError("S0/S1/S2 first DataLoader traces differ")
            prompt_snapshots = [result["arms"][arm]["initialization"]["prompt_snapshot_before"] for arm in ARMS]
            for arm_snapshot in prompt_snapshots[1:]:
                for name in ("sketch_prompt", "photo_prompt"):
                    if arm_snapshot[name]["sha256"] != prompt_snapshots[0][name]["sha256"]:
                        raise AssertionError(f"visual prompt initialization differs across arms: {name}")
            context_s1 = result["arms"]["semantic_text_S1"]["initialization"]["soft_context_snapshot_before"]["context"]["sha256"]
            context_s2 = result["arms"]["semantic_text_S2"]["initialization"]["soft_context_snapshot_before"]["context"]["sha256"]
            if context_s1 != context_s2:
                raise AssertionError("S1/S2 context initialization differs")
            result["cross_arm"] = {
                "first_batch_trace_byte_identity": True,
                "visual_prompt_initialization_byte_identity": True,
                "s1_s2_context_initialization_byte_identity": True,
                "all_arms_same_loader_identity": True,
            }
            result["status"] = "PASS"
    except Exception as error:
        result["errors"]["global"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=16),
        }
        result["status"] = "FAIL"
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "preflight_config.json").write_text(json.dumps({
            "preflight_kind": result["preflight_kind"],
            "authorization": result["authorization"],
            "campaign": CAMPAIGN,
            "clip_identity": result.get("clip_identity"),
            "source": result.get("source"),
            "environment": result["environment"],
            "official_test_loaded": False,
            "online_tracking": False,
        }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (out / "preflight_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(out), "errors": result["errors"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
