"""Primary frozen-CLIP visual-prompt campaign.

The predictor has a sketch-only inference API. Photo and text branches exist
only to form training losses and to build identity-checked evaluation caches.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .config.data import load_data_config
from .data.datasets import MultiPositiveRetrievalTrainDataset, RetrievalEvalDataset
from .data.manifest import read_class_map, read_manifest
from .data.splits import make_classwise_retrieval_split, split_manifest_identity
from .evaluation.frozen_prompt import (
    cache_identity,
    encode_prompted_loader,
    evaluate_prompted,
    geometry_payload,
    hash_state,
    load_prompt_cache,
    save_prompt_cache,
)
from .evaluation.text_bank import SoftPromptTextBank, encode_class_text_bank
from .models.clip import FrozenClipEncoder, load_frozen_clip, load_trainable_sketch_hidden_encoder
from .models.frozen_prompt import FrozenPromptModel
from .models.jepa import classification_accuracy, jepa_text_classification_loss
from .provenance import capture_provenance, capture_rng_state, restore_rng_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")
CAMPAIGN = "frozen_prompt_probe_2026-09-03"
PROBE_STEPS = (0, 15, 44, 73, 100, 250, 500, 1000, 1800, 5400)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _path(value: object) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() or candidate.exists() else PROJECT_ROOT / candidate


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    random.seed(torch.initial_seed() % (2**32) + worker_id)


def _validate(args: DictConfig) -> None:
    role = str(args.experiment_role)
    roles = {
        "frozen_prompt_FP0", "frozen_prompt_FP1", "frozen_prompt_FP2",
        "frozen_prompt_FP3", "frozen_prompt_FP4", "frozen_prompt_FP5",
        "frozen_prompt_FP_LN",
    }
    if role not in roles:
        raise ValueError(f"experiment_role must be exactly one of {sorted(roles)}")
    if str(args.experiment_campaign) != CAMPAIGN:
        raise ValueError("wrong frozen-prompt campaign")
    if tuple(int(x) for x in args.probe_steps) != PROBE_STEPS and not bool(args.allow_short_run):
        raise ValueError(f"probe_steps must be exactly {list(PROBE_STEPS)}")
    if int(args.pseudo_val_seed) != 3407 or str(args.train_class_scope) != "pseudo_train":
        raise ValueError("selection requires pseudo-train classes and split seed 3407")
    expected = {
        "frozen_prompt_FP0": (0, "vanilla", "none", False, 0.0),
        "frozen_prompt_FP1": (3, "prompt_only", "none", False, 0.0),
        "frozen_prompt_FP2": (3, "prompt_only", "hard", False, 1.0),
        "frozen_prompt_FP3": (3, "prompt_only", "soft", False, 1.0),
        "frozen_prompt_FP4": (0, "vanilla", "soft", False, 1.0),
        "frozen_prompt_FP5": (0, "early_adapt_then_freeze", "hard", False, 1.0),
        "frozen_prompt_FP_LN": (3, "prompt_plus_layernorm", "hard", True, 1.0),
    }[role]
    observed = (int(args.visual_prompt_length), str(args.prompt_mode), str(args.text_mode), bool(args.train_visual_layernorm), float(args.lambda_cls))
    if observed != expected:
        raise ValueError(f"{role} has ambiguous treatment: observed {observed}, expected {expected}")
    if role == "frozen_prompt_FP0" and int(args.max_steps) != 0:
        raise ValueError("FP0 is an evaluation-only cell and must not create optimizer updates")
    if role != "frozen_prompt_FP0" and int(args.max_steps) <= 0:
        raise ValueError("trainable cells require positive max_steps")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0 or int(args.num_workers) < 0:
        raise ValueError("invalid loader settings")
    if float(args.learning_rate) <= 0 or float(args.soft_prompt_learning_rate) <= 0:
        raise ValueError("learning rates must be positive")
    if float(args.lambda_cls) > 0 and str(args.text_mode) not in {"hard", "soft"}:
        raise ValueError("classification loss requires a text bank")


def _load_split(data: Any, args: DictConfig):
    names = read_class_map(data.train.class_map)
    sketches = read_manifest(data.train.sketch_manifest, data.root)
    photos = read_manifest(data.train.photo_manifest, data.root)
    split = make_classwise_retrieval_split(
        sketches, photos, names,
        num_validation_classes=int(args.pseudo_val_num_classes),
        seed=int(args.pseudo_val_seed),
    )
    manifests = {"train_sketch": data.train.sketch_manifest, "train_photo": data.train.photo_manifest, "train_class_map": data.train.class_map}
    identity = split_manifest_identity(split, dataset_name=data.name, dataset_root=data.root, manifest_paths=manifests)
    if set(split.train_class_ids) & set(split.validation_class_ids):
        raise AssertionError("pseudo-train and pseudo-unseen classes overlap")
    return split, names, identity


def _loader(entries, transform, args: DictConfig, *, train: bool = False, seed: int = 42) -> DataLoader:
    if train:
        dataset = MultiPositiveRetrievalTrainDataset(
            entries[0], entries[1], transform, transform, num_positive_photos=1,
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True,
                          num_workers=int(args.num_workers), pin_memory=bool(args.pin_memory),
                          drop_last=bool(args.drop_last), generator=generator,
                          worker_init_fn=_worker_seed, persistent_workers=int(args.num_workers) > 0)
    return DataLoader(RetrievalEvalDataset(entries, transform), batch_size=int(args.eval_batch_size),
                      shuffle=False, num_workers=int(args.num_workers), pin_memory=bool(args.pin_memory),
                      persistent_workers=int(args.num_workers) > 0)


def _prompt_model(
    args: DictConfig,
    device: torch.device,
    photo_clip: Any,
) -> tuple[torch.nn.Module, Any]:
    if str(args.prompt_mode) == "early_adapt_then_freeze":
        bundle = load_trainable_sketch_hidden_encoder(
            model_name=str(args.model_name), pretrained=args.pretrained, device=device,
            mode="partial", unfreeze_depth=4,
        )
        return _EarlyAdaptModel(bundle).to(device), bundle.transform
    # Reuse the already-loaded frozen CLIP visual tower. This keeps one model
    # copy in memory while the same frozen weights serve text supervision and
    # prompted image encoding.
    model = FrozenPromptModel(
        photo_clip.encoder.model.visual,
        prompt_length=int(args.visual_prompt_length),
        train_visual_layernorm=bool(args.train_visual_layernorm),
    ).to(device)
    return model, photo_clip.transform


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    return hash_state({name: value for name, value in model.state_dict().items()})


def _classify(query: Tensor, bank: Tensor, bank_labels: Tensor, labels: Tensor, tau: float) -> tuple[Tensor, Tensor]:
    return jepa_text_classification_loss(query, bank, bank_labels, labels, temperature=tau, detach_text=False)


def _ranking_loss(query: Tensor, positive: Tensor, negative: Tensor, margin: float) -> Tensor:
    """Ranking loss whose photo prompts remain in the loss graph."""
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    return F.softplus(margin - (query * positive).sum(-1) + (query * negative).sum(-1)).mean()


class _FrozenEncoderAdapter:
    def __init__(self, encoder: FrozenClipEncoder) -> None:
        self.encoder = encoder
        self.device = encoder.device

    def eval(self) -> None:
        self.encoder.eval()

    def __call__(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def encode_photo(self, images: Tensor) -> Tensor:
        return self.encoder(images)


def _encode_vanilla(encoder: FrozenClipEncoder, loader: DataLoader) -> Any:
    return encode_prompted_loader(_FrozenEncoderAdapter(encoder), loader)


class _EarlyAdaptModel(torch.nn.Module):
    """The existing depth-4, q=z0 semantic-origin reference."""
    def __init__(self, bundle: Any) -> None:
        super().__init__()
        self.encoder = bundle.encoder
        self.embedding_dim = bundle.encoder.embedding_dim
        self.projection = bundle.projection

    @property
    def device(self) -> torch.device:
        return self.projection.matrix.device

    def forward(self, images: Tensor) -> Tensor:
        return F.normalize(self.projection(self.encoder(images)), dim=-1)

    def encode_photo(self, images: Tensor) -> Tensor:
        return self.forward(images)

    def eval(self) -> "_EarlyAdaptModel":
        super().eval()
        return self


def _save_checkpoint(path: Path, *, model: torch.nn.Module, text_bank: SoftPromptTextBank | None,
                     optimizer: torch.optim.Optimizer | None, scheduler: Any, step: int, args: DictConfig,
                     split_identity: dict[str, object], manifest_identity: dict[str, object],
                     loader_generator: torch.Generator, provenance: dict[str, Any],
                     text_names: tuple[str, ...], initial_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_state = model.state_dict()
    # Prompt cells need only their trainable state; the pretrained CLIP
    # backbone is reconstructed from model/pretrained identity on resume.
    compact = True
    model_state = {
        name: value.detach().cpu()
        for name, value in full_state.items()
        if not compact or name in {n for n, p in model.named_parameters() if p.requires_grad}
    }
    text_state = None if text_bank is None else {name: value.detach().cpu() for name, value in text_bank.state_dict().items()}
    payload = {
        "format_version": 1, "model_type": "frozen_prompt", "step": step,
        "model_state_dict": model_state, "model_state_complete": not compact,
        "soft_prompt_state_dict": text_state,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "rng_state": capture_rng_state(loader_generator), "global_step": step,
        "resolved_config": OmegaConf.to_container(args, resolve=True),
        "experiment_role": str(args.experiment_role), "campaign": str(args.experiment_campaign),
        "data_split_identity": split_identity, "data_manifest_identity": manifest_identity,
        "model_state_hash": _state_hash(model), "initial_model_state_hash": initial_hash,
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "text": {"soft_prompt": text_bank is not None, "prompt_length": 0 if text_bank is None else text_bank.prompt_length,
                 "initialization": "first prompt_length token-embedding rows for hard prefix 'a photo of a'; normal noise tail if shorter",
                 "class_names_used_for_training": list(text_names),
                 "parameter_count": 0 if text_bank is None else text_bank.parameter_count},
        "provenance": provenance,
    }
    torch.save(payload, path)


def _restore_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    text_bank: SoftPromptTextBank | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    args: DictConfig,
    split_identity: dict[str, object],
    manifest_identity: dict[str, object],
    loader_generator: torch.Generator,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("invalid frozen-prompt checkpoint")
    if payload.get("model_type") != "frozen_prompt":
        raise ValueError("checkpoint is not a frozen-prompt checkpoint")
    if payload.get("experiment_role") != str(args.experiment_role):
        raise ValueError("checkpoint experiment role does not match")
    if payload.get("campaign") != str(args.experiment_campaign):
        raise ValueError("checkpoint campaign does not match")
    if payload.get("data_split_identity") != split_identity or payload.get("data_manifest_identity") != manifest_identity:
        raise ValueError("checkpoint data identity does not match")
    model.load_state_dict(payload["model_state_dict"], strict=bool(payload.get("model_state_complete", False)))
    if text_bank is not None:
        state = payload.get("soft_prompt_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing soft-prompt state")
        text_bank.load_state_dict(state, strict=True)
    if optimizer is not None:
        state = payload.get("optimizer_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing optimizer state")
        optimizer.load_state_dict(state)
    if scheduler is not None:
        state = payload.get("scheduler_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing scheduler state")
        scheduler.load_state_dict(state)
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        raise ValueError("checkpoint is missing RNG state")
    restore_rng_state(rng_state, loader_generator)
    return int(payload.get("global_step", payload.get("step", -1)))


def _metrics(evaluation: Any) -> dict[str, Any]:
    return {"full_mAP": evaluation.metrics.mean_average_precision,
            "P@200": evaluation.metrics.precision_at_k.get(200),
            "mAP@200": evaluation.metrics.mean_average_precision_at_k.get(200),
            "average_precision_per_query": evaluation.average_precision_per_query.tolist(),
            "num_queries": evaluation.metrics.num_queries,
            "num_gallery_items": evaluation.metrics.num_gallery_items}


def run(args: DictConfig) -> None:
    _validate(args)
    seed = int(args.seed)
    _seed(seed)
    device = _device(str(args.device))
    data = load_data_config(_path(args.data_config))
    split, names, manifest_identity = _load_split(data, args)
    split_identity = {"seed": split.seed, "train_class_ids": list(split.train_class_ids), "validation_class_ids": list(split.validation_class_ids), "sha256": hashlib.sha256(json.dumps([split.train_class_ids, split.validation_class_ids]).encode()).hexdigest()}
    train_names = {i: names[i] for i in split.train_class_ids}
    photo_clip = load_frozen_clip(model_name=str(args.model_name), pretrained=args.pretrained, device=device)
    prompt_model, transform = _prompt_model(args, device, photo_clip)
    photo_model: Any = prompt_model
    if str(args.experiment_role) == "frozen_prompt_FP5":
        photo_model = _FrozenEncoderAdapter(photo_clip.encoder)
    train_loader = _loader((split.train_sketch_entries, split.train_photo_entries), transform, args, train=True, seed=seed)
    # Classification geometry is measured on a deterministic small seen-class
    # batch; retrieval probes remain full pseudo-unseen.
    train_eval_loader = _loader(split.train_sketch_entries[: int(args.eval_batch_size)], transform, args)
    val_sketch_loader = _loader(split.validation_sketch_entries, transform, args)
    val_photo_loader = _loader(split.validation_photo_entries, transform, args)
    if len(train_loader) == 0:
        raise ValueError("training loader has no batches")

    # Vanilla references use the independent frozen CLIP image tower, never a
    # prompted cache. All prompted cells re-encode both query and gallery.
    vanilla_sketch = _encode_vanilla(photo_clip.encoder, val_sketch_loader)
    vanilla_photo = _encode_vanilla(photo_clip.encoder, val_photo_loader)
    text_bank: SoftPromptTextBank | None = None
    hard_bank = None
    if str(args.text_mode) == "soft":
        text_bank = SoftPromptTextBank(photo_clip.encoder, photo_clip.tokenizer, train_names, prompt_length=int(args.soft_prompt_length)).to(device)
    elif str(args.text_mode) == "hard":
        hard_bank = encode_class_text_bank(photo_clip.encoder, photo_clip.tokenizer, train_names, prompt_template=str(args.prompt_template))
    trainable = str(args.experiment_role) != "frozen_prompt_FP0"
    optimizer = None
    scheduler = None
    if trainable:
        parameters = [p for p in prompt_model.parameters() if p.requires_grad]
        if text_bank is not None:
            parameters += list(text_bank.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    initial_hash = _state_hash(prompt_model)
    initial_clip_state = {
        name: value.detach().cpu().clone()
        for name, value in prompt_model.named_parameters()
        if name.startswith("visual.") or name.startswith("encoder.visual.")
    }
    if str(args.experiment_role) == "frozen_prompt_FP5":
        # The reference is adapted through step 73, then held fixed. The
        # current trainer's max_steps is 73; later probes are intentionally not
        # fabricated as training points.
        if int(args.max_steps) != 73 and not bool(args.allow_short_run):
            raise ValueError("FP5 reference must end adaptation at step 73")
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_provenance(PROJECT_ROOT, resolved_config=OmegaConf.to_container(args, resolve=True), command=[sys.executable, *sys.argv])
    history: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    loader_generator = train_loader.generator
    if loader_generator is None:
        raise RuntimeError("training loader has no reproducible generator")
    step = 0
    probe_set = set(int(value) for value in args.probe_steps)

    def probe(probe_step: int, loss_rank: float | None = None, loss_cls: float | None = None, cls_acc: float | None = None) -> None:
        checkpoint = output_dir / "checkpoints" / f"frozen_prompt_step{probe_step}.pt"
        _save_checkpoint(checkpoint, model=prompt_model, text_bank=text_bank, optimizer=optimizer, scheduler=scheduler, step=probe_step,
                         args=args, split_identity=split_identity, manifest_identity=manifest_identity,
                         loader_generator=loader_generator, provenance=provenance, text_names=tuple(train_names.values()), initial_hash=initial_hash)
        checkpoint_hash = _hash_file(checkpoint)
        val_sketch = encode_prompted_loader(prompt_model, val_sketch_loader)
        val_photo = encode_prompted_loader(photo_model, val_photo_loader, photo=True)
        prompted_gallery = str(args.experiment_role) != "frozen_prompt_FP5"
        identity = cache_identity(prompt_checkpoint_hash=checkpoint_hash, prompt_length=int(args.visual_prompt_length) if prompted_gallery else 0,
                                  prompt_mode=str(args.prompt_mode), modality="photo", model_name=str(args.model_name),
                                  pretrained=None if args.pretrained is None else str(args.pretrained), data_manifest_identity=manifest_identity)
        cache_path = output_dir / "gallery_cache" / f"photo_step{probe_step}.pt"
        save_prompt_cache(val_photo, cache_path, identity=identity)
        # Read it back immediately; an incompatible gallery must never be
        # silently substituted for the prompted photo gallery.
        loaded_photo = load_prompt_cache(cache_path, expected_identity=identity)
        evaluation = evaluate_prompted(val_sketch, loaded_photo, query_chunk_size=int(args.query_chunk_size), device=device)
        geometry = geometry_payload(
            val_sketch, loaded_photo, sketch_reference=vanilla_sketch,
            photo_reference=vanilla_photo, max_samples=512,
        )
        text_cosine = None
        if hard_bank is not None or text_bank is not None:
            text_values = text_bank() if text_bank is not None else hard_bank.embeddings.to(device)
            text_labels = (text_bank.class_labels if text_bank is not None else hard_bank.labels).to(device)
            train_batch = next(iter(train_eval_loader))
            train_query = prompt_model(train_batch["image"].to(device))
            positions = torch.searchsorted(text_labels, train_batch["label"].long().to(device))
            text_cosine = float((F.normalize(train_query, dim=-1) * F.normalize(text_values[positions], dim=-1)).sum(-1).mean().item())
        prompt_tensors = [
            value.detach().flatten()
            for name, value in prompt_model.named_parameters()
            if "prompt" in name
        ]
        prompt_norm = float(torch.cat(prompt_tensors).norm().item()) if prompt_tensors else 0.0
        prompt_cosine = None
        if int(args.visual_prompt_length) > 0 and prompt_tensors:
            prompt_cosine = F.cosine_similarity(
                prompt_model.sketch_prompt.flatten(), prompt_model.photo_prompt.flatten(), dim=0
            ).item()
        row = {"step": probe_step, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
               "val": _metrics(evaluation), "rank_loss": loss_rank, "classification_loss": loss_cls,
               "seen_classification_accuracy": cls_acc, "pseudo_unseen_classification_accuracy": None,
               "semantic_margin": geometry["cross_modal"]["semantic_margin"], "sketch_photo_cosine": geometry["cross_modal"]["same_class_sketch_photo_cosine"],
               "sketch_text_cosine": text_cosine, "prompt_gradient_norm": prompt_grad_norm(), "prompt_parameter_norm": prompt_norm,
               "prompt_token_cosine_sketch_photo": prompt_cosine, "geometry": geometry,
               "visual_embedding_max_abs_delta": max(
                   float((val_sketch.embeddings - vanilla_sketch.embeddings).abs().max().item()),
                   float((loaded_photo.embeddings - vanilla_photo.embeddings).abs().max().item()),
               ),
               "prompt_attention": prompt_model.attention_diagnostics(next(iter(val_sketch_loader))["image"][: min(8, int(args.eval_batch_size))]) if int(args.visual_prompt_length) and hasattr(prompt_model, "attention_diagnostics") else None,
               "protocol": {"official_unseen_used_for_selection": False, "text_used_for_inference": False, "gallery_reencoded_with_photo_prompt": prompted_gallery,
                            "cache_identity": identity}}
        history.append(row)
        (output_dir / f"probe_step{probe_step}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")

    last_rank = last_cls = last_acc = None
    def prompt_grad_norm() -> float:
        values = [p.grad.detach().norm().item() for name, p in prompt_model.named_parameters() if "prompt" in name and p.grad is not None]
        values += [p.grad.detach().norm().item() for p in (text_bank.parameters() if text_bank is not None else []) if p.grad is not None]
        return float(math.sqrt(sum(value * value for value in values))) if values else 0.0

    if args.resume_checkpoint_path is not None:
        step = _restore_checkpoint(
            _path(args.resume_checkpoint_path), model=prompt_model, text_bank=text_bank,
            optimizer=optimizer, scheduler=scheduler, args=args, split_identity=split_identity,
            manifest_identity=manifest_identity, loader_generator=loader_generator,
        )
    if 0 in probe_set and step == 0:
        probe(0)
    while step < int(args.max_steps):
        for batch in train_loader:
            if step >= int(args.max_steps):
                break
            images = batch["sketch"].to(device)
            positives = batch["positive_photos"][:, 0].to(device)
            negatives = batch["negative_photo"].to(device)
            labels = batch["label"].long().to(device)
            optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
            query = prompt_model(images)
            photo_values = photo_model.encode_photo(torch.cat((positives, negatives), dim=0))
            positive, negative = photo_values.split((images.shape[0], images.shape[0]), dim=0)
            rank = (
                _ranking_loss(query, positive, negative, margin=float(args.margin))
                if str(args.experiment_role) != "frozen_prompt_FP4"
                else query.new_zeros(())
            )
            cls = query.new_zeros(())
            acc = query.new_zeros(())
            if hard_bank is not None or text_bank is not None:
                bank = text_bank() if text_bank is not None else hard_bank.embeddings.to(device)
                bank_labels = text_bank.class_labels if text_bank is not None else hard_bank.labels
                cls, logits = _classify(query, bank, bank_labels.to(device), labels, float(args.tau_cls))
                acc = classification_accuracy(logits, bank_labels.to(device), labels)
            total = rank + float(args.lambda_cls) * cls
            total.backward()
            for name, parameter in prompt_model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    raise RuntimeError(f"intended trainable parameter has no gradient: {name}")
                if not parameter.requires_grad and parameter.grad is not None:
                    raise RuntimeError(f"frozen CLIP parameter received gradient: {name}")
            if text_bank is not None and text_bank.context.grad is None:
                raise RuntimeError("soft prompt did not receive gradient")
            optimizer.step()  # type: ignore[union-attr]
            scheduler.step()  # type: ignore[union-attr]
            if not bool(args.train_visual_layernorm) and str(args.experiment_role) != "frozen_prompt_FP5":
                for name, parameter in prompt_model.named_parameters():
                    if (
                        (name.startswith("visual.") or name.startswith("encoder.visual."))
                        and not torch.equal(parameter.detach().cpu(), initial_clip_state[name])
                    ):
                        raise RuntimeError(f"CLIP-owned parameter changed: {name}")
            step += 1
            last_rank, last_cls, last_acc = rank.item(), cls.item(), acc.item()
            training_history.append({"step": step, "rank_loss": last_rank, "classification_loss": last_cls, "seen_classification_accuracy": last_acc, "prompt_gradient_norm": prompt_grad_norm()})
            if step in probe_set:
                probe(step, last_rank, last_cls, last_acc)
    if text_bank is not None:
        torch.save(
            {"state_dict": {name: value.detach().cpu() for name, value in text_bank.state_dict().items()},
             "prompt_length": text_bank.prompt_length,
             "initialization": "first prompt_length token-embedding rows for hard prefix 'a photo of a'; normal noise tail if shorter",
             "class_names_used_for_training": list(text_bank.class_names),
             "parameter_count": text_bank.parameter_count},
            output_dir / "soft_prompt.pt",
        )
    if str(args.experiment_role) == "frozen_prompt_FP5":
        prompt_model.requires_grad_(False)
    if int(args.max_steps) < step:
        raise ValueError("max_steps must be at least the resumed global step")
    # FP0 and FP5 have no updates after their semantic origin. Repeating the
    # frozen state at the requested logical probe steps makes late stability
    # explicit without pretending those points were training updates.
    if str(args.experiment_role) in {"frozen_prompt_FP0", "frozen_prompt_FP5"}:
        existing = {int(row["step"]) for row in history}
        for logical_step in PROBE_STEPS:
            if logical_step not in existing:
                probe(logical_step, last_rank, last_cls, last_acc)
    report = {"schema_version": 1, "experiment_role": str(args.experiment_role), "campaign": CAMPAIGN,
              "seed": seed, "dataset": data.name, "pseudo_split": split_identity, "manifest_identity": manifest_identity,
              "provenance": provenance, "parameter_counts": {"prompt_parameters": sum(p.numel() for n, p in prompt_model.named_parameters() if "prompt" in n),
              "clip_parameters": sum(p.numel() for n, p in prompt_model.named_parameters() if n.startswith("visual.")),
              "trainable_parameters": sum(p.numel() for p in prompt_model.parameters() if p.requires_grad) + (0 if text_bank is None else text_bank.parameter_count),
              "soft_prompt_parameters": 0 if text_bank is None else text_bank.parameter_count},
              "trainable_parameter_names": [n for n, p in prompt_model.named_parameters() if p.requires_grad] + ([] if text_bank is None else [f"soft_prompt.{n}" for n, p in text_bank.named_parameters() if p.requires_grad]),
              "history": history, "training_history": training_history,
              "official_unseen_used_for_selection": False,
              "clip_owned_parameters_byte_identical": (
                  False if bool(args.train_visual_layernorm) else all(
                      torch.equal(parameter.detach().cpu(), initial_clip_state[name])
                      for name, parameter in prompt_model.named_parameters()
                      if name.startswith("visual.") or name.startswith("encoder.visual.")
                  )
              ),
              "approved_layernorm_parameter_names": [
                  name for name, parameter in prompt_model.named_parameters()
                  if (name.startswith("visual.") or name.startswith("encoder.visual."))
                  and "ln" in name.lower() and parameter.requires_grad
              ],
              "inference_contract": {"required_inputs": ["raw_sketch_image"], "text_required": False, "oracle_class_required": False},
              "checkpoint": history[-1]["checkpoint"] if history else None, "checkpoint_sha256": history[-1]["checkpoint_sha256"] if history else None}
    (output_dir / "run_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


@hydra.main(version_base="1.3", config_path=HYDRA_CONFIG_DIR, config_name="train_frozen_prompt")
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
