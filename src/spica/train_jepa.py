"""Train the predictive cross-modal SPICA JEPA family.

The predictor consumes only raw sketch images.  Frozen CLIP photo embeddings
provide stop-gradient targets, and an optional frozen seen-class CLIP text bank
is used only by a classification loss outside the model forward pass.
"""

import json
import math
from pathlib import Path
import random

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config.data import load_data_config
from .data.datasets import MultiPositiveRetrievalTrainDataset, RetrievalEvalDataset
from .data.manifest import read_class_map, read_manifest
from .data.splits import ClasswiseRetrievalSplit, make_classwise_retrieval_split
from .evaluation.embeddings import (
    EncodedRetrievalSet,
    load_encoded_retrieval_set,
    encode_retrieval_loader,
)
from .evaluation.jepa import (
    JepaFeatureSet,
    encode_jepa_loader,
    evaluate_jepa_features,
    feature_probe_dict,
)
from .evaluation.metrics import CategoryRetrievalEvaluation
from .evaluation.text_bank import (
    EncodedTextBank,
    SoftPromptTextBank,
    encode_class_text_bank,
)
from .models.clip import (
    FrozenClipEncoder,
    load_frozen_clip,
    load_trainable_sketch_encoder,
)
from .models.jepa import (
    JepaPrediction,
    SignatureRegularizer,
    SketchPhotoJepa,
    SpicaJepaPredictor,
    classification_accuracy,
    jepa_prediction_loss,
    jepa_ranking_loss,
    jepa_text_classification_loss,
    photo_semantic_target,
    vicreg_latent_regularization,
)
from .tracking.wandb import WandbExperiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")
OBJECTIVE_NAME = "cross_modal_jepa_prediction_plus_ranking"
PSEUDO_SPLIT_SEED = 3407


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_project_path(configured_path: object) -> Path:
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _resolve_checkpoint_path(configured_path: object) -> Path:
    if configured_path is None:
        return Path(HydraConfig.get().runtime.output_dir) / "jepa_final.pt"
    return _resolve_project_path(configured_path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    # PyTorch sets the worker seed before this hook; deriving Python's RNG from
    # it keeps the dataset's positive/negative sampling reproducible.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)


def _validate_options(args: DictConfig) -> None:
    encoder_mode = str(args.encoder_mode)
    if encoder_mode not in {"frozen", "partial", "full"}:
        raise ValueError("encoder_mode must be frozen, partial, or full")
    depth = int(args.encoder_unfreeze_depth)
    if encoder_mode == "partial" and depth <= 0:
        raise ValueError("partial encoder mode requires encoder_unfreeze_depth > 0")
    if encoder_mode != "partial" and depth != 0:
        raise ValueError("encoder_unfreeze_depth must be zero outside partial mode")
    if int(args.num_positive_photos) <= 0:
        raise ValueError("num_positive_photos must be positive")
    for name in (
        "learning_rate",
        "encoder_learning_rate",
        "weight_decay",
        "margin",
        "lambda_pred",
        "lambda_rank",
        "lambda_cls",
        "lambda_regularizer",
        "vicreg_variance_weight",
        "vicreg_covariance_weight",
        "vicreg_target_std",
        "tau_cls",
        "soft_prompt_learning_rate",
    ):
        value = float(args[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(args.learning_rate) <= 0 or float(args.encoder_learning_rate) <= 0:
        raise ValueError("learning_rate and encoder_learning_rate must be positive")
    if float(args.lambda_pred) == 0 and float(args.lambda_rank) == 0:
        raise ValueError("At least one of lambda_pred or lambda_rank must be positive")
    if float(args.tau_cls) <= 0:
        raise ValueError("tau_cls must be positive")
    if not isinstance(args.soft_prompt, bool):
        raise ValueError("soft_prompt must be a boolean")
    if int(args.soft_prompt_length) <= 0:
        raise ValueError("soft_prompt_length must be positive")
    if float(args.soft_prompt_learning_rate) <= 0:
        raise ValueError("soft_prompt_learning_rate must be positive")
    if bool(args.soft_prompt) and float(args.lambda_cls) <= 0:
        raise ValueError("soft_prompt requires lambda_cls > 0")
    if str(args.regularizer) not in {"none", "vicreg", "sigreg"}:
        raise ValueError("regularizer must be none, vicreg, or sigreg")
    if str(args.regularizer) == "none" and float(args.lambda_regularizer) != 0:
        raise ValueError("regularizer=none requires lambda_regularizer=0")
    if int(args.pseudo_val_num_classes) <= 0:
        raise ValueError("pseudo_val_num_classes must be positive")
    if int(args.pseudo_val_seed) < 0:
        raise ValueError("pseudo_val_seed must be non-negative")
    if str(args.train_class_scope) not in {"pseudo_train", "all_seen"}:
        raise ValueError("train_class_scope must be pseudo_train or all_seen")
    if int(args.max_steps) <= 0 or int(args.log_every) <= 0:
        raise ValueError("max_steps and log_every must be positive")
    if int(args.batch_size) <= 0 or int(args.num_workers) < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if int(args.eval_batch_size) <= 0 or int(args.prediction_batch_size) <= 0:
        raise ValueError("evaluation batch sizes must be positive")
    if int(args.query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    if not isinstance(args.run_probes, bool):
        raise ValueError("run_probes must be a boolean")
    if not str(args.prompt_template).count("{}") == 1:
        raise ValueError("prompt_template must contain exactly one '{}' placeholder")


def _build_split(
    data_config,
    *,
    num_validation_classes: int,
    seed: int,
) -> tuple[ClasswiseRetrievalSplit, dict[int, str]]:
    class_names = read_class_map(data_config.train.class_map)
    sketch_entries = read_manifest(
        data_config.train.sketch_manifest,
        data_config.root,
    )
    photo_entries = read_manifest(
        data_config.train.photo_manifest,
        data_config.root,
    )
    split = make_classwise_retrieval_split(
        sketch_entries,
        photo_entries,
        class_names,
        num_validation_classes=num_validation_classes,
        seed=seed,
    )
    return split, class_names


def _build_train_loader(
    sketch_entries,
    photo_entries,
    transform,
    *,
    num_positive_photos: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    dataset = MultiPositiveRetrievalTrainDataset(
        sketch_entries=sketch_entries,
        photo_entries=photo_entries,
        sketch_transform=transform,
        photo_transform=transform,
        num_positive_photos=num_positive_photos,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )


def _build_eval_loader(
    entries,
    transform,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = RetrievalEvalDataset(entries=entries, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _encode_photo_targets(
    encoder: FrozenClipEncoder,
    positive_images: Tensor,
    negative_images: Tensor,
) -> tuple[Tensor, Tensor]:
    if positive_images.ndim != 5 or negative_images.ndim != 4:
        raise ValueError("invalid photo image batch dimensions")
    if positive_images.shape[0] != negative_images.shape[0]:
        raise ValueError("positive and negative batch sizes must match")
    batch_size, num_positives = positive_images.shape[:2]
    if positive_images.shape[2:] != negative_images.shape[1:]:
        raise ValueError("positive and negative image shapes must match")
    flattened = positive_images.flatten(0, 1)
    with torch.no_grad():
        encoded = encoder(torch.cat((flattened, negative_images), dim=0))
    positives, negative = encoded.split((batch_size * num_positives, batch_size), dim=0)
    return positives.reshape(batch_size, num_positives, -1), negative


def _check_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains non-finite values")


def _parameter_counts(
    model: SketchPhotoJepa,
    photo_encoder: FrozenClipEncoder,
    regularizer: SignatureRegularizer | None,
    text_bank: EncodedTextBank | SoftPromptTextBank | None,
) -> dict[str, int]:
    total = model.total_parameter_count
    trainable = model.trainable_parameter_count
    soft_prompt_parameters = (
        text_bank.parameter_count if isinstance(text_bank, SoftPromptTextBank) else 0
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "predictor_parameters": model.predictor_parameter_count,
        "sketch_encoder_trainable_parameters": model.sketch_encoder_trainable_parameter_count,
        "regularizer_parameters": 0
        if regularizer is None
        else regularizer.parameter_count,
        "soft_prompt_parameters": soft_prompt_parameters,
        "optimizer_trainable_parameters": trainable + soft_prompt_parameters,
        "frozen_photo_encoder_parameters": sum(
            parameter.numel() for parameter in photo_encoder.parameters()
        ),
    }


def _save_text_bank(
    path: Path,
    text_bank: EncodedTextBank | SoftPromptTextBank,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text_bank, SoftPromptTextBank):
        with torch.no_grad():
            embeddings = text_bank().detach().cpu()
        payload = {
            "embeddings": embeddings,
            "labels": text_bank.class_labels.cpu(),
            "class_names": text_bank.class_names,
            "prompts": text_bank.prompts,
            "frozen": False,
            "soft_prompt": True,
            "soft_prompt_length": text_bank.prompt_length,
            "soft_prompt_state_dict": {
                key: value.detach().cpu()
                for key, value in text_bank.state_dict().items()
            },
            "used_only_as_training_classifier": True,
        }
    else:
        payload = {
            "embeddings": text_bank.embeddings,
            "labels": text_bank.labels,
            "class_names": text_bank.class_names,
            "prompts": text_bank.prompts,
            "frozen": True,
            "soft_prompt": False,
            "used_only_as_training_classifier": True,
        }
    torch.save(payload, path)


def _cpu_state_dict(model: SketchPhotoJepa) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _save_checkpoint(
    path: Path,
    *,
    model: SketchPhotoJepa,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_name: str,
    args: DictConfig,
    split: ClasswiseRetrievalSplit,
    parameter_counts: dict[str, int],
    include_optimizer: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_type": "cross_modal_jepa",
        "step": step,
        "model_config": {
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.predictor.hidden_dim,
            "encoder_mode": model.sketch_context_encoder.mode,
            "encoder_unfreeze_depth": model.sketch_context_encoder.unfreeze_depth,
            "encoder_model_name": str(args.model_name),
            "encoder_pretrained": None
            if args.pretrained is None
            else str(args.pretrained),
            "parameter_counts": parameter_counts,
        },
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if include_optimizer else None,
        "metadata": {
            "dataset": data_name,
            "split": "train",
            "model_name": str(args.model_name),
            "pretrained": None if args.pretrained is None else str(args.pretrained),
            "objective": OBJECTIVE_NAME,
            "model_family": "cross_modal_jepa",
            "frozen_photo_encoder": True,
            "photo_target_stop_gradient": True,
            "sketch_encoder_initialized_from_clip": True,
            "text_enters_predictor": False,
            "text_conditioning": False,
            "inference_inputs": ["raw_sketch_image"],
            "encoder_mode": str(args.encoder_mode),
            "encoder_unfreeze_depth": int(args.encoder_unfreeze_depth),
            "num_positive_photos": int(args.num_positive_photos),
            "photo_target": "normalized_mean",
            "lambda_pred": float(args.lambda_pred),
            "lambda_rank": float(args.lambda_rank),
            "lambda_cls": float(args.lambda_cls),
            "regularizer": str(args.regularizer),
            "lambda_regularizer": float(args.lambda_regularizer),
            "vicreg_variance_weight": float(args.vicreg_variance_weight),
            "vicreg_covariance_weight": float(args.vicreg_covariance_weight),
            "tau_cls": float(args.tau_cls),
            "soft_prompt": bool(args.soft_prompt),
            "soft_prompt_length": int(args.soft_prompt_length),
            "soft_prompt_learning_rate": float(args.soft_prompt_learning_rate),
            "margin": float(args.margin),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "encoder_learning_rate": float(args.encoder_learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "data_config": str(args.data_config),
            "train_class_scope": str(args.train_class_scope),
            "pseudo_validation_seed": split.seed,
            "pseudo_validation_class_ids": list(split.validation_class_ids),
            "pseudo_train_class_ids": list(split.train_class_ids),
            "checkpoint_contains_optimizer": include_optimizer,
        },
    }
    torch.save(payload, path)


def _metrics_summary(evaluation: CategoryRetrievalEvaluation) -> dict[str, object]:
    return {
        "mAP": evaluation.metrics.mean_average_precision,
        "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
        "precision_at_k": evaluation.metrics.precision_at_k,
        "num_queries": evaluation.metrics.num_queries,
        "num_gallery_items": evaluation.metrics.num_gallery_items,
    }


def _feature_set_metadata(features: JepaFeatureSet) -> dict[str, int]:
    return {
        "num_items": int(features.q.shape[0]),
        "embedding_dim": int(features.q.shape[1]),
    }


def _probe_checkpoint(
    *,
    model: SketchPhotoJepa,
    val_loader: DataLoader,
    val_gallery: EncodedRetrievalSet,
    test_loader: DataLoader,
    test_gallery: EncodedRetrievalSet,
    device: torch.device,
    args: DictConfig,
) -> tuple[dict[str, object], JepaFeatureSet, JepaFeatureSet]:
    val_features = encode_jepa_loader(model, val_loader, device=device)
    test_features = encode_jepa_loader(model, test_loader, device=device)
    precision = tuple(int(k) for k in args.precision_at_k)
    map_at_k = tuple(int(k) for k in args.map_at_k)
    val_evaluation = evaluate_jepa_features(
        val_features,
        val_gallery,
        precision_at_k=precision,
        map_at_k=map_at_k,
        map_at_k_denominator=str(args.map_at_k_denominator),
        query_chunk_size=int(args.query_chunk_size),
        device=device,
    )
    test_evaluation = evaluate_jepa_features(
        test_features,
        test_gallery,
        precision_at_k=precision,
        map_at_k=map_at_k,
        map_at_k_denominator=str(args.map_at_k_denominator),
        query_chunk_size=int(args.query_chunk_size),
        device=device,
    )
    return (
        {
            "val": _metrics_summary(val_evaluation),
            "diagnostic_test": _metrics_summary(test_evaluation),
            "val_features": _feature_set_metadata(val_features),
            "test_features": _feature_set_metadata(test_features),
            "val_geometry": feature_probe_dict(val_features, val_gallery),
            "diagnostic_test_geometry": feature_probe_dict(test_features, test_gallery),
            "protocol": {
                "val_is_pseudo_unseen": str(args.train_class_scope) == "pseudo_train",
                "validation_classes_seen_during_training": str(args.train_class_scope)
                == "all_seen",
                "official_test_is_diagnostic_only": True,
                "text_used_for_evaluation": False,
                "photo_gallery_reencoded": False,
                "map_at_k_denominator": str(args.map_at_k_denominator),
            },
        },
        val_features,
        test_features,
    )


def _log_training_metrics(
    experiment: WandbExperiment,
    *,
    step: int,
    values: dict[str, float],
    equivalent_epochs: float,
) -> None:
    experiment.log_metrics(
        {
            **values,
            "steps": step,
            "equivalent_epochs": equivalent_epochs,
        },
        step=step,
    )


def run(args: DictConfig) -> None:
    _validate_options(args)
    seed = int(args.seed)
    _seed_everything(seed)
    device = _resolve_device(str(args.device))
    data = load_data_config(_resolve_project_path(args.data_config))
    split, all_class_names = _build_split(
        data,
        num_validation_classes=int(args.pseudo_val_num_classes),
        seed=int(args.pseudo_val_seed),
    )

    # ``pseudo_train`` is the model-selection protocol.  ``all_seen`` is used
    # only for the post-selection retraining pass; its held-out metrics are
    # explicitly diagnostic because those classes are then seen during fitting.
    if str(args.train_class_scope) == "pseudo_train":
        train_sketch_entries = split.train_sketch_entries
        train_photo_entries = split.train_photo_entries
        text_class_ids = split.train_class_ids
    else:
        train_sketch_entries = tuple(
            read_manifest(data.train.sketch_manifest, data.root)
        )
        train_photo_entries = tuple(read_manifest(data.train.photo_manifest, data.root))
        text_class_ids = tuple(sorted(all_class_names))
    train_class_names = {
        class_id: all_class_names[class_id] for class_id in text_class_ids
    }

    pretrained = None if args.pretrained is None else str(args.pretrained)
    print(f"Loading frozen photo CLIP {args.model_name} ({pretrained}) on {device}...")
    photo_clip = load_frozen_clip(
        model_name=str(args.model_name),
        pretrained=pretrained,
        device=device,
    )
    sketch_clip = load_trainable_sketch_encoder(
        model_name=str(args.model_name),
        pretrained=pretrained,
        device=device,
        mode=str(args.encoder_mode),
        unfreeze_depth=int(args.encoder_unfreeze_depth),
    )
    predictor = SpicaJepaPredictor(
        embedding_dim=sketch_clip.encoder.embedding_dim,
        hidden_dim=int(args.hidden_dim),
    )
    model = SketchPhotoJepa(sketch_clip.encoder, predictor).to(device)
    model.train()

    train_loader = _build_train_loader(
        train_sketch_entries,
        train_photo_entries,
        sketch_clip.transform,
        num_positive_photos=int(args.num_positive_photos),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=bool(args.drop_last),
        seed=seed,
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader has no batches")

    val_loader = _build_eval_loader(
        split.validation_sketch_entries,
        sketch_clip.transform,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )
    val_photo_loader = _build_eval_loader(
        split.validation_photo_entries,
        photo_clip.transform,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )
    test_data_sketch = read_manifest(data.test.sketch_manifest, data.root)
    test_loader = _build_eval_loader(
        test_data_sketch,
        sketch_clip.transform,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )
    # The frozen gallery cache is the same stable CLIP photo space used by the
    # official diagnostic evaluator.  Validation photos are encoded once here.
    print("Encoding pseudo-unseen validation gallery with frozen photo CLIP...")
    val_gallery = encode_retrieval_loader(photo_clip.encoder, val_photo_loader)
    test_gallery = load_encoded_retrieval_set(
        _resolve_project_path(args.embedding_dir) / "photos.pt"
    )

    text_bank: EncodedTextBank | SoftPromptTextBank | None = None
    if float(args.lambda_cls) > 0:
        if bool(args.soft_prompt):
            text_bank = SoftPromptTextBank(
                photo_clip.encoder,
                photo_clip.tokenizer,
                train_class_names,
                prompt_length=int(args.soft_prompt_length),
            ).to(device)
        else:
            text_bank = encode_class_text_bank(
                photo_clip.encoder,
                photo_clip.tokenizer,
                train_class_names,
                prompt_template=str(args.prompt_template),
            )
        _save_text_bank(
            Path(HydraConfig.get().runtime.output_dir) / "seen_text_bank.pt",
            text_bank,
        )
        prompt_kind = (
            "soft-prompt" if isinstance(text_bank, SoftPromptTextBank) else "frozen"
        )
        print(
            f"Cached {len(text_bank.class_names)} seen-class {prompt_kind} text embeddings; "
            "text is training supervision only."
        )
    text_embeddings = (
        None
        if text_bank is None or isinstance(text_bank, SoftPromptTextBank)
        else text_bank.embeddings.to(device)
    )
    text_labels = (
        None
        if text_bank is None
        else (
            text_bank.class_labels
            if isinstance(text_bank, SoftPromptTextBank)
            else text_bank.labels
        ).to(device)
    )

    signature_regularizer: SignatureRegularizer | None = None
    if str(args.regularizer) == "sigreg":
        signature_regularizer = SignatureRegularizer(
            model.embedding_dim,
            num_projections=int(args.sigreg_num_projections),
            num_frequencies=int(args.sigreg_num_frequencies),
            frequency_max=float(args.sigreg_frequency_max),
            seed=int(args.sigreg_seed),
        ).to(device)

    predictor_parameters = [
        parameter
        for parameter in model.predictor.parameters()
        if parameter.requires_grad
    ]
    encoder_parameters = [
        parameter
        for parameter in model.sketch_context_encoder.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = [
        {
            "params": predictor_parameters,
            "lr": float(args.learning_rate),
        }
    ]
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": float(args.encoder_learning_rate),
            }
        )
    if isinstance(text_bank, SoftPromptTextBank):
        parameter_groups.append(
            {
                "params": list(text_bank.parameters()),
                "lr": float(args.soft_prompt_learning_rate),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(args.weight_decay),
    )
    parameter_counts = _parameter_counts(
        model,
        photo_clip.encoder,
        signature_regularizer,
        text_bank,
    )
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equivalent_epochs_per_step = int(args.batch_size) / len(train_loader.dataset)
    run_config = {
        "model_family": "cross_modal_jepa",
        "objective": OBJECTIVE_NAME,
        "encoder_mode": str(args.encoder_mode),
        "encoder_unfreeze_depth": int(args.encoder_unfreeze_depth),
        "M": int(args.num_positive_photos),
        "lambda_pred": float(args.lambda_pred),
        "lambda_rank": float(args.lambda_rank),
        "lambda_cls": float(args.lambda_cls),
        "regularizer": str(args.regularizer),
        "lambda_regularizer": float(args.lambda_regularizer),
        "tau_cls": float(args.tau_cls),
        "predictor_learning_rate": float(args.learning_rate),
        "encoder_learning_rate": float(args.encoder_learning_rate),
        "soft_prompt": bool(args.soft_prompt),
        "soft_prompt_length": int(args.soft_prompt_length),
        "soft_prompt_learning_rate": float(args.soft_prompt_learning_rate),
        "seed": seed,
        "steps": int(args.max_steps),
        "equivalent_epochs": int(args.max_steps) * equivalent_epochs_per_step,
        **parameter_counts,
        "train_class_scope": str(args.train_class_scope),
        "pseudo_validation_seed": split.seed,
        "pseudo_train_classes": len(split.train_class_ids),
        "pseudo_validation_classes": len(split.validation_class_ids),
        "text_conditioning": False,
        "wandb_project": str(args.wandb_project),
        "wandb_group": None if args.wandb_group is None else str(args.wandb_group),
        "wandb_mode": str(args.wandb_mode),
    }

    history: list[dict[str, object]] = []
    training_history: list[dict[str, float | int]] = []
    probe_steps = {int(step) for step in args.probe_steps}
    if bool(args.run_probes):
        probe_steps.add(int(args.max_steps))
    non_blocking = device.type == "cuda" and bool(args.pin_memory)
    experiment_mode = str(args.wandb_mode)
    with WandbExperiment(
        project=str(args.wandb_project),
        entity=None if args.wandb_entity is None else str(args.wandb_entity),
        group=None if args.wandb_group is None else str(args.wandb_group),
        name=None if args.wandb_run_name is None else str(args.wandb_run_name),
        config=run_config,
        tags=(
            "cross-modal-jepa",
            str(args.encoder_mode),
            f"M{args.num_positive_photos}",
        ),
        mode=experiment_mode,
        job_type="training",
        directory=output_dir,
    ) as experiment:
        print(
            f"Training cross-modal JEPA: mode={args.encoder_mode}, "
            f"M={args.num_positive_photos}, classes={len(text_class_ids)}, "
            f"steps={args.max_steps}, trainable={parameter_counts['trainable_parameters']}"
        )
        if experiment_mode != "disabled":
            print(f"W&B run: {experiment.run_url}")
        if text_bank is not None:
            print("Text enters predictor: NO (classification bank is loss-only).")

        step = 0
        window: dict[str, float] = {
            "loss_total": 0.0,
            "loss_pred": 0.0,
            "loss_rank": 0.0,
            "loss_cls": 0.0,
            "loss_var": 0.0,
            "loss_cov": 0.0,
            "loss_sigreg": 0.0,
            "train_pred_cosine": 0.0,
            "classification_accuracy": 0.0,
        }
        window_count = 0
        while step < int(args.max_steps):
            for batch in train_loader:
                sketch_images = batch["sketch"].to(device, non_blocking=non_blocking)
                positive_images = batch["positive_photos"].to(
                    device, non_blocking=non_blocking
                )
                negative_images = batch["negative_photo"].to(
                    device, non_blocking=non_blocking
                )
                labels = batch["label"].to(device=device, dtype=torch.long)
                positive_targets, negative_target = _encode_photo_targets(
                    photo_clip.encoder,
                    positive_images,
                    negative_images,
                )
                target = photo_semantic_target(positive_targets)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction: JepaPrediction = model(sketch_images)
                loss_pred = jepa_prediction_loss(prediction.q, target)
                loss_rank = jepa_ranking_loss(
                    prediction.q,
                    target,
                    negative_target,
                    margin=float(args.margin),
                )
                loss_cls = prediction.q.new_zeros(())
                cls_accuracy = prediction.q.new_zeros(())
                batch_text_embeddings = text_embeddings
                if isinstance(text_bank, SoftPromptTextBank):
                    batch_text_embeddings = text_bank()
                if batch_text_embeddings is not None and text_labels is not None:
                    loss_cls, logits = jepa_text_classification_loss(
                        prediction.q,
                        batch_text_embeddings,
                        text_labels,
                        labels,
                        temperature=float(args.tau_cls),
                        detach_text=not isinstance(text_bank, SoftPromptTextBank),
                    )
                    cls_accuracy = classification_accuracy(logits, text_labels, labels)

                loss_var = prediction.u.new_zeros(())
                loss_cov = prediction.u.new_zeros(())
                loss_sigreg = prediction.u.new_zeros(())
                regularizer_loss = prediction.u.new_zeros(())
                if str(args.regularizer) == "vicreg":
                    vicreg = vicreg_latent_regularization(
                        prediction.u,
                        variance_weight=float(args.vicreg_variance_weight),
                        covariance_weight=float(args.vicreg_covariance_weight),
                        target_std=float(args.vicreg_target_std),
                    )
                    loss_var = vicreg.variance
                    loss_cov = vicreg.covariance
                    regularizer_loss = vicreg.total
                elif signature_regularizer is not None:
                    loss_sigreg = signature_regularizer(prediction.u)
                    regularizer_loss = loss_sigreg

                loss_total = (
                    float(args.lambda_pred) * loss_pred
                    + float(args.lambda_rank) * loss_rank
                    + float(args.lambda_cls) * loss_cls
                    + float(args.lambda_regularizer) * regularizer_loss
                )
                for name, value in (
                    ("loss_total", loss_total),
                    ("loss_pred", loss_pred),
                    ("loss_rank", loss_rank),
                    ("loss_cls", loss_cls),
                    ("loss_var", loss_var),
                    ("loss_cov", loss_cov),
                    ("loss_sigreg", loss_sigreg),
                ):
                    _check_finite(name, value)
                _check_finite("prediction.h", prediction.h)
                _check_finite("prediction.u", prediction.u)
                _check_finite("prediction.q", prediction.q)
                loss_total.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        if parameter.grad is None:
                            raise RuntimeError(
                                f"Trainable parameter has no gradient: {name}"
                            )
                        _check_finite(f"gradient {name}", parameter.grad)
                    elif parameter.grad is not None:
                        raise RuntimeError(
                            f"Frozen parameter received a gradient: {name}"
                        )
                if isinstance(text_bank, SoftPromptTextBank):
                    for name, parameter in text_bank.named_parameters():
                        if parameter.grad is None:
                            raise RuntimeError(
                                f"Trainable soft-prompt parameter has no gradient: {name}"
                            )
                        _check_finite(f"gradient soft_prompt.{name}", parameter.grad)
                loss_total_value = loss_total.item()
                optimizer.step()
                for name, parameter in model.named_parameters():
                    _check_finite(f"parameter {name}", parameter)
                if isinstance(text_bank, SoftPromptTextBank):
                    for name, parameter in text_bank.named_parameters():
                        _check_finite(f"soft-prompt parameter {name}", parameter)

                with torch.no_grad():
                    pred_cosine = (prediction.q * target).sum(dim=-1).mean()
                step += 1
                current = {
                    "loss_total": loss_total_value,
                    "loss_pred": loss_pred.item(),
                    "loss_rank": loss_rank.item(),
                    "loss_cls": loss_cls.item(),
                    "loss_var": loss_var.item(),
                    "loss_cov": loss_cov.item(),
                    "loss_sigreg": loss_sigreg.item(),
                    "train_pred_cosine": pred_cosine.item(),
                    "classification_accuracy": cls_accuracy.item(),
                }
                for name, value in current.items():
                    window[name] += value
                window_count += 1

                if step % int(args.log_every) == 0 or step == int(args.max_steps):
                    means = {
                        name: value / window_count for name, value in window.items()
                    }
                    equivalent_epochs = step * equivalent_epochs_per_step
                    _log_training_metrics(
                        experiment,
                        step=step,
                        values=means,
                        equivalent_epochs=equivalent_epochs,
                    )
                    training_history.append(
                        {
                            "step": step,
                            "equivalent_epochs": equivalent_epochs,
                            **means,
                        }
                    )
                    print(
                        f"step={step:04d} total={means['loss_total']:.5f} "
                        f"pred={means['loss_pred']:.5f} rank={means['loss_rank']:.5f} "
                        f"cls={means['loss_cls']:.5f} cosine={means['train_pred_cosine']:.4f} "
                        f"epochs={equivalent_epochs:.4f}"
                    )
                    window = dict.fromkeys(window, 0.0)
                    window_count = 0

                if bool(args.run_probes) and step in probe_steps:
                    checkpoint_path = output_dir / "checkpoints" / f"jepa_step{step}.pt"
                    _save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        data_name=data.name,
                        args=args,
                        split=split,
                        parameter_counts=parameter_counts,
                    )
                    print(
                        f"Probing pseudo-unseen and diagnostic test at step {step}..."
                    )
                    probe, _, _ = _probe_checkpoint(
                        model=model,
                        val_loader=val_loader,
                        val_gallery=val_gallery,
                        test_loader=test_loader,
                        test_gallery=test_gallery,
                        device=device,
                        args=args,
                    )
                    probe["step"] = step
                    probe["equivalent_epochs"] = step * equivalent_epochs_per_step
                    probe["checkpoint"] = str(checkpoint_path)
                    probe["parameter_counts"] = parameter_counts
                    probe["training"] = {
                        **current,
                        "classification_accuracy": current["classification_accuracy"],
                    }
                    history.append(probe)
                    (output_dir / f"probe_step{step}.json").write_text(
                        json.dumps(probe, indent=2, sort_keys=True) + "\n"
                    )
                    val_metrics = probe["val"]
                    test_metrics = probe["diagnostic_test"]
                    assert isinstance(val_metrics, dict)
                    assert isinstance(test_metrics, dict)
                    val_map = float(val_metrics["mAP"])
                    val_p200 = float(val_metrics["precision_at_k"][200])
                    test_map = float(test_metrics["mAP"])
                    test_p200 = float(test_metrics["precision_at_k"][200])
                    geometry = probe["val_geometry"]
                    assert isinstance(geometry, dict)
                    semantic = geometry["semantic"]
                    assert isinstance(semantic, dict)
                    q_geometry = geometry["q"]
                    assert isinstance(q_geometry, dict)
                    photo_targets = geometry["photo_targets"]
                    assert isinstance(photo_targets, dict)
                    experiment.log_metrics(
                        {
                            "val_mAP": val_map,
                            "val_P200": val_p200,
                            "diagnostic_test_mAP": test_map,
                            "diagnostic_test_P200": test_p200,
                            "effective_rank": float(q_geometry["effective_rank"]),
                            "h_effective_rank": float(geometry["h"]["effective_rank"]),
                            "u_effective_rank": float(geometry["u"]["effective_rank"]),
                            "q_effective_rank": float(q_geometry["effective_rank"]),
                            "mean_feature_variance": float(
                                q_geometry["mean_feature_variance"]
                            ),
                            "minimum_feature_variance": float(
                                q_geometry["minimum_feature_variance"]
                            ),
                            "near_zero_variance_fraction": float(
                                q_geometry["near_zero_variance_fraction"]
                            ),
                            "covariance_offdiag": float(
                                q_geometry["covariance_offdiag"]
                            ),
                            "global_anisotropy": float(q_geometry["global_anisotropy"]),
                            "mean_pairwise_cosine": float(
                                q_geometry["mean_pairwise_cosine"]
                            ),
                            "semantic_margin": float(semantic["semantic_margin"]),
                            "predicted_target_cosine": float(
                                semantic["predicted_target_cosine"]
                            ),
                            "individual_positive_cosine": float(
                                photo_targets["individual_positive_cosine"]
                            ),
                            "positive_centroid_cosine": float(
                                photo_targets["positive_centroid_cosine"]
                            ),
                            "negative_gallery_cosine": float(
                                photo_targets["negative_gallery_cosine"]
                            ),
                            "positive_negative_margin": float(
                                photo_targets["positive_negative_margin"]
                            ),
                        },
                        step=step,
                    )
                    print(
                        f"  val_mAP={val_map:.6f} val_P200={val_p200:.6f} "
                        f"diagnostic_mAP={test_map:.6f} test_P200={test_p200:.6f} "
                        f"q_rank={float(q_geometry['effective_rank']):.2f} "
                        f"margin={float(semantic['semantic_margin']):.4f}"
                    )

                if step >= int(args.max_steps):
                    break

        final_path = _resolve_checkpoint_path(args.checkpoint_path)
        _save_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            step=step,
            data_name=data.name,
            args=args,
            split=split,
            parameter_counts=parameter_counts,
            include_optimizer=bool(args.save_optimizer),
        )
        (output_dir / "training_history.json").write_text(
            json.dumps(training_history, indent=2, sort_keys=True) + "\n"
        )
        if text_bank is not None:
            _save_text_bank(output_dir / "seen_text_bank.pt", text_bank)
        report = {
            "config": run_config,
            "checkpoint": str(final_path),
            "step": step,
            "equivalent_epochs": step * equivalent_epochs_per_step,
            "parameter_counts": parameter_counts,
            "pseudo_split": {
                "seed": split.seed,
                "train_class_ids": list(split.train_class_ids),
                "validation_class_ids": list(split.validation_class_ids),
                "train_sketches": len(split.train_sketch_entries),
                "train_photos": len(split.train_photo_entries),
                "validation_sketches": len(split.validation_sketch_entries),
                "validation_photos": len(split.validation_photo_entries),
            },
            "text": {
                "enabled": text_bank is not None,
                "enters_predictor": False,
                "seen_class_count": 0
                if text_bank is None
                else len(text_bank.class_names),
                "prompt_template": str(args.prompt_template),
                "soft_prompt": isinstance(text_bank, SoftPromptTextBank),
                "soft_prompt_length": (
                    0
                    if not isinstance(text_bank, SoftPromptTextBank)
                    else text_bank.prompt_length
                ),
                "text_gradients_enabled": isinstance(text_bank, SoftPromptTextBank),
            },
            "training_history": training_history,
            "probe_history": history,
            "inference_contract": {
                "required_inputs": ["raw_sketch_image"],
                "text_required": False,
                "photo_required": False,
                "positive_set_required": False,
                "oracle_class_required": False,
            },
            "wandb": {
                "mode": experiment_mode,
                "project": str(args.wandb_project),
                "group": None if args.wandb_group is None else str(args.wandb_group),
                "run_id": experiment.run_id,
                "run_url": experiment.run_url,
            },
        }
        (output_dir / "run_result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(f"Checkpoint saved to {final_path}")
        print("Text enters predictor: NO")
        print(f"JEPA run completed at step {step}.")


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="train_jepa",
)
def main(args: DictConfig) -> None:
    run(args)


if __name__ == "__main__":
    main()
