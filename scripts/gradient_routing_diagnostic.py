#!/usr/bin/env python3
"""One-batch gradient attribution for the Stage-E mechanism audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from spica.config.data import load_data_config
from spica.data.loaders import build_multi_positive_retrieval_train_loader
from spica.evaluate_deterministic_k3 import _load_predictor as _load_k3_predictor
from spica.evaluate_deterministic import _resolve_device
from spica.evaluate_movmf import _load_predictor as _load_movmf_predictor
from spica.models.retrieval import (
    deterministic_angular_positive_assignment_loss,
    deterministic_k3_multi_positive_retrieval_loss,
)
from spica.models.vmf import mo_vmf_multi_positive_retrieval_loss
from spica.train_deterministic import (
    _resolve_data_config_path,
    _seed_everything,
)
from spica.training_utils import encode_multi_positive_images
from spica.models.clip import load_frozen_clip
from spica.train_movmf_ablation import _scheduled_prediction as _scheduled_movmf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute() or value.exists():
        return value
    return PROJECT_ROOT / value


def _group_gradient_norms(
    predictor: torch.nn.Module,
    gradients: tuple[torch.Tensor | None, ...],
    *,
    num_components: int,
) -> dict[str, float | None]:
    groups: dict[str, list[torch.Tensor]] = {
        "direction_branch": [],
        **{
            f"direction_component_{index}_output": []
            for index in range(num_components)
        },
        "gate_or_mixture_branch": [],
        "kappa_branch": [],
    }
    for (name, _), gradient in zip(predictor.named_parameters(), gradients):
        if gradient is None:
            continue
        value = gradient.detach().float()
        if name.startswith("direction_head"):
            groups["direction_branch"].append(value)
            if name in {"direction_head.3.weight", "direction_head.3.bias"}:
                chunks = value.chunk(num_components, dim=0)
                for index, chunk in enumerate(chunks):
                    groups[f"direction_component_{index}_output"].append(chunk)
        elif name.startswith(("gate_head", "mixture_head")):
            groups["gate_or_mixture_branch"].append(value)
        elif name.startswith("concentration_head"):
            groups["kappa_branch"].append(value)

    def norm(values: list[torch.Tensor]) -> float | None:
        if not values:
            return None
        return float(torch.sqrt(sum(value.square().sum() for value in values)).item())

    return {name: norm(values) for name, values in groups.items()}


def _gradient_record(
    predictor: torch.nn.Module,
    losses: dict[str, torch.Tensor],
    *,
    num_components: int,
) -> dict[str, Any]:
    parameters = tuple(predictor.parameters())
    result: dict[str, Any] = {}
    for loss_name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        result[loss_name] = {
            "loss": loss.item(),
            "gradient_norms": _group_gradient_norms(
                predictor, gradients, num_components=num_components
            ),
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movmf-checkpoint", required=True)
    parser.add_argument("--no-vmf-checkpoint", required=True)
    parser.add_argument("--angular-checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/sketchy_104_21.yaml")
    parser.add_argument("--model-name", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the checkpoint seed used to reproduce the training batch",
    )
    parser.add_argument(
        "--assignment-temperature",
        type=float,
        default=None,
        help="Override the angular checkpoint temperature",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    data = load_data_config(_resolve_data_config_path(args.data_config))

    movmf_path = _resolve(args.movmf_checkpoint)
    movmf_payload = torch.load(movmf_path, map_location="cpu", weights_only=True)
    no_vmf_path = _resolve(args.no_vmf_checkpoint)
    no_vmf_payload = torch.load(no_vmf_path, map_location="cpu", weights_only=True)
    angular_path = _resolve(args.angular_checkpoint)
    angular_payload = torch.load(angular_path, map_location="cpu", weights_only=True)
    payloads = (movmf_payload, no_vmf_payload, angular_payload)
    if not all(isinstance(payload, dict) for payload in payloads):
        raise TypeError("All gradient diagnostic checkpoints must be dictionaries")
    metadata = [payload.get("metadata", {}) for payload in payloads]
    if not all(isinstance(values, dict) for values in metadata):
        raise TypeError("All gradient diagnostic checkpoint metadata must be dictionaries")
    positive_counts = {
        int(values.get("positives_per_anchor_per_step", 3)) for values in metadata
    }
    if len(positive_counts) != 1:
        raise ValueError(
            "Gradient diagnostic checkpoints disagree on positives per anchor: "
            f"{sorted(positive_counts)}"
        )
    num_positive_photos = positive_counts.pop()
    seed = int(args.seed) if args.seed is not None else int(metadata[0].get("seed", 42))
    assignment_temperature = (
        float(args.assignment_temperature)
        if args.assignment_temperature is not None
        else float(metadata[2].get("assignment_temperature", 0.05))
    )
    clip = load_frozen_clip(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    loader = build_multi_positive_retrieval_train_loader(
        data,
        clip.transform,
        clip.transform,
        num_positive_photos=num_positive_photos,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    _seed_everything(seed)
    batch = next(iter(loader))
    sketch, positives, negative = encode_multi_positive_images(
        clip.encoder,
        batch["sketch"].to(device),
        batch["positive_photos"].to(device),
        batch["negative_photo"].to(device),
    )

    output: dict[str, Any] = {
        "diagnostic": "single real multi-positive training batch",
        "batch_size": args.batch_size,
        "num_positive_photos": num_positive_photos,
        "seed": seed,
        "assignment_temperature": assignment_temperature,
        "checkpoint_steps": [int(payload["step"]) for payload in payloads],
        "models": {},
    }

    movmf_metadata = metadata[0]
    movmf, _ = _load_movmf_predictor(
        movmf_path,
        expected_num_components=int(movmf_payload["model_config"]["num_components"]),
        device=device,
    )
    movmf_step = int(movmf_payload["step"])
    movmf_prediction, _, _ = _scheduled_movmf(
        movmf(sketch),
        step=movmf_step,
        warmup_steps=int(movmf_metadata.get("warmup_steps", 0)),
        warmup_concentration=float(
            movmf_metadata.get("initial_concentration", movmf.initial_concentration)
        ),
        gate_temperature_start=float(
            movmf_metadata.get("gate_temperature_start", 1.0)
        ),
        gate_temperature_anneal_steps=int(
            movmf_metadata.get("gate_temperature_anneal_steps", 1)
        ),
        warmup_dominant_weight=(
            None
            if movmf_metadata.get("target_dominant_weight") is None
            else float(movmf_metadata["target_dominant_weight"])
        ),
    )
    nll_weight = float(movmf_metadata.get("nll_weight", 1.0))
    ranking_weight = float(movmf_metadata.get("ranking_weight", 1.0))
    movmf_losses = mo_vmf_multi_positive_retrieval_loss(
        movmf_prediction,
        positives,
        negative,
        margin=float(movmf_metadata.get("margin", 0.2)),
        nll_weight=nll_weight,
        ranking_weight=ranking_weight,
        balance_weight=float(movmf_metadata.get("balance_weight", 0.0)),
        sharpness_weight=float(movmf_metadata.get("sharpness_weight", 0.0)),
        diversity_weight=float(movmf_metadata.get("diversity_weight", 0.0)),
        assignment_weight=float(movmf_metadata.get("assignment_weight", 0.0)),
        diversity_cosine_threshold=float(
            movmf_metadata.get("diversity_cosine_threshold", 0.9)
        ),
        ranking_score_transform=str(
            movmf_metadata.get("ranking_score_transform", "identity")
        ),
    )
    output["models"]["movmf"] = _gradient_record(
        movmf,
        {
            "positive_vmf_nll_unweighted": movmf_losses.positive_nll,
            "positive_vmf_nll_weighted": nll_weight * movmf_losses.positive_nll,
            "semantic_barycenter_ranking": ranking_weight
            * movmf_losses.density_ranking,
        },
        num_components=movmf.num_components,
    )

    for name, path, payload, angular in (
        ("no_vmf", no_vmf_path, no_vmf_payload, False),
        ("angular_routing", angular_path, angular_payload, True),
    ):
        predictor, _ = _load_k3_predictor(path, device)
        if predictor.num_components != 3:
            raise ValueError("Gradient diagnostic K3 checkpoints must have K=3")
        prediction = predictor(sketch)
        checkpoint_metadata = payload["metadata"]
        margin = float(checkpoint_metadata.get("margin", 0.2))
        loss_map: dict[str, torch.Tensor] = {}
        if angular:
            routing = deterministic_angular_positive_assignment_loss(
                prediction,
                positives,
                negative,
                margin=margin,
                assignment_temperature=assignment_temperature,
            )
            loss_map["angular_assignment"] = float(
                checkpoint_metadata.get("angular_assignment_weight", 1.0)
            ) * routing.total
        else:
            # Report the actual shared Stage-E ranking gradient.  The arm has
            # no separate positive-to-component routing loss to attribute.
            loss_map["semantic_barycenter_ranking"] = float(
                checkpoint_metadata.get("ranking_weight", 1.0)
            ) * deterministic_k3_multi_positive_retrieval_loss(
                prediction,
                positives,
                negative,
                margin=margin,
            )
        output["models"][name] = _gradient_record(
            predictor, loss_map, num_components=predictor.num_components
        )
        if not angular:
            output["models"][name]["explicit_routing"] = {
                "gradient_norms": None,
                "note": "No explicit positive-to-component routing objective in this arm.",
            }

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
