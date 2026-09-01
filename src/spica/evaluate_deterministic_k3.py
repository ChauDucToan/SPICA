"""Evaluation for the deterministic K=3/M=3 control using cached CLIP outputs."""

import json
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from torch import Tensor
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .evaluate_deterministic import (
    _metrics_dict,
    _resolve_device,
    _resolve_project_path,
    _validate_provenance,
    _validate_zero_shot_split,
)
from .evaluation.embeddings import EncodedRetrievalSet, load_encoded_retrieval_set
from .evaluation.metrics import (
    CategoryRetrievalEvaluation,
    CategoryRetrievalMetrics,
    _average_precision_from_relevance,
    evaluate_category_retrieval,
)
from .models.retrieval import DeterministicK3PhotoPredictor
from .train_deterministic import HYDRA_CONFIG_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_NAME = "deterministic_k3_multi_positive_gate_barycenter_ranking"
SUPPORTED_OBJECTIVES = {
    OBJECTIVE_NAME,
    "deterministic_k3_stageE_no_vmf",
    "deterministic_k3_stageE_angular_routing",
}


def _load_predictor(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Deterministic K3 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format_version",
        "model_type",
        "step",
        "model_config",
        "model_state_dict",
        "metadata",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        missing = required - payload.keys() if isinstance(payload, dict) else required
        raise ValueError(
            f"Invalid deterministic K3 checkpoint; missing {sorted(missing)}"
        )
    if payload["format_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint format version: {payload['format_version']}"
        )
    if payload["model_type"] != "deterministic_k3_photo_predictor":
        raise ValueError(f"Unsupported model type: {payload['model_type']!r}")
    config = payload["model_config"]
    metadata = payload["metadata"]
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise TypeError("K3 checkpoint model_config and metadata must be dictionaries")
    if config.get("num_components") != 3 or metadata.get("num_components") != 3:
        raise ValueError("Deterministic K3 checkpoint must contain three components")
    if metadata.get("objective") not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"Unsupported deterministic K3 objective: {metadata.get('objective')!r}"
        )
    if metadata.get("frozen_clip") is not True:
        raise ValueError("Deterministic K3 checkpoint must use a frozen CLIP encoder")
    if metadata.get("positives_per_anchor_per_step") != 3:
        raise ValueError("Deterministic K3 checkpoint must use M=3 positives")
    try:
        predictor = DeterministicK3PhotoPredictor(
            int(config["embedding_dim"]), int(config["hidden_dim"])
        )
        predictor.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError(
            "Invalid deterministic K3 model configuration or state"
        ) from error
    for name, parameter in predictor.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"Checkpoint contains a non-finite parameter: {name}")
    predictor.to(device).eval()
    return predictor, payload


def _predict(predictor, sketches, device, batch_size):
    if batch_size <= 0:
        raise ValueError(f"prediction_batch_size must be positive, got {batch_size}")
    if not torch.isfinite(sketches.embeddings).all().item():
        raise ValueError("Sketch cache contains non-finite embeddings")
    directions, gates = [], []
    with torch.inference_mode():
        for batch in sketches.embeddings.split(batch_size):
            output = predictor(batch.to(device))
            directions.append(output.directions.float().cpu())
            gates.append(output.gate_logits.float().cpu())
    directions = torch.cat(directions)
    gates = torch.cat(gates)
    if (
        not torch.isfinite(directions).all().item()
        or not torch.isfinite(gates).all().item()
    ):
        raise FloatingPointError("Predictor returned non-finite K3 outputs")
    return directions, gates


def _score_queries(directions, gate_logits, mode: str, temperature: float) -> Tensor:
    weights = gate_logits.softmax(-1)
    if mode == "learned_barycenter":
        return F.normalize((weights[..., None] * directions).sum(1), dim=-1)
    if mode == "uniform_barycenter":
        return F.normalize(directions.mean(1), dim=-1)
    if mode.startswith("component_"):
        try:
            component = int(mode.removeprefix("component_"))
        except ValueError as error:
            raise ValueError(
                "component scoring mode must be component_0, component_1, or component_2"
            ) from error
        if component not in range(directions.shape[1]):
            raise ValueError(
                f"component index must be in [0, {directions.shape[1] - 1}]"
            )
        return directions[:, component]
    # The remaining modes retain all K directional responses and are scored in
    # the ranking loop.
    return directions


def _evaluate_mode(directions, gates, sketches, photos, args, device):
    mode = str(args.scoring_mode)
    if mode in {"learned_barycenter", "uniform_barycenter"}:
        queries = _score_queries(directions, gates, mode, float(args.temperature))
        return evaluate_category_retrieval(
            EncodedRetrievalSet(
                queries, sketches.labels, sketches.paths, sketches.metadata
            ),
            photos,
            precision_at_k=tuple(int(k) for k in args.precision_at_k),
            map_at_k=tuple(int(k) for k in args.map_at_k),
            map_at_k_denominator=str(args.map_at_k_denominator),
            query_chunk_size=int(args.query_chunk_size),
            top_k=max(
                max(int(k) for k in args.precision_at_k),
                *(int(k) for k in args.map_at_k),
            ),
            device=device,
        )
    if mode.startswith("component_"):
        queries = _score_queries(directions, gates, mode, float(args.temperature))
        return evaluate_category_retrieval(
            EncodedRetrievalSet(
                queries, sketches.labels, sketches.paths, sketches.metadata
            ),
            photos,
            precision_at_k=tuple(int(k) for k in args.precision_at_k),
            map_at_k=tuple(int(k) for k in args.map_at_k),
            map_at_k_denominator=str(args.map_at_k_denominator),
            query_chunk_size=int(args.query_chunk_size),
            top_k=max(
                max(int(k) for k in args.precision_at_k),
                *(int(k) for k in args.map_at_k),
            ),
            device=device,
        )
    gallery = F.normalize(photos.embeddings.to(device), dim=-1)
    scores = []
    with torch.inference_mode():
        for start in range(0, directions.shape[0], int(args.query_chunk_size)):
            d = directions[start : start + int(args.query_chunk_size)].to(device)
            cosine = torch.einsum("bkd,gd->bkg", d, gallery)
            if mode == "max_directional":
                chunk = cosine.max(dim=1).values
            elif mode == "gate_weighted_directional":
                chunk = (
                    gates[start : start + d.shape[0]].to(device).softmax(-1)[..., None]
                    * cosine
                ).sum(1)
            elif mode == "logsumexp":
                temperature = float(args.temperature)
                if temperature <= 0:
                    raise ValueError("temperature must be positive for logsumexp")
                # Preserve the historical SPICA score convention.
                chunk = torch.logsumexp(temperature * cosine, dim=1) / temperature
            elif mode == "angular_logsumexp":
                temperature = float(args.temperature)
                if temperature <= 0:
                    raise ValueError(
                        "temperature must be positive for angular_logsumexp"
                    )
                # tau is the angular temperature: tau*logsumexp(cos/tau).
                chunk = temperature * torch.logsumexp(cosine / temperature, dim=1)
            else:
                raise ValueError("unsupported scoring_mode")
            scores.append(chunk.cpu())
    score_matrix = torch.cat(scores)
    # Convert cached scores to a retrieval-set-like evaluator without encoding
    # any gallery image; each row is ranked by the precomputed score matrix.
    return _evaluate_score_matrix(score_matrix, sketches.labels, photos, args, device)


def _evaluate_score_matrix(scores, query_labels, photos, args, device):
    if scores.ndim != 2 or scores.shape[0] != query_labels.shape[0]:
        raise ValueError("scores must have shape [num_queries, num_gallery]")
    if scores.shape[1] != photos.embeddings.shape[0] or scores.shape[1] == 0:
        raise ValueError("scores must contain one column per non-empty gallery item")
    if not torch.isfinite(scores).all().item():
        raise ValueError("Score matrix contains non-finite values")
    precision_ks = tuple(sorted({int(k) for k in args.precision_at_k}))
    map_ks = tuple(sorted({int(k) for k in args.map_at_k}))
    gallery_size = scores.shape[1]
    if not precision_ks or any(k <= 0 or k > gallery_size for k in precision_ks):
        raise ValueError(f"precision_at_k values must be in [1, {gallery_size}]")
    if any(k <= 0 or k > gallery_size for k in map_ks):
        raise ValueError(f"map_at_k values must be in [1, {gallery_size}]")
    ranks = torch.argsort(scores.to(device), dim=1, descending=True, stable=True)
    relevant = photos.labels.to(device)[ranks].eq(query_labels.to(device)[:, None])
    num_positives = relevant.sum(dim=1)
    if torch.any(num_positives == 0).item():
        raise ValueError("Every query must have at least one positive gallery item")
    positions = torch.arange(1, gallery_size + 1, device=device, dtype=torch.float32)
    full, truncated = _average_precision_from_relevance(
        relevant,
        positions,
        map_ks,
        map_at_k_denominator=str(args.map_at_k_denominator),
    )
    precision = {
        k: relevant[:, :k].float().mean(dim=1).double().mean().item()
        for k in precision_ks
    }
    top_k = max(*precision_ks, *map_ks)
    top = ranks[:, :top_k]
    return CategoryRetrievalEvaluation(
        metrics=CategoryRetrievalMetrics(
            full.double().mean().item(),
            precision,
            len(query_labels),
            len(photos.labels),
            {k: value.double().mean().item() for k, value in truncated.items()},
        ),
        average_precision_per_query=full.cpu(),
        top_indices=top.cpu(),
        top_scores=scores.to(device).gather(1, top).float().cpu(),
    )


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="evaluate_deterministic_k3",
)
def main(args: DictConfig) -> None:
    device = _resolve_device(str(args.device))
    checkpoint_path = _resolve_project_path(str(args.checkpoint_path))
    embedding_dir = _resolve_project_path(str(args.embedding_dir))
    data_path = _resolve_project_path(str(args.data_config))
    predictor, checkpoint = _load_predictor(checkpoint_path, device)
    sketches = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
    photos = load_encoded_retrieval_set(embedding_dir / "photos.pt")
    split_identities = _validate_zero_shot_split(
        data_path, dataset_name=str(args.dataset_name), sketches=sketches, photos=photos
    )
    _validate_provenance(
        checkpoint=checkpoint,
        sketches=sketches,
        photos=photos,
        dataset_name=str(args.dataset_name),
        model_name=str(args.model_name),
        pretrained=None if args.pretrained is None else str(args.pretrained),
    )
    directions, gates = _predict(
        predictor, sketches, device, int(args.prediction_batch_size)
    )
    evaluation = _evaluate_mode(directions, gates, sketches, photos, args, device)
    result = {
        "protocol": "standard_zs_sbir_sketch_only_inference",
        "diagnostic_test_evaluation": True,
        "uses_test_labels_for_scoring": False,
        "leakage_flags": {
            "gallery_reencoded": False,
            "test_labels_used_for_scoring": False,
        },
        "model": "deterministic_k3",
        "objective": checkpoint.get("metadata", {}).get("objective", OBJECTIVE_NAME),
        "scoring_mode": str(args.scoring_mode),
        "temperature": float(args.temperature),
        "map_at_k_denominator": str(args.map_at_k_denominator),
        "checkpoint": str(checkpoint_path),
        "trainable_parameters": sum(
            parameter.numel() for parameter in predictor.parameters()
        ),
        "embedding_cache_metadata": {
            "sketches": sketches.metadata,
            "photos": photos.metadata,
        },
        "checkpoint_step": int(checkpoint["step"]),
        "split_identities": split_identities,
        "params": {
            "K": 3,
            "M": 3,
            "prediction_batch_size": int(args.prediction_batch_size),
            "query_chunk_size": int(args.query_chunk_size),
        },
        "steps": int(checkpoint["step"]),
        "training_metadata": checkpoint.get("metadata", {}),
        "metrics": {
            **_metrics_dict(evaluation.metrics),
            "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
        },
    }
    output = Path(HydraConfig.get().runtime.output_dir) / "metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"mAP: {evaluation.metrics.mean_average_precision:.6f}\nMetrics saved to {output}"
    )


if __name__ == "__main__":
    main()
