"""P3: fit W_CLIP compatibility controls on pseudo-train, evaluate pseudo-unseen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spica.config.data import load_data_config
from spica.data.datasets import RetrievalEvalDataset
from spica.evaluation.embeddings import EncodedRetrievalSet, encode_retrieval_loader
from spica.evaluation.metrics import evaluate_category_retrieval
from spica.evaluation.transport import hidden_space_compatibility
from spica.evaluate_transport import _load_model
from spica.models.clip import load_frozen_clip, visual_pre_projection
from spica.provenance import capture_provenance
from spica.train_transport import PROJECT_ROOT, _build_split


def _loader(entries, transform, batch_size: int) -> DataLoader:
    return DataLoader(
        RetrievalEvalDataset(entries, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


@torch.inference_mode()
def _hidden(loader: DataLoader, model, reference_visual, device: torch.device):
    adapted, reference, labels, paths = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        adapted.append(model.sketch_context_encoder(images).float().cpu())
        reference.append(visual_pre_projection(reference_visual, images).float().cpu())
        labels.append(batch["label"].long())
        paths.extend(str(path) for path in batch["path"])
    return torch.cat(adapted), torch.cat(reference), torch.cat(labels), tuple(paths)


def _effective_rank(values: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(
        values.float() - values.float().mean(0, keepdim=True)
    )
    energy = singular.square()
    return float(energy.sum().square().div(energy.square().sum().clamp_min(1e-12)))


def _evaluate(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    paths: tuple[str, ...],
    gallery: EncodedRetrievalSet,
    device: torch.device,
) -> dict[str, object]:
    evaluation = evaluate_category_retrieval(
        EncodedRetrievalSet(
            embeddings=F.normalize(embeddings, dim=-1), labels=labels, paths=paths
        ),
        gallery,
        precision_at_k=(1, 5, 10, 100, 200),
        map_at_k=(200,),
        map_at_k_denominator="prefix_positive",
        top_k=200,
        device=device,
    )
    return {
        "mAP": evaluation.metrics.mean_average_precision,
        "mAP_at_k": evaluation.metrics.mean_average_precision_at_k,
        "precision_at_k": evaluation.metrics.precision_at_k,
        "per_query_AP": evaluation.average_precision_per_query.tolist(),
        "effective_rank": _effective_rank(embeddings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--data-config", type=Path, default=Path("configs/data/sketchy_104_21.yaml")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    data_path = (
        args.data_config
        if args.data_config.is_absolute()
        else PROJECT_ROOT / args.data_config
    )
    data = load_data_config(data_path)
    split, _ = _build_split(data, num_validation_classes=20, seed=3407)
    photo_clip = load_frozen_clip(
        model_name="ViT-B-32-quickgelu", pretrained="openai", device=device
    )
    gallery = encode_retrieval_loader(
        photo_clip.encoder,
        _loader(split.validation_photo_entries, photo_clip.transform, args.batch_size),
    )
    values = []
    for configured_checkpoint in args.checkpoints:
        checkpoint = (
            configured_checkpoint
            if configured_checkpoint.is_absolute()
            else PROJECT_ROOT / configured_checkpoint
        )
        model, payload, transform = _load_model(checkpoint, device=device)
        train_h, train_ref, _, _ = _hidden(
            _loader(split.train_sketch_entries, transform, args.batch_size),
            model,
            photo_clip.encoder.model.visual,
            device,
        )
        val_h, val_ref, val_labels, val_paths = _hidden(
            _loader(split.validation_sketch_entries, transform, args.batch_size),
            model,
            photo_clip.encoder.model.visual,
            device,
        )
        projection = model.photo_projection.cpu()
        train_h = train_h.float()
        train_ref = train_ref.float()
        cross = train_h.T @ train_ref
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        rotation = u @ vh
        x = torch.cat((train_h, torch.ones(train_h.shape[0], 1)), dim=1)
        target = F.normalize(projection(train_ref), dim=-1)
        gram = x.T @ x
        regularizer = torch.eye(gram.shape[0]) * args.ridge
        regularizer[-1, -1] = 0
        ridge = torch.linalg.solve(gram + regularizer, x.T @ target)
        val_x = torch.cat((val_h, torch.ones(val_h.shape[0], 1)), dim=1)
        reference_queries = F.normalize(projection(val_ref), dim=-1)
        methods = {
            "frozen_original_W_CLIP": projection(val_h),
            "orthogonal_adapter": projection(val_h @ rotation),
            "ridge_projection": val_x @ ridge,
        }
        method_results = {}
        for name, embeddings in methods.items():
            result = _evaluate(embeddings, val_labels, val_paths, gallery, device)
            result["frozen_W_compatibility_cosine"] = float(
                F.cosine_similarity(
                    F.normalize(embeddings, dim=-1), reference_queries, dim=-1
                ).mean()
            )
            method_results[name] = result
        values.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_step": int(payload["step"]),
                "fit_split": "pseudo_train only",
                "evaluation_split": "pseudo_unseen only",
                "data_split_identity": payload.get("data_split_identity"),
                "hidden_geometry": hidden_space_compatibility(
                    val_ref, val_h, projection
                ),
                "methods": method_results,
                "checkpoint_provenance": payload.get("metadata", {}).get("provenance"),
            }
        )
    result = {
        "schema_version": 1,
        "probe": "projection_refit_control",
        "seed": 3407,
        "official_unseen_used": False,
        "ridge": args.ridge,
        "values": values,
        "provenance": capture_provenance(
            PROJECT_ROOT, command=[str(value) for value in __import__("sys").argv]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
