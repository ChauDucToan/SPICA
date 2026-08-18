from pathlib import Path

import torch

from .arg_parser import parse_args
from .config.data import load_data_config
from .data.loaders import build_retrieval_eval_loaders
from .evaluation.embeddings import (
    EncodedRetrievalSet,
    encode_retrieval_loader,
    save_encoded_retrieval_set,
)
from .evaluation.metrics import evaluate_category_retrieval
from .evaluation.reporting import (
    RETRIEVAL_TABLE_COLUMNS,
    build_retrieval_table_rows,
)
from .models.clip import load_frozen_clip_image_encoder
from .tracking.wandb import WandbExperiment


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _save_embeddings(
    output_dir: Path,
    *,
    sketches: EncodedRetrievalSet,
    photos: EncodedRetrievalSet,
) -> None:
    save_encoded_retrieval_set(sketches, output_dir / "sketches.pt")
    save_encoded_retrieval_set(photos, output_dir / "photos.pt")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    config = load_data_config(args.data_config)
    precision_at_k = tuple(args.precision_at_k)
    stored_top_k = max(max(precision_at_k), args.wandb_table_results)

    run_config = {
        "dataset": config.name,
        "data_config": str(args.data_config),
        "split": args.split,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "query_chunk_size": args.query_chunk_size,
        "precision_at_k": list(precision_at_k),
        "seed": args.seed,
    }

    with WandbExperiment(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=run_config,
        tags=("clip-baseline", config.name, args.split),
        mode=args.wandb_mode,
        job_type="evaluation",
    ) as experiment:
        print(f"Loading {args.model_name} ({args.pretrained}) on {device}...")
        clip_bundle = load_frozen_clip_image_encoder(
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=device,
        )

        loaders = build_retrieval_eval_loaders(
            config=config,
            transform=clip_bundle.transform,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        print("Encoding sketches...")
        sketches = encode_retrieval_loader(
            clip_bundle.encoder,
            loaders.sketch,
        )
        print("Encoding photos...")
        photos = encode_retrieval_loader(
            clip_bundle.encoder,
            loaders.photo,
        )

        print("Computing category-level retrieval metrics...")
        evaluation = evaluate_category_retrieval(
            sketches,
            photos,
            precision_at_k=precision_at_k,
            query_chunk_size=args.query_chunk_size,
            top_k=stored_top_k,
            device=device,
        )

        logged_metrics = evaluation.metrics.to_log_dict()
        experiment.log_metrics(logged_metrics)

        table_rows = build_retrieval_table_rows(
            evaluation,
            sketches,
            photos,
            loaders.class_names,
            num_queries=args.wandb_table_queries,
            results_per_query=args.wandb_table_results,
            selection="worst",
        )
        experiment.log_table(
            "retrieval/worst_queries",
            columns=RETRIEVAL_TABLE_COLUMNS,
            rows=table_rows,
        )

        if args.output_dir is not None:
            _save_embeddings(
                args.output_dir,
                sketches=sketches,
                photos=photos,
            )
            experiment.log_artifact(
                args.output_dir,
                name=f"{config.name}-{args.split}-clip-embeddings",
                artifact_type="embeddings",
                description="Frozen CLIP sketch and photo embeddings.",
                metadata={
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                },
            )

        print(f"mAP: {evaluation.metrics.mean_average_precision:.6f}")
        for k, value in evaluation.metrics.precision_at_k.items():
            print(f"P@{k}: {value:.6f}")
        if experiment.run_url is not None:
            print(f"W&B run: {experiment.run_url}")


if __name__ == "__main__":
    main()
