from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from .config.data import load_data_config
from .data.manifest import read_class_map
from .evaluation.embeddings import (
    EncodedRetrievalSet,
    load_encoded_retrieval_set,
)
from .evaluation.fusion import (
    build_text_conditioned_queries,
    classify_sketches_with_text_bank,
)
from .evaluation.metrics import (
    CategoryRetrievalEvaluation,
    evaluate_category_retrieval,
)
from .evaluation.reporting import build_per_class_metric_rows
from .evaluation.text_bank import encode_class_text_bank
from .models.clip import load_frozen_clip
from .tracking.wandb import WandbExperiment


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Evaluate frozen CLIP text and static fusion baselines."
    )
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--model-name", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prompt-template",
        default="a photo of a {}",
        help="Python format string with one positional class-name placeholder.",
    )
    parser.add_argument(
        "--fusion-alphas",
        type=float,
        nargs="+",
        default=tuple(index / 10 for index in range(11)),
        help="Text weights to sweep; endpoints 0 and 1 are always included.",
    )
    parser.add_argument(
        "--precision-at-k",
        type=int,
        nargs="+",
        default=(1, 5, 10, 100),
    )
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help="Allow cache files without model provenance metadata.",
    )
    parser.add_argument("--wandb-project", default="spica")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: list[str] | None = None) -> Namespace:
    return build_parser().parse_args(argv)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _validate_cache_metadata(
    encoded_set: EncodedRetrievalSet,
    *,
    modality: str,
    dataset_name: str,
    split: str,
    model_name: str,
    pretrained: str | None,
    allow_legacy_cache: bool,
) -> None:
    if not encoded_set.metadata:
        if allow_legacy_cache:
            print(
                f"Warning: accepting legacy {modality} cache without provenance "
                "metadata. The caller is responsible for model compatibility."
            )
            return
        raise ValueError(
            f"The {modality} cache has no provenance metadata. Recreate it with "
            "spica-eval-clip or pass --allow-legacy-cache after manually "
            "verifying the model and checkpoint."
        )

    expected = {
        "format_version": 1,
        "dataset": dataset_name,
        "split": split,
        "model_name": model_name,
        "pretrained": pretrained,
        "modality": modality,
    }
    mismatches = {
        key: (encoded_set.metadata.get(key), expected_value)
        for key, expected_value in expected.items()
        if encoded_set.metadata.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Incompatible {modality} cache metadata; "
            f"observed/expected values: {mismatches}"
        )


def _class_macro_map(
    evaluation: CategoryRetrievalEvaluation,
    queries: EncodedRetrievalSet,
) -> float:
    class_maps = [
        evaluation.average_precision_per_query[queries.labels.eq(class_label)]
        .double()
        .mean()
        for class_label in torch.unique(queries.labels)
    ]
    return torch.stack(class_maps).mean().item()


def _metric_row(
    mode: str,
    alpha: float,
    evaluation: CategoryRetrievalEvaluation,
    ks: tuple[int, ...],
) -> list[float | str]:
    return [
        mode,
        alpha,
        evaluation.metrics.mean_average_precision,
        *(evaluation.metrics.precision_at_k[k] for k in ks),
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    config = load_data_config(args.data_config)
    split_config = config.train if args.split == "train" else config.test

    class_names = read_class_map(split_config.class_map)
    sketches = load_encoded_retrieval_set(args.embedding_dir / "sketches.pt")
    photos = load_encoded_retrieval_set(args.embedding_dir / "photos.pt")
    _validate_cache_metadata(
        sketches,
        modality="sketch",
        dataset_name=config.name,
        split=args.split,
        model_name=args.model_name,
        pretrained=args.pretrained,
        allow_legacy_cache=args.allow_legacy_cache,
    )
    _validate_cache_metadata(
        photos,
        modality="photo",
        dataset_name=config.name,
        split=args.split,
        model_name=args.model_name,
        pretrained=args.pretrained,
        allow_legacy_cache=args.allow_legacy_cache,
    )

    alphas = tuple(sorted(set(args.fusion_alphas) | {0.0, 1.0}))
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError(f"All fusion alphas must be between 0 and 1: {alphas}")
    ks = tuple(sorted(set(args.precision_at_k)))
    stored_top_k = max(ks)

    if args.split == "test":
        print(
            "Warning: alpha selection on the test split is exploratory only. "
            "Choose final hyperparameters on a held-out seen-class split."
        )

    run_config = {
        "dataset": config.name,
        "split": args.split,
        "embedding_dir": str(args.embedding_dir),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "device": str(device),
        "prompt_template": args.prompt_template,
        "fusion_alphas": list(alphas),
        "precision_at_k": list(ks),
        "query_chunk_size": args.query_chunk_size,
        "seed": args.seed,
        "exploratory_test_sweep": args.split == "test",
    }

    with WandbExperiment(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=run_config,
        tags=(
            "text-fusion-baseline",
            "exploratory-sweep" if args.split == "test" else "model-selection",
            config.name,
            args.split,
        ),
        mode=args.wandb_mode,
        job_type="exploratory-evaluation" if args.split == "test" else "evaluation",
    ) as experiment:
        print(f"Loading {args.model_name} ({args.pretrained}) on {device}...")
        clip_bundle = load_frozen_clip(
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=device,
        )
        text_bank = encode_class_text_bank(
            clip_bundle.encoder,
            clip_bundle.tokenizer,
            class_names,
            prompt_template=args.prompt_template,
        )

        oracle_text = text_bank.embeddings_for_labels(sketches.labels)
        classification = classify_sketches_with_text_bank(sketches, text_bank)
        predicted_text = text_bank.embeddings_for_labels(
            classification.predicted_labels
        )

        print("Evaluating sketch-only baseline from cache...")
        sketch_evaluation = evaluate_category_retrieval(
            sketches,
            photos,
            precision_at_k=ks,
            query_chunk_size=args.query_chunk_size,
            top_k=stored_top_k,
            device=device,
        )

        table_columns = (
            "mode",
            "alpha",
            "mAP",
            *(f"P@{k}" for k in ks),
        )
        table_rows = [
            _metric_row("sketch", 0.0, sketch_evaluation, ks),
            _metric_row("oracle", 0.0, sketch_evaluation, ks),
            _metric_row("predicted", 0.0, sketch_evaluation, ks),
        ]
        best_oracle_alpha = 0.0
        best_oracle = sketch_evaluation
        best_predicted_alpha = 0.0
        best_predicted = sketch_evaluation
        oracle_text_only: CategoryRetrievalEvaluation | None = None
        predicted_text_only: CategoryRetrievalEvaluation | None = None

        for alpha in alphas:
            if alpha == 0.0:
                continue

            print(f"Evaluating oracle fusion at alpha={alpha:.2f}...")
            oracle_queries = build_text_conditioned_queries(
                sketches,
                oracle_text,
                alpha=alpha,
            )
            oracle_evaluation = evaluate_category_retrieval(
                oracle_queries,
                photos,
                precision_at_k=ks,
                query_chunk_size=args.query_chunk_size,
                top_k=stored_top_k,
                device=device,
            )
            table_rows.append(_metric_row("oracle", alpha, oracle_evaluation, ks))
            if (
                oracle_evaluation.metrics.mean_average_precision
                > best_oracle.metrics.mean_average_precision
            ):
                best_oracle_alpha = alpha
                best_oracle = oracle_evaluation
            if alpha == 1.0:
                oracle_text_only = oracle_evaluation

            print(f"Evaluating predicted fusion at alpha={alpha:.2f}...")
            predicted_queries = build_text_conditioned_queries(
                sketches,
                predicted_text,
                alpha=alpha,
            )
            predicted_evaluation = evaluate_category_retrieval(
                predicted_queries,
                photos,
                precision_at_k=ks,
                query_chunk_size=args.query_chunk_size,
                top_k=stored_top_k,
                device=device,
            )
            table_rows.append(_metric_row("predicted", alpha, predicted_evaluation, ks))
            if (
                predicted_evaluation.metrics.mean_average_precision
                > best_predicted.metrics.mean_average_precision
            ):
                best_predicted_alpha = alpha
                best_predicted = predicted_evaluation
            if alpha == 1.0:
                predicted_text_only = predicted_evaluation

        if oracle_text_only is None or predicted_text_only is None:
            raise RuntimeError("The forced alpha=1.0 text-only evaluation was not run")

        experiment.log_metrics(
            {
                **sketch_evaluation.metrics.to_log_dict("sketch"),
                "text/classification_accuracy": classification.accuracy,
                **best_oracle.metrics.to_log_dict("fusion/oracle_best"),
                "fusion/oracle/best_alpha": best_oracle_alpha,
                **best_predicted.metrics.to_log_dict("fusion/predicted_best"),
                "fusion/predicted/best_alpha": best_predicted_alpha,
                **oracle_text_only.metrics.to_log_dict("text/oracle"),
                **predicted_text_only.metrics.to_log_dict("text/predicted"),
                "sketch/class_macro_mAP": _class_macro_map(sketch_evaluation, sketches),
                "fusion/oracle_best/class_macro_mAP": _class_macro_map(
                    best_oracle, sketches
                ),
                "fusion/predicted_best/class_macro_mAP": _class_macro_map(
                    best_predicted, sketches
                ),
            }
        )
        experiment.log_table(
            "fusion/alpha_sweep",
            columns=table_columns,
            rows=table_rows,
        )
        experiment.log_table(
            "text/class_prompts",
            columns=("class_label", "class_name", "prompt"),
            rows=[
                [int(label), name, prompt]
                for label, name, prompt in zip(
                    text_bank.labels.tolist(),
                    text_bank.class_names,
                    text_bank.prompts,
                    strict=True,
                )
            ],
        )

        per_class_columns, per_class_rows = build_per_class_metric_rows(
            {
                "sketch": sketch_evaluation,
                f"oracle_best_alpha_{best_oracle_alpha:.2f}": best_oracle,
                "oracle_text_alpha_1.00": oracle_text_only,
                f"predicted_best_alpha_{best_predicted_alpha:.2f}": best_predicted,
                "predicted_text_alpha_1.00": predicted_text_only,
            },
            sketches,
            photos,
            class_names,
            precision_at_k=ks,
        )
        experiment.log_table(
            "fusion/per_class",
            columns=per_class_columns,
            rows=per_class_rows,
        )

        print(f"Text classification accuracy: {classification.accuracy:.6f}")
        print(
            f"Sketch-only mAP: {sketch_evaluation.metrics.mean_average_precision:.6f}"
        )
        print(
            f"Best oracle fusion: alpha={best_oracle_alpha:.2f}, "
            f"mAP={best_oracle.metrics.mean_average_precision:.6f}"
        )
        print(
            f"Best predicted fusion: alpha={best_predicted_alpha:.2f}, "
            f"mAP={best_predicted.metrics.mean_average_precision:.6f}"
        )
        print(
            "Oracle text-only mAP: "
            f"{oracle_text_only.metrics.mean_average_precision:.6f}"
        )
        print(
            "Predicted text-only mAP: "
            f"{predicted_text_only.metrics.mean_average_precision:.6f}"
        )
        if experiment.run_url is not None:
            print(f"W&B run: {experiment.run_url}")


if __name__ == "__main__":
    main()
