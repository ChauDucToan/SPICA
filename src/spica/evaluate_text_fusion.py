import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .config.data import load_data_config
from .data.manifest import read_class_map, read_manifest
from .evaluate_deterministic import _validate_cache_against_manifest
from .evaluation.embeddings import (
    EncodedRetrievalSet,
    load_encoded_retrieval_set,
)
from .evaluation.fusion import (
    build_soft_query_fusion,
    build_text_conditioned_queries,
    classify_sketches_with_text_bank,
    soft_post_sketches_with_text_bank,
)
from .evaluation.metrics import (
    CategoryRetrievalEvaluation,
    evaluate_category_retrieval,
)
from .evaluation.reporting import build_per_class_metric_rows
from .evaluation.text_bank import encode_class_text_bank
from .models.clip import load_frozen_clip
from .tracking.wandb import WandbExperiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = str(PROJECT_ROOT / "configs")


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


@hydra.main(
    version_base="1.3",
    config_path=HYDRA_CONFIG_DIR,
    config_name="evaluate_text_fusion",
)
def main(args: DictConfig) -> None:
    torch.manual_seed(int(args.seed))
    data_config_path = Path(str(args.data_config))
    embedding_dir = Path(str(args.embedding_dir))

    device = _resolve_device(str(args.device))
    config = load_data_config(data_config_path)
    split = str(args.split)
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    split_config = config.train if split == "train" else config.test

    class_names = read_class_map(split_config.class_map)
    sketches = load_encoded_retrieval_set(embedding_dir / "sketches.pt")
    photos = load_encoded_retrieval_set(embedding_dir / "photos.pt")
    _validate_cache_metadata(
        sketches,
        modality="sketch",
        dataset_name=config.name,
        split=split,
        model_name=args.model_name,
        pretrained=args.pretrained,
        allow_legacy_cache=args.allow_legacy_cache,
    )
    _validate_cache_metadata(
        photos,
        modality="photo",
        dataset_name=config.name,
        split=split,
        model_name=str(args.model_name),
        pretrained=(None if args.pretrained is None else str(args.pretrained)),
        allow_legacy_cache=bool(args.allow_legacy_cache),
    )
    split_identities = {
        "sketch_manifest_sha256": _validate_cache_against_manifest(
            modality="sketch",
            encoded_set=sketches,
            manifest_entries=read_manifest(
                split_config.sketch_manifest, config.root
            ),
        ),
        "photo_manifest_sha256": _validate_cache_against_manifest(
            modality="photo",
            encoded_set=photos,
            manifest_entries=read_manifest(
                split_config.photo_manifest, config.root
            ),
        ),
    }

    alphas = tuple(sorted({float(alpha) for alpha in args.fusion_alphas} | {0.0, 1.0}))
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError(f"All fusion alphas must be between 0 and 1: {alphas}")

    temperatures = tuple(
        sorted(float(temperature) for temperature in args.posterior_temperatures)
    )
    if any(
        not math.isfinite(temperature) or temperature <= 0
        for temperature in temperatures
    ):
        raise ValueError(
            "All posterior temperatures must be finite and greater "
            f"than 0: {temperatures}"
        )
    ks = tuple(sorted({int(k) for k in args.precision_at_k}))
    map_ks = tuple(sorted({int(k) for k in args.map_at_k}))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("precision_at_k must contain positive integers")
    if any(k <= 0 for k in map_ks):
        raise ValueError("map_at_k must contain positive integers")
    stored_top_k = max(max(ks), *map_ks)

    if split == "test":
        print(
            "Warning: alpha and temperature selection on the test split "
            "is exploratory only. "
            "Choose final hyperparameters on a held-out seen-class split."
        )

    run_config = {
        "dataset": config.name,
        "split": split,
        "embedding_dir": str(args.embedding_dir),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "device": str(device),
        "prompt_template": args.prompt_template,
        "fusion_alphas": list(alphas),
        "posterior_temperatures": list(temperatures),
        "precision_at_k": list(ks),
        "map_at_k": list(map_ks),
        "map_at_k_denominator": str(args.map_at_k_denominator),
        "query_chunk_size": args.query_chunk_size,
        "seed": args.seed,
        "exploratory_test_sweep": split == "test",
        "split_identities": split_identities,
    }

    with WandbExperiment(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=run_config,
        tags=(
            "text-fusion-baseline",
            "exploratory-sweep" if split == "test" else "model-selection",
            config.name,
            split,
        ),
        mode=args.wandb_mode,
        job_type="exploratory-evaluation" if split == "test" else "evaluation",
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
            map_at_k=map_ks,
            map_at_k_denominator=str(args.map_at_k_denominator),
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
                map_at_k=map_ks,
                map_at_k_denominator=str(args.map_at_k_denominator),
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
                map_at_k=map_ks,
                map_at_k_denominator=str(args.map_at_k_denominator),
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

        sketch_ap = sketch_evaluation.average_precision_per_query
        predicted_text_ap = predicted_text_only.average_precision_per_query

        if (
            sketch_ap.shape != predicted_text_ap.shape
            or sketch_ap.shape != sketches.labels.shape
        ):
            raise RuntimeError(
                "Per-query AP and query labels must have matching "
                f"shapes, got sketch AP {tuple(sketch_ap.shape)}, "
                f"predicted-text AP {tuple(predicted_text_ap.shape)}, "
                f"and labels {tuple(sketches.labels.shape)}"
            )

        correct_text_mask = classification.predicted_labels.eq(sketches.labels)
        incorrect_text_mask = ~correct_text_mask

        if not correct_text_mask.any():
            raise RuntimeError("No correctly classified sketch queries were found")
        if not incorrect_text_mask.any():
            raise RuntimeError("No incorrectly classified sketch queries were found")

        num_correct_text = int(correct_text_mask.sum().item())
        num_incorrect_text = int(incorrect_text_mask.sum().item())

        sketch_map_when_text_correct = (
            sketch_ap[correct_text_mask].double().mean().item()
        )
        predicted_map_when_text_correct = (
            predicted_text_ap[correct_text_mask].double().mean().item()
        )

        sketch_map_when_text_incorrect = (
            sketch_ap[incorrect_text_mask].double().mean().item()
        )
        predicted_map_when_text_incorrect = (
            predicted_text_ap[incorrect_text_mask].double().mean().item()
        )

        oracle_routed_ap = torch.maximum(
            sketch_ap,
            predicted_text_ap,
        )
        oracle_routed_map = oracle_routed_ap.double().mean().item()

        oracle_routing_gain = (
            oracle_routed_map - predicted_text_only.metrics.mean_average_precision
        )

        sketch_rescues_incorrect = (
            sketch_ap[incorrect_text_mask] > predicted_text_ap[incorrect_text_mask]
        )
        sketch_rescue_rate = sketch_rescues_incorrect.double().mean().item()

        mean_sketch_gain_when_text_incorrect = (
            (sketch_ap[incorrect_text_mask] - predicted_text_ap[incorrect_text_mask])
            .double()
            .mean()
            .item()
        )

        print("Branch complementarity diagnostic:")
        print(
            f"  Correct text queries: {num_correct_text}, "
            f"sketch mAP={sketch_map_when_text_correct:.6f}, "
            f"text mAP={predicted_map_when_text_correct:.6f}"
        )
        print(
            f"  Incorrect text queries: {num_incorrect_text}, "
            f"sketch mAP={sketch_map_when_text_incorrect:.6f}, "
            f"text mAP={predicted_map_when_text_incorrect:.6f}"
        )
        print(f"  Sketch rescue rate on incorrect queries: {sketch_rescue_rate:.6f}")
        print(
            f"  Mean sketch gain on incorrect queries: "
            f"{mean_sketch_gain_when_text_incorrect:.6f}"
        )
        print(
            f"  Oracle routing mAP: {oracle_routed_map:.6f}, "
            f"gain over predicted text: "
            f"{oracle_routing_gain:.6f}"
        )

        if classification.class_scores.ndim != 2:
            raise RuntimeError(
                "Class scores must have shape [N, C], got "
                f"{tuple(classification.class_scores.shape)}"
            )

        if classification.class_scores.shape[1] < 2:
            raise RuntimeError(
                "At least two text classes are required to compute "
                "the top-1/top-2 score margin"
            )

        top_two_scores = torch.topk(
            classification.class_scores,
            k=2,
            dim=-1,
        ).values
        score_margin = top_two_scores[:, 0] - top_two_scores[:, 1]

        if not torch.isfinite(score_margin).all():
            raise RuntimeError("Score margins contain non-finite values")

        sorted_margin_indices = torch.argsort(score_margin)
        margin_buckets = torch.tensor_split(
            sorted_margin_indices,
            4,
        )

        margin_bucket_names = (
            "lowest_25_percent",
            "lower_middle_25_percent",
            "upper_middle_25_percent",
            "highest_25_percent",
        )

        margin_bucket_rows: list[list[int | float | str]] = []
        total_bucket_queries = sum(bucket.numel() for bucket in margin_buckets)

        for bucket_name, bucket_indices in zip(
            margin_bucket_names,
            margin_buckets,
            strict=True,
        ):
            bucket_sketch_ap = sketch_ap[bucket_indices]
            bucket_text_ap = predicted_text_ap[bucket_indices]
            bucket_correct = correct_text_mask[bucket_indices]

            bucket_sketch_wins = bucket_sketch_ap > bucket_text_ap

            margin_bucket_rows.append(
                [
                    bucket_name,
                    int(bucket_indices.numel()),
                    score_margin[bucket_indices].double().mean().item(),
                    score_margin[bucket_indices].double().min().item(),
                    score_margin[bucket_indices].double().max().item(),
                    bucket_correct.double().mean().item(),
                    bucket_sketch_ap.double().mean().item(),
                    bucket_text_ap.double().mean().item(),
                    bucket_sketch_wins.double().mean().item(),
                    (bucket_sketch_ap - bucket_text_ap).double().mean().item(),
                ]
            )
        print("Score-margin bucket diagnostic:")
        for row in margin_bucket_rows:
            (
                bucket_name,
                bucket_size,
                mean_margin,
                min_margin,
                max_margin,
                text_accuracy,
                bucket_sketch_map,
                bucket_text_map,
                sketch_win_rate,
                mean_ap_difference,
            ) = row

            print(
                f"  {bucket_name}: "
                f"queries={bucket_size}, "
                f"margin=[{min_margin:.6f}, {max_margin:.6f}], "
                f"mean_margin={mean_margin:.6f}, "
                f"text_accuracy={text_accuracy:.6f}, "
                f"sketch_mAP={bucket_sketch_map:.6f}, "
                f"text_mAP={bucket_text_map:.6f}, "
                f"sketch_win_rate={sketch_win_rate:.6f}, "
                f"mean_sketch_minus_text_AP="
                f"{mean_ap_difference:.6f}"
            )

        experiment.log_table(
            "score_margin_buckets",
            columns=(
                "bucket",
                "num_queries",
                "mean_margin",
                "min_margin",
                "max_margin",
                "text_accuracy",
                "sketch_map",
                "predicted_text_map",
                "sketch_win_rate",
                "mean_sketch_minus_text_ap",
            ),
            rows=margin_bucket_rows,
        )

        if total_bucket_queries != score_margin.numel():
            raise RuntimeError(
                "Margin buckets do not cover every query: "
                f"{total_bucket_queries} bucketed versus "
                f"{score_margin.numel()} total"
            )

        best_soft_evaluation: CategoryRetrievalEvaluation | None = None
        best_soft_temperature: float | None = None
        best_soft_alpha: float | None = None
        best_soft_mean_entropy: float | None = None
        best_soft_mean_expected_text_norm: float | None = None

        for temperature in temperatures:
            soft_result = soft_post_sketches_with_text_bank(
                sketches,
                text_bank,
                temperature=temperature,
            )

            posterior_row_sum_error = (
                (soft_result.posterior.sum(dim=-1) - 1.0).abs().max().item()
            )

            if posterior_row_sum_error > 1e-5:
                raise RuntimeError(
                    "Soft-posterior rows do not sum to one; "
                    f"temperature={temperature}, "
                    f"max_error={posterior_row_sum_error}"
                )

            mean_entropy = soft_result.normalized_entropy.double().mean().item()
            mean_expected_text_norm = (
                soft_result.expected_text_embeddings.norm(dim=-1).double().mean().item()
            )

            print(
                f"Soft posterior: temperature={temperature:.6f}, "
                f"mean_entropy={mean_entropy:.6f}, "
                f"mean_expected_text_norm="
                f"{mean_expected_text_norm:.6f}, "
                f"max_row_sum_error="
                f"{posterior_row_sum_error:.3e}"
            )

            for alpha in alphas:
                if alpha == 0.0:
                    continue

                print(
                    "Evaluating soft posterior fusion: "
                    f"temperature={temperature:.6f}, "
                    f"alpha={alpha:.2f}..."
                )

                soft_queries = build_soft_query_fusion(
                    sketches,
                    soft_result.expected_text_embeddings,
                    alpha=alpha,
                )

                soft_evaluation = evaluate_category_retrieval(
                    soft_queries,
                    photos,
                    precision_at_k=ks,
                    map_at_k=map_ks,
                    map_at_k_denominator=str(args.map_at_k_denominator),
                    query_chunk_size=args.query_chunk_size,
                    top_k=stored_top_k,
                    device=device,
                )

                if (
                    best_soft_evaluation is None
                    or soft_evaluation.metrics.mean_average_precision
                    > best_soft_evaluation.metrics.mean_average_precision
                ):
                    best_soft_evaluation = soft_evaluation
                    best_soft_temperature = temperature
                    best_soft_alpha = alpha
                    best_soft_mean_entropy = mean_entropy
                    best_soft_mean_expected_text_norm = mean_expected_text_norm

                print(
                    "Soft posterior result: "
                    f"temperature={temperature:.6f}, "
                    f"alpha={alpha:.2f}, "
                    "mAP="
                    f"{soft_evaluation.metrics.mean_average_precision:.6f}"
                )

        if (
            best_soft_evaluation is None
            or best_soft_temperature is None
            or best_soft_alpha is None
            or best_soft_mean_entropy is None
            or best_soft_mean_expected_text_norm is None
        ):
            raise RuntimeError("No soft-posterior fusion configuration was evaluated")

        print(
            "Best soft posterior fusion: "
            f"temperature={best_soft_temperature:.6f}, "
            f"alpha={best_soft_alpha:.2f}, "
            "mAP="
            f"{best_soft_evaluation.metrics.mean_average_precision:.6f}, "
            f"mean_entropy={best_soft_mean_entropy:.6f}, "
            "mean_expected_text_norm="
            f"{best_soft_mean_expected_text_norm:.6f}"
        )

        experiment.log_metrics(
            {
                **sketch_evaluation.metrics.to_log_dict(
                    "sketch",
                    map_at_k_denominator=str(args.map_at_k_denominator),
                ),
                "text/classification_accuracy": classification.accuracy,
                **best_oracle.metrics.to_log_dict(
                    "fusion/oracle_best",
                    map_at_k_denominator=str(args.map_at_k_denominator),
                ),
                "fusion/oracle/best_alpha": best_oracle_alpha,
                **best_predicted.metrics.to_log_dict(
                    "fusion/predicted_best",
                    map_at_k_denominator=str(args.map_at_k_denominator),
                ),
                "fusion/predicted/best_alpha": best_predicted_alpha,
                **oracle_text_only.metrics.to_log_dict(
                    "text/oracle",
                    map_at_k_denominator=str(args.map_at_k_denominator),
                ),
                **predicted_text_only.metrics.to_log_dict(
                    "text/predicted",
                    map_at_k_denominator=str(args.map_at_k_denominator),
                ),
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
