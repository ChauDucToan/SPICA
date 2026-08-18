from argparse import ArgumentParser, Namespace
from pathlib import Path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train or evaluate Spica.")

    parser.add_argument(
        "--data-config",
        type=Path,
        required=True,
        help="Path to a dataset YAML configuration.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--model-name",
        default="ViT-B-32-quickgelu",
        help="OpenCLIP model architecture name.",
    )
    parser.add_argument(
        "--pretrained",
        default="openai",
        help="OpenCLIP pretrained checkpoint tag.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as 'cpu', 'cuda', 'cuda:0', or 'auto'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of samples in each batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=256,
        help="Number of queries ranked against the gallery at once.",
    )
    parser.add_argument(
        "--precision-at-k",
        type=int,
        nargs="+",
        default=(1, 5, 10, 100),
        help="K values used for Precision@K.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for cached sketch and photo embeddings.",
    )
    parser.add_argument(
        "--wandb-project",
        default="spica",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional Weights & Biases entity or team.",
    )
    parser.add_argument(
        "--wandb-run-name",
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
        help="Weights & Biases synchronization mode.",
    )
    parser.add_argument(
        "--wandb-table-queries",
        type=int,
        default=10,
        help="Number of worst-AP queries recorded in the retrieval table.",
    )
    parser.add_argument(
        "--wandb-table-results",
        type=int,
        default=5,
        help="Number of ranked photos recorded for each table query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> Namespace:
    return build_parser().parse_args(argv)
