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
        "--image-size",
        type=int,
        default=224,
        help="Input image size after preprocessing.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> Namespace:
    return build_parser().parse_args(argv)
