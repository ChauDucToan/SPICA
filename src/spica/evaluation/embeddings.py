from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.clip import FrozenClipEncoder


@dataclass(frozen=True, slots=True)
class EncodedRetrievalSet:
    embeddings: Tensor
    labels: Tensor
    paths: tuple[str, ...]
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape [num_items, embedding_dim], "
                f"got {tuple(self.embeddings.shape)}"
            )
        if self.labels.ndim != 1:
            raise ValueError(
                f"labels must have shape [num_items], got {tuple(self.labels.shape)}"
            )

        num_items = self.embeddings.shape[0]
        if self.labels.shape[0] != num_items or len(self.paths) != num_items:
            raise ValueError(
                "embeddings, labels, and paths must contain the same number "
                f"of items, got {num_items}, {self.labels.shape[0]}, "
                f"and {len(self.paths)}"
            )


@torch.inference_mode()
def encode_retrieval_loader(
    encoder: FrozenClipEncoder,
    loader: DataLoader,
) -> EncodedRetrievalSet:
    encoder.eval()

    embedding_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    paths: list[str] = []

    non_blocking = encoder.device.type == "cuda" and loader.pin_memory

    for batch in loader:
        images = batch["image"].to(
            device=encoder.device,
            non_blocking=non_blocking,
        )
        embeddings = encoder(images)

        embedding_batches.append(embeddings.float().cpu())
        label_batches.append(batch["label"].to(dtype=torch.long, device="cpu"))
        paths.extend(str(path) for path in batch["path"])

    if not embedding_batches:
        raise ValueError("Cannot encode an empty retrieval loader")

    return EncodedRetrievalSet(
        embeddings=torch.cat(embedding_batches, dim=0),
        labels=torch.cat(label_batches, dim=0),
        paths=tuple(paths),
    )


def load_encoded_retrieval_set(input_path: Path) -> EncodedRetrievalSet:
    if not input_path.is_file():
        raise FileNotFoundError(f"Encoded retrieval set not found: {input_path}")

    payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid encoded retrieval payload in {input_path}")

    required_keys = {"embeddings", "labels", "paths"}
    missing_keys = required_keys - payload.keys()
    if missing_keys:
        raise ValueError(f"Missing keys in {input_path}: {sorted(missing_keys)}")

    embeddings = payload["embeddings"]
    labels = payload["labels"]
    paths = payload["paths"]
    metadata = payload.get("metadata", {})
    if not isinstance(embeddings, Tensor) or not isinstance(labels, Tensor):
        raise TypeError(f"Embeddings and labels in {input_path} must be tensors")
    if not isinstance(paths, (list, tuple)):
        raise TypeError(f"Paths in {input_path} must be a list or tuple")
    if not isinstance(metadata, dict):
        raise TypeError(f"Metadata in {input_path} must be a dictionary")

    return EncodedRetrievalSet(
        embeddings=embeddings.float(),
        labels=labels.long(),
        paths=tuple(str(path) for path in paths),
        metadata=dict(metadata),
    )


def save_encoded_retrieval_set(
    encoded_set: EncodedRetrievalSet,
    output_path: Path,
    *,
    metadata: dict[str, str | int | float | None] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": encoded_set.embeddings,
            "labels": encoded_set.labels,
            "paths": encoded_set.paths,
            "metadata": dict(
                metadata if metadata is not None else encoded_set.metadata
            ),
        },
        output_path,
    )
