from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.clip import FrozenClipImageEncoder


@dataclass(frozen=True, slots=True)
class EncodedRetrievalSet:
    embeddings: Tensor
    labels: Tensor
    paths: tuple[str, ...]

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
    encoder: FrozenClipImageEncoder,
    loader: DataLoader,
) -> EncodedRetrievalSet:
    """Encode every image from a retrieval loader and keep results on the CPU."""
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


def save_encoded_retrieval_set(
    encoded_set: EncodedRetrievalSet,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": encoded_set.embeddings,
            "labels": encoded_set.labels,
            "paths": encoded_set.paths,
        },
        output_path,
    )
