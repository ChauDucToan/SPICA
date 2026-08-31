from math import isfinite

import torch.nn.functional as F
from torch import Tensor, nn


def pairwise_ranking_loss(
    predicted_embeddings: Tensor,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    *,
    margin: float = 0.2,
) -> Tensor:
    if margin < 0 or not isfinite(margin):
        raise ValueError("Margin must be finite and non-negative")

    named_embeddings = (
        ("predicted_embeddings", predicted_embeddings),
        ("positive_embeddings", positive_embeddings),
        ("negative_embeddings", negative_embeddings),
    )

    for name, embeddings in named_embeddings:
        if embeddings.ndim != 2:
            raise ValueError(
                f"{name} must have shape [batch_size, embedding_dim], "
                f"got {tuple(embeddings.shape)}"
            )

        if not embeddings.is_floating_point():
            raise TypeError(f"{name} must be floating-point, got {embeddings.dtype}")

    expected_shape = predicted_embeddings.shape

    for name, embeddings in named_embeddings[1:]:
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"{name} must match predicted_embeddings shape "
                f"{tuple(expected_shape)}, got {tuple(embeddings.shape)}"
            )

    positive_scores = (predicted_embeddings * positive_embeddings).sum(dim=-1)
    negative_scores = (predicted_embeddings * negative_embeddings).sum(dim=-1)

    losses = F.softplus(margin - positive_scores + negative_scores)

    return losses.mean()


class DeterministicPhotoPredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lnorm = nn.LayerNorm(normalized_shape=embedding_dim)

        output_layer = nn.Linear(hidden_dim, embedding_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        self.stack = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.GELU(), output_layer
        )

    def forward(self, sketch_embeddings: Tensor) -> Tensor:
        if sketch_embeddings.ndim != 2:
            raise ValueError("sketch_embeddings must have ndim = 2")
        if sketch_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "sketch_embeddings.shape[1] must equal to embedding_dim, "
                f"got {sketch_embeddings.shape[1]}"
            )
        if not sketch_embeddings.is_floating_point():
            raise TypeError("sketch_embeddings must be floating-point")
        residual = self.stack(self.lnorm(sketch_embeddings))
        output = F.normalize(sketch_embeddings + residual, dim=-1)
        return output
