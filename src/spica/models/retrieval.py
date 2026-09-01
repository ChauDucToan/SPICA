from dataclasses import dataclass
from math import isfinite, log

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


@dataclass(frozen=True, slots=True)
class VmfPrediction:
    mean_direction: Tensor
    concentration: Tensor


class K1VmfPhotoPredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        *,
        min_concentration: float = 1e-4,
        max_concentration: float = 2048.0,
        initial_concentration: float = 512.0,
    ) -> None:
        super().__init__()
        if not isfinite(min_concentration) or min_concentration < 0:
            raise ValueError(
                "min_concentration must be finite and non-negative, "
                f"got {min_concentration}"
            )
        if not isfinite(max_concentration) or max_concentration <= min_concentration:
            raise ValueError(
                "max_concentration must be finite and greater than "
                f"min_concentration, got {max_concentration}"
            )
        if not isfinite(initial_concentration) or not (
            min_concentration < initial_concentration < max_concentration
        ):
            raise ValueError(
                "initial_concentration must be finite and strictly inside the "
                f"concentration bounds, got {initial_concentration}"
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.initial_concentration = initial_concentration
        self.direction_predictor = DeterministicPhotoPredictor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        concentration_output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(concentration_output.weight)
        initial_fraction = (initial_concentration - min_concentration) / (
            max_concentration - min_concentration
        )
        initial_raw = log(initial_fraction / (1.0 - initial_fraction))
        nn.init.constant_(concentration_output.bias, initial_raw)
        self.concentration_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            concentration_output,
        )

    def forward(self, sketch_embeddings: Tensor) -> VmfPrediction:
        mean_direction = self.direction_predictor(sketch_embeddings)
        raw_concentration = self.concentration_head(sketch_embeddings).squeeze(-1)
        concentration = (
            self.min_concentration
            + (self.max_concentration - self.min_concentration)
            * raw_concentration.sigmoid()
        )
        return VmfPrediction(
            mean_direction=mean_direction,
            concentration=concentration,
        )


@dataclass(frozen=True, slots=True)
class MoVmfPrediction:
    mean_directions: Tensor
    concentrations: Tensor
    mixture_logits: Tensor


class MoVmfPhotoPredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        *,
        num_components: int = 2,
        min_concentration: float = 1e-4,
        max_concentration: float = 2048.0,
        initial_concentration: float = 512.0,
        component_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_components < 2:
            raise ValueError(f"num_components must be at least 2, got {num_components}")
        if not isfinite(min_concentration) or min_concentration < 0:
            raise ValueError(
                "min_concentration must be finite and non-negative, "
                f"got {min_concentration}"
            )
        if not isfinite(max_concentration) or max_concentration <= min_concentration:
            raise ValueError(
                "max_concentration must be finite and greater than "
                f"min_concentration, got {max_concentration}"
            )
        if not isfinite(initial_concentration) or not (
            min_concentration < initial_concentration < max_concentration
        ):
            raise ValueError(
                "initial_concentration must be finite and strictly inside the "
                f"concentration bounds, got {initial_concentration}"
            )
        if not isfinite(component_init_std) or component_init_std <= 0:
            raise ValueError(
                "component_init_std must be finite and positive, "
                f"got {component_init_std}"
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.initial_concentration = initial_concentration
        self.component_init_std = component_init_std

        direction_output = nn.Linear(
            hidden_dim,
            num_components * embedding_dim,
        )
        nn.init.normal_(direction_output.weight, std=component_init_std)
        nn.init.zeros_(direction_output.bias)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            direction_output,
        )

        concentration_output = nn.Linear(hidden_dim, num_components)
        nn.init.zeros_(concentration_output.weight)
        initial_fraction = (initial_concentration - min_concentration) / (
            max_concentration - min_concentration
        )
        initial_raw = log(initial_fraction / (1.0 - initial_fraction))
        nn.init.constant_(concentration_output.bias, initial_raw)
        self.concentration_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            concentration_output,
        )

        mixture_output = nn.Linear(hidden_dim, num_components)
        nn.init.zeros_(mixture_output.weight)
        nn.init.zeros_(mixture_output.bias)
        self.mixture_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            mixture_output,
        )

    def forward(self, sketch_embeddings: Tensor) -> MoVmfPrediction:
        if sketch_embeddings.ndim != 2:
            raise ValueError("sketch_embeddings must have ndim = 2")
        if sketch_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "sketch_embeddings.shape[1] must equal to embedding_dim, "
                f"got {sketch_embeddings.shape[1]}"
            )
        if not sketch_embeddings.is_floating_point():
            raise TypeError("sketch_embeddings must be floating-point")

        batch_size = sketch_embeddings.shape[0]
        residuals = self.direction_head(sketch_embeddings).reshape(
            batch_size,
            self.num_components,
            self.embedding_dim,
        )
        mean_directions = F.normalize(
            sketch_embeddings[:, None, :] + residuals,
            dim=-1,
        )
        raw_concentrations = self.concentration_head(sketch_embeddings)
        concentrations = (
            self.min_concentration
            + (self.max_concentration - self.min_concentration)
            * raw_concentrations.sigmoid()
        )
        mixture_logits = self.mixture_head(sketch_embeddings)
        return MoVmfPrediction(
            mean_directions=mean_directions,
            concentrations=concentrations,
            mixture_logits=mixture_logits,
        )
