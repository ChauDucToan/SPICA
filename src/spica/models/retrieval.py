from dataclasses import dataclass
from math import isfinite, log

import torch
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


def _validate_multi_positive_embeddings(
    predicted_directions: Tensor,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
) -> None:
    if (
        predicted_directions.ndim != 3
        or positive_embeddings.ndim != 3
        or negative_embeddings.ndim != 2
    ):
        raise ValueError(
            "expected directions [B,K,D], positives [B,M,D], negative [B,D]"
        )
    if predicted_directions.shape[1] != 3:
        raise ValueError("deterministic K3 control requires exactly three directions")
    if positive_embeddings.shape[1] <= 0:
        raise ValueError("positive_embeddings must contain at least one positive")
    for name, embeddings in (
        ("predicted_directions", predicted_directions),
        ("positive_embeddings", positive_embeddings),
        ("negative_embeddings", negative_embeddings),
    ):
        if not embeddings.is_floating_point():
            raise TypeError(f"{name} must be floating-point, got {embeddings.dtype}")
        if not torch.isfinite(embeddings).all().item():
            raise ValueError(f"{name} must contain only finite values")
    batch_size, _, embedding_dim = predicted_directions.shape
    if (
        positive_embeddings.shape[0] != batch_size
        or positive_embeddings.shape[2] != embedding_dim
    ):
        raise ValueError("positive_embeddings dimensions must match directions")
    if negative_embeddings.shape != (batch_size, embedding_dim):
        raise ValueError(
            f"negative_embeddings must have shape {(batch_size, embedding_dim)}"
        )


def deterministic_gate_weighted_barycenter(
    directions: Tensor, gate_logits: Tensor
) -> Tensor:
    if directions.ndim != 3 or gate_logits.shape != directions.shape[:2]:
        raise ValueError("directions must be [B,K,D] and gate_logits must be [B,K]")
    if (
        not torch.isfinite(directions).all().item()
        or not torch.isfinite(gate_logits).all().item()
    ):
        raise ValueError("directions and gate_logits must be finite")
    weights = gate_logits.softmax(dim=-1)
    return F.normalize((weights[..., None] * directions).sum(dim=1), dim=-1)


@dataclass(frozen=True, slots=True)
class DeterministicK3Prediction:
    directions: Tensor
    gate_logits: Tensor

    @property
    def mean_directions(self) -> Tensor:
        return self.directions


def deterministic_k3_multi_positive_retrieval_loss(
    prediction: DeterministicK3Prediction,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    *,
    margin: float = 0.2,
    gate_prior_weight: float = 0.0,
    anchor_weight: float = 0.0,
    diversity_weight: float = 0.0,
    sketch_embeddings: Tensor | None = None,
) -> Tensor:
    """Gate-weighted barycenter ranking without concentrations or vMF terms."""
    if not isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    for name, value in (
        ("gate_prior_weight", gate_prior_weight),
        ("anchor_weight", anchor_weight),
        ("diversity_weight", diversity_weight),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    _validate_multi_positive_embeddings(
        prediction.directions, positive_embeddings, negative_embeddings
    )
    directions = F.normalize(prediction.directions, dim=-1)
    barycenter = deterministic_gate_weighted_barycenter(
        directions, prediction.gate_logits
    )
    positive_scores = torch.einsum("bd,bmd->bm", barycenter, positive_embeddings)
    negative_scores = (barycenter * negative_embeddings).sum(dim=-1)
    loss = F.softplus(margin - positive_scores + negative_scores[:, None]).mean()
    if gate_prior_weight:
        log_uniform = -log(directions.shape[1])
        log_probs = prediction.gate_logits.log_softmax(dim=-1)
        loss = (
            loss
            + gate_prior_weight
            * (log_probs.exp() * (log_probs - log_uniform)).sum(dim=-1).mean()
        )
    if anchor_weight:
        if sketch_embeddings is None or sketch_embeddings.shape != barycenter.shape:
            raise ValueError(
                "sketch_embeddings must have shape [B,D] when anchor_weight > 0"
            )
        loss = (
            loss
            + anchor_weight
            * (
                1 - (barycenter * F.normalize(sketch_embeddings, dim=-1)).sum(dim=-1)
            ).mean()
        )
    if diversity_weight and directions.shape[1] > 1:
        pairwise = torch.einsum("bkd,bjd->bkj", directions, directions)
        off_diagonal = ~torch.eye(
            directions.shape[1], dtype=torch.bool, device=directions.device
        )
        loss = loss + diversity_weight * pairwise[:, off_diagonal].pow(2).mean()
    return loss


# Descriptive aliases for callers that do not want the K-specific name.
deterministic_multi_positive_retrieval_loss = (
    deterministic_k3_multi_positive_retrieval_loss
)
deterministic_multi_positive_ranking_loss = (
    deterministic_k3_multi_positive_retrieval_loss
)


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


class DeterministicK3PhotoPredictor(nn.Module):
    """Three normalized deterministic directions and learned gate logits."""

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_components = 3
        direction_output = nn.Linear(hidden_dim, 3 * embedding_dim)
        nn.init.normal_(direction_output.weight, std=1e-4)
        nn.init.zeros_(direction_output.bias)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            direction_output,
        )
        gate_output = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(gate_output.weight)
        nn.init.zeros_(gate_output.bias)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            gate_output,
        )

    def forward(self, sketch_embeddings: Tensor) -> DeterministicK3Prediction:
        if (
            sketch_embeddings.ndim != 2
            or sketch_embeddings.shape[1] != self.embedding_dim
        ):
            raise ValueError("sketch_embeddings must have shape [B, embedding_dim]")
        if not sketch_embeddings.is_floating_point():
            raise TypeError("sketch_embeddings must be floating-point")
        batch_size = sketch_embeddings.shape[0]
        residuals = self.direction_head(sketch_embeddings).reshape(
            batch_size, 3, self.embedding_dim
        )
        return DeterministicK3Prediction(
            directions=F.normalize(sketch_embeddings[:, None, :] + residuals, dim=-1),
            gate_logits=self.gate_head(sketch_embeddings),
        )


# Short name for the new control's public API.
DeterministicK3Predictor = DeterministicK3PhotoPredictor


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
        initial_dominant_weight: float | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_components < 1:
            raise ValueError(f"num_components must be at least 1, got {num_components}")
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
        if initial_dominant_weight is not None:
            if num_components < 2:
                raise ValueError(
                    "initial_dominant_weight requires at least two components"
                )
            if not isfinite(initial_dominant_weight) or not (
                1.0 / num_components < initial_dominant_weight < 1.0
            ):
                raise ValueError(
                    "initial_dominant_weight must be finite and strictly between "
                    f"1 / num_components and 1, got {initial_dominant_weight}"
                )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.initial_concentration = initial_concentration
        self.component_init_std = component_init_std
        self.initial_dominant_weight = initial_dominant_weight

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
        if initial_dominant_weight is None:
            nn.init.zeros_(mixture_output.bias)
        else:
            satellite_weight = (1.0 - initial_dominant_weight) / (num_components - 1)
            initial_weights = mixture_output.bias.new_full(
                (num_components,),
                satellite_weight,
            )
            initial_weights[0] = initial_dominant_weight
            with torch.no_grad():
                mixture_output.bias.copy_(initial_weights.log())
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
