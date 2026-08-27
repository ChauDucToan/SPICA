import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .embeddings import EncodedRetrievalSet
from .text_bank import EncodedTextBank


@dataclass(frozen=True, slots=True)
class TextClassificationResult:
    predicted_labels: Tensor
    accuracy: float


@dataclass(frozen=True, slots=True)
class SoftSemanticResult:
    class_scores: Tensor  # [N, C]
    posterior: Tensor  # [N, C]
    expected_text_embeddings: Tensor  # [N, D]
    normalized_entropy: Tensor  # [N]


@torch.inference_mode()
def soft_post_sketches_with_text_bank(
    sketches: EncodedRetrievalSet,
    text_bank: EncodedTextBank,
    *,
    temperature: float,
) -> SoftSemanticResult:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"Temperature must be finite and greater than 0, got {temperature}"
        )

    if sketches.embeddings.shape[1] != text_bank.embeddings.shape[1]:
        raise ValueError(
            "Sketch and text embedding dimensions must match, got "
            f"{sketches.embeddings.shape[1]} and {text_bank.embeddings.shape[1]}"
        )
    num_classes = text_bank.embeddings.shape[0]
    if num_classes < 2:
        raise ValueError(
            f"Soft posterior requires at least two text classes, got {num_classes}"
        )

    sketch_embeddings = F.normalize(sketches.embeddings, dim=-1)
    text_embeddings = F.normalize(text_bank.embeddings, dim=-1)
    class_scores = sketch_embeddings @ text_embeddings.T
    scaled_scores = class_scores / temperature

    log_posterior = F.log_softmax(scaled_scores, dim=-1)
    posterior = log_posterior.exp()
    expected_text_embeddings = posterior @ text_embeddings

    entropy = -(posterior * log_posterior).sum(dim=-1)
    normalized_entropy = entropy / math.log(num_classes)

    return SoftSemanticResult(
        class_scores=class_scores,
        posterior=posterior,
        expected_text_embeddings=expected_text_embeddings,
        normalized_entropy=normalized_entropy,
    )


@torch.inference_mode()
def classify_sketches_with_text_bank(
    sketches: EncodedRetrievalSet,
    text_bank: EncodedTextBank,
) -> TextClassificationResult:
    if sketches.embeddings.shape[1] != text_bank.embeddings.shape[1]:
        raise ValueError(
            "Sketch and text embedding dimensions must match, got "
            f"{sketches.embeddings.shape[1]} and {text_bank.embeddings.shape[1]}"
        )

    sketch_embeddings = F.normalize(sketches.embeddings, dim=-1)
    text_embeddings = F.normalize(text_bank.embeddings, dim=-1)
    class_scores = sketch_embeddings @ text_embeddings.T
    predicted_positions = class_scores.argmax(dim=1)
    predicted_labels = text_bank.labels[predicted_positions]
    accuracy = predicted_labels.eq(sketches.labels).double().mean().item()

    return TextClassificationResult(
        predicted_labels=predicted_labels,
        accuracy=accuracy,
    )


def build_soft_query_fusion(
    sketches: EncodedRetrievalSet,
    expected_text_embeddings: Tensor,
    *,
    alpha: float,
) -> EncodedRetrievalSet:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if expected_text_embeddings.shape != sketches.embeddings.shape:
        raise ValueError(
            "Each sketch must have one matching text embedding; expected "
            f"{tuple(sketches.embeddings.shape)}, got {tuple(expected_text_embeddings.shape)}"
        )

    sketch_embeddings = F.normalize(sketches.embeddings, dim=-1)
    fused_embeddings = F.normalize(
        (1.0 - alpha) * sketch_embeddings + alpha * expected_text_embeddings,
        dim=-1,
    )

    return EncodedRetrievalSet(
        embeddings=fused_embeddings,
        labels=sketches.labels,
        paths=sketches.paths,
        metadata=sketches.metadata,
    )


def build_text_conditioned_queries(
    sketches: EncodedRetrievalSet,
    text_embeddings: Tensor,
    *,
    alpha: float,
) -> EncodedRetrievalSet:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if text_embeddings.shape != sketches.embeddings.shape:
        raise ValueError(
            "Each sketch must have one matching text embedding; expected "
            f"{tuple(sketches.embeddings.shape)}, got {tuple(text_embeddings.shape)}"
        )

    sketch_embeddings = F.normalize(sketches.embeddings, dim=-1)
    normalized_text = F.normalize(text_embeddings, dim=-1)
    fused_embeddings = F.normalize(
        (1.0 - alpha) * sketch_embeddings + alpha * normalized_text,
        dim=-1,
    )

    return EncodedRetrievalSet(
        embeddings=fused_embeddings,
        labels=sketches.labels,
        paths=sketches.paths,
        metadata=sketches.metadata,
    )
