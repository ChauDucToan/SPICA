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
