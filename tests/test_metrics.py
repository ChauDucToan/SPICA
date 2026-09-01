import pytest
import torch

from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.metrics import evaluate_category_retrieval


def test_truncated_map_preserves_full_map_and_uses_all_positive_denominator():
    queries = EncodedRetrievalSet(torch.tensor([[1.0, 0.0]]), torch.tensor([1]), ("q",))
    # Stable descending ranking is labels [1, 2, 1, 2].
    gallery = EncodedRetrievalSet(
        torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.4, 0.916515]]),
        torch.tensor([1, 2, 1, 2]),
        ("a", "b", "c", "d"),
    )
    evaluation = evaluate_category_retrieval(
        queries, gallery, precision_at_k=(1, 2), map_at_k=(1, 2, 4), top_k=2
    )
    assert evaluation.metrics.mean_average_precision == pytest.approx(5 / 6)
    # The official SBIR apsak convention truncates both scores and relevance
    # before AP, so the denominator is positives in the returned prefix.
    assert evaluation.metrics.mean_average_precision_at_k[1] == pytest.approx(1.0)
    assert evaluation.metrics.mean_average_precision_at_k[2] == pytest.approx(1.0)
    assert evaluation.metrics.mean_average_precision_at_k[4] == pytest.approx(5 / 6)
    assert evaluation.top_indices.shape == (1, 2)
    assert "retrieval/mAP" in evaluation.metrics.to_log_dict()
    assert "retrieval/mAP@2" in evaluation.metrics.to_log_dict()


def test_map_cutoff_validation():
    q = EncodedRetrievalSet(torch.eye(2)[:1], torch.tensor([1]), ("q",))
    g = EncodedRetrievalSet(torch.eye(2), torch.tensor([1, 2]), ("a", "b"))
    with pytest.raises(ValueError):
        evaluate_category_retrieval(q, g, map_at_k=(0,))
    with pytest.raises(ValueError):
        evaluate_category_retrieval(q, g, map_at_k=(3,))
