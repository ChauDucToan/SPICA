import pytest
import torch

from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.metrics import evaluate_category_retrieval


def test_truncated_map_preserves_full_map_and_uses_prefix_positive_denominator():
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
    # Historical SPICA AP@K truncates both scores and relevance before AP,
    # so the denominator is positives found in the returned prefix.
    assert evaluation.metrics.mean_average_precision_at_k[1] == pytest.approx(1.0)
    assert evaluation.metrics.mean_average_precision_at_k[2] == pytest.approx(1.0)
    assert evaluation.metrics.mean_average_precision_at_k[4] == pytest.approx(5 / 6)
    assert evaluation.top_indices.shape == (1, 2)
    assert "retrieval/mAP" in evaluation.metrics.to_log_dict()
    assert "retrieval/mAP@2" in evaluation.metrics.to_log_dict()
    assert "retrieval/mAP@2_prefix_positive" in evaluation.metrics.to_log_dict(
        map_at_k_denominator="prefix_positive"
    )


def test_map_denominator_variants_are_explicit():
    # Ranking relevance is [1, 0, 1, 0]: full AP=5/6, while AP@2 is
    # respectively 1 (prefix positives), 1/2 (all relevant), and 1/2
    # (min(total relevant, K)).
    q = EncodedRetrievalSet(torch.tensor([[1.0, 0.0]]), torch.tensor([1]), ("q",))
    g = EncodedRetrievalSet(
        torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.4, 0.916515]]),
        torch.tensor([1, 2, 1, 2]),
        ("a", "b", "c", "d"),
    )
    for convention, expected in (
        ("prefix_positive", 1.0),
        ("all_relevant", 0.5),
        ("min_relevant_k", 0.5),
    ):
        result = evaluate_category_retrieval(
            q, g, precision_at_k=(1,), map_at_k=(2,), map_at_k_denominator=convention
        )
        assert result.metrics.mean_average_precision == pytest.approx(5 / 6)
        assert result.metrics.mean_average_precision_at_k[2] == pytest.approx(expected)


def test_map_cutoff_validation():
    q = EncodedRetrievalSet(torch.eye(2)[:1], torch.tensor([1]), ("q",))
    g = EncodedRetrievalSet(torch.eye(2), torch.tensor([1, 2]), ("a", "b"))
    with pytest.raises(ValueError):
        evaluate_category_retrieval(q, g, map_at_k=(0,))
    with pytest.raises(ValueError):
        evaluate_category_retrieval(q, g, map_at_k=(3,))
