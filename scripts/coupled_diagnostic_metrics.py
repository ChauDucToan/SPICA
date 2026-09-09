"""CPU-only geometry and paired ranking metrics for coupled diagnostics."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def _unit_rows(value: np.ndarray) -> np.ndarray:
    """Normalize rows without overflowing while computing their norms."""
    scale = np.max(np.abs(value), axis=1)
    if np.any(scale == 0):
        raise ValueError("zero-norm feature/query rows are not valid")
    scaled = value / scale[:, None]
    return scaled / np.sqrt(np.sum(scaled * scaled, axis=1))[:, None]


def _feature_input(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("features and labels must be array-like") from exc
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("features must have shape (N, D), with N >= 2 and D >= 1")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must have shape (N,) matching features")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must be finite")
    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        raise ValueError("numeric labels must be finite")
    return x, y


def _group_indices(labels: np.ndarray) -> dict[Any, np.ndarray]:
    groups: dict[Any, list[int]] = {}
    for index, raw_label in enumerate(labels):
        label = raw_label.item() if isinstance(raw_label, np.generic) else raw_label
        try:
            groups.setdefault(label, []).append(index)
        except TypeError as exc:
            raise ValueError("labels must contain hashable values") from exc
    return {label: np.asarray(indices, dtype=np.int64) for label, indices in groups.items()}


def _pair_mean(total: float, pairs: int) -> float | None:
    return None if pairs == 0 else float(total / pairs)


def feature_geometry(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Return JSON-safe, non-whitened geometry summaries for feature rows.

    All pair means are ordered-pair means, so each unordered pair has exactly
    twice the weight.  No N-by-N similarity matrix is constructed.
    """
    x, y = _feature_input(features, labels)
    n, dimension = x.shape
    row_scale = np.max(np.abs(x), axis=1)
    if np.any(row_scale == 0):
        raise ValueError("zero-norm feature rows are not valid")
    row_norms = row_scale * np.sqrt(np.sum((x / row_scale[:, None]) ** 2, axis=1))
    if not np.all(np.isfinite(row_norms)):
        raise ValueError("raw row norms overflow float64")
    raw_centered = x - x.mean(axis=0)
    raw_covariance_trace = float(np.sum(raw_centered * raw_centered) / n)

    unit = _unit_rows(x)
    row_sum = unit.sum(axis=0)
    off_diagonal_pairs = n * (n - 1)
    mean_offdiagonal = _pair_mean(
        float(np.dot(row_sum, row_sum) - n), off_diagonal_pairs
    )

    groups = _group_indices(y)
    within_sum = 0.0
    within_pairs = 0
    group_summaries: dict[str, int] = {}
    for label, indices in groups.items():
        group_sum = unit[indices].sum(axis=0)
        size = int(indices.size)
        within_sum += float(np.dot(group_sum, group_sum) - size)
        within_pairs += size * (size - 1)
        group_summaries[str(label)] = size
    total_offdiagonal_sum = float(np.dot(row_sum, row_sum) - n)
    between_pairs = off_diagonal_pairs - within_pairs
    between_sum = total_offdiagonal_sum - within_sum

    # Subtract a reference row first so identical directions remain exactly zero.
    centered = unit - unit[0]
    centered -= centered.mean(axis=0)
    centered_variance = float(np.sum(centered * centered) / n)
    covariance = (centered.T @ centered) / n
    eigenvalues = np.linalg.eigh(covariance)[0].astype(np.float64, copy=False)
    # Only remove negative round-off from the PSD covariance eigensolve.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if centered_variance == 0.0:
        rank = 0
        effective_rank = 0.0
        participation_ratio = 0.0
        top_fraction = 0.0
        condition = None
    else:
        positive = eigenvalues[eigenvalues > 0.0]
        rank = int(positive.size)
        probabilities = eigenvalues / centered_variance
        entropy = float(-np.sum(probabilities[probabilities > 0] * np.log(
            probabilities[probabilities > 0]
        )))
        effective_rank = float(np.exp(entropy))
        participation_ratio = float(centered_variance**2 / np.sum(eigenvalues**2))
        top_fraction = float(eigenvalues[-1] / centered_variance)
        condition = float(eigenvalues[-1] / positive[0]) if rank else None

    return {
        "n": int(n),
        "dimension": int(dimension),
        "class_count": int(len(groups)),
        "class_counts": group_summaries,
        "counts": {"samples": int(n), "classes": int(len(groups))},
        "raw_row_norms": {
            "min": float(np.min(row_norms)),
            "max": float(np.max(row_norms)),
            "mean": float(np.mean(row_norms)),
            "std": float(np.std(row_norms)),
        },
        "raw_centered_covariance_trace": raw_covariance_trace,
        "mean_offdiagonal_cosine": mean_offdiagonal,
        "mean_within_class_cosine": _pair_mean(within_sum, within_pairs),
        "mean_between_class_cosine": _pair_mean(between_sum, between_pairs),
        "centered_variance": centered_variance,
        "covariance_eigenvalues": [float(value) for value in eigenvalues],
        "spectrum": [float(value) for value in eigenvalues],
        "centered_rank": rank,
        "covariance_entropy_effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "anisotropy": {
            "largest_eigenvalue_fraction": top_fraction,
            "largest_to_smallest_nonzero_eigenvalue": condition,
            "effective_rank_fraction_of_dimension": float(effective_rank / dimension),
        },
        "definitions": {
            "unit_features": "u_i = x_i / ||x_i||_2",
            "mean_offdiagonal_cosine": "(||sum_i u_i||_2^2 - N) / (N*(N-1))",
            "within_between": "group-sum identity, excluding self, weighted by exact ordered-pair counts",
            "centered_covariance": "C = (U - mean(U))^T (U - mean(U)) / N",
            "centered_variance": "trace(C)",
            "effective_rank": "exp(-sum_k p_k log(p_k)), p_k = eigenvalue_k / trace(C)",
            "participation_ratio": "trace(C)^2 / sum_k eigenvalue_k^2",
            "rank": "number of positive clipped covariance eigenvalues; exactly 0 when centered variance is 0",
        },
    }


def _paired_input(*values: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("all ranking inputs must have shape (N, D)")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all ranking inputs must have the same shape")
    if arrays[0].shape[0] < 1 or arrays[0].shape[1] < 1:
        raise ValueError("ranking inputs must have N >= 1 and D >= 1")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("ranking inputs must be finite")
    return arrays


def paired_rank_report(
    query: np.ndarray,
    positive: np.ndarray,
    negative_full: np.ndarray,
    negative_canonical: np.ndarray,
) -> dict[str, Any]:
    """Compare paired full and canonical negatives with stable softplus ranking."""
    q, p, nf, nc = _paired_input(query, positive, negative_full, negative_canonical)
    q, p, nf, nc = (_unit_rows(value) for value in (q, p, nf, nc))
    positive_score = np.sum(q * p, axis=1)
    negative_full_score = np.sum(q * nf, axis=1)
    negative_canonical_score = np.sum(q * nc, axis=1)
    margin_full = positive_score - negative_full_score
    margin_canonical = positive_score - negative_canonical_score
    loss_full = np.logaddexp(0.0, 0.2 - margin_full)
    loss_canonical = np.logaddexp(0.0, 0.2 - margin_canonical)
    loss_delta = loss_canonical - loss_full
    names = (
        "positive_score", "negative_full_score", "negative_canonical_score",
        "margin_full", "margin_canonical", "loss_full", "loss_canonical",
        "loss_delta_canonical_minus_full",
    )
    columns = (positive_score, negative_full_score, negative_canonical_score,
               margin_full, margin_canonical, loss_full, loss_canonical, loss_delta)
    means = {name: float(np.mean(column)) for name, column in zip(names, columns)}
    rows = [
        {name: float(column[index]) for name, column in zip(names, columns)}
        for index in range(q.shape[0])
    ]
    se = None if q.shape[0] < 2 else float(np.std(loss_delta, ddof=1) / np.sqrt(q.shape[0]))
    result: dict[str, Any] = {
        "n": int(q.shape[0]),
        "means": means,
        "paired_delta_se": {"loss_delta_canonical_minus_full": se},
        "paired_delta_se_note": "descriptive sample SE only; not a significance claim",
        "per_query": rows,
    }
    result.update({f"mean_{name}": value for name, value in means.items()})
    result["se_loss_delta_canonical_minus_full"] = se
    result["loss_delta_se"] = se
    return result


def self_check() -> dict[str, Any]:
    constant = feature_geometry(np.array([[3.0, 0.0], [6.0, 0.0]]), np.array([0, 1]))
    assert constant["centered_rank"] == 0
    assert constant["mean_offdiagonal_cosine"] == 1.0
    repeated = feature_geometry(np.tile([0.3, 0.7, 1.9], (101, 1)), np.zeros(101))
    assert repeated["centered_variance"] == repeated["covariance_entropy_effective_rank"] == 0

    diverse_features = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    diverse_labels = np.array([0, 0, 1, 1])
    diverse = feature_geometry(diverse_features, diverse_labels)
    assert diverse["centered_rank"] == 2
    assert np.isclose(diverse["covariance_entropy_effective_rank"], 2.0)
    assert np.isclose(diverse["participation_ratio"], 2.0)

    unit = _unit_rows(diverse_features)
    brute = unit @ unit.T
    mask = ~np.eye(4, dtype=bool)
    within = (diverse_labels[:, None] == diverse_labels[None, :]) & mask
    between = (diverse_labels[:, None] != diverse_labels[None, :])
    assert np.isclose(diverse["mean_within_class_cosine"], brute[within].mean())
    assert np.isclose(diverse["mean_between_class_cosine"], brute[between].mean())

    query = np.array([[1e300, -1e300], [-1e300, 1e300]])
    positive = query.copy()
    negative = -query
    report = paired_rank_report(query, positive, negative, negative)
    assert all(np.isfinite(row["loss_full"]) for row in report["per_query"])
    assert report["means"]["loss_delta_canonical_minus_full"] == 0.0
    scaled = paired_rank_report(query * 7.0, positive * 3.0, negative * 11.0, negative * 5.0)
    assert np.allclose(report["means"]["loss_full"], scaled["means"]["loss_full"])
    assert np.allclose(
        feature_geometry(diverse_features * np.array([[2.0], [3.0], [4.0], [5.0]]), diverse_labels)["covariance_eigenvalues"],
        diverse["covariance_eigenvalues"],
    )
    return {"status": "PASS", "checks": ["constant_rank", "spectrum", "pair_means", "ranking", "norm_invariance"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-self-check", action="store_true", help="run NumPy-only checks")
    args = parser.parse_args()
    if args.cpu_self_check:
        print(json.dumps(self_check(), sort_keys=True))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
