from collections.abc import Mapping
from typing import Any, Literal

import torch

from .embeddings import EncodedRetrievalSet
from .metrics import CategoryRetrievalEvaluation

QuerySelection = Literal["first", "best", "worst"]

RETRIEVAL_TABLE_COLUMNS = (
    "query_index",
    "query_path",
    "query_label",
    "query_class",
    "query_AP",
    "rank",
    "gallery_path",
    "gallery_label",
    "gallery_class",
    "cosine_similarity",
    "relevant",
)


def build_retrieval_table_rows(
    evaluation: CategoryRetrievalEvaluation,
    queries: EncodedRetrievalSet,
    gallery: EncodedRetrievalSet,
    class_names: Mapping[int, str],
    *,
    num_queries: int = 10,
    results_per_query: int = 5,
    selection: QuerySelection = "worst",
) -> list[list[Any]]:
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")
    if results_per_query <= 0:
        raise ValueError(f"results_per_query must be positive, got {results_per_query}")
    if results_per_query > evaluation.top_indices.shape[1]:
        raise ValueError(
            f"Evaluation only stores {evaluation.top_indices.shape[1]} results "
            f"per query, but {results_per_query} were requested"
        )

    available_queries = queries.embeddings.shape[0]
    num_selected = min(num_queries, available_queries)
    average_precision = evaluation.average_precision_per_query

    if selection == "first":
        selected_indices = torch.arange(num_selected)
    elif selection == "best":
        selected_indices = torch.argsort(
            average_precision,
            descending=True,
            stable=True,
        )[:num_selected]
    elif selection == "worst":
        selected_indices = torch.argsort(
            average_precision,
            stable=True,
        )[:num_selected]
    else:
        raise ValueError(f"Unsupported query selection: {selection!r}")

    rows: list[list[Any]] = []
    for query_index_tensor in selected_indices:
        query_index = int(query_index_tensor.item())
        query_label = int(queries.labels[query_index].item())
        query_class = class_names.get(query_label, f"class_{query_label}")
        query_ap = float(average_precision[query_index].item())

        for rank_offset in range(results_per_query):
            gallery_index = int(evaluation.top_indices[query_index, rank_offset].item())
            gallery_label = int(gallery.labels[gallery_index].item())
            gallery_class = class_names.get(
                gallery_label,
                f"class_{gallery_label}",
            )

            rows.append(
                [
                    query_index,
                    queries.paths[query_index],
                    query_label,
                    query_class,
                    query_ap,
                    rank_offset + 1,
                    gallery.paths[gallery_index],
                    gallery_label,
                    gallery_class,
                    float(evaluation.top_scores[query_index, rank_offset].item()),
                    query_label == gallery_label,
                ]
            )

    return rows
