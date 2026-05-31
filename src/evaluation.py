from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .indexing import ImageIndex, load_index
from .search import SearchResult, search_index


@dataclass
class QueryEvaluation:
    query_image_path: str
    query_label: str
    precision: float
    relevant_count: int
    results: list[SearchResult]


@dataclass
class EvaluationSummary:
    top_k: int
    query_count: int
    mean_precision: float
    evaluations: list[QueryEvaluation]


def get_category_label(image_path: str | Path) -> str:
    """Use the parent folder name as the image category label."""
    return Path(image_path).parent.name


def is_same_image(first_path: str | Path, second_path: str | Path) -> bool:
    """Compare two image paths after resolving them."""
    return Path(first_path).resolve() == Path(second_path).resolve()


def filter_self_match(
    query_image_path: str | Path,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Remove the query image itself from search results."""
    return [
        result
        for result in results
        if not is_same_image(query_image_path, result.image_path)
    ]


def evaluate_query(
    query_image_path: str | Path,
    index: ImageIndex,
    top_k: int = 10,
) -> QueryEvaluation:
    """Evaluate one query using precision@k based on folder category labels."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    response = search_index(
        query_image_path=query_image_path,
        index=index,
        top_k=top_k + 1,
    )
    results = filter_self_match(query_image_path, response.results)[:top_k]

    query_label = get_category_label(query_image_path)
    relevant_count = sum(
        get_category_label(result.image_path) == query_label
        for result in results
    )
    precision = relevant_count / top_k

    return QueryEvaluation(
        query_image_path=str(query_image_path),
        query_label=query_label,
        precision=precision,
        relevant_count=relevant_count,
        results=results,
    )


def evaluate_index(
    index: ImageIndex,
    top_k: int = 10,
    max_queries: int | None = None,
) -> EvaluationSummary:
    """Evaluate an index using all indexed images or a limited number of queries."""
    query_paths = index.image_paths
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than zero")
        query_paths = query_paths[:max_queries]

    evaluations = [
        evaluate_query(query_image_path=query_path, index=index, top_k=top_k)
        for query_path in query_paths
    ]

    mean_precision = float(np.mean([item.precision for item in evaluations]))
    return EvaluationSummary(
        top_k=top_k,
        query_count=len(evaluations),
        mean_precision=mean_precision,
        evaluations=evaluations,
    )


def evaluate_index_file(
    index_path: str | Path,
    top_k: int = 10,
    max_queries: int | None = None,
) -> EvaluationSummary:
    """Load an index file and evaluate it with precision@k."""
    index = load_index(index_path)
    return evaluate_index(index=index, top_k=top_k, max_queries=max_queries)
