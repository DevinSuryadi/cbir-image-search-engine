from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .orb import compute_orb_keypoints, count_good_orb_matches
from .search import SearchResult


@dataclass
class RerankResponse:
    results: list[SearchResult]
    rerank_seconds: float


def rerank_results_with_orb(
    query_image_path: str | Path,
    candidate_results: list[SearchResult],
    top_k: int = 10,
    ratio: float = 0.75,
    n_features: int = 1000,
    strategy: str = "weighted",
    distance_weight: float = 0.4,
    orb_weight: float = 0.6,
) -> RerankResponse:
    """Rerank initial vector-search candidates using ORB keypoint matching."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if strategy not in ("orb_matches", "weighted"):
        raise ValueError("strategy must be 'orb_matches' or 'weighted'")

    if distance_weight < 0 or orb_weight < 0:
        raise ValueError("Rerank weights must be non-negative")

    if distance_weight == 0 and orb_weight == 0:
        raise ValueError("At least one rerank weight must be greater than zero")

    start_time = time.perf_counter()
    _, query_descriptors = compute_orb_keypoints(query_image_path, n_features=n_features)
    reranked_results = []

    for result in candidate_results:
        _, candidate_descriptors = compute_orb_keypoints(result.image_path, n_features=n_features)
        match_count = count_good_orb_matches(
            query_descriptors=query_descriptors,
            candidate_descriptors=candidate_descriptors,
            ratio=ratio,
        )
        reranked_results.append(
            SearchResult(
                image_path=result.image_path,
                distance=result.distance,
                orb_matches=match_count,
                initial_distance=result.distance,
            )
        )

    if strategy == "weighted":
        apply_weighted_rerank_scores(
            results=reranked_results,
            distance_weight=distance_weight,
            orb_weight=orb_weight,
        )
        reranked_results.sort(key=lambda result: result.rerank_score)
    else:
        reranked_results.sort(key=lambda result: (-result.orb_matches, result.initial_distance))

    rerank_seconds = time.perf_counter() - start_time

    return RerankResponse(
        results=reranked_results[:top_k],
        rerank_seconds=rerank_seconds,
    )


def apply_weighted_rerank_scores(
    results: list[SearchResult],
    distance_weight: float,
    orb_weight: float,
) -> None:
    """Assign weighted rerank scores using normalized distance and ORB match quality."""
    if not results:
        return

    distances = [result.initial_distance for result in results]
    match_counts = [result.orb_matches or 0 for result in results]

    min_distance = min(distances)
    max_distance = max(distances)
    max_matches = max(match_counts)
    distance_range = max_distance - min_distance

    total_weight = distance_weight + orb_weight
    normalized_distance_weight = distance_weight / total_weight
    normalized_orb_weight = orb_weight / total_weight

    for result in results:
        if distance_range == 0:
            distance_penalty = 0.0
        else:
            distance_penalty = (result.initial_distance - min_distance) / distance_range

        if max_matches == 0:
            orb_penalty = 1.0
        else:
            orb_penalty = 1.0 - ((result.orb_matches or 0) / max_matches)

        result.rerank_score = (
            normalized_distance_weight * distance_penalty
            + normalized_orb_weight * orb_penalty
        )
