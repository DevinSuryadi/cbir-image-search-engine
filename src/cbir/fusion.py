from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bovw import search_bovw_index_file
from .labels import get_category_label, is_same_image
from .indexing import load_index
from .reranking import rerank_results_with_orb
from .search import SearchResponse, SearchResult, search_from_index_file, search_index


@dataclass
class FusionSource:
    name: str
    method: str
    index_path: str
    weight: float = 1.0
    depth: int = 50
    use_orb_rerank: bool = False
    candidate_k: int = 50
    rerank_strategy: str = "weighted"
    distance_weight: float = 0.4
    orb_weight: float = 0.6
    verify_top_k: int = 0


@dataclass
class FusionEvaluation:
    top_k: int
    query_count: int
    mean_precision: float


DEFAULT_FUSION_SOURCES = [
    FusionSource(
        name="hog_weighted_orb",
        method="classic",
        index_path="models/index-hog.pkl",
        weight=1.0,
        depth=50,
        use_orb_rerank=True,
        candidate_k=50,
        rerank_strategy="weighted",
        distance_weight=0.4,
        orb_weight=0.6,
    ),
    FusionSource(
        name="hsv_hog",
        method="classic",
        index_path="models/index-hsv-hog.pkl",
        weight=0.7,
        depth=50,
    ),
    FusionSource(
        name="bovw",
        method="bovw",
        index_path="models/index-bovw.pkl",
        weight=1.0,
        depth=50,
        verify_top_k=50,
    ),
]


def source_from_config(config: dict) -> FusionSource:
    """Create a FusionSource from a config dictionary."""
    return FusionSource(
        name=config["name"],
        method=config["method"],
        index_path=config["index_path"],
        weight=float(config.get("weight", 1.0)),
        depth=int(config.get("depth", 50)),
        use_orb_rerank=bool(config.get("use_orb_rerank", False)),
        candidate_k=int(config.get("candidate_k", 50)),
        rerank_strategy=config.get("rerank_strategy", "weighted"),
        distance_weight=float(config.get("distance_weight", 0.4)),
        orb_weight=float(config.get("orb_weight", 0.6)),
        verify_top_k=int(config.get("verify_top_k", 0)),
    )


def sources_from_config(configs: list[dict] | None) -> list[FusionSource]:
    """Create fusion sources from config, falling back to the default setup."""
    if not configs:
        return DEFAULT_FUSION_SOURCES

    return [source_from_config(config) for config in configs]


def search_source(query_image_path: str | Path, source: FusionSource) -> SearchResponse:
    """Run one configured search source."""
    if source.method == "classic":
        if not source.use_orb_rerank:
            return search_from_index_file(
                query_image_path=query_image_path,
                index_path=source.index_path,
                top_k=source.depth,
            )

        index = load_index(source.index_path)
        response = search_index(
            query_image_path=query_image_path,
            index=index,
            top_k=max(source.candidate_k, source.depth),
        )
        rerank_response = rerank_results_with_orb(
            query_image_path=query_image_path,
            candidate_results=response.results,
            top_k=source.depth,
            strategy=source.rerank_strategy,
            distance_weight=source.distance_weight,
            orb_weight=source.orb_weight,
        )
        response.results = rerank_response.results
        response.query_seconds += rerank_response.rerank_seconds
        return response

    if source.method == "bovw":
        return search_bovw_index_file(
            query_image_path=query_image_path,
            index_path=source.index_path,
            top_k=source.depth,
            verify_top_k=source.verify_top_k,
        )

    raise ValueError(f"Unsupported fusion source method: {source.method}")


def reciprocal_rank_fusion(
    ranked_results: list[tuple[FusionSource, list[SearchResult]]],
    top_k: int,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Fuse ranked search results with weighted Reciprocal Rank Fusion."""
    fused: dict[str, dict[str, object]] = {}

    for source, results in ranked_results:
        for rank, result in enumerate(results, start=1):
            key = str(Path(result.image_path))
            rank_score = source.weight / (rrf_k + rank)

            if key not in fused:
                fused[key] = {
                    "image_path": result.image_path,
                    "score": 0.0,
                    "source_count": 0,
                    "best_distance": result.distance,
                }

            fused[key]["score"] = float(fused[key]["score"]) + rank_score
            fused[key]["source_count"] = int(fused[key]["source_count"]) + 1
            fused[key]["best_distance"] = min(float(fused[key]["best_distance"]), result.distance)

    final_results = [
        SearchResult(
            image_path=str(item["image_path"]),
            distance=float(item["best_distance"]),
            rerank_score=float(item["score"]),
            source_count=int(item["source_count"]),
        )
        for item in fused.values()
    ]
    final_results.sort(key=lambda result: (-(result.rerank_score or 0.0), result.distance))
    return final_results[:top_k]


def search_fusion(
    query_image_path: str | Path,
    sources: list[FusionSource] | None = None,
    top_k: int = 10,
    rrf_k: int = 60,
) -> SearchResponse:
    """Search with multiple sources and fuse their rankings."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    active_sources = sources or DEFAULT_FUSION_SOURCES
    start_time = time.perf_counter()
    ranked_results = []
    total_query_seconds = 0.0

    for source in active_sources:
        response = search_source(query_image_path=query_image_path, source=source)
        ranked_results.append((source, response.results))
        total_query_seconds += response.query_seconds

    results = reciprocal_rank_fusion(
        ranked_results=ranked_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )
    elapsed_seconds = time.perf_counter() - start_time
    return SearchResponse(results=results, query_seconds=max(elapsed_seconds, total_query_seconds))


def evaluate_fusion(
    query_paths: list[str],
    sources: list[FusionSource] | None = None,
    top_k: int = 10,
    rrf_k: int = 60,
    max_queries: int | None = None,
) -> FusionEvaluation:
    """Evaluate fusion precision@k using parent-folder labels."""
    if max_queries is not None:
        query_paths = query_paths[:max_queries]

    precisions = []
    for query_path in query_paths:
        response = search_fusion(
            query_image_path=query_path,
            sources=sources,
            top_k=top_k + 1,
            rrf_k=rrf_k,
        )
        results = [
            result
            for result in response.results
            if not is_same_image(query_path, result.image_path)
        ][:top_k]

        query_label = get_category_label(query_path)
        relevant_count = sum(
            get_category_label(result.image_path) == query_label
            for result in results
        )
        precisions.append(relevant_count / top_k)

    return FusionEvaluation(
        top_k=top_k,
        query_count=len(query_paths),
        mean_precision=float(np.mean(precisions)) if precisions else 0.0,
    )
