from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .descriptors import (
    compute_combined_hsv_hog_descriptor,
    compute_hog_descriptor,
    compute_region_hsv_descriptor,
)
from .indexing import (
    DESCRIPTOR_HOG,
    DESCRIPTOR_HSV,
    DESCRIPTOR_HSV_HOG,
    DESCRIPTOR_LEGACY_HSV,
    ImageIndex,
    load_index,
)
from .preprocessing import read_image, read_preprocess


@dataclass
class SearchResult:
    image_path: str
    distance: float
    orb_matches: int | None = None
    initial_distance: float | None = None


@dataclass
class SearchResponse:
    results: list[SearchResult]
    query_seconds: float


def cosine_distances(query_descriptor: np.ndarray, descriptor_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine distance between one query descriptor and all indexed descriptors."""
    if query_descriptor.ndim != 1:
        raise ValueError("Query descriptor must be a 1D vector")

    if descriptor_matrix.ndim != 2:
        raise ValueError("Descriptor matrix must be a 2D matrix")

    if descriptor_matrix.shape[1] != query_descriptor.shape[0]:
        raise ValueError("Query descriptor length must match index descriptor length")

    query_norm = np.linalg.norm(query_descriptor)
    matrix_norm = np.linalg.norm(descriptor_matrix, axis=1)

    similarity = descriptor_matrix @ query_descriptor
    similarity = similarity / ((matrix_norm * query_norm) + 1e-10)

    return (1.0 - similarity).astype(np.float32)


def search_index(
    query_image_path: str | Path,
    index: ImageIndex,
    top_k: int = 10,
) -> SearchResponse:
    """Search top-k similar images from a loaded index."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    start_time = time.perf_counter()

    if index.descriptor_name in (DESCRIPTOR_HSV, DESCRIPTOR_LEGACY_HSV):
        query_hsv = read_preprocess(query_image_path)
        query_descriptor = compute_region_hsv_descriptor(query_hsv, bins=index.bins)
    elif index.descriptor_name == DESCRIPTOR_HOG:
        query_bgr = read_image(query_image_path)
        query_descriptor = compute_hog_descriptor(query_bgr)
    elif index.descriptor_name == DESCRIPTOR_HSV_HOG:
        query_bgr = read_image(query_image_path)
        query_descriptor = compute_combined_hsv_hog_descriptor(query_bgr, bins=index.bins)
    else:
        raise ValueError(f"Unsupported descriptor type in index: {index.descriptor_name}")

    distances = cosine_distances(query_descriptor, index.descriptors)

    result_count = min(top_k, len(index.image_paths))
    result_indices = np.argsort(distances)[:result_count]

    results = [
        SearchResult(
            image_path=index.image_paths[index_position],
            distance=float(distances[index_position]),
        )
        for index_position in result_indices
    ]

    query_seconds = time.perf_counter() - start_time
    return SearchResponse(results=results, query_seconds=query_seconds)


def search_from_index_file(
    query_image_path: str | Path,
    index_path: str | Path,
    top_k: int = 10,
) -> SearchResponse:
    """Load an index file and search top-k similar images."""
    index = load_index(index_path)
    return search_index(query_image_path=query_image_path, index=index, top_k=top_k)
