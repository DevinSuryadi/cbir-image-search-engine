from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_distances

from .preprocessing import list_image_files, read_image
from .search import SearchResult, SearchResponse


BOVW_FEATURE_SIFT = "sift"
BOVW_FEATURE_AKAZE = "akaze"
BOVW_FEATURE_ORB = "orb"
SUPPORTED_BOVW_FEATURES = (BOVW_FEATURE_SIFT, BOVW_FEATURE_AKAZE, BOVW_FEATURE_ORB)


@dataclass
class BoVWIndex:
    image_paths: list[str]
    vocabulary: MiniBatchKMeans
    tfidf_transformer: TfidfTransformer
    tfidf_matrix: np.ndarray
    feature_type: str
    vocabulary_size: int
    build_seconds: float


@dataclass
class BoVWEvaluation:
    top_k: int
    query_count: int
    mean_precision: float


def create_feature_extractor(feature_type: str):
    """Create a local feature extractor for BoVW."""
    if feature_type == BOVW_FEATURE_SIFT:
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is not available in this OpenCV installation")
        return cv2.SIFT_create()

    if feature_type == BOVW_FEATURE_AKAZE:
        return cv2.AKAZE_create()

    if feature_type == BOVW_FEATURE_ORB:
        return cv2.ORB_create(nfeatures=1500)

    raise ValueError(f"Unsupported BoVW feature type: {feature_type}")


def extract_local_descriptors(image_path: str | Path, feature_type: str) -> np.ndarray | None:
    """Extract local descriptors from one image."""
    image_bgr = read_image(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    extractor = create_feature_extractor(feature_type)
    _, descriptors = extractor.detectAndCompute(image_gray, None)

    if descriptors is None or len(descriptors) == 0:
        return None

    return descriptors.astype(np.float32)


def collect_training_descriptors(
    image_paths: list[Path],
    feature_type: str,
    max_descriptors: int = 50000,
) -> np.ndarray:
    """Collect local descriptors used to train the visual vocabulary."""
    collected = []

    for image_path in image_paths:
        descriptors = extract_local_descriptors(image_path, feature_type)
        if descriptors is not None:
            collected.append(descriptors)

    if not collected:
        raise ValueError("No local descriptors were extracted from the dataset")

    all_descriptors = np.vstack(collected).astype(np.float32)
    if len(all_descriptors) <= max_descriptors:
        return all_descriptors

    rng = np.random.default_rng(seed=42)
    selected_indices = rng.choice(len(all_descriptors), size=max_descriptors, replace=False)
    return all_descriptors[selected_indices]


def build_visual_vocabulary(
    descriptors: np.ndarray,
    vocabulary_size: int,
    random_state: int = 42,
) -> MiniBatchKMeans:
    """Cluster local descriptors into visual words."""
    if len(descriptors) < vocabulary_size:
        vocabulary_size = len(descriptors)

    vocabulary = MiniBatchKMeans(
        n_clusters=vocabulary_size,
        batch_size=2048,
        random_state=random_state,
        n_init=3,
    )
    vocabulary.fit(descriptors)
    return vocabulary


def encode_bovw_histogram(
    descriptors: np.ndarray | None,
    vocabulary: MiniBatchKMeans,
) -> np.ndarray:
    """Encode local descriptors as one visual word histogram."""
    vocabulary_size = vocabulary.n_clusters
    histogram = np.zeros(vocabulary_size, dtype=np.float32)

    if descriptors is None or len(descriptors) == 0:
        return histogram

    visual_words = vocabulary.predict(descriptors.astype(np.float32))
    histogram += np.bincount(visual_words, minlength=vocabulary_size).astype(np.float32)
    return histogram


def build_bovw_index(
    image_dir: str | Path,
    feature_type: str = BOVW_FEATURE_SIFT,
    vocabulary_size: int = 256,
    max_descriptors: int = 50000,
    verbose: bool = False,
) -> BoVWIndex:
    """Build a BoVW + TF-IDF image index."""
    if feature_type not in SUPPORTED_BOVW_FEATURES:
        raise ValueError(f"Unsupported BoVW feature type: {feature_type}")

    image_paths = list_image_files(image_dir)
    if not image_paths:
        raise ValueError(f"No supported image files found in: {image_dir}")

    start_time = time.perf_counter()

    if verbose:
        print("Collecting local descriptors...")
    training_descriptors = collect_training_descriptors(
        image_paths=image_paths,
        feature_type=feature_type,
        max_descriptors=max_descriptors,
    )

    if verbose:
        print("Building visual vocabulary...")
    vocabulary = build_visual_vocabulary(
        descriptors=training_descriptors,
        vocabulary_size=vocabulary_size,
    )

    histograms = []
    indexed_paths = []
    for position, image_path in enumerate(image_paths, start=1):
        if verbose:
            print(f"[{position}/{len(image_paths)}] Encoding {image_path}")

        descriptors = extract_local_descriptors(image_path, feature_type)
        histogram = encode_bovw_histogram(descriptors, vocabulary)
        histograms.append(histogram)
        indexed_paths.append(str(image_path))

    histogram_matrix = np.vstack(histograms).astype(np.float32)
    tfidf_transformer = TfidfTransformer(norm="l2")
    tfidf_matrix = tfidf_transformer.fit_transform(histogram_matrix).toarray().astype(np.float32)
    build_seconds = time.perf_counter() - start_time

    return BoVWIndex(
        image_paths=indexed_paths,
        vocabulary=vocabulary,
        tfidf_transformer=tfidf_transformer,
        tfidf_matrix=tfidf_matrix,
        feature_type=feature_type,
        vocabulary_size=vocabulary.n_clusters,
        build_seconds=build_seconds,
    )


def save_bovw_index(index: BoVWIndex, index_path: str | Path) -> None:
    """Save a BoVW index to disk."""
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as index_file:
        pickle.dump(index, index_file, pickle.HIGHEST_PROTOCOL)


def load_bovw_index(index_path: str | Path) -> BoVWIndex:
    """Load a BoVW index from disk."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"BoVW index file was not found: {path}")

    with path.open("rb") as index_file:
        index = pickle.load(index_file)

    if not isinstance(index, BoVWIndex):
        raise TypeError("Loaded object is not a valid BoVWIndex")

    return index


def build_and_save_bovw_index(
    image_dir: str | Path,
    index_path: str | Path,
    feature_type: str = BOVW_FEATURE_SIFT,
    vocabulary_size: int = 256,
    max_descriptors: int = 50000,
    verbose: bool = False,
) -> BoVWIndex:
    """Build and save a BoVW index."""
    index = build_bovw_index(
        image_dir=image_dir,
        feature_type=feature_type,
        vocabulary_size=vocabulary_size,
        max_descriptors=max_descriptors,
        verbose=verbose,
    )
    save_bovw_index(index, index_path)
    return index


def search_bovw_index(
    query_image_path: str | Path,
    index: BoVWIndex,
    top_k: int = 10,
    verify_top_k: int = 0,
) -> SearchResponse:
    """Search similar images using a BoVW index."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    start_time = time.perf_counter()
    descriptors = extract_local_descriptors(query_image_path, index.feature_type)
    histogram = encode_bovw_histogram(descriptors, index.vocabulary).reshape(1, -1)
    query_tfidf = index.tfidf_transformer.transform(histogram).toarray().astype(np.float32)

    distances = cosine_distances(index.tfidf_matrix, query_tfidf).reshape(-1)
    candidate_count = min(max(top_k, verify_top_k), len(index.image_paths))
    candidate_indices = np.argsort(distances)[:candidate_count]

    results = [
        SearchResult(
            image_path=index.image_paths[index_position],
            distance=float(distances[index_position]),
        )
        for index_position in candidate_indices
    ]

    if verify_top_k > 0:
        results = rerank_bovw_with_geometric_verification(
            query_image_path=query_image_path,
            results=results,
            top_k=top_k,
        )
    else:
        results = results[:top_k]

    query_seconds = time.perf_counter() - start_time
    return SearchResponse(results=results, query_seconds=query_seconds)


def search_bovw_index_file(
    query_image_path: str | Path,
    index_path: str | Path,
    top_k: int = 10,
    verify_top_k: int = 0,
) -> SearchResponse:
    """Load a BoVW index file and search similar images."""
    index = load_bovw_index(index_path)
    return search_bovw_index(
        query_image_path=query_image_path,
        index=index,
        top_k=top_k,
        verify_top_k=verify_top_k,
    )


def rerank_bovw_with_geometric_verification(
    query_image_path: str | Path,
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Rerank BoVW candidates using ORB homography inlier counts."""
    query_keypoints, query_descriptors = compute_orb_keypoints(query_image_path)
    verified_results = []

    for result in results:
        candidate_keypoints, candidate_descriptors = compute_orb_keypoints(result.image_path)
        inlier_count = count_homography_inliers(
            query_keypoints=query_keypoints,
            query_descriptors=query_descriptors,
            candidate_keypoints=candidate_keypoints,
            candidate_descriptors=candidate_descriptors,
        )
        verified_results.append(
            SearchResult(
                image_path=result.image_path,
                distance=result.distance,
                orb_matches=inlier_count,
                initial_distance=result.distance,
            )
        )

    verified_results.sort(key=lambda item: (-(item.orb_matches or 0), item.initial_distance))
    return verified_results[:top_k]


def compute_orb_keypoints(image_path: str | Path):
    """Compute ORB keypoints and descriptors for geometric verification."""
    image_bgr = read_image(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1500)
    return orb.detectAndCompute(image_gray, None)


def count_homography_inliers(
    query_keypoints,
    query_descriptors,
    candidate_keypoints,
    candidate_descriptors,
    ratio: float = 0.75,
) -> int:
    """Count geometrically consistent matches using homography RANSAC."""
    if query_descriptors is None or candidate_descriptors is None:
        return 0

    if len(query_descriptors) < 4 or len(candidate_descriptors) < 4:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue

        best_match, second_best_match = pair
        if best_match.distance < ratio * second_best_match.distance:
            good_matches.append(best_match)

    if len(good_matches) < 4:
        return len(good_matches)

    query_points = np.float32(
        [query_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    candidate_points = np.float32(
        [candidate_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    _, inlier_mask = cv2.findHomography(
        query_points,
        candidate_points,
        cv2.RANSAC,
        5.0,
    )
    if inlier_mask is None:
        return 0

    return int(inlier_mask.sum())


def get_category_label(image_path: str | Path) -> str:
    """Use parent folder as category label."""
    return Path(image_path).parent.name


def evaluate_bovw_index(
    index: BoVWIndex,
    top_k: int = 10,
    max_queries: int | None = None,
    verify_top_k: int = 0,
) -> BoVWEvaluation:
    """Evaluate BoVW precision@k using parent-folder labels."""
    query_paths = index.image_paths
    if max_queries is not None:
        query_paths = query_paths[:max_queries]

    precisions = []
    for query_path in query_paths:
        response = search_bovw_index(
            query_image_path=query_path,
            index=index,
            top_k=top_k + 1,
            verify_top_k=verify_top_k,
        )
        results = [
            result
            for result in response.results
            if Path(result.image_path).resolve() != Path(query_path).resolve()
        ][:top_k]

        query_label = get_category_label(query_path)
        relevant_count = sum(
            get_category_label(result.image_path) == query_label
            for result in results
        )
        precisions.append(relevant_count / top_k)

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    return BoVWEvaluation(
        top_k=top_k,
        query_count=len(query_paths),
        mean_precision=mean_precision,
    )


def evaluate_bovw_index_file(
    index_path: str | Path,
    top_k: int = 10,
    max_queries: int | None = None,
    verify_top_k: int = 0,
) -> BoVWEvaluation:
    """Load and evaluate a BoVW index file."""
    index = load_bovw_index(index_path)
    return evaluate_bovw_index(
        index=index,
        top_k=top_k,
        max_queries=max_queries,
        verify_top_k=verify_top_k,
    )
