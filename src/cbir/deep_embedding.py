from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .files import canonicalize_image_path, list_image_files
from .labels import is_same_image
from .labels import get_category_label, is_same_image
from .search import SearchResult, SearchResponse


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class DeepEmbeddingIndex:
    image_paths: list[str]
    embeddings: np.ndarray
    model_name: str
    build_seconds: float


@dataclass
class DeepEmbeddingEvaluation:
    top_k: int
    query_count: int
    mean_precision: float


def get_device(preferred_device: str = "auto") -> str:
    """Return the device used for deep embedding extraction."""
    import torch

    if preferred_device != "auto":
        return preferred_device

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(model_name: str = DEFAULT_CLIP_MODEL, device: str = "auto"):
    """Load a pretrained CLIP model and processor."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    resolved_device = get_device(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    torch.set_grad_enabled(False)
    return model, processor, resolved_device


def read_pil_rgb(image_path: str | Path) -> Image.Image:
    """Read an image as RGB PIL image."""
    return Image.open(image_path).convert("RGB")


def l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """L2 normalize every row in a matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-10)


def cosine_distances_numpy(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute cosine distance between an embedding matrix and one query embedding."""
    matrix_normalized = l2_normalize_matrix(matrix)
    query_normalized = l2_normalize_matrix(query)
    similarities = matrix_normalized @ query_normalized.T
    return (1.0 - similarities.reshape(-1)).astype(np.float32)


def compute_clip_embeddings(
    image_paths: list[str | Path],
    model,
    processor,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute normalized CLIP image embeddings for image paths."""
    import torch

    embeddings = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [read_pil_rgb(path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = encode_clip_images(model, inputs)

        batch_embeddings = image_features.detach().cpu().numpy().astype(np.float32)
        embeddings.append(batch_embeddings)

    embedding_matrix = np.vstack(embeddings).astype(np.float32)
    return l2_normalize_matrix(embedding_matrix).astype(np.float32)


def encode_clip_images(model, inputs):
    """Encode images with CLIP vision features and return one embedding per image.

    Some transformers installations return incompatible projected CLIP outputs.
    For image-to-image retrieval, using the vision pooled output is stable because
    the dataset and query are encoded by the same visual encoder.
    """
    if "pixel_values" not in inputs:
        raise ValueError("CLIP image inputs must contain pixel_values")

    pixel_values = inputs["pixel_values"]

    if hasattr(model, "vision_model"):
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            return vision_outputs.pooler_output
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state.mean(dim=1)

    image_features = model.get_image_features(pixel_values=pixel_values)
    if hasattr(image_features, "detach"):
        return image_features

    raise TypeError(f"Unsupported CLIP image feature output: {type(image_features)}")


def build_deep_embedding_index(
    image_dir: str | Path,
    model_name: str = DEFAULT_CLIP_MODEL,
    batch_size: int = 16,
    device: str = "auto",
    verbose: bool = False,
) -> DeepEmbeddingIndex:
    """Build an image index using pretrained CLIP embeddings."""
    image_files = list_image_files(image_dir)
    if not image_files:
        raise ValueError(f"No supported image files found in: {image_dir}")

    start_time = time.perf_counter()
    model, processor, resolved_device = load_clip_model(model_name=model_name, device=device)

    if verbose:
        print(f"Model: {model_name}")
        print(f"Device: {resolved_device}")
        print(f"Images: {len(image_files)}")

    embeddings = compute_clip_embeddings(
        image_paths=image_files,
        model=model,
        processor=processor,
        device=resolved_device,
        batch_size=batch_size,
    )
    build_seconds = time.perf_counter() - start_time

    return DeepEmbeddingIndex(
        image_paths=[canonicalize_image_path(path) for path in image_files],
        embeddings=embeddings,
        model_name=model_name,
        build_seconds=build_seconds,
    )


def save_deep_embedding_index(index: DeepEmbeddingIndex, index_path: str | Path) -> None:
    """Save a deep embedding index to disk."""
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as index_file:
        pickle.dump(index, index_file, pickle.HIGHEST_PROTOCOL)


def load_deep_embedding_index(index_path: str | Path) -> DeepEmbeddingIndex:
    """Load a deep embedding index from disk."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Deep embedding index file was not found: {path}")

    with path.open("rb") as index_file:
        index = pickle.load(index_file)

    if not isinstance(index, DeepEmbeddingIndex):
        raise TypeError("Loaded object is not a valid DeepEmbeddingIndex")

    return index


def build_and_save_deep_embedding_index(
    image_dir: str | Path,
    index_path: str | Path,
    model_name: str = DEFAULT_CLIP_MODEL,
    batch_size: int = 16,
    device: str = "auto",
    verbose: bool = False,
) -> DeepEmbeddingIndex:
    """Build and save a deep embedding image index."""
    index = build_deep_embedding_index(
        image_dir=image_dir,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )
    save_deep_embedding_index(index=index, index_path=index_path)
    return index


def search_deep_embedding_index(
    query_image_path: str | Path,
    index: DeepEmbeddingIndex,
    top_k: int = 10,
    batch_size: int = 1,
    device: str = "auto",
) -> SearchResponse:
    """Search similar images using a deep embedding index."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    load_start = time.perf_counter()
    model, processor, resolved_device = load_clip_model(
        model_name=index.model_name,
        device=device,
    )
    load_seconds = time.perf_counter() - load_start
    response = search_deep_embedding_index_with_model(
        query_image_path=query_image_path,
        index=index,
        model=model,
        processor=processor,
        device=resolved_device,
        top_k=top_k,
        batch_size=batch_size,
    )
    response.load_seconds = load_seconds
    return response


def search_deep_embedding_index_with_model(
    query_image_path: str | Path,
    index: DeepEmbeddingIndex,
    model,
    processor,
    device: str,
    top_k: int = 10,
    batch_size: int = 1,
) -> SearchResponse:
    """Search using an already-loaded deep model and index."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    start_time = time.perf_counter()
    query_embedding = compute_clip_embeddings(
        image_paths=[query_image_path],
        model=model,
        processor=processor,
        device=device,
        batch_size=batch_size,
    )

    distances = cosine_distances_numpy(index.embeddings, query_embedding)
    result_count = min(top_k, len(index.image_paths))
    result_indices = np.argsort(distances)[:result_count]

    results = []
    for index_position in result_indices:
        if is_same_image(query_image_path, index.image_paths[index_position]):
            continue

        results.append(
            SearchResult(
                image_path=index.image_paths[index_position],
                distance=float(distances[index_position]),
            )
        )
        if len(results) >= result_count:
            break

    query_seconds = time.perf_counter() - start_time
    return SearchResponse(results=results, query_seconds=query_seconds)


def search_deep_embedding_index_file(
    query_image_path: str | Path,
    index_path: str | Path,
    top_k: int = 10,
    device: str = "auto",
) -> SearchResponse:
    """Load a deep embedding index and search similar images."""
    index = load_deep_embedding_index(index_path)
    return search_deep_embedding_index(
        query_image_path=query_image_path,
        index=index,
        top_k=top_k,
        device=device,
    )


def evaluate_deep_embedding_index(
    index: DeepEmbeddingIndex,
    top_k: int = 10,
    max_queries: int | None = None,
    device: str = "auto",
) -> DeepEmbeddingEvaluation:
    """Evaluate deep embedding precision@k using parent-folder labels."""
    query_paths = index.image_paths
    if max_queries is not None:
        query_paths = query_paths[:max_queries]

    model, processor, resolved_device = load_clip_model(
        model_name=index.model_name,
        device=device,
    )

    precisions = []
    for query_path in query_paths:
        query_embedding = compute_clip_embeddings(
            image_paths=[query_path],
            model=model,
            processor=processor,
            device=resolved_device,
            batch_size=1,
        )
        distances = cosine_distances_numpy(index.embeddings, query_embedding)
        result_indices = np.argsort(distances)[: top_k + 1]

        results = [
            index.image_paths[index_position]
            for index_position in result_indices
            if not is_same_image(query_path, index.image_paths[index_position])
        ][:top_k]

        query_label = get_category_label(query_path)
        relevant_count = sum(get_category_label(result_path) == query_label for result_path in results)
        precisions.append(relevant_count / top_k)

    return DeepEmbeddingEvaluation(
        top_k=top_k,
        query_count=len(query_paths),
        mean_precision=float(np.mean(precisions)) if precisions else 0.0,
    )


def evaluate_deep_embedding_index_file(
    index_path: str | Path,
    top_k: int = 10,
    max_queries: int | None = None,
    device: str = "auto",
) -> DeepEmbeddingEvaluation:
    """Load and evaluate a deep embedding index file."""
    index = load_deep_embedding_index(index_path)
    return evaluate_deep_embedding_index(
        index=index,
        top_k=top_k,
        max_queries=max_queries,
        device=device,
    )
