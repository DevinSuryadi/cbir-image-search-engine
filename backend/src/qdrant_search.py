"""
Qdrant Cloud search utilities for the CBIR FastAPI backend.

Provides a singleton QdrantClient and search functions used by the API endpoints.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the backend root directory (backend/src/qdrant_search.py → backend/)
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_BACKEND_ROOT / ".env")

COLLECTION_NAME = "caltech101_clip"
DEFAULT_TOP_K = 20


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Return a cached singleton QdrantClient connected to Qdrant Cloud."""
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("Cluster_Endpoint") or os.getenv("CLUSTER_ENDPOINT")
    qdrant_api_key = os.getenv("api_key") or os.getenv("API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError(
            "Qdrant Cloud credentials missing. "
            "Set 'Cluster_Endpoint' and 'api_key' in backend/.env"
        )

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30.0)


def _qdrant_hits_to_results(hits: list) -> list[dict]:
    """Convert Qdrant ScoredPoint hits into a list of result dicts."""
    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                "point_id": str(hit.id),
                "image_url": payload.get("image_path", ""),
                "category": payload.get("category", ""),
                "score": round(float(hit.score), 6),
            }
        )
    return results


def search_by_vector(
    query_vector: list[float],
    top_k: int = DEFAULT_TOP_K,
    exclude_image_url: str | None = None,
) -> list[dict]:
    """Search Qdrant for the top-k most similar vectors.

    Args:
        query_vector: Normalized CLIP embedding as a float list.
        top_k: Number of results to return.
        exclude_image_url: Optionally filter out a specific image URL from results
                           (used to exclude the query image from its own results).

    Returns:
        List of result dicts with keys: point_id, image_url, category, score.
    """
    client = get_qdrant_client()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k + 1,  # Fetch one extra to allow excluding the query itself
        with_payload=True,
    )

    results = _qdrant_hits_to_results(hits)

    if exclude_image_url:
        results = [r for r in results if r["image_url"] != exclude_image_url]

    return results[:top_k]


def search_by_text(
    query_text: str,
    model,
    processor,
    device: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Encode a text query with CLIP and search Qdrant.

    Args:
        query_text: Natural language description to search for.
        model: Loaded CLIP model.
        processor: Loaded CLIP processor.
        device: Torch device string.
        top_k: Number of results to return.

    Returns:
        List of result dicts with keys: point_id, image_url, category, score.
    """
    # Use absolute import — works regardless of how the package is loaded
    from src.cbir.deep_embedding import encode_text_clip

    query_vector = encode_text_clip(
        text=query_text,
        model=model,
        processor=processor,
        device=device,
    )
    return search_by_vector(query_vector=query_vector.tolist(), top_k=top_k)


def more_like_this(
    image_url: str,
    model,
    processor,
    device: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Find images similar to a given Supabase image URL.

    Downloads the image from the URL, encodes it with CLIP, then searches Qdrant.

    Args:
        image_url: Public Supabase Storage URL of the reference image.
        model: Loaded CLIP model.
        processor: Loaded CLIP processor.
        device: Torch device string.
        top_k: Number of results to return.

    Returns:
        List of result dicts with keys: point_id, image_url, category, score.
    """
    import io

    import httpx
    import numpy as np
    import torch
    from PIL import Image

    # Use absolute imports — consistent with how main.py imports these
    from src.cbir.deep_embedding import encode_clip_images, l2_normalize_matrix

    # Download the image from Supabase URL
    response = httpx.get(image_url, timeout=15.0)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    # Encode with CLIP
    inputs = processor(images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        image_features = encode_clip_images(model, inputs)

    embedding = image_features.detach().cpu().numpy().astype(np.float32)
    query_vector = l2_normalize_matrix(embedding)[0]

    return search_by_vector(
        query_vector=query_vector.tolist(),
        top_k=top_k,
        exclude_image_url=image_url,
    )
