"""
CBIR Image Search Engine — FastAPI Backend

Endpoints:
  GET  /                              Health check
  POST /api/search/image              Search by uploaded query image
  POST /api/search/text               Search by text description
  POST /api/search/more-like-this     Find images similar to a given Supabase image URL
"""
from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.cbir.deep_embedding import encode_clip_images, encode_text_clip, l2_normalize_matrix, load_clip_model
from src.qdrant_search import search_by_vector, search_by_text as _qdrant_search_by_text, more_like_this

# ---------------------------------------------------------------------------
# App state — holds the CLIP model loaded once at startup
# ---------------------------------------------------------------------------

clip_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CLIP model on startup and release on shutdown."""
    print("Loading CLIP model...")
    model, processor, device = load_clip_model()
    clip_state["model"] = model
    clip_state["processor"] = processor
    clip_state["device"] = device
    print(f"CLIP model loaded on device: {device}")
    yield
    clip_state.clear()
    print("CLIP model released.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CBIR Image Search Engine API",
    description="Content-Based Image Retrieval using CLIP embeddings and Qdrant Cloud",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SearchResultItem(BaseModel):
    point_id: str
    image_url: str
    category: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query_seconds: float
    result_count: int


class MoreLikeThisRequest(BaseModel):
    image_url: str
    top_k: int = 20


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_clip_state() -> tuple:
    """Return the loaded CLIP model, processor, and device, or raise 503."""
    if not clip_state:
        raise HTTPException(status_code=503, detail="CLIP model is not loaded yet.")
    return clip_state["model"], clip_state["processor"], clip_state["device"]


def _encode_image_to_vector(image: Image.Image, model, processor, device: str) -> list[float]:
    """Encode a PIL RGB image into a normalized CLIP embedding vector (list of floats)."""
    inputs = processor(images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        image_features = encode_clip_images(model, inputs)

    embedding = image_features.detach().cpu().numpy().astype(np.float32)
    return l2_normalize_matrix(embedding)[0].tolist()


def _build_response(raw_results: list[dict], elapsed: float) -> SearchResponse:
    """Convert raw Qdrant result dicts into the API response model."""
    items = [SearchResultItem(**r) for r in raw_results]
    return SearchResponse(
        results=items,
        query_seconds=round(elapsed, 4),
        result_count=len(items),
    )


def _validate_top_k(top_k: int) -> None:
    """Raise 422 if top_k is outside the accepted range."""
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 100.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """Health check — returns API status."""
    return {"status": "ok", "message": "CBIR API is running"}


@app.post("/api/search/image", response_model=SearchResponse, tags=["Search"])
async def search_by_image(
    file: Annotated[UploadFile, File(description="Query image file (JPEG/PNG)")],
    top_k: Annotated[int, Form(description="Number of results to return")] = 20,
):
    """Search for visually similar images by uploading a query image.

    Accepts a multipart/form-data request with an image file and optional top_k.
    The image is encoded with CLIP and the resulting vector is queried against
    the Qdrant collection.
    """
    _validate_top_k(top_k)
    model, processor, device = _get_clip_state()

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    start = time.perf_counter()
    query_vector = _encode_image_to_vector(image, model, processor, device)
    results = search_by_vector(query_vector=query_vector, top_k=top_k)
    elapsed = time.perf_counter() - start

    return _build_response(results, elapsed)


@app.post("/api/search/text", response_model=SearchResponse, tags=["Search"])
def search_by_text(body: TextSearchRequest):
    """Search for images using a natural language text description.

    Uses CLIP's text encoder to convert the query into an embedding vector,
    then searches the Qdrant collection for semantically similar images.
    """
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query text must not be empty.")
    _validate_top_k(body.top_k)
    model, processor, device = _get_clip_state()

    start = time.perf_counter()
    results = _qdrant_search_by_text(
        query_text=body.query,
        model=model,
        processor=processor,
        device=device,
        top_k=body.top_k,
    )
    elapsed = time.perf_counter() - start

    return _build_response(results, elapsed)


@app.post("/api/search/more-like-this", response_model=SearchResponse, tags=["Search"])
def search_more_like_this(body: MoreLikeThisRequest):
    """Find images visually similar to a given image URL.

    Downloads the image from a public Supabase Storage URL, encodes it with
    CLIP, and returns visually similar images — excluding the reference image
    itself from the results.
    """
    if not body.image_url.strip():
        raise HTTPException(status_code=422, detail="image_url must not be empty.")
    _validate_top_k(body.top_k)
    model, processor, device = _get_clip_state()

    start = time.perf_counter()

    import httpx

    try:
        results = more_like_this(
            image_url=body.image_url,
            model=model,
            processor=processor,
            device=device,
            top_k=body.top_k,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download image from the provided URL: {exc}",
        )

    elapsed = time.perf_counter() - start
    return _build_response(results, elapsed)
