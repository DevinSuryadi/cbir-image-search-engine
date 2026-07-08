"""
CLIP deep embedding utilities for the CBIR FastAPI backend.

This module provides functions to load the CLIP model and compute
image and text embeddings used for Qdrant vector search.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .files import canonicalize_image_path, list_image_files
from .labels import get_category_label, is_same_image


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


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


def encode_clip_images(model, inputs):
    """Encode images with CLIP vision features and return one embedding per image.

    Uses vision pooler output for stable image-to-image retrieval since
    both query and dataset are encoded by the same visual encoder.
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


def compute_clip_embeddings(
    image_paths: list[str | Path],
    model,
    processor,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute normalized CLIP image embeddings for a list of image paths."""
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


def encode_text_clip(
    text: str,
    model,
    processor,
    device: str,
) -> np.ndarray:
    """Encode a text query into a normalized CLIP embedding vector.

    Returns a 1D float32 numpy array suitable for Qdrant vector search.
    """
    import torch

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    embedding = text_features.detach().cpu().numpy().astype(np.float32)
    normalized = l2_normalize_matrix(embedding)
    return normalized[0]  # Return 1D vector
