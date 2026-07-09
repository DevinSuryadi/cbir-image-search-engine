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



# Must match the model used when the Qdrant collection was indexed.
# The collection 'caltech101_clip' uses 768-dim vectors → clip-vit-large-patch14.
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"




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
    """Encode images with CLIP and return projected embeddings (one per image).

    Uses model.get_image_features() which applies CLIP's projection head,
    producing embeddings in the same 768-dim space as text embeddings and
    as the vectors stored in the Qdrant collection.

    Note: model.vision_model().pooler_output returns the raw ViT hidden-state
    (1024-dim for ViT-L) and must NOT be used here — it would mismatch Qdrant.
    """
    if "pixel_values" not in inputs:
        raise ValueError("CLIP image inputs must contain pixel_values")

    pixel_values = inputs["pixel_values"]

    image_features = model.get_image_features(pixel_values=pixel_values)

    # Handle both plain tensor and model output wrapper objects
    if not hasattr(image_features, "detach"):
        if hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
            image_features = image_features.image_embeds
        elif hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
            image_features = image_features.pooler_output
        else:
            raise TypeError(
                f"Unsupported CLIP image feature output type: {type(image_features)}. "
                "Expected a tensor or object with image_embeds/pooler_output attribute."
            )

    return image_features

    # NOTE: Do not use model.vision_model(pixel_values).pooler_output here.
    # That gives the raw ViT hidden size (e.g. 1024 for ViT-L), not the projected
    # 768-dim CLIP embedding space. get_image_features() is the correct call.



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

    Handles both plain tensor outputs and BaseModelOutputWithPooling objects
    returned by newer versions of the transformers library.
    """
    import torch

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # Newer transformers versions may return a BaseModelOutputWithPooling object
    # instead of a plain tensor — unwrap it the same way encode_clip_images does.
    if not hasattr(text_features, "detach"):
        if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
            text_features = text_features.pooler_output
        elif hasattr(text_features, "last_hidden_state"):
            # Fall back to the CLS token (first position) of the last hidden state
            text_features = text_features.last_hidden_state[:, 0, :]
        else:
            raise TypeError(
                f"Unsupported text feature output type: {type(text_features)}. "
                "Expected a tensor or BaseModelOutputWithPooling."
            )

    embedding = text_features.detach().cpu().numpy().astype(np.float32)
    normalized = l2_normalize_matrix(embedding)
    return normalized[0]  # Return 1D vector
