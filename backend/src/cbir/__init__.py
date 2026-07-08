"""CBIR package — CLIP embedding utilities for the FastAPI backend."""

from .deep_embedding import (
    DEFAULT_CLIP_MODEL,
    compute_clip_embeddings,
    encode_clip_images,
    encode_text_clip,
    get_device,
    l2_normalize_matrix,
    load_clip_model,
    read_pil_rgb,
)
from .files import canonicalize_image_path, list_image_files
from .labels import get_category_label, is_same_image

__all__ = [
    "DEFAULT_CLIP_MODEL",
    "compute_clip_embeddings",
    "encode_clip_images",
    "encode_text_clip",
    "get_device",
    "l2_normalize_matrix",
    "load_clip_model",
    "read_pil_rgb",
    "canonicalize_image_path",
    "list_image_files",
    "get_category_label",
    "is_same_image",
]
