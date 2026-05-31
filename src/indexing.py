from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .descriptors import DEFAULT_HSV_BINS, compute_region_hsv_descriptor
from .preprocessing import list_image_files, read_preprocess


@dataclass
class ImageIndex:
    image_paths: list[str]
    descriptors: np.ndarray
    descriptor_name: str
    bins: tuple[int, int, int]
    build_seconds: float


def validate_index(index: ImageIndex) -> None:
    """Validate an image index before saving or using it."""
    if not index.image_paths:
        raise ValueError("Index must contain at least one image path")

    if not isinstance(index.descriptors, np.ndarray):
        raise TypeError("Index descriptors must be a NumPy array")

    if index.descriptors.ndim != 2:
        raise ValueError("Index descriptors must be a 2D matrix")

    if index.descriptors.shape[0] != len(index.image_paths):
        raise ValueError("Number of descriptors must match number of image paths")


def build_image_index(
    image_dir: str | Path,
    bins: tuple[int, int, int] = DEFAULT_HSV_BINS,
    verbose: bool = False,
) -> ImageIndex:
    """Build an image index from all supported images in a directory."""
    image_files = list_image_files(image_dir)
    if not image_files:
        raise ValueError(f"No supported image files found in: {image_dir}")

    start_time = time.perf_counter()
    descriptors = []
    image_paths = []

    for position, image_file in enumerate(image_files, start=1):
        if verbose:
            print(f"[{position}/{len(image_files)}] Indexing {image_file}")

        image_hsv = read_preprocess(image_file)
        descriptor = compute_region_hsv_descriptor(image_hsv, bins=bins)

        descriptors.append(descriptor)
        image_paths.append(str(image_file))

    descriptor_matrix = np.vstack(descriptors).astype(np.float32)
    build_seconds = time.perf_counter() - start_time

    index = ImageIndex(
        image_paths=image_paths,
        descriptors=descriptor_matrix,
        descriptor_name="region_hsv_histogram",
        bins=bins,
        build_seconds=build_seconds,
    )
    validate_index(index)
    return index


def save_index(index: ImageIndex, index_path: str | Path) -> None:
    """Save an image index to disk."""
    validate_index(index)

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(index, file, pickle.HIGHEST_PROTOCOL)


def load_index(index_path: str | Path) -> ImageIndex:
    """Load an image index from disk."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file was not found: {path}")

    with path.open("rb") as file:
        index = pickle.load(file)

    if not isinstance(index, ImageIndex):
        raise TypeError("Loaded object is not a valid ImageIndex")

    validate_index(index)
    return index


def build_and_save_index(
    image_dir: str | Path,
    index_path: str | Path,
    bins: tuple[int, int, int] = DEFAULT_HSV_BINS,
    verbose: bool = False,
) -> ImageIndex:
    """Build an image index and save it to disk."""
    index = build_image_index(image_dir=image_dir, bins=bins, verbose=verbose)
    save_index(index, index_path=index_path)
    return index
