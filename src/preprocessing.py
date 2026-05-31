from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_supported_image_file(path: str | Path) -> bool:
    """Return True if the path has a supported image extension."""
    image_path = Path(path)
    return image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def read_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk in OpenCV BGR format."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file was not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")

    image_buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image could not be read: {path}")

    return image


def validate_color_image(image: np.ndarray) -> None:
    """Validate that an image is a non-empty BGR color image."""
    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")

    if image.size == 0:
        raise ValueError("Image is empty")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have 3 color channels")


def convert_bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to HSV color space."""
    validate_color_image(image_bgr)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def read_preprocess(image_path: str | Path) -> np.ndarray:
    """Read an image from disk and convert it to HSV color space."""
    image_bgr = read_image(image_path)
    return convert_bgr_to_hsv(image_bgr)


def list_image_files(image_dir: str | Path) -> list[Path]:
    """List supported image files from a dataset directory."""
    directory = Path(image_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Image directory was not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    image_files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(image_files)
