from __future__ import annotations

import cv2
import numpy as np


DEFAULT_HSV_BINS = (8, 12, 3)
HSV_RANGES = (0, 180, 0, 256, 0, 256)
DEFAULT_HOG_SIZE = (128, 128)


def get_five_regions(image_shape: tuple[int, int, int] | tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Return top-left, top-right, bottom-right, bottom-left, and center regions."""
    height, width = image_shape[:2]
    center_x = width // 2
    center_y = height // 2
    quarter_width = width // 4
    quarter_height = height // 4

    return [
        (0, 0, center_x, center_y),
        (center_x, 0, width, center_y),
        (center_x, center_y, width, height),
        (0, center_y, center_x, height),
        (
            center_x - quarter_width,
            center_y - quarter_height,
            center_x + quarter_width,
            center_y + quarter_height,
        ),
    ]


def create_region_mask(
    image_shape: tuple[int, int, int] | tuple[int, int],
    region: tuple[int, int, int, int],
) -> np.ndarray:
    """Create a binary mask for one image region."""
    start_x, start_y, end_x, end_y = region
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 255, -1)
    return mask


def compute_hsv_histogram(
    image_hsv: np.ndarray,
    mask: np.ndarray,
    bins: tuple[int, int, int] = DEFAULT_HSV_BINS,
) -> np.ndarray:
    """Compute a normalized HSV histogram for a masked image region."""
    histogram = cv2.calcHist(
        [image_hsv],
        [0, 1, 2],
        mask,
        bins,
        HSV_RANGES,
    )
    normalized = cv2.normalize(histogram, None).flatten()
    return normalized.astype(np.float32)


def compute_region_hsv_descriptor(
    image_hsv: np.ndarray,
    bins: tuple[int, int, int] = DEFAULT_HSV_BINS,
) -> np.ndarray:
    """Compute one descriptor by combining HSV histograms from five image regions."""
    if image_hsv.ndim != 3 or image_hsv.shape[2] != 3:
        raise ValueError("HSV image must have 3 color channels")

    descriptors = []
    for region in get_five_regions(image_hsv.shape):
        mask = create_region_mask(image_hsv.shape, region)
        histogram = compute_hsv_histogram(image_hsv, mask, bins)
        descriptors.append(histogram)

    return np.concatenate(descriptors).astype(np.float32)


def compute_hog_descriptor(
    image_bgr: np.ndarray,
    image_size: tuple[int, int] = DEFAULT_HOG_SIZE,
) -> np.ndarray:
    """Compute a HOG descriptor for image shape and texture information."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("BGR image must have 3 color channels")

    resized = cv2.resize(image_bgr, image_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor(
        _winSize=image_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    descriptor = hog.compute(gray)
    return descriptor.flatten().astype(np.float32)
