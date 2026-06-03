from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from .preprocessing import read_image
from .search import SearchResult


def read_image_rgb(image_path: str | Path):
    """Read an image and convert it from BGR to RGB for Matplotlib."""
    image_bgr = read_image(image_path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_search_results(
    query_image_path: str | Path,
    results: list[SearchResult],
    columns: int = 5,
) -> None:
    """Show the query image and search results in a Matplotlib grid."""
    if columns <= 0:
        raise ValueError("columns must be greater than zero")

    total_images = len(results) + 1
    rows = (total_images + columns - 1) // columns

    figure = plt.figure(figsize=(columns * 3, rows * 3))

    query_image = read_image_rgb(query_image_path)
    axis = figure.add_subplot(rows, columns, 1)
    axis.imshow(query_image)
    axis.set_title("Query")
    axis.axis("off")

    for position, result in enumerate(results, start=2):
        result_image = read_image_rgb(result.image_path)
        axis = figure.add_subplot(rows, columns, position)
        axis.imshow(result_image)
        if result.orb_matches is None:
            title = f"{position - 1}. d={result.distance:.3f}"
        else:
            title = f"{position - 1}. m={result.orb_matches}"
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def save_search_results(
    query_image_path: str | Path,
    results: list[SearchResult],
    output_path: str | Path,
    columns: int = 5,
) -> None:
    """Save the query image and search results as one grid image."""
    if columns <= 0:
        raise ValueError("columns must be greater than zero")

    total_images = len(results) + 1
    rows = (total_images + columns - 1) // columns

    figure = plt.figure(figsize=(columns * 3, rows * 3))

    query_image = read_image_rgb(query_image_path)
    axis = figure.add_subplot(rows, columns, 1)
    axis.imshow(query_image)
    axis.set_title("Query")
    axis.axis("off")

    for position, result in enumerate(results, start=2):
        result_image = read_image_rgb(result.image_path)
        axis = figure.add_subplot(rows, columns, position)
        axis.imshow(result_image)
        if result.orb_matches is None:
            title = f"{position - 1}. d={result.distance:.3f}"
        else:
            title = f"{position - 1}. m={result.orb_matches}"
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
