from __future__ import annotations

from pathlib import Path


def get_category_label(image_path: str | Path) -> str:
    """Use the parent folder name as the image category label."""
    return Path(image_path).parent.name


def is_same_image(first_path: str | Path, second_path: str | Path) -> bool:
    """Compare two image paths after resolving them."""
    return Path(first_path).resolve() == Path(second_path).resolve()
