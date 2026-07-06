from __future__ import annotations

from pathlib import Path

from .app_config import SUPPORTED_IMAGE_EXTENSIONS


def list_image_files(image_dir: str | Path) -> list[Path]:
    """List supported image files from a dataset directory."""
    directory = Path(image_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory was not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
