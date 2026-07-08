from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (previously in app_config.py which has been removed)
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

# Backend root is two levels above this file (backend/src/cbir/files.py → backend/)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def canonicalize_image_path(image_path: str | Path) -> str:
    """Return a stable, portable path string for indexing and self-match checks.

    Normalizes path separators to forward slashes so paths are consistent
    across Windows and Linux environments.
    """
    return Path(str(image_path).replace("\\", "/")).as_posix()


def list_image_files(image_dir: str | Path) -> list[Path]:
    """List supported image files recursively from a dataset directory."""
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
