from __future__ import annotations

from pathlib import Path

from .app_config import DATASET_PATH, PROJECT_ROOT, SUPPORTED_IMAGE_EXTENSIONS


def resolve_image_path(image_path: str | Path) -> Path:
    """Resolve image paths saved on Windows, Linux, or as project-relative paths."""
    raw_path = str(image_path)
    normalized_path = raw_path.replace("\\", "/")
    candidates = [
        Path(raw_path),
        Path(normalized_path),
    ]

    if not Path(normalized_path).is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)
        candidates.append(PROJECT_ROOT / normalized_path)

    dataset_marker = "data/images/"
    if dataset_marker in normalized_path:
        relative_dataset_path = normalized_path.split(dataset_marker, 1)[1]
        candidates.append(DATASET_PATH / relative_dataset_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if DATASET_PATH.exists():
        filename_matches = list(DATASET_PATH.rglob(Path(normalized_path).name))
        if filename_matches:
            return filename_matches[0].resolve()

    return Path(image_path)


def canonicalize_image_path(image_path: str | Path) -> str:
    """Return a stable path string for indexing and self-match checks."""
    resolved_path = resolve_image_path(image_path)
    if resolved_path.exists():
        try:
            return str(resolved_path.resolve().relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            return str(resolved_path.resolve().as_posix())

    return Path(str(image_path).replace("\\", "/")).as_posix()


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
