from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "search_config.json"
DATASET_PATH = PROJECT_ROOT / "data" / "images"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_IMAGE_DIR = "data/images"
DEFAULT_MODELS_DIR = "models"
DEFAULT_OUTPUTS_DIR = "outputs"
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_STREAMLIT_TOP_K = 15
DEFAULT_DEEP_INDEX_PATH = f"{DEFAULT_MODELS_DIR}/index-clip.pkl"
DEFAULT_DEEP_DEVICE = "auto"

DEFAULT_SEARCH_CONFIG = {
    "method": "fusion",
    "top_k": DEFAULT_STREAMLIT_TOP_K,
    "rrf_k": DEFAULT_RRF_K,
    "deep_index_path": DEFAULT_DEEP_INDEX_PATH,
    "deep_device": DEFAULT_DEEP_DEVICE,
}

DEFAULT_CLASSIC_INDEX_PATH = f"{DEFAULT_MODELS_DIR}/index.pkl"
DEFAULT_HSV_INDEX_PATH = f"{DEFAULT_MODELS_DIR}/index-hsv.pkl"
DEFAULT_HOG_INDEX_PATH = f"{DEFAULT_MODELS_DIR}/index-hog.pkl"
DEFAULT_HSV_HOG_INDEX_PATH = f"{DEFAULT_MODELS_DIR}/index-hsv-hog.pkl"
DEFAULT_BOVW_INDEX_PATH = f"{DEFAULT_MODELS_DIR}/index-bovw.pkl"
DEFAULT_CLIP_INDEX_PATH = f"{DEFAULT_MODELS_DIR}/index-clip.pkl"

DEFAULT_VOCABULARY_SIZE = 256
DEFAULT_MAX_DESCRIPTORS = 50000
DEFAULT_BATCH_SIZE = 16


def load_search_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """Load dashboard search settings and merge them with safe defaults."""
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_SEARCH_CONFIG.copy()

    with path.open("r", encoding="utf-8") as config_file:
        loaded_config = json.load(config_file)

    return {**DEFAULT_SEARCH_CONFIG, **loaded_config}
