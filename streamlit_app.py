from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image
import streamlit as st

from src.cbir.deep_embedding import (
    load_clip_model,
    load_deep_embedding_index,
    search_deep_embedding_index_with_model,
)


CONFIG_PATH = Path("config/search_config.json")
DATASET_PATH = Path("data/images")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_SEARCH_CONFIG = {
    "method": "fusion",
    "top_k": 15,
    "rrf_k": 60,
    "deep_index_path": "models/index-clip.pkl",
    "deep_device": "auto",
}

METHOD_OPTIONS = {
    "Deep Learning CLIP": "deep",
    "Classic Fusion": "fusion",
}

METHOD_DESCRIPTIONS = {
    "deep": (
        "Deep Learning CLIP menggunakan pretrained image encoder untuk mengubah "
        "setiap gambar menjadi embedding semantik. Pencarian dilakukan dengan "
        "membandingkan embedding gambar query terhadap embedding dataset yang "
        "sudah di-index menggunakan cosine distance."
    ),
    "fusion": (
        "Classic Fusion menggabungkan beberapa descriptor visual klasik dan sinyal "
        "ranking menjadi satu urutan hasil akhir. Metode ini mencari kemiripan "
        "berdasarkan komposisi warna, struktur bentuk, pola visual lokal, dan "
        "konsistensi keypoint."
    ),
}

METHOD_WORKFLOWS = {
    "deep": [
        ("Ekstraksi embedding", "Gambar query diubah menjadi embedding menggunakan pretrained CLIP image encoder."),
        ("Perbandingan vector", "Embedding query dibandingkan dengan embedding dataset yang sudah tersimpan di index."),
        ("Pengurutan hasil", "Gambar dengan cosine distance paling kecil ditempatkan pada ranking lebih tinggi."),
        ("Penampilan hasil", "Gambar dengan kemiripan semantik tertinggi ditampilkan sebagai hasil teratas."),
    ],
    "fusion": [
        ("Ekstraksi fitur", "Sistem mengekstraksi beberapa fitur visual klasik dari gambar query."),
        ("Pencarian independen", "Setiap metode fitur menghasilkan daftar kandidat berdasarkan index masing-masing."),
        ("Penggabungan ranking", "Ranking kandidat digabungkan menggunakan Reciprocal Rank Fusion untuk menghasilkan satu urutan akhir."),
        ("Penampilan hasil", "Gambar dengan sinyal ranking gabungan terkuat ditampilkan sebagai hasil teratas."),
    ],
}


def load_search_config() -> dict:
    """Load search parameters used by the dashboard."""
    if not CONFIG_PATH.exists():
        return DEFAULT_SEARCH_CONFIG

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        loaded_config = json.load(config_file)

    return {**DEFAULT_SEARCH_CONFIG, **loaded_config}


def save_uploaded_query(uploaded_file) -> str:
    """Save an uploaded query image to a temporary file and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        return temporary_file.name


def read_image_for_display(image_path: str | Path) -> Image.Image:
    """Read an image for Streamlit display without importing OpenCV."""
    return Image.open(image_path).convert("RGB")


def get_supported_categories() -> list[str]:
    """Return dataset category names from immediate child folders."""
    if not DATASET_PATH.exists():
        return []

    return sorted(path.name.replace("-", " ") for path in DATASET_PATH.iterdir() if path.is_dir())


def get_sample_query_file() -> Path | None:
    """Return one dataset image that can be used as a query example."""
    if not DATASET_PATH.exists():
        return None

    for category_path in sorted(path for path in DATASET_PATH.iterdir() if path.is_dir()):
        for image_path in sorted(category_path.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                return image_path

    return None


@st.cache_resource(show_spinner="Loading deep embedding index...")
def get_cached_deep_index(index_path: str):
    """Load the deep embedding index once per Streamlit session."""
    return load_deep_embedding_index(index_path)


@st.cache_resource(show_spinner="Loading CLIP model...")
def get_cached_clip_model(model_name: str, device: str):
    """Load the CLIP model once and reuse it for subsequent searches."""
    return load_clip_model(model_name=model_name, device=device)


def validate_required_indexes(search_config: dict) -> list[str]:
    """Return missing index paths required by the selected method."""
    if search_config["method"] == "deep":
        index_path = search_config["deep_index_path"]
        return [] if Path(index_path).exists() else [index_path]

    from src.cbir.fusion import DEFAULT_FUSION_SOURCES

    return [
        source.index_path
        for source in DEFAULT_FUSION_SOURCES
        if not Path(source.index_path).exists()
    ]


def run_search(query_path: str, search_config: dict):
    """Run search with the selected method."""
    top_k = int(search_config["top_k"])

    if search_config["method"] == "deep":
        index = get_cached_deep_index(search_config["deep_index_path"])
        model, processor, resolved_device = get_cached_clip_model(
            model_name=index.model_name,
            device=search_config["deep_device"],
        )
        return search_deep_embedding_index_with_model(
            query_image_path=query_path,
            index=index,
            model=model,
            processor=processor,
            device=resolved_device,
            top_k=top_k,
        )

    from src.cbir.fusion import DEFAULT_FUSION_SOURCES, search_fusion

    return search_fusion(
        query_image_path=query_path,
        sources=DEFAULT_FUSION_SOURCES,
        top_k=top_k,
        rrf_k=int(search_config["rrf_k"]),
    )


def show_sample_dataset_download() -> None:
    """Render a download button for one sample query image from the repository dataset."""
    sample_file = get_sample_query_file()
    if sample_file is None:
        return

    suffix = sample_file.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
    st.download_button(
        label="Download Sample Query Image",
        data=sample_file.read_bytes(),
        file_name=f"sample-query{suffix}",
        mime=mime,
        help="Download one example image from the repository dataset for query testing.",
    )
    st.caption("The sample image is selected from the dataset available in this app.")


def show_supported_categories() -> None:
    """Render supported image categories based on dataset folders."""
    categories = get_supported_categories()
    if not categories:
        return

    with st.expander("Kategori Gambar yang Didukung", expanded=False):
        st.markdown(
            "Sistem melakukan pencarian terhadap dataset yang tersedia. "
            "Kategori berikut terdeteksi dari folder dataset:"
        )
        st.markdown(", ".join(f"`{category}`" for category in categories))


def show_method_description(method: str) -> None:
    """Render explanation for the selected method."""
    st.info(METHOD_DESCRIPTIONS[method])


def show_method_workflow(method: str) -> None:
    """Render a concise workflow explanation for the selected method."""
    with st.expander("Cara Kerja Metode", expanded=False):
        for title, description in METHOD_WORKFLOWS[method]:
            st.markdown(f"**{title}.** {description}")


def show_configuration(search_config: dict) -> None:
    """Render compact search configuration."""
    st.markdown("**Search Configuration**")
    st.caption(f"Method: `{search_config['method']}`")
    st.caption(f"Top-k: `{search_config['top_k']}`")

    if search_config["method"] == "deep":
        st.caption(f"Index: `{search_config['deep_index_path']}`")
        st.caption(f"Device: `{search_config['deep_device']}`")
    else:
        st.caption(f"RRF k: `{search_config['rrf_k']}`")


def show_missing_index_message(missing_indexes: list[str]) -> None:
    """Render missing index instructions."""
    st.warning("Required index files were not found.")
    st.write(missing_indexes)
    st.code(
        "python manage.py deep-build --image-dir data/images --index-path models/index-clip.pkl --verbose\n"
        "python manage.py prepare --image-dir data/images --models-dir models --top-k 10 --verbose",
        language="bash",
    )


def show_search_results(results) -> None:
    """Display search results below the query image."""
    st.markdown("### Search Results")

    if not results:
        st.info("No results found.")
        return

    columns = st.columns(5)
    for position, result in enumerate(results, start=1):
        with columns[(position - 1) % len(columns)]:
            st.image(read_image_for_display(result.image_path), use_container_width=True)

            if result.source_count is not None and result.rerank_score is not None:
                st.caption(
                    f"{position}. fusion={result.rerank_score:.4f} | "
                    f"sources={result.source_count}"
                )
            else:
                st.caption(f"{position}. distance={result.distance:.4f}")

            st.caption(result.image_path)


def render_header() -> None:
    """Render page title and project explanation."""
    st.title("CBIR Image Search Engine")
    st.markdown(
        "Content-Based Image Retrieval (CBIR) adalah sistem pencarian gambar "
        "berdasarkan isi visual gambar. Project ini membandingkan pendekatan "
        "CBIR klasik dan deep learning embedding untuk mencari gambar yang "
        "paling mirip dari dataset."
    )


def main() -> None:
    st.set_page_config(page_title="CBIR Image Search Engine", layout="wide")
    search_config = load_search_config()

    render_header()
    show_sample_dataset_download()

    left_column, right_column = st.columns([2, 1], gap="large")
    with left_column:
        selected_method_label = st.radio(
            "Search Method",
            options=list(METHOD_OPTIONS.keys()),
            horizontal=True,
        )
        search_config["method"] = METHOD_OPTIONS[selected_method_label]
        show_method_description(search_config["method"])
        show_method_workflow(search_config["method"])

    with right_column:
        show_configuration(search_config)

    missing_indexes = validate_required_indexes(search_config)
    if missing_indexes:
        show_missing_index_message(missing_indexes)
        return

    st.markdown("### Query Image")
    upload_column, preview_column = st.columns([1, 1], gap="large")

    with upload_column:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=("jpg", "jpeg", "png", "bmp", "webp"),
        )
        st.caption("Gambar yang di-upload digunakan sebagai contoh query.")
        show_supported_categories()

    if uploaded_file is None:
        st.info("Upload an image to start searching.")
        return

    query_path = save_uploaded_query(uploaded_file)

    with preview_column:
        st.image(read_image_for_display(query_path), caption="Uploaded query", width=360)

    try:
        with st.spinner("Searching similar images..."):
            response = run_search(query_path=query_path, search_config=search_config)
    except Exception as error:
        st.error(str(error))
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Query Time", f"{response.query_seconds * 1000:.2f} ms")
    metric_columns[1].metric("Results", len(response.results))
    metric_columns[2].metric("Method", selected_method_label)

    show_search_results(response.results)


if __name__ == "__main__":
    main()
