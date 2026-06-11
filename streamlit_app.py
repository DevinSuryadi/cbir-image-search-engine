from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.cbir.bovw import search_bovw_index_file
from src.cbir.deep_embedding import (
    load_clip_model,
    load_deep_embedding_index,
    search_deep_embedding_index_with_model,
)
from src.cbir.fusion import DEFAULT_FUSION_SOURCES, search_fusion
from src.cbir.indexing import load_index
from src.cbir.reranking import rerank_results_with_orb
from src.cbir.search import search_from_index_file, search_index
from src.cbir.visualization import read_image_rgb


CONFIG_PATH = Path("config/search_config.json")
DATASET_PATH = Path("data/images")
DEFAULT_SEARCH_CONFIG = {
    "method": "fusion",
    "top_k": 15,
    "rrf_k": 60,
    "deep_index_path": "models/index-clip.pkl",
    "deep_device": "auto",
    "index_path": "models/index-best.pkl",
    "use_orb_rerank": True,
    "candidate_k": 50,
    "rerank_strategy": "weighted",
    "distance_weight": 0.4,
    "orb_weight": 0.6,
    "verify_top_k": 50,
}


METHOD_OPTIONS = {
    "Classic Fusion": "fusion",
    "Deep Learning CLIP": "deep",
}


METHOD_DESCRIPTIONS = {
    "fusion": (
        "Classic Fusion menggabungkan beberapa descriptor visual klasik dan sinyal "
        "ranking menjadi satu urutan hasil akhir. Metode ini mencari kemiripan "
        "berdasarkan komposisi warna, struktur bentuk, pola visual lokal, dan "
        "konsistensi keypoint."
    ),
    "deep": (
        "Deep Learning CLIP menggunakan pretrained image encoder untuk mengubah "
        "setiap gambar menjadi embedding semantik. Pencarian dilakukan dengan "
        "membandingkan embedding gambar query terhadap embedding dataset yang "
        "sudah di-index menggunakan cosine distance."
    ),
    "bovw": (
        "BoVW represents an image as a histogram of visual words built from local "
        "features. It is focused on matching repeated local patterns and details."
    ),
    "classic": (
        "Classic search uses one selected descriptor index and ranks dataset images "
        "by descriptor distance."
    ),
}


METHOD_WORKFLOWS = {
    "fusion": [
        ("Ekstraksi fitur", "Sistem mengekstraksi beberapa fitur visual klasik dari gambar query."),
        ("Pencarian independen", "Setiap metode fitur menghasilkan daftar kandidat berdasarkan index masing-masing."),
        ("Penggabungan ranking", "Ranking kandidat digabungkan menggunakan Reciprocal Rank Fusion untuk menghasilkan satu urutan akhir."),
        ("Penampilan hasil", "Gambar dengan sinyal ranking gabungan terkuat ditampilkan sebagai hasil teratas."),
    ],
    "deep": [
        ("Ekstraksi embedding", "Gambar query diubah menjadi embedding menggunakan pretrained CLIP image encoder."),
        ("Perbandingan vector", "Embedding query dibandingkan dengan embedding dataset yang sudah tersimpan di index."),
        ("Pengurutan hasil", "Gambar dengan cosine distance paling kecil ditempatkan pada ranking lebih tinggi."),
        ("Penampilan hasil", "Gambar dengan kemiripan semantik tertinggi ditampilkan sebagai hasil teratas."),
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


def get_supported_categories(dataset_path: Path = DATASET_PATH) -> list[str]:
    """Return dataset category names from immediate child folders."""
    if not dataset_path.exists():
        return []

    return sorted(
        path.name.replace("-", " ")
        for path in dataset_path.iterdir()
        if path.is_dir()
    )


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
    method = search_config["method"]

    if method == "deep":
        index_path = search_config["deep_index_path"]
        return [] if Path(index_path).exists() else [index_path]

    if method == "fusion":
        return [
            source.index_path
            for source in DEFAULT_FUSION_SOURCES
            if not Path(source.index_path).exists()
        ]

    return [] if Path(search_config["index_path"]).exists() else [search_config["index_path"]]


def run_search(query_path: str, search_config: dict):
    """Run search with stored parameters."""
    method = search_config["method"]
    top_k = int(search_config["top_k"])

    if method == "fusion":
        return search_fusion(
            query_image_path=query_path,
            sources=DEFAULT_FUSION_SOURCES,
            top_k=top_k,
            rrf_k=int(search_config["rrf_k"]),
        )

    if method == "deep":
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

    if method == "bovw":
        return search_bovw_index_file(
            query_image_path=query_path,
            index_path=search_config["index_path"],
            top_k=top_k,
            verify_top_k=int(search_config["verify_top_k"]),
        )

    if method != "classic":
        raise ValueError(f"Unsupported search method: {method}")

    if not bool(search_config["use_orb_rerank"]):
        return search_from_index_file(
            query_image_path=query_path,
            index_path=search_config["index_path"],
            top_k=top_k,
        )

    index = load_index(search_config["index_path"])
    response = search_index(
        query_image_path=query_path,
        index=index,
        top_k=max(int(search_config["candidate_k"]), top_k),
    )
    rerank_response = rerank_results_with_orb(
        query_image_path=query_path,
        candidate_results=response.results,
        top_k=top_k,
        strategy=search_config["rerank_strategy"],
        distance_weight=float(search_config["distance_weight"]),
        orb_weight=float(search_config["orb_weight"]),
    )
    response.results = rerank_response.results
    response.query_seconds += rerank_response.rerank_seconds
    return response


def show_configuration(search_config: dict) -> None:
    """Render compact search configuration."""
    st.markdown("**Search Configuration**")
    st.caption(f"Method: `{search_config['method']}`")
    st.caption(f"Top-k: `{search_config['top_k']}`")

    if search_config["method"] == "fusion":
        st.caption(f"RRF k: `{search_config['rrf_k']}`")
    elif search_config["method"] == "deep":
        st.caption(f"Index: `{search_config['deep_index_path']}`")
        st.caption(f"Device: `{search_config['deep_device']}`")
    else:
        st.caption(f"Index: `{search_config['index_path']}`")


def show_method_description(method: str) -> None:
    """Render explanation for the selected method."""
    st.info(METHOD_DESCRIPTIONS.get(method, "Metode pencarian tidak dikenali."))


def show_method_workflow(method: str) -> None:
    """Render a concise workflow explanation for the selected method."""
    workflow = METHOD_WORKFLOWS.get(method, [])
    if not workflow:
        return

    with st.expander("Cara Kerja Metode", expanded=False):
        for title, description in workflow:
            st.markdown(f"**{title}.** {description}")


def show_missing_index_message(missing_indexes: list[str]) -> None:
    """Render missing index instructions."""
    st.warning("Required index files were not found.")
    st.write(missing_indexes)
    st.code(
        "python manage.py prepare --image-dir data/images --models-dir models --top-k 10 --verbose\n"
        "python manage.py deep-build --image-dir data/images --index-path models/index-clip.pkl --verbose",
        language="bash",
    )


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


def show_search_results(results) -> None:
    """Display search results below the query image."""
    st.markdown("### Search Results")

    if not results:
        st.info("No results found.")
        return

    columns = st.columns(5)
    for position, result in enumerate(results, start=1):
        with columns[(position - 1) % len(columns)]:
            st.image(read_image_rgb(result.image_path), use_container_width=True)

            if result.source_count is not None and result.rerank_score is not None:
                st.caption(
                    f"{position}. fusion={result.rerank_score:.4f} | "
                    f"sources={result.source_count}"
                )
            elif result.orb_matches is not None:
                st.caption(
                    f"{position}. matches={result.orb_matches} | "
                    f"distance={result.initial_distance:.4f}"
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
        st.image(read_image_rgb(query_path), caption="Uploaded query", width=360)

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
