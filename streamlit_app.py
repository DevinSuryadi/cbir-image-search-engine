from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.cbir.bovw import search_bovw_index_file
from src.cbir.deep_embedding import search_deep_embedding_index_file
from src.cbir.fusion import DEFAULT_FUSION_SOURCES, search_fusion
from src.cbir.indexing import load_index
from src.cbir.reranking import rerank_results_with_orb
from src.cbir.search import search_from_index_file, search_index
from src.cbir.visualization import read_image_rgb


CONFIG_PATH = Path("config/search_config.json")
DEFAULT_SEARCH_CONFIG = {
    "method": "fusion",
    "top_k": 10,
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


def show_query_image(query_path: str) -> None:
    """Display the uploaded query image."""
    st.subheader("Query Image")
    st.image(read_image_rgb(query_path), width=360)


def show_search_results(results) -> None:
    """Display search results below the query image."""
    st.subheader("Search Results")

    if not results:
        st.info("No results found.")
        return

    columns = st.columns(5)
    for position, result in enumerate(results, start=1):
        with columns[(position - 1) % len(columns)]:
            st.image(read_image_rgb(result.image_path), use_container_width=True)

            if result.source_count is not None and result.rerank_score is not None:
                st.caption(
                    f"{position}. fusion score={result.rerank_score:.4f} | "
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
        return search_deep_embedding_index_file(
            query_image_path=query_path,
            index_path=search_config["deep_index_path"],
            top_k=top_k,
            device=search_config["deep_device"],
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


def show_search_config(search_config: dict) -> None:
    """Show current search configuration without making it user-controlled."""
    with st.expander("Search configuration", expanded=False):
        st.write(f"Method: `{search_config['method']}`")
        st.write(f"Top-k: `{search_config['top_k']}`")
        if search_config["method"] == "fusion":
            st.write(f"RRF k: `{search_config['rrf_k']}`")
            st.write("Fusion sources:")
            for source in DEFAULT_FUSION_SOURCES:
                st.write(f"- `{source.name}`: `{source.index_path}`")
        elif search_config["method"] == "deep":
            st.write(f"Deep index: `{search_config['deep_index_path']}`")
            st.write(f"Deep device: `{search_config['deep_device']}`")
        else:
            st.write(f"Index: `{search_config['index_path']}`")


def main() -> None:
    st.set_page_config(page_title="CBIR Image Search Engine", layout="wide")
    search_config = load_search_config()

    st.title("CBIR Image Search Engine")
    st.caption("Automatic rank-fusion image search")
    st.markdown(
        "Upload gambar query, lalu pilih apakah pencarian memakai metode klasik "
        "atau deep learning embedding."
    )

    search_mode = st.radio(
        "Search method",
        options=("Classic Fusion", "Deep Learning CLIP"),
        horizontal=True,
    )
    if search_mode == "Deep Learning CLIP":
        search_config["method"] = "deep"
    else:
        search_config["method"] = "fusion"

    show_search_config(search_config)

    missing_indexes = validate_required_indexes(search_config)
    if missing_indexes:
        st.warning("Required index files were not found.")
        st.write(missing_indexes)
        st.code(
            "python manage.py rebuild --image-dir data/images --models-dir models --top-k 10 --verbose\n"
            "python manage.py bovw-build --image-dir data/images --index-path models/index-bovw.pkl --feature sift --vocabulary-size 256 --verbose\n"
            "python manage.py deep-build --image-dir data/images --index-path models/index-clip.pkl --verbose",
            language="bash",
        )
        return

    uploaded_file = st.file_uploader(
        "Upload query image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
    )
    if uploaded_file is None:
        st.info("Upload an image to start searching.")
        return

    query_path = save_uploaded_query(uploaded_file)

    try:
        response = run_search(query_path=query_path, search_config=search_config)
    except Exception as error:
        st.error(str(error))
        return

    st.metric("Query Time", f"{response.query_seconds * 1000:.2f} ms")
    show_query_image(query_path)
    show_search_results(response.results)


if __name__ == "__main__":
    main()
