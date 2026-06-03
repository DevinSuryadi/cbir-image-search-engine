from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.cbir.indexing import load_index
from src.cbir.reranking import rerank_results_with_orb
from src.cbir.search import search_from_index_file, search_index
from src.cbir.visualization import read_image_rgb


CONFIG_PATH = Path("config/search_config.json")
DEFAULT_SEARCH_CONFIG = {
    "index_path": "models/index-best.pkl",
    "top_k": 10,
    "use_orb_rerank": True,
    "candidate_k": 50,
    "rerank_strategy": "weighted",
    "distance_weight": 0.4,
    "orb_weight": 0.6,
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

            if result.orb_matches is None:
                st.caption(f"{position}. distance={result.distance:.4f}")
            else:
                st.caption(
                    f"{position}. ORB matches={result.orb_matches} | "
                    f"distance={result.initial_distance:.4f}"
                )
                if result.rerank_score is not None:
                    st.caption(f"rerank score={result.rerank_score:.4f}")

            st.caption(result.image_path)


def run_search(query_path: str, search_config: dict):
    """Run search with stored best parameters."""
    index_path = search_config["index_path"]
    top_k = int(search_config["top_k"])
    use_orb_rerank = bool(search_config["use_orb_rerank"])

    if not use_orb_rerank:
        return search_from_index_file(
            query_image_path=query_path,
            index_path=index_path,
            top_k=top_k,
        )

    index = load_index(index_path)
    candidate_k = max(int(search_config["candidate_k"]), top_k)
    response = search_index(
        query_image_path=query_path,
        index=index,
        top_k=candidate_k,
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
        st.write(f"Index: `{search_config['index_path']}`")
        st.write(f"Top-k: `{search_config['top_k']}`")
        st.write(f"ORB reranking: `{search_config['use_orb_rerank']}`")
        st.write(f"Candidate-k: `{search_config['candidate_k']}`")
        st.write(f"Rerank strategy: `{search_config['rerank_strategy']}`")
        st.write(f"Distance weight: `{search_config['distance_weight']}`")
        st.write(f"ORB weight: `{search_config['orb_weight']}`")


def main() -> None:
    st.set_page_config(page_title="CBIR Image Search Engine", layout="wide")
    search_config = load_search_config()
    index_path = search_config["index_path"]

    st.title("CBIR Image Search Engine")
    st.caption("Automatic best-parameter search configuration")
    st.markdown(
        "Upload gambar query, lalu sistem otomatis memakai konfigurasi pencarian "
        "yang tersimpan untuk dataset ini."
    )
    show_search_config(search_config)

    uploaded_file = st.file_uploader(
        "Upload query image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
    )

    if not Path(index_path).exists():
        st.warning("Index file was not found. Build the index before searching.")
        st.code(
            "python manage.py rebuild --image-dir data/images --models-dir models --top-k 10 --verbose",
            language="bash",
        )
        return

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
