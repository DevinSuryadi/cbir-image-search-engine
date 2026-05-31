from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.indexing import load_index
from src.reranking import rerank_results_with_orb
from src.search import search_index
from src.search import search_from_index_file
from src.visualization import read_image_rgb


INDEX_OPTIONS = {
    "HOG (best precision@10)": "models/index-hog.pkl",
    "HSV + HOG": "models/index-hsv-hog.pkl",
    "HSV": "models/index-hsv.pkl",
    "Default index": "models/index.pkl",
}
DEFAULT_TOP_K = 10


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
                    f"HOG distance={result.initial_distance:.4f}"
                )
            st.caption(result.image_path)


def main() -> None:
    st.set_page_config(page_title="CBIR Image Search Engine", layout="wide")

    st.title("CBIR Image Search Engine")
    st.caption("Region-based HSV histogram with cosine distance")

    st.markdown(
        "Atur **Top-k results** untuk menentukan berapa banyak gambar paling mirip "
        "yang ditampilkan. Nilai yang lebih besar membantu membandingkan lebih banyak "
        "hasil, sedangkan nilai yang lebih kecil membuat hasil lebih fokus."
    )

    top_k = st.slider("Top-k results", min_value=1, max_value=20, value=DEFAULT_TOP_K)

    selected_index_label = st.selectbox(
        "Search descriptor",
        options=list(INDEX_OPTIONS.keys()),
        help="Pilih index/descriptor yang digunakan untuk pencarian. HOG menjadi default karena memiliki precision@10 terbaik pada eksperimen dataset ini.",
    )
    index_path = INDEX_OPTIONS[selected_index_label]

    use_orb_rerank = st.checkbox(
        "Use ORB reranking",
        value=False,
        help="Ambil kandidat awal dari descriptor terpilih, lalu urutkan ulang dengan ORB keypoint matching.",
    )
    candidate_k = st.slider(
        "Initial candidates for reranking",
        min_value=top_k,
        max_value=50,
        value=max(30, top_k),
        disabled=not use_orb_rerank,
    )

    uploaded_file = st.file_uploader(
        "Upload query image",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
    )

    if not Path(index_path).exists():
        st.warning("Index file was not found. Build the index before searching.")
        st.code(
            f"python build_index.py --image-dir data/images --index-path {index_path} --verbose",
            language="bash",
        )
        return

    if uploaded_file is None:
        st.info("Upload an image to start searching.")
        return

    query_path = save_uploaded_query(uploaded_file)

    try:
        if use_orb_rerank:
            index = load_index(index_path)
            response = search_index(
                query_image_path=query_path,
                index=index,
                top_k=candidate_k,
            )
            rerank_response = rerank_results_with_orb(
                query_image_path=query_path,
                candidate_results=response.results,
                top_k=top_k,
            )
            response.results = rerank_response.results
            response.query_seconds += rerank_response.rerank_seconds
        else:
            response = search_from_index_file(
                query_image_path=query_path,
                index_path=index_path,
                top_k=top_k,
            )
    except Exception as error:
        st.error(str(error))
        return

    st.metric("Query Time", f"{response.query_seconds * 1000:.2f} ms")
    show_query_image(query_path)
    show_search_results(response.results)


if __name__ == "__main__":
    main()
