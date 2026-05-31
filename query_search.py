from __future__ import annotations

import argparse

from src.indexing import load_index
from src.reranking import rerank_results_with_orb
from src.search import search_index
from src.search import search_from_index_file
from src.visualization import save_search_results, show_search_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query CBIR image search engine")
    parser.add_argument(
        "--query",
        required=True,
        help="Path to the query image",
    )
    parser.add_argument(
        "--index-path",
        default="models/index.pkl",
        help="Path to the saved image index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of similar images to return",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show query and search results in a Matplotlib window",
    )
    parser.add_argument(
        "--save",
        help="Save query and search results grid to an image file",
    )
    parser.add_argument(
        "--orb-rerank",
        action="store_true",
        help="Rerank initial candidates using ORB keypoint matching",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
        help="Number of initial candidates used before ORB reranking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.orb_rerank:
        index = load_index(args.index_path)
        response = search_index(
            query_image_path=args.query,
            index=index,
            top_k=max(args.candidate_k, args.top_k),
        )
        rerank_response = rerank_results_with_orb(
            query_image_path=args.query,
            candidate_results=response.results,
            top_k=args.top_k,
        )
        response.results = rerank_response.results
        response.query_seconds += rerank_response.rerank_seconds
    else:
        response = search_from_index_file(
            query_image_path=args.query,
            index_path=args.index_path,
            top_k=args.top_k,
        )

    print(f"Query time: {response.query_seconds * 1000:.2f} ms")
    print(f"Top-{len(response.results)} results:")

    for position, result in enumerate(response.results, start=1):
        if result.orb_matches is None:
            print(f"{position:02d}. distance={result.distance:.6f} | {result.image_path}")
        else:
            print(
                f"{position:02d}. "
                f"orb_matches={result.orb_matches} | "
                f"initial_distance={result.initial_distance:.6f} | "
                f"{result.image_path}"
            )

    if args.save:
        save_search_results(
            query_image_path=args.query,
            results=response.results,
            output_path=args.save,
        )
        print(f"Result grid saved to: {args.save}")

    if args.show:
        show_search_results(
            query_image_path=args.query,
            results=response.results,
        )


if __name__ == "__main__":
    main()
