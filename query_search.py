from __future__ import annotations

import argparse

from src.search import search_from_index_file


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = search_from_index_file(
        query_image_path=args.query,
        index_path=args.index_path,
        top_k=args.top_k,
    )

    print(f"Query time: {response.query_seconds * 1000:.2f} ms")
    print(f"Top-{len(response.results)} results:")

    for position, result in enumerate(response.results, start=1):
        print(f"{position:02d}. distance={result.distance:.6f} | {result.image_path}")


if __name__ == "__main__":
    main()
