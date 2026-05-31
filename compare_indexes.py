from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import evaluate_index_file
from src.indexing import load_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple CBIR indexes with precision@k")
    parser.add_argument(
        "--index-paths",
        nargs="+",
        required=True,
        help="Index files to compare",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved results used for precision@k",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Optional limit for the number of query images evaluated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_rows = []

    for index_path in args.index_paths:
        path = Path(index_path)
        index = load_index(path)
        summary = evaluate_index_file(
            index_path=path,
            top_k=args.top_k,
            max_queries=args.max_queries,
        )
        comparison_rows.append(
            {
                "index_path": str(path),
                "descriptor": index.descriptor_name,
                "query_count": summary.query_count,
                "mean_precision": summary.mean_precision,
            }
        )

    comparison_rows.sort(key=lambda row: row["mean_precision"], reverse=True)

    print(f"Metric: precision@{args.top_k}")
    print()
    print("Rank | Descriptor | Mean Precision | Queries | Index")
    print("-----|------------|----------------|---------|------")

    for rank, row in enumerate(comparison_rows, start=1):
        print(
            f"{rank:04d} | "
            f"{row['descriptor']} | "
            f"{row['mean_precision']:.4f} | "
            f"{row['query_count']} | "
            f"{row['index_path']}"
        )

    best = comparison_rows[0]
    print()
    print(
        "Best index: "
        f"{best['index_path']} "
        f"({best['descriptor']}, precision@{args.top_k}={best['mean_precision']:.4f})"
    )


if __name__ == "__main__":
    main()
