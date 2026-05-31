from __future__ import annotations

import argparse

from src.evaluation import evaluate_index_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CBIR search precision")
    parser.add_argument(
        "--index-path",
        default="models/index.pkl",
        help="Path to the saved image index",
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
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print per-query precision results",
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
    summary = evaluate_index_file(
        index_path=args.index_path,
        top_k=args.top_k,
        max_queries=args.max_queries,
        orb_rerank=args.orb_rerank,
        candidate_k=args.candidate_k,
    )

    print(f"Evaluated queries: {summary.query_count}")
    method = "precision with ORB rerank" if args.orb_rerank else "precision"
    print(f"Metric: {method}@{summary.top_k}")
    print(f"Mean precision: {summary.mean_precision:.4f}")

    if args.show_details:
        print()
        print("Per-query results:")
        for evaluation in summary.evaluations:
            print(
                f"precision={evaluation.precision:.4f} "
                f"({evaluation.relevant_count}/{summary.top_k}) | "
                f"label={evaluation.query_label} | "
                f"{evaluation.query_image_path}"
            )


if __name__ == "__main__":
    main()
