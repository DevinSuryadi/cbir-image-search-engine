from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import evaluate_index_file
from src.indexing import SUPPORTED_DESCRIPTORS, build_and_save_index, load_index, save_index


INDEX_FILENAMES = {
    "hsv": "index-hsv.pkl",
    "hog": "index-hog.pkl",
    "hsv_hog": "index-hsv-hog.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild all CBIR indexes after dataset changes")
    parser.add_argument(
        "--image-dir",
        default="data/images",
        help="Directory containing image dataset files",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory used to store generated index files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k value used for precision evaluation",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Optional limit for the number of query images evaluated",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only rebuild indexes without comparing precision",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print indexing progress",
    )
    return parser.parse_args()


def build_all_indexes(image_dir: str, models_dir: str, verbose: bool) -> dict[str, Path]:
    index_paths = {}
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for descriptor in SUPPORTED_DESCRIPTORS:
        index_path = output_dir / INDEX_FILENAMES[descriptor]
        print(f"Building {descriptor} index -> {index_path}")
        index = build_and_save_index(
            image_dir=image_dir,
            index_path=index_path,
            descriptor_type=descriptor,
            verbose=verbose,
        )
        print(f"Indexed images: {len(index.image_paths)}")
        print(f"Descriptor shape: {index.descriptors.shape}")
        print(f"Build time: {index.build_seconds:.2f} seconds")
        print()
        index_paths[descriptor] = index_path

    return index_paths


def evaluate_indexes(
    index_paths: dict[str, Path],
    top_k: int,
    max_queries: int | None,
) -> list[dict[str, object]]:
    rows = []

    for descriptor, index_path in index_paths.items():
        summary = evaluate_index_file(
            index_path=index_path,
            top_k=top_k,
            max_queries=max_queries,
        )
        rows.append(
            {
                "descriptor": descriptor,
                "index_path": index_path,
                "query_count": summary.query_count,
                "mean_precision": summary.mean_precision,
            }
        )

    rows.sort(key=lambda row: row["mean_precision"], reverse=True)
    return rows


def save_best_index(best_index_path: Path, models_dir: str) -> Path:
    best_index = load_index(best_index_path)
    output_path = Path(models_dir) / "index-best.pkl"
    save_index(best_index, output_path)
    return output_path


def print_evaluation(rows: list[dict[str, object]], top_k: int) -> None:
    print(f"Metric: precision@{top_k}")
    print()
    print("Rank | Descriptor | Mean Precision | Queries | Index")
    print("-----|------------|----------------|---------|------")

    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:04d} | "
            f"{row['descriptor']} | "
            f"{row['mean_precision']:.4f} | "
            f"{row['query_count']} | "
            f"{row['index_path']}"
        )


def main() -> None:
    args = parse_args()
    index_paths = build_all_indexes(
        image_dir=args.image_dir,
        models_dir=args.models_dir,
        verbose=args.verbose,
    )

    if args.skip_evaluation:
        return

    rows = evaluate_indexes(
        index_paths=index_paths,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )
    print_evaluation(rows, top_k=args.top_k)

    best_row = rows[0]
    best_index_path = save_best_index(
        best_index_path=best_row["index_path"],
        models_dir=args.models_dir,
    )

    print()
    print(
        "Best index saved to: "
        f"{best_index_path} "
        f"({best_row['descriptor']}, precision@{args.top_k}={best_row['mean_precision']:.4f})"
    )


if __name__ == "__main__":
    main()
