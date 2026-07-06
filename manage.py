from __future__ import annotations

import argparse
from pathlib import Path

from src.cbir.app_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOVW_INDEX_PATH,
    DEFAULT_CLASSIC_INDEX_PATH,
    DEFAULT_DEEP_DEVICE,
    DEFAULT_DEEP_INDEX_PATH,
    DEFAULT_IMAGE_DIR,
    DEFAULT_HOG_INDEX_PATH,
    DEFAULT_HSV_HOG_INDEX_PATH,
    DEFAULT_HSV_INDEX_PATH,
    DEFAULT_MAX_DESCRIPTORS,
    DEFAULT_MODELS_DIR,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    DEFAULT_VOCABULARY_SIZE,
)
from src.cbir.bovw import (
    SUPPORTED_BOVW_FEATURES,
    build_and_save_bovw_index,
    evaluate_bovw_index_file,
    load_bovw_index,
    search_bovw_index_file,
)
from src.cbir.deep_embedding import (
    DEFAULT_CLIP_MODEL,
    build_and_save_deep_embedding_index,
    evaluate_deep_embedding_index_file,
    search_deep_embedding_index_file,
)
from src.cbir.evaluation import evaluate_index_file
from src.cbir.fusion import DEFAULT_FUSION_SOURCES, evaluate_fusion, search_fusion
from src.cbir.indexing import (
    SUPPORTED_DESCRIPTORS,
    build_and_save_index,
    load_index,
    save_index,
)
from src.cbir.reranking import rerank_results_with_orb
from src.cbir.search import search_from_index_file, search_index
from src.cbir.visualization import save_search_results, show_search_results


INDEX_FILENAMES = {
    "hsv": Path(DEFAULT_HSV_INDEX_PATH).name,
    "hog": Path(DEFAULT_HOG_INDEX_PATH).name,
    "hsv_hog": Path(DEFAULT_HSV_HOG_INDEX_PATH).name,
}


def add_build_parser(subparsers) -> None:
    parser = subparsers.add_parser("build", help="Build one descriptor index")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--index-path", default=DEFAULT_CLASSIC_INDEX_PATH)
    parser.add_argument("--descriptor", choices=SUPPORTED_DESCRIPTORS, default="hsv")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(handler=handle_build)


def add_query_parser(subparsers) -> None:
    parser = subparsers.add_parser("query", help="Search similar images")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-path", default=DEFAULT_CLASSIC_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save")
    parser.add_argument("--orb-rerank", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rerank-strategy", choices=("weighted", "orb_matches"), default="weighted")
    parser.add_argument("--distance-weight", type=float, default=0.4)
    parser.add_argument("--orb-weight", type=float, default=0.6)
    parser.set_defaults(handler=handle_query)


def add_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate precision@k")
    parser.add_argument("--index-path", default=DEFAULT_CLASSIC_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--show-details", action="store_true")
    parser.add_argument("--orb-rerank", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rerank-strategy", choices=("weighted", "orb_matches"), default="weighted")
    parser.add_argument("--distance-weight", type=float, default=0.4)
    parser.add_argument("--orb-weight", type=float, default=0.6)
    parser.set_defaults(handler=handle_evaluate)


def add_compare_parser(subparsers) -> None:
    parser = subparsers.add_parser("compare", help="Compare multiple indexes")
    parser.add_argument("--index-paths", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-queries", type=int)
    parser.set_defaults(handler=handle_compare)


def add_rebuild_parser(subparsers) -> None:
    parser = subparsers.add_parser("rebuild", help="Rebuild all descriptor indexes")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(handler=handle_rebuild)


def add_bovw_build_parser(subparsers) -> None:
    parser = subparsers.add_parser("bovw-build", help="Build a BoVW + TF-IDF index")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--index-path", default=DEFAULT_BOVW_INDEX_PATH)
    parser.add_argument("--feature", choices=SUPPORTED_BOVW_FEATURES, default="sift")
    parser.add_argument("--vocabulary-size", type=int, default=DEFAULT_VOCABULARY_SIZE)
    parser.add_argument("--max-descriptors", type=int, default=DEFAULT_MAX_DESCRIPTORS)
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(handler=handle_bovw_build)


def add_bovw_query_parser(subparsers) -> None:
    parser = subparsers.add_parser("bovw-query", help="Search with a BoVW index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-path", default=DEFAULT_BOVW_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--verify-top-k", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save")
    parser.set_defaults(handler=handle_bovw_query)


def add_bovw_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser("bovw-evaluate", help="Evaluate a BoVW index")
    parser.add_argument("--index-path", default=DEFAULT_BOVW_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--verify-top-k", type=int, default=0)
    parser.set_defaults(handler=handle_bovw_evaluate)


def add_fusion_query_parser(subparsers) -> None:
    parser = subparsers.add_parser("fusion-query", help="Search with rank fusion")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save")
    parser.set_defaults(handler=handle_fusion_query)


def add_fusion_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser("fusion-evaluate", help="Evaluate rank fusion")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--max-queries", type=int)
    parser.set_defaults(handler=handle_fusion_evaluate)


def add_prepare_parser(subparsers) -> None:
    parser = subparsers.add_parser("prepare", help="Build all indexes required by the app")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--bovw-feature", choices=SUPPORTED_BOVW_FEATURES, default="sift")
    parser.add_argument("--vocabulary-size", type=int, default=DEFAULT_VOCABULARY_SIZE)
    parser.add_argument("--max-descriptors", type=int, default=DEFAULT_MAX_DESCRIPTORS)
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(handler=handle_prepare)


def add_deep_build_parser(subparsers) -> None:
    parser = subparsers.add_parser("deep-build", help="Build a pretrained deep embedding index")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--index-path", default=DEFAULT_DEEP_INDEX_PATH)
    parser.add_argument("--model-name", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEEP_DEVICE)
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(handler=handle_deep_build)


def add_deep_query_parser(subparsers) -> None:
    parser = subparsers.add_parser("deep-query", help="Search with a pretrained deep embedding index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-path", default=DEFAULT_DEEP_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--device", default=DEFAULT_DEEP_DEVICE)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save")
    parser.set_defaults(handler=handle_deep_query)


def add_deep_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser("deep-evaluate", help="Evaluate a pretrained deep embedding index")
    parser.add_argument("--index-path", default=DEFAULT_DEEP_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--device", default=DEFAULT_DEEP_DEVICE)
    parser.set_defaults(handler=handle_deep_evaluate)


def handle_build(args: argparse.Namespace) -> None:
    index = build_and_save_index(
        image_dir=args.image_dir,
        index_path=args.index_path,
        descriptor_type=args.descriptor,
        verbose=args.verbose,
    )

    print(f"Indexed images: {len(index.image_paths)}")
    print(f"Descriptor type: {index.descriptor_name}")
    print(f"Descriptor shape: {index.descriptors.shape}")
    print(f"Build time: {index.build_seconds:.2f} seconds")
    print(f"Index saved to: {args.index_path}")


def handle_query(args: argparse.Namespace) -> None:
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
            strategy=args.rerank_strategy,
            distance_weight=args.distance_weight,
            orb_weight=args.orb_weight,
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
        show_search_results(query_image_path=args.query, results=response.results)


def handle_evaluate(args: argparse.Namespace) -> None:
    summary = evaluate_index_file(
        index_path=args.index_path,
        top_k=args.top_k,
        max_queries=args.max_queries,
        orb_rerank=args.orb_rerank,
        candidate_k=args.candidate_k,
        rerank_strategy=args.rerank_strategy,
        distance_weight=args.distance_weight,
        orb_weight=args.orb_weight,
    )

    method = "precision with ORB rerank" if args.orb_rerank else "precision"
    if args.orb_rerank:
        method = f"{method} ({args.rerank_strategy})"
    print(f"Evaluated queries: {summary.query_count}")
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


def handle_compare(args: argparse.Namespace) -> None:
    rows = []

    for index_path in args.index_paths:
        path = Path(index_path)
        index = load_index(path)
        summary = evaluate_index_file(
            index_path=path,
            top_k=args.top_k,
            max_queries=args.max_queries,
        )
        rows.append(
            {
                "index_path": str(path),
                "descriptor": index.descriptor_name,
                "query_count": summary.query_count,
                "mean_precision": summary.mean_precision,
            }
        )

    rows.sort(key=lambda row: row["mean_precision"], reverse=True)
    print_precision_table(rows, top_k=args.top_k)

    best = rows[0]
    print()
    print(
        "Best index: "
        f"{best['index_path']} "
        f"({best['descriptor']}, precision@{args.top_k}={best['mean_precision']:.4f})"
    )


def handle_rebuild(args: argparse.Namespace) -> None:
    index_paths = build_all_indexes(
        image_dir=args.image_dir,
        models_dir=args.models_dir,
        verbose=args.verbose,
    )

    if args.skip_evaluation:
        return

    rows = evaluate_built_indexes(
        index_paths=index_paths,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )
    print_precision_table(rows, top_k=args.top_k)

    best_row = rows[0]
    best_index = load_index(best_row["index_path"])
    best_output_path = Path(args.models_dir) / "index-best.pkl"
    save_index(best_index, best_output_path)

    print()
    print(
        "Best index saved to: "
        f"{best_output_path} "
        f"({best_row['descriptor']}, precision@{args.top_k}={best_row['mean_precision']:.4f})"
    )


def handle_bovw_build(args: argparse.Namespace) -> None:
    index = build_and_save_bovw_index(
        image_dir=args.image_dir,
        index_path=args.index_path,
        feature_type=args.feature,
        vocabulary_size=args.vocabulary_size,
        max_descriptors=args.max_descriptors,
        verbose=args.verbose,
    )

    print(f"Indexed images: {len(index.image_paths)}")
    print(f"Feature type: {index.feature_type}")
    print(f"Vocabulary size: {index.vocabulary_size}")
    print(f"TF-IDF shape: {index.tfidf_matrix.shape}")
    print(f"Build time: {index.build_seconds:.2f} seconds")
    print(f"BoVW index saved to: {args.index_path}")


def handle_bovw_query(args: argparse.Namespace) -> None:
    response = search_bovw_index_file(
        query_image_path=args.query,
        index_path=args.index_path,
        top_k=args.top_k,
        verify_top_k=args.verify_top_k,
    )

    print(f"Query time: {response.query_seconds * 1000:.2f} ms")
    print(f"Top-{len(response.results)} results:")

    for position, result in enumerate(response.results, start=1):
        if result.orb_matches is None:
            print(f"{position:02d}. distance={result.distance:.6f} | {result.image_path}")
        else:
            print(
                f"{position:02d}. "
                f"inliers={result.orb_matches} | "
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
        show_search_results(query_image_path=args.query, results=response.results)


def handle_bovw_evaluate(args: argparse.Namespace) -> None:
    summary = evaluate_bovw_index_file(
        index_path=args.index_path,
        top_k=args.top_k,
        max_queries=args.max_queries,
        verify_top_k=args.verify_top_k,
    )

    metric = "precision"
    if args.verify_top_k > 0:
        metric = f"{metric} with geometric verification"

    print(f"Evaluated queries: {summary.query_count}")
    print(f"Metric: {metric}@{summary.top_k}")
    print(f"Mean precision: {summary.mean_precision:.4f}")


def handle_fusion_query(args: argparse.Namespace) -> None:
    response = search_fusion(
        query_image_path=args.query,
        sources=DEFAULT_FUSION_SOURCES,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )

    print(f"Query time: {response.query_seconds * 1000:.2f} ms")
    print(f"Top-{len(response.results)} fusion results:")

    for position, result in enumerate(response.results, start=1):
        print(
            f"{position:02d}. "
            f"fusion_score={result.rerank_score:.6f} | "
            f"sources={result.source_count} | "
            f"best_distance={result.distance:.6f} | "
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
        show_search_results(query_image_path=args.query, results=response.results)


def handle_fusion_evaluate(args: argparse.Namespace) -> None:
    query_paths = get_fusion_query_paths()
    summary = evaluate_fusion(
        query_paths=query_paths,
        sources=DEFAULT_FUSION_SOURCES,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        max_queries=args.max_queries,
    )

    print(f"Evaluated queries: {summary.query_count}")
    print(f"Metric: fusion precision@{summary.top_k}")
    print(f"Mean precision: {summary.mean_precision:.4f}")


def handle_prepare(args: argparse.Namespace) -> None:
    if args.models_dir != DEFAULT_MODELS_DIR:
        raise ValueError("prepare currently expects --models-dir models because app fusion config uses models/*.pkl")

    print("Step 1/3: building classic indexes")
    build_all_indexes(
        image_dir=args.image_dir,
        models_dir=args.models_dir,
        verbose=args.verbose,
    )

    print("Step 2/3: building BoVW index")
    bovw_index = build_and_save_bovw_index(
        image_dir=args.image_dir,
        index_path=DEFAULT_BOVW_INDEX_PATH,
        feature_type=args.bovw_feature,
        vocabulary_size=args.vocabulary_size,
        max_descriptors=args.max_descriptors,
        verbose=args.verbose,
    )
    print(f"Indexed images: {len(bovw_index.image_paths)}")
    print(f"Feature type: {bovw_index.feature_type}")
    print(f"Vocabulary size: {bovw_index.vocabulary_size}")
    print(f"TF-IDF shape: {bovw_index.tfidf_matrix.shape}")
    print(f"Build time: {bovw_index.build_seconds:.2f} seconds")
    print()

    print("Step 3/3: evaluating fusion")
    query_paths = get_fusion_query_paths()
    summary = evaluate_fusion(
        query_paths=query_paths,
        sources=DEFAULT_FUSION_SOURCES,
        top_k=args.top_k,
        rrf_k=DEFAULT_RRF_K,
    )
    print(f"Evaluated queries: {summary.query_count}")
    print(f"Metric: fusion precision@{summary.top_k}")
    print(f"Mean precision: {summary.mean_precision:.4f}")


def handle_deep_build(args: argparse.Namespace) -> None:
    index = build_and_save_deep_embedding_index(
        image_dir=args.image_dir,
        index_path=args.index_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"Indexed images: {len(index.image_paths)}")
    print(f"Model: {index.model_name}")
    print(f"Embedding shape: {index.embeddings.shape}")
    print(f"Build time: {index.build_seconds:.2f} seconds")
    print(f"Deep embedding index saved to: {args.index_path}")


def handle_deep_query(args: argparse.Namespace) -> None:
    response = search_deep_embedding_index_file(
        query_image_path=args.query,
        index_path=args.index_path,
        top_k=args.top_k,
        device=args.device,
    )

    print(f"Query time: {response.query_seconds * 1000:.2f} ms")
    print(f"Top-{len(response.results)} deep embedding results:")

    for position, result in enumerate(response.results, start=1):
        print(f"{position:02d}. distance={result.distance:.6f} | {result.image_path}")

    if args.save:
        save_search_results(
            query_image_path=args.query,
            results=response.results,
            output_path=args.save,
        )
        print(f"Result grid saved to: {args.save}")

    if args.show:
        show_search_results(query_image_path=args.query, results=response.results)


def handle_deep_evaluate(args: argparse.Namespace) -> None:
    summary = evaluate_deep_embedding_index_file(
        index_path=args.index_path,
        top_k=args.top_k,
        max_queries=args.max_queries,
        device=args.device,
    )

    print(f"Evaluated queries: {summary.query_count}")
    print(f"Metric: deep embedding precision@{summary.top_k}")
    print(f"Mean precision: {summary.mean_precision:.4f}")


def get_fusion_query_paths() -> list[str]:
    """Use the first available fusion source as the evaluation query set."""
    for source in DEFAULT_FUSION_SOURCES:
        index_path = Path(source.index_path)
        if not index_path.exists():
            continue

        if source.method == "classic":
            return load_index(index_path).image_paths

        if source.method == "bovw":
            return load_bovw_index(index_path).image_paths

    raise FileNotFoundError("No fusion source index was found")


def build_all_indexes(image_dir: str, models_dir: str, verbose: bool) -> dict[str, Path]:
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {}

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


def evaluate_built_indexes(
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


def print_precision_table(rows: list[dict[str, object]], top_k: int) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CBIR Image Search Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_build_parser(subparsers)
    add_query_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_compare_parser(subparsers)
    add_rebuild_parser(subparsers)
    add_bovw_build_parser(subparsers)
    add_bovw_query_parser(subparsers)
    add_bovw_evaluate_parser(subparsers)
    add_fusion_query_parser(subparsers)
    add_fusion_evaluate_parser(subparsers)
    add_prepare_parser(subparsers)
    add_deep_build_parser(subparsers)
    add_deep_query_parser(subparsers)
    add_deep_evaluate_parser(subparsers)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
