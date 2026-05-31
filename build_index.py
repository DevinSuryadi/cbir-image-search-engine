from __future__ import annotations

import argparse

from src.indexing import build_and_save_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CBIR image index")
    parser.add_argument(
        "--image-dir",
        default="data/images",
        help="Directory containing image dataset files",
    )
    parser.add_argument(
        "--index-path",
        default="models/index.pkl",
        help="Output path for the generated index file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print indexing progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_and_save_index(
        image_dir=args.image_dir,
        index_path=args.index_path,
        verbose=args.verbose,
    )

    print(f"Indexed images: {len(index.image_paths)}")
    print(f"Descriptor shape: {index.descriptors.shape}")
    print(f"Build time: {index.build_seconds:.2f} seconds")
    print(f"Index saved to: {args.index_path}")


if __name__ == "__main__":
    main()
