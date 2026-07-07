from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.cbir.app_config import DEFAULT_SEARCH_CONFIG, load_search_config
from src.cbir.files import list_image_files
from src.cbir.indexing import build_image_index
from src.cbir.search import search_index


def create_solid_color_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> None:
    """Create a simple RGB image for deterministic retrieval tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    image.save(path)


class SearchPipelineTests(unittest.TestCase):
    def test_load_search_config_merges_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "search_config.json"
            config_path.write_text(json.dumps({"method": "deep", "top_k": 3}), encoding="utf-8")

            config = load_search_config(config_path)

        self.assertEqual(config["method"], "deep")
        self.assertEqual(config["top_k"], 3)
        self.assertEqual(config["rrf_k"], DEFAULT_SEARCH_CONFIG["rrf_k"])
        self.assertEqual(config["deep_index_path"], DEFAULT_SEARCH_CONFIG["deep_index_path"])
        self.assertEqual(config["deep_device"], DEFAULT_SEARCH_CONFIG["deep_device"])

    def test_hsv_search_returns_matching_category_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            query_path = root / "query_red.jpg"

            red_image = dataset_dir / "Red" / "red.jpg"
            blue_image = dataset_dir / "Blue" / "blue.jpg"

            create_solid_color_image(red_image, (255, 0, 0))
            create_solid_color_image(blue_image, (0, 0, 255))
            create_solid_color_image(query_path, (250, 10, 10))

            index = build_image_index(dataset_dir, descriptor_type="hsv")
            response = search_index(query_image_path=query_path, index=index, top_k=1)

        self.assertEqual(len(response.results), 1)
        self.assertTrue(response.results[0].image_path.endswith("red.jpg"))
        self.assertTrue(Path(response.results[0].image_path).is_absolute())

    def test_search_excludes_self_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"

            query_image = dataset_dir / "Red" / "query.jpg"
            other_image = dataset_dir / "Blue" / "blue.jpg"

            create_solid_color_image(query_image, (255, 0, 0))
            create_solid_color_image(other_image, (0, 0, 255))

            index = build_image_index(dataset_dir, descriptor_type="hsv")
            response = search_index(query_image_path=query_image, index=index, top_k=1)

            self.assertEqual(len(response.results), 1)
            self.assertTrue(Path(response.results[0].image_path).samefile(other_image))

    def test_list_image_files_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_solid_color_image(root / "keep.jpg", (255, 255, 255))
            (root / "skip.txt").write_text("not an image", encoding="utf-8")

            image_files = list_image_files(root)

        self.assertEqual([path.name for path in image_files], ["keep.jpg"])


if __name__ == "__main__":
    unittest.main()
