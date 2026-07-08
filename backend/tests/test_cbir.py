"""
Unit and integration tests for the CBIR FastAPI backend.

Tests cover:
  - Core CBIR utility functions (files, embeddings)
  - FastAPI endpoint responses via TestClient (no real Qdrant/model calls)
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


# ---------------------------------------------------------------------------
# Utility: create a minimal in-memory JPEG bytes object
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (64, 64)) -> bytes:
    """Return raw JPEG bytes for a solid-color image."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Tests: src.cbir.files
# ---------------------------------------------------------------------------

class FilesTests(unittest.TestCase):
    def test_list_image_files_finds_jpg(self) -> None:
        """list_image_files should discover .jpg files recursively."""
        from src.cbir.files import list_image_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "cat" / "image_0001.jpg"
            img.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), (0, 0, 0)).save(img)

            result = list_image_files(root)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "image_0001.jpg")

    def test_list_image_files_ignores_non_images(self) -> None:
        """list_image_files should skip non-image files like .txt."""
        from src.cbir.files import list_image_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("not an image", encoding="utf-8")
            img = root / "photo.png"
            Image.new("RGB", (32, 32), (255, 255, 255)).save(img)

            result = list_image_files(root)

        self.assertEqual([f.name for f in result], ["photo.png"])

    def test_list_image_files_raises_on_missing_dir(self) -> None:
        """list_image_files should raise FileNotFoundError for a non-existent path."""
        from src.cbir.files import list_image_files

        with self.assertRaises(FileNotFoundError):
            list_image_files("/this/path/does/not/exist")

    def test_canonicalize_image_path_normalizes_separators(self) -> None:
        """canonicalize_image_path should replace backslashes with forward slashes."""
        from src.cbir.files import canonicalize_image_path

        result = canonicalize_image_path(r"data\images\cat\photo.jpg")
        self.assertNotIn("\\", result)
        self.assertIn("/", result)


# ---------------------------------------------------------------------------
# Tests: src.cbir.deep_embedding (fast, no actual model loading)
# ---------------------------------------------------------------------------

class DeepEmbeddingTests(unittest.TestCase):
    def test_l2_normalize_matrix_produces_unit_vectors(self) -> None:
        """l2_normalize_matrix should produce rows with L2 norm ≈ 1.0."""
        import numpy as np
        from src.cbir.deep_embedding import l2_normalize_matrix

        matrix = np.random.randn(8, 768).astype(np.float32)
        normalized = l2_normalize_matrix(matrix)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.ones(8), atol=1e-5)

    def test_l2_normalize_matrix_handles_zero_vector(self) -> None:
        """l2_normalize_matrix should not produce NaN for zero vectors."""
        import numpy as np
        from src.cbir.deep_embedding import l2_normalize_matrix

        matrix = np.zeros((3, 512), dtype=np.float32)
        result = l2_normalize_matrix(matrix)
        self.assertFalse(np.any(np.isnan(result)))

    @patch("src.cbir.deep_embedding.load_clip_model")
    def test_encode_text_clip_returns_1d_vector(self, mock_load: MagicMock) -> None:
        """encode_text_clip should return a 1D float32 numpy array."""
        import numpy as np
        from src.cbir.deep_embedding import encode_text_clip

        # Mock model that returns a (1, 512) tensor
        import torch
        mock_model = MagicMock()
        mock_model.get_text_features.return_value = torch.randn(1, 512)

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        result = encode_text_clip(
            text="a photo of a cat",
            model=mock_model,
            processor=mock_processor,
            device="cpu",
        )

        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.dtype, np.float32)


# ---------------------------------------------------------------------------
# Tests: FastAPI endpoints via TestClient (mocked Qdrant + CLIP state)
# ---------------------------------------------------------------------------

MOCK_SEARCH_RESULTS = [
    {
        "point_id": "abc-123",
        "image_url": "https://example.supabase.co/storage/v1/object/public/images/cat/photo.jpg",
        "category": "cat",
        "score": 0.92,
    }
]


class APIEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        """Patch clip_state and Qdrant search before each test."""
        import main

        self.clip_patcher = patch.dict(
            main.clip_state,
            {"model": MagicMock(), "processor": MagicMock(), "device": "cpu"},
        )
        self.clip_patcher.start()

    def tearDown(self) -> None:
        self.clip_patcher.stop()

    def _get_client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app, raise_server_exceptions=True)

    def test_health_check(self) -> None:
        """GET / should return status ok."""
        client = self._get_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("main.search_by_vector", return_value=MOCK_SEARCH_RESULTS)
    @patch("main._encode_image_to_vector", return_value=[0.1] * 768)
    def test_search_image_returns_results(self, _mock_encode, _mock_search) -> None:
        """POST /api/search/image should return valid search results."""
        client = self._get_client()
        jpeg_bytes = _make_jpeg_bytes()
        response = client.post(
            "/api/search/image",
            files={"file": ("query.jpg", jpeg_bytes, "image/jpeg")},
            data={"top_k": "5"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["result_count"], 1)
        self.assertEqual(body["results"][0]["category"], "cat")

    @patch("main._qdrant_search_by_text", return_value=MOCK_SEARCH_RESULTS)
    def test_search_text_returns_results(self, _mock_search) -> None:
        """POST /api/search/text should return valid search results."""
        client = self._get_client()
        response = client.post(
            "/api/search/text",
            json={"query": "a dog running in a park", "top_k": 5},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result_count"], 1)

    def test_search_text_rejects_empty_query(self) -> None:
        """POST /api/search/text with empty query should return 422."""
        client = self._get_client()
        response = client.post("/api/search/text", json={"query": "  ", "top_k": 5})
        self.assertEqual(response.status_code, 422)

    def test_search_image_rejects_invalid_top_k(self) -> None:
        """POST /api/search/image with top_k=0 should return 422."""
        client = self._get_client()
        jpeg_bytes = _make_jpeg_bytes()
        response = client.post(
            "/api/search/image",
            files={"file": ("q.jpg", jpeg_bytes, "image/jpeg")},
            data={"top_k": "0"},
        )
        self.assertEqual(response.status_code, 422)

    def test_search_image_rejects_non_image(self) -> None:
        """POST /api/search/image with a text file should return 400."""
        client = self._get_client()
        response = client.post(
            "/api/search/image",
            files={"file": ("not_an_image.jpg", b"this is not an image", "image/jpeg")},
            data={"top_k": "5"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
