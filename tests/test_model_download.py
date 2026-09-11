import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.model_download import _download, download_model
from transcription.models import TranscriptionError


class TestModelDownloadSource(unittest.TestCase):
    def test_transport_reports_incremental_byte_progress(self):
        class Response:
            headers = {"Content-Length": "6"}

            def __init__(self):
                self.parts = iter((b"abc", b"def", b""))

            def read(self, _size):
                return next(self.parts)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        progress = []
        with tempfile.TemporaryDirectory() as folder, patch("transcription.model_download.urllib.request.urlopen", return_value=Response()):
            destination = Path(folder) / "model.safetensors"
            _download("https://example.test/model", destination, lambda fraction, message: progress.append((fraction, message)))
            self.assertEqual(destination.read_bytes(), b"abcdef")
        self.assertEqual([fraction for fraction, _message in progress], [.5, 1.0])
        self.assertIn("0.0 / 0.0 MiB", progress[-1][1])

    def test_auto_falls_back_to_huggingface(self):
        expected = Path("/tmp/model.safetensors")
        with patch("transcription.model_download._modelscope", side_effect=OSError("offline")), patch(
            "transcription.model_download._huggingface", return_value=expected
        ) as huggingface:
            self.assertEqual(download_model("small", "auto"), expected)
            huggingface.assert_called_once()

    def test_single_modelscope_source_does_not_fallback(self):
        with patch("transcription.model_download._modelscope", side_effect=OSError("offline")), patch(
            "transcription.model_download._huggingface"
        ) as huggingface:
            with self.assertRaises(TranscriptionError):
                download_model("small", "modelscope")
            huggingface.assert_not_called()

    def test_modelscope_cache_is_shared_by_device(self):
        with tempfile.TemporaryDirectory() as folder, patch("transcription.model_download._CACHE", Path(folder)):
            root = Path(folder) / "small"
            root.mkdir()
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "config.json").write_text('{"variant": "small"}')
            with patch("transcription.model_download._download") as download:
                self.assertEqual(download_model("small", "modelscope"), root / "model.safetensors")
                self.assertEqual(download_model("small", "modelscope"), root / "model.safetensors")
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
