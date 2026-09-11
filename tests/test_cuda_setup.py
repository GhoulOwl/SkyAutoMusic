import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.cuda_setup import ensure_cuda_runtime
from transcription.models import TranscriptionError


class TestCudaSetup(unittest.TestCase):
    def test_ready_runtime_does_not_start_installer(self):
        with patch("transcription.cuda_setup._cuda_is_ready", return_value=True), patch("transcription.cuda_setup.subprocess.Popen") as install:
            result = ensure_cuda_runtime()
        self.assertTrue(result.ready)
        install.assert_not_called()

    def test_missing_nvidia_gpu_has_actionable_error(self):
        with patch("transcription.cuda_setup._cuda_is_ready", return_value=False), patch("transcription.cuda_setup.sys.platform", "win32"), patch("transcription.cuda_setup._nvidia_gpu_name", return_value=None):
            with self.assertRaisesRegex(TranscriptionError, "NVIDIA"):
                ensure_cuda_runtime()

    def test_installs_cuda_runtime_and_requires_restart(self):
        process = MagicMock()
        process.stdout = ["Downloading torch\n", "Installing collected packages\n"]
        process.wait.return_value = 0
        progress = []
        with patch("transcription.cuda_setup._cuda_is_ready", return_value=False), patch("transcription.cuda_setup.sys.platform", "win32"), patch("transcription.cuda_setup._nvidia_gpu_name", return_value="NVIDIA RTX"), patch("transcription.cuda_setup.subprocess.run"), patch("transcription.cuda_setup.subprocess.Popen", return_value=process) as install:
            result = ensure_cuda_runtime(lambda fraction, message: progress.append((fraction, message)))
        self.assertTrue(result.restart_required)
        self.assertIn("--force-reinstall", install.call_args.args[0])
        self.assertIn("--extra-index-url", install.call_args.args[0])
        self.assertIn("https://download.pytorch.org/whl/cu128", install.call_args.args[0])
        self.assertIn("torch==2.7.1+cu128", install.call_args.args[0])
        self.assertTrue(progress)


if __name__ == "__main__":
    unittest.main()
