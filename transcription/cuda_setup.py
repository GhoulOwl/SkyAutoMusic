"""Detect and provision the optional Windows CUDA runtime for MuScriptor."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional

from .models import TranscriptionError


SetupProgress = Callable[[float, str], None]
_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
_TORCH_VERSION = "2.7.1"
_CUDA_BUILD = "cu128"


@dataclass(frozen=True)
class CudaSetupResult:
    state: str
    message: str

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def restart_required(self) -> bool:
        return self.state == "restart_required"


def _notify(progress: Optional[SetupProgress], fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _cuda_is_ready() -> bool:
    try:
        import torch
        return bool(torch.version.cuda and torch.cuda.is_available())
    except Exception:
        return False


def _nvidia_gpu_name() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return next((line.strip() for line in completed.stdout.splitlines() if line.strip()), None)


def ensure_cuda_runtime(progress: Optional[SetupProgress] = None) -> CudaSetupResult:
    """Install the CUDA PyTorch wheel after an explicit GPU-model action.

    PyTorch DLLs cannot safely be swapped in a live process.  A successful
    install therefore deliberately returns ``restart_required`` instead of
    pretending the current process has gained CUDA support.
    """
    if _cuda_is_ready():
        _notify(progress, .40, "CUDA 环境已就绪")
        return CudaSetupResult("ready", "CUDA 环境已就绪")
    if sys.platform != "win32":
        raise TranscriptionError("CUDA 自动配置仅适用于 Windows；Apple 芯片将使用 MPS。")
    gpu_name = _nvidia_gpu_name()
    if not gpu_name:
        raise TranscriptionError("未检测到可用的 NVIDIA CUDA 显卡。请选择 Small（CPU）。")
    if getattr(sys, "frozen", False):
        raise TranscriptionError("当前打包版无法安装 CUDA 运行环境；请使用包含 CUDA 的发布包，或在源码环境中运行自动配置。")

    _notify(progress, .03, f"检测到 {gpu_name}，正在安装 CUDA PyTorch 运行环境")
    try:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TranscriptionError(f"无法准备 Python 安装工具：{exc}") from exc
    command = [
        sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
        "--extra-index-url", _CUDA_INDEX,
        f"torch=={_TORCH_VERSION}+{_CUDA_BUILD}", f"torchaudio=={_TORCH_VERSION}+{_CUDA_BUILD}",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            text = line.strip()
            if text:
                _notify(progress, .12, f"正在配置 CUDA：{text[:180]}")
        if process.wait() != 0:
            raise TranscriptionError("CUDA PyTorch 安装失败，请检查网络连接、磁盘空间和 NVIDIA 驱动。")
    except OSError as exc:
        raise TranscriptionError(f"无法启动 CUDA 环境安装：{exc}") from exc
    _notify(progress, .40, "CUDA PyTorch 已安装；重启程序后将自动启用 Medium（CUDA）")
    return CudaSetupResult("restart_required", "CUDA PyTorch 已安装；请重启程序后继续下载 Medium 模型。")
