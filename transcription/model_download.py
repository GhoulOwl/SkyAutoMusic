"""Download MuScriptor checkpoints from ModelScope or Hugging Face safely."""
from __future__ import annotations

import os
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Literal, Optional

from .models import TranscriptionError

DownloadSource = Literal["auto", "modelscope", "huggingface"]
DownloadProgress = Callable[[float, str], None]
_SIZES = ("small", "medium", "large")
_CACHE = Path.home() / ".cache" / "muscriptor" / "modelscope"
_MODEL_SCOPE = "https://modelscope.cn/models/MuScriptor/muscriptor-{size}/resolve/master/{name}"
_MODEL_SCOPE_FILES = "https://modelscope.cn/api/v1/models/MuScriptor/muscriptor-{size}/repo/files?Revision=master&Recursive=true"


def modelscope_model_url(size: str) -> str:
    return f"https://modelscope.cn/models/MuScriptor/muscriptor-{size}"


def _emit(status: Optional[Callable[[str], None]], progress: Optional[DownloadProgress],
          fraction: float, message: str) -> None:
    if status:
        status(message)
    if progress:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _download(url: str, destination: Path, progress: Optional[DownloadProgress] = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response, open(destination, "wb") as output:
        total = int(response.headers.get("Content-Length", "0") or 0)
        completed = 0
        last_reported = -1.0
        while block := response.read(1024 * 1024):
            output.write(block)
            completed += len(block)
            if progress and total:
                fraction = completed / total
                if fraction >= 1.0 or fraction - last_reported >= .005:
                    progress(fraction, f"已下载 {completed / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MiB")
                    last_reported = fraction


def _modelscope_checksums(size: str) -> dict[str, str]:
    with urllib.request.urlopen(_MODEL_SCOPE_FILES.format(size=size), timeout=30) as response:
        payload = json.load(response)
    files = payload.get("Data", {}).get("Files", [])
    result = {item.get("Path"): item.get("Sha256") for item in files if item.get("Path") in ("config.json", "model.safetensors")}
    if not all(result.values()):
        raise TranscriptionError("魔搭仓库缺少模型校验信息")
    return result


def _verify(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest().lower() != expected.lower():
        raise TranscriptionError(f"模型文件校验失败：{path.name}")


def _modelscope(size: str, status: Optional[Callable[[str], None]], progress: Optional[DownloadProgress] = None) -> Path:
    root = _CACHE / size
    weights, config = root / "model.safetensors", root / "config.json"
    if weights.is_file() and config.is_file() and weights.stat().st_size > 0:
        try:
            if json.loads(config.read_text(encoding="utf-8")).get("variant") == size:
                _emit(status, progress, 1.0, f"MuScriptor {size} 模型已在本地缓存中")
                return weights
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{size}-", dir=root.parent))
    try:
        _emit(status, progress, .01, f"正在查询 MuScriptor {size} 模型文件")
        checksums = _modelscope_checksums(size)
        for name, start, span in (("config.json", .02, .03), ("model.safetensors", .05, .85)):
            if status:
                status(f"正在从魔搭社区下载 MuScriptor {size}：{name}")
            if progress:
                progress(start, f"正在从魔搭社区下载 MuScriptor {size}：{name}")
            _download(
                _MODEL_SCOPE.format(size=size, name=name), temporary / name,
                lambda fraction, message, start=start, span=span: _emit(
                    status, progress, start + span * fraction, f"{name}：{message}"
                ),
            )
            _emit(status, progress, start + span, f"正在校验 {name}")
            _verify(temporary / name, checksums[name])
        if not (temporary / "model.safetensors").stat().st_size:
            raise TranscriptionError("魔搭模型权重为空")
        root.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "model.safetensors"):
            os.replace(temporary / name, root / name)
        _emit(status, progress, 1.0, f"MuScriptor {size} 模型下载并校验完成")
        return weights
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _huggingface(size: str, token: Optional[str], status: Optional[Callable[[str], None]], progress: Optional[DownloadProgress] = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        _emit(status, progress, .05, f"正在从 Hugging Face 下载 MuScriptor {size} 权重")
        repo = f"MuScriptor/muscriptor-{size}"
        weights = hf_hub_download(repo_id=repo, filename="model.safetensors", token=token)
        _emit(status, progress, .85, "模型权重已下载，正在获取配置")
        config = hf_hub_download(repo_id=repo, filename="config.json", token=token)
        # MuScriptor accepts a local weights path and reads config beside it.
        local_root = Path(weights).parent
        local_config = local_root / "config.json"
        if not local_config.exists():
            shutil.copyfile(config, local_config)
        _emit(status, progress, 1.0, f"MuScriptor {size} 模型下载完成")
        return Path(weights)
    except Exception as exc:
        raise TranscriptionError(f"Hugging Face 下载失败：{exc}") from exc


def download_model(size: str, source: DownloadSource = "auto", token: Optional[str] = None,
                   status: Optional[Callable[[str], None]] = None,
                   progress: Optional[DownloadProgress] = None) -> Path:
    """Return a complete local checkpoint, preferring ModelScope in auto mode."""
    if size not in _SIZES:
        raise ValueError(f"unsupported MuScriptor model: {size}")
    if source not in ("auto", "modelscope", "huggingface"):
        raise ValueError(f"unsupported model download source: {source}")
    if source == "modelscope":
        try:
            return _modelscope(size, status, progress)
        except Exception as exc:
            raise TranscriptionError(f"魔搭社区下载失败：{exc}") from exc
    if source == "huggingface":
        return _huggingface(size, token, status, progress)
    try:
        return _modelscope(size, status, progress)
    except Exception as modelscope_error:
        if status:
            status(f"魔搭下载失败，正在切换 Hugging Face：{modelscope_error}")
        try:
            return _huggingface(size, token, status, progress)
        except Exception as hf_error:
            raise TranscriptionError(f"模型下载失败。魔搭：{modelscope_error}；Hugging Face：{hf_error}") from hf_error
