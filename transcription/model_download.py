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
_SIZES = ("small", "medium", "large")
_CACHE = Path.home() / ".cache" / "muscriptor" / "modelscope"
_MODEL_SCOPE = "https://modelscope.cn/models/MuScriptor/muscriptor-{size}/resolve/master/{name}"
_MODEL_SCOPE_FILES = "https://modelscope.cn/api/v1/models/MuScriptor/muscriptor-{size}/repo/files?Revision=master&Recursive=true"


def modelscope_model_url(size: str) -> str:
    return f"https://modelscope.cn/models/MuScriptor/muscriptor-{size}"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response, open(destination, "wb") as output:
        shutil.copyfileobj(response, output)


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


def _modelscope(size: str, status: Optional[Callable[[str], None]]) -> Path:
    root = _CACHE / size
    weights, config = root / "model.safetensors", root / "config.json"
    if weights.is_file() and config.is_file() and weights.stat().st_size > 0:
        try:
            if json.loads(config.read_text(encoding="utf-8")).get("variant") == size:
                return weights
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{size}-", dir=root.parent))
    try:
        checksums = _modelscope_checksums(size)
        for name in ("config.json", "model.safetensors"):
            if status:
                status(f"正在从魔搭社区下载 MuScriptor {size}：{name}")
            _download(_MODEL_SCOPE.format(size=size, name=name), temporary / name)
            _verify(temporary / name, checksums[name])
        if not (temporary / "model.safetensors").stat().st_size:
            raise TranscriptionError("魔搭模型权重为空")
        root.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "model.safetensors"):
            os.replace(temporary / name, root / name)
        return weights
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _huggingface(size: str, token: Optional[str], status: Optional[Callable[[str], None]]) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        if status:
            status(f"正在从 Hugging Face 下载 MuScriptor {size}")
        repo = f"MuScriptor/muscriptor-{size}"
        weights = hf_hub_download(repo_id=repo, filename="model.safetensors", token=token)
        config = hf_hub_download(repo_id=repo, filename="config.json", token=token)
        # MuScriptor accepts a local weights path and reads config beside it.
        local_root = Path(weights).parent
        local_config = local_root / "config.json"
        if not local_config.exists():
            shutil.copyfile(config, local_config)
        return Path(weights)
    except Exception as exc:
        raise TranscriptionError(f"Hugging Face 下载失败：{exc}") from exc


def download_model(size: str, source: DownloadSource = "auto", token: Optional[str] = None,
                   status: Optional[Callable[[str], None]] = None) -> Path:
    """Return a complete local checkpoint, preferring ModelScope in auto mode."""
    if size not in _SIZES:
        raise ValueError(f"unsupported MuScriptor model: {size}")
    if source not in ("auto", "modelscope", "huggingface"):
        raise ValueError(f"unsupported model download source: {source}")
    if source == "modelscope":
        try:
            return _modelscope(size, status)
        except Exception as exc:
            raise TranscriptionError(f"魔搭社区下载失败：{exc}") from exc
    if source == "huggingface":
        return _huggingface(size, token, status)
    try:
        return _modelscope(size, status)
    except Exception as modelscope_error:
        if status:
            status(f"魔搭下载失败，正在切换 Hugging Face：{modelscope_error}")
        try:
            return _huggingface(size, token, status)
        except Exception as hf_error:
            raise TranscriptionError(f"模型下载失败。魔搭：{modelscope_error}；Hugging Face：{hf_error}") from hf_error
