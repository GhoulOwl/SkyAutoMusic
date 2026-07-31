from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .models import (
    CancelledError,
    ProgressCallback,
    StemKind,
    TranscriptionError,
)


MODEL_NAME = "htdemucs_6s"
MODEL_FILENAME = "5c90dfd2-34c22ccb.th"
MODEL_CHECKSUM_PREFIX = "34c22ccb"
MODEL_MANIFEST_FILENAME = "model-manifest.json"
VISIBLE_STEMS = ("vocals", "drums", "piano", "bass", "guitar")


@dataclass(frozen=True)
class SeparatedAudio:
    paths: Dict[StemKind, str]
    model_name: str
    sample_rate: int
    duration_sec: float


def _resource_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)
    return Path(__file__).resolve().parents[1]


def bundled_model_dir() -> Path:
    override = os.environ.get("SKYAUTOMUSIC_SEPARATION_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _resource_root() / "assets" / "models" / MODEL_NAME


def _sha256_prefix(path: Path, length: int = 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:length]


def validate_model_repository(model_dir: Optional[Path] = None) -> Path:
    root = Path(model_dir) if model_dir is not None else bundled_model_dir()
    yaml_path = root / f"{MODEL_NAME}.yaml"
    checkpoint = root / MODEL_FILENAME
    missing = [
        path.name
        for path in (yaml_path, checkpoint)
        if not path.is_file()
    ]
    if missing:
        raise TranscriptionError(
            "六轨分离模型不完整，缺少 "
            + "、".join(missing)
            + "。请重新下载完整 ZIP 发布包，或切换到普通复音模式。"
        )
    actual_prefix = _sha256_prefix(checkpoint, len(MODEL_CHECKSUM_PREFIX))
    if actual_prefix.lower() != MODEL_CHECKSUM_PREFIX:
        raise TranscriptionError(
            f"六轨分离模型校验失败: {checkpoint.name}。"
            "请重新下载完整 ZIP 发布包。"
        )
    manifest_path = root / MODEL_MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = str(manifest.get("sha256", "")).strip().lower()
        except (OSError, ValueError, TypeError) as exc:
            raise TranscriptionError(f"分离模型清单无效: {exc}") from exc
        if expected:
            actual = _sha256_prefix(checkpoint, len(expected))
            if actual != expected:
                raise TranscriptionError(
                    f"六轨分离模型完整校验失败: {checkpoint.name}"
                )
    return root


class DemucsStemSeparator:
    """Thread-safe, lazily loaded CPU Demucs separator using bundled weights."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else bundled_model_dir()
        self._separator = None
        self._lock = threading.Lock()

    def _load(self, callback):
        if self._separator is not None:
            self._separator.update_parameter(callback=callback)
            return self._separator
        root = validate_model_repository(self.model_dir)
        try:
            from demucs.api import Separator
        except ImportError as exc:
            raise TranscriptionError(
                "缺少 Demucs / PyTorch，无法使用智能六轨融合。"
                "请安装完整依赖或使用 ZIP 发布版。"
            ) from exc
        try:
            self._separator = Separator(
                model=MODEL_NAME,
                repo=root,
                device="cpu",
                shifts=0,
                overlap=0.25,
                split=True,
                jobs=0,
                progress=False,
                callback=callback,
            )
        except Exception as exc:
            raise TranscriptionError(f"六轨分离模型加载失败: {exc}") from exc
        return self._separator

    def prepare(
        self,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> None:
        """Validate and load the bundled model without separating audio."""
        with self._lock:
            if progress_cb:
                progress_cb("separate", 0.0, "正在离线校验六轨分离模型")
            self._load(None)
            if progress_cb:
                progress_cb("separate", 1.0, "六轨分离模型已就绪")

    def separate(
        self,
        input_path: str,
        output_dir: str,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> SeparatedAudio:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("转写已取消")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        last_fraction = 0.0

        def callback(info: dict) -> None:
            nonlocal last_fraction
            if cancel_event is not None and cancel_event.is_set():
                raise KeyboardInterrupt
            audio_length = max(1, int(info.get("audio_length", 1)))
            offset = max(0, int(info.get("segment_offset", 0)))
            fraction = min(0.99, offset / audio_length)
            if info.get("state") == "end":
                fraction = min(0.99, max(fraction, last_fraction + 0.01))
            last_fraction = max(last_fraction, fraction)
            if progress_cb:
                progress_cb(
                    "separate",
                    last_fraction,
                    f"正在分离人声与乐器… {int(last_fraction * 100)}%",
                )

        with self._lock:
            if progress_cb:
                progress_cb("separate", 0.0, "正在加载六轨分离模型")
            separator = self._load(callback)
            separator.update_parameter(callback=callback)
            try:
                original, raw_stems = separator.separate_audio_file(Path(input_path))
            except KeyboardInterrupt as exc:
                raise CancelledError("转写已取消") from exc
            except CancelledError:
                raise
            except Exception as exc:
                raise TranscriptionError(f"六轨分离失败: {exc}") from exc

        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("转写已取消")
        missing = [stem for stem in VISIBLE_STEMS if stem not in raw_stems]
        if missing:
            raise TranscriptionError(
                "六轨分离结果缺少声部: " + "、".join(missing)
            )
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise TranscriptionError("缺少 numpy / soundfile，无法保存分轨音频") from exc

        sample_rate = int(separator.samplerate)
        tensors = {stem: raw_stems[stem] for stem in VISIBLE_STEMS}
        tensors["instrumental"] = original - raw_stems["vocals"]
        paths: Dict[StemKind, str] = {}
        for stem in (*VISIBLE_STEMS, "instrumental"):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("转写已取消")
            tensor = tensors[stem].detach().cpu().numpy().T
            tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            path = output / f"{stem}.wav"
            sf.write(path, tensor, sample_rate, subtype="PCM_16")
            paths[stem] = str(path.resolve())  # type: ignore[index]

        duration_sec = float(original.shape[-1]) / max(1, sample_rate)
        if progress_cb:
            progress_cb("separate", 1.0, "人声与乐器分离完成")
        return SeparatedAudio(
            paths=paths,
            model_name=MODEL_NAME,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
        )


_DEMUCS_BACKEND = DemucsStemSeparator()


def separate_audio(
    input_path: str,
    output_dir: str,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
    separator: Optional[DemucsStemSeparator] = None,
) -> SeparatedAudio:
    backend = separator or _DEMUCS_BACKEND
    return backend.separate(input_path, output_dir, cancel_event, progress_cb)
