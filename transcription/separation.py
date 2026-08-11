"""Internal vocal/accompaniment separation for the V2 arranger."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import CancelledError, ProgressCallback, TranscriptionError


# The bundled checkpoint is retained for its well-tested vocal estimate.  V2
# deliberately exposes only these two derived signals, never six musical stems.
MODEL_NAME = "htdemucs_6s"
MODEL_FILENAME = "5c90dfd2-34c22ccb.th"
MODEL_CHECKSUM_PREFIX = "34c22ccb"
MODEL_MANIFEST_FILENAME = "model-manifest.json"


@dataclass(frozen=True)
class TwoStemAudio:
    vocal_path: str
    accompaniment_path: str
    model_name: str
    sample_rate: int
    duration_sec: float


def _resource_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    return Path(bundle_root) if bundle_root else Path(__file__).resolve().parents[1]


def bundled_model_dir() -> Path:
    override = os.environ.get("SKYAUTOMUSIC_SEPARATION_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _resource_root() / "assets" / "models" / MODEL_NAME


def _sha256_prefix(path: Path, length: int = 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def validate_model_repository(model_dir: Optional[Path] = None) -> Path:
    root = Path(model_dir) if model_dir is not None else bundled_model_dir()
    yaml_path = root / f"{MODEL_NAME}.yaml"
    checkpoint = root / MODEL_FILENAME
    missing = [path.name for path in (yaml_path, checkpoint) if not path.is_file()]
    if missing:
        raise TranscriptionError(
            "人声/伴奏分离模型不完整，缺少 " + "、".join(missing) + "。请重新下载完整 ZIP 发布包。"
        )
    if _sha256_prefix(checkpoint, len(MODEL_CHECKSUM_PREFIX)).lower() != MODEL_CHECKSUM_PREFIX:
        raise TranscriptionError(f"人声/伴奏分离模型校验失败：{checkpoint.name}")
    manifest_path = root / MODEL_MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            expected = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("sha256", "")).lower()
        except (OSError, ValueError, TypeError) as exc:
            raise TranscriptionError(f"分离模型清单无效：{exc}") from exc
        if expected and _sha256_prefix(checkpoint, len(expected)) != expected:
            raise TranscriptionError(f"人声/伴奏分离模型完整校验失败：{checkpoint.name}")
    return root


class TwoStemSeparator:
    """Lazy, process-wide CPU separator that persists only vocals and accompaniment."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else bundled_model_dir()
        self._separator = None
        self._lock = threading.Lock()

    def _load(self, callback) -> object:
        if self._separator is not None:
            self._separator.update_parameter(callback=callback)
            return self._separator
        root = validate_model_repository(self.model_dir)
        try:
            from demucs.api import Separator
        except ImportError as exc:
            raise TranscriptionError("缺少 Demucs / PyTorch，无法分离人声与伴奏") from exc
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
            raise TranscriptionError(f"人声/伴奏分离模型加载失败：{exc}") from exc
        return self._separator

    def prepare(self, progress_cb: Optional[ProgressCallback] = None) -> None:
        with self._lock:
            if progress_cb:
                progress_cb("separate", 0.0, "正在校验人声/伴奏分离模型")
            self._load(None)
            if progress_cb:
                progress_cb("separate", 1.0, "人声/伴奏分离模型已就绪")

    def separate(
        self,
        input_path: str,
        output_dir: str,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> TwoStemAudio:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("转写已取消")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        last_fraction = 0.0

        def callback(info: dict) -> None:
            nonlocal last_fraction
            if cancel_event is not None and cancel_event.is_set():
                raise KeyboardInterrupt
            length = max(1, int(info.get("audio_length", 1)))
            offset = max(0, int(info.get("segment_offset", 0)))
            last_fraction = max(last_fraction, min(0.99, offset / length))
            if progress_cb:
                progress_cb("separate", last_fraction, f"正在分离人声与伴奏… {int(last_fraction * 100)}%")

        with self._lock:
            separator = self._load(callback)
            separator.update_parameter(callback=callback)
            try:
                original, raw_stems = separator.separate_audio_file(Path(input_path))
            except KeyboardInterrupt as exc:
                raise CancelledError("转写已取消") from exc
            except Exception as exc:
                raise TranscriptionError(f"人声/伴奏分离失败：{exc}") from exc

        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("转写已取消")
        if "vocals" not in raw_stems:
            raise TranscriptionError("分离模型没有返回人声")
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise TranscriptionError("缺少 numpy / soundfile，无法保存分离音频") from exc
        sample_rate = int(separator.samplerate)
        vocal = raw_stems["vocals"].detach().cpu().numpy().T
        accompaniment = (original - raw_stems["vocals"]).detach().cpu().numpy().T
        vocal_path = output / "vocals.wav"
        accompaniment_path = output / "accompaniment.wav"
        sf.write(vocal_path, np.nan_to_num(vocal), sample_rate, subtype="PCM_16")
        sf.write(accompaniment_path, np.nan_to_num(accompaniment), sample_rate, subtype="PCM_16")
        if progress_cb:
            progress_cb("separate", 1.0, "人声与伴奏分离完成")
        return TwoStemAudio(
            vocal_path=str(vocal_path.resolve()),
            accompaniment_path=str(accompaniment_path.resolve()),
            model_name=MODEL_NAME,
            sample_rate=sample_rate,
            duration_sec=float(original.shape[-1]) / max(1, sample_rate),
        )


_TWO_STEM_BACKEND = TwoStemSeparator()


def separate_audio(
    input_path: str,
    output_dir: str,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
    separator: Optional[TwoStemSeparator] = None,
) -> TwoStemAudio:
    return (separator or _TWO_STEM_BACKEND).separate(input_path, output_dir, cancel_event, progress_cb)
