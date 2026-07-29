from __future__ import annotations

import math
import os
import tempfile
import wave
from typing import Dict, Iterable, Optional

from .arranger import SKY_MIDI
from .models import TranscriptionError, TranscriptionResult


PREVIEW_SR = 22050
PREVIEW_NOTE_SECONDS = 0.30


def _key_index(note: Dict[str, object]) -> int:
    key = str(note.get("key", ""))
    if not key.startswith("1Key"):
        raise TranscriptionError(f"无法试听未知按键: {key}")
    index = int(key[4:])
    if not 0 <= index < len(SKY_MIDI):
        raise TranscriptionError(f"按键超出15键范围: {key}")
    return index


def render_preview_wav(
    song_notes: Iterable[Dict[str, object]],
    output_path: Optional[str] = None,
    sample_rate: int = PREVIEW_SR,
) -> str:
    try:
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError("缺少 numpy，无法生成本地试听") from exc

    notes = list(song_notes)
    if not notes:
        raise TranscriptionError("没有可试听的音符")
    first_ms = min(int(note["time"]) for note in notes)
    last_ms = max(int(note["time"]) for note in notes)
    total_seconds = max(0.1, (last_ms - first_ms) / 1000.0 + PREVIEW_NOTE_SECONDS + 0.05)
    audio = np.zeros(int(math.ceil(total_seconds * sample_rate)), dtype=np.float32)
    tone_length = max(1, int(PREVIEW_NOTE_SECONDS * sample_rate))
    t = np.arange(tone_length, dtype=np.float32) / sample_rate
    attack_samples = max(1, int(0.005 * sample_rate))
    envelope = np.exp(-7.0 * t / PREVIEW_NOTE_SECONDS)
    envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)

    tone_cache = {}
    for note in notes:
        index = _key_index(note)
        if index not in tone_cache:
            frequency = 440.0 * (2.0 ** ((SKY_MIDI[index] - 69) / 12.0))
            tone = (
                np.sin(2 * np.pi * frequency * t)
                + 0.35 * np.sin(2 * np.pi * frequency * 2 * t)
                + 0.15 * np.sin(2 * np.pi * frequency * 3 * t)
            )
            tone_cache[index] = (tone * envelope).astype(np.float32)
        start = int(round((int(note["time"]) - first_ms) / 1000.0 * sample_rate))
        end = min(len(audio), start + tone_length)
        audio[start:end] += tone_cache[index][: end - start]

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= 0.95 / peak
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")

    if output_path is None:
        descriptor, output_path = tempfile.mkstemp(prefix="sky-preview-", suffix=".wav")
        os.close(descriptor)
    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return output_path


def render_result_preview(result: TranscriptionResult, output_path: Optional[str] = None) -> str:
    return render_preview_wav(result.song_notes, output_path=output_path)


class PreviewPlayer:
    def __init__(self) -> None:
        self.path: Optional[str] = None

    def play(self, result: TranscriptionResult) -> str:
        self.stop()
        if os.name != "nt":
            raise TranscriptionError("本地异步试听目前仅支持 Windows")
        import winsound

        self.path = render_result_preview(result)
        winsound.PlaySound(self.path, winsound.SND_ASYNC | winsound.SND_FILENAME)
        return self.path

    def stop(self) -> None:
        if os.name == "nt":
            try:
                import winsound

                winsound.PlaySound(None, 0)
            except Exception:
                pass
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass
        self.path = None

