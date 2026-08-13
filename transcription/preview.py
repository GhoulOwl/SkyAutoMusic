from __future__ import annotations

import math
import os
import tempfile
import wave
from collections import defaultdict
from typing import Callable, Dict, Iterable, Optional

from audio_preview import AudioPreviewController, PianoSampleLibrary
from player import MusicPlayer, PlaybackState

from .arranger import SKY_MIDI
from .models import TranscriptionError, TranscriptionResult


PREVIEW_SR = 22050


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
    library: Optional[PianoSampleLibrary] = None,
) -> str:
    """Render with the bundled Sky samples; kept for export/tests, not live playback."""

    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa / numpy，无法生成钢琴音色试听") from exc

    notes = list(song_notes)
    if not notes:
        raise TranscriptionError("没有可试听的音符")
    sample_paths = (library or PianoSampleLibrary()).ensure_samples()
    used_indices = sorted({_key_index(note) for note in notes})
    decoded = {}
    for index in used_indices:
        try:
            audio, _sr = librosa.load(
                sample_paths[index],
                sr=sample_rate,
                mono=True,
            )
        except Exception as exc:
            raise TranscriptionError(f"无法读取钢琴音色 {index}: {exc}") from exc
        decoded[index] = np.asarray(audio, dtype=np.float32)

    first_ms = min(int(note["time"]) for note in notes)
    last_ms = max(int(note["time"]) for note in notes)
    longest = max(len(decoded[index]) for index in used_indices)
    total_samples = (
        int(math.ceil((last_ms - first_ms) / 1000.0 * sample_rate))
        + longest
        + 1
    )
    mixed = np.zeros(max(1, total_samples), dtype=np.float32)
    for note in notes:
        index = _key_index(note)
        sample = decoded[index]
        start = int(round((int(note["time"]) - first_ms) / 1000.0 * sample_rate))
        end = min(len(mixed), start + len(sample))
        mixed[start:end] += sample[: end - start]
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed *= 0.95 / peak
    pcm = np.clip(mixed * 32767.0, -32768, 32767).astype("<i2")

    if output_path is None:
        descriptor, output_path = tempfile.mkstemp(prefix="sky-preview-", suffix=".wav")
        os.close(descriptor)
    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return output_path


def render_result_preview(
    result: TranscriptionResult,
    output_path: Optional[str] = None,
) -> str:
    return render_preview_wav(result.song_notes, output_path=output_path)


class PreviewPlayer:
    """Live Sky-sample score preview driven by the normal timestamp scheduler."""

    def __init__(
        self,
        controller: Optional[AudioPreviewController] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.on_status = on_status or (lambda _message: None)
        self.on_finished = on_finished or (lambda: None)
        self.on_error = on_error or (lambda _exc: None)
        self.controller = controller or AudioPreviewController(
            on_error=self._handle_error
        )
        self.player = MusicPlayer(
            key_controller=self.controller,
            update_status=self._handle_status,
            update_finished=self._handle_finished,
        )

    @property
    def prepared(self) -> bool:
        return self.controller.prepared

    @property
    def state(self) -> PlaybackState:
        return self.player.state

    def prepare(self, progress_callback=None) -> None:
        self.controller.prepare(progress_callback=progress_callback)

    def play(self, result: TranscriptionResult) -> bool:
        if not self.prepared:
            raise TranscriptionError("钢琴音色尚未准备完成")
        self.stop()
        notes_by_time = defaultdict(list)
        for note in result.song_notes:
            time_ms = int(note["time"])
            key = str(note["key"])
            _key_index(note)
            if key not in notes_by_time[time_ms]:
                notes_by_time[time_ms].append(key)
        sorted_times = sorted(notes_by_time)
        if not sorted_times:
            raise TranscriptionError("没有可试听的音符")
        self.player.speed = 1.0
        self.player.simulate = False
        if not self.player.start(dict(notes_by_time), sorted_times):
            raise TranscriptionError("无法启动钢琴音色试听")
        return True

    def pause_or_resume(self) -> PlaybackState:
        if self.player.state == PlaybackState.PLAYING:
            self.player.pause()
            self.controller.stop_all()
        elif self.player.state == PlaybackState.PAUSED:
            self.player.resume()
        return self.player.state

    def stop(self) -> None:
        self.player.stop()
        self.controller.stop_all()

    def close(self) -> None:
        self.player.stop()
        self.controller.close()

    def _handle_status(self, message: str) -> None:
        if message.startswith("演奏进度"):
            message = "钢琴音色试听" + message[len("演奏进度"):]
        self.on_status(message)

    def _handle_finished(self) -> None:
        self.on_finished()

    def _handle_error(self, exc: Exception) -> None:
        self.on_error(exc)
