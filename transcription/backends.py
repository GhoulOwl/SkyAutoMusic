from __future__ import annotations

import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .models import (
    CancelledError,
    NoteEvent,
    ProgressCallback,
    TranscriptionError,
)


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
MIDI_EXTS = {".mid", ".midi"}
DEFAULT_SR = 22050
CHUNK_SECONDS = 30.0
CHUNK_OVERLAP_SECONDS = 1.0

_SENSITIVITY = {
    "low": {"onset": 0.60, "frame": 0.40, "voiced": 0.70},
    "normal": {"onset": 0.50, "frame": 0.30, "voiced": 0.50},
    "high": {"onset": 0.40, "frame": 0.20, "voiced": 0.30},
}


@dataclass
class BackendOutput:
    events: List[NoteEvent]
    bpm: float
    beat_times_ms: List[int]
    engine: str
    duration_sec: float
    warnings: List[str] = field(default_factory=list)


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


def is_midi_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MIDI_EXTS


def is_supported_input(path: str) -> bool:
    return is_audio_file(path) or is_midi_file(path)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("转写已取消")


def _notify(
    progress_cb: Optional[ProgressCallback],
    stage: str,
    fraction: float,
    message: str,
) -> None:
    if progress_cb:
        progress_cb(stage, max(0.0, min(1.0, float(fraction))), message)


def _normalized_interval_ms(start_ms: float, end_ms: float) -> Tuple[int, int]:
    """Normalize third-party timestamps before constructing a strict NoteEvent."""
    start_value = max(0, int(round(float(start_ms))))
    end_value = max(start_value + 1, int(round(float(end_ms))))
    return start_value, end_value


def _load_audio(path: str, progress_cb: Optional[ProgressCallback]):
    _notify(progress_cb, "decode", 0.0, "正在解码音频")
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa / numpy，无法解析音频") from exc
    try:
        y, sr = librosa.load(path, sr=DEFAULT_SR, mono=True)
    except Exception as exc:
        raise TranscriptionError(f"音频解码失败: {exc}") from exc
    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        raise TranscriptionError("音频为空或没有可识别的声音")
    _notify(progress_cb, "decode", 1.0, "音频解码完成")
    return y, int(sr)


def _estimate_beats(y, sr: int) -> Tuple[float, List[int]]:
    try:
        import librosa
        import numpy as np

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = float(np.asarray(tempo).reshape(-1)[0])
        if not math.isfinite(tempo_value) or tempo_value <= 0:
            tempo_value = 120.0
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return tempo_value, [int(round(float(value) * 1000)) for value in beat_times]
    except Exception:
        return 120.0, []


def _chunk_regions(duration_sec: float) -> List[Tuple[float, float, float, float]]:
    regions: List[Tuple[float, float, float, float]] = []
    core_start = 0.0
    while core_start < duration_sec:
        core_end = min(duration_sec, core_start + CHUNK_SECONDS)
        read_start = max(0.0, core_start - (CHUNK_OVERLAP_SECONDS if core_start else 0.0))
        read_end = min(
            duration_sec,
            core_end + (CHUNK_OVERLAP_SECONDS if core_end < duration_sec else 0.0),
        )
        regions.append((core_start, core_end, read_start, read_end))
        core_start = core_end
    return regions


def _owned_by_core(event: NoteEvent, core_start: float, core_end: float, final: bool) -> bool:
    start_sec = event.start_ms / 1000.0
    if final:
        return core_start <= start_sec <= core_end
    return core_start <= start_sec < core_end


def transcribe_midi(
    path: str,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> BackendOutput:
    _check_cancel(cancel_event)
    _notify(progress_cb, "decode", 0.0, "正在读取 MIDI")
    try:
        import pretty_midi
    except ImportError as exc:
        raise TranscriptionError("缺少 pretty_midi，无法导入 MIDI") from exc
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception as exc:
        raise TranscriptionError(f"MIDI 解析失败: {exc}") from exc

    events: List[NoteEvent] = []
    ignored_drums = 0
    for instrument in midi.instruments:
        _check_cancel(cancel_event)
        if instrument.is_drum:
            ignored_drums += len(instrument.notes)
            continue
        for note in instrument.notes:
            start_ms, end_ms = _normalized_interval_ms(
                note.start * 1000,
                note.end * 1000,
            )
            events.append(NoteEvent(
                start_ms=start_ms,
                end_ms=end_ms,
                midi_pitch=int(note.pitch),
                strength=max(0.0, min(1.0, note.velocity / 127.0)),
                source="midi",
            ))

    if not events:
        raise TranscriptionError("MIDI 中没有可用的非鼓轨音符")
    events.sort(key=lambda item: (item.start_ms, item.midi_pitch))
    try:
        _tempo_times, tempi = midi.get_tempo_changes()
        bpm = float(tempi[0]) if len(tempi) else 120.0
    except Exception:
        bpm = 120.0
    try:
        beat_times = [int(round(float(value) * 1000)) for value in midi.get_beats()]
    except Exception:
        beat_times = []
    warnings = [f"已忽略 {ignored_drums} 个鼓轨音符"] if ignored_drums else []
    duration = max(event.end_ms for event in events) / 1000.0
    _notify(progress_cb, "decode", 1.0, "MIDI 读取完成")
    return BackendOutput(events, bpm, beat_times, "midi", duration, warnings)


def _run_length_encode(values: Sequence[Optional[int]]) -> List[Tuple[Optional[int], int, int]]:
    if not values:
        return []
    runs: List[Tuple[Optional[int], int, int]] = []
    start = 0
    value = values[0]
    for index in range(1, len(values)):
        if values[index] != value:
            runs.append((value, start, index))
            value = values[index]
            start = index
    runs.append((value, start, len(values)))
    return runs


def _stabilize_pitch_frames(values: Sequence[Optional[int]], min_frames: int = 3) -> List[Optional[int]]:
    stable = list(values)
    runs = _run_length_encode(stable)
    for index, (value, start, end) in enumerate(runs):
        if value is None or end - start >= min_frames:
            continue
        previous = runs[index - 1][0] if index else None
        following = runs[index + 1][0] if index + 1 < len(runs) else None
        replacement = previous if previous is not None else following
        for frame in range(start, end):
            stable[frame] = replacement
    return stable


def _monophonic_chunk(
    y,
    sr: int,
    voiced_threshold: float,
    min_midi: int = 36,
    max_midi: int = 96,
) -> List[NoteEvent]:
    import librosa
    import numpy as np

    hop_length = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.midi_to_hz(min_midi),
        fmax=librosa.midi_to_hz(max_midi),
        sr=sr,
        hop_length=hop_length,
    )
    if f0 is None:
        return []
    values: List[Optional[int]] = []
    for frequency, is_voiced, probability in zip(f0, voiced_flag, voiced_prob):
        valid = bool(is_voiced) and math.isfinite(float(frequency))
        valid = valid and float(probability) >= voiced_threshold
        values.append(int(round(float(librosa.hz_to_midi(frequency)))) if valid else None)
    values = _stabilize_pitch_frames(values, min_frames=3)
    frame_times = librosa.frames_to_time(
        np.arange(len(values)), sr=sr, hop_length=hop_length
    )
    try:
        onset_times = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, units="time", backtrack=True
        )
    except Exception:
        onset_times = []

    events: List[NoteEvent] = []
    for value, start, end in _run_length_encode(values):
        if value is None or end - start < 3:
            continue
        start_sec = float(frame_times[start])
        end_index = min(end, len(frame_times) - 1)
        end_sec = float(frame_times[end_index]) if end_index < len(frame_times) else len(y) / sr
        if end_sec <= start_sec:
            end_sec = start_sec + hop_length / sr
        if len(onset_times):
            closest = min(onset_times, key=lambda item: abs(float(item) - start_sec))
            onset_sec = max(0.0, float(closest))
            # Backtracked onsets can occasionally land after a very short pYIN
            # segment.  Do not turn that segment into a reversed interval.
            if abs(onset_sec - start_sec) <= 0.08 and onset_sec < end_sec:
                start_sec = onset_sec
        strength_values = [
            float(probability)
            for probability in voiced_prob[start:end]
            if math.isfinite(float(probability))
        ]
        strength = sum(strength_values) / len(strength_values) if strength_values else 0.5
        start_ms, end_ms = _normalized_interval_ms(
            start_sec * 1000,
            end_sec * 1000,
        )
        events.append(NoteEvent(
            start_ms=start_ms,
            end_ms=end_ms,
            midi_pitch=int(value),
            strength=strength,
            source="pyin",
        ))
    return events


def transcribe_monophonic(
    path: str,
    sensitivity: str = "normal",
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
    min_midi: int = 36,
    max_midi: int = 96,
) -> BackendOutput:
    y, sr = _load_audio(path, progress_cb)
    _check_cancel(cancel_event)
    bpm, beat_times = _estimate_beats(y, sr)
    duration = len(y) / sr
    regions = _chunk_regions(duration)
    events: List[NoteEvent] = []
    voiced_threshold = _SENSITIVITY[sensitivity]["voiced"]
    for index, (core_start, core_end, read_start, read_end) in enumerate(regions):
        _check_cancel(cancel_event)
        _notify(
            progress_cb,
            "transcribe",
            index / max(1, len(regions)),
            f"正在识别单旋律 ({index + 1}/{len(regions)})",
        )
        start_sample = int(round(read_start * sr))
        end_sample = int(round(read_end * sr))
        if min_midi == 36 and max_midi == 96:
            local = _monophonic_chunk(
                y[start_sample:end_sample],
                sr,
                voiced_threshold,
            )
        else:
            local = _monophonic_chunk(
                y[start_sample:end_sample],
                sr,
                voiced_threshold,
                min_midi,
                max_midi,
            )
        final = index == len(regions) - 1
        for event in local:
            shifted = event.shifted(int(round(read_start * 1000)))
            if _owned_by_core(shifted, core_start, core_end, final):
                events.append(shifted)
    _check_cancel(cancel_event)
    if not events:
        raise TranscriptionError("没有识别到稳定的单旋律音符")
    events.sort(key=lambda item: (item.start_ms, item.midi_pitch))
    _notify(progress_cb, "transcribe", 1.0, "单旋律识别完成")
    return BackendOutput(events, bpm, beat_times, "pyin", duration)


class BasicPitchBackend:
    """复用同一个 Basic Pitch 模型，避免批量处理时反复加载。"""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model
        except ImportError as exc:
            raise TranscriptionError(
                "缺少 Basic Pitch / ONNX Runtime，无法使用复音高质量模式"
            ) from exc
        try:
            self._model = Model(ICASSP_2022_MODEL_PATH)
        except Exception as exc:
            raise TranscriptionError(f"Basic Pitch 模型加载失败: {exc}") from exc
        return self._model

    def transcribe(
        self,
        path: str,
        sensitivity: str = "normal",
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[ProgressCallback] = None,
        min_midi: int = 36,
        max_midi: int = 96,
    ) -> BackendOutput:
        y, sr = _load_audio(path, progress_cb)
        _check_cancel(cancel_event)
        bpm, beat_times = _estimate_beats(y, sr)
        duration = len(y) / sr
        regions = _chunk_regions(duration)
        settings = _SENSITIVITY[sensitivity]
        events: List[NoteEvent] = []

        try:
            import librosa
            import soundfile as sf
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise TranscriptionError(
                "缺少 Basic Pitch、soundfile 或 librosa，无法使用复音模式"
            ) from exc
        model = self._load_model()
        min_frequency = float(librosa.midi_to_hz(min_midi))
        max_frequency = float(librosa.midi_to_hz(max_midi))

        with tempfile.TemporaryDirectory(prefix="sky-transcribe-") as temp_dir:
            for index, (core_start, core_end, read_start, read_end) in enumerate(regions):
                _check_cancel(cancel_event)
                _notify(
                    progress_cb,
                    "transcribe",
                    index / max(1, len(regions)),
                    f"正在识别复音 ({index + 1}/{len(regions)})",
                )
                start_sample = int(round(read_start * sr))
                end_sample = int(round(read_end * sr))
                chunk_path = os.path.join(temp_dir, f"chunk-{index:04d}.wav")
                sf.write(chunk_path, y[start_sample:end_sample], sr, subtype="PCM_16")
                try:
                    with self._lock:
                        _model_output, _midi, note_events = predict(
                            chunk_path,
                            model,
                            onset_threshold=settings["onset"],
                            frame_threshold=settings["frame"],
                            minimum_note_length=127.7,
                            minimum_frequency=min_frequency,
                            maximum_frequency=max_frequency,
                            multiple_pitch_bends=False,
                        )
                except Exception as exc:
                    raise TranscriptionError(f"Basic Pitch 推理失败: {exc}") from exc

                final = index == len(regions) - 1
                for start_sec, end_sec, pitch, strength, _pitch_bends in note_events:
                    try:
                        absolute_start = (float(start_sec) + read_start) * 1000
                        absolute_end = (float(end_sec) + read_start) * 1000
                        if not (
                            math.isfinite(absolute_start)
                            and math.isfinite(absolute_end)
                            and math.isfinite(float(strength))
                        ):
                            continue
                        start_ms, end_ms = _normalized_interval_ms(
                            absolute_start,
                            absolute_end,
                        )
                        midi_pitch = int(round(float(pitch)))
                        if not 0 <= midi_pitch <= 127:
                            continue
                    except (TypeError, ValueError, OverflowError):
                        continue
                    event = NoteEvent(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        midi_pitch=midi_pitch,
                        strength=max(0.0, min(1.0, float(strength))),
                        source="basic_pitch",
                    )
                    if _owned_by_core(event, core_start, core_end, final):
                        events.append(event)

        _check_cancel(cancel_event)
        if not events:
            raise TranscriptionError("没有识别到复音音符")
        events.sort(key=lambda item: (item.start_ms, item.midi_pitch))
        _notify(progress_cb, "transcribe", 1.0, "复音识别完成")
        return BackendOutput(events, bpm, beat_times, "basic_pitch", duration)


_BASIC_PITCH_BACKEND = BasicPitchBackend()


def transcribe_polyphonic(
    path: str,
    sensitivity: str = "normal",
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
    min_midi: int = 36,
    max_midi: int = 96,
) -> BackendOutput:
    return _BASIC_PITCH_BACKEND.transcribe(
        path,
        sensitivity,
        cancel_event,
        progress_cb,
        min_midi,
        max_midi,
    )


def estimate_audio_timing(
    path: str,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[float, List[int], float]:
    """Estimate one shared beat map without producing pitched note events."""

    y, sr = _load_audio(path, progress_cb)
    _check_cancel(cancel_event)
    bpm, beat_times = _estimate_beats(y, sr)
    duration = len(y) / max(1, sr)
    return bpm, beat_times, duration
