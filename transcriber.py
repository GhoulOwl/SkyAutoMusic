"""向后兼容的扒谱入口。

新实现位于 :mod:`transcription`。本模块保留旧版 ``Transcriber``、``run``、
``write_song_json`` 与 ``_midi_to_key`` 接口，避免 GUI 和外部调用失效。
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from transcription.arranger import SKY_MIDI, fold_to_sky_range, nearest_sky_pitch
from transcription.backends import (
    AUDIO_EXTS,
    MIDI_EXTS,
    DEFAULT_SR,
    is_audio_file,
    is_midi_file,
    is_supported_input,
)
from transcription.models import (
    CancelledError,
    TranscriptionOptions,
    TranscriptionResult,
)
from transcription.pipeline import export_song_json, transcribe_draft


class TranscribeStats:
    """兼容旧版属性访问的统计对象。"""

    __slots__ = (
        "onset_count",
        "note_count",
        "clamped_low",
        "clamped_high",
        "duration_sec",
        "_extra",
    )

    def __init__(self, values: Optional[Dict[str, Any]] = None):
        values = dict(values or {})
        self.onset_count = int(values.get("raw_event_count", values.get("onset_count", 0)))
        self.note_count = int(values.get("arranged_note_count", values.get("note_count", 0)))
        self.clamped_low = int(values.get("clamped_low", 0))
        self.clamped_high = int(values.get("clamped_high", 0))
        self.duration_sec = float(values.get("duration_sec", 0.0))
        self._extra = values

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self._extra)
        result.update({
            "onset_count": self.onset_count,
            "note_count": self.note_count,
            "clamped_low": self.clamped_low,
            "clamped_high": self.clamped_high,
            "duration_sec": round(self.duration_sec, 3),
        })
        return result


class Transcriber:
    NUM_KEYS = 15
    DEFAULT_SR = DEFAULT_SR

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        midi_root: int = 60,
        options: Optional[TranscriptionOptions] = None,
    ):
        # sr/midi_root 为兼容旧构造参数保留；新流水线使用固定 22050Hz 与 C4-C6 映射。
        self.sr = int(sr)
        self.midi_root = int(midi_root)
        self.options = options or TranscriptionOptions()
        self.last_result: Optional[TranscriptionResult] = None

    def transcribe(
        self,
        input_path: str,
        options: Optional[TranscriptionOptions] = None,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], TranscribeStats]:
        result = transcribe_draft(
            input_path,
            options=options or self.options,
            cancel_event=cancel_event,
            progress_cb=progress_cb,
        )
        self.last_result = result
        return list(result.song_notes), TranscribeStats(result.stats)

    def _midi_to_key(self, midi: float) -> Tuple[str, int]:
        """将 MIDI 音高映射到 C4-C6 自然音键位。

        越界音先按八度折叠，升降音取最近自然音。返回值第二项仍沿用旧含义：
        原始音低于范围为 -1，高于范围为 1，范围内为 0。
        """
        rounded = int(round(float(midi)))
        range_state = -1 if rounded < SKY_MIDI[0] else (1 if rounded > SKY_MIDI[-1] else 0)
        folded, _ = fold_to_sky_range(rounded)
        sky_pitch, _ = nearest_sky_pitch(folded)
        return f"1Key{SKY_MIDI.index(sky_pitch)}", range_state

    def transcribe_to_song(
        self,
        input_path: str,
        song_name: Optional[str] = None,
        bpm: Optional[int] = None,
        options: Optional[TranscriptionOptions] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        result = transcribe_draft(
            input_path,
            options=options or self.options,
            cancel_event=cancel_event,
        )
        self.last_result = result
        name = song_name or os.path.splitext(os.path.basename(input_path))[0]
        return {
            "name": name,
            "transcribedBy": "SkyAutoMusic",
            "bpm": int(round(bpm if bpm is not None else result.bpm)),
            "songNotes": list(result.song_notes),
            "_transcribe": {
                "schemaVersion": 1,
                "engine": result.engine,
                "sourceFile": result.source_file,
                "detectedKey": result.detected_key,
                "semitoneShift": result.semitone_shift,
                "octaveShift": result.octave_shift,
                "quantize": result.options.quantize,
                "maxPolyphony": result.options.max_polyphony,
            },
            "_transcribe_stats": dict(result.stats),
        }

    def write_song_json(
        self,
        input_path: str,
        output_dir: str,
        song_name: Optional[str] = None,
        bpm: Optional[int] = None,
        options: Optional[TranscriptionOptions] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        result = transcribe_draft(
            input_path,
            options=options or self.options,
            cancel_event=cancel_event,
        )
        if bpm is not None:
            result.bpm = float(bpm)
        self.last_result = result
        name = song_name or os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, name + ".json")
        export_song_json(result, output_path, name)
        song = self.transcribe_to_song_dict(result, name)
        return output_path, song

    @staticmethod
    def transcribe_to_song_dict(
        result: TranscriptionResult,
        song_name: str,
    ) -> Dict[str, Any]:
        return {
            "name": song_name,
            "transcribedBy": "SkyAutoMusic",
            "bpm": int(round(result.bpm or 120.0)),
            "songNotes": list(result.song_notes),
            "_transcribe": {
                "schemaVersion": 1,
                "engine": result.engine,
                "sourceFile": result.source_file,
                "detectedKey": result.detected_key,
                "semitoneShift": result.semitone_shift,
                "octaveShift": result.octave_shift,
                "quantize": result.options.quantize,
                "maxPolyphony": result.options.max_polyphony,
            },
            "_transcribe_stats": dict(result.stats),
        }

    def run(
        self,
        files: Iterable[str],
        output_dir: str,
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
        bpm: Optional[int] = None,
        options: Optional[TranscriptionOptions] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[Dict[str, Any]]:
        inputs = [path for path in files if is_supported_input(path)]
        total = len(inputs)
        results: List[Dict[str, Any]] = []
        if not inputs:
            if progress_cb:
                progress_cb("", 1.0, "未选择任何音频或 MIDI 文件")
            return results

        for index, path in enumerate(inputs):
            if cancel_event is not None and cancel_event.is_set():
                break
            song_name = os.path.splitext(os.path.basename(path))[0]

            def pipeline_progress(stage: str, fraction: float, message: str) -> None:
                if progress_cb:
                    stage_start, stage_weight = {
                        "decode": (0.00, 0.10),
                        "transcribe": (0.10, 0.75),
                        "arrange": (0.85, 0.15),
                    }.get(stage, (0.0, 1.0))
                    file_fraction = stage_start + stage_weight * max(
                        0.0, min(1.0, fraction)
                    )
                    overall = (index + file_fraction) / total
                    progress_cb(path, overall, f"({index + 1}/{total}) {message}")

            try:
                result = transcribe_draft(
                    path,
                    options=options or self.options,
                    cancel_event=cancel_event,
                    progress_cb=pipeline_progress,
                )
                if bpm is not None:
                    result.bpm = float(bpm)
                output_path = os.path.join(output_dir, song_name + ".json")
                export_song_json(result, output_path, song_name)
                self.last_result = result
                results.append({
                    "input": path,
                    "output": output_path,
                    "ok": True,
                    "cancelled": False,
                    "error": None,
                    "stats": dict(result.stats),
                    "song_name": song_name,
                })
            except CancelledError as exc:
                results.append({
                    "input": path,
                    "output": None,
                    "ok": False,
                    "cancelled": True,
                    "error": str(exc),
                    "stats": None,
                    "song_name": song_name,
                })
                break
            except Exception as exc:
                results.append({
                    "input": path,
                    "output": None,
                    "ok": False,
                    "cancelled": False,
                    "error": str(exc),
                    "stats": None,
                    "song_name": song_name,
                })

        if progress_cb:
            ok_count = sum(1 for item in results if item["ok"])
            progress_cb("", 1.0, f"完成 {ok_count}/{total}")
        return results


__all__ = [
    "AUDIO_EXTS",
    "MIDI_EXTS",
    "CancelledError",
    "TranscribeStats",
    "Transcriber",
    "TranscriptionOptions",
    "is_audio_file",
    "is_midi_file",
    "is_supported_input",
]
