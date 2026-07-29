from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import replace
from typing import Optional

from .arranger import arrange_events
from .backends import (
    is_midi_file,
    is_supported_input,
    transcribe_midi,
    transcribe_monophonic,
    transcribe_polyphonic,
)
from .models import (
    CancelledError,
    ProgressCallback,
    SourceMetadata,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)


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


def transcribe_draft(
    path: str,
    options: Optional[TranscriptionOptions] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    options = options or TranscriptionOptions()
    if not os.path.isfile(path):
        raise TranscriptionError(f"输入文件不存在: {path}")
    if not is_supported_input(path):
        raise TranscriptionError(f"不支持的输入格式: {os.path.splitext(path)[1] or '无扩展名'}")
    _check_cancel(cancel_event)

    if is_midi_file(path) or options.mode == "midi":
        backend = transcribe_midi(path, cancel_event, progress_cb)
        effective_options = replace(options, mode="midi")
    elif options.mode == "monophonic":
        backend = transcribe_monophonic(
            path, options.sensitivity, cancel_event, progress_cb
        )
        effective_options = options
    else:
        backend = transcribe_polyphonic(
            path, options.sensitivity, cancel_event, progress_cb
        )
        effective_options = replace(options, mode="polyphonic")

    _check_cancel(cancel_event)
    _notify(progress_cb, "arrange", 0.0, "正在编配光遇15键乐谱")
    song_notes, detected_key, semitone_shift, octave_shift, arrange_stats = arrange_events(
        backend.events,
        effective_options,
        bpm=backend.bpm,
        beat_times_ms=backend.beat_times_ms,
    )
    if not song_notes:
        raise TranscriptionError("转写完成，但没有可编配的有效音符")
    _check_cancel(cancel_event)
    stats = dict(arrange_stats)
    stats.update({
        "duration_sec": round(float(backend.duration_sec), 3),
        "backend_event_count": len(backend.events),
    })
    _notify(progress_cb, "arrange", 1.0, "15键编配完成")
    return TranscriptionResult(
        events=list(backend.events),
        song_notes=song_notes,
        detected_key=detected_key,
        bpm=float(backend.bpm or 120.0),
        stats=stats,
        warnings=list(backend.warnings),
        engine=backend.engine,
        source_file=os.path.basename(path),
        semitone_shift=semitone_shift,
        octave_shift=octave_shift,
        beat_times_ms=list(backend.beat_times_ms),
        options=effective_options,
        source=SourceMetadata(
            platform="local",
            title=os.path.splitext(os.path.basename(path))[0],
            display_name=os.path.basename(path),
        ),
    )


def rearrange_draft(
    result: TranscriptionResult,
    options: TranscriptionOptions,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    _check_cancel(cancel_event)
    if result.engine == "midi":
        options = replace(options, mode="midi")
    _notify(progress_cb, "arrange", 0.0, "正在重新编配")
    notes, detected_key, semitone_shift, octave_shift, arrange_stats = arrange_events(
        result.events,
        options,
        bpm=result.bpm,
        beat_times_ms=result.beat_times_ms,
    )
    if not notes:
        raise TranscriptionError("当前参数没有产生可用音符")
    stats = dict(result.stats)
    stats.update(arrange_stats)
    _notify(progress_cb, "arrange", 1.0, "重新编配完成")
    return TranscriptionResult(
        events=list(result.events),
        song_notes=notes,
        detected_key=detected_key,
        bpm=result.bpm,
        stats=stats,
        warnings=list(result.warnings),
        engine=result.engine,
        source_file=result.source_file,
        semitone_shift=semitone_shift,
        octave_shift=octave_shift,
        beat_times_ms=list(result.beat_times_ms),
        options=options,
        source=result.source,
    )


def export_song_json(
    result: TranscriptionResult,
    output_path: str,
    song_name: Optional[str] = None,
) -> str:
    if not result.song_notes:
        raise TranscriptionError("不能导出空乐谱")
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    source = result.source
    if not song_name:
        if source is not None and source.platform == "netease" and source.title:
            song_name = source.title
        else:
            song_name = os.path.splitext(os.path.basename(output_path))[0]
    song = {
        "name": song_name,
        "transcribedBy": "SkyAutoMusic",
        "bpm": int(round(result.bpm or 120.0)),
        "songNotes": result.song_notes,
        "_transcribe": {
            "schemaVersion": 1,
            "engine": result.engine,
            "sourceFile": os.path.basename(result.source_file),
            "detectedKey": result.detected_key,
            "semitoneShift": int(result.semitone_shift),
            "octaveShift": int(result.octave_shift),
            "quantize": result.options.quantize,
            "maxPolyphony": int(result.options.max_polyphony),
        },
        "_transcribe_stats": result.stats,
    }
    if source is not None:
        song["_transcribe"]["sourcePlatform"] = source.platform
        if source.source_id:
            song["_transcribe"]["sourceId"] = source.source_id
        if source.webpage_url:
            song["_transcribe"]["sourceUrl"] = source.webpage_url
        if source.display_name:
            song["_transcribe"]["sourceFile"] = source.display_name
        if source.platform == "netease" and source.artists:
            song["author"] = " / ".join(source.artists)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=".sky-score-",
            dir=output_dir,
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump([song], handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        raise
    return output_path


def next_available_path(directory: str, stem: str, suffix: str = ".json") -> str:
    candidate = os.path.join(directory, stem + suffix)
    if not os.path.exists(candidate):
        return candidate
    index = 1
    while True:
        candidate = os.path.join(directory, f"{stem} ({index}){suffix}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def sanitize_filename_stem(value: str, fallback: str = "网易云歌曲") -> str:
    stem = _INVALID_FILENAME_CHARS.sub("_", str(value or ""))
    stem = re.sub(r"\s+", " ", stem).strip().rstrip(". ")
    if not stem:
        stem = fallback
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = "_" + stem
    return stem[:180].rstrip(". ") or fallback


def suggested_output_stem(result: TranscriptionResult) -> str:
    source = result.source
    if source is not None and source.platform == "netease":
        title = source.title or "网易云歌曲"
        artist = " / ".join(source.artists)
        artist_part = f" - {artist}" if artist else ""
        id_part = f" [网易云{source.source_id}]" if source.source_id else ""
        return sanitize_filename_stem(f"{title}{artist_part}{id_part}")
    source_name = source.display_name if source and source.display_name else result.source_file
    return sanitize_filename_stem(os.path.splitext(os.path.basename(source_name))[0])

