from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from dataclasses import replace
from typing import Dict, Optional

from .arranger import arrange_events
from .backends import (
    BackendOutput,
    estimate_audio_timing,
    is_midi_file,
    is_supported_input,
    transcribe_midi,
    transcribe_monophonic,
    transcribe_polyphonic,
)
from .models import (
    CancelledError,
    NoteEvent,
    ProgressCallback,
    SourceMetadata,
    StemKind,
    StemResult,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)
from .separation import separate_audio


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
    workspace_dir: Optional[str] = None,
) -> TranscriptionResult:
    options = options or TranscriptionOptions()
    if not os.path.isfile(path):
        raise TranscriptionError(f"输入文件不存在: {path}")
    if not is_supported_input(path):
        raise TranscriptionError(f"不支持的输入格式: {os.path.splitext(path)[1] or '无扩展名'}")
    _check_cancel(cancel_event)

    if not is_midi_file(path) and options.mode == "stem_fusion":
        return _transcribe_stem_fusion(
            path,
            options,
            cancel_event,
            progress_cb,
            workspace_dir,
        )
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


_STEM_TRANSCRIPTION_SETTINGS = {
    "vocals": ("monophonic", 36, 96),
    "piano": ("polyphonic", 21, 108),
    "bass": ("polyphonic", 24, 72),
    "guitar": ("polyphonic", 40, 96),
    "instrumental": ("polyphonic", 24, 108),
}


def _tag_stem_events(events, stem: StemKind):
    return [replace(event, stem=stem) for event in events]


def _transcribe_one_stem(
    stem: StemKind,
    audio_path: str,
    options: TranscriptionOptions,
    cancel_event: Optional[threading.Event],
    progress_cb: Optional[ProgressCallback],
) -> BackendOutput:
    engine, min_midi, max_midi = _STEM_TRANSCRIPTION_SETTINGS[stem]
    if engine == "monophonic":
        return transcribe_monophonic(
            audio_path,
            options.sensitivity,
            cancel_event,
            progress_cb,
            min_midi,
            max_midi,
        )
    return transcribe_polyphonic(
        audio_path,
        options.sensitivity,
        cancel_event,
        progress_cb,
        min_midi,
        max_midi,
    )


def _transcribe_stem_fusion(
    path: str,
    options: TranscriptionOptions,
    cancel_event: Optional[threading.Event],
    progress_cb: Optional[ProgressCallback],
    workspace_dir: Optional[str],
) -> TranscriptionResult:
    base_dir = os.path.abspath(workspace_dir) if workspace_dir else None
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    artifact_root = tempfile.mkdtemp(prefix="sky-six-stem-", dir=base_dir)
    try:
        separated = separate_audio(
            path,
            artifact_root,
            cancel_event,
            progress_cb,
        )
        _check_cancel(cancel_event)

        timing_warning = ""
        original_bpm, original_beats, _duration = estimate_audio_timing(
            path,
            cancel_event,
            None,
        )
        timing_maps = {
            "original": (float(original_bpm), list(original_beats)),
        }
        detected_drum_bpm = 0.0
        detected_drum_beats: list[int] = []
        try:
            drum_bpm, drum_beats, _duration = estimate_audio_timing(
                separated.paths["drums"],
                cancel_event,
                None,
            )
            if len(drum_beats) < 2:
                raise TranscriptionError("鼓轨没有检测到稳定节拍")
            detected_drum_bpm = float(drum_bpm)
            detected_drum_beats = list(drum_beats)
            timing_maps["drums"] = (float(drum_bpm), list(drum_beats))
        except CancelledError:
            raise
        except TranscriptionError as exc:
            timing_warning = f"鼓点节奏检测失败，已回退原曲：{exc}"
        timing_key = (
            "drums"
            if options.use_drum_timing and "drums" in timing_maps
            else "original"
        )
        bpm, beat_times = timing_maps[timing_key]

        stem_results: Dict[StemKind, StemResult] = {
            "drums": StemResult(
                kind="drums",
                audio_path=separated.paths["drums"],
                events=[],
                engine="librosa_beat",
                stats={
                    "bpm": round(detected_drum_bpm, 3),
                    "beat_count": len(detected_drum_beats),
                    "used_for_timing": bool(
                        options.use_drum_timing and "drums" in timing_maps
                    ),
                },
                warnings=[timing_warning] if timing_warning else [],
            )
        }
        all_events: list[NoteEvent] = []
        warnings = [timing_warning] if timing_warning else []
        stem_order = ("vocals", "piano", "bass", "guitar", "instrumental")
        total_stems = len(stem_order)
        for index, stem in enumerate(stem_order):
            _check_cancel(cancel_event)

            def stem_progress(
                _stage: str,
                fraction: float,
                message: str,
                *,
                _index=index,
                _stem=stem,
            ) -> None:
                if progress_cb:
                    combined = (_index + max(0.0, min(1.0, fraction))) / total_stems
                    progress_cb(
                        "transcribe",
                        combined,
                        f"{_stem}：{message}",
                    )

            try:
                backend = _transcribe_one_stem(
                    stem,
                    separated.paths[stem],
                    options,
                    cancel_event,
                    stem_progress,
                )
                tagged = _tag_stem_events(backend.events, stem)
                all_events.extend(tagged)
                stem_results[stem] = StemResult(
                    kind=stem,
                    audio_path=separated.paths[stem],
                    events=tagged,
                    engine=backend.engine,
                    stats={
                        "event_count": len(tagged),
                        "duration_sec": round(float(backend.duration_sec), 3),
                    },
                    warnings=list(backend.warnings),
                )
                warnings.extend(f"{stem}：{item}" for item in backend.warnings)
            except CancelledError:
                raise
            except TranscriptionError as exc:
                detail = f"{stem} 未识别到可用音符：{exc}"
                warnings.append(detail)
                stem_results[stem] = StemResult(
                    kind=stem,
                    audio_path=separated.paths[stem],
                    events=[],
                    engine=_STEM_TRANSCRIPTION_SETTINGS[stem][0],
                    stats={"event_count": 0},
                    warnings=[detail],
                )

        _check_cancel(cancel_event)
        if not all_events:
            raise TranscriptionError(
                "分轨完成，但所有有调声部都没有识别到可用音符"
            )
        all_events.sort(
            key=lambda item: (
                item.start_ms,
                item.midi_pitch,
                str(item.stem),
            )
        )
        _notify(progress_cb, "arrange", 0.0, "正在融合六轨并编配光遇15键乐谱")
        song_notes, detected_key, semitone_shift, octave_shift, arrange_stats = (
            arrange_events(
                all_events,
                options,
                bpm=bpm,
                beat_times_ms=beat_times,
            )
        )
        if not song_notes:
            raise TranscriptionError("六轨融合完成，但当前轨道设置没有产生有效音符")
        stats = dict(arrange_stats)
        stats.update({
            "duration_sec": round(float(separated.duration_sec), 3),
            "backend_event_count": len(all_events),
            "stem_event_counts": {
                stem: len(result.events)
                for stem, result in stem_results.items()
            },
        })
        _notify(progress_cb, "arrange", 1.0, "六轨融合与15键编配完成")
        return TranscriptionResult(
            events=all_events,
            song_notes=song_notes,
            detected_key=detected_key,
            bpm=float(bpm or 120.0),
            stats=stats,
            warnings=warnings,
            engine="stem_fusion",
            source_file=os.path.basename(path),
            semitone_shift=semitone_shift,
            octave_shift=octave_shift,
            beat_times_ms=list(beat_times),
            options=options,
            source=SourceMetadata(
                platform="local",
                title=os.path.splitext(os.path.basename(path))[0],
                display_name=os.path.basename(path),
            ),
            stems=stem_results,
            separation_model=separated.model_name,
            artifact_root=artifact_root,
            timing_maps=timing_maps,
        )
    except Exception:
        shutil.rmtree(artifact_root, ignore_errors=True)
        raise


def rearrange_draft(
    result: TranscriptionResult,
    options: TranscriptionOptions,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    _check_cancel(cancel_event)
    if result.engine == "midi":
        options = replace(options, mode="midi")
    bpm = result.bpm
    beat_times = result.beat_times_ms
    if result.engine == "stem_fusion" and result.timing_maps:
        timing_key = (
            "drums"
            if options.use_drum_timing and "drums" in result.timing_maps
            else "original"
        )
        bpm, beat_times = result.timing_maps.get(
            timing_key,
            (result.bpm, result.beat_times_ms),
        )
    _notify(progress_cb, "arrange", 0.0, "正在重新编配")
    notes, detected_key, semitone_shift, octave_shift, arrange_stats = arrange_events(
        result.events,
        options,
        bpm=bpm,
        beat_times_ms=beat_times,
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
        bpm=float(bpm),
        stats=stats,
        warnings=list(result.warnings),
        engine=result.engine,
        source_file=result.source_file,
        semitone_shift=semitone_shift,
        octave_shift=octave_shift,
        beat_times_ms=list(beat_times),
        options=options,
        source=result.source,
        stems=dict(result.stems),
        separation_model=result.separation_model,
        artifact_root=result.artifact_root,
        timing_maps=dict(result.timing_maps),
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
            "schemaVersion": 2,
            "engine": result.engine,
            "sourceFile": os.path.basename(result.source_file),
            "detectedKey": result.detected_key,
            "semitoneShift": int(result.semitone_shift),
            "octaveShift": int(result.octave_shift),
            "quantize": result.options.quantize,
            "maxPolyphony": int(result.options.max_polyphony),
            "repeatCleanup": result.options.repeat_cleanup,
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
    if result.options.mode == "stem_fusion":
        song["_transcribe"].update({
            "separationModel": result.separation_model,
            "enabledStems": list(result.options.enabled_stems),
            "useDrumTiming": bool(result.options.use_drum_timing),
            "fusionProfile": result.options.fusion_profile,
            "instrumentalPolicy": result.options.instrumental_policy,
            "stemEngines": {
                stem: stem_result.engine
                for stem, stem_result in result.stems.items()
            },
        })

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


def cleanup_result_artifacts(result: TranscriptionResult) -> None:
    root = str(result.artifact_root or "")
    if root and os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    result.artifact_root = ""

