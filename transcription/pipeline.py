"""Public V2 transcription pipeline and JSON export helpers."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import bisect
from dataclasses import replace
from typing import Dict, Optional

from .analysis import analyze_audio
from .arranger import arrange_analysis, arrange_events, key_transpose
from .backends import is_midi_file, is_supported_input, transcribe_midi
from .models import (
    CancelledError,
    ProgressCallback,
    SourceMetadata,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)
from .quality import analyze_quality_audio, arrange_quality_analysis


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("转写已取消")


def _notify(callback: Optional[ProgressCallback], stage: str, fraction: float, message: str) -> None:
    if callback:
        callback(stage, max(0.0, min(1.0, float(fraction))), message)


def transcribe_draft(
    path: str,
    options: Optional[TranscriptionOptions] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
    workspace_dir: Optional[str] = None,
) -> TranscriptionResult:
    options = options or TranscriptionOptions()
    if not os.path.isfile(path):
        raise TranscriptionError(f"输入文件不存在：{path}")
    if not is_supported_input(path):
        raise TranscriptionError(f"不支持的输入格式：{os.path.splitext(path)[1] or '无扩展名'}")
    _check_cancel(cancel_event)
    source = SourceMetadata(
        platform="local", title=os.path.splitext(os.path.basename(path))[0], display_name=os.path.basename(path)
    )
    if is_midi_file(path) or options.mode == "midi":
        _notify(progress_cb, "decode", 0.0, "正在读取 MIDI")
        backend = transcribe_midi(path, cancel_event, progress_cb)
        midi_options = replace(options, mode="midi")
        notes, key, shift, octave, stats = arrange_events(backend.events, midi_options, backend.bpm, backend.beat_times_ms)
        stats.update({"duration_sec": round(float(backend.duration_sec), 3), "backend_event_count": len(backend.events)})
        return TranscriptionResult(
            events=list(backend.events), song_notes=notes, detected_key=key, bpm=float(backend.bpm),
            stats=stats, warnings=list(backend.warnings), engine="midi", source_file=os.path.basename(path),
            semitone_shift=shift, octave_shift=octave, beat_times_ms=list(backend.beat_times_ms), options=midi_options, source=source,
        )

    # `auto` deliberately remains the portable V2 engine. Quality mode is an
    # explicit user choice because its model is separately licensed/downloaded.
    if options.engine == "quality":
        _notify(progress_cb, "quality", 0.0, "正在启动高质量整曲扒谱")
        quality, raw_events = analyze_quality_audio(path, options, cancel_event, progress_cb)
        _check_cancel(cancel_event)
        _notify(progress_cb, "arrange", 0.0, "正在进行旋律优先 15 键编配")
        notes, key, shift, stats = arrange_quality_analysis(quality, options)
        if not notes:
            raise TranscriptionError("高质量分析完成，但没有生成可用的 15 键音符")
        stats.update({"duration_sec": round(quality.duration_sec, 3), "backend_event_count": len(raw_events)})
        _notify(progress_cb, "arrange", 1.0, "高质量 15 键编配完成")
        return TranscriptionResult(
            events=raw_events, song_notes=notes, detected_key=key, bpm=quality.bpm,
            stats=stats, warnings=[], engine="arrangement_v3_quality", source_file=os.path.basename(path),
            semitone_shift=shift, beat_times_ms=list(quality.beat_times_ms), options=options,
            source=source, quality_analysis=quality,
        )

    analysis, raw_events, artifact_root, model, warnings = analyze_audio(
        path, options, cancel_event, progress_cb, workspace_dir
    )
    _check_cancel(cancel_event)
    _notify(progress_cb, "arrange", 0.0, "正在编配 15 键钢琴谱")
    notes, arrange_stats, _roles = arrange_analysis(analysis, options)
    if not notes:
        shutil.rmtree(artifact_root, ignore_errors=True)
        raise TranscriptionError("分析完成，但没有生成可用的 15 键音符")
    stats: Dict[str, object] = dict(arrange_stats)
    stats.update({
        "duration_sec": round(analysis.duration_sec, 3),
        "backend_event_count": len(raw_events),
        "timingConfidence": round(analysis.tempo_map.confidence, 3),
        "keyConfidence": round(analysis.key_confidence, 3),
        "melodyConfidence": round(analysis.melody_confidence, 3),
        "harmonyConfidence": round(analysis.harmony_confidence, 3),
        "structureConfidence": round(analysis.structure_confidence, 3),
        "sectionCount": len(analysis.sections),
    })
    _notify(progress_cb, "arrange", 1.0, "15 键钢琴编配完成")
    return TranscriptionResult(
        events=raw_events, song_notes=notes, detected_key=analysis.detected_key, bpm=analysis.tempo_map.bpm,
        stats=stats, warnings=warnings, engine="arrangement_v2", source_file=os.path.basename(path),
        semitone_shift=analysis.semitone_shift, octave_shift=0, beat_times_ms=list(analysis.tempo_map.beat_times_ms),
        options=options, source=source, analysis=analysis, separation_model=model, artifact_root=artifact_root,
    )


def rearrange_draft(
    result: TranscriptionResult,
    options: TranscriptionOptions,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    _check_cancel(cancel_event)
    _notify(progress_cb, "arrange", 0.0, "正在重新编配")
    if result.quality_analysis is not None:
        if options.bpm_override is not None or options.meter != "auto":
            raise TranscriptionError("高质量草稿的 BPM 或拍号修正需要重新生成当前歌曲")
        notes, key, shift, stats = arrange_quality_analysis(result.quality_analysis, options)
        _check_cancel(cancel_event)
        _notify(progress_cb, "arrange", 1.0, "高质量草稿重新编配完成")
        return TranscriptionResult(
            events=list(result.events), song_notes=notes, detected_key=key, bpm=result.quality_analysis.bpm,
            stats={**result.stats, **stats}, warnings=list(result.warnings), engine="arrangement_v3_quality",
            source_file=result.source_file, semitone_shift=shift, beat_times_ms=list(result.quality_analysis.beat_times_ms),
            options=options, source=result.source, quality_analysis=result.quality_analysis,
            artifact_root=result.artifact_root,
        )
    if result.analysis is None:
        notes, key, shift, octave, stats = arrange_events(result.events, replace(options, mode="midi"), result.bpm, result.beat_times_ms)
        return TranscriptionResult(
            events=list(result.events), song_notes=notes, detected_key=key, bpm=result.bpm, stats={**result.stats, **stats},
            warnings=list(result.warnings), engine=result.engine, source_file=result.source_file, semitone_shift=shift,
            octave_shift=octave, beat_times_ms=list(result.beat_times_ms), options=replace(options, mode="midi"), source=result.source,
        )
    if options.bpm_override is not None or options.meter != "auto":
        raise TranscriptionError("修改 BPM 或拍号需要重新生成当前歌曲；编配预设、调性、八度和复音可即时重编配")
    notes, stats, _roles = arrange_analysis(result.analysis, options)
    _check_cancel(cancel_event)
    _notify(progress_cb, "arrange", 1.0, "重新编配完成")
    return TranscriptionResult(
        events=list(result.events), song_notes=notes, detected_key=options.source_key or result.analysis.detected_key,
        bpm=result.analysis.tempo_map.bpm, stats={**result.stats, **stats}, warnings=list(result.warnings),
        engine="arrangement_v2", source_file=result.source_file,
        semitone_shift=key_transpose(options.source_key or result.analysis.detected_key), octave_shift=0,
        beat_times_ms=list(result.analysis.tempo_map.beat_times_ms), options=options, source=result.source,
        analysis=result.analysis, separation_model=result.separation_model, artifact_root=result.artifact_root,
    )


def refine_region(
    result: TranscriptionResult,
    source_path: str,
    start_ms: int,
    end_ms: int,
    options: Optional[TranscriptionOptions] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    """Re-transcribe one bar-aligned V3 region while keeping the song grid fixed.

    The selected range gets one bar of audio context on each side, but only the
    requested bar range replaces symbolic notes.  This prevents a local repair
    from shifting the rest of an already reviewed score.
    """
    quality = result.quality_analysis
    if quality is None:
        raise TranscriptionError("片段精修仅适用于高质量草稿")
    if not os.path.isfile(source_path):
        raise TranscriptionError(f"精修源文件不存在：{source_path}")
    if end_ms <= start_ms:
        raise TranscriptionError("精修结束时间必须晚于开始时间")
    _check_cancel(cancel_event)
    active_options = options or result.options
    if active_options.engine != "quality":
        active_options = replace(active_options, engine="quality")
    if not active_options.rights_confirmed:
        raise TranscriptionError("片段精修同样需要确认输入音频的使用权利")

    bars = quality.bar_starts_ms or quality.beat_times_ms or [0]
    core_start = max((bar for bar in bars if bar <= start_ms), default=0)
    core_end = next((bar for bar in bars if bar >= end_ms), int(round(quality.duration_sec * 1000)))
    if core_end <= core_start:
        core_end = int(round(quality.duration_sec * 1000))
    left_index = max(0, bisect.bisect_left(bars, core_start) - 1)
    right_index = min(len(bars) - 1, bisect.bisect_right(bars, core_end))
    context_start, context_end = bars[left_index], bars[right_index]
    context_end = max(context_end, core_end)
    if progress_cb:
        progress_cb("refine", 0.0, "正在准备小节片段精修")

    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa / soundfile，无法进行片段精修") from exc
    root = result.artifact_root or tempfile.mkdtemp(prefix="sky-arrangement-v3-")
    os.makedirs(root, exist_ok=True)
    clip_path = os.path.join(root, f"refine-{core_start}-{core_end}.wav")
    try:
        waveform, sr = librosa.load(source_path, sr=16000, mono=True, offset=context_start / 1000.0, duration=max(.01, (context_end - context_start) / 1000.0))
        sf.write(clip_path, waveform, sr, subtype="PCM_16")
        local_options = replace(active_options, quality_model="medium")
        local, _raw = analyze_quality_audio(clip_path, local_options, cancel_event, progress_cb, beam_size=4)
    finally:
        try:
            os.unlink(clip_path)
        except OSError:
            pass
    offset = context_start
    replacement = [replace(note, start_ms=note.start_ms + offset, end_ms=note.end_ms + offset) for note in local.symbolic_notes if core_start <= note.start_ms + offset < core_end]
    retained = [note for note in quality.symbolic_notes if not (core_start <= note.start_ms < core_end)]
    updated = replace(quality, symbolic_notes=sorted(retained + replacement, key=lambda note: (note.start_ms, note.midi_pitch)), refined_regions=[*quality.refined_regions, (core_start, core_end)])
    notes, key, shift, stats = arrange_quality_analysis(updated, active_options, forced_shift=result.semitone_shift)
    if progress_cb:
        progress_cb("refine", 1.0, "片段精修完成，等待确认替换")
    return TranscriptionResult(
        events=list(result.events), song_notes=notes, detected_key=key, bpm=updated.bpm,
        stats={**result.stats, **stats}, warnings=list(result.warnings), engine="arrangement_v3_quality",
        source_file=result.source_file, semitone_shift=shift, beat_times_ms=list(updated.beat_times_ms),
        options=active_options, source=result.source, quality_analysis=updated, artifact_root=root,
    )


def _metadata(result: TranscriptionResult) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "schemaVersion": 3 if result.quality_analysis is not None else 2,
        "engine": result.engine,
        "sourceFile": result.source_file,
        "detectedKey": result.detected_key,
        "semitoneShift": result.semitone_shift,
        "arrangementPreset": result.options.arrangement_preset,
        "maxPolyphony": result.options.max_polyphony,
    }
    if result.source is not None:
        metadata["sourcePlatform"] = result.source.platform
    if result.analysis is not None:
        analysis = result.analysis
        metadata.update({
            "analysisVersion": 2,
            "meter": analysis.tempo_map.meter,
            "tempoMode": "bar_smooth",
            "leadSource": analysis.lead_source,
            "separationModel": result.separation_model,
            "usedMixFallback": analysis.used_mix_fallback,
            "confidence": {
                "timing": round(analysis.tempo_map.confidence, 3), "key": round(analysis.key_confidence, 3),
                "melody": round(analysis.melody_confidence, 3), "harmony": round(analysis.harmony_confidence, 3),
                "structure": round(analysis.structure_confidence, 3),
            },
        })
    if result.quality_analysis is not None:
        quality = result.quality_analysis
        metadata.update({
            "analysisVersion": 3,
            "qualityArrangerVersion": 4,
            "qualityModel": quality.model_name,
            "qualityDevice": quality.device,
            "timingBackend": quality.timing_backend,
            "quantization": quality.quantization,
            "meter": quality.meter,
            "refinedRegions": [list(region) for region in quality.refined_regions],
            "confidence": {"timing": round(quality.timing_confidence, 3)},
        })
    return metadata


def export_song_json(result: TranscriptionResult, output_path: str, song_name: str = "") -> None:
    if not result.song_notes:
        raise TranscriptionError("没有可导出的有效音符")
    output_path = os.path.abspath(output_path)
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    source = result.source
    song: Dict[str, object] = {
        "name": song_name or (source.title if source else "") or os.path.splitext(os.path.basename(result.source_file))[0], "author": "", "transcribedBy": "SkyAutoMusic",
        "bpm": int(round(result.bpm or 120.0)), "songNotes": list(result.song_notes),
        "_transcribe": _metadata(result), "_transcribe_stats": dict(result.stats),
    }
    if source and source.platform == "netease":
        song["author"] = " / ".join(source.artists)
        if source.source_id:
            song["_transcribe"]["sourceId"] = source.source_id  # type: ignore[index]
        if source.webpage_url:
            song["_transcribe"]["sourceUrl"] = source.webpage_url  # type: ignore[index]
    temporary_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", prefix=".sky-score-", dir=directory, delete=False) as handle:
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


def next_available_path(directory: str, stem: str, suffix: str = ".json") -> str:
    index = 0
    while True:
        value = stem if index == 0 else f"{stem} ({index})"
        candidate = os.path.join(directory, f"{value}{suffix}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}


def sanitize_filename_stem(value: str, fallback: str = "网易云歌曲") -> str:
    stem = _INVALID_FILENAME_CHARS.sub("_", str(value or ""))
    stem = re.sub(r"\s+", " ", stem).strip().rstrip(". ") or fallback
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = "_" + stem
    return stem[:180].rstrip(". ") or fallback


def suggested_output_stem(result: TranscriptionResult) -> str:
    source = result.source
    if source and source.platform == "netease":
        artist = " / ".join(source.artists)
        if source.source_id:
            artist = f"\u7f51\u6613\u4e91{source.source_id}" + (" / " + artist if artist else "")
        return sanitize_filename_stem(f"{source.title or '网易云歌曲'}{' - ' + artist if artist else ''}")
    return sanitize_filename_stem(os.path.splitext(os.path.basename(result.source_file))[0])


def cleanup_result_artifacts(result: TranscriptionResult) -> None:
    if result.artifact_root and os.path.isdir(result.artifact_root):
        shutil.rmtree(result.artifact_root, ignore_errors=True)
    result.artifact_root = ""
