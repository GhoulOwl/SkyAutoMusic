"""Signal analysis used by the V2 melody-and-accompaniment arranger."""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .backends import transcribe_monophonic, transcribe_polyphonic
from .models import (
    AnalysisDraft,
    CancelledError,
    ChordSpan,
    LeadSource,
    MelodyNote,
    Meter,
    NoteEvent,
    ProgressCallback,
    Section,
    TempoMap,
    TranscriptionError,
    TranscriptionOptions,
)
from .separation import separate_audio


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
CHORD_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
}


def _notify(callback: Optional[ProgressCallback], stage: str, fraction: float, text: str) -> None:
    if callback:
        callback(stage, max(0.0, min(1.0, float(fraction))), text)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("转写已取消")


def _median(values: Iterable[float], fallback: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)) and value > 0)
    if not ordered:
        return fallback
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _nearest_grid(grid: Sequence[int], time_ms: int) -> int:
    if not grid:
        return int(time_ms)
    import bisect
    position = bisect.bisect_left(grid, time_ms)
    choices = grid[max(0, position - 1): min(len(grid), position + 1)]
    return min(choices, key=lambda value: (abs(value - time_ms), value))


def _load_audio(path: str):
    try:
        import librosa
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa，无法分析音频") from exc
    try:
        return librosa.load(path, sr=22050, mono=True)
    except Exception as exc:
        raise TranscriptionError(f"音频解码失败：{exc}") from exc


def _tempo_candidates(raw_tempo: float) -> List[float]:
    values = [raw_tempo / 2.0, raw_tempo, raw_tempo * 2.0]
    return sorted({round(value, 4) for value in values if 40.0 <= value <= 220.0})


def _choose_tempo(raw_tempo: float, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    candidates = _tempo_candidates(raw_tempo)
    if not candidates:
        return 90.0
    # Librosa commonly returns an eighth-note pulse for ballads.  The score
    # should use the musical quarter pulse, while retaining genuinely fast pop.
    def score(value: float) -> Tuple[float, float]:
        conventional = 0.0 if 55.0 <= value <= 145.0 else -1.0
        centre = -abs(value - 104.0) / 100.0
        return conventional + centre, -value
    return max(candidates, key=score)


def _extend_beats(beats: List[float], period: float, duration_ms: int) -> List[float]:
    if not beats:
        beats = [0.0]
    beats = sorted(set(max(0.0, value) for value in beats))
    while beats[0] > period * 0.55:
        beats.insert(0, max(0.0, beats[0] - period))
        if beats[0] == 0.0:
            break
    if beats[0] > 1.0:
        beats.insert(0, 0.0)
    while beats[-1] < duration_ms + period:
        beats.append(beats[-1] + period)
    return beats


def build_tempo_map(y, sr: int, options: TranscriptionOptions) -> TempoMap:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa / numpy，无法分析节奏") from exc
    duration_ms = int(round(len(y) / max(1, sr) * 1000))
    harmonic, percussive = librosa.effects.hpss(y)
    raw_tempo, frames = librosa.beat.beat_track(y=percussive, sr=sr, trim=False)
    raw_tempo = float(np.asarray(raw_tempo).reshape(-1)[0]) if np.size(raw_tempo) else 90.0
    raw_beats = [float(value * 1000) for value in librosa.frames_to_time(frames, sr=sr)]
    bpm = _choose_tempo(raw_tempo, options.bpm_override)
    period = 60000.0 / bpm
    raw_period = _median((right - left for left, right in zip(raw_beats, raw_beats[1:])), period)
    ratio = max(1, int(round(period / max(1.0, raw_period))))
    beats = raw_beats[::ratio] if raw_beats else []
    beats = _extend_beats(beats, period, duration_ms)

    meter: Meter = options.meter if options.meter != "auto" else "4/4"
    beats_per_bar = {"4/4": 4, "3/4": 3, "6/8": 6}[meter]
    subdivisions = 2 if meter == "6/8" else 4
    smoothed: List[float] = []
    for start in range(0, len(beats) - 1, beats_per_bar):
        block = beats[start: start + beats_per_bar + 1]
        if len(block) < 2:
            continue
        local_period = _median(
            (right - left for left, right in zip(block, block[1:])), period
        )
        local_period = max(period * 0.65, min(period * 1.35, local_period))
        base = block[0]
        for offset in range(min(beats_per_bar, len(block) - 1)):
            value = base + offset * local_period
            if not smoothed or value > smoothed[-1] + 1.0:
                smoothed.append(value)
    if len(smoothed) < 2:
        smoothed = beats
    while smoothed[-1] < duration_ms + period:
        smoothed.append(smoothed[-1] + period)
    beat_times = tuple(int(round(value)) for value in smoothed)
    bars = tuple(beat_times[index] for index in range(0, len(beat_times), beats_per_bar))
    grid: List[int] = []
    for left, right in zip(beat_times, beat_times[1:]):
        step = (right - left) / subdivisions
        grid.extend(int(round(left + offset * step)) for offset in range(subdivisions))
    if beat_times:
        grid.append(beat_times[-1])
    grid = sorted(set(value for value in grid if 0 <= value <= duration_ms + int(period)))
    consistency = 1.0 - min(1.0, abs(raw_tempo - bpm) / max(40.0, raw_tempo))
    confidence = max(0.25, min(0.95, 0.45 + 0.5 * consistency if len(raw_beats) >= 8 else 0.35))
    return TempoMap(bpm=bpm, meter=meter, beat_times_ms=beat_times, bar_starts_ms=bars, grid_times_ms=tuple(grid), confidence=confidence)


def _profile_score(chroma, profile: Sequence[float], root: int) -> float:
    import numpy as np
    target = np.array([profile[(pitch - root) % 12] for pitch in range(12)], dtype=float)
    value = float(np.dot(chroma, target))
    denom = float(np.linalg.norm(chroma) * np.linalg.norm(target))
    return value / denom if denom else 0.0


def detect_key_from_audio(y, sr: int, override: Optional[str]) -> Tuple[str, float, int]:
    if override:
        return override, 1.0, _key_transpose(override)
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa / numpy，无法识别调性") from exc
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    vector = np.mean(chroma, axis=1)
    candidates: List[Tuple[float, int, str]] = []
    for root in range(12):
        candidates.append((_profile_score(vector, MAJOR_PROFILE, root), root, "major"))
        candidates.append((_profile_score(vector, MINOR_PROFILE, root), root, "minor"))
    candidates.sort(reverse=True)
    score, root, mode = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
    return f"{NOTE_NAMES[root]} {mode}", max(0.0, min(1.0, score - runner_up + 0.45)), _key_transpose(f"{NOTE_NAMES[root]} {mode}")


def _key_transpose(key_name: str) -> int:
    parts = str(key_name).strip().replace("♯", "#").replace("♭", "b").split()
    note = (parts[0] if parts else "C").upper()
    aliases = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    root = NOTE_NAMES.index(aliases.get(note, note)) if aliases.get(note, note) in NOTE_NAMES else 0
    minor = len(parts) > 1 and parts[1].lower() in ("minor", "min", "m", "小调")
    target = 9 if minor else 0
    raw = (target - root) % 12
    return raw - 12 if raw > 6 else raw


def _events_to_melody(
    events: Sequence[NoteEvent],
    grid: Sequence[int],
    source: LeadSource,
    previous_bias: bool = True,
) -> List[MelodyNote]:
    by_time: Dict[int, List[NoteEvent]] = {}
    for event in events:
        if not 40 <= event.midi_pitch <= 96:
            continue
        start = _nearest_grid(grid, event.start_ms)
        by_time.setdefault(start, []).append(event)
    selected: List[MelodyNote] = []
    previous_pitch: Optional[int] = None
    for start in sorted(by_time):
        pool = by_time[start]
        def score(event: NoteEvent) -> Tuple[float, float, float]:
            continuity = 0.0 if previous_pitch is None or not previous_bias else -0.035 * abs(event.midi_pitch - previous_pitch)
            lead_bonus = 0.018 * event.midi_pitch if source == "instrumental" else 0.0
            return event.strength + continuity + lead_bonus, event.duration_ms, event.midi_pitch
        event = max(pool, key=score)
        end = max(start + 1, _nearest_grid(grid, event.end_ms))
        note = MelodyNote(start, end, event.midi_pitch, max(0.0, min(1.0, event.strength)), source)
        if selected and note.midi_pitch == selected[-1].midi_pitch and note.start_ms - selected[-1].end_ms <= 90:
            previous = selected[-1]
            selected[-1] = MelodyNote(previous.start_ms, max(previous.end_ms, note.end_ms), previous.midi_pitch, max(previous.confidence, note.confidence), source)
        else:
            selected.append(note)
            previous_pitch = note.midi_pitch
    return selected


def extract_melody(
    vocal_path: str,
    fallback_path: str,
    tempo: TempoMap,
    cancel_event: Optional[threading.Event],
    progress_cb: Optional[ProgressCallback],
) -> Tuple[List[MelodyNote], LeadSource, float, List[NoteEvent], List[str]]:
    warnings: List[str] = []
    _check_cancel(cancel_event)
    _notify(progress_cb, "melody", 0.0, "正在提取人声主旋律")
    try:
        pyin = transcribe_monophonic(vocal_path, "normal", cancel_event, None, 40, 96)
        basic = transcribe_polyphonic(vocal_path, "normal", cancel_event, None, 40, 96)
        pitch_events = list(pyin.events) + list(basic.events)
    except TranscriptionError:
        pitch_events = []
    melody = _events_to_melody(pitch_events, tempo.grid_times_ms, "vocal")
    duration = max(1, (tempo.grid_times_ms[-1] if tempo.grid_times_ms else 1))
    coverage = sum(note.end_ms - note.start_ms for note in melody) / duration
    agreement = sum(note.confidence for note in melody) / max(1, len(melody))
    confidence = max(0.0, min(1.0, agreement * min(1.0, coverage * 3.0)))
    if confidence >= 0.28 and len(melody) >= 8:
        _notify(progress_cb, "melody", 1.0, "人声主旋律提取完成")
        return melody, "vocal", confidence, pitch_events, warnings

    _check_cancel(cancel_event)
    warnings.append("人声旋律置信度较低，已尝试器乐主线")
    _notify(progress_cb, "melody", 0.55, "正在提取器乐主线")
    try:
        fallback = transcribe_polyphonic(fallback_path, "normal", cancel_event, None, 48, 100)
        fallback_events = list(fallback.events)
    except TranscriptionError:
        fallback_events = []
    instrumental = _events_to_melody(fallback_events, tempo.grid_times_ms, "instrumental")
    instrumental_confidence = (
        sum(note.confidence for note in instrumental) / max(1, len(instrumental))
        if instrumental else 0.0
    )
    if instrumental_confidence >= 0.35 and len(instrumental) >= 6:
        _notify(progress_cb, "melody", 1.0, "器乐主线提取完成")
        return instrumental, "instrumental", instrumental_confidence, fallback_events, warnings
    warnings.append("未检测到可靠主旋律，已生成节奏化和弦伴奏")
    _notify(progress_cb, "melody", 1.0, "将使用和弦伴奏")
    return [], "chords_only", 0.0, fallback_events, warnings


def _window_chroma(chroma, frame_times, start_ms: int, end_ms: int):
    import numpy as np
    mask = (frame_times * 1000 >= start_ms) & (frame_times * 1000 < end_ms)
    if not np.any(mask):
        return np.mean(chroma, axis=1)
    return np.mean(chroma[:, mask], axis=1)


def _best_chord(vector, previous: Optional[ChordSpan]) -> Tuple[int, str, float]:
    import numpy as np
    total = float(np.sum(vector)) or 1.0
    ranked: List[Tuple[float, int, str]] = []
    for root in range(12):
        for quality, intervals in CHORD_INTERVALS.items():
            pcs = {(root + interval) % 12 for interval in intervals}
            inside = sum(float(vector[pitch]) for pitch in pcs) / total
            outside = sum(float(vector[pitch]) for pitch in range(12) if pitch not in pcs) / total
            score = inside - outside * 0.32
            if previous and previous.root_pc == root and previous.quality == quality:
                score += 0.08
            ranked.append((score, root, quality))
    ranked.sort(reverse=True)
    best = ranked[0]
    margin = best[0] - ranked[1][0]
    return best[1], best[2], max(0.0, min(1.0, 0.45 + margin * 2.2))


def infer_chords(y, sr: int, tempo: TempoMap, key_name: str, cancel_event: Optional[threading.Event], progress_cb: Optional[ProgressCallback]) -> Tuple[List[ChordSpan], float]:
    try:
        import librosa
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa，无法识别和弦") from exc
    _notify(progress_cb, "harmony", 0.0, "正在识别和弦进行")
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=1024)
    frame_times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=1024)
    beats = list(tempo.beat_times_ms)
    if len(beats) < 2:
        return [], 0.0
    spans: List[ChordSpan] = []
    for index in range(0, len(beats) - 1, 2):
        _check_cancel(cancel_event)
        start = beats[index]
        end = beats[min(len(beats) - 1, index + 2)]
        if end <= start:
            continue
        root, quality, confidence = _best_chord(_window_chroma(chroma, frame_times, start, end), spans[-1] if spans else None)
        if spans and confidence < 0.5:
            root, quality = spans[-1].root_pc, spans[-1].quality
        spans.append(ChordSpan(start, end, root, quality, root, confidence))
    if not spans:
        root = NOTE_NAMES.index(key_name.split()[0]) if key_name.split()[0] in NOTE_NAMES else 0
        spans = [ChordSpan(0, beats[-1], root, "maj", root, 0.2)]
    merged: List[ChordSpan] = []
    for span in spans:
        if merged and span.root_pc == merged[-1].root_pc and span.quality == merged[-1].quality:
            previous = merged[-1]
            merged[-1] = ChordSpan(previous.start_ms, span.end_ms, previous.root_pc, previous.quality, previous.bass_pc, max(previous.confidence, span.confidence))
        else:
            merged.append(span)
    confidence = sum(item.confidence for item in merged) / len(merged)
    _notify(progress_cb, "harmony", 1.0, "和弦进行识别完成")
    return merged, confidence


def infer_sections(y, sr: int, tempo: TempoMap, melody: Sequence[MelodyNote], chords: Sequence[ChordSpan], cancel_event: Optional[threading.Event], progress_cb: Optional[ProgressCallback]) -> Tuple[List[Section], float]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError("缺少 librosa / numpy，无法分析段落") from exc
    _notify(progress_cb, "structure", 0.0, "正在识别歌曲段落")
    meter_beats = {"4/4": 4, "3/4": 3, "6/8": 6}[tempo.meter]
    bars = list(tempo.bar_starts_ms)
    if len(bars) < 2:
        end = int(round(len(y) / sr * 1000))
        return [Section(0, max(1, end), "verse", 1)], 0.2
    rms = librosa.feature.rms(y=y, hop_length=1024)[0]
    rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=1024) * 1000
    energies: List[float] = []
    vocal_activity: List[float] = []
    for left, right in zip(bars, bars[1:] + [int(round(len(y) / sr * 1000))]):
        _check_cancel(cancel_event)
        mask = (rms_times >= left) & (rms_times < right)
        energies.append(float(np.mean(rms[mask])) if np.any(mask) else 0.0)
        vocal_activity.append(sum(max(0, min(note.end_ms, right) - max(note.start_ms, left)) for note in melody) / max(1, right - left))
    high = float(np.percentile(energies, 70)) if energies else 0.0
    low = float(np.percentile(energies, 35)) if energies else 0.0
    labels: List[Tuple[str, int]] = []
    for index, (energy, activity) in enumerate(zip(energies, vocal_activity)):
        if index == 0 and activity < 0.08:
            labels.append(("intro", 1))
        elif index == len(energies) - 1 and activity < 0.08:
            labels.append(("outro", 0))
        elif activity < 0.035:
            labels.append(("instrumental", 1 if energy >= low else 0))
        elif energy >= high:
            labels.append(("chorus", 3))
        elif energy >= low:
            labels.append(("verse", 1))
        else:
            labels.append(("verse", 1))
    sections: List[Section] = []
    start_index = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start_index]:
            role, energy = labels[start_index]
            start_ms = bars[start_index]
            end_ms = bars[index] if index < len(bars) else int(round(len(y) / sr * 1000))
            if sections and index - start_index < 4:
                previous = sections[-1]
                sections[-1] = Section(previous.start_ms, end_ms, previous.role, previous.energy, previous.repeat_group)
            else:
                sections.append(Section(start_ms, max(start_ms + 1, end_ms), role, energy, len(sections) + 1))
            start_index = index
    if not sections:
        sections = [Section(0, int(round(len(y) / sr * 1000)), "verse", 1)]
    _notify(progress_cb, "structure", 1.0, "歌曲段落识别完成")
    return sections, 0.55 if len(sections) > 1 else 0.35


def analyze_audio(
    path: str,
    options: TranscriptionOptions,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
    workspace_dir: Optional[str] = None,
) -> Tuple[AnalysisDraft, List[NoteEvent], str, str, List[str]]:
    """Run the non-destructive V2 analysis and return its temporary artifact root."""
    base_dir = os.path.abspath(workspace_dir) if workspace_dir else None
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    artifact_root = tempfile.mkdtemp(prefix="sky-arrangement-v2-", dir=base_dir)
    warnings: List[str] = []
    try:
        _notify(progress_cb, "decode", 0.0, "正在解码音频")
        # Demucs imports a legacy package used by some audio stacks.  Load and
        # execute it before librosa so the two CPU runtimes remain compatible.
        original = sr = None
        _notify(progress_cb, "decode", 1.0, "音频解码完成")
        _check_cancel(cancel_event)
        used_mix_fallback = False
        separation_model = ""
        try:
            separated = separate_audio(path, artifact_root, cancel_event, progress_cb)
            vocal_path, accompaniment_path = separated.vocal_path, separated.accompaniment_path
            separation_model = separated.model_name
        except CancelledError:
            raise
        except TranscriptionError as exc:
            # A per-song Demucs failure should not make the rest of the app unusable.
            warnings.append(f"人声/伴奏分离失败，已回退原曲分析：{exc}")
            vocal_path = accompaniment_path = path
            used_mix_fallback = True
        original, sr = _load_audio(path)
        _check_cancel(cancel_event)
        _notify(progress_cb, "timing", 0.0, "正在分析节拍与小节")
        tempo = build_tempo_map(original, sr, options)
        _notify(progress_cb, "timing", 1.0, "节拍与小节分析完成")
        key, key_confidence, shift = detect_key_from_audio(original, sr, options.source_key)
        melody, lead_source, melody_confidence, raw_events, melody_warnings = extract_melody(vocal_path, accompaniment_path, tempo, cancel_event, progress_cb)
        warnings.extend(melody_warnings)
        accompaniment, accompaniment_sr = _load_audio(accompaniment_path)
        chords, harmony_confidence = infer_chords(accompaniment, accompaniment_sr, tempo, key, cancel_event, progress_cb)
        sections, structure_confidence = infer_sections(original, sr, tempo, melody, chords, cancel_event, progress_cb)
        analysis = AnalysisDraft(
            duration_sec=len(original) / max(1, sr), tempo_map=tempo, detected_key=key,
            key_confidence=key_confidence, semitone_shift=shift, melody=melody, chords=chords,
            sections=sections, lead_source=lead_source, melody_confidence=melody_confidence,
            harmony_confidence=harmony_confidence, structure_confidence=structure_confidence,
            used_mix_fallback=used_mix_fallback, vocal_path=vocal_path if not used_mix_fallback else "",
            accompaniment_path=accompaniment_path if not used_mix_fallback else "",
        )
        return analysis, raw_events, artifact_root, separation_model, warnings
    except Exception:
        shutil.rmtree(artifact_root, ignore_errors=True)
        raise
