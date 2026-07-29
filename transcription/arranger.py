from __future__ import annotations

import bisect
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import NoteEvent, TranscriptionOptions


SKY_MIDI: Tuple[int, ...] = (
    60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84
)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _weighted_pitch_histogram(events: Sequence[NoteEvent]) -> List[float]:
    hist = [0.0] * 12
    for event in events:
        hist[event.midi_pitch % 12] += max(1.0, event.weight)
    return hist


def _profile_score(hist: Sequence[float], profile: Sequence[float], root: int) -> float:
    # 将 profile 的 0 音级旋转到 root 后与输入直方图做余弦相似度。
    rotated = [profile[(pc - root) % 12] for pc in range(12)]
    dot = sum(a * b for a, b in zip(hist, rotated))
    norm_a = math.sqrt(sum(a * a for a in hist))
    norm_b = math.sqrt(sum(b * b for b in rotated))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def detect_key(events: Sequence[NoteEvent]) -> str:
    if not events:
        return "C major"
    hist = _weighted_pitch_histogram(events)
    candidates: List[Tuple[float, int, str]] = []
    for root in range(12):
        candidates.append((_profile_score(hist, MAJOR_PROFILE, root), root, "major"))
        candidates.append((_profile_score(hist, MINOR_PROFILE, root), root, "minor"))
    _, root, mode = max(candidates, key=lambda item: (item[0], item[2] == "major", -item[1]))
    return f"{NOTE_NAMES[root]} {mode}"


def parse_key(key_name: str) -> Tuple[int, str]:
    value = (key_name or "").strip().replace("♯", "#").replace("♭", "b")
    if not value:
        raise ValueError("调性不能为空")
    parts = value.split()
    note = parts[0].upper()
    aliases = {
        "DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#",
    }
    note = aliases.get(note, note)
    if note not in NOTE_NAMES:
        raise ValueError(f"无法识别调性: {key_name}")
    mode_text = " ".join(parts[1:]).lower()
    mode = "minor" if mode_text in ("minor", "min", "m", "小调") else "major"
    return NOTE_NAMES.index(note), mode


def key_transpose(key_name: str) -> int:
    root, mode = parse_key(key_name)
    target = 9 if mode == "minor" else 0
    raw = (target - root) % 12
    return raw - 12 if raw > 6 else raw


def _weighted_median(pitches_and_weights: Iterable[Tuple[int, float]]) -> float:
    ordered = sorted((int(p), max(0.0, float(w))) for p, w in pitches_and_weights)
    if not ordered:
        return 72.0
    total = sum(w for _, w in ordered)
    if total <= 0:
        return float(ordered[len(ordered) // 2][0])
    half = total / 2
    accumulated = 0.0
    for pitch, weight in ordered:
        accumulated += weight
        if accumulated >= half:
            return float(pitch)
    return float(ordered[-1][0])


def choose_octave_shift(events: Sequence[NoteEvent], semitone_shift: int) -> int:
    best: Optional[Tuple[Tuple[float, float, int], int]] = None
    for shift in (-24, -12, 0, 12, 24):
        shifted = [(event.midi_pitch + semitone_shift + shift, event.weight) for event in events]
        inside = sum(weight for pitch, weight in shifted if SKY_MIDI[0] <= pitch <= SKY_MIDI[-1])
        median = _weighted_median(shifted)
        score = (inside, -abs(median - 72), -abs(shift))
        if best is None or score > best[0]:
            best = (score, shift)
    return best[1] if best else 0


def fold_to_sky_range(pitch: int) -> Tuple[int, bool]:
    folded = int(pitch)
    changed = False
    while folded < SKY_MIDI[0]:
        folded += 12
        changed = True
    while folded > SKY_MIDI[-1]:
        folded -= 12
        changed = True
    return folded, changed


def nearest_sky_pitch(
    pitch: int,
    previous_source_pitch: Optional[int] = None,
    previous_sky_pitch: Optional[int] = None,
) -> Tuple[int, bool]:
    folded, folded_changed = fold_to_sky_range(pitch)
    distances = [abs(candidate - folded) for candidate in SKY_MIDI]
    minimum = min(distances)
    candidates = [candidate for candidate, distance in zip(SKY_MIDI, distances) if distance == minimum]
    if len(candidates) == 1:
        return candidates[0], folded_changed or candidates[0] != pitch

    # 半音恰好位于两个自然音之间时，优先保持前一个音的运动方向。
    chosen = candidates[0]
    if previous_source_pitch is not None and previous_sky_pitch is not None:
        direction = pitch - previous_source_pitch
        if direction > 0:
            chosen = max(candidates)
        elif direction < 0:
            chosen = min(candidates)
        else:
            chosen = min(candidates, key=lambda item: (abs(item - previous_sky_pitch), item))
    return chosen, True


def _build_quantize_grid(
    events: Sequence[NoteEvent],
    bpm: float,
    beat_times_ms: Optional[Sequence[int]],
    subdivisions: int,
) -> Tuple[List[float], float]:
    beats = sorted(float(t) for t in (beat_times_ms or []) if t >= 0)
    if len(beats) >= 2:
        intervals = [b - a for a, b in zip(beats, beats[1:]) if b > a]
        beat_ms = sorted(intervals)[len(intervals) // 2] if intervals else 60000.0 / max(1.0, bpm)
        grid: List[float] = []
        first_event = min((event.start_ms for event in events), default=0)
        cursor = beats[0]
        while cursor - beat_ms >= first_event:
            cursor -= beat_ms
            beats.insert(0, cursor)
        for start, end in zip(beats, beats[1:]):
            step = (end - start) / subdivisions
            grid.extend(start + i * step for i in range(subdivisions))
        grid.append(beats[-1])
        last_event = max((event.start_ms for event in events), default=0)
        cursor = beats[-1]
        while cursor < last_event:
            cursor += beat_ms / subdivisions
            grid.append(cursor)
        return sorted(set(grid)), beat_ms / subdivisions

    beat_ms = 60000.0 / max(1.0, bpm or 120.0)
    step = beat_ms / subdivisions
    anchor = min((event.start_ms for event in events), default=0)
    last = max((event.start_ms for event in events), default=anchor)
    count = int(math.ceil((last - anchor) / step)) + 1
    return [anchor + i * step for i in range(count + 1)], step


def quantize_events(
    events: Sequence[NoteEvent],
    mode: str,
    bpm: float,
    beat_times_ms: Optional[Sequence[int]] = None,
) -> List[NoteEvent]:
    if mode == "off" or not events:
        return list(events)
    subdivisions = 2 if mode == "1/8" else 4
    grid, typical_step = _build_quantize_grid(events, bpm, beat_times_ms, subdivisions)
    if not grid:
        return list(events)
    result: List[NoteEvent] = []
    for event in events:
        pos = bisect.bisect_left(grid, event.start_ms)
        choices = grid[max(0, pos - 1): min(len(grid), pos + 1)]
        target = min(choices, key=lambda item: abs(item - event.start_ms))
        if abs(target - event.start_ms) <= typical_step * 0.35:
            delta = int(round(target - event.start_ms))
            result.append(event.shifted(delta))
        else:
            result.append(event)
    return result


def _group_events(events: Sequence[NoteEvent], tolerance_ms: int = 40) -> List[List[NoteEvent]]:
    groups: List[List[NoteEvent]] = []
    for event in sorted(events, key=lambda item: (item.start_ms, item.midi_pitch)):
        if not groups or event.start_ms - groups[-1][0].start_ms > tolerance_ms:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def arrange_events(
    events: Sequence[NoteEvent],
    options: TranscriptionOptions,
    bpm: float = 120.0,
    beat_times_ms: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, object]], str, int, int, Dict[str, int]]:
    if not events:
        return [], options.source_key or "C major", 0, 0, {
            "raw_event_count": 0,
            "arranged_note_count": 0,
            "chord_count": 0,
            "folded_count": 0,
            "accidental_count": 0,
            "filtered_count": 0,
            "deduped_count": 0,
            "polyphony_reduced": 0,
        }

    detected_key = options.source_key or detect_key(events)
    semitone_shift = key_transpose(detected_key)
    octave_shift = (
        int(options.octave_shift) * 12
        if options.octave_shift is not None
        else choose_octave_shift(events, semitone_shift)
    )
    timed_events = quantize_events(events, options.quantize, bpm, beat_times_ms)
    groups = _group_events(timed_events)

    notes: List[Dict[str, object]] = []
    folded_count = 0
    accidental_count = 0
    deduped_count = 0
    polyphony_reduced = 0
    chord_count = 0
    previous_source_pitch: Optional[int] = None
    previous_sky_pitch: Optional[int] = None

    for group in groups:
        mapped_by_key: Dict[int, Tuple[NoteEvent, int, float]] = {}
        for event in group:
            shifted_pitch = event.midi_pitch + semitone_shift + octave_shift
            folded_pitch, was_folded = fold_to_sky_range(shifted_pitch)
            sky_pitch, was_adjusted = nearest_sky_pitch(
                folded_pitch, previous_source_pitch, previous_sky_pitch
            )
            if was_folded:
                folded_count += 1
            if was_adjusted and sky_pitch != folded_pitch:
                accidental_count += 1
            key_index = SKY_MIDI.index(sky_pitch)
            score = event.weight
            previous_source_pitch = folded_pitch
            previous_sky_pitch = sky_pitch
            existing = mapped_by_key.get(key_index)
            if existing is None or score > existing[2]:
                if existing is not None:
                    deduped_count += 1
                mapped_by_key[key_index] = (event, sky_pitch, score)
            else:
                deduped_count += 1

        candidates = [
            (key_index, event, pitch, score)
            for key_index, (event, pitch, score) in mapped_by_key.items()
        ]
        if len(candidates) > options.max_polyphony:
            melody = max(candidates, key=lambda item: item[2])
            others = sorted(
                (item for item in candidates if item is not melody),
                key=lambda item: (item[3], item[2]),
                reverse=True,
            )
            kept = [melody] + others[: options.max_polyphony - 1]
            polyphony_reduced += len(candidates) - len(kept)
            candidates = kept

        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1:
            chord_count += 1
        group_time = min(event.start_ms for event in group)
        for key_index, _event, _pitch, _score in candidates:
            notes.append({"time": int(group_time), "key": f"1Key{key_index}"})

    notes.sort(key=lambda item: (int(item["time"]), int(str(item["key"])[4:])))
    filtered = deduped_count + polyphony_reduced
    stats = {
        "raw_event_count": len(events),
        "arranged_note_count": len(notes),
        "chord_count": chord_count,
        "folded_count": folded_count,
        "accidental_count": accidental_count,
        "filtered_count": filtered,
        "deduped_count": deduped_count,
        "polyphony_reduced": polyphony_reduced,
    }
    return notes, detected_key, semitone_shift, octave_shift, stats
