from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import NoteEvent, StemKind, TranscriptionOptions


SKY_MIDI: Tuple[int, ...] = (
    60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84
)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _median(values: Sequence[float], fallback: float = 0.0) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float(fallback)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _normalize_beat_ms(value: float) -> float:
    """Fold half/double-tempo estimates into a useful quarter-note range."""
    if not math.isfinite(value) or value <= 0:
        return 500.0
    lower = 60000.0 / 180.0
    upper = 60000.0 / 60.0
    normalized = float(value)
    while normalized < lower - 1.0:
        normalized *= 2.0
    while normalized > upper + 1.0:
        normalized /= 2.0
    return _clamp(normalized, lower, upper)


@dataclass(frozen=True)
class _BeatInterval:
    midpoint_ms: float
    duration_ms: float


class _RhythmContext:
    def __init__(
        self,
        events: Sequence[NoteEvent],
        bpm: float,
        beat_times_ms: Optional[Sequence[int]],
    ) -> None:
        beats = sorted(set(float(value) for value in (beat_times_ms or []) if value >= 0))
        raw_intervals = [
            (start, end, end - start)
            for start, end in zip(beats, beats[1:])
            if end > start
        ]
        accepted = self._filter_intervals([item[2] for item in raw_intervals])
        accepted_ids = {round(value, 6) for value in accepted}
        self._intervals = [
            _BeatInterval(
                midpoint_ms=(start + end) / 2.0,
                duration_ms=_normalize_beat_ms(duration),
            )
            for start, end, duration in raw_intervals
            if round(duration, 6) in accepted_ids
        ]

        if self._intervals:
            reference = _median([item.duration_ms for item in self._intervals], 500.0)
        elif math.isfinite(float(bpm)) and float(bpm) > 0:
            reference = _normalize_beat_ms(60000.0 / float(bpm))
        else:
            unique_onsets = sorted(set(event.start_ms for event in events))
            onset_gaps = [
                float(end - start)
                for start, end in zip(unique_onsets, unique_onsets[1:])
                if 50 <= end - start <= 3000
            ]
            reference = _normalize_beat_ms(_median(onset_gaps, 500.0))
        self.reference_beat_ms = _normalize_beat_ms(reference)

    @staticmethod
    def _filter_intervals(values: Sequence[float]) -> List[float]:
        positive = [float(value) for value in values if math.isfinite(value) and value > 0]
        if len(positive) < 3:
            return positive
        center = _median(positive)
        deviations = [abs(value - center) for value in positive]
        mad = _median(deviations)
        tolerance = max(3.0 * mad, center * 0.35, 40.0)
        filtered = [value for value in positive if abs(value - center) <= tolerance]
        return filtered or positive

    @property
    def reference_bpm(self) -> float:
        return 60000.0 / self.reference_beat_ms

    def beat_ms_at(self, time_ms: int) -> float:
        if not self._intervals:
            return self.reference_beat_ms
        nearest = sorted(
            self._intervals,
            key=lambda item: abs(item.midpoint_ms - float(time_ms)),
        )[:5]
        return _median([item.duration_ms for item in nearest], self.reference_beat_ms)

    def fragment_gap_ms(self, time_ms: int) -> float:
        return _clamp(self.beat_ms_at(time_ms) * 0.15, 45.0, 120.0)

    def repeat_window_ms(self, time_ms: int) -> float:
        return _clamp(self.beat_ms_at(time_ms) * 0.55, 140.0, 300.0)


@dataclass
class _MappedNote:
    time_ms: int
    key_index: int
    sky_pitch: int
    source_pitch: int
    start_ms: int
    end_ms: int
    strength: float
    score: float
    stem: Optional[StemKind] = None

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)


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


def _merge_source_fragments(
    events: Sequence[NoteEvent],
    rhythm: _RhythmContext,
) -> Tuple[List[NoteEvent], int]:
    by_pitch: Dict[Tuple[Optional[StemKind], int], List[NoteEvent]] = {}
    for event in events:
        by_pitch.setdefault((event.stem, event.midi_pitch), []).append(event)

    merged_events: List[NoteEvent] = []
    merged_count = 0
    for pitch_events in by_pitch.values():
        merged_pitch: List[NoteEvent] = []
        for event in sorted(pitch_events, key=lambda item: (item.start_ms, item.end_ms)):
            if not merged_pitch:
                merged_pitch.append(event)
                continue
            previous = merged_pitch[-1]
            gap_ms = event.start_ms - previous.end_ms
            if gap_ms <= rhythm.fragment_gap_ms(event.start_ms):
                merged_pitch[-1] = NoteEvent(
                    start_ms=previous.start_ms,
                    end_ms=max(previous.end_ms, event.end_ms),
                    midi_pitch=previous.midi_pitch,
                    strength=max(previous.strength, event.strength),
                    source=previous.source,
                    stem=previous.stem,
                )
                merged_count += 1
            else:
                merged_pitch.append(event)
        merged_events.extend(merged_pitch)
    merged_events.sort(key=lambda item: (item.start_ms, item.midi_pitch, item.end_ms))
    return merged_events, merged_count


def _clean_mapped_groups(
    groups: Sequence[Sequence[_MappedNote]],
    mode: str,
    rhythm: _RhythmContext,
) -> Tuple[List[List[_MappedNote]], int, int, float]:
    if not groups:
        return [], 0, 0, rhythm.repeat_window_ms(0)

    windows = [
        rhythm.repeat_window_ms(min(note.time_ms for note in group))
        for group in groups
        if group
    ]
    average_window = _median(windows, rhythm.repeat_window_ms(0))
    if mode == "off":
        return [list(group) for group in groups], 0, 0, average_window

    cleaned: List[List[_MappedNote]] = []
    suppressed_count = 0
    preserved_count = 0

    for group in groups:
        if not group:
            continue
        previous_by_key = (
            {note.key_index: note for note in cleaned[-1]}
            if cleaned
            else {}
        )
        kept: List[_MappedNote] = []
        for note in group:
            previous = previous_by_key.get(note.key_index)
            if previous is None:
                kept.append(note)
                continue

            onset_gap = note.time_ms - previous.time_ms
            repeat_window = rhythm.repeat_window_ms(note.time_ms)
            if onset_gap < 0 or onset_gap > repeat_window:
                kept.append(note)
                continue

            suppress = mode == "strong"
            if mode == "auto":
                beat_ms = rhythm.beat_ms_at(note.time_ms)
                source_gap = note.start_ms - previous.end_ms
                mapping_collision = note.source_pitch != previous.source_pitch
                weak_short_note = (
                    note.duration_ms < beat_ms * 0.35
                    and note.strength < max(0.55, previous.strength * 0.85)
                )
                clearly_rearticulated = (
                    source_gap >= _clamp(beat_ms * 0.20, 70.0, 180.0)
                    and note.duration_ms >= beat_ms * 0.18
                    and note.strength >= max(0.45, previous.strength * 0.75)
                )
                suppress = (
                    source_gap <= rhythm.fragment_gap_ms(note.time_ms)
                    or mapping_collision
                    or weak_short_note
                )
                if clearly_rearticulated and not mapping_collision:
                    suppress = False

            if suppress:
                previous.end_ms = max(previous.end_ms, note.end_ms)
                previous.strength = max(previous.strength, note.strength)
                previous.score = max(previous.score, note.score)
                suppressed_count += 1
            else:
                kept.append(note)
                preserved_count += 1

        if kept:
            cleaned.append(kept)

    return cleaned, suppressed_count, preserved_count, average_window


_FUSION_ROLE_WEIGHTS = {
    "vocal_first": {
        "vocals": 5.0,
        "piano": 2.4,
        "guitar": 2.1,
        "bass": 1.8,
        "instrumental": 0.7,
    },
    "keyboard_first": {
        "piano": 5.0,
        "vocals": 3.4,
        "guitar": 2.0,
        "bass": 1.8,
        "instrumental": 0.7,
    },
    "balanced": {
        "vocals": 3.0,
        "piano": 2.8,
        "guitar": 2.6,
        "bass": 2.0,
        "instrumental": 0.8,
    },
}


def _is_near_specialized_event(
    event: NoteEvent,
    starts_by_pitch: Dict[int, List[int]],
    onset_tolerance_ms: int = 80,
) -> bool:
    for pitch in range(event.midi_pitch - 1, event.midi_pitch + 2):
        starts = starts_by_pitch.get(pitch)
        if not starts:
            continue
        position = bisect.bisect_left(starts, event.start_ms)
        for index in (position - 1, position):
            if 0 <= index < len(starts):
                if abs(starts[index] - event.start_ms) <= onset_tolerance_ms:
                    return True
    return False


def _prepare_fusion_events(
    events: Sequence[NoteEvent],
    options: TranscriptionOptions,
) -> Tuple[List[NoteEvent], Dict[str, int]]:
    enabled = set(options.enabled_stems)
    specialized = [
        event
        for event in events
        if event.stem in ("vocals", "piano", "bass", "guitar")
    ]
    starts_by_pitch: Dict[int, List[int]] = {}
    for event in specialized:
        starts_by_pitch.setdefault(event.midi_pitch, []).append(event.start_ms)
    for starts in starts_by_pitch.values():
        starts.sort()

    selected: List[NoteEvent] = []
    instrumental_suppressed = 0
    disabled_count = 0
    for event in events:
        stem = event.stem
        if stem in ("drums", None):
            disabled_count += 1
            continue
        if stem not in enabled:
            disabled_count += 1
            continue
        if stem == "instrumental":
            if options.instrumental_policy == "preview_only":
                disabled_count += 1
                continue
            if (
                options.instrumental_policy == "smart_fill"
                and _is_near_specialized_event(event, starts_by_pitch)
            ):
                instrumental_suppressed += 1
                continue
        selected.append(event)
    selected.sort(key=lambda item: (item.start_ms, item.midi_pitch))
    return selected, {
        "fusion_input_event_count": len(events),
        "fusion_selected_event_count": len(selected),
        "fusion_disabled_event_count": disabled_count,
        "instrumental_smart_fill_suppressed": instrumental_suppressed,
    }


def _role_score(event: NoteEvent, options: TranscriptionOptions) -> float:
    weights = _FUSION_ROLE_WEIGHTS[options.fusion_profile]
    return event.weight * weights.get(str(event.stem), 1.0)


def _limit_fusion_polyphony(
    candidates: Sequence[_MappedNote],
    limit: int,
    previous_lead_pitch: Optional[int],
    options: TranscriptionOptions,
) -> Tuple[List[_MappedNote], Optional[int]]:
    vocals = [item for item in candidates if item.stem == "vocals"]
    piano = [item for item in candidates if item.stem == "piano"]
    tonal = [
        item
        for item in candidates
        if item.stem in ("piano", "guitar", "instrumental")
    ]

    def lead_score(item: _MappedNote) -> Tuple[float, float]:
        continuity = (
            0.0
            if previous_lead_pitch is None
            else -abs(item.sky_pitch - previous_lead_pitch)
        )
        return item.score, continuity

    if options.fusion_profile == "keyboard_first" and piano:
        lead_pool = piano
    elif options.fusion_profile == "balanced":
        lead_pool = vocals + tonal
    else:
        lead_pool = vocals or tonal
    lead = max(lead_pool or list(candidates), key=lead_score)

    if len(candidates) <= limit:
        return list(candidates), lead.sky_pitch

    kept = [lead]
    # 非人声预设仍在有空间时保留可靠人声，避免键盘/平衡模式把旋律整段丢掉。
    if limit > 1 and lead.stem != "vocals" and vocals:
        kept.append(max(vocals, key=lead_score))
    if limit > 1:
        bass = [item for item in candidates if item.stem == "bass" and item is not lead]
        if bass and len(kept) < limit:
            kept.append(max(bass, key=lambda item: item.score))
    remaining = [
        item
        for item in candidates
        if item not in kept
    ]
    remaining.sort(
        key=lambda item: (
            item.stem != "instrumental",
            item.score,
            item.sky_pitch,
        ),
        reverse=True,
    )
    kept.extend(remaining[: max(0, limit - len(kept))])
    return kept, lead.sky_pitch


def _empty_stats(
    options: TranscriptionOptions,
    rhythm: _RhythmContext,
) -> Dict[str, object]:
    return {
        "raw_event_count": 0,
        "arranged_note_count": 0,
        "chord_count": 0,
        "folded_count": 0,
        "accidental_count": 0,
        "filtered_count": 0,
        "deduped_count": 0,
        "polyphony_reduced": 0,
        "timing_reference_bpm": round(rhythm.reference_bpm, 2),
        "source_fragment_merged_count": 0,
        "mapped_repeat_suppressed_count": 0,
        "intentional_repeat_preserved_count": 0,
        "repeat_cleanup": options.repeat_cleanup,
        "average_repeat_window_ms": round(rhythm.repeat_window_ms(0), 1),
    }


def arrange_events(
    events: Sequence[NoteEvent],
    options: TranscriptionOptions,
    bpm: float = 120.0,
    beat_times_ms: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, object]], str, int, int, Dict[str, object]]:
    fusion_stats: Dict[str, int] = {}
    arranged_input = list(events)
    if options.mode == "stem_fusion":
        arranged_input, fusion_stats = _prepare_fusion_events(events, options)
    rhythm = _RhythmContext(arranged_input, bpm, beat_times_ms)
    if not arranged_input:
        return [], options.source_key or "C major", 0, 0, _empty_stats(options, rhythm)

    if options.repeat_cleanup == "off":
        prepared_events = list(arranged_input)
        source_fragment_merged_count = 0
    else:
        prepared_events, source_fragment_merged_count = _merge_source_fragments(
            arranged_input, rhythm
        )

    detected_key = options.source_key or detect_key(prepared_events)
    semitone_shift = key_transpose(detected_key)
    octave_shift = (
        int(options.octave_shift) * 12
        if options.octave_shift is not None
        else choose_octave_shift(prepared_events, semitone_shift)
    )
    timed_events = quantize_events(
        prepared_events, options.quantize, bpm, beat_times_ms
    )
    groups = _group_events(timed_events)

    mapped_groups: List[List[_MappedNote]] = []
    folded_count = 0
    accidental_count = 0
    group_deduped_count = 0
    polyphony_reduced = 0
    previous_source_pitch: Optional[int] = None
    previous_sky_pitch: Optional[int] = None
    previous_lead_pitch: Optional[int] = None

    for group in groups:
        group_time = min(event.start_ms for event in group)
        mapped_by_key: Dict[int, _MappedNote] = {}
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
            score = (
                _role_score(event, options)
                if options.mode == "stem_fusion"
                else event.weight
            )
            previous_source_pitch = folded_pitch
            previous_sky_pitch = sky_pitch
            existing = mapped_by_key.get(key_index)
            candidate = _MappedNote(
                time_ms=int(group_time),
                key_index=key_index,
                sky_pitch=sky_pitch,
                source_pitch=shifted_pitch,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                strength=event.strength,
                score=score,
                stem=event.stem,
            )
            if existing is None or score > existing.score:
                if existing is not None:
                    group_deduped_count += 1
                mapped_by_key[key_index] = candidate
            else:
                group_deduped_count += 1

        candidates = list(mapped_by_key.values())
        if len(candidates) > options.max_polyphony:
            if options.mode == "stem_fusion":
                kept, previous_lead_pitch = _limit_fusion_polyphony(
                    candidates,
                    options.max_polyphony,
                    previous_lead_pitch,
                    options,
                )
            else:
                melody = max(candidates, key=lambda item: item.sky_pitch)
                others = sorted(
                    (item for item in candidates if item is not melody),
                    key=lambda item: (item.score, item.sky_pitch),
                    reverse=True,
                )
                kept = [melody] + others[: options.max_polyphony - 1]
            polyphony_reduced += len(candidates) - len(kept)
            candidates = kept
        elif options.mode == "stem_fusion" and candidates:
            _kept, previous_lead_pitch = _limit_fusion_polyphony(
                candidates,
                len(candidates),
                previous_lead_pitch,
                options,
            )

        candidates.sort(key=lambda item: item.key_index)
        mapped_groups.append(candidates)

    cleaned_groups, repeat_suppressed, repeat_preserved, average_window = (
        _clean_mapped_groups(mapped_groups, options.repeat_cleanup, rhythm)
    )
    notes = [
        {"time": note.time_ms, "key": f"1Key{note.key_index}"}
        for group in cleaned_groups
        for note in group
    ]
    notes.sort(key=lambda item: (int(item["time"]), int(str(item["key"])[4:])))
    chord_count = sum(1 for group in cleaned_groups if len(group) > 1)
    deduped_count = source_fragment_merged_count + group_deduped_count
    filtered = deduped_count + polyphony_reduced + repeat_suppressed
    stats = {
        "raw_event_count": len(events),
        "arranged_note_count": len(notes),
        "chord_count": chord_count,
        "folded_count": folded_count,
        "accidental_count": accidental_count,
        "filtered_count": filtered,
        "deduped_count": deduped_count,
        "polyphony_reduced": polyphony_reduced,
        "timing_reference_bpm": round(rhythm.reference_bpm, 2),
        "source_fragment_merged_count": source_fragment_merged_count,
        "mapped_repeat_suppressed_count": repeat_suppressed,
        "intentional_repeat_preserved_count": repeat_preserved,
        "repeat_cleanup": options.repeat_cleanup,
        "average_repeat_window_ms": round(average_window, 1),
    }
    stats.update(fusion_stats)
    return notes, detected_key, semitone_shift, octave_shift, stats
