"""Deterministic 15-key piano arranger for an analysed song."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .analysis import CHORD_INTERVALS, NOTE_NAMES
from .models import AnalysisDraft, ChordSpan, NoteEvent, Section, TranscriptionOptions


SKY_MIDI: Tuple[int, ...] = (60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84)
NATURAL_PCS = (0, 2, 4, 5, 7, 9, 11)
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def fold_to_sky_range(pitch: int) -> Tuple[int, bool]:
    value = int(pitch)
    changed = False
    while value < SKY_MIDI[0]:
        value += 12
        changed = True
    while value > SKY_MIDI[-1]:
        value -= 12
        changed = True
    return value, changed


def nearest_sky_pitch(
    pitch: int,
    previous_source_pitch: Optional[int] = None,
    previous_sky_pitch: Optional[int] = None,
) -> Tuple[int, bool]:
    folded, changed = fold_to_sky_range(pitch)
    distances = [abs(candidate - folded) for candidate in SKY_MIDI]
    minimum = min(distances)
    choices = [candidate for candidate, distance in zip(SKY_MIDI, distances) if distance == minimum]
    if len(choices) == 1:
        return choices[0], changed or choices[0] != pitch
    if previous_source_pitch is not None and previous_sky_pitch is not None:
        direction = folded - previous_source_pitch
        if direction > 0:
            return max(choices), True
        if direction < 0:
            return min(choices), True
        return min(choices, key=lambda value: abs(value - previous_sky_pitch)), True
    return min(choices), True


def parse_key(key_name: str) -> Tuple[int, str]:
    parts = str(key_name or "C major").strip().replace("♯", "#").replace("♭", "b").split()
    note = (parts[0] if parts else "C").upper()
    aliases = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    note = aliases.get(note, note)
    root = NOTE_NAMES.index(note) if note in NOTE_NAMES else 0
    mode = "minor" if len(parts) > 1 and parts[1].lower() in ("minor", "min", "m", "小调") else "major"
    return root, mode


def key_transpose(key_name: str) -> int:
    root, mode = parse_key(key_name)
    target = 9 if mode == "minor" else 0
    raw = (target - root) % 12
    return raw - 12 if raw > 6 else raw


def _profile_score(hist: Sequence[float], profile: Sequence[float], root: int) -> float:
    target = [profile[(pitch - root) % 12] for pitch in range(12)]
    dot = sum(left * right for left, right in zip(hist, target))
    norm = math.sqrt(sum(value * value for value in hist) * sum(value * value for value in target))
    return dot / norm if norm else 0.0


def detect_key(events: Sequence[NoteEvent]) -> str:
    if not events:
        return "C major"
    hist = [0.0] * 12
    for event in events:
        hist[event.midi_pitch % 12] += max(1.0, event.weight)
    candidates = []
    for root in range(12):
        candidates.append((_profile_score(hist, MAJOR_PROFILE, root), root, "major"))
        candidates.append((_profile_score(hist, MINOR_PROFILE, root), root, "minor"))
    _, root, mode = max(candidates)
    return f"{NOTE_NAMES[root]} {mode}"


def _nearest_natural_pc(pitch_class: int, preferred_direction: int = 0) -> int:
    pitch_class %= 12
    distances = {pc: min((pc - pitch_class) % 12, (pitch_class - pc) % 12) for pc in NATURAL_PCS}
    best = min(distances.values())
    choices = [pc for pc, distance in distances.items() if distance == best]
    if preferred_direction > 0:
        return max(choices)
    if preferred_direction < 0:
        return min(choices)
    return min(choices)


def _pitch_to_key(pitch: int, previous_pitch: Optional[int] = None, previous_sky: Optional[int] = None) -> Tuple[int, bool]:
    natural_pc = _nearest_natural_pc(pitch % 12, 0 if previous_pitch is None else pitch - previous_pitch)
    candidates = [candidate for candidate in SKY_MIDI if candidate % 12 == natural_pc]
    if not candidates:
        value, changed = nearest_sky_pitch(pitch, previous_pitch, previous_sky)
        return SKY_MIDI.index(value), changed
    folded, folded_changed = fold_to_sky_range(pitch)
    selected = min(candidates, key=lambda value: (abs(value - folded), abs(value - 72)))
    return SKY_MIDI.index(selected), folded_changed or selected != pitch


def _section_at(sections: Sequence[Section], time_ms: int) -> Section:
    for section in sections:
        if section.start_ms <= time_ms < section.end_ms:
            return section
    return sections[-1] if sections else Section(0, max(1, time_ms + 1), "verse", 1)


def _chord_at(chords: Sequence[ChordSpan], time_ms: int) -> Optional[ChordSpan]:
    for chord in chords:
        if chord.start_ms <= time_ms < chord.end_ms:
            return chord
    return chords[-1] if chords else None


def _effective_preset(options: TranscriptionOptions, section: Section) -> str:
    if options.arrangement_preset != "auto":
        return options.arrangement_preset
    if section.role == "chorus" or section.energy >= 3:
        return "full"
    if section.role in ("intro", "pre_chorus") or section.energy >= 2:
        return "standard"
    if section.role == "outro":
        return "simple"
    return "simple"


def _chord_key_indices(chord: ChordSpan, semitone_shift: int) -> List[int]:
    root = (chord.root_pc + semitone_shift) % 12
    intervals = CHORD_INTERVALS.get(chord.quality, CHORD_INTERVALS["maj"])
    values: List[int] = []
    for interval in intervals:
        pc = _nearest_natural_pc((root + interval) % 12)
        # Low register supplies bass/root; upper natural tones cross into C5
        # only when the chord genuinely needs them.
        pitch = 60 + pc
        if pitch > 71 and interval != 0:
            pitch -= 12
        index, _ = _pitch_to_key(pitch)
        if index not in values:
            values.append(index)
    root_index, _ = _pitch_to_key(60 + root)
    if root_index in values:
        values.remove(root_index)
    return [root_index] + values


def _bar_beat_index(analysis: AnalysisDraft, time_ms: int) -> int:
    beats = analysis.tempo_map.beat_times_ms
    if not beats:
        return 0
    closest = min(range(len(beats)), key=lambda index: abs(beats[index] - time_ms))
    beat_count = {"4/4": 4, "3/4": 3, "6/8": 6}[analysis.tempo_map.meter]
    return closest % beat_count


def arrange_analysis(analysis: AnalysisDraft, options: TranscriptionOptions) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[int, str]]:
    """Render one analysed song into playable Sky-key events."""
    semitone_shift = key_transpose(options.source_key or analysis.detected_key)
    by_time: Dict[int, Dict[int, Tuple[int, str]]] = defaultdict(dict)
    roles: Dict[int, str] = {}

    def add(time_ms: int, key_index: int, priority: int, role: str) -> None:
        current = by_time[time_ms].get(key_index)
        if current is None or priority > current[0]:
            by_time[time_ms][key_index] = (priority, role)

    previous_melody_pitch: Optional[int] = None
    previous_sky_pitch: Optional[int] = None
    for melody in analysis.melody:
        pitch = melody.midi_pitch + semitone_shift + (12 * (options.melody_octave_shift or 0))
        choices = [pitch + shift for shift in (-24, -12, 0, 12, 24)]
        pitch = min(choices, key=lambda value: (0 if 72 <= value <= 84 else 50 + abs(value - 78), abs(value - (previous_sky_pitch or 78))))
        key_index, _adjusted = _pitch_to_key(pitch, previous_melody_pitch, previous_sky_pitch)
        key_index = max(7, key_index) if key_index < 7 and pitch >= 72 else key_index
        add(melody.start_ms, key_index, 5, "melody")
        previous_melody_pitch = pitch
        previous_sky_pitch = SKY_MIDI[key_index]

    beats = list(analysis.tempo_map.beat_times_ms)
    # Strong beats carry the harmonic skeleton.  In an energetic/full texture,
    # add a controlled eighth-note re-articulation as well; this is what makes
    # a chorus feel like piano accompaniment instead of isolated bass markers.
    accompaniment_slots: List[Tuple[int, bool]] = []
    for left, right in zip(beats, beats[1:]):
        accompaniment_slots.append((left, True))
        midpoint = int(round((left + right) / 2.0))
        section = _section_at(analysis.sections, left)
        preset = _effective_preset(options, section)
        if preset == "full" or (preset == "standard" and section.energy >= 2):
            accompaniment_slots.append((midpoint, False))

    for time_ms, is_main_beat in accompaniment_slots:
        chord = _chord_at(analysis.chords, time_ms)
        if chord is None:
            continue
        section = _section_at(analysis.sections, time_ms)
        preset = _effective_preset(options, section)
        beat = _bar_beat_index(analysis, time_ms)
        strong = beat == 0 or (analysis.tempo_map.meter == "4/4" and beat == 2) or (analysis.tempo_map.meter == "6/8" and beat == 3)
        chord_keys = _chord_key_indices(chord, semitone_shift)
        if not chord_keys:
            continue
        if not is_main_beat:
            # Accompaniment is never placed on sixteenth weak positions.  Keep
            # this deliberately compact so a melody can still breathe.
            if preset == "full":
                add(time_ms, chord_keys[0], 4, "bass")
                if len(chord_keys) > 1:
                    add(time_ms, chord_keys[1], 3, "harmony")
            elif preset == "standard" and len(chord_keys) > 1:
                add(time_ms, chord_keys[1], 2, "harmony")
            continue
        if preset == "simple":
            if strong:
                add(time_ms, chord_keys[0], 4, "bass")
        elif preset == "standard":
            if strong:
                add(time_ms, chord_keys[0], 4, "bass")
                if len(chord_keys) > 1:
                    add(time_ms, chord_keys[1], 3, "harmony")
            elif beat % 2 == 1 and len(chord_keys) > 2:
                add(time_ms, chord_keys[2], 2, "harmony")
        else:  # full
            if strong:
                for index, key_index in enumerate(chord_keys[:3]):
                    add(time_ms, key_index, 4 - min(index, 2), "bass" if index == 0 else "harmony")
                if section.role == "chorus" and options.max_polyphony >= 5:
                    octave_key = chord_keys[0] + 7
                    if octave_key <= 14:
                        add(time_ms, octave_key, 1, "octave")
            elif len(chord_keys) > 1:
                add(time_ms, chord_keys[min(2, len(chord_keys) - 1)], 2, "harmony")

    notes: List[Dict[str, object]] = []
    for time_ms in sorted(by_time):
        selected = sorted(by_time[time_ms].items(), key=lambda item: (-item[1][0], item[0]))[: options.max_polyphony]
        for key_index, (_priority, role) in sorted(selected):
            notes.append({"time": int(time_ms), "key": f"1Key{key_index}"})
            roles[(int(time_ms) << 4) | key_index] = role
    grouped = Counter(int(note["time"]) for note in notes)
    chord_count = sum(count > 1 for count in grouped.values())
    melody_count = sum(1 for role in roles.values() if role == "melody")
    stats: Dict[str, object] = {
        "arranged_note_count": len(notes),
        "onset_count": len(grouped),
        "chord_count": chord_count,
        "chord_onset_ratio": round(chord_count / max(1, len(grouped)), 3),
        "average_polyphony": round(len(notes) / max(1, len(grouped)), 3),
        "melody_note_count": melody_count,
        "timing_grid_count": len(analysis.tempo_map.grid_times_ms),
        "leadSource": analysis.lead_source,
    }
    return notes, stats, roles


def arrange_events(
    events: Sequence[NoteEvent],
    options: TranscriptionOptions,
    bpm: float = 120.0,
    beat_times_ms: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, object]], str, int, int, Dict[str, object]]:
    """Compatibility MIDI arrangement: preserve simultaneous MIDI notes safely."""
    detected = options.source_key or detect_key(events)
    shift = key_transpose(detected)
    grouped: Dict[int, List[NoteEvent]] = defaultdict(list)
    for event in events:
        grouped[event.start_ms].append(event)
    notes: List[Dict[str, object]] = []
    previous_source: Optional[int] = None
    previous_sky: Optional[int] = None
    folded = 0
    for time_ms in sorted(grouped):
        selected = sorted(grouped[time_ms], key=lambda event: (event.midi_pitch, event.strength), reverse=True)[: options.max_polyphony]
        keys: Dict[int, NoteEvent] = {}
        for event in selected:
            key_index, adjusted = _pitch_to_key(event.midi_pitch + shift, previous_source, previous_sky)
            if adjusted:
                folded += 1
            keys[key_index] = event
            previous_source = event.midi_pitch + shift
            previous_sky = SKY_MIDI[key_index]
        notes.extend({"time": int(time_ms), "key": f"1Key{index}"} for index in sorted(keys))
    stats: Dict[str, object] = {
        "raw_event_count": len(events), "arranged_note_count": len(notes),
        "onset_count": len(grouped), "folded_count": folded,
    }
    return notes, detected, shift, 0, stats
