"""Conservative phrase selection and optional vocal-stem evidence.

Raw MuScriptor notes are immutable evidence. Stem observations never create a
symbolic note; they can rank a candidate or repair a strongly supported octave.
"""
from __future__ import annotations

import bisect
import tempfile
from collections import Counter
from dataclasses import replace
from typing import Sequence

from .arranger import SKY_MIDI, NATURAL_PCS, key_transpose
from .models import CancelledError, LeadSegment, QualityAnalysisDraft, SymbolicNote, TranscriptionError, VocalEvidence

EVIDENCE_VERSION = 2


def supported_pitch(note, evidence):
    spans = [item for item in evidence if item.activity == "present" and item.midi_pitch is not None
             and item.confidence is not None and item.confidence >= .85
             and item.start_ms < note.end_ms and item.end_ms > note.start_ms]
    weights = Counter()
    intervals = {}
    for item in spans:
        left, right = max(note.start_ms, item.start_ms), min(note.end_ms, item.end_ms)
        weights[item.midi_pitch] += (right - left) * item.confidence
        intervals.setdefault(item.midi_pitch, []).append((left, right))
    if not weights:
        return None
    pitch = max(weights, key=weights.get)
    # Union coverage avoids counting overlapping detector fragments twice.
    covered, end = 0, -1
    for left, right in sorted(intervals[pitch]):
        covered += max(0, right - max(left, end))
        end = max(end, right)
    return pitch if covered / (note.end_ms - note.start_ms) >= .70 else None


def select_phrase_melody(notes, beats, evidence=()):
    from .quality import _inside_or_near_vocal_phrase, _lead_cost, _lead_segments, _vocal_guard_ms
    vocals = [n for n in notes if n.instrument == "voice" and 45 <= n.midi_pitch <= 100]
    guard = _vocal_guard_ms(beats)
    candidates = [n for n in notes if n in vocals or (
        n.instrument != "voice" and n.role != "drums" and 54 <= n.midi_pitch <= 96
        and not _inside_or_near_vocal_phrase(n.start_ms, vocals, guard))]
    groups = []
    for note in sorted(set(candidates), key=lambda n: (n.start_ms, n.midi_pitch, n.instrument)):
        if not groups or note.start_ms - groups[-1][0].start_ms > 12:
            groups.append([note])
        else:
            groups[-1].append(note)
    selected = []
    begin = 0
    while begin < len(groups):
        end = begin + 1
        while end < len(groups) and groups[end][0].start_ms - max(n.end_ms for n in groups[end-1]) <= 1400:
            end += 1
        rows, scores, links = [], [], []
        for group in groups[begin:end]:
            voice = [n for n in group if n.instrument == "voice"]
            row = sorted(voice or group, key=lambda n: _lead_cost(n, None), reverse=True)[:8] + [None]
            row_scores, row_links = [], []
            for current in row:
                emission = -15.0 if current is None else _lead_cost(current, None) - 95.0
                if current is not None and current.instrument == "voice":
                    pitch = supported_pitch(current, evidence)
                    if pitch is not None:
                        emission += 30 if pitch == current.midi_pitch else -30
                if not rows:
                    row_scores.append(emission); row_links.append(-1)
                    continue
                choices = []
                for index, previous in enumerate(rows[-1]):
                    penalty = 0.0
                    if current is not None and previous is not None:
                        penalty += abs(current.midi_pitch - previous.midi_pitch) * .8
                        penalty += 7 if current.instrument != previous.instrument else 0
                        overlap = max(0, previous.end_ms - current.start_ms)
                        penalty += 70 * min(1, overlap / max(1, current.end_ms-current.start_ms))
                    choices.append(scores[-1][index] + emission - penalty)
                best = max(range(len(choices)), key=choices.__getitem__)
                row_scores.append(choices[best]); row_links.append(best)
            rows.append(row); scores.append(row_scores); links.append(row_links)
        state = max(range(len(scores[-1])), key=scores[-1].__getitem__)
        phrase = []
        for index in reversed(range(len(rows))):
            note = rows[index][state]
            if note is not None:
                supported = supported_pitch(note, evidence) if note.instrument == "voice" else None
                if supported is not None and abs(supported-note.midi_pitch) == 12:
                    note = replace(note, midi_pitch=supported)
                phrase.append(note)
            state = links[index][state]
        selected.extend(reversed(phrase))
        begin = end
    duration = max([n.end_ms for n in notes] + [1])
    return selected, _lead_segments(selected, duration)


def suspect_regions(draft):
    """Rank contexts, merge neighbours, cap each at 30 s and total at 30%."""
    vocals = sorted([n for n in draft.symbolic_notes if n.instrument == "voice"], key=lambda n: n.start_ms)
    bars = sorted(set([0, *draft.bar_starts_ms, int(draft.duration_sec*1000)]))
    flags = []
    for index, note in enumerate(vocals):
        previous = vocals[index-1] if index else None
        score = 0
        if previous:
            gap = note.start_ms - previous.end_ms
            score += 3 if abs(note.midi_pitch-previous.midi_pitch) >= 12 and gap < 600 else 0
            score += 2 if note.start_ms < previous.end_ms and note.midi_pitch != previous.midi_pitch else 0
            score += 2 if note.start_ms-previous.start_ms < 80 else 0
            score += 1 if 80 < gap < 600 else 0
        if not score:
            continue
        left = max(0, bisect.bisect_right(bars, note.start_ms)-2)
        right = min(len(bars)-1, bisect.bisect_right(bars, note.end_ms)+1)
        flags.append((score, bars[left], bars[right]))
    budget = int(draft.duration_sec * 1000 * .3)
    accepted = []
    for score, left, right in sorted(flags, reverse=True):
        right = min(right, left+30000)
        if any(a <= left and right <= b for a, b in accepted):
            continue
        # Union first, then enforce the actual decoded duration budget.
        merged = []
        for a, b in sorted([*accepted, (left, right)]):
            if merged and a <= merged[-1][1] and b-merged[-1][0] <= 30000:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        if sum(b-a for a, b in merged) <= budget:
            accepted = merged
    return accepted


def check_vocal_regions(path, draft, cancel_event=None, progress_cb=None):
    from .backends import transcribe_monophonic
    from .separation import separate_audio
    import librosa
    import soundfile as sf
    regions = suspect_regions(draft)
    evidence, failures = [], []
    for index, (left, right) in enumerate(regions):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("转写已取消")
        if progress_cb:
            progress_cb("vocal_check", index/max(1, len(regions)), "正在校验疑似人声片段")
        try:
            with tempfile.TemporaryDirectory(prefix="sky-vocal-check-") as root:
                audio, sr = librosa.load(path, sr=22050, mono=False, offset=left/1000, duration=(right-left)/1000)
                clip = root + "/source.wav"
                sf.write(clip, audio.T, sr)
                stems = separate_audio(clip, root, cancel_event)
                output = transcribe_monophonic(stems.vocal_path, "normal", cancel_event, min_midi=40, max_midi=100)
                for event in output.events:
                    evidence.append(VocalEvidence(event.start_ms+left, event.end_ms+left, event.midi_pitch,
                                                  event.strength, "present" if event.strength >= .85 else "uncertain"))
        except CancelledError:
            raise
        except Exception as exc:
            failures.append(str(exc))
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("转写已取消")
    draft.vocal_evidence = evidence
    draft.vocal_evidence_version = EVIDENCE_VERSION
    draft.timing_diagnostics.update({"vocalCheckRegions": [list(r) for r in regions],
                                     "vocalCheckFailures": failures,
                                     "vocalCheckDurationMs": sum(b-a for a, b in regions)})
    if progress_cb:
        progress_cb("vocal_check", 1, "疑似人声校验完成")


def map_phrase_keys(melody, options):
    """Jointly search transposition and phrase voicing over the 15-key lattice."""
    if not melody:
        return 0, []
    shifts = [key_transpose(options.source_key)] if options.source_key else list(range(-6, 6))
    best = None
    for shift in shifts:
        all_keys, total = [], 0.0
        begin = 0
        while begin < len(melody):
            end = begin+1
            while end < len(melody) and melody[end].instrument == melody[end-1].instrument and melody[end].start_ms-melody[end-1].end_ms <= 1400:
                end += 1
            phrase = melody[begin:end]
            offsets = [12*options.melody_octave_shift] if options.melody_octave_shift is not None else [-12, 0, 12]
            phrase_best = None
            for offset in offsets:
                rows, costs, links = [], [], []
                for i, note in enumerate(phrase):
                    pitch = note.midi_pitch+shift+offset
                    distance = min(abs(value-pitch) for value in SKY_MIDI)
                    row = [key for key, value in enumerate(SKY_MIDI) if abs(value-pitch) <= distance+2]
                    row_costs, row_links = [], []
                    for key in row:
                        weight = 1.5 if i == len(phrase)-1 else 1.0
                        emission = weight*abs(SKY_MIDI[key]-pitch)
                        emission += 2.5 if (note.midi_pitch+shift) % 12 not in NATURAL_PCS else 0
                        if not rows:
                            row_costs.append(emission); row_links.append(-1); continue
                        source_delta = note.midi_pitch-phrase[i-1].midi_pitch
                        transitions = []
                        for j, old_key in enumerate(rows[-1]):
                            delta = SKY_MIDI[key]-SKY_MIDI[old_key]
                            penalty = abs(delta-source_delta)*.6
                            if source_delta and delta and (source_delta > 0) != (delta > 0):
                                penalty += 8
                            transitions.append(costs[-1][j]+emission+penalty)
                        j = min(range(len(transitions)), key=transitions.__getitem__)
                        row_costs.append(transitions[j]); row_links.append(j)
                    rows.append(row); costs.append(row_costs); links.append(row_links)
                state = min(range(len(costs[-1])), key=costs[-1].__getitem__)
                cost = costs[-1][state]+abs(offset)*.02
                path = []
                for i in reversed(range(len(rows))):
                    path.append(rows[i][state]); state = links[i][state]
                value = (cost, list(reversed(path)))
                if phrase_best is None or value[0] < phrase_best[0]:
                    phrase_best = value
            total += phrase_best[0]
            all_keys.extend(phrase_best[1])
            begin = end
        candidate = (total, abs(shift), shift, all_keys)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    return best[2], best[3]
