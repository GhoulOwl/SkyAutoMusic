"""Event fusion: confidence reconstruction, octave & gap repair (维度2/3/5).

This is the key fix for the hard-coded ``confidence=1.0`` bug. For every
MuScriptor event we compare its pitch against an independently tracked F0
trajectory and assign a *real* confidence. The same F0 grid drives two repair
passes:

* **octave repair** -- folds an event's MIDI into the octave that best matches
  the F0 median (AMT octave errors are extremely common);
* **gap repair** -- synthesizes notes for voiced F0 segments that MuScriptor
  completely missed (this is what recovers "missing notes").

All steps are best-effort and wrapped by the caller, so a missing optional
dependency (``crepe``/``librosa``) simply yields the original events.
"""

from __future__ import annotations

import statistics
from typing import List, Optional, Tuple

from .models import PitchEvent, TranscribeConfig
from .onset import OnsetDetector
from .pitch import F0Tracker, F0Frame

_NO_EVIDENCE_CONFIDENCE = 0.55  # trust the model when F0 is silent/unavailable


def fuse_events(
    events: List[PitchEvent], audio_path: str, config: TranscribeConfig
) -> List[PitchEvent]:
    f0: List[F0Frame] = []
    if config.f0_enabled:
        try:
            f0 = F0Tracker(config).track(audio_path)
        except Exception:
            f0 = []

    if config.onset_enabled:
        try:
            OnsetDetector(config).reestimate_offsets(audio_path, events)
        except Exception:
            pass

    if f0:
        for ev in events:
            seg = _segment_midis(f0, ev.time_ms, ev.time_ms + max(1, ev.duration_ms))
            if config.repair_octave and seg:
                ev.midi = _repair_octave(ev.midi, seg)
            ev.confidence = _f0_agreement(f0, ev.time_ms, ev.time_ms + max(1, ev.duration_ms), ev.midi)
            ev.confidence_source = "f0"
    else:
        # No F0 evidence: keep the model's confidence but flag it as unverified.
        for ev in events:
            ev.confidence = _NO_EVIDENCE_CONFIDENCE if ev.confidence >= 1.0 else ev.confidence
            ev.confidence_source = "muscriptor"

    if config.repair_gaps and f0:
        try:
            events = _fill_gaps(events, f0, config)
        except Exception:
            pass

    return events


def _segment_midis(f0: List[F0Frame], t0: int, t1: int) -> List[float]:
    return [m for (t, m, _c, v) in f0 if v and m is not None and t0 <= t < t1]


def _f0_agreement(
    f0: List[F0Frame], t0: int, t1: int, midi: float
) -> float:
    seg = _segment_midis(f0, t0, t1)
    if not seg:
        return _NO_EVIDENCE_CONFIDENCE
    matches = 0
    for m in seg:
        diff = abs(m - midi)
        if diff <= 1.0 or (round(diff) % 12) <= 1:
            matches += 1
    ratio = matches / len(seg)
    return round(0.2 + 0.8 * ratio, 3)


def _repair_octave(midi: float, seg_midis: List[float]) -> float:
    med = statistics.median(seg_midis)
    best_shift = 0
    best_dist = abs(midi - med)
    for shift in (-24, -12, 0, 12, 24):
        cand = midi + shift
        dist = abs(cand - med)
        if dist < best_dist:
            best_dist = dist
            best_shift = shift
    # Only correct when the original is a large (>1 octave) jump from the F0
    # median, otherwise leave the model's choice intact.
    if abs(best_shift) >= 12 and abs(midi - med) >= 12:
        return midi + best_shift
    return midi


def _fill_gaps(
    events: List[PitchEvent], f0: List[F0Frame], config: TranscribeConfig
) -> List[PitchEvent]:
    """Synthesize notes for voiced F0 segments with no covering event."""
    voiced = [(t, m) for (t, m, _c, v) in f0 if v and m is not None]
    if len(voiced) < 2:
        return events

    step_ms = max(1.0, config.f0_step_ms)
    # group into runs of consecutive voiced frames
    runs: List[List[Tuple[int, float]]] = []
    current: List[Tuple[int, float]] = [voiced[0]]
    for (t, m) in voiced[1:]:
        prev_t = current[-1][0]
        if t - prev_t <= step_ms * 1.5 + 1:
            current.append((t, m))
        else:
            runs.append(current)
            current = [(t, m)]
    runs.append(current)

    min_dur = max(1, config.min_note_duration_ms)
    out = list(events)
    for run in runs:
        start_t = run[0][0]
        end_t = run[-1][0]
        if end_t - start_t < min_dur:
            continue
        med = statistics.median([m for (_t, m) in run])
        # only fill a *complete* gap: no overlapping event at any pitch
        if _any_event_overlaps(events, start_t, end_t):
            continue
        out.append(
            PitchEvent(
                time_ms=start_t,
                midi=round(med),
                confidence=0.5,
                source="gap",
                duration_ms=end_t - start_t,
                instrument="",
                confidence_source="gap",
            )
        )
    return out


def _any_event_overlaps(
    events: List[PitchEvent], start_t: int, end_t: int
) -> bool:
    for ev in events:
        ev_end = ev.time_ms + max(0, ev.duration_ms)
        if not (ev_end <= start_t or ev.time_ms >= end_t):
            return True
    return False
