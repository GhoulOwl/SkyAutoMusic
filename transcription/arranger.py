from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .mapping import SkyKeyMapper
from .models import PitchEvent, SkyNote, TranscribeConfig, TranscribeStats


@dataclass(frozen=True)
class _MelodyEvent:
    time_ms: int
    midi: float
    confidence: float
    duration_ms: int
    instrument: str
    score: float


@dataclass(frozen=True)
class _Choice:
    note: Optional[SkyNote]
    cost: float
    delete: bool = False


@dataclass
class _State:
    cost: float
    path: List[Optional[SkyNote]]
    last_key_index: Optional[int] = None
    last_time: Optional[int] = None
    edge_key: Optional[str] = None
    edge_run: int = 0
    selected_count: int = 0


class SkyMelodyArranger:
    """Turn raw transcription events into a Sky-friendly single melody line."""

    _MAX_STATES = 48

    def __init__(self, config: TranscribeConfig):
        self.config = config
        self.mapper = SkyKeyMapper(config)
        self.last_candidates: List[SkyNote] = []
        self.last_arranged: List[SkyNote] = []
        self.last_report: Dict[str, object] = {}
        self._instrument_avg_conf: Dict[str, float] = {}

    def arrange(
        self,
        events: Iterable[PitchEvent],
        stats: TranscribeStats,
    ) -> Tuple[List[SkyNote], TranscribeStats]:
        raw_events = sorted(list(events), key=lambda e: (int(e.time_ms), float(e.midi)))
        self.last_candidates = []
        self.last_arranged = []
        self.last_report = {}

        stats.arranger_enabled = True
        if not raw_events:
            self._finish_report(stats, raw_count=0, phrases=[])
            return [], stats

        filtered = self._filter_events(raw_events)
        if self.config.arranger_keep_chords:
            # 维度4：和弦模式保留同时刻的多个声部，跳过会把它们折叠成单音的聚类。
            clustered = filtered
        else:
            clustered = self._cluster_events(filtered)
        phrases = self._split_phrases(clustered)

        arranged: List[SkyNote] = []
        phrase_reports: List[Dict[str, object]] = []
        phrase_transposes: List[int] = []

        for phrase_index, phrase in enumerate(phrases):
            transpose = self.mapper.compute_transpose(event.midi for event in phrase)
            phrase_transposes.append(transpose)
            self.last_candidates.extend(
                self._preview_note(event, transpose) for event in phrase
            )

        if self.config.arranger_keep_chords:
            arranged, phrase_reports = self._arrange_chords(
                phrases, phrase_transposes
            )
        else:
            arranged, phrase_reports = self._arrange_melody(
                phrases, phrase_transposes
            )

        arranged.sort(key=lambda n: (int(n.time), SkyKeyMapper.key_to_index(n.key), n.key))
        self.last_arranged = arranged

        stats.transpose = phrase_transposes[0] if len(phrase_transposes) == 1 else 0
        stats.mapped_event_count = len(arranged)
        stats.arranger_phrase_count = len(phrases)
        stats.arranger_candidate_count = len(clustered)
        stats.arranger_selected_count = len(arranged)
        stats.arranger_deleted_count = max(0, len(raw_events) - len(arranged))
        stats.arranger_boundary_hit_count = sum(
            1
            for note in arranged
            if SkyKeyMapper.key_to_index(note.key) in {0, self.mapper.NUM_KEYS - 1}
        )
        stats.dropped_count += stats.arranger_deleted_count
        self._finish_report(stats, raw_count=len(raw_events), phrases=phrase_reports)
        return arranged, stats

    @property
    def last_report_bytes(self) -> bytes:
        return json.dumps(self.last_report, ensure_ascii=False, indent=2).encode("utf-8")

    def _arrange_melody(
        self,
        phrases: List[List[_MelodyEvent]],
        phrase_transposes: List[int],
    ) -> Tuple[List[SkyNote], List[Dict[str, object]]]:
        arranged: List[SkyNote] = []
        phrase_reports: List[Dict[str, object]] = []
        for phrase_index, phrase in enumerate(phrases):
            transpose = phrase_transposes[phrase_index]
            phrase_notes, phrase_report = self._arrange_phrase(
                phrase,
                transpose=transpose,
                phrase_index=phrase_index,
            )
            arranged.extend(phrase_notes)
            phrase_reports.append(phrase_report)
        return arranged, phrase_reports

    def _arrange_chords(
        self,
        phrases: List[List[_MelodyEvent]],
        phrase_transposes: List[int],
    ) -> Tuple[List[SkyNote], List[Dict[str, object]]]:
        """维度4：保留同时间的多个声部，输出可被 Sky 多键同按演奏的和弦。

        For each simultaneous time slot keep the top ``max_chord_voices`` events
        (by content-aware score), de-duplicated by Sky key. Unlike the melody
        DP this never deletes a simultaneous voice, so harmonic content is
        preserved.
        """
        arranged: List[SkyNote] = []
        phrase_reports: List[Dict[str, object]] = []
        for phrase_index, phrase in enumerate(phrases):
            transpose = phrase_transposes[phrase_index]
            by_time: Dict[int, List[_MelodyEvent]] = {}
            for ev in phrase:
                by_time.setdefault(int(ev.time_ms), []).append(ev)

            notes: List[SkyNote] = []
            for t in sorted(by_time):
                group = sorted(
                    by_time[t],
                    key=lambda e: -self._event_score(
                        e, self._instrument_priority(e.instrument)
                    ),
                )
                used_keys: set = set()
                picked = 0
                for ev in group:
                    if picked >= self.config.max_chord_voices:
                        break
                    key, _clamped = self.mapper.midi_to_key(
                        ev.midi, transpose=transpose
                    )
                    if key in used_keys:
                        continue
                    used_keys.add(key)
                    notes.append(self._preview_note(ev, transpose))
                    picked += 1
            arranged.extend(notes)
            phrase_reports.append(
                {
                    "phrase_index": phrase_index,
                    "start_ms": int(phrase[0].time_ms) if phrase else 0,
                    "end_ms": int(phrase[-1].time_ms) if phrase else 0,
                    "transpose": transpose,
                    "candidate_count": len(phrase),
                    "selected_count": len(notes),
                    "deleted_count": max(0, len(phrase) - len(notes)),
                    "cost": 0.0,
                }
            )
        return arranged, phrase_reports

    def _filter_events(self, events: List[PitchEvent]) -> List[_MelodyEvent]:
        # 维度4（内容感知）：先统计每个乐器被 F0 校验后的平均置信度，
        # 让真正承载主旋律的乐器在评分中胜出，而非死板按名字表。
        conf_sum: Dict[str, float] = {}
        conf_cnt: Dict[str, int] = {}
        for event in events:
            inst = str(event.instrument or "unknown")
            conf_sum[inst] = conf_sum.get(inst, 0.0) + float(event.confidence)
            conf_cnt[inst] = conf_cnt.get(inst, 0) + 1
        self._instrument_avg_conf = {
            inst: conf_sum[inst] / conf_cnt[inst] for inst in conf_sum
        }

        kept: List[_MelodyEvent] = []
        for event in events:
            instrument = str(event.instrument or "unknown")
            if self._is_excluded_instrument(instrument):
                continue
            confidence = float(event.confidence)
            duration_ms = int(event.duration_ms)
            if confidence < self.config.min_confidence:
                continue
            if 0 < duration_ms < self.config.min_note_duration_ms:
                continue
            priority = self._instrument_priority(instrument)
            if priority <= 0:
                continue
            kept.append(
                _MelodyEvent(
                    time_ms=max(0, int(event.time_ms)),
                    midi=float(event.midi),
                    confidence=confidence,
                    duration_ms=duration_ms,
                    instrument=instrument,
                    score=self._event_score(event, priority),
                )
            )
        return kept

    def _cluster_events(self, events: List[_MelodyEvent]) -> List[_MelodyEvent]:
        if not events:
            return []
        events = sorted(events, key=lambda e: (e.time_ms, -e.score, e.midi))
        window_ms = int(self.config.arranger_cluster_ms)
        if window_ms <= 0:
            return events

        clusters: List[List[_MelodyEvent]] = []
        current: List[_MelodyEvent] = []
        cluster_start = 0
        for event in events:
            if not current:
                current = [event]
                cluster_start = int(event.time_ms)
                continue
            if int(event.time_ms) - cluster_start <= window_ms:
                current.append(event)
            else:
                clusters.append(current)
                current = [event]
                cluster_start = int(event.time_ms)
        if current:
            clusters.append(current)

        selected: List[_MelodyEvent] = []
        for cluster in clusters:
            best = max(cluster, key=lambda e: (e.score, e.duration_ms, e.midi))
            selected.append(best)
        selected.sort(key=lambda e: (e.time_ms, e.midi))
        return selected

    def _split_phrases(self, events: List[_MelodyEvent]) -> List[List[_MelodyEvent]]:
        if not events:
            return []
        gap_ms = int(self.config.arranger_phrase_gap_ms)
        if gap_ms <= 0:
            return [events]

        phrases: List[List[_MelodyEvent]] = []
        current: List[_MelodyEvent] = []
        last_time: Optional[int] = None
        for event in events:
            if last_time is not None and int(event.time_ms) - last_time > gap_ms:
                phrases.append(current)
                current = [event]
            else:
                current.append(event)
            last_time = int(event.time_ms)
        if current:
            phrases.append(current)
        return phrases

    def _arrange_phrase(
        self,
        phrase: List[_MelodyEvent],
        transpose: int,
        phrase_index: int,
    ) -> Tuple[List[SkyNote], Dict[str, object]]:
        states = [_State(cost=0.0, path=[])]
        for event in phrase:
            choices = self._choices_for_event(event, transpose)
            next_states: List[_State] = []
            for state in states:
                for choice in choices:
                    next_states.append(self._advance_state(state, choice, event))
            next_states.sort(key=lambda s: (s.cost, -s.selected_count))
            states = next_states[: self._MAX_STATES]

        best = min(states, key=lambda s: (s.cost, -s.selected_count))
        notes = [note for note in best.path if note is not None]
        report = {
            "phrase_index": phrase_index,
            "start_ms": int(phrase[0].time_ms) if phrase else 0,
            "end_ms": int(phrase[-1].time_ms) if phrase else 0,
            "transpose": transpose,
            "candidate_count": len(phrase),
            "selected_count": len(notes),
            "deleted_count": max(0, len(phrase) - len(notes)),
            "cost": round(best.cost, 3),
        }
        return notes, report

    def _choices_for_event(self, event: _MelodyEvent, transpose: int) -> List[_Choice]:
        choices: Dict[str, _Choice] = {}
        for octave_shift in (0, -12, 12):
            key, clamped = self.mapper.midi_to_key(
                event.midi + octave_shift,
                transpose=transpose,
            )
            key_midi = self.mapper.key_to_midi(key)
            target_midi = float(event.midi) + float(transpose) + float(octave_shift)
            pitch_cost = abs(float(key_midi) - target_midi) * 3.5
            octave_cost = abs(octave_shift) / 12.0 * 1.6
            clamped_cost = abs(int(clamped)) * 0.8
            choice = _Choice(
                note=SkyNote(
                    time=int(event.time_ms),
                    key=key,
                    midi=float(key_midi),
                    confidence=float(event.confidence),
                    duration_ms=int(event.duration_ms),
                    instrument=str(event.instrument),
                ),
                cost=pitch_cost + octave_cost + clamped_cost,
            )
            existing = choices.get(key)
            if existing is None or choice.cost < existing.cost:
                choices[key] = choice

        delete_choice = _Choice(note=None, cost=self._delete_cost(event), delete=True)
        return list(choices.values()) + [delete_choice]

    def _advance_state(
        self,
        state: _State,
        choice: _Choice,
        event: _MelodyEvent,
    ) -> _State:
        if choice.delete or choice.note is None:
            return _State(
                cost=state.cost + choice.cost,
                path=state.path + [None],
                last_key_index=state.last_key_index,
                last_time=state.last_time,
                edge_key=state.edge_key,
                edge_run=state.edge_run,
                selected_count=state.selected_count,
            )

        note = choice.note
        key_index = SkyKeyMapper.key_to_index(note.key)
        cost = state.cost + choice.cost
        edge_key: Optional[str] = None
        edge_run = 0

        if state.last_key_index is not None and state.last_time is not None:
            jump = abs(key_index - state.last_key_index)
            if jump > 5:
                cost += (jump - 5) * 7.0
            else:
                cost += jump * 0.25

            gap = int(event.time_ms) - int(state.last_time)
            min_gap = int(self.config.arranger_min_note_gap_ms)
            if min_gap > 0 and gap < min_gap:
                cost += (float(min_gap - gap) / float(min_gap)) * 70.0

        if key_index in {0, self.mapper.NUM_KEYS - 1}:
            edge_key = note.key
            edge_run = state.edge_run + 1 if state.edge_key == note.key else 1
            if edge_run >= 2:
                cost += (edge_run - 1) * 22.0

        return _State(
            cost=cost,
            path=state.path + [note],
            last_key_index=key_index,
            last_time=int(event.time_ms),
            edge_key=edge_key,
            edge_run=edge_run,
            selected_count=state.selected_count + 1,
        )

    def _preview_note(self, event: _MelodyEvent, transpose: int) -> SkyNote:
        key, _clamped = self.mapper.midi_to_key(event.midi, transpose=transpose)
        return SkyNote(
            time=int(event.time_ms),
            key=key,
            midi=float(self.mapper.key_to_midi(key)),
            confidence=float(event.confidence),
            duration_ms=int(event.duration_ms),
            instrument=str(event.instrument),
        )

    def _delete_cost(self, event: _MelodyEvent) -> float:
        priority = self._instrument_priority(event.instrument)
        duration_bonus = min(max(0, int(event.duration_ms)), 900) / 900.0 * 12.0
        return 18.0 + float(event.confidence) * 10.0 + priority * 0.12 + duration_bonus

    def _event_score(self, event: PitchEvent, priority: float) -> float:
        # 维度4：将乐器平均置信度折叠进评分（0.5~1.0 区间），使被 F0 校验
        # 确认度更高的乐器在旋律/和弦选择中更有优势。
        avg_conf = self._instrument_avg_conf.get(event.instrument, 1.0)
        effective = priority * (0.5 + 0.5 * avg_conf)
        duration = min(max(0, int(event.duration_ms)), 1200) / 1200.0 * 12.0
        midrange = max(0.0, 14.0 - abs(float(event.midi) - 72.0) * 0.45)
        return float(event.confidence) * 100.0 + effective * 1.4 + duration + midrange

    def _finish_report(
        self,
        stats: TranscribeStats,
        raw_count: int,
        phrases: List[Dict[str, object]],
    ) -> None:
        self.last_report = {
            "arranger_enabled": True,
            "profile": self.config.arranger_profile,
            "raw_event_count": raw_count,
            "candidate_count": stats.arranger_candidate_count,
            "selected_count": stats.arranger_selected_count,
            "deleted_count": stats.arranger_deleted_count,
            "boundary_hit_count": stats.arranger_boundary_hit_count,
            "phrase_count": stats.arranger_phrase_count,
            "phrases": phrases,
        }

    @classmethod
    def _is_excluded_instrument(cls, instrument: str) -> bool:
        name = cls._normalized_instrument(instrument)
        return (
            name == "drums"
            or "drum" in name
            or "bass" in name
            or "timpani" in name
        )

    @classmethod
    def _instrument_priority(cls, instrument: str) -> float:
        name = cls._normalized_instrument(instrument)
        if not name:
            return 0.0
        priorities = [
            ("voice", 80.0),
            ("vocal", 80.0),
            ("synth_lead", 75.0),
            ("lead", 72.0),
            ("flute", 70.0),
            ("violin", 68.0),
            ("sax", 64.0),
            ("trumpet", 62.0),
            ("clarinet", 60.0),
            ("oboe", 60.0),
            ("cello", 58.0),
            ("acoustic_piano", 56.0),
            ("electric_piano", 54.0),
            ("piano", 54.0),
            ("guitar", 50.0),
            ("string", 46.0),
            ("organ", 35.0),
            ("chromatic_percussion", 30.0),
            ("synth_pad", 24.0),
        ]
        for needle, score in priorities:
            if needle in name:
                return score
        return 10.0

    @staticmethod
    def _normalized_instrument(instrument: str) -> str:
        return str(instrument or "").strip().lower().replace(" ", "_").replace("-", "_")
