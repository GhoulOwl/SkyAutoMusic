from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .mapping import SkyKeyMapper
from .models import SkyNote, TranscribeConfig


class NotePostProcessor:
    """Clean mapped notes before exporting playable Sky JSON."""

    def __init__(self, config: TranscribeConfig):
        self.config = config

    def process(self, notes: Iterable[SkyNote]) -> Tuple[List[SkyNote], int, Optional[int]]:
        work = sorted(list(notes), key=lambda n: (n.time, n.key, -n.confidence))
        if not work:
            return [], 0, None

        dropped = 0
        work, count = self._filter_noise_floor(work)
        dropped += count
        if not work:
            return [], dropped, None

        work, count = self._merge_close_onsets(work, self.config.merge_onset_ms)
        dropped += count
        work, count = self._dedupe_repeated_keys(work, self.config.dedupe_key_ms)
        dropped += count

        # 维度5：用初步 BPM 计算自适应 gap（避免快旋律被固定窗口抽稀）。
        preliminary_bpm = self._estimate_bpm(work)
        work, count = self._apply_profile(work, preliminary_bpm)
        dropped += count

        estimated_bpm = self._estimate_bpm(work)
        if self.config.quantize and estimated_bpm:
            work = self._quantize(work, estimated_bpm, self.config.quantize_strength)
            work, count = self._merge_exact_times(work)
            dropped += count
            work, count = self._dedupe_repeated_keys(work, self.config.dedupe_key_ms)
            dropped += count
            work, count = self._apply_profile(work, estimated_bpm)
            dropped += count

        work.sort(key=lambda n: (n.time, SkyKeyMapper.key_to_index(n.key), n.key))
        return work, dropped, estimated_bpm

    def _merge_close_onsets(
        self, notes: List[SkyNote], window_ms: int
    ) -> Tuple[List[SkyNote], int]:
        if window_ms <= 0:
            return self._merge_exact_times(notes)

        clusters: List[List[SkyNote]] = []
        current: List[SkyNote] = []
        cluster_start = 0
        for note in notes:
            if not current:
                current = [note]
                cluster_start = int(note.time)
                continue
            if int(note.time) - cluster_start <= window_ms:
                current.append(note)
            else:
                clusters.append(current)
                current = [note]
                cluster_start = int(note.time)
        if current:
            clusters.append(current)

        merged: List[SkyNote] = []
        dropped = 0
        for cluster in clusters:
            cluster_time = self._weighted_time(cluster)
            best_by_key: Dict[str, SkyNote] = {}
            for note in cluster:
                candidate = SkyNote(
                    time=cluster_time,
                    key=note.key,
                    midi=note.midi,
                    confidence=note.confidence,
                    duration_ms=note.duration_ms,
                    instrument=note.instrument,
                    confidence_source=note.confidence_source,
                )
                existing = best_by_key.get(note.key)
                if existing is None or candidate.confidence > existing.confidence:
                    if existing is not None:
                        dropped += 1
                    best_by_key[note.key] = candidate
                else:
                    dropped += 1
            merged.extend(best_by_key.values())
        merged.sort(key=lambda n: (n.time, n.key, -n.confidence))
        return merged, dropped

    def _merge_exact_times(self, notes: List[SkyNote]) -> Tuple[List[SkyNote], int]:
        best: Dict[Tuple[int, str], SkyNote] = {}
        dropped = 0
        for note in notes:
            key = (int(note.time), note.key)
            existing = best.get(key)
            if existing is None or note.confidence > existing.confidence:
                if existing is not None:
                    dropped += 1
                best[key] = note
            else:
                dropped += 1
        merged = list(best.values())
        merged.sort(key=lambda n: (n.time, n.key, -n.confidence))
        return merged, dropped

    def _dedupe_repeated_keys(
        self, notes: List[SkyNote], window_ms: int
    ) -> Tuple[List[SkyNote], int]:
        if window_ms <= 0:
            return list(notes), 0
        kept: List[SkyNote] = []
        last_time_by_key: Dict[str, int] = {}
        dropped = 0
        for note in sorted(notes, key=lambda n: (n.time, n.key, -n.confidence)):
            last = last_time_by_key.get(note.key)
            if last is not None and int(note.time) - last < window_ms:
                dropped += 1
                continue
            kept.append(note)
            last_time_by_key[note.key] = int(note.time)
        return kept, dropped

    def _cap_polyphony(
        self, notes: List[SkyNote], max_polyphony: int
    ) -> Tuple[List[SkyNote], int]:
        grouped: Dict[int, List[SkyNote]] = defaultdict(list)
        for note in notes:
            grouped[int(note.time)].append(note)

        capped: List[SkyNote] = []
        dropped = 0
        for time_ms in sorted(grouped):
            group = grouped[time_ms]
            group.sort(
                key=lambda n: (-n.confidence, SkyKeyMapper.key_to_index(n.key), n.key)
            )
            keep = group[:max_polyphony]
            dropped += max(0, len(group) - len(keep))
            keep.sort(key=lambda n: (SkyKeyMapper.key_to_index(n.key), n.key))
            capped.extend(keep)
        return capped, dropped

    def _estimate_bpm(self, notes: List[SkyNote]) -> Optional[int]:
        times = sorted({int(n.time) for n in notes})
        if len(times) < 3:
            return None
        intervals = [
            times[i + 1] - times[i]
            for i in range(len(times) - 1)
            if 80 <= times[i + 1] - times[i] <= 2000
        ]
        if len(intervals) < 2:
            return None

        median_interval = float(statistics.median(intervals))
        candidates = []
        for factor in (1, 2, 4, 8):
            beat_ms = median_interval * factor
            if beat_ms <= 0:
                continue
            bpm = 60000.0 / beat_ms
            if 50 <= bpm <= 220:
                candidates.append((abs(bpm - 120.0), factor, bpm))
        if not candidates:
            return None
        _distance, _factor, bpm = min(candidates)
        return int(round(bpm))

    def _quantize(self, notes: List[SkyNote], bpm: int, strength: float = 1.0) -> List[SkyNote]:
        if not notes or bpm <= 0:
            return list(notes)
        beat_ms = 60000.0 / float(bpm)
        grid_ms = beat_ms / 4.0
        if grid_ms <= 0:
            return list(notes)
        base = min(int(n.time) for n in notes)
        strength = max(0.0, min(1.0, float(strength)))
        quantized: List[SkyNote] = []
        for note in notes:
            offset = int(note.time) - base
            q_time = int(round(round(offset / grid_ms) * grid_ms + base))
            # 维度5：按 quantize_strength 部分吸附，避免强量化把全曲时间推偏。
            snapped = int(note.time + (q_time - int(note.time)) * strength)
            quantized.append(
                SkyNote(
                    time=max(0, snapped),
                    key=note.key,
                    midi=note.midi,
                    confidence=note.confidence,
                    duration_ms=note.duration_ms,
                    instrument=note.instrument,
                    confidence_source=note.confidence_source,
                )
            )
        quantized.sort(key=lambda n: (n.time, n.key, -n.confidence))
        return quantized

    def _filter_noise_floor(self, notes: List[SkyNote]) -> Tuple[List[SkyNote], int]:
        kept: List[SkyNote] = []
        dropped = 0
        for note in notes:
            if self.config.profile == "melody" and self._is_excluded_melody_instrument(
                note.instrument
            ):
                dropped += 1
                continue
            if float(note.confidence) < self.config.min_confidence:
                dropped += 1
                continue
            if (
                int(note.duration_ms) > 0
                and int(note.duration_ms) < self.config.min_note_duration_ms
            ):
                dropped += 1
                continue
            kept.append(note)
        return kept, dropped

    def _apply_profile(
        self, notes: List[SkyNote], bpm: Optional[int] = None
    ) -> Tuple[List[SkyNote], int]:
        if self.config.profile == "melody" and not self.config.arranger_keep_chords:
            work, dropped = self._select_melody_line(notes)
            gap = (
                self._adaptive_gap(bpm)
                if self.config.adaptive_gap
                else self.config.min_event_gap_ms
            )
            work, count = self._thin_dense_melody(work, gap)
            return work, dropped + count
        # 和弦 / 多声部：用复音上限封顶，不再做单旋律抽稀。
        if self.config.arranger_keep_chords:
            cap = max(self.config.max_polyphony, self.config.max_chord_voices)
            return self._cap_polyphony(notes, cap)
        return self._cap_polyphony(notes, self.config.max_polyphony)

    def _adaptive_gap(self, bpm: Optional[int]) -> int:
        """维度3/5：gap 取真实节拍的十六分音符，避免固定阈值误删快旋律。"""
        if bpm:
            grid = 60000.0 / float(bpm) / 4.0
            return max(self.config.min_event_gap_ms, int(round(grid)))
        return self.config.min_event_gap_ms

    def _select_melody_line(self, notes: List[SkyNote]) -> Tuple[List[SkyNote], int]:
        grouped: Dict[int, List[SkyNote]] = defaultdict(list)
        for note in notes:
            grouped[int(note.time)].append(note)

        kept: List[SkyNote] = []
        dropped = 0
        previous: Optional[SkyNote] = None
        for time_ms in sorted(grouped):
            group = grouped[time_ms]
            best = max(group, key=lambda note: self._melody_score(note, previous))
            kept.append(best)
            previous = best
            dropped += max(0, len(group) - 1)
        kept.sort(key=lambda n: (n.time, SkyKeyMapper.key_to_index(n.key), n.key))
        return kept, dropped

    def _thin_dense_melody(
        self, notes: List[SkyNote], window_ms: int
    ) -> Tuple[List[SkyNote], int]:
        if window_ms <= 0 or len(notes) <= 1:
            return list(notes), 0

        clusters: List[List[SkyNote]] = []
        current: List[SkyNote] = []
        cluster_start = 0
        for note in sorted(notes, key=lambda n: (n.time, -n.confidence)):
            if not current:
                current = [note]
                cluster_start = int(note.time)
                continue
            if int(note.time) - cluster_start < window_ms:
                current.append(note)
            else:
                clusters.append(current)
                current = [note]
                cluster_start = int(note.time)
        if current:
            clusters.append(current)

        kept: List[SkyNote] = []
        dropped = 0
        previous: Optional[SkyNote] = None
        for cluster in clusters:
            best = max(cluster, key=lambda note: self._melody_score(note, previous))
            kept.append(best)
            previous = best
            dropped += max(0, len(cluster) - 1)
        kept.sort(key=lambda n: (n.time, SkyKeyMapper.key_to_index(n.key), n.key))
        return kept, dropped

    @staticmethod
    def _melody_score(note: SkyNote, previous: Optional[SkyNote]) -> float:
        key_index = SkyKeyMapper.key_to_index(note.key)
        score = float(note.confidence) * 100.0
        score += NotePostProcessor._instrument_priority(note.instrument)
        score += max(0.0, 12.0 - abs(float(note.midi) - 72.0) * 0.55)
        score += min(max(0, int(note.duration_ms)), 1200) / 1200.0 * 8.0
        score += key_index * 0.4
        if previous is not None:
            previous_index = SkyKeyMapper.key_to_index(previous.key)
            score -= min(abs(key_index - previous_index), 12) * 2.0
        return score

    @staticmethod
    def _normalized_instrument(instrument: str) -> str:
        return str(instrument or "").strip().lower().replace(" ", "_").replace("-", "_")

    @classmethod
    def _is_excluded_melody_instrument(cls, instrument: str) -> bool:
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
    def _weighted_time(notes: List[SkyNote]) -> int:
        total = sum(max(0.001, float(n.confidence)) for n in notes)
        value = sum(int(n.time) * max(0.001, float(n.confidence)) for n in notes)
        return int(round(value / total))
