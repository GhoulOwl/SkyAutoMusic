from __future__ import annotations

from typing import Iterable, List, Tuple

from .models import PitchEvent, SkyNote, TranscribeConfig


SKY_MAJOR_OFFSETS = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24]
CHROMATIC_OFFSETS = list(range(15))
SKY_MAJOR_MIN_OFFSET = SKY_MAJOR_OFFSETS[0]
SKY_MAJOR_MAX_OFFSET = SKY_MAJOR_OFFSETS[-1]
SKY_MAJOR_EDGE_SNAP_SEMITONES = 1
SKY_MAJOR_TRANSPOSE_RANGE = range(-24, 25)


class SkyKeyMapper:
    """Map MIDI pitches into the 15 playable Sky keys."""

    NUM_KEYS = 15

    def __init__(self, config: TranscribeConfig):
        self.config = config
        self.mapping = config.mapping
        self.midi_root = config.midi_root

    @property
    def offsets(self) -> List[int]:
        if self.mapping == "chromatic":
            return CHROMATIC_OFFSETS
        return SKY_MAJOR_OFFSETS

    def compute_transpose(self, midis: Iterable[float]) -> int:
        midis = list(midis)
        if not midis:
            return 0
        if self.mapping == "chromatic":
            return self._compute_chromatic_transpose(midis)
        return self._compute_sky_major_transpose(midis)

    def midi_to_key(self, midi: float, transpose: int = 0) -> Tuple[str, int]:
        if self.mapping == "chromatic":
            return self._chromatic_midi_to_key(midi, transpose)
        return self._sky_major_midi_to_key(midi, transpose)

    def map_event(self, event: PitchEvent, transpose: int = 0) -> Tuple[SkyNote, int]:
        key, clamped = self.midi_to_key(event.midi, transpose=transpose)
        note_midi = self.key_to_midi(key)
        mapped = SkyNote(
            time=int(event.time_ms),
            key=key,
            midi=float(note_midi),
            confidence=float(event.confidence),
            duration_ms=int(event.duration_ms),
            instrument=str(event.instrument or ""),
        )
        return mapped, clamped

    def key_to_midi(self, key: str) -> int:
        idx = self.key_to_index(key)
        offsets = self.offsets
        idx = max(0, min(len(offsets) - 1, idx))
        return int(self.midi_root + offsets[idx])

    def is_playable_after_transpose(self, midi: float, transpose: int = 0) -> bool:
        """Return whether a pitch is inside Sky's two-octave playable window."""
        diff = int(round(midi)) + int(transpose) - self.midi_root
        if self.mapping == "chromatic":
            return 0 <= diff < self.NUM_KEYS
        return SKY_MAJOR_MIN_OFFSET <= diff <= SKY_MAJOR_MAX_OFFSET

    @staticmethod
    def key_to_index(key: str) -> int:
        if isinstance(key, str) and key.startswith("1Key"):
            try:
                return int(key[4:])
            except ValueError:
                return 0
        if isinstance(key, str) and key.startswith("2Key"):
            try:
                return int(key[4:])
            except ValueError:
                return 0
        return 0

    def nearest_mapping_distance(self, midi: float, transpose: int = 0) -> int:
        diff = int(round(midi)) + int(transpose) - self.midi_root
        if self.mapping == "chromatic":
            folded = self._fold_offset(diff, CHROMATIC_OFFSETS[-1])
            return 0 if 0 <= folded < self.NUM_KEYS else 999
        bounded = self._clamp_offset(diff, SKY_MAJOR_MIN_OFFSET, SKY_MAJOR_MAX_OFFSET)
        outside = self._outside_distance(diff, SKY_MAJOR_MIN_OFFSET, SKY_MAJOR_MAX_OFFSET)
        return outside * 8 + min(abs(bounded - offset) for offset in SKY_MAJOR_OFFSETS)

    def _compute_chromatic_transpose(self, midis: List[float]) -> int:
        best_t = 0
        best_score = -1
        int_midis = [int(round(m)) for m in midis]
        for transpose in range(-6, 7):
            score = sum(
                1
                for midi in int_midis
                if 0 <= midi + transpose - self.midi_root < self.NUM_KEYS
            )
            if score > best_score or (
                score == best_score and abs(transpose) < abs(best_t)
            ):
                best_score = score
                best_t = transpose
        return best_t

    def _compute_sky_major_transpose(self, midis: List[float]) -> int:
        best = None
        best_t = 0
        for transpose in SKY_MAJOR_TRANSPOSE_RANGE:
            total_distance = 0
            outside_distance = 0
            outside_count = 0
            native_count = 0
            for midi in midis:
                diff = int(round(midi)) + transpose - self.midi_root
                outside = self._outside_distance(
                    diff,
                    SKY_MAJOR_MIN_OFFSET,
                    SKY_MAJOR_MAX_OFFSET,
                )
                if outside == 0:
                    native_count += 1
                else:
                    outside_count += 1
                    outside_distance += outside
                bounded = self._clamp_offset(
                    diff,
                    SKY_MAJOR_MIN_OFFSET,
                    SKY_MAJOR_MAX_OFFSET,
                )
                total_distance += min(abs(bounded - off) for off in SKY_MAJOR_OFFSETS)
            candidate = (
                outside_count,
                outside_distance,
                total_distance,
                -native_count,
                abs(transpose),
                transpose,
            )
            if best is None or candidate < best:
                best = candidate
                best_t = transpose
        return best_t

    def _chromatic_midi_to_key(self, midi: float, transpose: int = 0) -> Tuple[str, int]:
        index = int(round(midi)) + int(transpose) - self.midi_root
        clamped = -1 if index < 0 else 1 if index >= self.NUM_KEYS else 0
        while index < 0:
            index += 12
        while index >= self.NUM_KEYS:
            index -= 12
        index = max(0, min(self.NUM_KEYS - 1, index))
        return f"1Key{index}", clamped

    def _sky_major_midi_to_key(self, midi: float, transpose: int = 0) -> Tuple[str, int]:
        diff = int(round(midi)) + int(transpose) - self.midi_root
        clamped = (
            -1
            if diff < SKY_MAJOR_MIN_OFFSET
            else 1
            if diff > SKY_MAJOR_MAX_OFFSET
            else 0
        )
        index = self._project_sky_major_offset_to_index(diff)
        return f"1Key{index}", clamped

    def _project_sky_major_offset_to_index(self, offset: int) -> int:
        offset = int(offset)
        if SKY_MAJOR_MIN_OFFSET <= offset <= SKY_MAJOR_MAX_OFFSET:
            return self._nearest_sky_major_index(offset)

        policy = str(getattr(self.config, "out_of_range_policy", "adaptive") or "adaptive")
        if policy == "drop":
            policy = "adaptive"
        if policy == "clamp":
            bounded = self._clamp_offset(
                offset,
                SKY_MAJOR_MIN_OFFSET,
                SKY_MAJOR_MAX_OFFSET,
            )
            return self._nearest_sky_major_index(bounded)
        if policy == "octave":
            return self._octave_project_sky_major_index(offset)

        outside = self._outside_distance(
            offset,
            SKY_MAJOR_MIN_OFFSET,
            SKY_MAJOR_MAX_OFFSET,
        )
        if outside <= SKY_MAJOR_EDGE_SNAP_SEMITONES:
            bounded = self._clamp_offset(
                offset,
                SKY_MAJOR_MIN_OFFSET,
                SKY_MAJOR_MAX_OFFSET,
            )
            return self._nearest_sky_major_index(bounded)
        return self._octave_project_sky_major_index(offset)

    @staticmethod
    def _nearest_sky_major_index(offset: int) -> int:
        return min(
            range(len(SKY_MAJOR_OFFSETS)),
            key=lambda i: (abs(SKY_MAJOR_OFFSETS[i] - int(offset)), i),
        )

    @staticmethod
    def _octave_project_sky_major_index(offset: int) -> int:
        """Project out-of-range notes by preserving scale degree when possible."""
        offset = int(offset)
        return min(
            range(len(SKY_MAJOR_OFFSETS)),
            key=lambda i: (
                SkyKeyMapper._pitch_class_distance(offset, SKY_MAJOR_OFFSETS[i]),
                abs(offset - SKY_MAJOR_OFFSETS[i]),
                i,
            ),
        )

    @staticmethod
    def _pitch_class_distance(left: int, right: int) -> int:
        diff = abs((int(left) - int(right)) % 12)
        return min(diff, 12 - diff)

    @staticmethod
    def _clamp_offset(offset: int, min_offset: int, max_offset: int) -> int:
        return max(min_offset, min(max_offset, int(offset)))

    @staticmethod
    def _outside_distance(offset: int, min_offset: int, max_offset: int) -> int:
        offset = int(offset)
        if offset < min_offset:
            return min_offset - offset
        if offset > max_offset:
            return offset - max_offset
        return 0

    @staticmethod
    def _fold_offset(offset: int, max_offset: int) -> int:
        folded = int(offset)
        while folded < 0:
            folded += 12
        while folded > max_offset:
            folded -= 12
        if folded < 0:
            return 0
        if folded > max_offset:
            return max_offset
        return folded
