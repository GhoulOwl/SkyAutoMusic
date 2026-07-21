from __future__ import annotations

import json
import struct
from typing import Any, Dict, Iterable, List, Tuple

from .mapping import SkyKeyMapper
from .models import PitchEvent, SkyNote, TranscribeConfig


class SkyJsonExporter:
    """Write SkyAutoMusic-compatible JSON."""

    def write_song(self, song: Dict[str, Any], out_path: str) -> str:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([song], f, ensure_ascii=False)
        return out_path


class MidiExporter:
    """Small Standard MIDI File writer used for previews and diagnostics."""

    def __init__(self, config: TranscribeConfig):
        self.config = config
        self.mapper = SkyKeyMapper(config)

    def write_song_notes(
        self,
        notes: Iterable[Dict[str, Any]],
        out_path: str,
        bpm: int = 120,
    ) -> str:
        midi_events: List[Tuple[int, int]] = []
        for note in notes:
            time_ms = int(note.get("time", 0))
            key = str(note.get("key", "1Key0"))
            midi_events.append((time_ms, self.mapper.key_to_midi(key)))
        return self.write_midi_events(midi_events, out_path, bpm=bpm)

    def write_sky_notes(
        self,
        notes: Iterable[SkyNote],
        out_path: str,
        bpm: int = 120,
    ) -> str:
        midi_events = [
            (int(note.time), self.mapper.key_to_midi(note.key))
            for note in notes
        ]
        return self.write_midi_events(midi_events, out_path, bpm=bpm)

    def write_pitch_events(
        self,
        events: Iterable[PitchEvent],
        out_path: str,
        bpm: int = 120,
    ) -> str:
        midi_events = [
            (int(event.time_ms), int(round(event.midi)))
            for event in events
        ]
        return self.write_midi_events(midi_events, out_path, bpm=bpm)

    def write_midi_events(
        self,
        midi_events: Iterable[Tuple[int, int]],
        out_path: str,
        bpm: int = 120,
    ) -> str:
        events_list = [(max(0, int(t)), int(m)) for t, m in midi_events]
        division = 480
        tempo_us = self._tempo_us(bpm)

        def ms_to_ticks(ms: float) -> int:
            return int(round(ms * 1000 * division / tempo_us))

        times = sorted({t for t, _m in events_list})
        dur_for_time: Dict[int, int] = {}
        for i, time_ms in enumerate(times):
            duration = times[i + 1] - time_ms if i + 1 < len(times) else 250
            dur_for_time[time_ms] = max(60, min(duration, 800))

        track_events: List[Tuple[int, int, int, int]] = []
        for time_ms, midi in events_list:
            note = max(0, min(127, int(midi)))
            start = ms_to_ticks(time_ms)
            end = ms_to_ticks(time_ms + dur_for_time.get(time_ms, 250))
            track_events.append((start, 1, 0x90, note))
            track_events.append((end, 0, 0x80, note))

        track_events.sort(key=lambda event: (event[0], event[1], event[3]))

        track = bytearray()
        track += b"\x00\xFF\x51\x03" + struct.pack(">I", tempo_us)[1:]
        prev_tick = 0
        for tick, _order, status, note in track_events:
            delta = max(0, tick - prev_tick)
            track += self._vlq(delta)
            track += bytes((status, note, 80))
            prev_tick = tick
        track += b"\x00\xFF\x2F\x00"

        with open(out_path, "wb") as f:
            f.write(b"MThd" + struct.pack(">I", 6) + struct.pack(">HHH", 0, 1, division))
            f.write(b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
        return out_path

    @staticmethod
    def _tempo_us(bpm: int) -> int:
        if not bpm:
            return 500000
        return min(int(60000000 / max(1, int(bpm))), 0xFFFFFF)

    @staticmethod
    def _vlq(value: int) -> bytes:
        if value < 0:
            value = 0
        buf = [value & 0x7F]
        value >>= 7
        while value > 0:
            buf.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(buf))
