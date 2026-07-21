from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .analysis import AnalyzerFactory
from .arranger import SkyMelodyArranger
from .exporters import MidiExporter, SkyJsonExporter
from .mapping import SkyKeyMapper
from .models import PitchEvent, SkyNote, TranscribeConfig, TranscribeStats
from .postprocess import NotePostProcessor


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


class Transcriber:
    """Compatibility facade for the refactored transcription pipeline."""

    NUM_KEYS = 15
    DEFAULT_SR = 44100
    ONSET_PITCH_WINDOW = 0.13
    MAX_POLYPHONY = 3
    PITCH_MAG_THRESHOLD = 0.10
    ONSET_DELTA = 0.07

    def __init__(
        self,
        config: Optional[TranscribeConfig] = None,
        sr: int = DEFAULT_SR,
        midi_root: int = 60,
    ):
        if isinstance(config, int):
            legacy_sr = int(config)
            legacy_midi_root = int(sr) if sr != self.DEFAULT_SR else midi_root
            sr = legacy_sr
            midi_root = legacy_midi_root
            config = None
        if config is None:
            config = TranscribeConfig(sr=sr, midi_root=midi_root)
        elif sr != self.DEFAULT_SR or midi_root != 60:
            config = replace(config, sr=sr, midi_root=midi_root)

        self.config = config
        self.sr = config.sr
        self.midi_root = config.midi_root
        self.mapper = SkyKeyMapper(config)
        self.arranger = SkyMelodyArranger(config)
        self.postprocessor = NotePostProcessor(config)
        self.json_exporter = SkyJsonExporter()
        self.midi_exporter = MidiExporter(config)
        self.analyzer, self._engine_error = AnalyzerFactory.create(config)

        self._last_raw_events: List[PitchEvent] = []
        self._last_melody_candidates: List[SkyNote] = []
        self._last_arranged_notes: List[SkyNote] = []
        self._last_mapped_notes: List[SkyNote] = []
        self._last_final_notes: List[SkyNote] = []

    def transcribe(self, audio_path: str) -> Tuple[List[Dict[str, Any]], TranscribeStats]:
        raw_events, stats = self._analyze(audio_path)
        stats.mapping = self.config.mapping
        stats.out_of_range_policy = self.config.out_of_range_policy
        stats.raw_event_count = len(raw_events)
        stats.arranger_enabled = bool(self.config.arranger_enabled)
        self._last_raw_events = list(raw_events)
        self._last_melody_candidates = []
        self._last_arranged_notes = []
        self._last_mapped_notes = []
        self._last_final_notes = []

        if not raw_events:
            return [], stats

        if self.config.arranger_enabled:
            mapped_notes, stats = self.arranger.arrange(raw_events, stats)
            self._last_melody_candidates = list(self.arranger.last_candidates)
            self._last_arranged_notes = list(self.arranger.last_arranged)
        else:
            mapped_notes = self._map_events_legacy(raw_events, stats)
        self._last_mapped_notes = list(mapped_notes)

        final_notes, dropped_count, estimated_bpm = self.postprocessor.process(mapped_notes)
        stats.dropped_count += dropped_count
        stats.estimated_bpm = estimated_bpm
        stats.note_count = len(final_notes)
        self._last_final_notes = list(final_notes)

        return [note.to_song_note() for note in final_notes], stats

    def _map_events_legacy(
        self,
        raw_events: List[PitchEvent],
        stats: TranscribeStats,
    ) -> List[SkyNote]:
        stats.arranger_enabled = False
        transpose_events = self._events_for_transpose(raw_events)
        transpose = self.mapper.compute_transpose(event.midi for event in transpose_events)
        stats.transpose = transpose

        mapped_notes: List[SkyNote] = []
        for event in raw_events:
            if self._should_drop_out_of_range(event, transpose):
                stats.dropped_count += 1
                continue
            note, clamped = self.mapper.map_event(event, transpose=transpose)
            if clamped < 0:
                stats.clamped_low += 1
            elif clamped > 0:
                stats.clamped_high += 1
            mapped_notes.append(note)
        stats.mapped_event_count = len(mapped_notes)
        return mapped_notes

    def transcribe_to_song(
        self,
        audio_path: str,
        song_name: Optional[str] = None,
        bpm: int = 120,
    ) -> Dict[str, Any]:
        notes, stats = self.transcribe(audio_path)
        if song_name is None:
            song_name = os.path.splitext(os.path.basename(audio_path))[0]
        return {
            "name": song_name,
            "bpm": int(bpm),
            "songNotes": notes,
            "_transcribe_stats": stats.to_dict(),
        }

    def write_song_json(
        self,
        audio_path: str,
        output_dir: str,
        song_name: Optional[str] = None,
        bpm: int = 120,
    ) -> Tuple[str, Dict[str, Any]]:
        song = self.transcribe_to_song(audio_path, song_name=song_name, bpm=bpm)
        os.makedirs(output_dir, exist_ok=True)
        base = song_name or os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(output_dir, base + ".json")
        self.json_exporter.write_song(song, out_path)

        midi_path = os.path.join(output_dir, base + ".mid")
        try:
            self.write_midi_file(song["songNotes"], midi_path, bpm=int(bpm))
            song["_midi_path"] = midi_path
        except Exception:
            song["_midi_path"] = None

        if self.config.debug_outputs:
            song["_debug_outputs"] = self._write_debug_outputs(output_dir, base, bpm=int(bpm))
        return out_path, song

    def write_midi_file(
        self,
        notes: List[Dict[str, Any]],
        out_path: str,
        bpm: int = 120,
    ) -> str:
        exporter = MidiExporter(self._config_for_instance())
        return exporter.write_song_notes(notes, out_path, bpm=bpm)

    @staticmethod
    def _vlq(value: int) -> bytes:
        return MidiExporter._vlq(value)

    def run(
        self,
        files: Iterable[str],
        output_dir: str,
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
        bpm: int = 120,
    ) -> List[Dict[str, Any]]:
        files = [path for path in files if is_audio_file(path)]
        results: List[Dict[str, Any]] = []
        total = len(files)
        if total == 0:
            if progress_cb:
                progress_cb("", 1.0, "未选择任何音频文件")
            return results

        for index, file_path in enumerate(files):
            song_name = os.path.splitext(os.path.basename(file_path))[0]
            if progress_cb:
                progress_cb(file_path, index / total, f"({index + 1}/{total}) 正在转写: {song_name}")
            try:
                out_path, song = self.write_song_json(
                    file_path,
                    output_dir,
                    song_name=song_name,
                    bpm=bpm,
                )
                results.append(
                    {
                        "input": file_path,
                        "output": out_path,
                        "midi_output": song.get("_midi_path"),
                        "ok": True,
                        "error": None,
                        "stats": song.get("_transcribe_stats"),
                        "song_name": song_name,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "input": file_path,
                        "output": None,
                        "ok": False,
                        "error": str(exc),
                        "stats": None,
                        "song_name": song_name,
                    }
                )

        if progress_cb:
            ok_count = sum(1 for result in results if result["ok"])
            progress_cb("", 1.0, f"完成 {ok_count}/{total}")
        return results

    def _compute_transpose(self, midis: List[float]) -> int:
        return SkyKeyMapper(self._config_for_instance()).compute_transpose(midis)

    def _midi_to_key(self, midi: float, transpose: int = 0) -> Tuple[str, int]:
        return SkyKeyMapper(self._config_for_instance()).midi_to_key(
            midi,
            transpose=transpose,
        )

    def _analyze(self, audio_path: str) -> Tuple[List[PitchEvent], TranscribeStats]:
        raw_events, stats = self.analyzer.analyze(audio_path)
        if getattr(self, "_engine_error", None) and not stats.engine_error:
            stats.engine_error = self._engine_error
        return raw_events, stats

    def _events_for_transpose(self, events: List[PitchEvent]) -> List[PitchEvent]:
        if self.config.profile != "melody":
            return list(events)
        melody_events = [
            event
            for event in events
            if not NotePostProcessor._is_excluded_melody_instrument(event.instrument)
        ]
        return melody_events or list(events)

    def _should_drop_out_of_range(self, event: PitchEvent, transpose: int) -> bool:
        should_drop = (
            bool(self.config.drop_out_of_range)
            or self.config.out_of_range_policy == "drop"
        )
        if not should_drop:
            return False
        if self.config.profile != "melody":
            return False
        return not self.mapper.is_playable_after_transpose(event.midi, transpose=transpose)

    def _write_debug_outputs(self, output_dir: str, base: str, bpm: int) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        exporter = MidiExporter(self._config_for_instance())
        debug_specs = [
            ("raw", exporter.write_pitch_events, self._last_raw_events),
            ("melody_candidates", exporter.write_sky_notes, self._last_melody_candidates),
            ("arranged", exporter.write_sky_notes, self._last_arranged_notes),
            ("mapped", exporter.write_sky_notes, self._last_mapped_notes),
            ("final", exporter.write_sky_notes, self._last_final_notes),
        ]
        for suffix, writer, payload in debug_specs:
            path = os.path.join(output_dir, f"{base}.{suffix}.mid")
            try:
                writer(payload, path, bpm=bpm)
                paths[suffix] = path
            except Exception:
                continue
        artifacts = getattr(getattr(self, "analyzer", None), "last_artifacts", {})
        artifact_specs = {
            "events_jsonl": f"{base}.muscriptor.events.jsonl",
            "arranger_report": f"{base}.arranger.report.json",
        }
        if self.config.arranger_enabled and self.arranger.last_report:
            artifacts = dict(artifacts)
            artifacts["arranger_report"] = self.arranger.last_report_bytes
        for artifact_name, filename in artifact_specs.items():
            payload = artifacts.get(artifact_name)
            if not payload:
                continue
            path = os.path.join(output_dir, filename)
            try:
                with open(path, "wb") as file_obj:
                    file_obj.write(payload)
                paths[f"muscriptor_{artifact_name}"] = path
            except Exception:
                continue
        return paths

    def _config_for_instance(self) -> TranscribeConfig:
        config = getattr(self, "config", None)
        if isinstance(config, TranscribeConfig):
            return config
        return TranscribeConfig(
            sr=int(getattr(self, "sr", self.DEFAULT_SR)),
            midi_root=int(getattr(self, "midi_root", 60)),
            max_polyphony=int(getattr(self, "MAX_POLYPHONY", 1)),
            onset_pitch_window=float(getattr(self, "ONSET_PITCH_WINDOW", 0.13)),
            pitch_mag_threshold=float(getattr(self, "PITCH_MAG_THRESHOLD", 0.10)),
            onset_delta=float(getattr(self, "ONSET_DELTA", 0.07)),
        )
