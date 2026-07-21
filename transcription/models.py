from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PitchEvent:
    """Raw detected pitch at an onset before Sky key mapping."""

    time_ms: int
    midi: float
    confidence: float = 1.0
    source: str = "muscriptor"
    duration_ms: int = 0
    instrument: str = ""


@dataclass
class SkyNote:
    """Mapped playable Sky note with diagnostic metadata."""

    time: int
    key: str
    midi: float
    confidence: float = 1.0
    duration_ms: int = 0
    instrument: str = ""

    def to_song_note(self) -> Dict[str, Any]:
        return {"time": int(self.time), "key": self.key}


@dataclass
class TranscribeConfig:
    """User-tunable transcription behavior."""

    engine: str = "muscriptor"
    profile: str = "melody"
    mapping: str = "sky_major"
    max_polyphony: int = 3
    quantize: bool = True
    debug_outputs: bool = False
    sr: int = 44100
    midi_root: int = 60
    onset_pitch_window: float = 0.13
    pitch_mag_threshold: float = 0.10
    onset_delta: float = 0.07
    merge_onset_ms: int = 55
    dedupe_key_ms: int = 120
    min_confidence: float = 0.35
    min_note_duration_ms: int = 90
    min_event_gap_ms: int = 100
    drop_out_of_range: bool = False
    out_of_range_policy: str = "adaptive"
    arranger_enabled: bool = True
    arranger_profile: str = "sky_melody"
    arranger_cluster_ms: int = 90
    arranger_phrase_gap_ms: int = 900
    arranger_min_note_gap_ms: int = 120
    arranger_keep_chords: bool = False
    muscriptor_model: str = "large"
    muscriptor_device: str = "auto"
    muscriptor_instruments: Optional[List[str]] = None
    muscriptor_batch_size: Optional[int] = None
    muscriptor_beam_size: int = 1
    muscriptor_cfg_coef: float = 1.0
    muscriptor_prelude_forcing: bool = True

    def __post_init__(self) -> None:
        self.engine = str(self.engine or "muscriptor").lower()
        if self.engine != "muscriptor":
            raise ValueError(
                "Only the MuScriptor transcription engine is supported. "
                "Set engine='muscriptor' or omit the engine setting."
            )
        self.profile = str(self.profile or "melody").lower()
        if self.profile not in {"melody", "chord"}:
            raise ValueError(f"Unknown transcription profile: {self.profile}")
        self.max_polyphony = max(1, int(self.max_polyphony))
        self.sr = int(self.sr)
        self.midi_root = int(self.midi_root)
        self.merge_onset_ms = max(0, int(self.merge_onset_ms))
        self.dedupe_key_ms = max(0, int(self.dedupe_key_ms))
        self.min_confidence = max(0.0, min(1.0, float(self.min_confidence)))
        self.min_note_duration_ms = max(0, int(self.min_note_duration_ms))
        self.min_event_gap_ms = max(0, int(self.min_event_gap_ms))
        self.drop_out_of_range = bool(self.drop_out_of_range)
        self.out_of_range_policy = str(
            self.out_of_range_policy or "adaptive"
        ).strip().lower()
        if self.out_of_range_policy not in {"adaptive", "clamp", "octave", "drop"}:
            raise ValueError(f"Unknown out-of-range policy: {self.out_of_range_policy}")
        self.arranger_enabled = bool(self.arranger_enabled)
        self.arranger_profile = str(self.arranger_profile or "sky_melody").strip().lower()
        if self.arranger_profile != "sky_melody":
            raise ValueError(f"Unknown arranger profile: {self.arranger_profile}")
        self.arranger_cluster_ms = max(0, int(self.arranger_cluster_ms))
        self.arranger_phrase_gap_ms = max(0, int(self.arranger_phrase_gap_ms))
        self.arranger_min_note_gap_ms = max(0, int(self.arranger_min_note_gap_ms))
        self.arranger_keep_chords = bool(self.arranger_keep_chords)
        self.muscriptor_model = str(self.muscriptor_model or "large").strip().lower()
        if self.muscriptor_model not in {"small", "medium", "large"}:
            raise ValueError(f"Unknown MuScriptor model size: {self.muscriptor_model}")
        self.muscriptor_device = str(self.muscriptor_device or "auto").lower()
        if self.muscriptor_instruments is not None:
            self.muscriptor_instruments = [
                str(name).strip()
                for name in self.muscriptor_instruments
                if str(name).strip()
            ]
        self.muscriptor_batch_size = (
            None
            if self.muscriptor_batch_size is None
            else max(1, int(self.muscriptor_batch_size))
        )
        self.muscriptor_beam_size = max(1, int(self.muscriptor_beam_size))
        self.muscriptor_cfg_coef = float(self.muscriptor_cfg_coef)


@dataclass
class TranscribeStats:
    """Single transcription run statistics for UI and tests."""

    onset_count: int = 0
    note_count: int = 0
    clamped_low: int = 0
    clamped_high: int = 0
    duration_sec: float = 0.0
    transpose: int = 0
    raw_event_count: int = 0
    mapped_event_count: int = 0
    dropped_count: int = 0
    estimated_bpm: Optional[int] = None
    mapping: str = "sky_major"
    out_of_range_policy: str = "adaptive"
    engine_requested: str = "muscriptor"
    engine_used: str = ""
    engine_error: Optional[str] = None
    muscriptor_model: str = "large"
    muscriptor_device: str = "auto"
    instrument_counts: Dict[str, int] = field(default_factory=dict)
    arranger_enabled: bool = False
    arranger_phrase_count: int = 0
    arranger_candidate_count: int = 0
    arranger_selected_count: int = 0
    arranger_deleted_count: int = 0
    arranger_boundary_hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "onset_count": self.onset_count,
            "note_count": self.note_count,
            "clamped_low": self.clamped_low,
            "clamped_high": self.clamped_high,
            "duration_sec": round(self.duration_sec, 3),
            "transpose": self.transpose,
            "raw_event_count": self.raw_event_count,
            "mapped_event_count": self.mapped_event_count,
            "dropped_count": self.dropped_count,
            "estimated_bpm": self.estimated_bpm,
            "mapping": self.mapping,
            "out_of_range_policy": self.out_of_range_policy,
            "engine_requested": self.engine_requested,
            "engine_used": self.engine_used,
            "engine_error": self.engine_error,
            "muscriptor_model": self.muscriptor_model,
            "muscriptor_device": self.muscriptor_device,
            "instrument_counts": dict(self.instrument_counts),
            "arranger_enabled": self.arranger_enabled,
            "arranger_phrase_count": self.arranger_phrase_count,
            "arranger_candidate_count": self.arranger_candidate_count,
            "arranger_selected_count": self.arranger_selected_count,
            "arranger_deleted_count": self.arranger_deleted_count,
            "arranger_boundary_hit_count": self.arranger_boundary_hit_count,
        }
