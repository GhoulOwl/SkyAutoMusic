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
    confidence_source: str = "muscriptor"  # muscriptor | f0 | gap

    def to_sky_note_kwargs(self) -> Dict[str, Any]:
        return {
            "midi": self.midi,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "instrument": self.instrument,
            "confidence_source": self.confidence_source,
        }


@dataclass
class SkyNote:
    """Mapped playable Sky note with diagnostic metadata."""

    time: int
    key: str
    midi: float
    confidence: float = 1.0
    duration_ms: int = 0
    instrument: str = ""
    confidence_source: str = "muscriptor"

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
    merge_onset_ms: int = 25
    dedupe_key_ms: int = 120
    min_confidence: float = 0.35
    min_note_duration_ms: int = 50
    min_event_gap_ms: int = 0
    adaptive_gap: bool = True
    drop_out_of_range: bool = False
    out_of_range_policy: str = "adaptive"
    arranger_enabled: bool = True
    arranger_profile: str = "sky_melody"
    arranger_cluster_ms: int = 90
    arranger_phrase_gap_ms: int = 900
    arranger_min_note_gap_ms: int = 120
    arranger_keep_chords: bool = False
    max_chord_voices: int = 3
    quantize_strength: float = 0.3
    muscriptor_model: str = "large"
    muscriptor_device: str = "auto"
    muscriptor_instruments: Optional[List[str]] = None
    muscriptor_batch_size: Optional[int] = None
    muscriptor_beam_size: int = 1
    muscriptor_cfg_coef: float = 1.0
    muscriptor_prelude_forcing: bool = True
    # ---- 维度1：音频预处理 ----
    preprocess_enabled: bool = True
    preprocess_separate: str = "hpss"  # none | hpss | demucs
    preprocess_denoise: bool = False
    preprocess_hp_hz: float = 70.0
    preprocess_target_sr: int = 44100
    preprocess_lufs: float = -14.0
    # ---- 维度2/3：并行 F0 基频跟踪与起始点检测 ----
    f0_enabled: bool = True
    f0_method: str = "crepe"  # crepe | pyin
    f0_step_ms: float = 10.0
    f0_voicing_threshold: float = 0.5
    onset_enabled: bool = True
    # ---- 维度5：后处理修补 ----
    repair_octave: bool = True
    repair_gaps: bool = True

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
        # ---- 五维重构新增字段校验 ----
        self.preprocess_enabled = bool(self.preprocess_enabled)
        self.preprocess_separate = str(self.preprocess_separate or "hpss").strip().lower()
        if self.preprocess_separate not in {"none", "hpss", "demucs"}:
            raise ValueError(
                f"Unknown preprocess_separate mode: {self.preprocess_separate}"
            )
        self.preprocess_denoise = bool(self.preprocess_denoise)
        self.preprocess_hp_hz = max(0.0, float(self.preprocess_hp_hz))
        self.preprocess_target_sr = max(4000, int(self.preprocess_target_sr))
        self.preprocess_lufs = float(self.preprocess_lufs)
        self.f0_enabled = bool(self.f0_enabled)
        self.f0_method = str(self.f0_method or "crepe").strip().lower()
        if self.f0_method not in {"crepe", "pyin"}:
            raise ValueError(f"Unknown f0_method: {self.f0_method}")
        self.f0_step_ms = max(1.0, float(self.f0_step_ms))
        self.f0_voicing_threshold = max(0.0, min(1.0, float(self.f0_voicing_threshold)))
        self.onset_enabled = bool(self.onset_enabled)
        self.adaptive_gap = bool(self.adaptive_gap)
        self.max_chord_voices = max(1, int(self.max_chord_voices))
        self.quantize_strength = max(0.0, min(1.0, float(self.quantize_strength)))
        self.repair_octave = bool(self.repair_octave)
        self.repair_gaps = bool(self.repair_gaps)


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
