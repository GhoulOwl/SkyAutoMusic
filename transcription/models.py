"""Audio-to-15-key arrangement pipeline data models."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


ArrangementPreset = Literal["auto", "simple", "standard", "full"]
TranscriptionMode = Literal["audio_arrangement", "midi"]
Meter = Literal["auto", "4/4", "3/4", "6/8"]
LeadSource = Literal["vocal", "instrumental", "chords_only"]
TranscriptionEngine = Literal["auto", "fast", "quality"]
QualityModel = Literal["auto", "small", "medium"]
SymbolicRole = Literal["melody", "bass", "harmony", "other", "drums"]
SourcePlatform = Literal["local", "netease"]
ProgressCallback = Callable[[str, float, str], None]


class TranscriptionError(RuntimeError):
    """A transcription failure that can be shown directly to the user."""


class CancelledError(TranscriptionError):
    """The user cancelled an in-flight transcription."""


@dataclass(frozen=True)
class SourceMetadata:
    platform: SourcePlatform = "local"
    title: str = ""
    artists: Tuple[str, ...] = ()
    source_id: str = ""
    webpage_url: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class NoteEvent:
    """Compatibility event used by MIDI import and low-level pitch backends."""

    start_ms: int
    end_ms: int
    midi_pitch: int
    strength: float
    source: str

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("invalid note interval")
        if not 0 <= self.midi_pitch <= 127:
            raise ValueError("midi pitch must be in 0..127")

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)

    @property
    def weight(self) -> float:
        return self.duration_ms * max(0.0, float(self.strength))

    def shifted(self, offset_ms: int) -> "NoteEvent":
        return replace(
            self,
            start_ms=max(0, self.start_ms + offset_ms),
            end_ms=max(0, self.end_ms + offset_ms),
        )


@dataclass(frozen=True)
class SymbolicNote:
    """Instrument-aware note emitted by the V3 whole-song transcription backend."""

    start_ms: int
    end_ms: int
    midi_pitch: int
    instrument: str
    role: SymbolicRole = "other"

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("symbolic note must have a positive interval")
        if not 0 <= self.midi_pitch <= 127:
            raise ValueError("symbolic MIDI pitch must be in 0..127")


@dataclass
class QualityAnalysisDraft:
    """Cached V3 symbolic transcription and its non-uniform beat grid."""

    duration_sec: float
    symbolic_notes: List[SymbolicNote]
    beat_times_ms: List[int]
    downbeat_times_ms: List[int]
    bar_starts_ms: List[int]
    bpm: float
    meter: Meter
    timing_confidence: float
    model_name: str
    device: str
    timing_backend: str = "beat_this"
    quantization: str = "adaptive_8th_triplet_16th"
    refined_regions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptionOptions:
    mode: TranscriptionMode = "audio_arrangement"
    arrangement_preset: ArrangementPreset = "auto"
    source_key: Optional[str] = None
    melody_octave_shift: Optional[int] = None
    max_polyphony: int = 4
    bpm_override: Optional[float] = None
    meter: Meter = "auto"
    engine: TranscriptionEngine = "auto"
    quality_model: QualityModel = "auto"
    rights_confirmed: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("audio_arrangement", "midi"):
            raise ValueError(f"unsupported transcription mode: {self.mode}")
        if self.arrangement_preset not in ("auto", "simple", "standard", "full"):
            raise ValueError(f"unsupported arrangement preset: {self.arrangement_preset}")
        if self.melody_octave_shift is not None and self.melody_octave_shift not in (-1, 0, 1):
            raise ValueError("melody_octave_shift must be -1, 0, 1, or None")
        if not 2 <= int(self.max_polyphony) <= 10:
            raise ValueError("max_polyphony must be in 2..10")
        if self.bpm_override is not None and not 40.0 <= float(self.bpm_override) <= 220.0:
            raise ValueError("bpm_override must be in 40..220")
        if self.meter not in ("auto", "4/4", "3/4", "6/8"):
            raise ValueError(f"unsupported meter: {self.meter}")
        if self.engine not in ("auto", "fast", "quality"):
            raise ValueError(f"unsupported transcription engine: {self.engine}")
        if self.quality_model not in ("auto", "small", "medium"):
            raise ValueError(f"unsupported quality model: {self.quality_model}")


@dataclass(frozen=True)
class TempoMap:
    bpm: float
    meter: Meter
    beat_times_ms: Tuple[int, ...]
    bar_starts_ms: Tuple[int, ...]
    grid_times_ms: Tuple[int, ...]
    confidence: float


@dataclass(frozen=True)
class MelodyNote:
    start_ms: int
    end_ms: int
    midi_pitch: int
    confidence: float
    source: LeadSource

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError("melody note must have a positive duration")


@dataclass(frozen=True)
class ChordSpan:
    start_ms: int
    end_ms: int
    root_pc: int
    quality: str
    bass_pc: int
    confidence: float

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError("chord span must have a positive duration")
        if not 0 <= self.root_pc <= 11 or not 0 <= self.bass_pc <= 11:
            raise ValueError("chord pitch class must be in 0..11")


@dataclass(frozen=True)
class Section:
    start_ms: int
    end_ms: int
    role: Literal["intro", "verse", "pre_chorus", "chorus", "bridge", "instrumental", "outro"]
    energy: int
    repeat_group: int = 0

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError("section must have a positive duration")
        if not 0 <= self.energy <= 3:
            raise ValueError("section energy must be in 0..3")


@dataclass
class AnalysisDraft:
    duration_sec: float
    tempo_map: TempoMap
    detected_key: str
    key_confidence: float
    semitone_shift: int
    melody: List[MelodyNote]
    chords: List[ChordSpan]
    sections: List[Section]
    lead_source: LeadSource
    melody_confidence: float
    harmony_confidence: float
    structure_confidence: float
    used_mix_fallback: bool = False
    vocal_path: str = ""
    accompaniment_path: str = ""


@dataclass
class TranscriptionResult:
    events: List[NoteEvent]
    song_notes: List[Dict[str, Any]]
    detected_key: str
    bpm: float
    stats: Dict[str, Any]
    warnings: List[str]
    engine: str = "arrangement_v2"
    source_file: str = ""
    semitone_shift: int = 0
    octave_shift: int = 0
    beat_times_ms: List[int] = field(default_factory=list, repr=False)
    options: TranscriptionOptions = field(default_factory=TranscriptionOptions, repr=False)
    source: Optional[SourceMetadata] = None
    analysis: Optional[AnalysisDraft] = field(default=None, repr=False)
    quality_analysis: Optional[QualityAnalysisDraft] = field(default=None, repr=False)
    separation_model: str = ""
    artifact_root: str = field(default="", repr=False)
