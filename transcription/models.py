from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


StemKind = Literal[
    "vocals",
    "drums",
    "piano",
    "bass",
    "guitar",
    "instrumental",
]
TranscriptionMode = Literal["midi", "monophonic", "polyphonic", "stem_fusion"]
Sensitivity = Literal["low", "normal", "high"]
QuantizeMode = Literal["off", "1/8", "1/16"]
RepeatCleanupMode = Literal["off", "auto", "strong"]
FusionProfile = Literal["vocal_first", "keyboard_first", "balanced"]
InstrumentalPolicy = Literal["smart_fill", "always", "preview_only"]
SourcePlatform = Literal["local", "netease"]
ProgressCallback = Callable[[str, float, str], None]

STEM_KINDS: Tuple[StemKind, ...] = (
    "vocals",
    "drums",
    "piano",
    "bass",
    "guitar",
    "instrumental",
)
DEFAULT_ENABLED_STEMS: Tuple[StemKind, ...] = (
    "vocals",
    "piano",
    "bass",
    "guitar",
    "instrumental",
)


class TranscriptionError(RuntimeError):
    """用户可理解的转写失败。"""


class CancelledError(TranscriptionError):
    """用户主动取消转写。"""


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
    start_ms: int
    end_ms: int
    midi_pitch: int
    strength: float
    source: str
    stem: Optional[StemKind] = None

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError("start_ms 不能为负数")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms 不能早于 start_ms")
        if not 0 <= self.midi_pitch <= 127:
            raise ValueError("midi_pitch 必须位于 0..127")

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
class TranscriptionOptions:
    mode: TranscriptionMode = "polyphonic"
    sensitivity: Sensitivity = "normal"
    source_key: Optional[str] = None
    # 用户界面以八度表示：-2、-1、0、1、2；None 表示自动。
    octave_shift: Optional[int] = None
    quantize: QuantizeMode = "off"
    max_polyphony: int = 3
    repeat_cleanup: RepeatCleanupMode = "auto"
    enabled_stems: Tuple[StemKind, ...] = DEFAULT_ENABLED_STEMS
    use_drum_timing: bool = True
    fusion_profile: FusionProfile = "vocal_first"
    instrumental_policy: InstrumentalPolicy = "smart_fill"

    def __post_init__(self) -> None:
        if self.mode not in ("midi", "monophonic", "polyphonic", "stem_fusion"):
            raise ValueError(f"不支持的转写模式: {self.mode}")
        if self.sensitivity not in ("low", "normal", "high"):
            raise ValueError(f"不支持的灵敏度: {self.sensitivity}")
        if self.quantize not in ("off", "1/8", "1/16"):
            raise ValueError(f"不支持的量化方式: {self.quantize}")
        if self.repeat_cleanup not in ("off", "auto", "strong"):
            raise ValueError(f"不支持的重复音清理方式: {self.repeat_cleanup}")
        if self.octave_shift is not None and not -2 <= self.octave_shift <= 2:
            raise ValueError("octave_shift 必须位于 -2..2")
        if not 1 <= int(self.max_polyphony) <= 5:
            raise ValueError("max_polyphony 必须位于 1..5")
        enabled_stems = tuple(self.enabled_stems)
        if len(enabled_stems) != len(set(enabled_stems)):
            raise ValueError("enabled_stems 不能包含重复声部")
        if any(stem not in STEM_KINDS for stem in enabled_stems):
            raise ValueError(f"enabled_stems 包含未知声部: {enabled_stems}")
        if "drums" in enabled_stems:
            raise ValueError("鼓轨只能用于节奏，不能作为有调音符声部")
        object.__setattr__(self, "enabled_stems", enabled_stems)
        if self.fusion_profile not in ("vocal_first", "keyboard_first", "balanced"):
            raise ValueError(f"不支持的融合预设: {self.fusion_profile}")
        if self.instrumental_policy not in ("smart_fill", "always", "preview_only"):
            raise ValueError(f"不支持的伴奏策略: {self.instrumental_policy}")


@dataclass
class StemResult:
    kind: StemKind
    audio_path: str
    events: List[NoteEvent]
    engine: str
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    events: List[NoteEvent]
    song_notes: List[Dict[str, Any]]
    detected_key: str
    bpm: float
    stats: Dict[str, Any]
    warnings: List[str]
    engine: str = ""
    source_file: str = ""
    semitone_shift: int = 0
    # 实际应用的半音数，始终是 12 的整数倍。
    octave_shift: int = 0
    beat_times_ms: List[int] = field(default_factory=list, repr=False)
    options: TranscriptionOptions = field(default_factory=TranscriptionOptions, repr=False)
    source: Optional[SourceMetadata] = None
    stems: Dict[StemKind, StemResult] = field(default_factory=dict, repr=False)
    separation_model: str = ""
    artifact_root: str = field(default="", repr=False)
    timing_maps: Dict[str, Tuple[float, List[int]]] = field(
        default_factory=dict,
        repr=False,
    )

