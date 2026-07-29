from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Literal, Optional


TranscriptionMode = Literal["midi", "monophonic", "polyphonic"]
Sensitivity = Literal["low", "normal", "high"]
QuantizeMode = Literal["off", "1/8", "1/16"]
ProgressCallback = Callable[[str, float, str], None]


class TranscriptionError(RuntimeError):
    """用户可理解的转写失败。"""


class CancelledError(TranscriptionError):
    """用户主动取消转写。"""


@dataclass(frozen=True)
class NoteEvent:
    start_ms: int
    end_ms: int
    midi_pitch: int
    strength: float
    source: str

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

    def __post_init__(self) -> None:
        if self.mode not in ("midi", "monophonic", "polyphonic"):
            raise ValueError(f"不支持的转写模式: {self.mode}")
        if self.sensitivity not in ("low", "normal", "high"):
            raise ValueError(f"不支持的灵敏度: {self.sensitivity}")
        if self.quantize not in ("off", "1/8", "1/16"):
            raise ValueError(f"不支持的量化方式: {self.quantize}")
        if self.octave_shift is not None and not -2 <= self.octave_shift <= 2:
            raise ValueError("octave_shift 必须位于 -2..2")
        if not 1 <= int(self.max_polyphony) <= 5:
            raise ValueError("max_polyphony 必须位于 1..5")


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

