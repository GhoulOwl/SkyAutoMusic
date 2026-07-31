"""SkyAutoMusic 音频/MIDI 扒谱流水线。"""

from .models import (
    CancelledError,
    DEFAULT_ENABLED_STEMS,
    FusionProfile,
    InstrumentalPolicy,
    NoteEvent,
    ProgressCallback,
    RepeatCleanupMode,
    SourceMetadata,
    StemKind,
    StemResult,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)
from .pipeline import (
    cleanup_result_artifacts,
    export_song_json,
    next_available_path,
    rearrange_draft,
    sanitize_filename_stem,
    suggested_output_stem,
    transcribe_draft,
)

__all__ = [
    "CancelledError",
    "DEFAULT_ENABLED_STEMS",
    "FusionProfile",
    "InstrumentalPolicy",
    "NoteEvent",
    "ProgressCallback",
    "RepeatCleanupMode",
    "SourceMetadata",
    "StemKind",
    "StemResult",
    "TranscriptionError",
    "TranscriptionOptions",
    "TranscriptionResult",
    "cleanup_result_artifacts",
    "export_song_json",
    "next_available_path",
    "rearrange_draft",
    "sanitize_filename_stem",
    "suggested_output_stem",
    "transcribe_draft",
]
