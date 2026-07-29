"""SkyAutoMusic 音频/MIDI 扒谱流水线。"""

from .models import (
    CancelledError,
    NoteEvent,
    ProgressCallback,
    SourceMetadata,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)
from .pipeline import (
    export_song_json,
    next_available_path,
    rearrange_draft,
    sanitize_filename_stem,
    suggested_output_stem,
    transcribe_draft,
)

__all__ = [
    "CancelledError",
    "NoteEvent",
    "ProgressCallback",
    "SourceMetadata",
    "TranscriptionError",
    "TranscriptionOptions",
    "TranscriptionResult",
    "export_song_json",
    "next_available_path",
    "rearrange_draft",
    "sanitize_filename_stem",
    "suggested_output_stem",
    "transcribe_draft",
]
