"""SkyAutoMusic audio-to-15-key arranging pipeline."""

from .models import (
    AnalysisDraft,
    CancelledError,
    ChordSpan,
    LeadSource,
    MelodyNote,
    Meter,
    NoteEvent,
    ProgressCallback,
    QualityAnalysisDraft,
    QualityModel,
    Section,
    SourceMetadata,
    SymbolicNote,
    SymbolicRole,
    TempoMap,
    TranscriptionError,
    TranscriptionEngine,
    TranscriptionOptions,
    TranscriptionResult,
)
from .pipeline import (
    cleanup_result_artifacts,
    export_song_json,
    next_available_path,
    rearrange_draft,
    refine_region,
    sanitize_filename_stem,
    suggested_output_stem,
    transcribe_draft,
)

__all__ = [
    "AnalysisDraft", "CancelledError", "ChordSpan", "LeadSource", "MelodyNote", "Meter", "NoteEvent",
    "ProgressCallback", "QualityAnalysisDraft", "QualityModel", "Section", "SourceMetadata", "SymbolicNote", "SymbolicRole", "TempoMap", "TranscriptionEngine", "TranscriptionError", "TranscriptionOptions",
    "TranscriptionResult", "cleanup_result_artifacts", "export_song_json", "next_available_path",
    "rearrange_draft", "refine_region", "sanitize_filename_stem", "suggested_output_stem", "transcribe_draft",
]
