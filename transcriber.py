"""Compatibility entrypoint for audio-to-Sky-score transcription.

The implementation lives in the internal ``transcription`` package. This
module keeps the historical imports used by the GUI and tests:

    from transcriber import Transcriber, is_audio_file
"""

from __future__ import annotations

from transcription import (
    PitchEvent,
    SkyNote,
    TranscribeConfig,
    Transcriber,
    TranscribeStats,
    is_audio_file,
)
from transcription.core import AUDIO_EXTS

__all__ = [
    "AUDIO_EXTS",
    "PitchEvent",
    "SkyNote",
    "TranscribeConfig",
    "TranscribeStats",
    "Transcriber",
    "is_audio_file",
]
