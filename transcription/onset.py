"""Onset detection and offset re-estimation (维度3).

MuScriptor provides note on/off times directly, but they are not validated
against the audio. This module adds an *independent* onset detection (spectral
flux) used by :mod:`fusion` to:

* re-detect onsets the model may have missed (feeds gap-filling);
* re-estimate offsets when the F0 tracker shows the pitch stopped earlier than
  the model's ``end_time`` (trims over-long notes).

Lazy imports keep the pipeline runnable without ``librosa``.
"""

from __future__ import annotations

import os
from typing import List

from .models import TranscribeConfig


class OnsetDetector:
    """Detect note onsets from the audio signal."""

    def __init__(self, config: TranscribeConfig):
        self.config = config

    def detect(self, audio_path: str, sr: int = None) -> List[int]:
        """Return onset times (ms) detected in ``audio_path``."""
        sr = int(sr or self.config.preprocess_target_sr)
        if not audio_path or not os.path.exists(audio_path):
            return []
        try:
            import librosa

            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            onsets = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                delta=self.config.onset_delta,
                units="time",
                backtrack=True,
            )
            return [int(round(t * 1000.0)) for t in onsets]
        except Exception:
            return []

    def reestimate_offsets(
        self, audio_path: str, events: List, sr: int = None
    ) -> List:
        """Trim note durations when the audio energy drops early.

        ``events`` is a list of :class:`~transcription.models.PitchEvent`.
        Returns the (possibly truncated) events.
        """
        sr = int(sr or self.config.preprocess_target_sr)
        if not audio_path or not os.path.exists(audio_path) or not events:
            return events
        try:
            import librosa
            import numpy as np

            y, _ = librosa.load(audio_path, sr=sr, mono=True)
        except Exception:
            return events

        for ev in events:
            if ev.duration_ms <= 0:
                continue
            start_s = ev.time_ms / 1000.0
            end_s = (ev.time_ms + ev.duration_ms) / 1000.0
            seg = y[int(start_s * sr): int(end_s * sr)]
            if seg.size < 512:
                continue
            rms = librosa.feature.rms(
                y=seg, frame_length=512, hop_length=128
            )[0]
            if rms.size == 0:
                continue
            peak = rms.max()
            if peak <= 0:
                continue
            idx = np.where(rms < 0.2 * peak)[0]
            if idx.size:
                stop_s = start_s + (idx[0] * 128) / sr
                new_dur = max(30, int(round((stop_s - start_s) * 1000.0)))
                if 0 < new_dur < ev.duration_ms:
                    ev.duration_ms = new_dur
        return events
