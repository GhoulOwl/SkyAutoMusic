"""Parallel F0 (fundamental frequency) tracking (维度2).

Runs alongside MuScriptor to provide an *independent* pitch estimate. The two
uses are:

* **Confidence reconstruction** -- MuScriptor (0.2.1) does not emit per-note
  confidence, so the pipeline used a hard-coded ``1.0``. We compare each
  MuScriptor event against the F0 trajectory and assign a real confidence,
  which re-enables noise gating, melody selection and polyphony capping.
* **Octave / gap repair** -- the F0 grid is used by :mod:`fusion` to correct
  octave errors and to fill notes MuScriptor missed.

Two backends are supported: ``crepe`` (CNN, robust) and ``pyin`` (probabilistic
YIN, no extra model download). Both are imported lazily so the rest of the
pipeline works without them.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from .models import TranscribeConfig

F0Frame = Tuple[int, Optional[float], float, bool]  # (t_ms, midi|None, conf, voiced)


class F0Tracker:
    """Frame-level fundamental-frequency tracker."""

    def __init__(self, config: TranscribeConfig):
        self.config = config

    def track(
        self,
        audio_path: str,
        sr: Optional[int] = None,
        step_ms: Optional[float] = None,
        voicing: Optional[float] = None,
    ) -> List[F0Frame]:
        sr = int(sr or self.config.preprocess_target_sr)
        step_ms = float(step_ms if step_ms is not None else self.config.f0_step_ms)
        voicing = float(
            voicing if voicing is not None else self.config.f0_voicing_threshold
        )
        if not audio_path or not os.path.exists(audio_path):
            return []

        if self.config.f0_method == "pyin":
            return self._track_pyin(audio_path, sr, step_ms, voicing)
        return self._track_crepe(audio_path, sr, step_ms, voicing)

    def _track_crepe(self, audio_path, sr, step_ms, voicing) -> List[F0Frame]:
        try:
            import crepe
            import librosa
            import numpy as np
        except Exception:
            return []

        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        step_size = max(1, int(round(step_ms)))
        try:
            _, f0, conf, _ = crepe.predict(
                y, sr=sr, viterbi=True, step_size=step_size
            )
        except Exception:
            return []

        frames: List[F0Frame] = []
        for i, (f, c) in enumerate(zip(f0, conf)):
            t_ms = i * step_size
            voiced = float(c) > voicing and f is not None and not np.isnan(f)
            midi = 69.0 + 12.0 * np.log2(f / 440.0) if voiced else None
            frames.append((t_ms, midi, float(c), bool(voiced)))
        return frames

    def _track_pyin(self, audio_path, sr, step_ms, voicing) -> List[F0Frame]:
        try:
            import librosa
            import numpy as np
        except Exception:
            return []

        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz("C7")
        # Use a long analysis window (good frequency resolution for low/440Hz
        # content) and a hop that matches the requested step size.
        frame_length = 2048
        hop_length = max(128, int(round(sr * step_ms / 1000.0)))
        try:
            f0, voiced_flag, conf = librosa.pyin(
                y,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
        except Exception:
            return []

        frames: List[F0Frame] = []
        ms_per_frame = 1000.0 * hop_length / sr
        for i, (f, v, c) in enumerate(zip(f0, voiced_flag, conf)):
            t_ms = int(round(i * ms_per_frame))
            voiced = bool(v) and c is not None and not np.isnan(c) and float(c) > voicing
            midi = 69.0 + 12.0 * np.log2(f / 440.0) if (voiced and f is not None and not np.isnan(f)) else None
            frames.append((t_ms, midi, 0.0 if c is None else float(c), bool(voiced)))
        return frames


def frame_midi_at(frames: List[F0Frame], t_ms: int) -> Optional[float]:
    """Nearest voiced F0 midi value around time ``t_ms`` (±one frame)."""
    best = None
    best_dt = None
    for t, midi, _conf, voiced in frames:
        if not voiced or midi is None:
            continue
        dt = abs(t - t_ms)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = midi
    return best
