"""Audio preprocessing for the transcription pipeline (维度1).

Runs **before** the MuScriptor engine and the parallel F0 tracker so that the
model sees a cleaned signal: de-noised, separated from percussion / other
stems, high-passed and loudness-normalized.

The module degrades gracefully: if ``librosa`` / ``scipy`` / ``soundfile`` are
not installed (or the input path is missing), :meth:`AudioPreprocessor.run`
returns the original path unchanged so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Optional

from .models import TranscribeConfig


class AudioPreprocessor:
    """Clean a raw audio file before transcription.

    Steps (all optional, controlled by ``config``):

    * resample / mono downmix to ``preprocess_target_sr``;
    * high-pass filter (``preprocess_hp_hz``) to remove rumble / DC offset;
    * harmonic-percussive separation (``hpss``) or source separation
      (``demucs``) to isolate the melodic carrier;
    * peak normalization to a fixed amplitude;
    * optional noise gate (``preprocess_denoise``).
    """

    def __init__(self, config: TranscribeConfig):
        self.config = config

    def run(self, audio_path: str) -> str:
        if not self.config.preprocess_enabled:
            return audio_path
        if not audio_path or not os.path.exists(audio_path):
            return audio_path

        try:
            return self._run(audio_path)
        except Exception as exc:  # never block transcription on preprocessing
            warnings.warn(f"audio preprocessing skipped: {exc}")
            return audio_path

    def _run(self, audio_path: str) -> str:
        import librosa
        import numpy as np
        import soundfile as sf
        from scipy.signal import butter, sosfilt

        sr = self.config.preprocess_target_sr
        y, _ = librosa.load(audio_path, sr=sr, mono=True)

        # 1) 去直流 + 高通去轰鸣
        y = self._highpass(y, sr, self.config.preprocess_hp_hz, butter, sosfilt)

        # 2) 谐波/打击乐分离，或人声/乐器干茎分离
        if self.config.preprocess_separate == "hpss":
            harmonic, _ = librosa.effects.hpss(y, margin=3.0)
            y = harmonic
        elif self.config.preprocess_separate == "demucs":
            y = self._demucs(audio_path, sr, y)

        # 3) 响度归一化（峰值）
        if y.size and np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y, norm=np.inf, amp=0.95)

        # 4) 可选噪声门
        if self.config.preprocess_denoise:
            y = self._noise_gate(y, sr)

        out = tempfile.mktemp(suffix=".wav")
        sf.write(out, y.astype(np.float32), sr)
        return out

    @staticmethod
    def _highpass(y, sr, hp_hz, butter, sosfilt):
        if not hp_hz or hp_hz <= 0:
            return y
        nyq = sr / 2.0
        high = min(hp_hz / nyq, 0.99)
        if high <= 0:
            return y
        sos = butter(4, high, btype="high", output="sos")
        return sosfilt(sos, y)

    def _demucs(self, audio_path: str, sr: int, fallback):
        """Extract the melodic carrier using Demucs (vocals/other stems).

        Falls back to the input signal when Demucs is unavailable or fails.
        """
        try:
            import librosa
            import numpy as np

            from demucs.apply import apply_model
            from demucs.pretrained import get_model
            from demucs.audio import AudioFile, convert_audio

            model = get_model(name="htdemucs")
            wav, rate = librosa.load(audio_path, sr=model.samplerate, mono=False)
            wav = np.asarray(wav)
            ref = wav.mean(0)
            ref_tensor = (
                librosa.util.normalize(ref, norm=np.inf, amp=1.0).reshape(1, -1)
                if False
                else None
            )
            # demucs expects (channels, length) float tensor on CPU
            import torch

            src = torch.tensor(wav).float()
            src = convert_audio(src, rate, model.samplerate, model.audio_channels)
            with torch.no_grad():
                stems = apply_model(model, src[None], device="cpu", progress=False)[0]
            # stems order: drums, bass, other, vocals
            vocals = stems[3].numpy()
            other = stems[2].numpy()
            mixed = (vocals + other) / 2.0
            if mixed.size:
                return librosa.resample(mixed.mean(0), orig_sr=model.samplerate, target_sr=sr)
        except Exception as exc:
            warnings.warn(f"demucs separation skipped: {exc}")
        return fallback

    @staticmethod
    def _noise_gate(y, sr, threshold_db: float = -45.0):
        """Simple spectral-gated attenuation of low-energy frames."""
        try:
            import librosa
            import numpy as np

            frame_length = 2048
            hop = 512
            rms = librosa.feature.rms(
                y=y, frame_length=frame_length, hop_length=hop
            )[0]
            if rms.size == 0:
                return y
            db = librosa.amplitude_to_db(rms + 1e-9)
            mask = db > threshold_db
            mask = np.convolve(mask.astype(float), np.ones(3) / 3, mode="same") > 0.5
            # smooth mask to per-sample gate
            idx = np.linspace(0, len(mask) - 1, len(y)).round().astype(int)
            gate = mask[idx]
            return y * gate
        except Exception:
            return y
