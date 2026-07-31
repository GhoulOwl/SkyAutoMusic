"""Run a real, short CPU Demucs separation and verify six aligned WAV stems."""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.models import STEM_KINDS
from transcription.separation import DemucsStemSeparator


def _generate_fixture(path: Path) -> None:
    sample_rate = 16000
    duration_sec = 0.8
    frames = []
    for index in range(int(sample_rate * duration_sec)):
        time_sec = index / sample_rate
        envelope = min(1.0, time_sec / 0.02) * math.exp(-1.2 * time_sec)
        sample = (
            math.sin(2 * math.pi * 220.0 * time_sec)
            + 0.5 * math.sin(2 * math.pi * 329.627557 * time_sec)
        )
        value = int(max(-1.0, min(1.0, sample / 1.5 * envelope * 0.7)) * 32767)
        frames.append(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(frames))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sky-six-stem-smoke-") as temp_dir:
        root = Path(temp_dir)
        source = root / "fixture.wav"
        output = root / "stems"
        _generate_fixture(source)
        separated = DemucsStemSeparator().separate(
            str(source),
            str(output),
        )
        if set(separated.paths) != set(STEM_KINDS):
            raise RuntimeError(
                f"Unexpected stems: {sorted(separated.paths)}"
            )
        wav_shapes = set()
        for stem, path in separated.paths.items():
            with wave.open(path, "rb") as handle:
                shape = (
                    handle.getnchannels(),
                    handle.getframerate(),
                    handle.getnframes(),
                    handle.getsampwidth(),
                )
            wav_shapes.add(shape)
            if Path(path).stat().st_size <= 44:
                raise RuntimeError(f"Empty stem: {stem}")
        if len(wav_shapes) != 1:
            raise RuntimeError(f"Stem alignment mismatch: {sorted(wav_shapes)}")
        print(
            f"{separated.model_name}: six aligned stems, "
            f"{separated.duration_sec:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
