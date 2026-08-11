"""Run a short CPU V2 vocal/accompaniment separation smoke test."""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transcription.separation import TwoStemSeparator


def _fixture(path: Path) -> None:
    rate = 16000
    frames = []
    for index in range(int(rate * .8)):
        time = index / rate
        value = (math.sin(2 * math.pi * 220 * time) + .5 * math.sin(2 * math.pi * 330 * time)) * min(1, time / .02) * math.exp(-1.2 * time) / 1.5
        frames.append(struct.pack("<h", int(max(-1, min(1, value * .7)) * 32767)))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate); stream.writeframes(b"".join(frames))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sky-v2-separation-") as folder:
        root = Path(folder); source = root / "fixture.wav"; _fixture(source)
        pair = TwoStemSeparator().separate(str(source), str(root / "output"))
        for path in (pair.vocal_path, pair.accompaniment_path):
            if not Path(path).is_file() or Path(path).stat().st_size <= 44:
                raise RuntimeError(f"empty separation output: {path}")
        print(f"{pair.model_name}: V2 vocal/accompaniment pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
