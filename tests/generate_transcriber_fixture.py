"""生成供 CI 打包自检使用的短 C 大三和弦 WAV。"""
from __future__ import annotations

import math
import struct
import sys
import wave


def generate(path: str) -> None:
    sample_rate = 22050
    duration = 1.2
    frequencies = (261.625565, 329.627557, 391.995436)
    frames = []
    for index in range(int(sample_rate * duration)):
        time_s = index / sample_rate
        if time_s < 0.2:
            value = 0.0
        else:
            local = time_s - 0.2
            attack = min(1.0, local / 0.01)
            decay = math.exp(-2.5 * local)
            value = sum(math.sin(2 * math.pi * freq * local) for freq in frequencies)
            value = value / len(frequencies) * attack * decay * 0.8
        frames.append(struct.pack("<h", max(-32768, min(32767, int(value * 32767)))))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(frames))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_transcriber_fixture.py OUTPUT.wav")
    generate(sys.argv[1])
