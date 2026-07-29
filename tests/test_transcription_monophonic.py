import math
import os
import struct
import sys
import tempfile
import unittest
import wave


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.backends import transcribe_monophonic  # noqa: E402
from transcription.models import TranscriptionError  # noqa: E402


def _write_sequence(path, midi_notes, rich=False):
    sample_rate = 22050
    samples = []
    for midi in midi_notes:
        samples.extend([0.0] * int(0.2 * sample_rate))
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        for index in range(int(0.45 * sample_rate)):
            time_s = index / sample_rate
            envelope = min(1.0, time_s / 0.01) * max(
                0.0, min(1.0, (0.45 - time_s) / 0.03)
            )
            value = math.sin(2 * math.pi * frequency * time_s)
            if rich:
                value += 0.6 * math.sin(2 * math.pi * frequency * 2 * time_s)
                value += 0.35 * math.sin(2 * math.pi * frequency * 3 * time_s)
                value /= 1.95
            samples.append(value * envelope * 0.7)
    samples.extend([0.0] * int(0.2 * sample_rate))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
                for value in samples
            )
        )


class TestMonophonicIntegration(unittest.TestCase):
    def test_d4_e4_sequence_has_no_release_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sequence.wav")
            _write_sequence(path, (62, 64))
            output = transcribe_monophonic(path)
        self.assertEqual([item.midi_pitch for item in output.events], [62, 64])
        self.assertEqual(len(output.events), 2)
        self.assertLess(abs(output.events[0].start_ms - 200), 90)
        self.assertLess(abs(output.events[1].start_ms - 850), 90)

    def test_rich_c4_remains_one_fundamental(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rich.wav")
            _write_sequence(path, (60,), rich=True)
            output = transcribe_monophonic(path)
        self.assertEqual([item.midi_pitch for item in output.events], [60])

    def test_silence_and_corrupt_audio_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            silent = os.path.join(directory, "silent.wav")
            with wave.open(silent, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(22050)
                handle.writeframes(b"\0\0" * 2205)
            with self.assertRaises(TranscriptionError):
                transcribe_monophonic(silent)

            corrupt = os.path.join(directory, "corrupt.wav")
            with open(corrupt, "wb") as handle:
                handle.write(b"not a wave file")
            with self.assertRaises(TranscriptionError):
                transcribe_monophonic(corrupt)


if __name__ == "__main__":
    unittest.main()
