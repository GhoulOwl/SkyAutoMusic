"""兼容入口与15键自然音映射测试。"""
import os
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcriber import (  # noqa: E402
    Transcriber,
    is_audio_file,
    is_midi_file,
    is_supported_input,
)
from transcription.models import (  # noqa: E402
    NoteEvent,
    TranscriptionOptions,
    TranscriptionResult,
)


def _bare_transcriber() -> Transcriber:
    transcriber = Transcriber.__new__(Transcriber)
    transcriber.NUM_KEYS = 15
    transcriber.midi_root = 60
    transcriber.sr = 22050
    return transcriber


class TestMidiToSkyKey(unittest.TestCase):
    def setUp(self):
        self.transcriber = _bare_transcriber()

    def test_all_c_major_natural_notes_map_to_15_keys(self):
        midi_notes = (60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84)
        for index, midi in enumerate(midi_notes):
            with self.subTest(midi=midi):
                key, range_state = self.transcriber._midi_to_key(midi)
                self.assertEqual(key, f"1Key{index}")
                self.assertEqual(range_state, 0)

    def test_accidental_uses_nearest_natural_note_and_ties_down(self):
        expected = {
            61: "1Key0",
            63: "1Key1",
            66: "1Key3",
            68: "1Key4",
            70: "1Key5",
            73: "1Key7",
            75: "1Key8",
            78: "1Key10",
            80: "1Key11",
            82: "1Key12",
        }
        for midi, key in expected.items():
            with self.subTest(midi=midi):
                self.assertEqual(self.transcriber._midi_to_key(midi), (key, 0))

    def test_out_of_range_notes_fold_by_octave(self):
        # D3 -> D4，F#6 -> F5（等距时向下）
        self.assertEqual(self.transcriber._midi_to_key(50), ("1Key1", -1))
        self.assertEqual(self.transcriber._midi_to_key(90), ("1Key10", 1))

    def test_fractional_midi_is_rounded_before_mapping(self):
        self.assertEqual(self.transcriber._midi_to_key(64.6), ("1Key3", 0))


class TestInputExtensions(unittest.TestCase):
    def test_audio_extensions(self):
        for path in ["a.mp3", "b.WAV", "c.Flac", "d.ogg", "e.m4a", "f.AAC"]:
            self.assertTrue(is_audio_file(path), path)

    def test_midi_extensions(self):
        for path in ["a.mid", "b.MIDI"]:
            self.assertTrue(is_midi_file(path), path)
            self.assertTrue(is_supported_input(path), path)

    def test_unsupported_extensions(self):
        for path in ["a.json", "b.txt", "c", "d.mp4"]:
            self.assertFalse(is_supported_input(path), path)


class TestCompatibilityFacade(unittest.TestCase):
    def test_run_keeps_legacy_result_shape_and_writes_compatible_json(self):
        draft = TranscriptionResult(
            events=[NoteEvent(0, 100, 60, 1.0, "test")],
            song_notes=[{"time": 0, "key": "1Key0"}],
            detected_key="C major",
            bpm=120,
            stats={"arranged_note_count": 1},
            warnings=[],
            engine="pyin",
            source_file="input.wav",
            options=TranscriptionOptions(mode="audio_arrangement"),
        )
        callbacks = []
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "input.wav")
            with open(input_path, "wb") as handle:
                handle.write(b"fixture")
            with mock.patch("transcriber.transcribe_draft", return_value=draft):
                results = Transcriber().run(
                    [input_path],
                    directory,
                    progress_cb=lambda path, fraction, status: callbacks.append(
                        (path, fraction, status)
                    ),
                )
            self.assertTrue(results[0]["ok"])
            self.assertTrue(os.path.isfile(results[0]["output"]))
            self.assertEqual(results[0]["song_name"], "input")
            self.assertTrue(callbacks)


if __name__ == "__main__":
    unittest.main()
