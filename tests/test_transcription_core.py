import json
import os
import sys
import tempfile
import threading
import types
import unittest
import wave
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_loader import load_score  # noqa: E402
from transcription.arranger import (  # noqa: E402
    SKY_MIDI,
    arrange_events,
    detect_key,
    key_transpose,
    quantize_events,
)
from transcription.backends import (  # noqa: E402
    BasicPitchBackend,
    BackendOutput,
    _chunk_regions,
    transcribe_monophonic,
    transcribe_midi,
)
from transcription.models import (  # noqa: E402
    CancelledError,
    NoteEvent,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)
from transcription.pipeline import (  # noqa: E402
    export_song_json,
    next_available_path,
    transcribe_draft,
)
from transcription.preview import PREVIEW_SR, render_preview_wav  # noqa: E402


def event(start, pitch, strength=1.0, duration=200, source="test"):
    return NoteEvent(start, start + duration, pitch, strength, source)


class TestKeyAndArrangement(unittest.TestCase):
    def test_detects_clear_c_major_material(self):
        events = [
            event(0, 60, duration=400),
            event(500, 64, duration=350),
            event(1000, 67, duration=500),
            event(1600, 72, duration=500),
        ]
        self.assertEqual(detect_key(events), "C major")

    def test_major_and_minor_transpose_targets(self):
        self.assertEqual(key_transpose("C major"), 0)
        self.assertEqual(key_transpose("G major"), 5)
        self.assertEqual(key_transpose("A minor"), 0)
        self.assertEqual(key_transpose("E minor"), 5)

    def test_all_natural_notes_arrange_to_all_keys(self):
        events = [event(index * 100, pitch) for index, pitch in enumerate(SKY_MIDI)]
        notes, key, semitone, octave, stats = arrange_events(
            events,
            TranscriptionOptions(source_key="C major", octave_shift=0),
        )
        self.assertEqual(key, "C major")
        self.assertEqual(semitone, 0)
        self.assertEqual(octave, 0)
        self.assertEqual([note["key"] for note in notes], [f"1Key{i}" for i in range(15)])
        self.assertEqual(stats["arranged_note_count"], 15)

    def test_g_major_is_transposed_to_c_major_keyboard(self):
        # G-B-D 三和弦整体上移5半音，得到 C-E-G。
        events = [event(100, pitch) for pitch in (67, 71, 74)]
        notes, _key, semitone, _octave, _stats = arrange_events(
            events,
            TranscriptionOptions(source_key="G major", octave_shift=0),
        )
        self.assertEqual(semitone, 5)
        self.assertEqual([note["key"] for note in notes], ["1Key7", "1Key9", "1Key11"])

    def test_manual_octave_shift_is_reported_in_semitones(self):
        notes, _key, _semitone, octave, stats = arrange_events(
            [event(0, 48), event(100, 50)],
            TranscriptionOptions(
                source_key="C major", octave_shift=1, max_polyphony=3
            ),
        )
        self.assertEqual(octave, 12)
        self.assertEqual([note["key"] for note in notes], ["1Key0", "1Key1"])
        self.assertEqual(stats["folded_count"], 0)

    def test_auto_octave_places_low_material_near_keyboard_center(self):
        notes, _key, _semitone, octave, stats = arrange_events(
            [event(0, 36), event(0, 40), event(0, 43)],
            TranscriptionOptions(source_key="C major"),
        )
        self.assertEqual(octave, 24)
        self.assertEqual(
            [note["key"] for note in notes],
            ["1Key0", "1Key2", "1Key4"],
        )
        self.assertEqual(stats["folded_count"], 0)

    def test_events_within_40ms_form_chord_and_are_deduplicated(self):
        events = [
            event(100, 60, strength=0.6),
            event(125, 60, strength=0.9),
            event(139, 64),
            event(141, 67),
        ]
        notes, *_rest, stats = arrange_events(
            events,
            TranscriptionOptions(source_key="C major", octave_shift=0),
        )
        self.assertEqual(
            notes,
            [
                {"time": 100, "key": "1Key0"},
                {"time": 100, "key": "1Key2"},
                {"time": 141, "key": "1Key4"},
            ],
        )
        self.assertEqual(stats["deduped_count"], 1)

    def test_polyphony_limit_keeps_melody_and_strongest_accompaniment(self):
        events = [
            event(0, 60, strength=1.0),
            event(0, 64, strength=0.9),
            event(0, 67, strength=0.1),
            event(0, 72, strength=0.2),
        ]
        notes, *_rest, stats = arrange_events(
            events,
            TranscriptionOptions(
                source_key="C major", octave_shift=0, max_polyphony=2
            ),
        )
        self.assertEqual([note["key"] for note in notes], ["1Key0", "1Key7"])
        self.assertEqual(stats["polyphony_reduced"], 2)

    def test_quantization_moves_only_nearby_events(self):
        events = [event(105, 60), event(190, 62)]
        quantized = quantize_events(events, "1/16", 120, [0, 500, 1000])
        self.assertEqual(quantized[0].start_ms, 125)
        self.assertEqual(quantized[1].start_ms, 190)

    def test_polyphony_options_accept_one_through_five(self):
        for value in range(1, 6):
            self.assertEqual(TranscriptionOptions(max_polyphony=value).max_polyphony, value)
        with self.assertRaises(ValueError):
            TranscriptionOptions(max_polyphony=0)
        with self.assertRaises(ValueError):
            TranscriptionOptions(max_polyphony=6)


class TestMidiBackend(unittest.TestCase):
    def test_midi_merges_non_drum_tracks_and_ignores_drums(self):
        class FakeNote:
            def __init__(self, start, end, pitch, velocity):
                self.start, self.end = start, end
                self.pitch, self.velocity = pitch, velocity

        class FakeInstrument:
            def __init__(self, is_drum, notes):
                self.is_drum, self.notes = is_drum, notes

        class FakeMidi:
            def __init__(self, _path):
                self.instruments = [
                    FakeInstrument(False, [FakeNote(0.1, 0.4, 60, 100)]),
                    FakeInstrument(False, [FakeNote(0.1, 0.5, 64, 80)]),
                    FakeInstrument(True, [FakeNote(0.0, 0.2, 36, 127)]),
                ]

            def get_tempo_changes(self):
                return [0.0], [90.0]

            def get_beats(self):
                return [0.0, 2 / 3, 4 / 3]

        fake_module = types.SimpleNamespace(PrettyMIDI=FakeMidi)
        with mock.patch.dict(sys.modules, {"pretty_midi": fake_module}):
            output = transcribe_midi("fake.mid")
        self.assertEqual([item.midi_pitch for item in output.events], [60, 64])
        self.assertEqual(output.bpm, 90.0)
        self.assertEqual(output.beat_times_ms[:2], [0, 667])
        self.assertIn("已忽略 1 个鼓轨音符", output.warnings)

    def test_chunk_regions_overlap_but_core_regions_do_not(self):
        regions = _chunk_regions(65)
        self.assertEqual(
            regions,
            [
                (0.0, 30.0, 0.0, 31.0),
                (30.0, 60.0, 29.0, 61.0),
                (60.0, 65, 59.0, 65),
            ],
        )

    def test_cancel_is_checked_between_audio_chunks(self):
        import numpy as np

        cancelled = threading.Event()

        def fake_chunk(_audio, _sr, _threshold):
            cancelled.set()
            return [event(0, 60, source="pyin")]

        with mock.patch(
            "transcription.backends._load_audio",
            return_value=(np.ones(65 * 100, dtype=np.float32), 100),
        ), mock.patch(
            "transcription.backends._estimate_beats", return_value=(120.0, [])
        ), mock.patch(
            "transcription.backends._monophonic_chunk", side_effect=fake_chunk
        ):
            with self.assertRaises(CancelledError):
                transcribe_monophonic("fake.wav", cancel_event=cancelled)


class TestBasicPitchBackend(unittest.TestCase):
    def test_chunked_predictions_are_offset_and_owned_once(self):
        import numpy as np

        fake_inference = types.ModuleType("basic_pitch.inference")

        def fake_predict(*_args, **_kwargs):
            return {}, object(), [(1.5, 1.8, 60, 0.8, None)]

        fake_inference.predict = fake_predict
        fake_package = types.ModuleType("basic_pitch")
        fake_package.__path__ = []
        backend = BasicPitchBackend()
        backend._model = object()
        with mock.patch.dict(
            sys.modules,
            {
                "basic_pitch": fake_package,
                "basic_pitch.inference": fake_inference,
            },
        ), mock.patch(
            "transcription.backends._load_audio",
            return_value=(np.ones(65 * 100, dtype=np.float32), 100),
        ), mock.patch(
            "transcription.backends._estimate_beats", return_value=(120.0, [])
        ):
            output = backend.transcribe("fake.wav")
        self.assertEqual(
            [item.start_ms for item in output.events],
            [1500, 30500, 60500],
        )
        self.assertTrue(all(item.source == "basic_pitch" for item in output.events))


class TestPipelineAndExport(unittest.TestCase):
    def test_pipeline_uses_backend_then_arranger(self):
        backend = BackendOutput(
            events=[event(100, 60), event(100, 64), event(100, 67)],
            bpm=100,
            beat_times_ms=[0, 600],
            engine="basic_pitch",
            duration_sec=1.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            with mock.patch(
                "transcription.pipeline.transcribe_polyphonic", return_value=backend
            ):
                result = transcribe_draft(
                    handle.name,
                    TranscriptionOptions(source_key="C major", octave_shift=0),
                )
        self.assertEqual(result.engine, "basic_pitch")
        self.assertEqual(
            [note["key"] for note in result.song_notes],
            ["1Key0", "1Key2", "1Key4"],
        )

    def test_cancelled_pipeline_does_not_load_backend(self):
        with tempfile.NamedTemporaryFile(suffix=".mid") as handle:
            cancelled = threading.Event()
            cancelled.set()
            with self.assertRaises(CancelledError):
                transcribe_draft(handle.name, cancel_event=cancelled)

    def test_atomic_json_export_and_loader_compatibility(self):
        result = TranscriptionResult(
            events=[event(0, 60)],
            song_notes=[{"time": 0, "key": "1Key0"}],
            detected_key="C major",
            bpm=120,
            stats={"arranged_note_count": 1},
            warnings=[],
            engine="pyin",
            source_file="/private/path/source.wav",
            options=TranscriptionOptions(mode="monophonic"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "song.json")
            export_song_json(result, output, "Song")
            score = load_score(output, valid_keys={"1Key0"})
            self.assertEqual(score["notes_by_time"][0], ["1Key0"])
            with open(output, "r", encoding="utf-8") as handle:
                payload = json.load(handle)[0]
            self.assertEqual(payload["transcribedBy"], "SkyAutoMusic")
            self.assertEqual(payload["_transcribe"]["sourceFile"], "source.wav")
            self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(directory)))

    def test_next_available_path_never_silently_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            open(os.path.join(directory, "song.json"), "w").close()
            open(os.path.join(directory, "song (1).json"), "w").close()
            self.assertEqual(
                next_available_path(directory, "song"),
                os.path.join(directory, "song (2).json"),
            )

    def test_empty_result_is_not_exported(self):
        result = TranscriptionResult(
            events=[],
            song_notes=[],
            detected_key="C major",
            bpm=120,
            stats={},
            warnings=[],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "empty.json")
            with self.assertRaises(TranscriptionError):
                export_song_json(result, output, "empty")
            self.assertFalse(os.path.exists(output))


class TestPreview(unittest.TestCase):
    def test_preview_is_mono_16bit_and_not_clipped(self):
        notes = [
            {"time": 0, "key": "1Key0"},
            {"time": 0, "key": "1Key2"},
            {"time": 200, "key": "1Key14"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = render_preview_wav(notes, os.path.join(directory, "preview.wav"))
            with wave.open(path, "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), PREVIEW_SR)
                frames = handle.readframes(handle.getnframes())
            # -32768 表示负向削波；归一化后不应出现。
            samples = memoryview(frames).cast("h")
            self.assertLess(max(abs(int(value)) for value in samples), 32768)


if __name__ == "__main__":
    unittest.main()
