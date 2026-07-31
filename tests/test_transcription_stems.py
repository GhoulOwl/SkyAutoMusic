import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.arranger import arrange_events  # noqa: E402
from transcription.backends import (  # noqa: E402
    BackendOutput,
    BasicPitchBackend,
    _monophonic_chunk,
)
from transcription.models import (  # noqa: E402
    CancelledError,
    NoteEvent,
    StemResult,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)
from transcription.pipeline import (  # noqa: E402
    cleanup_result_artifacts,
    export_song_json,
    rearrange_draft,
    transcribe_draft,
)
from transcription.preview import PreviewPlayer  # noqa: E402
from transcription.separation import (  # noqa: E402
    DemucsStemSeparator,
    MODEL_FILENAME,
    MODEL_NAME,
    SeparatedAudio,
    validate_model_repository,
)


def stem_event(start, pitch, stem, strength=1.0, duration=240):
    return NoteEvent(
        start,
        start + duration,
        pitch,
        strength,
        f"{stem}-test",
        stem=stem,
    )


class TestStemOptionsAndFusion(unittest.TestCase):
    def test_defaults_enable_smart_fill_but_not_drum_notes(self):
        options = TranscriptionOptions(mode="stem_fusion")
        self.assertEqual(options.fusion_profile, "vocal_first")
        self.assertEqual(options.instrumental_policy, "smart_fill")
        self.assertTrue(options.use_drum_timing)
        self.assertIn("instrumental", options.enabled_stems)
        self.assertNotIn("drums", options.enabled_stems)

    def test_drums_cannot_be_enabled_as_pitched_notes(self):
        with self.assertRaisesRegex(ValueError, "鼓轨只能用于节奏"):
            TranscriptionOptions(
                mode="stem_fusion",
                enabled_stems=("vocals", "drums"),
            )

    def test_vocal_is_kept_as_lead_when_polyphony_is_limited(self):
        notes, _key, _semitone, _octave, stats = arrange_events(
            [
                stem_event(0, 64, "vocals", strength=0.55),
                stem_event(0, 72, "piano", strength=1.0),
                stem_event(0, 67, "guitar", strength=1.0),
            ],
            TranscriptionOptions(
                mode="stem_fusion",
                source_key="C major",
                octave_shift=0,
                max_polyphony=1,
                repeat_cleanup="off",
            ),
        )
        self.assertEqual(notes, [{"time": 0, "key": "1Key2"}])
        self.assertEqual(stats["polyphony_reduced"], 2)

    def test_keyboard_profile_can_promote_piano_to_lead(self):
        notes, _key, _semitone, _octave, _stats = arrange_events(
            [
                stem_event(0, 64, "vocals"),
                stem_event(0, 72, "piano"),
                stem_event(0, 67, "guitar"),
            ],
            TranscriptionOptions(
                mode="stem_fusion",
                fusion_profile="keyboard_first",
                source_key="C major",
                octave_shift=0,
                max_polyphony=1,
                repeat_cleanup="off",
            ),
        )
        self.assertEqual(notes, [{"time": 0, "key": "1Key7"}])

    def test_disabled_specialized_track_still_masks_instrumental_duplicate(self):
        notes, _key, _semitone, _octave, stats = arrange_events(
            [
                stem_event(0, 64, "vocals"),
                stem_event(500, 67, "piano"),
                stem_event(530, 67, "instrumental"),
            ],
            TranscriptionOptions(
                mode="stem_fusion",
                source_key="C major",
                octave_shift=0,
                enabled_stems=("vocals", "instrumental"),
                repeat_cleanup="off",
            ),
        )
        self.assertEqual(notes, [{"time": 0, "key": "1Key2"}])
        self.assertEqual(stats["instrumental_smart_fill_suppressed"], 1)
        self.assertEqual(stats["fusion_disabled_event_count"], 1)

    def test_instrumental_only_fills_uncovered_material(self):
        notes, _key, _semitone, _octave, stats = arrange_events(
            [
                stem_event(0, 64, "vocals"),
                stem_event(20, 64, "instrumental"),
                stem_event(500, 69, "instrumental"),
            ],
            TranscriptionOptions(
                mode="stem_fusion",
                source_key="C major",
                octave_shift=0,
                repeat_cleanup="off",
            ),
        )
        self.assertEqual(
            notes,
            [
                {"time": 0, "key": "1Key2"},
                {"time": 500, "key": "1Key5"},
            ],
        )
        self.assertEqual(stats["instrumental_smart_fill_suppressed"], 1)

    def test_silent_drums_never_generate_sky_notes(self):
        notes, _key, _semitone, _octave, _stats = arrange_events(
            [stem_event(0, 36, "drums")],
            TranscriptionOptions(mode="stem_fusion"),
        )
        self.assertEqual(notes, [])


class TestBackendTimestampRepair(unittest.TestCase):
    def test_pyin_ignores_backtracked_onset_after_note_end(self):
        import numpy as np

        frequencies = np.full(4, 440.0)
        voiced = np.ones(4, dtype=bool)
        probabilities = np.full(4, 0.9)
        with (
            mock.patch(
                "librosa.pyin",
                return_value=(frequencies, voiced, probabilities),
            ),
            mock.patch(
                "librosa.frames_to_time",
                return_value=np.array([0.0, 0.01, 0.02, 0.03]),
            ),
            mock.patch(
                "librosa.onset.onset_detect",
                return_value=np.array([0.08]),
            ),
            mock.patch("librosa.hz_to_midi", return_value=69.0),
        ):
            events = _monophonic_chunk(
                np.ones(1024, dtype=np.float32),
                22050,
                0.5,
            )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_ms, 0)
        self.assertGreater(events[0].end_ms, events[0].start_ms)

    def test_basic_pitch_repairs_reversed_third_party_interval(self):
        import numpy as np
        import types

        fake_inference = types.ModuleType("basic_pitch.inference")
        fake_inference.predict = lambda *_args, **_kwargs: (
            {},
            object(),
            [(0.8, 0.5, 60, 0.8, None)],
        )
        fake_package = types.ModuleType("basic_pitch")
        fake_package.__path__ = []
        backend = BasicPitchBackend()
        backend._model = object()
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "basic_pitch": fake_package,
                    "basic_pitch.inference": fake_inference,
                },
            ),
            mock.patch(
                "transcription.backends._load_audio",
                return_value=(np.ones(100, dtype=np.float32), 100),
            ),
            mock.patch(
                "transcription.backends._estimate_beats",
                return_value=(120.0, []),
            ),
        ):
            output = backend.transcribe("fake.wav")
        self.assertEqual(len(output.events), 1)
        self.assertEqual(output.events[0].start_ms, 800)
        self.assertEqual(output.events[0].end_ms, 801)


class TestStemPipeline(unittest.TestCase):
    @staticmethod
    def _separated_audio(output_dir):
        paths = {}
        for stem in (
            "vocals",
            "drums",
            "piano",
            "bass",
            "guitar",
            "instrumental",
        ):
            path = os.path.join(output_dir, f"{stem}.wav")
            Path(path).write_bytes(b"RIFF-test")
            paths[stem] = path
        return SeparatedAudio(
            paths=paths,
            model_name=MODEL_NAME,
            sample_rate=44100,
            duration_sec=2.0,
        )

    @staticmethod
    def _backend_for(path, *_args, **_kwargs):
        stem = Path(path).stem
        pitches = {
            "vocals": 64,
            "piano": 67,
            "bass": 48,
            "guitar": 72,
            "instrumental": 76,
        }
        return BackendOutput(
            events=[
                NoteEvent(
                    0 if stem == "vocals" else 500,
                    300 if stem == "vocals" else 800,
                    pitches[stem],
                    0.9,
                    "mock",
                )
            ],
            bpm=120.0,
            beat_times_ms=[0, 500, 1000],
            duration_sec=2.0,
            engine="pyin" if stem == "vocals" else "basic_pitch_onnx",
            warnings=[],
        )

    def test_full_six_track_result_and_cached_drum_timing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "song.wav")
            Path(source).write_bytes(b"RIFF-source")

            def fake_separate(_path, output_dir, *_args):
                return self._separated_audio(output_dir)

            with (
                mock.patch(
                    "transcription.pipeline.separate_audio",
                    side_effect=fake_separate,
                ),
                mock.patch(
                    "transcription.pipeline.estimate_audio_timing",
                    side_effect=[
                        (100.0, [0, 600, 1200], 2.0),
                        (120.0, [0, 500, 1000], 2.0),
                    ],
                ),
                mock.patch(
                    "transcription.pipeline.transcribe_monophonic",
                    side_effect=self._backend_for,
                ),
                mock.patch(
                    "transcription.pipeline.transcribe_polyphonic",
                    side_effect=self._backend_for,
                ),
            ):
                result = transcribe_draft(
                    source,
                    TranscriptionOptions(
                        mode="stem_fusion",
                        source_key="C major",
                        octave_shift=0,
                        repeat_cleanup="off",
                    ),
                    workspace_dir=temp_dir,
                )

            self.assertEqual(result.engine, "stem_fusion")
            self.assertEqual(result.separation_model, MODEL_NAME)
            self.assertEqual(set(result.stems), {
                "vocals",
                "drums",
                "piano",
                "bass",
                "guitar",
                "instrumental",
            })
            self.assertEqual(result.stems["drums"].events, [])
            self.assertEqual(result.stems["drums"].stats["bpm"], 120.0)
            self.assertEqual(result.stems["drums"].stats["beat_count"], 3)
            self.assertTrue(result.stems["drums"].stats["used_for_timing"])
            self.assertEqual(result.bpm, 120.0)
            self.assertEqual(result.beat_times_ms, [0, 500, 1000])
            self.assertEqual(set(result.timing_maps), {"original", "drums"})
            self.assertTrue(all(event.stem for event in result.events))

            original_timing = rearrange_draft(
                result,
                TranscriptionOptions(
                    mode="stem_fusion",
                    source_key="C major",
                    octave_shift=0,
                    repeat_cleanup="off",
                    use_drum_timing=False,
                ),
            )
            self.assertEqual(original_timing.bpm, 100.0)
            self.assertEqual(original_timing.beat_times_ms, [0, 600, 1200])
            self.assertEqual(original_timing.artifact_root, result.artifact_root)

            artifact_root = result.artifact_root
            cleanup_result_artifacts(result)
            self.assertFalse(os.path.exists(artifact_root))
            self.assertEqual(result.artifact_root, "")

    def test_drum_timing_failure_falls_back_to_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "song.wav")
            Path(source).write_bytes(b"RIFF-source")

            def fake_separate(_path, output_dir, *_args):
                return self._separated_audio(output_dir)

            with (
                mock.patch(
                    "transcription.pipeline.separate_audio",
                    side_effect=fake_separate,
                ),
                mock.patch(
                    "transcription.pipeline.estimate_audio_timing",
                    side_effect=[
                        (90.0, [0, 667, 1334], 2.0),
                        TranscriptionError("静音"),
                    ],
                ),
                mock.patch(
                    "transcription.pipeline.transcribe_monophonic",
                    side_effect=self._backend_for,
                ),
                mock.patch(
                    "transcription.pipeline.transcribe_polyphonic",
                    side_effect=self._backend_for,
                ),
            ):
                result = transcribe_draft(
                    source,
                    TranscriptionOptions(mode="stem_fusion"),
                    workspace_dir=temp_dir,
                )
            try:
                self.assertEqual(result.bpm, 90.0)
                self.assertEqual(set(result.timing_maps), {"original"})
                self.assertEqual(result.stems["drums"].stats["beat_count"], 0)
                self.assertFalse(result.stems["drums"].stats["used_for_timing"])
                self.assertTrue(
                    any("鼓点节奏检测失败" in warning for warning in result.warnings)
                )
            finally:
                cleanup_result_artifacts(result)

    def test_separation_failure_cleans_session_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "song.wav")
            Path(source).write_bytes(b"RIFF-source")
            with mock.patch(
                "transcription.pipeline.separate_audio",
                side_effect=TranscriptionError("模型损坏"),
            ):
                with self.assertRaisesRegex(TranscriptionError, "模型损坏"):
                    transcribe_draft(
                        source,
                        TranscriptionOptions(mode="stem_fusion"),
                        workspace_dir=temp_dir,
                    )
            leftovers = [
                name for name in os.listdir(temp_dir)
                if name.startswith("sky-six-stem-")
            ]
            self.assertEqual(leftovers, [])

    def test_cancel_before_separation_cleans_session_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "song.wav")
            Path(source).write_bytes(b"RIFF-source")
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(CancelledError):
                transcribe_draft(
                    source,
                    TranscriptionOptions(mode="stem_fusion"),
                    cancel_event=cancel,
                    workspace_dir=temp_dir,
                )
            self.assertFalse(
                any(name.startswith("sky-six-stem-") for name in os.listdir(temp_dir))
            )

    def test_export_schema_v2_contains_engines_but_not_temp_paths(self):
        options = TranscriptionOptions(mode="stem_fusion")
        result = TranscriptionResult(
            events=[stem_event(0, 64, "vocals")],
            song_notes=[{"time": 0, "key": "1Key2"}],
            detected_key="C major",
            bpm=120.0,
            stats={"arranged_note_count": 1},
            warnings=[],
            engine="stem_fusion",
            source_file="song.wav",
            options=options,
            stems={
                "vocals": StemResult(
                    "vocals",
                    r"C:\Temp\secret-vocals.wav",
                    [stem_event(0, 64, "vocals")],
                    "pyin",
                ),
                "drums": StemResult(
                    "drums",
                    r"C:\Temp\secret-drums.wav",
                    [],
                    "librosa_beat",
                ),
            },
            separation_model=MODEL_NAME,
            artifact_root=r"C:\Temp\secret-session",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "score.json")
            export_song_json(result, output)
            text = Path(output).read_text(encoding="utf-8")
            payload = json.loads(text)[0]
        metadata = payload["_transcribe"]
        self.assertEqual(metadata["schemaVersion"], 2)
        self.assertEqual(metadata["separationModel"], MODEL_NAME)
        self.assertEqual(metadata["stemEngines"]["vocals"], "pyin")
        self.assertEqual(metadata["fusionProfile"], "vocal_first")
        self.assertNotIn("secret-", text)
        self.assertNotIn("artifactRoot", metadata)


class TestSeparationResources(unittest.TestCase):
    def test_separator_derives_aligned_instrumental_from_original_minus_vocals(self):
        import numpy as np
        import soundfile as sf

        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            @property
            def shape(self):
                return self.values.shape

            def __sub__(self, other):
                return FakeTensor(self.values - other.values)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class FakeDemucs:
            samplerate = 8000

            def __init__(self):
                self.callback = None

            def update_parameter(self, callback=None, **_kwargs):
                self.callback = callback

            def separate_audio_file(self, _path):
                if self.callback:
                    self.callback({
                        "state": "start",
                        "audio_length": 128,
                        "segment_offset": 0,
                    })
                original = FakeTensor(np.full((2, 128), 0.75))
                stems = {
                    "vocals": FakeTensor(np.full((2, 128), 0.25)),
                    "drums": FakeTensor(np.full((2, 128), 0.10)),
                    "piano": FakeTensor(np.full((2, 128), 0.10)),
                    "bass": FakeTensor(np.full((2, 128), 0.10)),
                    "guitar": FakeTensor(np.full((2, 128), 0.10)),
                    "other": FakeTensor(np.full((2, 128), 0.10)),
                }
                return original, stems

        backend = DemucsStemSeparator(Path("unused"))
        fake = FakeDemucs()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            source.write_bytes(b"fixture")
            with mock.patch.object(backend, "_load", return_value=fake):
                result = backend.separate(
                    str(source),
                    str(Path(temp_dir) / "stems"),
                )
            self.assertEqual(set(result.paths), {
                "vocals",
                "drums",
                "piano",
                "bass",
                "guitar",
                "instrumental",
            })
            instrumental, sample_rate = sf.read(result.paths["instrumental"])
            self.assertEqual(instrumental.shape, (128, 2))
            self.assertEqual(sample_rate, 8000)
            self.assertTrue(np.allclose(instrumental, 0.5, atol=1e-4))

    def test_missing_model_reports_offline_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(TranscriptionError, "模型不完整"):
                validate_model_repository(Path(temp_dir))

    def test_corrupt_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / f"{MODEL_NAME}.yaml").write_text(
                "models: [5c90dfd2]\n",
                encoding="utf-8",
            )
            (root / MODEL_FILENAME).write_bytes(b"not-the-real-checkpoint")
            with self.assertRaisesRegex(TranscriptionError, "校验失败"):
                validate_model_repository(root)


class TestSkyPreviewPlayer(unittest.TestCase):
    def test_score_preview_uses_normal_game_sample_controller(self):
        controller = mock.Mock()
        controller.prepared = True
        preview = PreviewPlayer(controller=controller)
        preview.player.start = mock.Mock(return_value=True)
        preview.player.stop = mock.Mock()
        result = TranscriptionResult(
            events=[],
            song_notes=[
                {"time": 100, "key": "1Key0"},
                {"time": 100, "key": "1Key4"},
                {"time": 600, "key": "1Key7"},
            ],
            detected_key="C major",
            bpm=120.0,
            stats={},
            warnings=[],
        )

        self.assertTrue(preview.play(result))
        notes_by_time, sorted_times = preview.player.start.call_args.args
        self.assertEqual(sorted_times, [100, 600])
        self.assertEqual(notes_by_time[100], ["1Key0", "1Key4"])
        self.assertIs(preview.player.key_controller, controller)
        preview.stop()
        controller.stop_all.assert_called()


if __name__ == "__main__":
    unittest.main()
