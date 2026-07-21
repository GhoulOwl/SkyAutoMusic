import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcriber import TranscribeConfig, Transcriber, is_audio_file  # noqa: E402
from transcription.analysis import (  # noqa: E402
    AnalyzerFactory,
    AnalyzerUnavailableError,
    MuScriptorAnalyzer,
)
from transcription.arranger import SkyMelodyArranger  # noqa: E402
from transcription.mapping import SkyKeyMapper  # noqa: E402
from transcription.models import PitchEvent, SkyNote, TranscribeStats  # noqa: E402
from transcription.postprocess import NotePostProcessor  # noqa: E402


def _bare_transcriber(mapping="sky_major", **config_kwargs) -> Transcriber:
    t = Transcriber.__new__(Transcriber)
    t.config = TranscribeConfig(mapping=mapping, **config_kwargs)
    t.NUM_KEYS = 15
    t.midi_root = 60
    t.sr = 44100
    return t


@dataclass
class FakeStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class FakeEndEvent:
    end_time: float
    start_event: FakeStartEvent

    @property
    def start_event_index(self):
        return self.start_event.index


@dataclass
class FakeProgressEvent:
    completed: int
    total: int


class TestSkyMajorMapping(unittest.TestCase):
    def setUp(self):
        self.t = _bare_transcriber("sky_major")

    def test_sky_major_scale_maps_to_15_keys(self):
        cases = [
            (60, "1Key0"),
            (62, "1Key1"),
            (64, "1Key2"),
            (65, "1Key3"),
            (67, "1Key4"),
            (69, "1Key5"),
            (71, "1Key6"),
            (72, "1Key7"),
            (74, "1Key8"),
            (76, "1Key9"),
            (77, "1Key10"),
            (79, "1Key11"),
            (81, "1Key12"),
            (83, "1Key13"),
            (84, "1Key14"),
        ]
        for midi, key in cases:
            with self.subTest(midi=midi):
                mapped, clamped = self.t._midi_to_key(midi)
                self.assertEqual(mapped, key)
                self.assertEqual(clamped, 0)

    def test_sky_major_accidentals_snap_to_nearest_scale_key(self):
        self.assertEqual(self.t._midi_to_key(61), ("1Key0", 0))
        self.assertEqual(self.t._midi_to_key(63), ("1Key1", 0))

    def test_sky_major_out_of_range_uses_adaptive_projection(self):
        self.assertEqual(self.t._midi_to_key(48), ("1Key0", -1))
        self.assertEqual(self.t._midi_to_key(85), ("1Key14", 1))
        self.assertEqual(self.t._midi_to_key(86), ("1Key8", 1))
        self.assertEqual(self.t._midi_to_key(91), ("1Key11", 1))
        self.assertEqual(self.t._midi_to_key(96), ("1Key14", 1))
        self.assertEqual(self.t._midi_to_key(55), ("1Key4", -1))
        self.assertEqual(self.t._midi_to_key(59), ("1Key0", -1))

    def test_sky_major_clamp_policy_remains_available(self):
        t = _bare_transcriber("sky_major", out_of_range_policy="clamp")
        self.assertEqual(t._midi_to_key(86), ("1Key14", 1))
        self.assertEqual(t._midi_to_key(55), ("1Key0", -1))

    def test_sky_major_transpose_minimizes_scale_distance(self):
        self.assertEqual(self.t._compute_transpose([60, 62, 64, 65]), 0)
        self.assertEqual(self.t._compute_transpose([61, 63, 65]), -1)

    def test_sky_major_transpose_moves_register_into_two_octaves(self):
        high_register = [72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96]
        low_register = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72]
        self.assertEqual(self.t._compute_transpose(high_register), -12)
        self.assertEqual(self.t._compute_transpose(low_register), 12)


class TestChromaticCompatibility(unittest.TestCase):
    def setUp(self):
        self.t = _bare_transcriber("chromatic")

    def test_chromatic_mapping_matches_legacy_behavior(self):
        self.assertEqual(self.t._midi_to_key(60), ("1Key0", 0))
        self.assertEqual(self.t._midi_to_key(70), ("1Key10", 0))
        self.assertEqual(self.t._midi_to_key(74), ("1Key14", 0))
        self.assertEqual(self.t._midi_to_key(50), ("1Key2", -1))
        self.assertEqual(self.t._midi_to_key(90), ("1Key6", 1))

    def test_chromatic_transpose_matches_legacy_behavior(self):
        self.assertEqual(self.t._compute_transpose([]), 0)
        self.assertEqual(self.t._compute_transpose([60, 61, 62, 70]), 0)
        self.assertEqual(self.t._compute_transpose([54, 55, 56]), 6)
        self.assertEqual(self.t._compute_transpose([78, 79, 80]), -6)


class TestPostProcessing(unittest.TestCase):
    def _process(self, notes, **kwargs):
        kwargs.setdefault("profile", "chord")
        kwargs.setdefault("max_polyphony", 3)
        config = TranscribeConfig(quantize=False, **kwargs)
        return NotePostProcessor(config).process(notes)

    def test_close_onsets_are_merged(self):
        notes = [
            SkyNote(100, "1Key0", 60, 0.9),
            SkyNote(125, "1Key1", 62, 0.5),
        ]
        processed, dropped, bpm = self._process(notes, merge_onset_ms=35)
        self.assertEqual(len(processed), 2)
        self.assertEqual(len({n.time for n in processed}), 1)
        self.assertEqual(dropped, 0)
        self.assertIsNone(bpm)

    def test_repeated_key_is_deduped(self):
        notes = [
            SkyNote(100, "1Key0", 60, 0.9),
            SkyNote(150, "1Key0", 60, 0.8),
        ]
        processed, dropped, _bpm = self._process(
            notes,
            merge_onset_ms=0,
            dedupe_key_ms=90,
        )
        self.assertEqual([n.time for n in processed], [100])
        self.assertEqual(dropped, 1)

    def test_polyphony_is_capped_by_confidence(self):
        notes = [
            SkyNote(100, "1Key0", 60, 0.1),
            SkyNote(100, "1Key1", 62, 0.9),
            SkyNote(100, "1Key2", 64, 0.8),
            SkyNote(100, "1Key3", 65, 0.7),
        ]
        processed, dropped, _bpm = self._process(
            notes,
            merge_onset_ms=0,
            max_polyphony=3,
        )
        self.assertEqual([n.key for n in processed], ["1Key1", "1Key2", "1Key3"])
        self.assertEqual(dropped, 1)

    def test_reasonable_tempo_quantizes_to_sixteenth_grid(self):
        config = TranscribeConfig(
            profile="chord",
            quantize=True,
            merge_onset_ms=0,
            dedupe_key_ms=0,
            max_polyphony=4,
        )
        notes = [
            SkyNote(0, "1Key0", 60, 1.0),
            SkyNote(248, "1Key1", 62, 1.0),
            SkyNote(510, "1Key2", 64, 1.0),
            SkyNote(760, "1Key3", 65, 1.0),
        ]
        processed, dropped, bpm = NotePostProcessor(config).process(notes)
        self.assertEqual([n.time for n in processed], [0, 250, 500, 750])
        self.assertEqual(bpm, 120)
        self.assertEqual(dropped, 0)

    def test_melody_profile_drops_weak_and_short_notes(self):
        config = TranscribeConfig(
            profile="melody",
            quantize=False,
            min_confidence=0.35,
            min_note_duration_ms=90,
            merge_onset_ms=0,
            dedupe_key_ms=0,
            min_event_gap_ms=0,
        )
        notes = [
            SkyNote(0, "1Key0", 60, 0.9, duration_ms=140, instrument="voice"),
            SkyNote(200, "1Key1", 62, 0.2, duration_ms=140, instrument="voice"),
            SkyNote(400, "1Key2", 64, 0.9, duration_ms=40, instrument="voice"),
        ]
        processed, dropped, _bpm = NotePostProcessor(config).process(notes)
        self.assertEqual([n.key for n in processed], ["1Key0"])
        self.assertEqual(dropped, 2)

    def test_melody_profile_filters_drums_and_bass(self):
        config = TranscribeConfig(
            profile="melody",
            quantize=False,
            merge_onset_ms=0,
            dedupe_key_ms=0,
            min_event_gap_ms=0,
        )
        notes = [
            SkyNote(0, "1Key0", 60, 1.0, duration_ms=160, instrument="drums"),
            SkyNote(100, "1Key1", 62, 1.0, duration_ms=160, instrument="electric_bass"),
            SkyNote(200, "1Key7", 72, 1.0, duration_ms=160, instrument="voice"),
        ]
        processed, dropped, _bpm = NotePostProcessor(config).process(notes)
        self.assertEqual([n.instrument for n in processed], ["voice"])
        self.assertEqual(dropped, 2)

    def test_melody_profile_prefers_melodic_instrument(self):
        config = TranscribeConfig(
            profile="melody",
            quantize=False,
            merge_onset_ms=0,
            dedupe_key_ms=0,
            min_event_gap_ms=0,
        )
        notes = [
            SkyNote(100, "1Key2", 64, 1.0, duration_ms=160, instrument="organ"),
            SkyNote(100, "1Key5", 69, 1.0, duration_ms=160, instrument="voice"),
            SkyNote(100, "1Key9", 76, 1.0, duration_ms=160, instrument="synth_pad"),
        ]
        processed, dropped, _bpm = NotePostProcessor(config).process(notes)
        self.assertEqual([n.instrument for n in processed], ["voice"])
        self.assertEqual(dropped, 2)

    def test_melody_profile_thins_dense_events(self):
        config = TranscribeConfig(
            profile="melody",
            quantize=False,
            merge_onset_ms=0,
            dedupe_key_ms=0,
            min_event_gap_ms=100,
        )
        notes = [
            SkyNote(0, "1Key0", 60, 0.6, duration_ms=160, instrument="voice"),
            SkyNote(70, "1Key5", 69, 0.9, duration_ms=160, instrument="voice"),
            SkyNote(190, "1Key6", 71, 0.8, duration_ms=160, instrument="voice"),
        ]
        processed, dropped, _bpm = NotePostProcessor(config).process(notes)
        self.assertEqual([n.time for n in processed], [70, 190])
        self.assertEqual(dropped, 1)


class TestSkyMelodyArranger(unittest.TestCase):
    def _arrange(self, events, **kwargs):
        config = TranscribeConfig(
            quantize=False,
            arranger_cluster_ms=kwargs.pop("arranger_cluster_ms", 90),
            arranger_phrase_gap_ms=kwargs.pop("arranger_phrase_gap_ms", 900),
            arranger_min_note_gap_ms=kwargs.pop("arranger_min_note_gap_ms", 120),
            **kwargs,
        )
        stats = TranscribeStats(raw_event_count=len(events))
        return SkyMelodyArranger(config).arrange(events, stats)

    def test_arranger_filters_drums_and_bass(self):
        events = [
            PitchEvent(0, 60, 1.0, duration_ms=200, instrument="drums"),
            PitchEvent(100, 40, 1.0, duration_ms=200, instrument="electric_bass"),
            PitchEvent(200, 64, 1.0, duration_ms=200, instrument="voice"),
        ]
        notes, stats = self._arrange(events)
        self.assertEqual([n.instrument for n in notes], ["voice"])
        self.assertEqual(stats.arranger_candidate_count, 1)
        self.assertEqual(stats.arranger_selected_count, 1)
        self.assertEqual(stats.arranger_deleted_count, 2)

    def test_arranger_cluster_prefers_main_melody_candidate(self):
        events = [
            PitchEvent(100, 64, 1.0, duration_ms=200, instrument="organ"),
            PitchEvent(130, 69, 1.0, duration_ms=200, instrument="voice"),
            PitchEvent(150, 76, 1.0, duration_ms=200, instrument="synth_pad"),
        ]
        notes, stats = self._arrange(events, arranger_cluster_ms=90)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].instrument, "voice")
        self.assertEqual(stats.arranger_candidate_count, 1)

    def test_arranger_phrase_transpose_moves_high_register_into_sky_range(self):
        events = [
            PitchEvent(0, 96, 1.0, duration_ms=240, instrument="voice"),
            PitchEvent(240, 98, 1.0, duration_ms=240, instrument="voice"),
            PitchEvent(480, 100, 1.0, duration_ms=240, instrument="voice"),
        ]
        notes, stats = self._arrange(events, arranger_cluster_ms=0)
        self.assertEqual(stats.arranger_phrase_count, 1)
        self.assertTrue(-24 <= stats.transpose <= 24)
        self.assertEqual(len(notes), 3)
        self.assertTrue(all(60 <= n.midi <= 84 for n in notes))
        self.assertEqual(
            [
                SkyKeyMapper.key_to_index(notes[1].key) - SkyKeyMapper.key_to_index(notes[0].key),
                SkyKeyMapper.key_to_index(notes[2].key) - SkyKeyMapper.key_to_index(notes[1].key),
            ],
            [1, 1],
        )

    def test_arranger_dynamic_path_avoids_repeated_boundary_key(self):
        events = [
            PitchEvent(0, 84, 1.0, duration_ms=250, instrument="voice"),
            PitchEvent(240, 84, 1.0, duration_ms=250, instrument="voice"),
            PitchEvent(480, 84, 1.0, duration_ms=250, instrument="voice"),
        ]
        notes, stats = self._arrange(events, arranger_cluster_ms=0)
        self.assertEqual(len(notes), 3)
        self.assertLess(stats.arranger_boundary_hit_count, 3)
        self.assertNotEqual([n.key for n in notes], ["1Key14", "1Key14", "1Key14"])

    def test_arranger_deletes_dense_ornament_and_keeps_long_notes(self):
        events = [
            PitchEvent(0, 60, 1.0, duration_ms=420, instrument="voice"),
            PitchEvent(50, 62, 1.0, duration_ms=120, instrument="voice"),
            PitchEvent(220, 64, 1.0, duration_ms=420, instrument="voice"),
        ]
        notes, stats = self._arrange(
            events,
            arranger_cluster_ms=0,
            arranger_min_note_gap_ms=120,
        )
        self.assertEqual([n.time for n in notes], [0, 220])
        self.assertEqual(stats.arranger_selected_count, 2)
        self.assertEqual(stats.arranger_deleted_count, 1)


class TestMuScriptorAnalyzer(unittest.TestCase):
    def test_mocked_muscriptor_events_are_parsed(self):
        start_voice = FakeStartEvent(64, 0.25, 1, "voice")
        start_bass = FakeStartEvent(40, 0.5, 2, "electric_bass")

        class FakeModel:
            def transcribe(self, **kwargs):
                self.kwargs = kwargs
                yield FakeProgressEvent(0, 1)
                yield start_voice
                yield FakeEndEvent(0.75, start_voice)
                yield start_bass
                yield FakeEndEvent(0.65, start_bass)

        class FakeModelClass:
            loaded = None

            @classmethod
            def load_model(cls, weights_path=None, device=None):
                cls.loaded = (weights_path, device)
                return FakeModel()

        config = TranscribeConfig(muscriptor_model="medium", muscriptor_device="auto")
        with patch(
            "transcription.analysis._load_muscriptor_model_class",
            return_value=FakeModelClass,
        ):
            analyzer = MuScriptorAnalyzer(config)
            events, stats = analyzer.analyze("fake.wav")

        self.assertEqual(FakeModelClass.loaded, ("medium", None))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].time_ms, 250)
        self.assertEqual(events[0].midi, 64)
        self.assertEqual(events[0].duration_ms, 500)
        self.assertEqual(events[0].instrument, "voice")
        self.assertEqual(events[0].source, "muscriptor")
        self.assertEqual(stats.engine_used, "muscriptor")
        self.assertEqual(stats.muscriptor_model, "medium")
        self.assertEqual(stats.instrument_counts, {"electric_bass": 1, "voice": 1})
        self.assertIn(b'"type": "start"', analyzer.last_artifacts["events_jsonl"])

    def test_unsupported_muscriptor_kwargs_are_not_sent(self):
        start = FakeStartEvent(67, 0.1, 1, "flute")

        class FakeModel:
            def __init__(self):
                self.received = {}

            def transcribe(
                self,
                audio,
                instruments=None,
                batch_size=None,
                no_eos_is_ok=True,
                beam_size=1,
                cfg_coef=1.0,
            ):
                self.received = {
                    "audio": audio,
                    "instruments": instruments,
                    "batch_size": batch_size,
                    "no_eos_is_ok": no_eos_is_ok,
                    "beam_size": beam_size,
                    "cfg_coef": cfg_coef,
                }
                yield start
                yield FakeEndEvent(0.4, start)

        class FakeModelClass:
            model = FakeModel()

            @classmethod
            def load_model(cls, weights_path=None, device=None):
                return cls.model

        config = TranscribeConfig(muscriptor_prelude_forcing=True)
        with patch(
            "transcription.analysis._load_muscriptor_model_class",
            return_value=FakeModelClass,
        ):
            analyzer = MuScriptorAnalyzer(config)
            events, _stats = analyzer.analyze("fake.wav")

        self.assertEqual(len(events), 1)
        self.assertEqual(FakeModelClass.model.received["audio"], "fake.wav")
        self.assertEqual(FakeModelClass.model.received["cfg_coef"], 1.0)
        self.assertNotIn("prelude_forcing", FakeModelClass.model.received)

    def test_load_failure_mentions_hugging_face_and_license(self):
        class FailingModelClass:
            @classmethod
            def load_model(cls, weights_path=None, device=None):
                raise RuntimeError("401 gated repo")

        with patch(
            "transcription.analysis._load_muscriptor_model_class",
            return_value=FailingModelClass,
        ):
            with self.assertRaisesRegex(
                AnalyzerUnavailableError,
                "Hugging Face|license|HF_TOKEN",
            ):
                MuScriptorAnalyzer(TranscribeConfig())


class TestAnalyzerFactory(unittest.TestCase):
    def test_default_engine_is_muscriptor_only(self):
        config = TranscribeConfig()
        self.assertEqual(config.engine, "muscriptor")
        self.assertEqual(config.profile, "melody")
        self.assertEqual(config.max_polyphony, 3)
        self.assertFalse(config.drop_out_of_range)
        self.assertEqual(config.out_of_range_policy, "adaptive")
        self.assertEqual(config.muscriptor_model, "large")
        self.assertTrue(config.arranger_enabled)
        self.assertEqual(config.arranger_profile, "sky_melody")
        self.assertEqual(config.arranger_cluster_ms, 90)
        self.assertEqual(config.arranger_phrase_gap_ms, 900)
        self.assertEqual(config.arranger_min_note_gap_ms, 120)
        self.assertFalse(config.arranger_keep_chords)

    def test_muscriptor_model_uses_official_names_only(self):
        self.assertEqual(TranscribeConfig(muscriptor_model="large").muscriptor_model, "large")
        with self.assertRaisesRegex(ValueError, "Unknown MuScriptor model size"):
            TranscribeConfig(muscriptor_model="high")

    def test_factory_creates_muscriptor_analyzer(self):
        class FakeMuScriptor:
            def __init__(self, config):
                self.config = config

        with patch("transcription.analysis.MuScriptorAnalyzer", FakeMuScriptor):
            analyzer, error = AnalyzerFactory.create(TranscribeConfig())
        self.assertIsInstance(analyzer, FakeMuScriptor)
        self.assertIsNone(error)

    def test_non_muscriptor_engine_is_rejected(self):
        for engine in ["auto", "librosa", "basic_pitch", "mt3"]:
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(ValueError, "Only the MuScriptor"):
                    TranscribeConfig(engine=engine)


class TestTranscriberEngines(unittest.TestCase):
    def test_mocked_muscriptor_events_flow_to_json_and_stats(self):
        class FakeAnalyzer:
            def analyze(self, audio_path):
                return [
                    PitchEvent(
                        time_ms=0,
                        midi=60,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="voice",
                    ),
                    PitchEvent(
                        time_ms=250,
                        midi=64,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="voice",
                    ),
                ], TranscribeStats(
                    mapping="sky_major",
                    engine_requested="muscriptor",
                    engine_used="muscriptor",
                    raw_event_count=2,
                    onset_count=2,
                    muscriptor_model="medium",
                    instrument_counts={"voice": 2},
                )

        with patch(
            "transcription.core.AnalyzerFactory.create",
            return_value=(FakeAnalyzer(), None),
        ):
            transcriber = Transcriber(TranscribeConfig(quantize=False))
        notes, stats = transcriber.transcribe("fake.wav")
        self.assertEqual(notes, [{"time": 0, "key": "1Key0"}, {"time": 250, "key": "1Key2"}])
        self.assertEqual(stats.engine_requested, "muscriptor")
        self.assertEqual(stats.engine_used, "muscriptor")
        self.assertEqual(stats.instrument_counts, {"voice": 2})
        self.assertTrue(stats.arranger_enabled)
        self.assertEqual(stats.arranger_selected_count, 2)

    def test_melody_transpose_uses_melody_range_and_drops_unplayable_bass(self):
        class FakeAnalyzer:
            def analyze(self, audio_path):
                return [
                    PitchEvent(
                        time_ms=0,
                        midi=96,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="voice",
                    ),
                    PitchEvent(
                        time_ms=250,
                        midi=36,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="electric_bass",
                    ),
                ], TranscribeStats(
                    mapping="sky_major",
                    engine_requested="muscriptor",
                    engine_used="muscriptor",
                    raw_event_count=2,
                    onset_count=2,
                    muscriptor_model="medium",
                    instrument_counts={"electric_bass": 1, "voice": 1},
                )

        with patch(
            "transcription.core.AnalyzerFactory.create",
            return_value=(FakeAnalyzer(), None),
        ):
            transcriber = Transcriber(TranscribeConfig(quantize=False))
        notes, stats = transcriber.transcribe("fake.wav")
        self.assertEqual(stats.transpose, -12)
        self.assertEqual(notes, [{"time": 0, "key": "1Key14"}])
        self.assertEqual(stats.mapped_event_count, 1)
        self.assertGreaterEqual(stats.dropped_count, 1)

    def test_drop_policy_removes_unplayable_melody_notes(self):
        class FakeAnalyzer:
            def analyze(self, audio_path):
                return [
                    PitchEvent(
                        time_ms=0,
                        midi=60,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="voice",
                    ),
                    PitchEvent(
                        time_ms=250,
                        midi=96,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="voice",
                    ),
                ], TranscribeStats(
                    mapping="sky_major",
                    engine_requested="muscriptor",
                    engine_used="muscriptor",
                    raw_event_count=2,
                    onset_count=2,
                    muscriptor_model="medium",
                    instrument_counts={"voice": 2},
                )

        with patch(
            "transcription.core.AnalyzerFactory.create",
            return_value=(FakeAnalyzer(), None),
        ):
            transcriber = Transcriber(
                TranscribeConfig(
                    quantize=False,
                    arranger_enabled=False,
                    out_of_range_policy="drop",
                )
            )
        notes, stats = transcriber.transcribe("fake.wav")
        self.assertFalse(stats.arranger_enabled)
        self.assertEqual(stats.transpose, 0)
        self.assertEqual(notes, [{"time": 0, "key": "1Key0"}])
        self.assertEqual(stats.mapped_event_count, 1)
        self.assertEqual(stats.dropped_count, 1)

    def test_muscriptor_runtime_failure_is_not_fallbacked(self):
        class FailingMuScriptor:
            def analyze(self, audio_path):
                raise RuntimeError("muscriptor exploded")

        with patch(
            "transcription.core.AnalyzerFactory.create",
            return_value=(FailingMuScriptor(), None),
        ):
            transcriber = Transcriber(TranscribeConfig(quantize=False))
            with self.assertRaisesRegex(RuntimeError, "muscriptor exploded"):
                transcriber.transcribe("fake.wav")

    def test_debug_outputs_include_arranger_artifacts(self):
        class FakeAnalyzer:
            def analyze(self, audio_path):
                return [
                    PitchEvent(
                        time_ms=0,
                        midi=60,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="voice",
                    ),
                    PitchEvent(
                        time_ms=250,
                        midi=64,
                        confidence=1.0,
                        source="muscriptor",
                        duration_ms=250,
                        instrument="voice",
                    ),
                ], TranscribeStats(
                    mapping="sky_major",
                    engine_requested="muscriptor",
                    engine_used="muscriptor",
                    raw_event_count=2,
                    onset_count=2,
                    muscriptor_model="medium",
                    instrument_counts={"voice": 2},
                )

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "transcription.core.AnalyzerFactory.create",
                return_value=(FakeAnalyzer(), None),
            ):
                transcriber = Transcriber(
                    TranscribeConfig(quantize=False, debug_outputs=True)
                )
            _out_path, song = transcriber.write_song_json(
                "fake.wav",
                tmp,
                song_name="debug-song",
            )

            outputs = song["_debug_outputs"]
            self.assertTrue(os.path.exists(outputs["melody_candidates"]))
            self.assertTrue(os.path.exists(outputs["arranged"]))
            self.assertTrue(os.path.exists(outputs["final"]))
            report_path = outputs["muscriptor_arranger_report"]
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as file_obj:
                report = json.load(file_obj)
            self.assertTrue(report["arranger_enabled"])
            self.assertEqual(report["selected_count"], 2)


class TestIsAudioFile(unittest.TestCase):
    def test_recognized(self):
        for path in ["a.mp3", "b.WAV", "c.Flac", "d.ogg", "e.m4a", "f.AAC"]:
            self.assertTrue(is_audio_file(path), path)

    def test_rejected(self):
        for path in ["a.json", "b.txt", "c", "d.mp4"]:
            self.assertFalse(is_audio_file(path), path)


class TestWriteMidi(unittest.TestCase):
    def _write(self, notes, bpm=120, mapping="sky_major"):
        transcriber = _bare_transcriber(mapping)
        fd, path = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        self.addCleanup(os.remove, path)
        transcriber.write_midi_file(notes, path, bpm=bpm)
        with open(path, "rb") as f:
            return path, f.read()

    def test_valid_smf_header(self):
        notes = [{"time": 0, "key": "1Key0"}, {"time": 500, "key": "1Key7"}]
        path, data = self._write(notes)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(len(data), 22)
        self.assertTrue(data.startswith(b"MThd"))
        self.assertEqual(data[8:10], b"\x00\x00")
        self.assertEqual(data[10:12], b"\x00\x01")
        self.assertEqual(data[12:14], b"\x01\xe0")
        self.assertIn(b"MTrk", data)
        self.assertIn(b"\xFF\x2F\x00", data)

    def test_sky_major_note_events_present(self):
        notes = [{"time": 0, "key": "1Key0"}, {"time": 500, "key": "1Key14"}]
        _path, data = self._write(notes)
        self.assertIn(b"\x90\x3C\x50", data)
        self.assertIn(b"\x80\x3C\x50", data)
        self.assertIn(b"\x90\x54\x50", data)
        self.assertIn(b"\xFF\x51\x03\x07\xA1\x20", data)

    def test_chromatic_midi_preview_remains_available(self):
        _path, data = self._write(
            [{"time": 0, "key": "1Key14"}],
            mapping="chromatic",
        )
        self.assertIn(b"\x90\x4A\x50", data)

    def test_tempo_respects_bpm(self):
        _path, data = self._write([{"time": 0, "key": "1Key0"}], bpm=60)
        self.assertIn(b"\xFF\x51\x03\x0F\x42\x40", data)

    def test_empty_notes_still_valid(self):
        _path, data = self._write([])
        self.assertTrue(data.startswith(b"MThd"))
        self.assertIn(b"\xFF\x2F\x00", data)

    def test_chord_same_time(self):
        notes = [{"time": 0, "key": "1Key0"}, {"time": 0, "key": "1Key4"}]
        _path, data = self._write(notes)
        self.assertEqual(data.count(b"\x90\x3C\x50"), 1)
        self.assertEqual(data.count(b"\x90\x43\x50"), 1)
        self.assertEqual(data.count(b"\x80\x3C\x50"), 1)
        self.assertEqual(data.count(b"\x80\x43\x50"), 1)

    def test_note_off_precedes_note_on_at_same_tick(self):
        notes = [{"time": 0, "key": "1Key0"}, {"time": 60, "key": "1Key4"}]
        _path, data = self._write(notes)
        first_off = data.find(b"\x80\x3C\x50")
        second_on = data.find(b"\x90\x43\x50")
        self.assertGreaterEqual(first_off, 0)
        self.assertGreaterEqual(second_on, 0)
        self.assertLess(first_off, second_on)


if __name__ == "__main__":
    unittest.main()
