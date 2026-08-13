import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.benchmark import _score_notes, evaluate_15_key
from transcription.models import QualityAnalysisDraft, SymbolicNote, TranscriptionError, TranscriptionOptions
from transcription.pipeline import export_song_json, transcribe_draft
from transcription.quality import arrange_quality_analysis, resolve_quality_model


def quality_draft():
    return QualityAnalysisDraft(
        duration_sec=4.0,
        symbolic_notes=[
            SymbolicNote(12, 400, 72, "voice", "melody"),
            SymbolicNote(505, 900, 74, "voice", "melody"),
            SymbolicNote(1000, 1300, 76, "voice", "melody"),
            SymbolicNote(0, 500, 48, "acoustic_bass", "bass"),
            SymbolicNote(500, 1000, 55, "acoustic_guitar", "harmony"),
            SymbolicNote(1000, 1500, 60, "acoustic_guitar", "harmony"),
        ],
        beat_times_ms=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        downbeat_times_ms=[0, 2000, 4000], bar_starts_ms=[0, 2000, 4000],
        bpm=120.0, meter="4/4", timing_confidence=.9, model_name="small", device="cpu",
    )


class TestQualityArrangement(unittest.TestCase):
    def test_quality_requires_explicit_rights_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "input.wav")
            open(source, "wb").close()
            with self.assertRaisesRegex(TranscriptionError, "确认"):
                transcribe_draft(source, TranscriptionOptions(engine="quality"))

    def test_melody_survives_polyphony_limit_and_quantizes_small_error(self):
        notes, _key, _shift, stats = arrange_quality_analysis(quality_draft(), TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", max_polyphony=2, arrangement_preset="full"))
        by_time = {}
        for note in notes: by_time.setdefault(note["time"], []).append(note["key"])
        self.assertIn("1Key7", by_time[0])
        self.assertTrue(all(len(keys) <= 2 for keys in by_time.values()))
        self.assertEqual(stats["melody_note_count"], 3)

    def test_device_model_defaults(self):
        self.assertEqual(resolve_quality_model("auto", "cpu"), "small")
        self.assertEqual(resolve_quality_model("auto", "mps"), "medium")
        self.assertEqual(resolve_quality_model("small", "cuda"), "small")

    def test_v3_export_metadata_keeps_song_schema(self):
        draft = quality_draft()
        notes, key, shift, stats = arrange_quality_analysis(draft, TranscriptionOptions(engine="quality", rights_confirmed=True))
        from transcription.models import TranscriptionResult
        result = TranscriptionResult([], notes, key, 120, stats, [], engine="arrangement_v3_quality", options=TranscriptionOptions(engine="quality", rights_confirmed=True), quality_analysis=draft, semitone_shift=shift)
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "out.json"); export_song_json(result, output, "demo")
            with open(output, encoding="utf-8") as handle:
                data = json.load(handle)[0]
        self.assertEqual(data["_transcribe"]["schemaVersion"], 3)
        self.assertEqual(data["_transcribe"]["qualityModel"], "small")
        self.assertTrue(data["songNotes"])


class TestQualityMetrics(unittest.TestCase):
    def test_score_metrics_report_timing_and_pitch_separately(self):
        reference = [{"time": 0, "key": "1Key7"}, {"time": 500, "key": "1Key8"}]
        estimate = [{"time": 30, "key": "1Key7"}, {"time": 500, "key": "1Key6"}]
        metrics = evaluate_15_key(reference, estimate)
        self.assertEqual(metrics["onset"]["f1"], 1.0)
        self.assertEqual(metrics["key_onset"]["matched"], 1)
        self.assertEqual(metrics["key_onset"]["onset_mae_ms"], 30)

    def test_utf16_partial_score_is_read_and_scores_only_covered_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hand-score.json")
            with open(path, "w", encoding="utf-16") as handle:
                json.dump([{"songNotes": [{"time": 0, "key": "1Key1"}, {"time": 500, "key": "1Key2"}]}], handle)
            reference = _score_notes(Path(path))
        estimate = [*reference, {"time": 4000, "key": "1Key9"}]
        metrics = evaluate_15_key(reference, estimate)
        self.assertEqual(metrics["key_onset"]["f1"], 1.0)
        self.assertEqual(metrics["estimate_note_count"], 2)


if __name__ == "__main__": unittest.main()
