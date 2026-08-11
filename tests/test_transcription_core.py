import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_loader import load_score
from transcription.arranger import SKY_MIDI, arrange_analysis, arrange_events, key_transpose
from transcription.analysis import _choose_tempo, _nearest_grid
from transcription.models import AnalysisDraft, ChordSpan, MelodyNote, NoteEvent, Section, TempoMap, TranscriptionError, TranscriptionOptions, TranscriptionResult
from transcription.pipeline import export_song_json, rearrange_draft


def draft() -> AnalysisDraft:
    beats = tuple(range(0, 8001, 500))
    return AnalysisDraft(
        duration_sec=8.0,
        tempo_map=TempoMap(120.0, "4/4", beats, (0, 2000, 4000, 6000, 8000), tuple(range(0, 8001, 125)), 0.9),
        detected_key="C major", key_confidence=0.9, semitone_shift=0,
        melody=[MelodyNote(0, 450, 72, .9, "vocal"), MelodyNote(1000, 1450, 74, .9, "vocal"), MelodyNote(4000, 4450, 76, .9, "vocal")],
        chords=[ChordSpan(0, 2000, 0, "maj", 0, .9), ChordSpan(2000, 4000, 5, "maj", 5, .9), ChordSpan(4000, 6000, 7, "7", 7, .9), ChordSpan(6000, 8000, 0, "maj", 0, .9)],
        sections=[Section(0, 4000, "verse", 1), Section(4000, 8000, "chorus", 3)],
        lead_source="vocal", melody_confidence=.9, harmony_confidence=.9, structure_confidence=.8,
    )


class TestV2Options(unittest.TestCase):
    def test_halftime_tempo_avoids_double_time(self):
        self.assertAlmostEqual(_choose_tempo(161.0, None), 80.5)
        self.assertAlmostEqual(_choose_tempo(80.5, None), 80.5)
        self.assertEqual(_choose_tempo(161.0, 100.0), 100.0)

    def test_grid_nearest_is_monotonic_by_construction(self):
        grid = [0, 125, 250, 375]
        self.assertEqual(_nearest_grid(grid, 188), 250)
        self.assertEqual([_nearest_grid(grid, time) for time in (1, 124, 249, 374)], [0, 125, 250, 375])

    def test_v2_option_bounds(self):
        self.assertEqual(TranscriptionOptions().arrangement_preset, "auto")
        for value in range(2, 6):
            self.assertEqual(TranscriptionOptions(max_polyphony=value).max_polyphony, value)
        with self.assertRaises(ValueError):
            TranscriptionOptions(max_polyphony=1)
        with self.assertRaises(ValueError):
            TranscriptionOptions(mode="stem_fusion")

    def test_key_mapping_and_midi_compatibility(self):
        events = [NoteEvent(index * 100, index * 100 + 80, pitch, 1.0, "midi") for index, pitch in enumerate(SKY_MIDI)]
        notes, key, shift, _octave, _stats = arrange_events(events, TranscriptionOptions(mode="midi", source_key="C major"))
        self.assertEqual(key, "C major")
        self.assertEqual(shift, 0)
        self.assertEqual([note["key"] for note in notes], [f"1Key{i}" for i in range(15)])
        self.assertEqual(key_transpose("G major"), 5)


class TestV2Arrangement(unittest.TestCase):
    def test_chorus_is_denser_and_respects_polyphony(self):
        analysis = draft()
        notes, stats, _roles = arrange_analysis(analysis, TranscriptionOptions(arrangement_preset="auto", max_polyphony=4))
        grouped = {}
        for note in notes:
            grouped.setdefault(note["time"], []).append(note["key"])
        self.assertTrue(all(0 <= int(key[-1]) <= 14 for keys in grouped.values() for key in keys))
        self.assertTrue(all(len(keys) <= 4 and len(keys) == len(set(keys)) for keys in grouped.values()))
        verse = sum(len(keys) for time, keys in grouped.items() if time < 4000)
        chorus = sum(len(keys) for time, keys in grouped.items() if time >= 4000)
        self.assertGreater(chorus / 4, verse / 4 * 1.25)
        self.assertGreater(stats["chord_onset_ratio"], 0.3)

    def test_rearrange_uses_cached_analysis(self):
        analysis = draft()
        original = TranscriptionResult([], [], "C major", 120.0, {}, [], analysis=analysis)
        result = rearrange_draft(original, TranscriptionOptions(arrangement_preset="simple", max_polyphony=2))
        self.assertTrue(result.song_notes)
        self.assertTrue(all(note["key"].startswith("1Key") for note in result.song_notes))
        with self.assertRaises(TranscriptionError):
            rearrange_draft(original, TranscriptionOptions(bpm_override=100))


class TestV2Export(unittest.TestCase):
    def test_json_is_compatible_and_has_no_artifact_path(self):
        analysis = draft()
        result = TranscriptionResult([], [{"time": 0, "key": "1Key7"}], "C major", 120.0, {}, [], source_file="secret.wav", analysis=analysis)
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "song.json")
            export_song_json(result, output, "Song")
            self.assertEqual(load_score(output, valid_keys={"1Key7"})["notes_by_time"][0], ["1Key7"])
            with open(output, encoding="utf-8") as handle:
                payload = json.load(handle)[0]
        self.assertEqual(payload["_transcribe"]["engine"], "arrangement_v2")
        self.assertEqual(payload["_transcribe"]["meter"], "4/4")
        self.assertNotIn("artifact", json.dumps(payload).lower())


if __name__ == "__main__":
    unittest.main()
