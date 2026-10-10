import threading
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

from transcription.analysis import _events_to_melody, extract_melody
from transcription.backends import BackendOutput, _monophonic_chunk
from transcription.fidelity import check_vocal_regions, map_phrase_keys, select_phrase_melody, supported_pitch, suspect_regions
from transcription.models import CancelledError, NoteEvent, QualityAnalysisDraft, SymbolicNote, TempoMap, TranscriptionError, TranscriptionOptions, VocalEvidence
from transcription.quality import _timing_from_model, _beat_score, _best_shift, _arrangement_shift, _map_quality_melody
from transcription.benchmark import _load_quality_cache, _save_quality_cache
from transcription.models import TranscriptionResult
from transcription.arranger import arrange_analysis
from transcription.models import AnalysisDraft, MelodyNote, Section


def draft(notes, duration=40):
    beats = list(range(0, int(duration*1000)+1, 500))
    return QualityAnalysisDraft(duration, notes, beats, beats[::4], beats[::4], 120, '4/4', .9, 'small', 'cpu')


class VocalFidelityTests(unittest.TestCase):
    def test_independent_same_pitch_attacks_survive_fast_arrangement(self):
        events = [NoteEvent(t, t+200, 72, .9, 'pyin') for t in (0, 250, 500)]
        actual = _events_to_melody(events, list(range(0, 1000, 125)), 'vocal')
        self.assertEqual([n.start_ms for n in actual], [0, 250, 500])

    def test_fast_quantization_does_not_merge_independent_attacks(self):
        events = [NoteEvent(t, t+100, 72, .9, 'pyin') for t in (90, 125)]
        actual = _events_to_melody(events, list(range(0, 1000, 125)), 'vocal')
        self.assertEqual([n.start_ms for n in actual], [90, 125])

    def test_cross_detector_duplicates_remain_one_attack(self):
        events = [NoteEvent(0, 200, 72, .9, 'pyin'), NoteEvent(20, 220, 72, .8, 'basic_pitch')]
        actual = _events_to_melody(events, list(range(0, 1000, 125)), 'vocal')
        self.assertEqual(len(actual), 1)

    def test_long_detector_sustain_does_not_swallow_a_new_attack(self):
        events = [NoteEvent(0, 600, 72, .99, 'basic_pitch'), NoteEvent(50, 250, 72, .9, 'pyin')]
        actual = _events_to_melody(events, list(range(0, 1000, 125)), 'vocal')
        self.assertEqual([n.start_ms for n in actual], [0, 50])

    def test_steady_pitch_run_splits_at_rearticulations(self):
        import numpy as np
        values = (np.full(100, 440.), np.ones(100, dtype=bool), np.full(100, .99))
        with patch('librosa.pyin', return_value=values), patch('librosa.onset.onset_detect', return_value=np.array([0, .4, .8])):
            events = _monophonic_chunk(np.zeros(25600), 22050, .5)
        self.assertEqual(len(events), 3)
        self.assertTrue(all(event.midi_pitch == 69 and event.end_ms > event.start_ms for event in events))

    def test_one_detector_failure_keeps_other_vocal_result(self):
        events = [NoteEvent(t, t+200, 72, .99, 'basic_pitch') for t in range(0, 2000, 250)]
        tempo = TempoMap(120, '4/4', tuple(range(0, 2500, 500)), (0, 2000), tuple(range(0, 2500, 125)), .9)
        def failed(*_args):
            raise TranscriptionError('offline')
        with patch('transcription.analysis.transcribe_monophonic', failed), patch('transcription.analysis.transcribe_polyphonic', return_value=BackendOutput(events, 120, [], 'basic_pitch', 2)):
            melody, source, _confidence, _raw, warnings = extract_melody('vocal.wav', 'mix.wav', tempo, None, None)
        self.assertEqual(source, 'vocal')
        self.assertEqual(len(melody), 8)
        self.assertTrue(warnings)

    def test_vocal_breath_is_not_filled_by_piano(self):
        voice = SymbolicNote(0, 300, 72, 'voice', 'melody')
        nearby = SymbolicNote(500, 900, 76, 'acoustic_piano', 'melody')
        later = SymbolicNote(1600, 2000, 76, 'acoustic_piano', 'melody')
        selected, _segments = select_phrase_melody([voice, nearby, later], [0, 500, 1000, 1500, 2000])
        self.assertEqual(selected, [voice, later])

    def test_octave_repair_needs_confidence_and_union_coverage(self):
        note = SymbolicNote(0, 400, 84, 'voice', 'melody')
        strong = [VocalEvidence(0, 400, 72, .95, 'present')]
        selected, _segments = select_phrase_melody([note], [0, 500], strong)
        self.assertEqual(selected[0].midi_pitch, 72)
        self.assertEqual(note.midi_pitch, 84)
        self.assertIsNone(supported_pitch(note, [VocalEvidence(0, 200, 72, .99, 'present')]*4))
        self.assertIsNone(supported_pitch(note, [VocalEvidence(0, 400, 72, .8, 'present')]))
        selected, _segments = select_phrase_melody([], [0, 500], strong)
        self.assertEqual(selected, [])

    def test_phrase_dp_uses_later_notes_to_resolve_early_harmony(self):
        notes = [SymbolicNote(0, 180, 74, 'voice', 'melody'), SymbolicNote(0, 180, 84, 'voice', 'melody'),
                 SymbolicNote(250, 450, 84, 'voice', 'melody'), SymbolicNote(500, 700, 86, 'voice', 'melody')]
        selected, _segments = select_phrase_melody(notes, [0, 500, 1000])
        self.assertEqual([n.midi_pitch for n in selected], [84, 84, 86])

    def test_suspect_contexts_respect_limits(self):
        notes = [SymbolicNote(t, t+200, 72+(12 if i % 2 else 0), 'voice', 'melody') for i, t in enumerate(range(0, 40000, 250))]
        regions = suspect_regions(draft(notes))
        self.assertTrue(regions)
        self.assertLessEqual(sum(b-a for a, b in regions), 12000)
        self.assertTrue(all(b-a <= 30000 for a, b in regions))
        self.assertTrue(all(a >= previous[1] for previous, (a, b) in zip(regions, regions[1:])))

    def test_cancel_during_evidence_is_not_a_fallback(self):
        event = threading.Event(); event.set()
        value = draft([SymbolicNote(0, 200, 72, 'voice', 'melody'), SymbolicNote(250, 450, 84, 'voice', 'melody')])
        with self.assertRaises(CancelledError):
            check_vocal_regions('missing.wav', value, event)

    def test_actual_downbeats_are_preserved(self):
        tracker = lambda _path: ([t/1000 for t in range(0, 6001, 500)], [0, 1.5, 3.5, 5.5])
        with patch('transcription.quality._beat_tracker', return_value=tracker):
            _beats, downbeats, _bars, *_other = _timing_from_model(None, 'x.wav', [], TranscriptionOptions(meter='4/4'))
        self.assertEqual(downbeats, [0, 1500, 3500, 5500])

    def test_accompaniment_accent_follows_actual_bar_anchor(self):
        value = draft([], duration=6)
        value.downbeat_times_ms = [0, 1500, 3500, 5500]
        self.assertEqual(_beat_score(value, 1500, 500), 6)
        self.assertEqual(_beat_score(value, 2000, 500), 1)
        self.assertEqual(_beat_score(value, 2500, 500), 4)

    def test_joint_mapping_keeps_contour_and_manual_transpose(self):
        notes = [SymbolicNote(i*250, i*250+200, pitch, 'voice', 'melody') for i, pitch in enumerate([72, 74, 76, 79])]
        shift, keys = map_phrase_keys(notes, TranscriptionOptions(source_key='C major', melody_octave_shift=0))
        self.assertEqual(shift, 0)
        self.assertEqual(keys, [7, 8, 9, 11])
        shift, _keys = map_phrase_keys(notes, TranscriptionOptions(source_key='D major'))
        self.assertEqual(shift, -2)

    def test_fast_arranger_respects_manual_octave(self):
        tempo = TempoMap(120, '4/4', (0, 500, 1000), (0,), (0, 125, 250), .9)
        analysis = AnalysisDraft(1, tempo, 'C major', .9, 0, [MelodyNote(0, 300, 72, .9, 'vocal')], [], [Section(0, 1000, 'verse', 1)], 'vocal', .9, 0, .5)
        low, _stats, _roles = arrange_analysis(analysis, TranscriptionOptions(melody_octave_shift=-1))
        high, _stats, _roles = arrange_analysis(analysis, TranscriptionOptions(melody_octave_shift=1))
        self.assertEqual(low[0]['key'], '1Key0')
        self.assertEqual(high[0]['key'], '1Key14')

    def test_versioned_evidence_cache_restores_typed_observations_without_changing_raw_notes(self):
        value = draft([SymbolicNote(0, 400, 84, 'voice', 'melody')])
        value.vocal_evidence = [VocalEvidence(0, 400, 72, .95, 'present')]
        value.vocal_evidence_version = 2
        options = TranscriptionOptions(engine='quality', source_key='C major', fidelity_mode='candidate', vocal_check='auto')
        result = TranscriptionResult([], [{'time': 0, 'key': '1Key14'}], 'C major', 120, {}, [],
                                     options=options, quality_analysis=value)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'cache.json'
            _save_quality_cache(path, result)
            restored = _load_quality_cache(path, options)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.quality_analysis.symbolic_notes[0].midi_pitch, 84)
        self.assertIsInstance(restored.quality_analysis.vocal_evidence[0], VocalEvidence)
        self.assertEqual(restored.song_notes[0]['key'], '1Key7')

    def test_low_voice_is_voiced_by_octave_instead_of_clamped_to_c(self):
        notes = [SymbolicNote(i*250, i*250+200, pitch, 'voice', 'melody') for i, pitch in enumerate([45, 47, 48])]
        shift, keys = map_phrase_keys(notes, TranscriptionOptions(source_key='C major'))
        self.assertEqual(shift, 0)
        self.assertEqual(keys, [5, 6, 7])

    def test_forced_negative_tritone_is_preserved(self):
        notes = [SymbolicNote(0, 200, 72, 'voice', 'melody')]
        shift, keys = map_phrase_keys(notes, TranscriptionOptions(melody_octave_shift=0), forced_shift=-6)
        self.assertEqual(shift, -6)
        self.assertIn(keys[0], [3, 4])

    def test_corrected_direction_cost_and_frozen_stable_transpose_are_separate(self):
        notes = [SymbolicNote(0, 200, 78, 'voice', 'melody'), SymbolicNote(250, 450, 76, 'voice', 'melody')]
        options = TranscriptionOptions()
        self.assertEqual(_best_shift(notes, options), -4)
        self.assertEqual(_best_shift(notes, options, preserve_v8=True), -2)
        self.assertEqual(_arrangement_shift(notes, options), -2)

    def test_collision_with_an_exact_grid_attack_restores_both_source_attacks(self):
        notes = [SymbolicNote(90, 200, 72, 'voice', 'melody'), SymbolicNote(125, 250, 72, 'voice', 'melody')]
        value = draft(notes)
        options = TranscriptionOptions(fidelity_mode='candidate', source_key='C major', melody_octave_shift=0)
        mapped = _map_quality_melody(notes, value, options, 0)
        self.assertEqual([time for _note, time, _key in mapped], [90, 125])

    def test_compatibility_export_reports_actual_arranger_and_evidence_versions(self):
        from transcriber import Transcriber
        value = draft([SymbolicNote(0, 200, 72, 'voice', 'melody')])
        value.vocal_evidence_version = 2
        result = TranscriptionResult([], [{'time': 0, 'key': '1Key7'}], 'C major', 120,
                                     {'qualityArrangerVersion': 9}, [], quality_analysis=value,
                                     options=TranscriptionOptions(fidelity_mode='candidate'))
        song = Transcriber.transcribe_to_song_dict(result, 'example')
        self.assertEqual(song['_transcribe']['qualityArrangerVersion'], 9)
        self.assertEqual(song['_transcribe']['vocalEvidenceVersion'], 2)
        self.assertEqual(song['_transcribe']['fidelityMode'], 'candidate')
