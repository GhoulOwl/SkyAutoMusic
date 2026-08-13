import bisect
import json
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.arranger import SKY_MIDI, _pitch_to_key, key_transpose
from transcription.benchmark import _cache_path, _load_quality_cache, _save_quality_cache, _score_notes, evaluate_15_key
from transcription.models import QualityAnalysisDraft, SymbolicNote, TranscriptionError, TranscriptionOptions, TranscriptionResult
from transcription.pipeline import export_song_json, transcribe_draft
from transcription.quality import _role, arrange_quality_analysis, resolve_quality_model, select_melody


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


# Frozen original-V3 arranger reference. It deliberately has the historical
# accompaniment predicate; tests compare the candidate to it only on drafts
# without drums, or to the same clean draft after a drum layer is added.
_LEAD = {"voice": 8, "synth_lead": 7, "violin": 6, "flutes": 6,
         "soprano_and_alto_sax": 5, "tenor_sax": 5, "clarinet": 5,
         "oboe": 5, "trumpet": 5, "acoustic_piano": 4, "electric_piano": 4}


def _reference_quantize(time_ms, draft):
    beats = draft.beat_times_ms
    if len(beats) < 2:
        return time_ms
    index = max(0, min(len(beats) - 2, bisect.bisect_right(beats, time_ms) - 1))
    left, right = beats[index], beats[index + 1]
    choices = [int(round(left + (right - left) * unit / division)) for division in (2, 3, 4) for unit in range(division + 1)]
    candidate = min(choices, key=lambda value: (abs(value - time_ms), value))
    return candidate if abs(candidate - time_ms) <= min(70, int(round(max(1, right - left) * .18))) else time_ms


def _reference_cost(note, previous):
    value = 100.0 + 12.0 * _LEAD.get(note.instrument, 0) - abs(note.midi_pitch - 74) * .6
    if previous:
        value -= abs(note.midi_pitch - previous.midi_pitch) * .55
        if note.instrument != previous.instrument:
            value -= 7.0
        if note.start_ms - previous.end_ms > 1400:
            value -= 3.0
    return value


def _reference_select(notes, beats):
    candidates = [note for note in notes if note.role == "melody" and 45 <= note.midi_pitch <= 100]
    if not candidates:
        candidates = [note for note in notes if note.role != "drums" and 54 <= note.midi_pitch <= 96]
    grouped = defaultdict(list)
    for note in candidates:
        grouped[int(round(note.start_ms / 25.0) * 25) if len(beats) >= 2 else note.start_ms].append(note)
    windows, previous, selected = defaultdict(list), None, []
    for onset in sorted(grouped):
        beat_index = max(0, bisect.bisect_right(beats, onset) - 1) if beats else onset // 4000
        row = sorted(grouped[onset], key=lambda note: (_LEAD.get(note.instrument, 0), note.end_ms - note.start_ms), reverse=True)[:8]
        windows[beat_index // 8].append(row)
    for window in (windows[index] for index in sorted(windows)):
        scores, links = [], []
        for row, candidates_at_onset in enumerate(window):
            row_scores, row_links = [], []
            for current in candidates_at_onset:
                if row == 0:
                    row_scores.append(_reference_cost(current, previous)); row_links.append(-1)
                else:
                    choices = [scores[-1][old] + _reference_cost(current, window[row - 1][old]) for old in range(len(window[row - 1]))]
                    best = max(range(len(choices)), key=choices.__getitem__)
                    row_scores.append(choices[best]); row_links.append(best)
            scores.append(row_scores); links.append(row_links)
        if not scores:
            continue
        index, chosen = max(range(len(scores[-1])), key=scores[-1].__getitem__), []
        for row in range(len(window) - 1, -1, -1):
            chosen.append(window[row][index]); index = links[row][index]
            if index < 0:
                break
        for current in reversed(chosen):
            if previous and current.start_ms < previous.end_ms and current.midi_pitch != previous.midi_pitch:
                continue
            selected.append(current); previous = current
    return selected


def _reference_shift(melody, options):
    if not melody:
        return 0
    shifts = (key_transpose(options.source_key),) if options.source_key else range(-6, 6)
    best_shift, best_cost = 0, float("inf")
    for shift in shifts:
        previous_source = previous_sky = None; cost = 0.0
        for note in melody:
            key, adjusted = _pitch_to_key(note.midi_pitch + shift, previous_source, previous_sky)
            cost += (2.5 if adjusted else 0.0) + abs(SKY_MIDI[key] - (note.midi_pitch + shift)) * .35
            if previous_sky is not None:
                source_delta, sky_delta = note.midi_pitch - previous_source, SKY_MIDI[key] - previous_sky
                if source_delta and sky_delta and (source_delta > 0) != (sky_delta > 0):
                    cost += 4.0
            previous_source, previous_sky = note.midi_pitch + shift, SKY_MIDI[key]
        if cost < best_cost:
            best_shift, best_cost = shift, cost
    return best_shift


def _reference_arrange(draft, options):
    melody = _reference_select(draft.symbolic_notes, draft.beat_times_ms)
    shift = _reference_shift(melody, options)
    identities = {(note.start_ms, note.end_ms, note.midi_pitch, note.instrument) for note in melody}
    by_time, previous_source, previous_sky = defaultdict(dict), None, None

    def add(time_ms, pitch, priority, role):
        nonlocal previous_source, previous_sky
        index, _adjusted = _pitch_to_key(pitch + shift, previous_source, previous_sky)
        old = by_time[time_ms].get(index)
        if old is None or priority > old[0]:
            by_time[time_ms][index] = (priority, role)
        if role == "melody":
            previous_source, previous_sky = pitch + shift, SKY_MIDI[index]

    for note in melody:
        add(_reference_quantize(note.start_ms, draft), note.midi_pitch + 12 * (options.melody_octave_shift or 0), 100, "melody")
    density = {"simple": 4, "standard": 2, "full": 1, "auto": 2}[options.arrangement_preset]
    accompaniment = [note for note in draft.symbolic_notes if note.role not in ("drums", "melody") or (note.start_ms, note.end_ms, note.midi_pitch, note.instrument) not in identities]
    for ordinal, note in enumerate(accompaniment):
        if ordinal % density == 0:
            add(_reference_quantize(note.start_ms, draft), note.midi_pitch, 40 if note.role == "bass" else 20, note.role)
    output = []
    for time_ms in sorted(by_time):
        for key, _value in sorted(sorted(by_time[time_ms].items(), key=lambda item: (-item[1][0], item[0]))[:options.max_polyphony]):
            output.append({"time": int(time_ms), "key": f"1Key{key}"})
    return output, melody


class TestQualityArrangement(unittest.TestCase):
    def test_quality_requires_explicit_rights_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "input.wav")
            open(source, "wb").close()
            with self.assertRaisesRegex(TranscriptionError, "确认"):
                transcribe_draft(source, TranscriptionOptions(engine="quality"))

    def test_original_v3_reference_matches_without_drums_for_all_presets(self):
        for preset in ("simple", "standard", "full", "auto"):
            options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset=preset, max_polyphony=2)
            expected, expected_melody = _reference_arrange(quality_draft(), options)
            actual, _key, _shift, stats = arrange_quality_analysis(quality_draft(), options)
            self.assertEqual(actual, expected)
            self.assertEqual(select_melody(quality_draft().symbolic_notes, quality_draft().beat_times_ms), expected_melody)
            self.assertEqual(stats["filteredDrumCount"], 0)

    def test_only_drum_derived_output_is_removed(self):
        clean = quality_draft()
        clean.symbolic_notes = [
            SymbolicNote(0, 300, 48, "acoustic_bass", "bass"),
            SymbolicNote(500, 800, 55, "acoustic_guitar", "harmony"),
            SymbolicNote(1000, 1300, 60, "acoustic_guitar", "harmony"),
        ]
        noisy = quality_draft()
        noisy.symbolic_notes = [
            clean.symbolic_notes[0],
            SymbolicNote(250, 350, 36, "drums", "drums"),
            clean.symbolic_notes[1],
            SymbolicNote(750, 850, 38, "drums", "drums"),
            clean.symbolic_notes[2],
        ]
        for preset in ("simple", "standard", "full", "auto"):
            options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset=preset)
            expected, _melody = _reference_arrange(noisy, options)
            expected = [note for note in expected if note["time"] not in (250, 750)]
            actual, _key, _shift, stats = arrange_quality_analysis(noisy, options)
            self.assertEqual(actual, expected)
            self.assertEqual(select_melody(noisy.symbolic_notes, noisy.beat_times_ms), _melody)
            self.assertEqual(stats["filteredDrumCount"], 2)

    def test_original_path_covers_piano_short_notes_leaps_and_fallback(self):
        draft = quality_draft()
        draft.symbolic_notes = [
            SymbolicNote(0, 55, 60, "acoustic_piano", "melody"),
            SymbolicNote(500, 950, 84, "acoustic_piano", "melody"),
            SymbolicNote(1000, 1400, 56, "acoustic_piano", "melody"),
            SymbolicNote(1500, 1950, 79, "acoustic_piano", "melody"),
        ]
        expected, reference_selected = _reference_arrange(draft, TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major"))
        actual, _key, _shift, _stats = arrange_quality_analysis(draft, TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major"))
        self.assertEqual(_role("acoustic_piano"), "melody")
        self.assertEqual(select_melody(draft.symbolic_notes, draft.beat_times_ms), reference_selected)
        self.assertEqual(actual, expected)

    def test_melody_survives_polyphony_limit_and_quantizes_small_error(self):
        notes, _key, _shift, stats = arrange_quality_analysis(quality_draft(), TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", max_polyphony=2, arrangement_preset="full"))
        by_time = {}
        for note in notes:
            by_time.setdefault(note["time"], []).append(note["key"])
        self.assertIn("1Key7", by_time[0])
        self.assertTrue(all(len(keys) <= 2 for keys in by_time.values()))
        self.assertEqual(stats["melody_note_count"], 3)

    def test_quality_arrangement_allows_ten_notes_at_one_onset(self):
        draft = quality_draft()
        draft.symbolic_notes = [SymbolicNote(0, 400, pitch, "acoustic_guitar", "harmony") for pitch in SKY_MIDI[:10]]
        notes, _key, _shift, _stats = arrange_quality_analysis(
            draft, TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="full", max_polyphony=10)
        )
        self.assertEqual(len([note for note in notes if note["time"] == 0]), 10)

    def test_device_model_defaults(self):
        self.assertEqual(resolve_quality_model("auto", "cpu"), "small")
        self.assertEqual(resolve_quality_model("auto", "mps"), "medium")
        self.assertEqual(resolve_quality_model("small", "cuda"), "small")

    def test_v3_export_metadata_keeps_song_schema(self):
        draft = quality_draft()
        options = TranscriptionOptions(engine="quality", rights_confirmed=True)
        notes, key, shift, stats = arrange_quality_analysis(draft, options)
        result = TranscriptionResult([], notes, key, 120, stats, [], engine="arrangement_v3_quality", options=options, quality_analysis=draft, semitone_shift=shift)
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "out.json")
            export_song_json(result, output, "demo")
            with open(output, encoding="utf-8") as handle:
                data = json.load(handle)[0]
        metadata = data["_transcribe"]
        self.assertEqual(metadata["schemaVersion"], 3)
        self.assertEqual(metadata["qualityArrangerVersion"], 4)
        self.assertNotIn("onsetDelayMs", metadata)
        self.assertTrue(data["songNotes"])


class TestQualityMetrics(unittest.TestCase):
    def test_score_metrics_report_top_voice_selected_melody_and_compatibility_alias(self):
        reference = [{"time": 0, "key": "1Key7"}, {"time": 500, "key": "1Key8"}]
        estimate = [{"time": 30, "key": "1Key7"}, {"time": 500, "key": "1Key6"}]
        selected = [{"time": 30, "key": "1Key7"}, {"time": 500, "key": "1Key8"}]
        metrics = evaluate_15_key(reference, estimate, selected_melody=selected)
        self.assertEqual(metrics["onset"]["f1"], 1.0)
        self.assertEqual(metrics["key_onset"]["matched"], 1)
        self.assertEqual(metrics["key_onset"]["onset_mae_ms"], 30)
        self.assertEqual(metrics["melody"], metrics["top_voice"])
        self.assertEqual(metrics["selected_melody"]["f1"], 1.0)

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

    def test_quality_cache_restores_symbolic_draft_and_rearranges(self):
        options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="simple")
        notes, key, shift, stats = arrange_quality_analysis(quality_draft(), options)
        result = TranscriptionResult([], notes, key, 120, stats, [], engine="arrangement_v3_quality", options=options, quality_analysis=quality_draft(), semitone_shift=shift)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "demo.wav"; audio.write_bytes(b"audio-content")
            path = _cache_path(root, "demo", options, audio)
            audio.write_bytes(b"changed-audio-content")
            self.assertNotEqual(path, _cache_path(root, "demo", options, audio))
            audio.write_bytes(b"audio-content")
            _save_quality_cache(path, result)
            restored = _load_quality_cache(path, TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="full"))
        self.assertIsNotNone(restored)
        self.assertEqual(restored.quality_analysis.model_name, "small")
        self.assertEqual(restored.options.arrangement_preset, "full")

    def test_old_quality_cache_is_rejected_before_it_can_restore_legacy_timing(self):
        options = TranscriptionOptions(engine="quality", rights_confirmed=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.json"
            path.write_text(json.dumps({"cacheVersion": 2, "draft": {"onset_delay_ms": 42}}), encoding="utf-8")
            self.assertIsNone(_load_quality_cache(path, options))


if __name__ == "__main__":
    unittest.main()
