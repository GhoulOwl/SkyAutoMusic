import bisect
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.arranger import SKY_MIDI, _pitch_to_key, key_transpose
from transcription.benchmark import _cache_path, _load_quality_cache, _save_quality_cache, _score_notes, evaluate_15_key
from transcription.models import QualityAnalysisDraft, SymbolicNote, TranscriptionError, TranscriptionOptions, TranscriptionResult
from transcription.pipeline import export_song_json, transcribe_draft
from transcription.quality import _notify, _role, _timing_from_model, arrange_quality_analysis, arrange_quality_analysis_with_roles, arrange_quality_melody, choose_quality_device, resolve_quality_model, select_melody, select_melody_with_segments


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


def reliable_budget_draft():
    """Eight 4/4 bars with enough material for every V7 budget."""
    notes = []
    for bar in range(8):
        start = bar * 2000
        for beat in range(4):
            time = start + beat * 500
            notes.append(SymbolicNote(time, time + 360, 76 + (beat % 2) * 2, "voice", "melody"))
        notes.append(SymbolicNote(start, start + 700, 48, "acoustic_bass", "bass"))
        notes.append(SymbolicNote(start + 1000, start + 1700, 64, "acoustic_guitar", "harmony"))
    # These deliberately have unique times/keys, so a leaked source would be visible.
    notes.extend([
        SymbolicNote(250, 850, 84, "acoustic_piano", "melody"),
        SymbolicNote(750, 760, 36, "drums", "drums"),
        SymbolicNote(1250, 1850, 67, "percussion_fx", "other"),
    ])
    beats = list(range(0, 16001, 500))
    return QualityAnalysisDraft(
        duration_sec=16.0, symbolic_notes=notes, beat_times_ms=beats,
        downbeat_times_ms=list(range(0, 16001, 2000)), bar_starts_ms=list(range(0, 16001, 2000)),
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

    def test_v7_presets_lock_melody_and_increase_density_when_safe_candidates_exist(self):
        draft = reliable_budget_draft()
        counts = {}
        for preset in ("simple", "standard", "auto", "full"):
            options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset=preset, max_polyphony=3)
            notes, roles, _key, shift, stats = arrange_quality_analysis_with_roles(draft, options)
            mapped = {(int(note["time"]), str(note["key"])) for note in arrange_quality_melody(draft, options, shift)}
            rendered = {(int(note["time"]), str(note["key"])) for note in notes}
            self.assertTrue(mapped <= rendered)
            self.assertTrue(all(role in ("melody", "bass", "harmony") for role in roles.values()))
            self.assertLessEqual(stats["accompaniment_note_count"], stats["melody_note_count"])
            if preset == "simple":
                self.assertEqual(stats["accompaniment_note_count"], 0)
            if preset == "auto":
                self.assertGreaterEqual(stats["melody_note_ratio"], .75)
            counts[preset] = stats["accompaniment_note_count"]
        self.assertLess(counts["simple"], counts["standard"])
        self.assertLess(counts["standard"], counts["auto"])
        self.assertLess(counts["auto"], counts["full"])

    def test_source_texture_can_support_vocal_while_drums_and_other_never_render(self):
        draft = reliable_budget_draft()
        options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="full")
        _notes, roles, _key, _shift, stats = arrange_quality_analysis_with_roles(draft, options)
        self.assertGreater(stats["reused_source_texture_count"], 0)
        self.assertGreater(stats["filteredDrumCount"], 0)
        self.assertGreater(stats["filtered_other_count"], 0)
        self.assertNotIn("other", roles.values())
        self.assertNotIn("drums", roles.values())

    def test_locked_melody_survives_every_valid_polyphony(self):
        draft = reliable_budget_draft()
        for polyphony in range(2, 11):
            options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="full", max_polyphony=polyphony)
            notes, _roles, _key, shift, _stats = arrange_quality_analysis_with_roles(draft, options)
            expected = {(int(note["time"]), str(note["key"])) for note in arrange_quality_melody(draft, options, shift)}
            actual = {(int(note["time"]), str(note["key"])) for note in notes}
            self.assertTrue(expected <= actual, polyphony)

    def test_original_path_covers_piano_short_notes_leaps_and_fallback(self):
        draft = quality_draft()
        draft.symbolic_notes = [
            SymbolicNote(0, 55, 60, "acoustic_piano", "melody"),
            SymbolicNote(500, 950, 84, "acoustic_piano", "melody"),
            SymbolicNote(1000, 1400, 56, "acoustic_piano", "melody"),
            SymbolicNote(1500, 1950, 79, "acoustic_piano", "melody"),
        ]
        options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="simple")
        expected, reference_selected = _reference_arrange(draft, options)
        actual, _key, _shift, _stats = arrange_quality_analysis(draft, options)
        self.assertEqual(_role("acoustic_piano"), "melody")
        self.assertEqual(select_melody(draft.symbolic_notes, draft.beat_times_ms), reference_selected)
        self.assertEqual(actual, expected)

    def test_overlapping_voice_syllables_are_retained(self):
        draft = quality_draft()
        first = SymbolicNote(0, 450, 72, "voice", "melody")
        second = SymbolicNote(250, 700, 74, "voice", "melody")
        draft.symbolic_notes = [first, second]

        self.assertEqual(select_melody(draft.symbolic_notes, draft.beat_times_ms), [first, second])

    def test_voice_wins_same_onset_while_piano_remains_available_for_accompaniment(self):
        draft = quality_draft()
        first_voice = SymbolicNote(0, 400, 72, "voice", "melody")
        second_voice = SymbolicNote(500, 900, 74, "voice", "melody")
        draft.symbolic_notes = [
            SymbolicNote(0, 900, 84, "acoustic_piano", "melody"), first_voice,
            SymbolicNote(500, 1300, 86, "acoustic_piano", "melody"), second_voice,
        ]

        self.assertEqual(select_melody(draft.symbolic_notes, draft.beat_times_ms), [first_voice, second_voice])

    def test_voice_collision_uses_original_onset_but_piano_keeps_quantized_timing(self):
        draft = quality_draft()
        options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major")
        first_voice = SymbolicNote(0, 100, 72, "voice", "melody")
        second_voice = SymbolicNote(60, 160, 72, "voice", "melody")
        draft.symbolic_notes = [first_voice, second_voice]
        vocal_path = arrange_quality_melody(draft, options)
        self.assertEqual([note["time"] for note in vocal_path], [0, 60])
        vocal_notes, _key, _shift, _stats = arrange_quality_analysis(draft, options)
        self.assertEqual([(note["time"], note["key"]) for note in vocal_notes], [(0, vocal_path[0]["key"]), (60, vocal_path[1]["key"])])

        piano = quality_draft()
        piano.symbolic_notes = [
            SymbolicNote(0, 100, 72, "acoustic_piano", "melody"),
            SymbolicNote(60, 160, 72, "acoustic_piano", "melody"),
        ]
        self.assertEqual([note["time"] for note in arrange_quality_melody(piano, options)], [0, 0])

    def test_short_vocal_breath_remains_silent_before_instrumental_takeover(self):
        draft = quality_draft()
        voice = SymbolicNote(0, 300, 72, "voice", "melody")
        piano = SymbolicNote(500, 900, 76, "acoustic_piano", "melody")
        draft.symbolic_notes = [voice, piano]

        selected, segments = select_melody_with_segments(draft.symbolic_notes, draft.beat_times_ms)
        self.assertEqual(selected, [voice])
        self.assertEqual([segment.source for segment in segments], ["vocal"])

        later = SymbolicNote(1600, 2000, 76, "acoustic_piano", "melody")
        self.assertEqual(select_melody([voice, later], draft.beat_times_ms), [voice, later])

    def test_direct_beat_tracker_keeps_variable_grid_and_scales_override(self):
        tracker = lambda _path: ([.0, .5, 1.03, 1.50, 2.04], [.0, 2.04])
        options = TranscriptionOptions(engine="quality", rights_confirmed=True, bpm_override=100, meter="4/4")
        with patch("transcription.quality._beat_tracker", return_value=tracker):
            beats, downbeats, bars, bpm, meter, confidence, backend, diagnostics = _timing_from_model(object(), "song.wav", [], options)
        self.assertEqual(backend, "beat_this_direct")
        self.assertEqual(meter, "4/4")
        self.assertEqual(bpm, 100)
        self.assertGreater(confidence, .5)
        self.assertEqual(downbeats, bars)
        self.assertEqual(diagnostics["rawBeatCount"], 5)
        self.assertNotEqual(beats[2] - beats[1], beats[3] - beats[2])

    def test_fallback_grid_does_not_quantize_note_attacks(self):
        draft = quality_draft()
        draft.timing_backend = "fixed_grid_fallback"
        draft.timing_confidence = .13
        draft.symbolic_notes = [SymbolicNote(61, 400, 72, "voice", "melody")]
        notes = arrange_quality_melody(draft, TranscriptionOptions(engine="quality", rights_confirmed=True))
        self.assertEqual(notes[0]["time"], 61)

    def test_melody_survives_polyphony_limit_and_quantizes_small_error(self):
        notes, _key, _shift, stats = arrange_quality_analysis(quality_draft(), TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", max_polyphony=2, arrangement_preset="full"))
        by_time = {}
        for note in notes:
            by_time.setdefault(note["time"], []).append(note["key"])
        self.assertIn("1Key7", by_time[0])
        self.assertTrue(all(len(keys) <= 2 for keys in by_time.values()))
        self.assertEqual(stats["melody_note_count"], 3)

    def test_dense_thousands_are_filtered_before_bar_and_global_budgets(self):
        draft = reliable_budget_draft()
        for index in range(3000):
            time = 100 + (index % 20) * 10
            draft.symbolic_notes.append(SymbolicNote(time, time + 300, 60 + (index % 12), "acoustic_guitar", "harmony"))
        options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="full", max_polyphony=3)
        notes, _key, _shift, stats = arrange_quality_analysis(draft, options)
        self.assertGreater(
            stats["filtered_dense_accompaniment_count"] + stats["filtered_duplicate_accompaniment_count"],
            1000,
        )
        self.assertLessEqual(stats["accompaniment_note_count"], int(stats["melody_note_count"] * .50))
        self.assertLessEqual(len(notes), stats["melody_note_count"] + int(stats["melody_note_count"] * .50))

    def test_low_timing_protection_attaches_v7_support_to_melody_onsets(self):
        draft = reliable_budget_draft()
        draft.timing_confidence = .13
        draft.timing_backend = "fixed_120_fallback"
        counts = {}
        for preset in ("simple", "standard", "auto", "full"):
            options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset=preset, max_polyphony=3)
            notes, roles, _key, shift, stats = arrange_quality_analysis_with_roles(draft, options)
            self.assertTrue(stats["low_timing_protection"])
            melody_times = {int(note["time"]) for note in notes if roles[f"{int(note['time'])}:{note['key']}"] == "melody"}
            accompaniment_times = {int(note["time"]) for note in notes if roles[f"{int(note['time'])}:{note['key']}"] != "melody"}
            self.assertTrue(accompaniment_times <= melody_times)
            self.assertTrue(all(role in ("melody", "bass", "harmony") for role in roles.values()))
            counts[preset] = stats["accompaniment_note_count"]
            if preset == "simple":
                self.assertEqual(stats["accompaniment_note_count"], 0)
            if preset == "full":
                self.assertGreater(stats["generated_accompaniment_count"], 0)
        self.assertLess(counts["simple"], counts["standard"])
        self.assertLess(counts["standard"], counts["auto"])
        self.assertLess(counts["auto"], counts["full"])

    def test_v7_infers_safe_chords_from_melody_without_source_accompaniment(self):
        draft = quality_draft()
        draft.timing_confidence = .13
        draft.timing_backend = "fixed_120_fallback"
        draft.symbolic_notes = [
            SymbolicNote(time, time + 340, pitch, "voice", "melody")
            for time, pitch in ((0, 72), (500, 76), (1000, 79), (1500, 76), (2000, 72), (2500, 74))
        ]
        options = TranscriptionOptions(engine="quality", rights_confirmed=True, source_key="C major", arrangement_preset="full", max_polyphony=3)
        notes, roles, _key, _shift, stats = arrange_quality_analysis_with_roles(draft, options)
        melody_times = {int(note["time"]) for note in notes if roles[f"{int(note['time'])}:{note['key']}"] == "melody"}
        for note in notes:
            role = roles[f"{int(note['time'])}:{note['key']}"]
            if role == "melody":
                continue
            self.assertIn(int(note["time"]), melody_times)
            lead_keys = [int(item["key"].replace("1Key", "")) for item in notes if int(item["time"]) == int(note["time"]) and roles[f"{int(item['time'])}:{item['key']}"] == "melody"]
            self.assertTrue(any(2 <= lead - int(note["key"].replace("1Key", "")) <= 10 for lead in lead_keys))
        self.assertEqual(stats["harmonic_key"], "C major")
        self.assertEqual(stats["source_accompaniment_count"], 0)
        self.assertGreater(stats["generated_accompaniment_count"], 0)
        self.assertLessEqual(stats["accompaniment_note_count"], stats["melody_note_count"] // 2)

    def test_device_model_defaults(self):
        self.assertEqual(resolve_quality_model("auto", "cpu"), "small")
        self.assertEqual(resolve_quality_model("auto", "mps"), "medium")
        self.assertEqual(resolve_quality_model("auto", "cuda"), "medium")
        self.assertEqual(resolve_quality_model("small", "cuda"), "small")
        self.assertEqual(TranscriptionOptions(quality_model="large").quality_model, "large")
        self.assertEqual(TranscriptionOptions(quality_device="cpu").quality_device, "cpu")

    def test_cpu_preference_never_selects_accelerator(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(choose_quality_device("cpu"), "cpu")

    def test_quality_model_download_progress_callback_is_safe_and_bounded(self):
        progress = []
        _notify(lambda stage, fraction, message: progress.append((stage, fraction, message)), "quality", 1.5, "正在下载模型")
        _notify(None, "quality", .02, "ignored")
        self.assertEqual(progress, [("quality", 1.0, "正在下载模型")])

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
        self.assertEqual(metadata["qualityArrangerVersion"], 8)
        self.assertEqual(stats["qualityArrangerVersion"], 8)
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
