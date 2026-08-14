import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.draft_store import DraftStore
from transcription.models import (
    AnalysisDraft, ChordSpan, MelodyNote, NoteEvent, QualityAnalysisDraft,
    Section, SourceMetadata, SymbolicNote, TempoMap, TranscriptionOptions,
    TranscriptionResult,
)


def _v2_result():
    analysis = AnalysisDraft(
        duration_sec=2.0,
        tempo_map=TempoMap(120.0, "4/4", (0, 500, 1000), (0,), (0, 125, 250), .9),
        detected_key="C major", key_confidence=.8, semitone_shift=0,
        melody=[MelodyNote(0, 300, 72, .8, "vocal")],
        chords=[ChordSpan(0, 1000, 0, "maj", 0, .7)],
        sections=[Section(0, 1000, "verse", 1)], lead_source="vocal",
        melody_confidence=.8, harmony_confidence=.7, structure_confidence=.6,
        vocal_path="transient.wav", accompaniment_path="transient-accompaniment.wav",
    )
    return TranscriptionResult(
        events=[NoteEvent(0, 300, 72, 1.0, "test")],
        song_notes=[{"time": 0, "key": "1Key7"}], detected_key="C major", bpm=120,
        stats={"onset_count": 1}, warnings=[], source_file="local.wav",
        options=TranscriptionOptions(arrangement_preset="simple"), analysis=analysis,
        artifact_root="transient-root",
    )


def _quality_result():
    quality = QualityAnalysisDraft(
        duration_sec=2.0, symbolic_notes=[SymbolicNote(0, 250, 72, "voice", "melody")],
        beat_times_ms=[0, 500, 1000], downbeat_times_ms=[0], bar_starts_ms=[0], bpm=120,
        meter="4/4", timing_confidence=.9, model_name="small", device="cpu",
        refined_regions=[(0, 1000)],
    )
    return TranscriptionResult(
        events=[NoteEvent(0, 250, 72, 1.0, "quality:voice")],
        song_notes=[{"time": 0, "key": "1Key7"}], detected_key="C major", bpm=120,
        stats={"qualityArrangerVersion": 5}, warnings=[], engine="arrangement_v3_quality",
        source_file="online.mp3", options=TranscriptionOptions(engine="quality", rights_confirmed=True),
        source=SourceMetadata(platform="netease", title="Online", artists=("Artist",), source_id="42"),
        quality_analysis=quality,
    )


class TestDraftStore(unittest.TestCase):
    def test_v2_round_trip_keeps_analysis_but_excludes_transient_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "local.wav")
            open(source, "wb").close()
            store = DraftStore(os.path.join(folder, "Drafts"))
            record = DraftStore.new_record("本地草稿", source, _v2_result())
            store.upsert(record)
            records, warnings = store.load_all()
        self.assertFalse(warnings)
        restored = records[0]
        self.assertEqual(restored.name, "本地草稿")
        self.assertEqual(restored.source_path, source)
        self.assertEqual(restored.result.analysis.tempo_map.beat_times_ms, (0, 500, 1000))
        self.assertEqual(restored.result.analysis.vocal_path, "")
        self.assertEqual(restored.result.artifact_root, "")

    def test_online_source_is_managed_and_delete_does_not_touch_original_download(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "download.mp3")
            with open(source, "wb") as stream:
                stream.write(b"audio")
            store = DraftStore(os.path.join(folder, "Drafts"))
            record = DraftStore.new_record("在线草稿", source, _quality_result())
            store.upsert(record, managed_source=source)
            managed_path = record.source_path
            with open(os.path.join(folder, "Drafts", record.id, "draft.json"), encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["source"]["kind"], "managed")
            self.assertNotIn("path", payload["source"])
            records, warnings = store.load_all()
            self.assertFalse(warnings)
            self.assertTrue(records[0].source_available)
            self.assertEqual(records[0].result.quality_analysis.refined_regions, [(0, 1000)])
            store.delete(records[0])
            self.assertFalse(os.path.exists(managed_path))
            self.assertTrue(os.path.exists(source))

    def test_rename_and_corrupt_entries_are_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "local.mid")
            open(source, "wb").close()
            root = os.path.join(folder, "Drafts")
            store = DraftStore(root)
            record = DraftStore.new_record("旧名称", source, _v2_result())
            store.upsert(record)
            store.rename(record, "新名称")
            os.makedirs(os.path.join(root, "broken"))
            with open(os.path.join(root, "broken", "draft.json"), "w", encoding="utf-8") as stream:
                stream.write("not json")
            records, warnings = store.load_all()
        self.assertEqual([item.name for item in records], ["新名称"])
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
