import json
import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_loader import ScoreValidationError, load_score  # noqa: E402


def _write_score(payload):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


class TestScoreLoader(unittest.TestCase):
    def test_loads_valid_score_and_dedupes_duplicate_notes(self):
        path = _write_score([
            {
                "name": "demo",
                "bpm": 90,
                "songNotes": [
                    {"time": 100, "key": "1Key0"},
                    {"time": 100, "key": "1Key0"},
                    {"time": 150.4, "key": "1Key1"},
                ],
            }
        ])
        try:
            score = load_score(path, valid_keys={"1Key0", "1Key1"})
        finally:
            os.remove(path)

        self.assertEqual(score["meta"]["name"], "demo")
        self.assertEqual(score["sorted_times"], [100, 150])
        self.assertEqual(score["notes_by_time"][100], ["1Key0"])
        self.assertEqual(score["notes_by_time"][150], ["1Key1"])
        self.assertEqual(score["duration_ms"], 50)
        self.assertEqual(score["note_count"], 2)
        self.assertEqual(len(score["warnings"]), 1)

    def test_metrics_use_playback_span_and_deduplicated_notes(self):
        path = _write_score([{
            "songNotes": [
                {"time": 1000, "key": "1Key0"},
                {"time": 1000, "key": "1Key1"},
                {"time": 1000, "key": "1Key1"},
                {"time": 61000, "key": "1Key2"},
            ],
        }])
        try:
            score = load_score(path, valid_keys={"1Key0", "1Key1", "1Key2"})
        finally:
            os.remove(path)

        self.assertEqual(score["duration_ms"], 60000)
        self.assertEqual(score["note_count"], 3)

    def test_single_event_has_zero_duration(self):
        path = _write_score([{
            "songNotes": [{"time": 500, "key": "1Key0"}],
        }])
        try:
            score = load_score(path, valid_keys={"1Key0"})
        finally:
            os.remove(path)

        self.assertEqual(score["duration_ms"], 0)
        self.assertEqual(score["note_count"], 1)

    def test_rejects_unknown_key(self):
        path = _write_score([{"songNotes": [{"time": 0, "key": "bad"}]}])
        try:
            with self.assertRaises(ScoreValidationError) as ctx:
                load_score(path, valid_keys={"1Key0"})
        finally:
            os.remove(path)

        self.assertIn("未在按键映射中定义", str(ctx.exception))

    def test_rejects_invalid_time(self):
        path = _write_score([{"songNotes": [{"time": "soon", "key": "1Key0"}]}])
        try:
            with self.assertRaises(ScoreValidationError) as ctx:
                load_score(path, valid_keys={"1Key0"})
        finally:
            os.remove(path)

        self.assertIn("time 不是数字", str(ctx.exception))

    def test_rejects_missing_song_notes(self):
        path = _write_score([{"name": "empty"}])
        try:
            with self.assertRaises(ScoreValidationError):
                load_score(path, valid_keys={"1Key0"})
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
