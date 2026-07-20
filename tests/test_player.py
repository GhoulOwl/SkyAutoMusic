import os
import sys
import unittest
from collections import defaultdict


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from player import MusicPlayer, PlaybackState  # noqa: E402


class FakeKeyController:
    def __init__(self):
        self.events = []

    def press_keys(self, notes):
        self.events.append(("press", tuple(notes)))

    def release_keys(self, notes):
        self.events.append(("release", tuple(notes)))


class TestMusicPlayer(unittest.TestCase):
    def test_plays_notes_in_timestamp_order(self):
        keys = FakeKeyController()
        notes_by_time = defaultdict(list, {0: ["1Key0"], 1: ["1Key1", "1Key2"]})
        seen_notes = []
        statuses = []
        player = MusicPlayer(
            keys,
            update_status=statuses.append,
            update_note=lambda idx, t_ms, notes: seen_notes.append((idx, t_ms, tuple(notes))),
        )
        player.START_DELAY = 0
        player.NOTE_HOLD = 0

        self.assertTrue(player.start(notes_by_time, [0, 1]))
        player.thread.join(timeout=1)

        self.assertEqual(player.state, PlaybackState.STOPPED)
        self.assertEqual(keys.events[:4], [
            ("press", ("1Key0",)),
            ("release", ("1Key0",)),
            ("press", ("1Key1", "1Key2")),
            ("release", ("1Key1", "1Key2")),
        ])
        self.assertIn((0, 0, ("1Key0",)), seen_notes)
        self.assertIn((1, 1, ("1Key1", "1Key2")), seen_notes)
        self.assertTrue(any("演奏结束" in s for s in statuses))

    def test_finished_callback_fires_on_natural_end(self):
        keys = FakeKeyController()
        finished = []
        notes_by_time = defaultdict(list, {0: ["1Key0"]})
        player = MusicPlayer(
            keys,
            update_finished=lambda: finished.append(True),
        )
        player.START_DELAY = 0
        player.NOTE_HOLD = 0

        self.assertTrue(player.start(notes_by_time, [0]))
        player.thread.join(timeout=1)

        self.assertEqual(player.state, PlaybackState.STOPPED)
        self.assertEqual(finished, [True])

    def test_finished_callback_not_fired_on_user_stop(self):
        keys = FakeKeyController()
        finished = []
        notes_by_time = defaultdict(list, {0: ["1Key0"], 1000: ["1Key1"]})
        player = MusicPlayer(
            keys,
            update_finished=lambda: finished.append(True),
        )
        player.START_DELAY = 0
        player.NOTE_HOLD = 0

        self.assertTrue(player.start(notes_by_time, [0, 1000]))
        player.stop()
        player.thread.join(timeout=1)

        self.assertEqual(player.state, PlaybackState.STOPPED)
        self.assertEqual(finished, [])


if __name__ == "__main__":
    unittest.main()
