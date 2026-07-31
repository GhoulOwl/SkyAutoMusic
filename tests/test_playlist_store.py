import json
import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playlist_store import PlaybackSession, PlaylistStore  # noqa: E402


class TestPlaylistStore(unittest.TestCase):
    def test_missing_playlist_starts_empty_and_first_add_creates_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "playlist.json")
            store = PlaylistStore(path)

            self.assertEqual(store.load({"a.json"}), [])
            self.assertFalse(os.path.exists(path))
            self.assertTrue(store.add("a.json"))
            self.assertTrue(os.path.exists(path))

            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), ["a.json"])

    def test_load_cleans_duplicates_invalid_values_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "playlist.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    ["a.json", "a.json", "../bad.json", 123, "missing.json", "b.json"],
                    f,
                )

            store = PlaylistStore(path)
            self.assertEqual(store.load({"a.json", "b.json"}), ["a.json", "b.json"])

            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), ["a.json", "b.json"])

    def test_corrupt_playlist_is_repaired_to_empty_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "playlist.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not-json")

            store = PlaylistStore(path)
            self.assertEqual(store.load({"a.json"}), [])
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])

    def test_add_remove_move_and_refresh_preserve_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PlaylistStore(os.path.join(directory, "playlist.json"))
            store.load()

            self.assertTrue(store.add("a.json"))
            self.assertTrue(store.add("b.json"))
            self.assertTrue(store.add("c.json"))
            self.assertFalse(store.add("b.json"))
            self.assertTrue(store.move("c.json", -1))
            self.assertEqual(store.items, ["a.json", "c.json", "b.json"])
            self.assertFalse(store.move("a.json", -1))
            self.assertTrue(store.remove("c.json"))
            self.assertFalse(store.remove("missing.json"))
            self.assertTrue(store.refresh({"b.json"}))
            self.assertEqual(store.items, ["b.json"])


class TestPlaybackSession(unittest.TestCase):
    def test_session_uses_stable_order_and_reports_candidates(self):
        source = ["a.json", "b.json", "c.json"]
        session = PlaybackSession.create(
            source,
            "b.json",
            mode="preview",
            auto_advance=True,
            source_tab="播放列表",
        )
        source.reverse()

        self.assertEqual(session.items, ("a.json", "b.json", "c.json"))
        self.assertEqual(list(session.candidate_indexes(1)), [2])
        self.assertEqual(list(session.candidate_indexes(-1)), [0])
        self.assertTrue(session.can_move(1))
        self.assertTrue(session.can_move(-1))


if __name__ == "__main__":
    unittest.main()
