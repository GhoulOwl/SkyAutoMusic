import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from player import PlaybackState
from transcription.models import TranscriptionResult
from transcription.preview import PreviewPlayer


class _Controller:
    prepared = True

    def stop_all(self):
        pass

    def close(self):
        pass


class _Player:
    def __init__(self):
        self.state = PlaybackState.STOPPED
        self.started = None
        self.seek_index = None
        self.speed = 1.0
        self.simulate = False

    def stop(self):
        self.state = PlaybackState.STOPPED

    def start(self, notes_by_time, sorted_times):
        self.started = (notes_by_time, sorted_times)
        self.state = PlaybackState.PLAYING
        return True

    def seek(self, index):
        self.seek_index = index


class TestPreviewRange(unittest.TestCase):
    def _preview(self):
        preview = PreviewPlayer(controller=_Controller())
        preview.player = _Player()
        return preview

    @staticmethod
    def _result():
        return TranscriptionResult(
            events=[], song_notes=[
                {"time": 100, "key": "1Key1"},
                {"time": 500, "key": "1Key2"},
                {"time": 900, "key": "1Key3"},
            ], detected_key="C major", bpm=120, stats={}, warnings=[],
        )

    def test_preview_filters_to_the_requested_region(self):
        preview = self._preview()
        preview.play(self._result(), start_ms=300, end_ms=700)
        notes_by_time, times = preview.player.started
        self.assertEqual(times, [500])
        self.assertEqual(notes_by_time[500], ["1Key2"])

    def test_seek_uses_the_first_note_at_or_after_playhead(self):
        preview = self._preview()
        preview.play(self._result())
        self.assertEqual(preview.seek_ms(600), 900)
        self.assertEqual(preview.player.seek_index, 2)


if __name__ == "__main__":
    unittest.main()
