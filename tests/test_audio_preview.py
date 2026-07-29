import io
import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_preview import (  # noqa: E402
    AudioPreviewController,
    MciSamplePlayer,
    PianoSampleLibrary,
    SAMPLE_COUNT,
    note_to_sample_index,
)


FAKE_MP3 = b"ID3" + (b"\0" * 256)


class FakeResponse(io.BytesIO):
    pass


class FakeBackend:
    def __init__(self, paths):
        self.paths = paths
        self.played = []
        self.stopped = 0
        self.closed = 0

    def play_indices(self, indices):
        self.played.append(tuple(indices))

    def stop_all(self):
        self.stopped += 1

    def close(self):
        self.closed += 1


class TestNoteMapping(unittest.TestCase):
    def test_both_keyboards_share_the_same_samples(self):
        for index in range(SAMPLE_COUNT):
            self.assertEqual(note_to_sample_index(f"1Key{index}"), index)
            self.assertEqual(note_to_sample_index(f"2Key{index}"), index)

    def test_unknown_notes_are_ignored(self):
        for note in ("3Key0", "1Key15", "1Key-1", "bad", None):
            self.assertIsNone(note_to_sample_index(note))


class TestSampleLibrary(unittest.TestCase):
    def test_uses_cached_samples_without_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index in range(SAMPLE_COUNT):
                with open(os.path.join(temp_dir, f"{index}.mp3"), "wb") as stream:
                    stream.write(FAKE_MP3)

            def fail_opener(*args, **kwargs):
                self.fail("network should not be used for cached samples")

            library = PianoSampleLibrary(
                cache_dir=temp_dir,
                search_dirs=[temp_dir],
                opener=fail_opener,
            )
            paths = library.ensure_samples()

        self.assertEqual(len(paths), SAMPLE_COUNT)

    def test_downloads_missing_samples_and_reports_progress(self):
        requested = []

        def opener(request, timeout):
            requested.append((request.full_url, timeout))
            return FakeResponse(FAKE_MP3)

        with tempfile.TemporaryDirectory() as temp_dir:
            progress = []
            library = PianoSampleLibrary(
                cache_dir=temp_dir,
                search_dirs=[],
                base_url="https://example.test/piano",
                opener=opener,
            )
            paths = library.ensure_samples(
                lambda completed, total, index: progress.append(
                    (completed, total, index)
                )
            )
            self.assertTrue(all(os.path.exists(path) for path in paths))

        self.assertEqual(len(requested), SAMPLE_COUNT)
        self.assertEqual(progress[-1], (SAMPLE_COUNT, SAMPLE_COUNT, SAMPLE_COUNT - 1))

    def test_invalid_download_is_removed(self):
        def opener(request, timeout):
            return FakeResponse(b"not an mp3")

        with tempfile.TemporaryDirectory() as temp_dir:
            library = PianoSampleLibrary(
                cache_dir=temp_dir,
                search_dirs=[],
                opener=opener,
            )
            with self.assertRaises(Exception):
                library.ensure_samples()
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "0.mp3.part")))


class TestAudioPreviewController(unittest.TestCase):
    def _prepared_controller(self):
        paths = [f"{index}.mp3" for index in range(SAMPLE_COUNT)]

        class FakeLibrary:
            def ensure_samples(self, progress_callback=None):
                return paths

        created = []

        def factory(sample_paths):
            backend = FakeBackend(sample_paths)
            created.append(backend)
            return backend

        controller = AudioPreviewController(
            library=FakeLibrary(),
            backend_factory=factory,
        )
        controller.prepare()
        return controller, created[0]

    def test_dispatches_chords_and_dedupes_keyboard_layers(self):
        controller, backend = self._prepared_controller()
        controller.press_keys(["1Key0", "2Key0", "1Key4", "bad"])
        self.assertEqual(backend.played, [(0, 4)])

    def test_repeated_notes_are_retriggered(self):
        controller, backend = self._prepared_controller()
        controller.press_keys(["1Key3"])
        controller.press_keys(["1Key3"])
        self.assertEqual(backend.played, [(3,), (3,)])

    def test_stop_and_close_delegate_to_backend(self):
        controller, backend = self._prepared_controller()
        controller.stop_all()
        controller.close()
        self.assertEqual(backend.stopped, 1)
        self.assertEqual(backend.closed, 1)
        self.assertFalse(controller.prepared)


class TestMciSamplePlayer(unittest.TestCase):
    def test_chord_commands_use_independent_aliases(self):
        commands = []
        paths = [f"C:/samples/{index}.mp3" for index in range(SAMPLE_COUNT)]
        player = MciSamplePlayer(paths, command_sender=commands.append)
        self.addCleanup(player.close)
        commands.clear()

        player.play_indices([0, 4])

        play_commands = [command for command in commands if command.startswith("play ")]
        self.assertEqual(len(play_commands), 2)
        self.assertNotEqual(play_commands[0], play_commands[1])


if __name__ == "__main__":
    unittest.main()
