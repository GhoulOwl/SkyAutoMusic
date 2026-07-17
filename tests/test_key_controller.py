import os
import sys
import types
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_module(name, **attrs):
    if name not in sys.modules:
        sys.modules[name] = types.SimpleNamespace(**attrs)


_stub_module("pyautogui", keyDown=lambda key: None, keyUp=lambda key: None)
_stub_module(
    "keyboard",
    press=lambda key: None,
    release=lambda key: None,
    add_hotkey=lambda *args, **kwargs: None,
    unhook_all_hotkeys=lambda: None,
)
_stub_module("win32gui")
_stub_module("win32process")
_stub_module("win32con")
_stub_module("psutil")

from play_music_gui import (  # noqa: E402
    KeyController,
    is_sky_game_window_identity,
    note_to_key,
)


class TestKeyController(unittest.TestCase):
    def test_letter_keys_are_normalized_before_debug_output(self):
        logs = []
        controller = KeyController(
            {"1Key0": "Y", "1Key1": "U", "1Key9": ";"},
            debug=True,
            log_func=logs.append,
        )

        controller.press("1Key0")
        controller.release("1Key1")
        controller.press("1Key9")

        self.assertEqual(logs[0], "[DEBUG] press  1Key0 -> y")
        self.assertEqual(logs[1], "[DEBUG] release 1Key1 -> u")
        self.assertEqual(logs[2], "[DEBUG] press  1Key9 -> ;")

    def test_default_mapping_sends_sky_letter_keys_as_lowercase(self):
        logs = []
        controller = KeyController(note_to_key, debug=True, log_func=logs.append)

        for note in ("1Key0", "1Key5", "1Key10", "2Key4"):
            controller.press(note)

        self.assertEqual(
            logs,
            [
                "[DEBUG] press  1Key0 -> y",
                "[DEBUG] press  1Key5 -> h",
                "[DEBUG] press  1Key10 -> n",
                "[DEBUG] press  2Key4 -> p",
            ],
        )

    def test_missing_note_is_ignored(self):
        logs = []
        controller = KeyController({"1Key0": "Y"}, debug=True, log_func=logs.append)

        controller.press("missing")
        controller.release("missing")

        self.assertEqual(logs, [])


class TestSkyWindowIdentity(unittest.TestCase):
    def test_current_skyautomusic_process_is_not_game_window(self):
        self.assertFalse(
            is_sky_game_window_identity(
                "SkyAutoMusic.exe",
                "SkyAutoMusic auto player",
                pid=123,
                current_pid=123,
            )
        )

    def test_packaged_app_name_is_not_treated_as_sky_game(self):
        self.assertFalse(
            is_sky_game_window_identity(
                "SkyAutoMusic.exe",
                "SkyAutoMusic auto player",
                pid=123,
                current_pid=999,
            )
        )

    def test_real_game_process_or_title_matches(self):
        self.assertTrue(is_sky_game_window_identity("Sky.exe", "", pid=1, current_pid=2))
        self.assertTrue(is_sky_game_window_identity("wrapped.exe", "Sky", pid=1, current_pid=2))
        self.assertTrue(is_sky_game_window_identity("光遇.exe", "", pid=1, current_pid=2))


if __name__ == "__main__":
    unittest.main()
