import importlib
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestPlatformCompatibility(unittest.TestCase):
    def test_platform_capabilities_are_consistent(self):
        from platform_support import CAPABILITIES

        if sys.platform == "darwin":
            self.assertTrue(CAPABILITIES.is_macos)
            self.assertFalse(CAPABILITIES.game_playback_supported)
            self.assertFalse(CAPABILITIES.global_hotkeys_supported)
            self.assertFalse(CAPABILITIES.overlay_supported)
            self.assertEqual(CAPABILITIES.audio_backend, "pygame")
            self.assertEqual(CAPABILITIES.credential_backend, "keychain")
        elif sys.platform == "win32":
            self.assertTrue(CAPABILITIES.is_windows)
            self.assertTrue(CAPABILITIES.game_playback_supported)

    def test_windows_helpers_are_importable_without_win32_on_non_windows(self):
        import key_controller
        import window_focus
        import score_overlay

        if sys.platform != "win32":
            self.assertIsNone(key_controller.keyboard)
            self.assertIsNone(key_controller.pyautogui)
            self.assertIsNone(window_focus.find_sky_game_window())
            self.assertFalse(window_focus.bring_window_to_front(1))
            self.assertIsNone(window_focus.get_window_rect(1))
            self.assertIsNone(score_overlay.win32gui)

    def test_gui_imports_when_tk_is_available(self):
        try:
            import tkinter  # noqa: F401
        except ImportError:
            self.skipTest("当前 Python 没有 Tk；macOS CI 使用带 Tk 的 Python 3.10")
        importlib.import_module("play_music_gui")


if __name__ == "__main__":
    unittest.main()
