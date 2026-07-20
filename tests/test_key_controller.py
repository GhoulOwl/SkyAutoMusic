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

from key_controller import KeyController, note_to_key  # noqa: E402
from score_overlay import display_key  # noqa: E402
from window_focus import is_sky_game_window_identity  # noqa: E402


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


class _FakeBackend:
    """可控的后端替身：可记录调用、模拟失败或不可用。"""

    def __init__(self, name, available=True, fail=False):
        self.name = name
        self.label = name
        self._available = available
        self._fail = fail
        self.calls = []

    @property
    def available(self):
        return self._available

    def press(self, key):
        if self._fail:
            raise RuntimeError("boom")
        self.calls.append(("press", key))

    def release(self, key):
        if self._fail:
            raise RuntimeError("boom")
        self.calls.append(("release", key))


class TestInputBackends(unittest.TestCase):
    def _force_driver_unavailable(self, controller):
        controller._backends["interception"]._available = False

    def test_auto_falls_back_to_keyboard_without_driver(self):
        controller = KeyController(note_to_key)
        self._force_driver_unavailable(controller)
        controller.set_backend("auto")
        self.assertFalse(controller.is_driver_available())
        self.assertEqual(controller.effective_backend(), "keyboard")
        self.assertTrue(controller.is_auto())

    def test_backend_options_include_auto_and_all_backends(self):
        controller = KeyController(note_to_key)
        names = [name for name, _ in controller.backend_options()]
        self.assertEqual(names[0], "auto")
        for expected in ("interception", "keyboard", "pyautogui"):
            self.assertIn(expected, names)

    def test_set_invalid_backend_raises(self):
        controller = KeyController(note_to_key)
        with self.assertRaises(ValueError):
            controller.set_backend("nonexistent")

    def test_explicit_interception_falls_back_when_driver_missing(self):
        controller = KeyController(note_to_key, backend="interception")
        self._force_driver_unavailable(controller)
        self.assertFalse(controller.is_auto())
        # 显式选择 interception 但驱动不可用时，effective 应回退到 keyboard
        self.assertEqual(controller.effective_backend(), "keyboard")

    def test_press_routes_to_active_keyboard_backend(self):
        controller = KeyController({"1Key0": "Y"}, backend="keyboard")
        fake_kb = _FakeBackend("keyboard")
        controller._backends["keyboard"] = fake_kb
        controller._backends["pyautogui"] = _FakeBackend("pyautogui")
        controller.press("1Key0")
        self.assertEqual(fake_kb.calls, [("press", "y")])

    def test_send_falls_back_to_pyautogui_when_keyboard_fails(self):
        controller = KeyController({"1Key0": "Y"}, backend="keyboard")
        controller._backends["keyboard"] = _FakeBackend("keyboard", fail=True)
        fake_pag = _FakeBackend("pyautogui")
        controller._backends["pyautogui"] = fake_pag
        controller.press("1Key0")
        self.assertEqual(fake_pag.calls, [("press", "y")])

    def test_all_backends_failing_logs_warning(self):
        logs = []
        controller = KeyController(
            {"1Key0": "Y"}, backend="keyboard", log_func=logs.append)
        self._force_driver_unavailable(controller)
        controller._backends["keyboard"] = _FakeBackend("keyboard", fail=True)
        controller._backends["pyautogui"] = _FakeBackend("pyautogui", fail=True)
        controller.press("1Key0")
        self.assertTrue(any("按键press失败" in m for m in logs))


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


class TestScoreOverlayHelpers(unittest.TestCase):
    def test_second_keyboard_keys_share_display_slots(self):
        self.assertEqual(display_key("2Key14"), "1Key14")
        self.assertEqual(display_key("1Key3"), "1Key3")


if __name__ == "__main__":
    unittest.main()
