import keyboard
import pyautogui


note_to_key = {
    "1Key0": "Y", "1Key1": "U", "1Key2": "I", "1Key3": "O", "1Key4": "P",
    "1Key5": "H", "1Key6": "J", "1Key7": "K", "1Key8": "L", "1Key9": ";",
    "1Key10": "N", "1Key11": "M", "1Key12": ",", "1Key13": ".", "1Key14": "/",
    "2Key0": "Y", "2Key1": "U", "2Key2": "I", "2Key3": "O", "2Key4": "P",
    "2Key5": "H", "2Key6": "J", "2Key7": "K", "2Key8": "L", "2Key9": ";",
    "2Key10": "N", "2Key11": "M", "2Key12": ",", "2Key13": ".", "2Key14": "/",
}


class KeyController:
    """Keyboard abstraction for Sky note playback."""

    def __init__(self, mapping=None, debug=False, log_func=None):
        self.mapping = dict(mapping or {})
        self.debug = debug
        self.log_func = log_func or (lambda msg: print(msg))

    def set_mapping(self, mapping):
        self.mapping = dict(mapping or {})

    def set_debug(self, debug):
        self.debug = debug

    def _resolve(self, note):
        return self.mapping.get(note)

    def _normalize_key(self, key):
        """Normalize letter key names to physical lowercase keys."""
        if isinstance(key, str) and len(key) == 1 and key.isalpha():
            return key.lower()
        return key

    def press(self, note):
        key = self._resolve(note)
        if not key:
            return
        key = self._normalize_key(key)
        if self.debug:
            self.log_func(f"[DEBUG] press  {note} -> {key}")
            return
        try:
            keyboard.press(key)
        except Exception:
            try:
                pyautogui.keyDown(key)
            except Exception:
                self.log_func(f"[WARN] 按键失败: {key}")

    def release(self, note):
        key = self._resolve(note)
        if not key:
            return
        key = self._normalize_key(key)
        if self.debug:
            self.log_func(f"[DEBUG] release {note} -> {key}")
            return
        try:
            keyboard.release(key)
        except Exception:
            try:
                pyautogui.keyUp(key)
            except Exception:
                self.log_func(f"[WARN] 释放失败: {key}")

    def press_keys(self, notes):
        for note in notes:
            self.press(note)

    def release_keys(self, notes):
        for note in notes:
            self.release(note)
