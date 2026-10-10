"""Keyboard abstraction for Sky note playback.

支持多种按键后端，可按需切换：

- ``interception``：基于 Interception 内核驱动的"虚拟 HID / 驱动级键盘"，
  在驱动层注入按键事件，能绕过部分游戏对 SendInput 的屏蔽，默认优先使用。
- ``keyboard``：基于 ``keyboard`` 库的用户级按键（SendInput），作为回退方案。
- ``pyautogui``：最后的兜底方案。

当指定 ``backend="auto"``（默认）时，按 interception -> keyboard -> pyautogui
的顺序自动选择第一个可用的后端；单次按键失败时也会沿同一链路自动回退。
"""

import sys
import threading
from dataclasses import dataclass, field


@dataclass
class KeyBatchResult:
    """Acknowledges input delivery to an API, never receipt by the game."""
    sent_notes: list = field(default_factory=list)
    failed_notes: list = field(default_factory=list)
    backend_by_note: dict = field(default_factory=dict)
    fallback_count: int = 0
    fatal_error: str = ""

if sys.platform == "win32":
    try:
        import keyboard
    except Exception:  # keyboard is optional even on Windows.
        keyboard = None
    try:
        import pyautogui
    except Exception:  # pyautogui is the final Windows fallback.
        pyautogui = None
else:
    # Do not import Windows input packages on macOS/Linux.  The GUI hides game
    # playback there, but keeping the backend objects preserves the public API.
    keyboard = None
    pyautogui = None


note_to_key = {
    "1Key0": "Y", "1Key1": "U", "1Key2": "I", "1Key3": "O", "1Key4": "P",
    "1Key5": "H", "1Key6": "J", "1Key7": "K", "1Key8": "L", "1Key9": ";",
    "1Key10": "N", "1Key11": "M", "1Key12": ",", "1Key13": ".", "1Key14": "/",
    "2Key0": "Y", "2Key1": "U", "2Key2": "I", "2Key3": "O", "2Key4": "P",
    "2Key5": "H", "2Key6": "J", "2Key7": "K", "2Key8": "L", "2Key9": ";",
    "2Key10": "N", "2Key11": "M", "2Key12": ",", "2Key13": ".", "2Key14": "/",
}


# 后端名称常量
BACKEND_AUTO = "auto"
BACKEND_INTERCEPTION = "interception"
BACKEND_KEYBOARD = "keyboard"
BACKEND_PYAUTOGUI = "pyautogui"

# 自动选择 / 回退的优先级顺序
_BACKEND_ORDER = (BACKEND_INTERCEPTION, BACKEND_KEYBOARD, BACKEND_PYAUTOGUI)


class _InterceptionBackend:
    """基于 Interception 内核驱动的虚拟 HID / 驱动级键盘后端。

    需要先安装 ``interception-driver`` 内核驱动，否则该后端不可用，
    ``available`` 会返回 ``False``，由控制器自动回退到其它后端。
    """

    name = BACKEND_INTERCEPTION
    label = "虚拟HID/驱动级键盘 (Interception)"

    def __init__(self):
        self._module = None
        self._available = None  # None=未检测, bool=检测结果

    def _load(self):
        """惰性加载 interception 模块；驱动未安装时返回 None。"""
        if sys.platform != "win32":
            self._available = False
            return None
        if self._module is not None or self._available is False:
            return self._module
        try:
            import interception as _ic  # noqa: F401
            # 模块导入时会创建全局 Interception 上下文并打开全部设备句柄；
            # 若驱动未安装，构造过程会抛异常，这里统一捕获。
            self._module = _ic
            self._available = True
        except Exception:
            self._module = None
            self._available = False
        return self._module

    @property
    def available(self):
        if self._available is None:
            self._load()
        return bool(self._available)

    def press(self, key):
        mod = self._load()
        if mod is None:
            raise RuntimeError("interception 驱动不可用")
        mod.key_down(key)

    def release(self, key):
        mod = self._load()
        if mod is None:
            raise RuntimeError("interception 驱动不可用")
        mod.key_up(key)

    def calibrate(self):
        """交互式校准键盘设备编号（需用户按一次键以识别设备）。

        调用 ``interception.auto_capture_devices(keyboard=True, mouse=False)``，
        该函数会阻塞直到捕获到一次按键事件。
        """
        mod = self._load()
        if mod is None:
            raise RuntimeError("interception 驱动不可用")
        mod.auto_capture_devices(keyboard=True, mouse=False)


class _KeyboardBackend:
    """基于 ``keyboard`` 库的用户级键盘后端（SendInput）。"""

    name = BACKEND_KEYBOARD
    label = "常规键盘 (keyboard)"

    @property
    def available(self):
        return keyboard is not None

    def press(self, key):
        if keyboard is None:
            raise RuntimeError("keyboard 后端不可用")
        keyboard.press(key)

    def release(self, key):
        if keyboard is None:
            raise RuntimeError("keyboard 后端不可用")
        keyboard.release(key)

    def send_many(self, action, keys):
        """Insert one chord as a single scan-code SendInput array."""
        import ctypes
        from ctypes import wintypes
        class KeyboardInput(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_size_t)]
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
        class HardwareInput(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]
        class Payload(ctypes.Union):
            _fields_ = [("ki", KeyboardInput), ("mi", MouseInput), ("hi", HardwareInput)]
        class Input(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("payload", Payload)]
        inputs = (Input * len(keys))()
        for index, key in enumerate(keys):
            scan = keyboard.key_to_scan_codes(key)[0]
            flags = 0x0008 | (0x0002 if action == "release" else 0) | (0x0001 if scan > 255 else 0)
            inputs[index].type = 1
            inputs[index].payload.ki = KeyboardInput(0, scan & 255, flags, 0, 0)
        send = ctypes.WinDLL("user32", use_last_error=True).SendInput
        send.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        send.restype = wintypes.UINT
        return int(send(len(inputs), inputs, ctypes.sizeof(Input)))


class _PyAutoGUIBackend:
    """基于 ``pyautogui`` 的兼容兜底后端。"""

    name = BACKEND_PYAUTOGUI
    label = "兼容模式 (pyautogui)"

    @property
    def available(self):
        return pyautogui is not None

    def press(self, key):
        if pyautogui is None:
            raise RuntimeError("pyautogui 后端不可用")
        pyautogui.keyDown(key, _pause=False)

    def release(self, key):
        if pyautogui is None:
            raise RuntimeError("pyautogui 后端不可用")
        try:
            pyautogui.keyUp(key, _pause=False)
        except Exception as exc:
            if isinstance(exc, getattr(pyautogui, "FailSafeException", ())):
                # Corner-stop must still let go of keys already pressed. This
                # uses the same Windows key-up implementation, never key-down.
                pyautogui._platformModule._keyUp(key)
            else:
                raise


_BACKEND_CLASSES = {
    BACKEND_INTERCEPTION: _InterceptionBackend,
    BACKEND_KEYBOARD: _KeyboardBackend,
    BACKEND_PYAUTOGUI: _PyAutoGUIBackend,
}


class KeyController:
    """Keyboard abstraction for Sky note playback."""

    def __init__(self, mapping=None, debug=False, log_func=None, backend=BACKEND_AUTO):
        self.mapping = dict(mapping or {})
        self.debug = debug
        self.log_func = log_func or (lambda msg: print(msg))
        self._backends = {name: cls() for name, cls in _BACKEND_CLASSES.items()}
        self._backend_name = BACKEND_KEYBOARD
        self._auto = False
        self._held_backend = {}
        self._input_lock = threading.RLock()
        self.last_backend = None
        self.set_backend(backend)

    # ---------- 后端管理 ----------
    def set_mapping(self, mapping):
        self.mapping = dict(mapping or {})

    def set_debug(self, debug):
        self.debug = debug

    def backend_options(self):
        """返回 [(name, label), ...]，按优先级排列，供 UI 选择使用。"""
        opts = [(BACKEND_AUTO, "自动（优先驱动级）")]
        for name in _BACKEND_ORDER:
            opts.append((name, self._backends[name].label))
        return opts

    def available_backends(self):
        """返回当前可用的后端名称列表（按优先级）。"""
        return [name for name in _BACKEND_ORDER if self._backends[name].available]

    def is_driver_available(self):
        return self._backends[BACKEND_INTERCEPTION].available

    def set_backend(self, name):
        """切换按键后端。``auto`` 表示自动选择第一个可用后端。"""
        if name == BACKEND_AUTO:
            chosen = None
            for cand in _BACKEND_ORDER:
                if self._backends[cand].available:
                    chosen = cand
                    break
            if chosen is None:
                chosen = BACKEND_KEYBOARD  # 兜底
            self._backend_name = chosen
            self._auto = True
        else:
            if name not in self._backends:
                raise ValueError(f"未知按键后端: {name}")
            self._backend_name = name
            self._auto = False
        return self._backend_name

    def get_backend(self):
        """返回当前（auto 解析后）的后端名称。"""
        return self._backend_name

    def is_auto(self):
        return self._auto

    def effective_backend(self):
        """实际会使用的后端名称（结合可用性判断，非 auto 时可能回退）。"""
        name = self._backend_name
        if self._backends[name].available:
            return name
        for cand in _BACKEND_ORDER:
            if self._backends[cand].available:
                return cand
        return name

    def get_backend_label(self, name=None):
        if name is None:
            name = self.effective_backend()
        return self._backends[name].label

    def calibrate_driver_keyboard(self):
        """交互式校准驱动级键盘设备（仅 interception 后端可用时有效）。"""
        self._backends[BACKEND_INTERCEPTION].calibrate()

    # ---------- 按键逻辑 ----------
    def _resolve(self, note):
        return self.mapping.get(note)

    def physical_key(self, note):
        return self._normalize_key(self._resolve(note)) or note

    def _normalize_key(self, key):
        """Normalize letter key names to physical lowercase keys.

        驱动级 / 用户级后端均依据键名字符串查表（VkKeyScan），
        大写字母会被识别为 Shift+字母，因此必须归一为小写。
        """
        if isinstance(key, str) and len(key) == 1 and key.isalpha():
            return key.lower()
        return key

    def _send(self, action, key):
        """向后端发送按键，失败时按优先级回退到其他后端。"""
        # 候选顺序：当前后端优先，其余按 _BACKEND_ORDER
        order = [self._backend_name] + [n for n in _BACKEND_ORDER if n != self._backend_name]
        last_err = None
        for name in order:
            backend = self._backends[name]
            if not backend.available:
                continue
            try:
                getattr(backend, action)(key)
                return name
            except Exception as e:
                last_err = e
                continue
        if last_err:
            self.log_func(f"[WARN] 按键{action}失败: {key} ({last_err})")
        else:
            self.log_func(f"[WARN] 当前平台不支持键盘注入: {action} {key}")

    def press(self, note):
        key = self._resolve(note)
        if not key:
            return
        key = self._normalize_key(key)
        if self.debug:
            self.log_func(f"[DEBUG] press  {note} -> {key}")
            return
        self._send("press", key)

    def release(self, note):
        key = self._resolve(note)
        if not key:
            return
        key = self._normalize_key(key)
        if self.debug:
            self.log_func(f"[DEBUG] release {note} -> {key}")
            return
        self._send("release", key)

    def _send_batch(self, action, notes):
        result = KeyBatchResult()
        with self._input_lock:
            pending = {}
            for note in dict.fromkeys(notes):
                key = self._resolve(note)
                if not key or self.debug:
                    if self.debug and key:
                        self.log_func(f"[DEBUG] {action} {note} -> {self._normalize_key(key)}")
                    result.failed_notes.append(note)
                else:
                    pending.setdefault(self._normalize_key(key), []).append(note)
            # Release each physical key on the backend that delivered key-down.
            groups = {}
            for key in pending:
                preferred = self._held_backend.get(key, self._backend_name) if action == "release" else self._backend_name
                groups.setdefault(preferred, []).append(key)
            for preferred, keys in groups.items():
                remaining = list(keys)
                order = [preferred] + [name for name in _BACKEND_ORDER if name != preferred]
                for name in order:
                    if result.fatal_error:
                        break
                    backend = self._backends[name]
                    if not backend.available or not remaining:
                        continue
                    succeeded = []
                    try:
                        if hasattr(backend, "send_many"):
                            count = backend.send_many(action, remaining)
                            succeeded = remaining[:max(0, min(len(remaining), count))]
                        else:
                            for key in remaining:
                                try:
                                    getattr(backend, action)(key)
                                    succeeded.append(key)
                                except Exception as exc:
                                    self.log_func(f"[WARN] {name} {action} {key}: {exc}")
                                    if isinstance(exc, getattr(pyautogui, "FailSafeException", ())) and action == "press":
                                        result.fatal_error = "鼠标角落紧急停止"
                                        break
                    except Exception as exc:
                        self.log_func(f"[WARN] {name} {action}: {exc}")
                    for key in succeeded:
                        for note in pending[key]:
                            result.sent_notes.append(note)
                            result.backend_by_note[note] = name
                        if name != preferred:
                            result.fallback_count += 1
                        if action == "press":
                            self._held_backend[key] = name
                        else:
                            self._held_backend.pop(key, None)
                        self.last_backend = name
                    remaining = [key for key in remaining if key not in succeeded]
                for key in remaining:
                    result.failed_notes.extend(pending[key])
                    self.log_func(f"[WARN] 按键{action}失败: {key}")
        return result

    def press_keys(self, notes):
        return self._send_batch("press", notes)

    def release_keys(self, notes):
        return self._send_batch("release", notes)
