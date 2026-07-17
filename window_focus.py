import ctypes
import os

import psutil
import win32con
import win32gui
import win32process


SKY_PROCESS_NAMES = {"sky.exe", "sky"}


def is_sky_game_window_identity(process_name, window_title="", pid=None, current_pid=None):
    """Return True for the game window, while excluding this app process."""
    if pid is not None and current_pid is not None and pid == current_pid:
        return False

    name = (process_name or "").strip()
    title = (window_title or "").strip()
    name_lower = name.lower()
    title_lower = title.lower()

    if name_lower in SKY_PROCESS_NAMES or title_lower == "sky":
        return True
    return "光遇" in name or "光遇" in title


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def describe_window(hwnd):
    if not hwnd:
        return "无"
    try:
        _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
        name = psutil.Process(pid).name()
        title = win32gui.GetWindowText(hwnd) or "(无标题)"
        return f"{title} / {name} / pid={pid} / hwnd={hwnd}"
    except Exception:
        return f"hwnd={hwnd}"


def describe_foreground_window():
    try:
        return describe_window(win32gui.GetForegroundWindow())
    except Exception:
        return "无法读取"


def find_sky_game_window(current_pid=None):
    current_pid = os.getpid() if current_pid is None else current_pid
    candidates = []

    def enum_windows_callback(hwnd, result):
        if not win32gui.IsWindowVisible(hwnd) or not win32gui.IsWindowEnabled(hwnd):
            return
        try:
            _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc_name = psutil.Process(pid).name()
            title = win32gui.GetWindowText(hwnd)
            if not is_sky_game_window_identity(proc_name, title, pid, current_pid):
                return
            name_lower = (proc_name or "").lower()
            title_lower = (title or "").lower()
            score = 2 if name_lower in SKY_PROCESS_NAMES or "光遇" in proc_name else 1
            if title_lower == "sky" or "光遇" in title:
                score = max(score, 1)
            result.append((score, hwnd))
        except Exception:
            pass

    win32gui.EnumWindows(enum_windows_callback, candidates)
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def bring_window_to_front(hwnd):
    if not hwnd:
        return False

    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception:
        pass

    try:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
        )
    except Exception:
        pass

    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

    try:
        if win32gui.GetForegroundWindow() == hwnd:
            return True
    except Exception:
        return False

    try:
        foreground = win32gui.GetForegroundWindow()
        if foreground:
            tid_target = win32process.GetWindowThreadProcessId(hwnd)[0]
            tid_fore = win32process.GetWindowThreadProcessId(foreground)[0]
            if tid_target and tid_fore and tid_target != tid_fore:
                win32process.AttachThreadInput(tid_fore, tid_target, True)
                try:
                    win32gui.SetForegroundWindow(hwnd)
                    win32gui.SetFocus(hwnd)
                finally:
                    win32process.AttachThreadInput(tid_fore, tid_target, False)
    except Exception:
        pass

    try:
        return win32gui.GetForegroundWindow() == hwnd
    except Exception:
        return False


def release_topmost(hwnd):
    if not hwnd:
        return
    try:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
        )
    except Exception:
        pass


def get_window_rect(hwnd):
    try:
        return win32gui.GetWindowRect(hwnd)
    except Exception:
        return None
