import tkinter as tk

import win32con
import win32gui

from window_focus import get_window_rect


KEY_ORDER = [f"1Key{i}" for i in range(15)]


def display_key(note_key):
    if isinstance(note_key, str) and note_key.startswith("2Key"):
        return "1Key" + note_key[4:]
    return note_key


class ScoreOverlay:
    """Semi-transparent click-through score overlay for the game window."""

    def __init__(self, root, geometry=None, locked=True, on_geometry_changed=None, log_func=None):
        self.root = root
        self.locked = bool(locked)
        self.on_geometry_changed = on_geometry_changed or (lambda geometry: None)
        self.log_func = log_func or (lambda msg: None)
        self.notes_by_time = {}
        self.sorted_times = []
        self.title = ""
        self.current_idx = -1
        self.current_notes = []
        self.progress = 0.0
        self.elapsed_sec = 0.0
        self._drag_start = None
        self._manual_position = bool(geometry)

        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.title("SkyAutoMusic 乐谱覆盖层")
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        try:
            self.window.attributes("-alpha", 0.72)
        except Exception:
            pass
        self.window.configure(bg="#101820")
        self.window.geometry(geometry or "560x190+80+80")

        self.canvas = tk.Canvas(
            self.window,
            bg="#101820",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._begin_drag)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)
        self.window.bind("<Configure>", self._on_configure)
        self.window.after(150, self._apply_clickthrough)

    def set_score(self, notes_by_time, sorted_times, title=""):
        self.notes_by_time = notes_by_time or {}
        self.sorted_times = list(sorted_times or [])
        self.title = title or ""
        self.current_idx = -1
        self.current_notes = []
        self.progress = 0.0
        self.elapsed_sec = 0.0
        self.redraw()

    def show(self, game_hwnd=None):
        if game_hwnd and not self._manual_position:
            self.place_near_window(game_hwnd)
        self.window.deiconify()
        self.window.lift()
        self._apply_clickthrough()
        self.redraw()

    def hide(self):
        self.window.withdraw()

    def toggle_lock(self):
        self.locked = not self.locked
        self._apply_clickthrough()
        self.redraw()
        return self.locked

    def set_locked(self, locked):
        self.locked = bool(locked)
        self._apply_clickthrough()
        self.redraw()

    def update_playback(self, progress=None, elapsed_sec=None, current_idx=None, current_notes=None):
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))
        if elapsed_sec is not None:
            self.elapsed_sec = max(0.0, float(elapsed_sec))
        if current_idx is not None:
            self.current_idx = int(current_idx)
        if current_notes is not None:
            self.current_notes = list(current_notes)
        if self.window.winfo_viewable():
            self.redraw()

    def place_near_window(self, hwnd):
        rect = get_window_rect(hwnd)
        if not rect:
            return
        left, top, right, bottom = rect
        width = min(620, max(420, right - left - 80))
        height = 190
        x = left + max(20, ((right - left) - width) // 2)
        y = bottom - height - 60
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def redraw(self):
        if not self.canvas.winfo_exists():
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")

        self.canvas.create_rectangle(0, 0, width, height, fill="#101820", outline="")
        title = self.title or "未加载乐谱"
        lock_text = "穿透锁定 F10" if self.locked else "移动模式 F10 锁定"
        self.canvas.create_text(14, 13, anchor="w", text=title, fill="#F6FBFF", font=("微软雅黑", 10, "bold"))
        self.canvas.create_text(width - 14, 13, anchor="e", text=lock_text, fill="#9FE6C8", font=("微软雅黑", 9))

        bar_x, bar_y = 14, 30
        bar_w, bar_h = width - 28, 8
        self.canvas.create_rectangle(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, fill="#243447", outline="")
        self.canvas.create_rectangle(bar_x, bar_y, bar_x + int(bar_w * self.progress), bar_y + bar_h, fill="#4F8CFF", outline="")

        m, s = divmod(int(self.elapsed_sec), 60)
        pct = int(self.progress * 100)
        self.canvas.create_text(width - 14, 47, anchor="e", text=f"{m}:{s:02d}  {pct}%", fill="#C8D5E0", font=("Consolas", 9))

        grid_top = 60
        pad = 10
        gap = 7
        cols, rows = 5, 3
        cell_w = (width - pad * 2 - gap * (cols - 1)) / cols
        cell_h = 28
        active = {display_key(note) for note in self.current_notes}

        for idx, key in enumerate(KEY_ORDER):
            row = idx // cols
            col = idx % cols
            x1 = pad + col * (cell_w + gap)
            y1 = grid_top + row * (cell_h + gap)
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            is_active = key in active
            fill = "#4F8CFF" if is_active else "#1B2A38"
            outline = "#BFE2FF" if is_active else "#315068"
            text_fill = "#FFFFFF" if is_active else "#AFC1D0"
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=2)
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=str(idx + 1),
                fill=text_fill,
                font=("Consolas", 10, "bold"),
            )

        self._draw_density(width, height)

    def _draw_density(self, width, height):
        x1, y1 = 14, height - 28
        x2, y2 = width - 14, height - 12
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#152332", outline="#2B4457")
        if not self.sorted_times:
            return
        start = self.sorted_times[0]
        span = max(1, self.sorted_times[-1] - start)
        max_notes = max((len(self.notes_by_time.get(t, [])) for t in self.sorted_times), default=1)
        for t in self.sorted_times[:: max(1, len(self.sorted_times) // 240)]:
            note_count = len(self.notes_by_time.get(t, []))
            frac = (t - start) / span
            x = x1 + frac * (x2 - x1)
            h = max(3, (note_count / max_notes) * (y2 - y1 - 2))
            self.canvas.create_line(x, y2 - 1, x, y2 - h, fill="#6DBBFF")
        cursor_x = x1 + self.progress * (x2 - x1)
        self.canvas.create_line(cursor_x, y1 - 4, cursor_x, y2 + 4, fill="#FFE08A", width=2)

    def _begin_drag(self, event):
        if self.locked:
            return
        self._drag_start = (event.x_root, event.y_root, self.window.winfo_x(), self.window.winfo_y())

    def _drag(self, event):
        if self.locked or not self._drag_start:
            return
        sx, sy, wx, wy = self._drag_start
        self.window.geometry(f"+{wx + event.x_root - sx}+{wy + event.y_root - sy}")

    def _end_drag(self, _event):
        self._drag_start = None
        self._manual_position = True
        self.on_geometry_changed(self.window.geometry())

    def _on_configure(self, _event):
        if self.window.winfo_viewable():
            self.redraw()

    def _apply_clickthrough(self):
        try:
            hwnd = self.window.winfo_id()
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            styles |= win32con.WS_EX_LAYERED | win32con.WS_EX_TOOLWINDOW
            if self.locked:
                styles |= win32con.WS_EX_TRANSPARENT
            else:
                styles &= ~win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        except Exception as exc:
            self.log_func(f"[WARN] 覆盖层点击穿透设置失败: {exc}")
