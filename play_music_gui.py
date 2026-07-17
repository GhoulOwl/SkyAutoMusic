import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pyautogui
import keyboard
from collections import defaultdict
import win32gui
import win32process
import win32con
import psutil
import sys
import ctypes
import webbrowser
import random
from enum import Enum

# 资源路径适配函数，兼容PyInstaller打包和开发环境
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # 打包后（PyInstaller onefile/onedir）：使用 exe 所在目录，
        # 使 Sheet Music/、config.json、favorites.json 等定位到 exe 同级，
        # 用户自行放置的乐谱能被找到，且配置可持久化。
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# 配置
SHEET_MUSIC_DIR = resource_path('Sheet Music')
if not os.path.exists(SHEET_MUSIC_DIR):
    os.makedirs(SHEET_MUSIC_DIR)
note_to_key = {
    "1Key0": "Y", "1Key1": "U", "1Key2": "I", "1Key3": "O", "1Key4": "P",
    "1Key5": "H", "1Key6": "J", "1Key7": "K", "1Key8": "L", "1Key9": ";",
    "1Key10": "N", "1Key11": "M", "1Key12": ",", "1Key13": ".", "1Key14": "/",
    "2Key0": "Y", "2Key1": "U", "2Key2": "I", "2Key3": "O", "2Key4": "P",
    "2Key5": "H", "2Key6": "J", "2Key7": "K", "2Key8": "L", "2Key9": ";",
    "2Key10": "N", "2Key11": "M", "2Key12": ",", "2Key13": ".", "2Key14": "/"
}

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


CONFIG_FILE = resource_path('config.json')

def is_dark_mode():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
            r'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize')
        value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
        return value == 0
    except Exception:
        return False

class PlaybackState(Enum):
    """三态状态机：未播放 / 播放中 / 暂停中。"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class KeyController:
    """统一的按键抽象层。

    - 调试模式：开启后只输出日志，不真正按下按键（PR 调试模式前置能力）。
    - 优先使用 keyboard 库，失败时回退 pyautogui。
    - 支持运行时切换按键映射（多套键盘映射的前置能力）。
    """

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
        """单字母键名统一小写，避免 keyboard / pyautogui 把大写解析成 'Shift+字母'。

        光遇 PC 版乐器对应键盘物理键（小写键名）。若发送 'Y' 这类大写，
        keyboard.press('Y') 会被理解为「按住 Shift 再按 y」，游戏收不到正确的
        音符键位 → 表现为「按键未生效」。统一转小写即可正确按到目标键。
        """
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
        for n in notes:
            self.press(n)

    def release_keys(self, notes):
        for n in notes:
            self.release(n)


class MusicPlayer:
    """播放内核：三态状态机 + 严格 time 差值 + 速度/模拟/调试参数。

    时序规则（依据 1.md）：严格按乐谱 time（毫秒绝对时间戳）的差值播放，
    不乘任何 bpm 系数；speed 为可实时调节的用户速度倍率（1.0 = 原速）。
    simulate 开启后加入自然的随机变速与随机失误（模拟真实演奏）。
    """

    DEFAULT_MISS_PROB = 0.03        # 每个音符被"失误跳过"的概率
    DEFAULT_JITTER = (0.85, 1.15)   # 间隔随机抖动范围
    NOTE_HOLD = 0.05                # 单音符按住基准时长（秒）

    def __init__(self, key_controller, update_status=None, update_elapsed=None,
                 update_total=None, update_progress=None):
        self.key_controller = key_controller
        self.update_status = update_status or (lambda msg: None)
        self.update_elapsed = update_elapsed or (lambda sec: None)
        self.update_total = update_total or (lambda sec: None)
        self.update_progress = update_progress or (lambda frac: None)
        self.state = PlaybackState.STOPPED
        self._stop = threading.Event()
        self.notes_by_time = None
        self.sorted_times = None
        self.current_idx = 0
        self.thread = None
        # 可调参数（由 UI 注入）
        self.speed = 1.0
        self.simulate = False
        self.miss_prob = self.DEFAULT_MISS_PROB
        self.jitter = self.DEFAULT_JITTER
        # 内部控制
        self._seek_requested = False
        self._seek_target_idx = 0
        self._held_keys = []

    # ---------- 状态查询 ----------
    def is_playing(self):
        return self.state == PlaybackState.PLAYING

    def is_paused(self):
        return self.state == PlaybackState.PAUSED

    def is_stopped(self):
        return self.state == PlaybackState.STOPPED

    # ---------- 控制入口（供 UI 调用） ----------
    def start(self, notes_by_time, sorted_times):
        """仅当处于 STOPPED 态时，从曲谱开头开始播放。"""
        if self.state != PlaybackState.STOPPED or not sorted_times:
            return False
        self.notes_by_time = notes_by_time
        self.sorted_times = sorted_times
        self.current_idx = 0
        self._stop.clear()
        self._seek_requested = False
        self._held_keys = []
        self.state = PlaybackState.PLAYING
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def pause(self):
        if self.state == PlaybackState.PLAYING:
            self.state = PlaybackState.PAUSED

    def resume(self):
        if self.state == PlaybackState.PAUSED:
            self.state = PlaybackState.PLAYING

    def stop(self):
        """切回 STOPPED 态（自动播放完成或点击结束按钮都走这里）。"""
        self._stop.set()
        self.state = PlaybackState.STOPPED
        self.current_idx = 0
        self._release_held()

    def seek(self, target_idx):
        """定位到指定音符索引（播放中/暂停中均可；STOPPED 态忽略）。"""
        if self.state == PlaybackState.STOPPED or not self.sorted_times:
            return
        self._seek_target_idx = max(0, min(int(target_idx), len(self.sorted_times) - 1))
        self._seek_requested = True

    # ---------- 内部实现 ----------
    def _release_held(self):
        if self._held_keys:
            try:
                self.key_controller.release_keys(self._held_keys)
            except Exception:
                pass
            self._held_keys = []

    def _run(self):
        try:
            self._playback_loop()
        finally:
            self._release_held()
            self.state = PlaybackState.STOPPED
            self.current_idx = 0
            # 仅自然播放完成时提示"结束"，手动停止由 UI 控制文案
            if not self._stop.is_set():
                self.update_progress(1.0)
                self.update_status("演奏结束！")

    def _playback_loop(self):
        notes_by_time = self.notes_by_time
        sorted_times = self.sorted_times
        total = len(sorted_times)
        if total == 0:
            return
        t0 = sorted_times[0]
        last_t = sorted_times[-1]
        span = max(1, last_t - t0)
        self.update_total((last_t - t0) / 1000.0)
        # 起播前给窗口聚焦预留一点时间
        if not self._sleep_with_controls(0.5):
            return
        idx = self.current_idx
        while idx < total:
            if self._stop.is_set():
                break
            if not self._wait_if_paused():
                break
            if self._seek_requested:
                idx = self._consume_seek(t0, last_t, span)
                continue
            t = sorted_times[idx]
            keys = notes_by_time[t]
            # 模拟真实演奏：按概率"失误"跳过该音符
            if self.simulate and random.random() < self.miss_prob:
                self.update_status(f"演奏进度: {idx + 1}/{total}（失误跳过）")
            else:
                held = list(keys)
                self._held_keys = held
                self.key_controller.press_keys(held)
                hold = self.NOTE_HOLD
                if self.simulate:
                    hold *= random.uniform(0.7, 1.4)
                if not self._sleep_with_controls(hold):
                    break
                self.key_controller.release_keys(held)
                if self._held_keys is held:
                    self._held_keys = []
                self.update_status(f"演奏进度: {idx + 1}/{total}")
            # 进度与时间显示（严格按 time 差值）
            elapsed_sec = max(0, (t - t0) / 1000.0)
            self.update_elapsed(elapsed_sec)
            self.update_progress((t - t0) / span)
            if idx < total - 1:
                interval = (sorted_times[idx + 1] - t) / 1000.0
                # 用户速度倍率：speed>1 更快，间隔相应缩短
                if self.speed and self.speed > 0:
                    interval = interval / self.speed
                # 模拟真实演奏：自然的随机变速
                if self.simulate:
                    interval *= random.uniform(*self.jitter)
                if interval > 0 and not self._sleep_with_controls(interval):
                    break
            idx += 1
            self.current_idx = idx

    def _wait_if_paused(self):
        """暂停态下阻塞，直到 resume / stop / seek。返回 False 表示被 stop。"""
        step = 0.05
        while self.state == PlaybackState.PAUSED:
            if self._stop.is_set():
                return False
            if self._seek_requested:
                return True
            time.sleep(step)
        return not self._stop.is_set()

    def _consume_seek(self, t0, last_t, span):
        self._seek_requested = False
        target = max(0, min(self._seek_target_idx, len(self.sorted_times) - 1))
        self.current_idx = target
        t = self.sorted_times[target]
        elapsed_sec = max(0, (t - t0) / 1000.0)
        self.update_elapsed(elapsed_sec)
        self.update_progress((t - t0) / span)
        return target

    def _sleep_with_controls(self, duration):
        """可在小步长内响应 stop / pause / seek 的睡眠。返回 False 表示被 stop 中断。"""
        step = 0.02
        slept = 0.0
        while slept < duration:
            if self._stop.is_set():
                return False
            if self._seek_requested:
                return True
            if self.state == PlaybackState.PAUSED:
                if not self._wait_if_paused():
                    return False
                if self._seek_requested:
                    return True
            time.sleep(step)
            slept += step
        return True

class MusicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SkyAutoMusic 自动弹琴")
        # 读取窗口配置
        win_w, win_h = 600, 480
        x, y = None, None
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                win_w = cfg.get('width', win_w)
                win_h = cfg.get('height', win_h)
                x = cfg.get('x')
                y = cfg.get('y')
            except Exception:
                pass
        if x is not None and y is not None:
            self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        else:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            x = (screen_w - win_w) // 2
            y = (screen_h - win_h) // 2
            self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.resizable(True, True)
        self.root.minsize(480, 360)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # 统一浅色风格
        self.bg_color = "#F7F9FB"
        self.fg_color = "#222"
        self.accent = "#4F8CFF"
        self.frame_bg = "#FFFFFF"
        self.entry_bg = "#F8F8F8"
        self.entry_fg = "#222"
        self.button_bg = "#4F8CFF"
        self.button_fg = "#FFFFFF"
        self.button_active_bg = "#3399FF"
        self.root.configure(bg=self.bg_color)
        self.set_style()
        self.default_hotkeys = {'start': 'F5', 'stop': 'F7'}
        self.hotkeys = self.default_hotkeys.copy()
        self.hotkey_vars = {k: tk.StringVar(value=v) for k, v in self.hotkeys.items()}
        self.status_var = tk.StringVar(value="请选择乐谱并点击开始演奏")
        self.elapsed_time_var = tk.StringVar(value="0:00")
        self.total_time_var = tk.StringVar(value="0:00")
        self.music_info_vars = {
            'filename': tk.StringVar(),
            'path': tk.StringVar(),
            'name': tk.StringVar(),
            'author': tk.StringVar(),
            'transcribedBy': tk.StringVar(),
        }
        self.filtered_music_files = []  # 先初始化，防止后续方法引用时报错
        self.favorites = set()  # 收藏的乐谱文件名集合，可持久化
        self.favorite_file = resource_path('favorites.json')  # 用resource_path，兼容打包
        self.load_favorites()  # 启动时加载收藏
        self.create_widgets()
        # 关键：初始化后立即加载乐谱列表并刷新
        self.all_music_files = self.get_all_music_files() or []
        self.filtered_music_files = self.all_music_files.copy()
        self.refresh_music_listbox()
        # 统一按键抽象层（调试模式、可切换映射的前置能力）
        self.key_controller = KeyController(
            mapping=note_to_key, debug=False, log_func=self._debug_log)
        # 播放内核（三态状态机），回调绑定到本类方法
        self.player = MusicPlayer(
            key_controller=self.key_controller,
            update_status=self._on_status,
            update_elapsed=self._on_elapsed,
            update_total=self._on_total,
            update_progress=self._on_progress,
        )
        self.music_data = None
        self.notes_by_time = None
        self.sorted_times = None
        # 播放相关可调参数（后续 UI 注入，这里给默认值）
        self.debug = False
        self.speed = 1.0
        self.simulate = False
        self.miss_prob = 0.03
        self._progress_frac = 0.0
        self._game_hwnd = None  # 当前已置顶/聚焦的游戏窗口句柄，stop 时解除置顶
        self.debug_logs = []
        self.last_music_files = set(self.all_music_files or [])
        self.schedule_music_dir_watch()
        # 启动进度条定时刷新（主线程驱动，读取播放线程写入的 _progress_frac）
        self._refresh_progress_ui()

    def set_style(self):
        style = ttk.Style()
        if sys.platform == "win32":
            try:
                style.theme_use('vista')
            except Exception:
                style.theme_use('clam')
        else:
            style.theme_use('clam')
        style.configure('.', font=('微软雅黑', 10))
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabelframe', background=self.bg_color, foreground=self.fg_color, borderwidth=0)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.accent, font=('微软雅黑', 9, 'bold'))
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('TButton', background=self.button_bg, foreground='#222', borderwidth=0, relief='flat', padding=4, font=('微软雅黑', 9, 'bold'))
        style.map('TButton', background=[('active', self.button_active_bg)], foreground=[('active', '#222')])
        style.configure('Accent.TButton', background=self.accent, foreground='#222', borderwidth=0, relief='flat', padding=4, font=('微软雅黑', 9, 'bold'))
        style.map('Accent.TButton', background=[('active', self.button_active_bg)], foreground=[('active', '#222')])
        style.configure('TEntry', fieldbackground=self.entry_bg, background=self.entry_bg, foreground=self.entry_fg, borderwidth=1, relief='flat')
        style.configure('TCombobox', fieldbackground=self.entry_bg, background=self.entry_bg, foreground=self.entry_fg, borderwidth=1, relief='flat')
        style.map('TCombobox', fieldbackground=[('readonly', self.entry_bg)], background=[('readonly', self.entry_bg)], foreground=[('readonly', self.entry_fg)])
        # 现代美观进度条样式
        style.layout('Modern.Horizontal.TProgressbar', [
            ('Horizontal.Progressbar.trough', {'children': [
                ('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})
            ], 'sticky': 'nswe'})
        ])
        style.configure('Modern.Horizontal.TProgressbar',
            troughcolor='#E6EAF0',
            background=self.accent,
            thickness=18,
            borderwidth=0,
            relief='flat',
            lightcolor='#A7C7FF',
            darkcolor='#4F8CFF',
            bordercolor='#E6EAF0',
            padding=2
        )
        # 渐变色和圆角效果（部分平台支持）
        try:
            style.element_create('Rounded.pbar', 'from', 'clam')
            style.layout('Modern.Horizontal.TProgressbar', [
                ('Horizontal.Progressbar.trough', {'children': [
                    ('Rounded.pbar', {'side': 'left', 'sticky': 'ns'})
                ], 'sticky': 'nswe'})
            ])
        except Exception:
            pass

    def create_widgets(self):
        # 主Notebook分页
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=0, pady=0)
        # 播放Tab
        play_tab = ttk.Frame(notebook)
        notebook.add(play_tab, text="播放")
        # 设置Tab
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="说明")
        # 播放Tab内容（三栏布局）
        main_frame = ttk.Frame(play_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        main_frame.columnconfigure(0, weight=0, minsize=220)
        main_frame.columnconfigure(1, weight=1, minsize=320)
        main_frame.columnconfigure(2, weight=0, minsize=0)
        # ====== 右侧主控区 ======
        center_frame = ttk.Frame(main_frame, width=320)
        center_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        center_frame.grid_propagate(False)
        center_frame.config(width=320)

        # ====== 曲谱信息展示区（右侧，按钮组上方） ======
        # 歌名、作者、制谱人在上方大字号高亮，文件名在下方小字号
        self.music_info_frame = ttk.LabelFrame(center_frame, text="曲谱信息", padding=6)
        self.music_info_frame.pack(fill="x", pady=(0, 8), anchor="n")
        # 歌名
        ttk.Label(self.music_info_frame, text="歌名:", width=7, anchor="e").grid(row=0, column=0, sticky="e", pady=(0,2))
        ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['name'], width=18, anchor="w", font=("微软雅黑", 12, "bold"), foreground=self.accent).grid(row=0, column=1, sticky="w", pady=(0,2))
        # 作者
        ttk.Label(self.music_info_frame, text="作者:", width=7, anchor="e").grid(row=1, column=0, sticky="e", pady=(0,2))
        ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['author'], width=18, anchor="w", font=("微软雅黑", 12, "bold"), foreground=self.accent).grid(row=1, column=1, sticky="w", pady=(0,2))
        # 制谱人
        ttk.Label(self.music_info_frame, text="制谱:", width=7, anchor="e").grid(row=2, column=0, sticky="e", pady=(0,2))
        ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['transcribedBy'], width=18, anchor="w", font=("微软雅黑", 12, "bold"), foreground=self.accent).grid(row=2, column=1, sticky="w", pady=(0,2))
        # 文件名
        ttk.Label(self.music_info_frame, text="文件名:", width=7, anchor="e").grid(row=3, column=0, sticky="e", pady=(6,0))
        ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['filename'], width=18, anchor="w", font=("微软雅黑", 9), foreground="#888").grid(row=3, column=1, sticky="w", pady=(6,0))
        # ====== 左侧乐谱区 ======
        left_frame = ttk.Frame(main_frame, width=220)
        left_frame.grid(row=0, column=0, sticky="nswe", padx=(10, 0), pady=10)
        left_frame.grid_propagate(False)
        left_frame.config(width=220)

        # ====== 乐谱分页按钮（全部/收藏） ======
        # 可自定义：tab_names 可扩展更多分页
        self.music_tabs = ["全部曲谱", "收藏曲谱"]  # 可自定义：分页名称
        self.current_music_tab = tk.StringVar(value=self.music_tabs[0])
        tab_frame = ttk.Frame(left_frame)
        tab_frame.pack(fill="x", pady=(0, 4))
        for name in self.music_tabs:
            btn = ttk.Radiobutton(tab_frame, text=name, value=name, variable=self.current_music_tab,
                                  command=self.on_music_tab_changed, style="Toolbutton")
            btn.pack(side="left", padx=2)

        # ====== 搜索栏 ======
        ttk.Label(left_frame, text="搜索乐谱:", font=("微软雅黑", 10, "bold"), foreground=self.accent, width=12, anchor="w").pack(anchor="w", pady=(0, 2))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(left_frame, textvariable=self.search_var, font=("微软雅黑", 10), width=18)
        self.search_entry.pack(fill="x", padx=(0, 2), pady=(0, 6))
        self.search_entry.bind('<KeyRelease>', self.on_search)

        # ====== 乐谱列表区 ======
        # 可自定义：height 控制显示行数，width 控制显示宽度
        self.music_listbox = tk.Listbox(left_frame, width=22, height=22, font=("微软雅黑", 10), activestyle='dotbox', borderwidth=1, relief='solid')
        self.music_listbox.pack(fill="both", expand=True)
        self.music_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        self.music_listbox.bind('<Button-3>', self.on_music_listbox_right_click)  # 右键菜单
        self.refresh_music_listbox()
        self.tooltip = None
        # 中间：歌曲信息三行+播放时长+主按钮组
        info_frame = ttk.Frame(center_frame, width=300)
        info_frame.pack(pady=(10, 8), fill="x")
        info_frame.pack_propagate(False)
        ttk.Label(info_frame, text="歌曲：", font=("微软雅黑", 10, "bold"), width=6, anchor="w").pack(anchor="w")
        ttk.Label(info_frame, textvariable=self.music_info_vars['name'], font=("微软雅黑", 10, "bold"), foreground=self.accent, width=20, anchor="w").pack(anchor="w", padx=(8, 0))
        ttk.Label(info_frame, text="作者：", font=("微软雅黑", 10, "bold"), width=6, anchor="w").pack(anchor="w", pady=(6, 0))
        ttk.Label(info_frame, textvariable=self.music_info_vars['author'], font=("微软雅黑", 10, "bold"), width=20, anchor="w").pack(anchor="w", padx=(8, 0))
        # 播放时长显示（紧凑居中）
        time_frame = ttk.Frame(center_frame, width=300)
        time_frame.pack(pady=(0, 8), fill="x")
        time_frame.pack_propagate(False)
        self.elapsed_time_var = tk.StringVar(value="0:00")
        self.total_time_var = tk.StringVar(value="0:00")
        time_inner = ttk.Frame(time_frame, width=180)
        time_inner.pack(anchor="center")
        time_inner.pack_propagate(False)
        ttk.Label(time_inner, textvariable=self.elapsed_time_var, font=("Consolas", 11, "bold"), foreground=self.accent, width=7, anchor="e").pack(side="left")
        ttk.Label(time_inner, text="/", font=("微软雅黑", 10, "bold"), foreground="#888", width=2, anchor="center").pack(side="left", padx=2)
        ttk.Label(time_inner, textvariable=self.total_time_var, font=("Consolas", 11, "bold"), foreground="#888", width=7, anchor="w").pack(side="left")
        # ====== 弹奏进度条 ======
        # 复用 set_style 中定义的 Modern.Horizontal.TProgressbar 样式，与主窗口 UI 风格统一。
        # maximum=1000 提供更平滑的分辨率；实际值由 _refresh_progress_ui 定时从 _progress_frac 读取更新，
        # 避免在播放线程里直接操作 Tk 组件（线程安全）。
        progress_frame = ttk.Frame(center_frame, width=300)
        progress_frame.pack(pady=(0, 8), fill="x")
        self.progress_bar = ttk.Progressbar(
            progress_frame, mode="determinate",
            style='Modern.Horizontal.TProgressbar', maximum=1000, value=0)
        self.progress_bar.pack(fill="x", padx=18)
        # 进度百分比文本，居中显示，弱化配色与整体风格协调
        self.progress_percent_var = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.progress_percent_var,
                  font=("Consolas", 9), foreground="#888",
                  background=self.bg_color, anchor="center").pack(fill="x", pady=(2, 0))
        # 操作按钮组
        btn_frame = ttk.Frame(center_frame, width=220)
        btn_frame.pack(pady=12)
        btn_frame.pack_propagate(False)
        self.start_btn = ttk.Button(btn_frame, text="开始演奏 (F5)", command=self.start_play, style='Accent.TButton', width=14)
        self.start_btn.grid(row=0, column=0, padx=10, pady=6)
        self.stop_btn = ttk.Button(btn_frame, text="停止 (F7)", command=self.stop_play, state="disabled", style='Accent.TButton', width=14)
        self.stop_btn.grid(row=0, column=1, padx=10, pady=6)
        # 扒谱（音频→乐谱）入口
        self.generate_btn = ttk.Button(btn_frame, text="生成乐谱", command=self.open_generate_dialog, style='Accent.TButton', width=14)
        self.generate_btn.grid(row=1, column=0, columnspan=2, padx=10, pady=6)
        # 状态栏
        self.status_label = ttk.Label(center_frame, textvariable=self.status_var, anchor="center", font=("微软雅黑", 10, "bold"), background=self.bg_color, foreground=self.accent, width=32)
        self.status_label.pack(pady=6, fill="x")
        # 设置Tab内容
        self.create_hotkey_settings(parent=settings_tab)

    def create_hotkey_settings(self, parent=None):
        frame = ttk.LabelFrame(parent or self.root, text="程序说明", padding=14)
        frame.pack(pady=10, fill="x", padx=8)
        # 作者超链接
        author_label = tk.Label(frame, text="作者: 傅卿何（点击访问主页）", fg="#3366cc", cursor="hand2", font=("微软雅黑", 10, "underline"))
        author_label.grid(row=0, column=0, sticky="w", padx=4, pady=4)
        author_label.bind("<Button-1>", lambda e: webbrowser.open("https://gitee.com/Tloml-Starry"))
        # 交流群超链接
        group_label = tk.Label(frame, text="交流群（点击加入）", fg="#3366cc", cursor="hand2", font=("微软雅黑", 10, "underline"))
        group_label.grid(row=1, column=0, sticky="w", padx=4, pady=4)
        group_label.bind("<Button-1>", lambda e: webbrowser.open("https://qm.qq.com/q/XVf2HjGJgK"))
        # 其它说明
        ttk.Label(frame, text="本程序完全免费，仅供学习交流，严禁商用.").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="右键曲谱可以收藏曲谱，方便下次演奏.").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        # 热键说明区
        frame = ttk.LabelFrame(parent or self.root, text="热键说明", padding=14)
        frame.pack(pady=10, fill="x", padx=8)
        ttk.Label(frame, text="开始/继续:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(frame, textvariable=self.hotkey_vars['start'], font=("微软雅黑", 10, "bold"), foreground=self.accent).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="停止:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(frame, textvariable=self.hotkey_vars['stop'], font=("微软雅黑", 10, "bold"), foreground=self.accent).grid(row=2, column=1, sticky="w", padx=4, pady=4)

    def get_all_music_files(self):
        return [f for f in os.listdir(SHEET_MUSIC_DIR) if f.endswith('.json')]

    def refresh_music_listbox(self):
        # 根据当前tab显示全部或收藏
        tab = getattr(self, 'current_music_tab', None)
        if tab and getattr(self, 'music_tabs', None):
            if self.current_music_tab.get() == "收藏曲谱":
                files = [f for f in self.filtered_music_files if f in self.favorites]
            else:
                files = self.filtered_music_files or []
        else:
            files = self.filtered_music_files or []
        self.music_listbox.delete(0, tk.END)
        # 只显示文件名（带.json），不做display_name截断，保证索引一一对应
        for f in files:
            self.music_listbox.insert(tk.END, f)
        if files:
            self.music_listbox.selection_set(0)
            self.update_song_info(files[0])
        else:
            self.update_song_info(None)

    def on_search(self, event=None):
        keyword = self.search_var.get().lower()
        if not keyword:
            self.filtered_music_files = self.all_music_files.copy()
        else:
            self.filtered_music_files = [f for f in self.all_music_files if keyword in f.lower()]
        self.refresh_music_listbox()

    def on_listbox_select(self, event=None):
        sel = self.music_listbox.curselection()
        if sel:
            filename = self.filtered_music_files[sel[0]]
            self.update_song_info(filename)
            self.status_var.set(f"已选择乐谱: {filename}")

    def refresh_music_list(self):
        # 兼容旧接口，实际不再用
        self.all_music_files = self.get_all_music_files() or []
        self.filtered_music_files = self.all_music_files.copy() if self.all_music_files else []
        self.refresh_music_listbox()

    def update_song_info(self, filename):
        # 页面信息区展示选中曲谱详细信息
        import os
        import json
        if not filename:
            self.music_info_vars['filename'].set("")
            self.music_info_vars['name'].set("")
            self.music_info_vars['author'].set("")
            self.music_info_vars['transcribedBy'].set("")
            return
        self.music_info_vars['filename'].set(filename)
        path = os.path.join(SHEET_MUSIC_DIR, filename)  # SHEET_MUSIC_DIR已用resource_path
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'utf-16', 'utf-16-le', 'utf-16-be']
        data = None
        last_err = None
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    data = json.load(f)
                break
            except Exception as e:
                last_err = e
                data = None
        if data is None:
            self.music_info_vars['name'].set('')
            self.music_info_vars['author'].set('')
            self.music_info_vars['transcribedBy'].set('')
            return
        meta = {}
        # 兼容多种结构
        if isinstance(data, dict):
            meta = data
        elif isinstance(data, list) and len(data) > 0:
            # 优先第一个元素
            if isinstance(data[0], dict):
                meta = data[0]
            else:
                for item in data:
                    if isinstance(item, dict) and ('songName' in item or 'name' in item):
                        meta = item
                        break
        # 兼容不同字段名
        name = meta.get('songName') or meta.get('name') or ''
        author = meta.get('author') or ''
        transcribed = meta.get('transcribedBy') or meta.get('transcriber') or ''
        self.music_info_vars['name'].set(name)
        self.music_info_vars['author'].set(author)
        self.music_info_vars['transcribedBy'].set(transcribed)

    def start_play(self):
        # 三态约束：从头开始必须处于"未播放"态，否则异常（依据 1.md）
        if self.player.state != PlaybackState.STOPPED:
            self.status_var.set("请先停止当前演奏（F7）再重新开始")
            return
        if not self.check_and_set_game_window():
            return
        if not self.load_music():
            return
        # 把当前可调参数注入播放器
        self.player.speed = self.speed
        self.player.simulate = self.simulate
        self.player.miss_prob = self.miss_prob
        self.key_controller.set_debug(self.debug)
        # 从头开始：清零进度显示，避免残留上一次的进度
        self._progress_frac = 0.0
        if getattr(self, 'progress_bar', None) is not None:
            self.progress_bar['value'] = 0
        if not self.player.start(self.notes_by_time, self.sorted_times):
            return
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("演奏中... F11 暂停 / F7 停止")

    def stop_play(self):
        if self.player:
            self.player.stop()
        # 解除游戏窗口置顶（若此前被本程序置顶），恢复正常桌面层级
        self._release_game_topmost()
        self.elapsed_time_var.set("0:00")
        self._progress_frac = 0.0
        if getattr(self, 'progress_bar', None) is not None:
            self.progress_bar['value'] = 0
        self.status_var.set("已停止，点击开始或按F5重新演奏")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def toggle_play_pause(self):
        """F11：播放中 <-> 暂停中 切换。STOPPED 态忽略（需点播放按钮从头开始）。"""
        if not self.player:
            return
        if self.player.state == PlaybackState.PLAYING:
            self.player.pause()
            self.status_var.set("已暂停（F11 继续 / F7 停止）")
        elif self.player.state == PlaybackState.PAUSED:
            self.player.resume()
            self.status_var.set("演奏中... F11 暂停 / F7 停止")

    # ---------- 播放内核回调 ----------
    def _on_status(self, msg):
        self.status_var.set(msg)

    def _on_elapsed(self, sec):
        m, s = divmod(int(sec), 60)
        self.elapsed_time_var.set(f"{m}:{s:02d}")

    def _on_total(self, sec):
        m, s = divmod(int(sec), 60)
        self.total_time_var.set(f"{m}:{s:02d}")

    def _on_progress(self, frac):
        # 仅在播放线程内记录进度值，真正的 UI 更新交给主线程的 _refresh_progress_ui。
        self._progress_frac = max(0.0, min(1.0, frac))

    def _refresh_progress_ui(self):
        """在 Tk 主线程内定时刷新进度条与百分比文本（线程安全）。

        播放内核在独立线程里只更新 self._progress_frac；这里以固定间隔读取该值
        并驱动进度条，避免跨线程直接操作 Tk 组件导致的不稳定。
        """
        try:
            frac = max(0.0, min(1.0, getattr(self, '_progress_frac', 0.0)))
            if getattr(self, 'progress_bar', None) is not None:
                self.progress_bar['value'] = int(frac * 1000)
            if getattr(self, 'progress_percent_var', None) is not None:
                self.progress_percent_var.set(f"{int(frac * 100)}%")
        except Exception:
            pass
        # 约 20fps 刷新，兼顾流畅度与开销
        self.root.after(50, self._refresh_progress_ui)

    def _debug_log(self, msg):
        # 调试模式日志：打印并保留最近若干条，供后续调试面板使用
        print(msg)
        self.debug_logs.append(msg)
        if len(self.debug_logs) > 200:
            self.debug_logs = self.debug_logs[-200:]

    def check_and_set_game_window(self):
        # 查找进程名为 'Sky' 或 '光遇' 的窗口，优先 'Sky'
        # 注：进程检测基于进程名，与游戏安装目录（如 Z:\FeverApps\sky）无关
        current_pid = os.getpid()
        candidates = []

        def enum_windows_callback(hwnd, result):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                tid, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    proc = psutil.Process(pid)
                    name = proc.name()
                    title = win32gui.GetWindowText(hwnd)
                    if not is_sky_game_window_identity(name, title, pid, current_pid):
                        return
                    name_lower = (name or "").lower()
                    title_lower = (title or "").lower()
                    # 精确进程名优先；title 兜底用于进程名被启动器包装的情况。
                    score = 2 if name_lower in SKY_PROCESS_NAMES or "光遇" in name else 1
                    if title_lower == "sky" or "光遇" in title:
                        score = max(score, 1)
                    result.append((score, hwnd))
                except Exception:
                    pass
        win32gui.EnumWindows(enum_windows_callback, candidates)
        hwnd = max(candidates, default=(0, None), key=lambda item: item[0])[1]
        if hwnd:
            self._game_hwnd = hwnd
            self._bring_window_to_front(hwnd)
            return True
        else:
            messagebox.showwarning("未检测到游戏", "未找到进程名为 'Sky' 或 '光遇' 的游戏窗口，请先打开游戏！")
            return False

    def _bring_window_to_front(self, hwnd):
        """将游戏窗口恢复并抢到键盘焦点（含 AttachThreadInput 兜底）。

        仅置顶 (TOPMOST) 不够——游戏要的是「键盘焦点」。keyboard / pyautogui 的
        按键事件只发往当前前台窗口；若焦点仍留在本程序窗口，游戏便收不到按键。
        这里先用 SetForegroundWindow，失败再用 AttachThreadInput 把本线程输入
        桥接到游戏线程后再抢前台，最大化把焦点交还给游戏的成功率。
        """
        # 若处于最小化状态先恢复
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        except Exception:
            pass
        # 置顶确保可见（演奏期间保持，stop 时解除）
        try:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        except Exception:
            pass
        # 直接尝试抢前台
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        # 兜底：前台仍不是游戏窗口时，桥接输入线程再抢一次
        if win32gui.GetForegroundWindow() != hwnd:
            foreground = win32gui.GetForegroundWindow()
            if foreground:
                try:
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
        # 仍拿不到焦点 → 提示用户手动点一下游戏窗口
        if win32gui.GetForegroundWindow() != hwnd:
            try:
                self.root.after(0, lambda: messagebox.showwarning(
                    "焦点切换失败",
                    "无法自动将游戏窗口置于前台（可能被系统限制）。\n请手动点击一下游戏窗口，再按 F5 / 开始演奏。"))
            except Exception:
                pass

    def _release_game_topmost(self):
        """演奏结束后解除游戏窗口置顶，恢复正常桌面层级。"""
        hwnd = getattr(self, '_game_hwnd', None)
        if hwnd:
            try:
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            except Exception:
                pass

    def load_music(self):
        sel = self.music_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先选择乐谱！")
            return False
        selected = self.filtered_music_files[sel[0]]
        path = os.path.join(SHEET_MUSIC_DIR, selected)  # SHEET_MUSIC_DIR已用resource_path
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'utf-16', 'utf-16-le', 'utf-16-be']
        last_err = None
        music_json = None
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    music_json = json.load(f)
                break
            except Exception as e:
                last_err = e
                music_json = None
        if music_json is None:
            messagebox.showerror("错误", f"乐谱文件解析失败: {last_err}")
            return False
        if isinstance(music_json, list) and 'songNotes' in music_json[0]:
            song_notes = music_json[0]['songNotes']
            # 读取bpm字段，若无则默认120
            self.bpm = music_json[0].get('bpm', 120)
        else:
            messagebox.showerror("错误", "乐谱文件格式不正确，未找到songNotes。")
            return False
        notes_by_time = defaultdict(list)
        for note in song_notes:
            notes_by_time[note['time']].append(note['key'])
        sorted_times = sorted(notes_by_time.keys())
        self.notes_by_time = notes_by_time
        self.sorted_times = sorted_times
        return True

    # ---------- 扒谱（音频 → 乐谱） ----------
    def open_generate_dialog(self):
        """打开文件选择对话框，启动后台线程批量转写音频为乐谱 JSON。

        软限幅到 0/14，保留原曲和弦（不人为叠音）。完成后写入 Sheet Music/ 并刷新列表。
        """
        from tkinter import filedialog
        from transcriber import Transcriber, is_audio_file

        # 检查依赖（numpy/librosa）—— Transcriber 会在 __init__ 内做懒导入 + 错误抛出
        try:
            transcriber = Transcriber()
        except Exception as e:
            messagebox.showerror(
                "依赖缺失",
                f"无法启动扒谱功能：\n{e}\n\n请先运行: pip install librosa numpy soundfile",
            )
            return

        files = filedialog.askopenfilenames(
            title="选择要转写的音频文件",
            filetypes=[("音频文件", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac"), ("全部", "*.*")],
        )
        if not files:
            return

        # 进度窗口
        win = tk.Toplevel(self.root)
        win.title("生成乐谱")
        win.geometry("520x220")
        win.transient(self.root)
        win.grab_set()
        ttk.Label(win, text="正在转写音频 → 乐谱 JSON ...", font=("微软雅黑", 10, "bold"), foreground=self.accent).pack(pady=(14, 6))
        info_var = tk.StringVar(value="准备中…")
        ttk.Label(win, textvariable=info_var, font=("微软雅黑", 9)).pack(pady=(0, 6), padx=10, anchor="w")
        bar = ttk.Progressbar(win, mode="determinate", style='Modern.Horizontal.TProgressbar', maximum=1000, value=0)
        bar.pack(fill="x", padx=14, pady=(2, 6))
        detail_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=detail_var, font=("微软雅黑", 8), foreground="#888").pack(pady=(0, 8), padx=10, anchor="w")
        cancel_flag = {"cancel": False}

        def on_cancel():
            cancel_flag["cancel"] = True
            info_var.set("正在取消…")

        ttk.Button(win, text="取消", command=on_cancel).pack(pady=(4, 8))

        def progress_cb(filename, frac, status):
            # 转到 Tk 主线程
            def _apply():
                if cancel_flag["cancel"]:
                    return
                bar["value"] = int(max(0.0, min(1.0, frac)) * 1000)
                info_var.set(status)
                if filename:
                    detail_var.set(os.path.basename(filename))
                else:
                    detail_var.set("")
            try:
                win.after(0, _apply)
            except Exception:
                pass

        def worker():
            try:
                # 用本地 list 防止被 Cancel 修改（iterable 一次性）
                files_list = list(files)
                # 用户取消：transcriber.run 不支持逐文件中断，最简做法：等本批跑完，UI 层关进度窗口
                transcriber.run(
                    files_list,
                    output_dir=SHEET_MUSIC_DIR,
                    progress_cb=progress_cb,
                )
            except Exception as e:
                win.after(0, lambda: messagebox.showerror("转写失败", str(e)))
            finally:
                def _finish():
                    # 通知主线程刷新列表（schedule_music_dir_watch 每秒会扫一次，但这里立刻刷更直观）
                    try:
                        self.all_music_files = self.get_all_music_files() or []
                        self.filtered_music_files = self.all_music_files.copy()
                        self.on_search()
                        self.last_music_files = set(self.all_music_files)
                    except Exception:
                        pass
                    # 计算本次成功数
                    ok_n = 0
                    for fp in files:
                        out = os.path.join(SHEET_MUSIC_DIR,
                                           os.path.splitext(os.path.basename(fp))[0] + ".json")
                        if os.path.exists(out):
                            ok_n += 1
                    try:
                        win.destroy()
                    except Exception:
                        pass
                    messagebox.showinfo(
                        "生成完成",
                        f"已生成 {ok_n}/{len(files)} 份乐谱到 Sheet Music/ 文件夹。\n"
                        f"点击列表中对应文件即可开始演奏。",
                    )
                try:
                    win.after(0, _finish)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self):
        # 退出软件前确保处于"未播放"态并释放可能按住的按键
        if getattr(self, 'player', None):
            self.player.stop()
        # 保存窗口大小和位置
        try:
            geo = self.root.geometry()
            size_pos = geo.split('+')
            size = size_pos[0].split('x')
            width, height = int(size[0]), int(size[1])
            x, y = int(size_pos[1]), int(size_pos[2])
            cfg = {'width': width, 'height': height, 'x': x, 'y': y}
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f)
        except Exception:
            pass
        self.root.destroy()

    def schedule_music_dir_watch(self):
        current_files = set(self.get_all_music_files())
        if current_files != self.last_music_files:
            self.all_music_files = list(current_files)
            self.filtered_music_files = self.all_music_files.copy()
            self.on_search()  # 保持搜索关键字过滤
            self.last_music_files = current_files
        self.root.after(1000, self.schedule_music_dir_watch)

    def bind_hotkeys(self):
        import keyboard
        # 先解绑，防止重复注册
        try:
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        # 注册开始、播放/暂停、停止热键
        try:
            keyboard.add_hotkey(self.hotkeys['start'], lambda: self.start_play())
            keyboard.add_hotkey('F11', lambda: self.toggle_play_pause())
            keyboard.add_hotkey(self.hotkeys['stop'], lambda: self.stop_play())
        except Exception as e:
            messagebox.showwarning("热键注册失败", f"全局热键注册失败，可能需要以管理员身份运行。\n详细信息：{e}")

    def on_music_tab_changed(self):
        # 分页切换时刷新乐谱列表
        self.refresh_music_listbox()

    def on_music_listbox_right_click(self, event):
        """
        右键菜单：仅保留收藏/取消收藏
        """
        idx = self.music_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.filtered_music_files or []):
            return
        self.music_listbox.selection_clear(0, tk.END)
        self.music_listbox.selection_set(idx)
        filename = self.filtered_music_files[idx]
        menu = tk.Menu(self.music_listbox, tearoff=0)
        # 只保留收藏/取消收藏
        if filename in self.favorites:
            menu.add_command(label="取消收藏", command=lambda: self.toggle_favorite(filename))
        else:
            menu.add_command(label="收藏", command=lambda: self.toggle_favorite(filename))
        menu.tk_popup(event.x_root, event.y_root)

    def toggle_favorite(self, filename):
        """
        收藏/取消收藏，并保存到本地
        """
        if filename in self.favorites:
            self.favorites.remove(filename)
        else:
            self.favorites.add(filename)
        self.save_favorites()
        self.refresh_music_listbox()

    def load_favorites(self):
        """
        加载收藏数据
        """
        import json
        try:
            with open(self.favorite_file, 'r', encoding='utf-8') as f:
                self.favorites = set(json.load(f))
        except Exception:
            self.favorites = set()

    def save_favorites(self):
        """
        保存收藏数据
        """
        import json
        try:
            with open(self.favorite_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.favorites), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', font=('微软雅黑', 10))
    app = MusicGUI(root)
    app.bind_hotkeys()
    root.mainloop() 
