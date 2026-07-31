import os
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import webbrowser

from audio_preview import AudioPreviewController
from key_controller import KeyController, note_to_key
from player import MusicPlayer, PlaybackState
from playlist_store import PlaybackSession, PlaylistStore
from score_loader import ScoreValidationError, load_score, summarize_meta
from score_overlay import ScoreOverlay
from window_focus import (
    bring_window_to_front,
    describe_foreground_window,
    describe_window,
    find_sky_game_window,
    is_admin,
    is_sky_game_window_identity,
    release_topmost,
    relaunch_as_admin_if_needed,
    switch_to_english_input,
)

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

class MusicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SkyAutoMusic 自动弹琴")
        # 读取窗口配置
        win_w, win_h = 760, 600
        x, y = None, None
        cfg = {}
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
        self.root.minsize(620, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.config = cfg
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
        self.default_hotkeys = {
            'start': 'F5',
            'previous': 'F6',
            'stop': 'F7',
            'next': 'F8',
            'overlay_lock': 'F10',
            'toggle_pause': 'F11',
        }
        self.hotkeys = self.default_hotkeys.copy()
        self.hotkey_vars = {k: tk.StringVar(value=v) for k, v in self.hotkeys.items()}
        self.status_var = tk.StringVar(value="请选择乐谱并点击开始演奏")
        self.elapsed_time_var = tk.StringVar(value="0:00")
        self.total_time_var = tk.StringVar(value="0:00")
        self.hotkey_status_var = tk.StringVar(value="未注册")
        self.game_window_var = tk.StringVar(value="未检测")
        self.foreground_window_var = tk.StringVar(value="未检测")
        self.admin_status_var = tk.StringVar(value="是" if is_admin() else "否")
        self.overlay_status_var = tk.StringVar(value="未显示")
        self.music_count_var = tk.StringVar(value="0 首")
        self.music_info_vars = {
            'filename': tk.StringVar(),
            'path': tk.StringVar(),
            'name': tk.StringVar(),
            'author': tk.StringVar(),
            'transcribedBy': tk.StringVar(),
            'duration': tk.StringVar(value="--"),
            'bpm': tk.StringVar(value="--"),
            'noteCount': tk.StringVar(value="--"),
        }
        self.filtered_music_files = []  # 先初始化，防止后续方法引用时报错
        self.visible_music_files = []
        self.favorites = set()  # 收藏的乐谱文件名集合，可持久化
        self.favorite_file = resource_path('favorites.json')  # 用resource_path，兼容打包
        self.load_favorites()  # 启动时加载收藏
        self.playlist_store = PlaylistStore(resource_path("playlist.json"))
        self.playlist_store.load(self.get_all_music_files())
        self.debug_logs = []
        self._progress_frac = 0.0
        self._elapsed_sec = 0.0
        self._current_note_info = (-1, None, [])
        self._game_hwnd = None  # 当前已置顶/聚焦的游戏窗口句柄，stop 时解除置顶
        self._active_mode = None  # None / game / preview_loading / preview / transition
        self._preview_loading = False
        self._preview_generation = 0
        self._transition_generation = 0
        self._playback_session = None
        self._playing_filename = None
        self._pending_selection_filename = None
        self._pending_selection_changed = False
        self._programmatic_selection = False
        self._transcription_dialog = None
        # 统一按键抽象层（调试模式、可切换映射的前置能力）
        # 输入方式：auto / interception(驱动级) / keyboard，默认优先驱动级
        # 需在 create_widgets 之前创建，诊断页 UI 会读取其后端列表与状态。
        self.input_method = self.config.get('input_method', 'auto')
        self.key_controller = KeyController(
            mapping=note_to_key, debug=False, log_func=self._debug_log,
            backend=self.input_method)
        self.create_widgets()
        # 关键：初始化后立即加载乐谱列表并刷新
        self.all_music_files = self.get_all_music_files() or []
        self.filtered_music_files = self.all_music_files.copy()
        self.refresh_music_listbox()
        # 播放内核（三态状态机），回调绑定到本类方法
        self.player = MusicPlayer(
            key_controller=self.key_controller,
            update_status=self._on_status,
            update_elapsed=self._on_elapsed,
            update_total=self._on_total,
            update_progress=self._on_progress,
            update_note=self._on_note,
            update_finished=self._on_finished,
        )
        self.audio_preview = AudioPreviewController(
            on_error=self._on_preview_audio_error,
        )
        self.preview_player = MusicPlayer(
            key_controller=self.audio_preview,
            update_status=self._on_preview_status,
            update_elapsed=self._on_elapsed,
            update_total=self._on_total,
            update_progress=self._on_progress,
            update_note=self._on_note,
            update_finished=self._on_preview_finished,
        )
        self._set_playback_controls(None)
        self.overlay = ScoreOverlay(
            self.root,
            geometry=self.config.get("overlay_geometry"),
            locked=self.config.get("overlay_locked", True),
            on_geometry_changed=self._on_overlay_geometry_changed,
            log_func=self._debug_log,
        )
        self.music_data = None
        self.notes_by_time = None
        self.sorted_times = None
        # 播放相关可调参数（后续 UI 注入，这里给默认值）
        self.debug = False
        self.speed = 1.0
        self.simulate = False
        self.miss_prob = 0.03
        self.last_music_files = set(self.all_music_files or [])
        self.schedule_music_dir_watch()
        # 启动进度条定时刷新（主线程驱动，读取播放线程写入的 _progress_frac）
        self._refresh_progress_ui()
        self._refresh_diagnostics()

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
        diagnostics_tab = ttk.Frame(notebook)
        notebook.add(diagnostics_tab, text="诊断")
        # 播放Tab内容（响应式双栏布局）
        main_frame = ttk.Frame(play_tab)
        main_frame.pack(fill="both", expand=True, padx=14, pady=14)
        # Keep the two columns in a fixed ratio. Without a shared uniform
        # group, a long selected title changes the right column's requested
        # width and makes the score list visibly jump.
        main_frame.columnconfigure(
            0, weight=2, minsize=250, uniform="main_columns"
        )
        main_frame.columnconfigure(
            1, weight=3, minsize=330, uniform="main_columns"
        )
        main_frame.rowconfigure(0, weight=1)
        # ====== 右侧主控区 ======
        center_frame = ttk.Frame(main_frame)
        center_frame.grid(row=0, column=1, sticky="nswe", padx=(14, 0))

        # ====== 曲谱信息展示区（右侧，按钮组上方） ======
        # 歌名、作者、制谱人在上方大字号高亮，文件名在下方小字号
        self.music_info_frame = ttk.LabelFrame(center_frame, text="曲谱信息", padding=12)
        self.music_info_frame.pack(fill="x", pady=(0, 16), anchor="n")
        self.music_info_frame.columnconfigure(1, weight=1)
        # 歌名
        ttk.Label(self.music_info_frame, text="歌名:", width=7, anchor="e").grid(row=0, column=0, sticky="e", padx=(0, 8), pady=(0, 5))
        self.music_name_value_label = ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['name'], anchor="w", font=("微软雅黑", 12, "bold"), foreground=self.accent, wraplength=260)
        self.music_name_value_label.grid(row=0, column=1, sticky="we", pady=(0, 5))
        # 作者
        ttk.Label(self.music_info_frame, text="作者:", width=7, anchor="e").grid(row=1, column=0, sticky="e", padx=(0, 8), pady=5)
        self.music_author_value_label = ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['author'], anchor="w", font=("微软雅黑", 11, "bold"), foreground=self.accent, wraplength=260)
        self.music_author_value_label.grid(row=1, column=1, sticky="we", pady=5)
        # 制谱人
        ttk.Label(self.music_info_frame, text="制谱:", width=7, anchor="e").grid(row=2, column=0, sticky="e", padx=(0, 8), pady=5)
        self.music_transcriber_value_label = ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['transcribedBy'], anchor="w", font=("微软雅黑", 11, "bold"), foreground=self.accent, wraplength=260)
        self.music_transcriber_value_label.grid(row=2, column=1, sticky="we", pady=5)
        # 播放指标
        ttk.Label(self.music_info_frame, text="总时长:", width=7, anchor="e").grid(row=3, column=0, sticky="e", padx=(0, 8), pady=3)
        ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['duration'], anchor="w").grid(row=3, column=1, sticky="we", pady=3)
        ttk.Label(self.music_info_frame, text="BPM:", width=7, anchor="e").grid(row=4, column=0, sticky="e", padx=(0, 8), pady=3)
        ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['bpm'], anchor="w").grid(row=4, column=1, sticky="we", pady=3)
        ttk.Label(self.music_info_frame, text="音符数:", width=7, anchor="e").grid(row=5, column=0, sticky="e", padx=(0, 8), pady=3)
        ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['noteCount'], anchor="w").grid(row=5, column=1, sticky="we", pady=3)
        # 文件名
        ttk.Label(self.music_info_frame, text="文件名:", width=7, anchor="e").grid(row=6, column=0, sticky="e", padx=(0, 8), pady=(8, 0))
        self.music_filename_value_label = ttk.Label(self.music_info_frame, textvariable=self.music_info_vars['filename'], anchor="w", font=("微软雅黑", 9), foreground="#888", wraplength=260)
        self.music_filename_value_label.grid(row=6, column=1, sticky="we", pady=(8, 0))
        # ====== 左侧乐谱区 ======
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nswe")

        # ====== 乐谱分页按钮（全部/收藏/播放列表） ======
        # 可自定义：tab_names 可扩展更多分页
        self.music_tabs = ["全部曲谱", "收藏曲谱", "播放列表"]
        self.current_music_tab = tk.StringVar(value=self.music_tabs[0])
        self.music_tab_buttons = {}
        tab_frame = ttk.Frame(left_frame)
        tab_frame.pack(fill="x", pady=(0, 4))
        for name in self.music_tabs:
            label = f"播放列表 ({len(self.playlist_store.items)})" if name == "播放列表" else name
            btn = ttk.Radiobutton(tab_frame, text=label, value=name, variable=self.current_music_tab,
                                  command=self.on_music_tab_changed, style="Toolbutton")
            btn.pack(side="left", padx=2)
            self.music_tab_buttons[name] = btn

        # ====== 搜索栏 ======
        search_header = ttk.Frame(left_frame)
        search_header.pack(fill="x", pady=(4, 2))
        ttk.Label(search_header, text="搜索乐谱", font=("微软雅黑", 10, "bold"), foreground=self.accent, anchor="w").pack(side="left")
        ttk.Label(search_header, textvariable=self.music_count_var, font=("微软雅黑", 9), foreground="#888", anchor="e").pack(side="right")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(left_frame, textvariable=self.search_var, font=("微软雅黑", 10))
        self.search_entry.pack(fill="x", pady=(0, 8))
        self.search_entry.bind('<KeyRelease>', self.on_search)

        # ====== 乐谱列表区 ======
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill="both", expand=True)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.music_listbox = tk.Listbox(
            list_frame,
            font=("微软雅黑", 10),
            activestyle="none",
            borderwidth=1,
            relief="solid",
            exportselection=False,
            selectbackground=self.accent,
            selectforeground="#FFFFFF",
        )
        self.music_listbox.grid(row=0, column=0, sticky="nswe")
        self.music_vscroll = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.music_listbox.yview,
        )
        self.music_vscroll.grid(row=0, column=1, sticky="ns")
        self.music_hscroll = ttk.Scrollbar(
            list_frame,
            orient="horizontal",
            command=self.music_listbox.xview,
        )
        self.music_hscroll.grid(row=1, column=0, sticky="ew")
        self.music_listbox.configure(
            yscrollcommand=self.music_vscroll.set,
            xscrollcommand=self.music_hscroll.set,
        )
        self.music_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        self.music_listbox.bind('<Button-3>', self.on_music_listbox_right_click)  # 右键菜单
        # 播放列表操作区：普通列表可添加，播放列表内可排序和移除。
        self.add_playlist_frame = ttk.Frame(left_frame)
        self.add_to_playlist_btn = ttk.Button(
            self.add_playlist_frame,
            text="加入播放列表",
            command=self.add_selected_to_playlist,
            style='Accent.TButton',
        )
        self.add_to_playlist_btn.pack(fill="x")
        self.manage_playlist_frame = ttk.Frame(left_frame)
        for column in range(3):
            self.manage_playlist_frame.columnconfigure(column, weight=1)
        self.playlist_up_btn = ttk.Button(
            self.manage_playlist_frame, text="上移",
            command=lambda: self.move_selected_playlist_item(-1))
        self.playlist_up_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.playlist_down_btn = ttk.Button(
            self.manage_playlist_frame, text="下移",
            command=lambda: self.move_selected_playlist_item(1))
        self.playlist_down_btn.grid(row=0, column=1, sticky="ew", padx=3)
        self.playlist_remove_btn = ttk.Button(
            self.manage_playlist_frame, text="移除",
            command=self.remove_selected_from_playlist)
        self.playlist_remove_btn.grid(row=0, column=2, sticky="ew", padx=(3, 0))
        self._update_playlist_action_visibility()
        self.refresh_music_listbox()
        self.tooltip = None
        # 播放时长显示（紧凑居中）
        time_frame = ttk.Frame(center_frame)
        time_frame.pack(pady=(6, 10), fill="x")
        time_inner = ttk.Frame(time_frame)
        time_inner.pack(anchor="center")
        ttk.Label(time_inner, textvariable=self.elapsed_time_var, font=("Consolas", 11, "bold"), foreground=self.accent, width=7, anchor="e").pack(side="left")
        ttk.Label(time_inner, text="/", font=("微软雅黑", 10, "bold"), foreground="#888", width=2, anchor="center").pack(side="left", padx=2)
        ttk.Label(time_inner, textvariable=self.total_time_var, font=("Consolas", 11, "bold"), foreground="#888", width=7, anchor="w").pack(side="left")
        # ====== 弹奏进度条 ======
        # 复用 set_style 中定义的 Modern.Horizontal.TProgressbar 样式，与主窗口 UI 风格统一。
        # maximum=1000 提供更平滑的分辨率；实际值由 _refresh_progress_ui 定时从 _progress_frac 读取更新，
        # 避免在播放线程里直接操作 Tk 组件（线程安全）。
        progress_frame = ttk.Frame(center_frame)
        progress_frame.pack(pady=(0, 14), fill="x")
        self.progress_bar = ttk.Progressbar(
            progress_frame, mode="determinate",
            style='Modern.Horizontal.TProgressbar', maximum=1000, value=0)
        self.progress_bar.pack(fill="x")
        # 进度百分比文本，居中显示，弱化配色与整体风格协调
        self.progress_percent_var = tk.StringVar(value="0%")
        ttk.Label(progress_frame, textvariable=self.progress_percent_var,
                  font=("Consolas", 9), foreground="#888",
                  background=self.bg_color, anchor="center").pack(fill="x", pady=(2, 0))
        # 操作按钮组
        btn_frame = ttk.Frame(center_frame)
        btn_frame.pack(pady=8, fill="x")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        self.start_btn = ttk.Button(btn_frame, text="游戏演奏 (F5)", command=self.start_play, style='Accent.TButton')
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.stop_btn = ttk.Button(btn_frame, text="停止 (F7)", command=self.stop_play, state="disabled", style='Accent.TButton')
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.preview_btn = ttk.Button(btn_frame, text="本地试听", command=self.start_preview, style='Accent.TButton')
        self.preview_btn.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=6)
        # 扒谱（音频→乐谱）入口
        self.generate_btn = ttk.Button(btn_frame, text="生成乐谱", command=self.open_generate_dialog, style='Accent.TButton')
        self.generate_btn.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.previous_btn = ttk.Button(
            btn_frame, text="上一首 (F6)", command=self.previous_track)
        self.previous_btn.grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.next_btn = ttk.Button(
            btn_frame, text="下一首 (F8)", command=self.next_track)
        self.next_btn.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=6)
        # 状态栏
        self.status_label = ttk.Label(center_frame, textvariable=self.status_var, anchor="center", justify="center", font=("微软雅黑", 10, "bold"), background=self.bg_color, foreground=self.accent, wraplength=380)
        self.status_label.pack(pady=(10, 6), fill="x")
        center_frame.bind("<Configure>", self._on_center_frame_configure)
        # 设置Tab内容
        self.create_hotkey_settings(parent=settings_tab)
        self.create_diagnostics_tab(parent=diagnostics_tab)

    def _on_center_frame_configure(self, event):
        wraplength = max(120, event.width - 120)
        for label in (
            self.music_name_value_label,
            self.music_author_value_label,
            self.music_transcriber_value_label,
            self.music_filename_value_label,
        ):
            label.configure(wraplength=wraplength)
        self.status_label.configure(wraplength=max(160, event.width - 24))

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
        hotkey_rows = [
            ("游戏演奏:", "start"),
            ("上一首:", "previous"),
            ("停止:", "stop"),
            ("下一首:", "next"),
            ("暂停/继续:", "toggle_pause"),
            ("覆盖层移动/锁定:", "overlay_lock"),
        ]
        for row, (label, name) in enumerate(hotkey_rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=4, pady=4)
            ttk.Label(
                frame,
                textvariable=self.hotkey_vars[name],
                font=("微软雅黑", 10, "bold"),
                foreground=self.accent,
            ).grid(row=row, column=1, sticky="w", padx=4, pady=4)

    def create_diagnostics_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="运行状态", padding=14)
        frame.pack(padx=10, pady=10, fill="x")
        rows = [
            ("管理员权限:", self.admin_status_var),
            ("热键状态:", self.hotkey_status_var),
            ("游戏窗口:", self.game_window_var),
            ("前台窗口:", self.foreground_window_var),
            ("覆盖层:", self.overlay_status_var),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label, width=12, anchor="e").grid(row=row, column=0, sticky="e", padx=4, pady=4)
            ttk.Label(frame, textvariable=var, anchor="w", wraplength=430).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        frame.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(btn_frame, text="刷新诊断", command=lambda: self._refresh_diagnostics(schedule=False), style='Accent.TButton').pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="切换覆盖层锁定 (F10)", command=self.toggle_overlay_lock, style='Accent.TButton').pack(side="left")

        # ====== 键盘输入方式选择（虚拟HID/驱动级键盘） ======
        input_frame = ttk.LabelFrame(parent, text="键盘输入方式", padding=10)
        input_frame.pack(padx=10, pady=(0, 8), fill="x")
        # 名称 <-> 标签 映射，用于下拉框显示与回写
        self._input_method_options = []
        self._input_label_to_name = {}
        for _name, _label in self.key_controller.backend_options():
            self._input_method_options.append(_label)
            self._input_label_to_name[_label] = _name
        self._input_name_to_label = {n: l for l, n in self._input_label_to_name.items()}
        # 当前生效后端对应的标签作为初始值
        _cur_label = self._input_name_to_label.get(self.key_controller.get_backend())
        self.input_method_var = tk.StringVar(value=_cur_label or self._input_method_options[0])
        ttk.Label(input_frame, text="输入方式:", width=10, anchor="e").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.input_method_combo = ttk.Combobox(
            input_frame, textvariable=self.input_method_var,
            values=self._input_method_options, state="readonly", width=28)
        self.input_method_combo.grid(row=0, column=1, sticky="we", padx=4, pady=4)
        self.input_method_combo.bind("<<ComboboxSelected>>", self._on_input_method_change)
        ttk.Button(input_frame, text="校准驱动级键盘", command=self.calibrate_driver_keyboard, style='Accent.TButton').grid(row=0, column=2, padx=4, pady=4)
        self.input_method_status_var = tk.StringVar(value="")
        ttk.Label(input_frame, textvariable=self.input_method_status_var, anchor="w", wraplength=430, foreground="#888").grid(row=1, column=0, columnspan=3, sticky="we", padx=4, pady=(2, 0))
        input_frame.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(parent, text="最近按键/警告日志", padding=8)
        log_frame.pack(padx=10, pady=8, fill="both", expand=True)
        self.debug_text = tk.Text(log_frame, height=10, wrap="word", font=("Consolas", 9), bg="#FFFFFF", fg="#222", relief="solid", borderwidth=1)
        self.debug_text.pack(fill="both", expand=True)
        self.debug_text.configure(state="disabled")

    def get_all_music_files(self):
        files = [f for f in os.listdir(SHEET_MUSIC_DIR) if f.lower().endswith('.json')]
        return sorted(files, key=str.casefold)

    def _selected_music_filename(self):
        if not getattr(self, "music_listbox", None):
            return None
        selected = self.music_listbox.curselection()
        if not selected or selected[0] >= len(self.visible_music_files):
            return None
        return self.visible_music_files[selected[0]]

    def refresh_music_listbox(self):
        previous = self._selected_music_filename()
        if previous is None:
            previous = self.music_info_vars["filename"].get() or None
        old_yview = self.music_listbox.yview()
        old_top = old_yview[0] if old_yview else 0.0
        old_xview = self.music_listbox.xview()
        old_left = old_xview[0] if old_xview else 0.0
        # 根据当前 tab 显示全部、收藏或保持自定义顺序的播放列表。
        tab = getattr(self, 'current_music_tab', None)
        if tab and getattr(self, 'music_tabs', None):
            current_tab = self.current_music_tab.get()
            if current_tab == "收藏曲谱":
                files = [f for f in self.filtered_music_files if f in self.favorites]
                files = sorted(files, key=str.casefold)
            elif current_tab == "播放列表":
                filtered = set(self.filtered_music_files)
                files = [f for f in self.playlist_store.items if f in filtered]
            else:
                files = sorted(self.filtered_music_files or [], key=str.casefold)
        else:
            files = sorted(self.filtered_music_files or [], key=str.casefold)
        self.music_listbox.delete(0, tk.END)
        self.visible_music_files = list(files)
        self.music_count_var.set(f"{len(files)} 首")
        playlist_tab = getattr(self, "music_tab_buttons", {}).get("播放列表")
        if playlist_tab is not None:
            playlist_tab.config(text=f"播放列表 ({len(self.playlist_store.items)})")
        # 只显示文件名（带.json），不做display_name截断，保证索引一一对应
        for f in files:
            self.music_listbox.insert(tk.END, f)
        if files:
            selected_index = files.index(previous) if previous in files else 0
            self.music_listbox.selection_set(selected_index)
            self.music_listbox.activate(selected_index)
            self.update_song_info(files[selected_index])
            if previous in files:
                self.music_listbox.yview_moveto(old_top)
                self.music_listbox.xview_moveto(old_left)
            else:
                self.music_listbox.see(selected_index)
        else:
            self.update_song_info(None)
        self._update_playlist_action_visibility()
        self._update_playlist_action_states()
        self._update_navigation_buttons()

    def on_search(self, event=None):
        keyword = self.search_var.get().lower()
        if not keyword:
            self.filtered_music_files = self.all_music_files.copy()
        else:
            self.filtered_music_files = [f for f in self.all_music_files if keyword in f.lower()]
        self.refresh_music_listbox()

    def on_listbox_select(self, event=None):
        sel = self.music_listbox.curselection()
        if sel and sel[0] < len(self.visible_music_files):
            filename = self.visible_music_files[sel[0]]
            programmatic = self._programmatic_selection
            readable = self.update_song_info(filename)
            if self._active_mode is not None and not programmatic:
                playing = self._playing_filename or "当前曲目"
                self.status_var.set(f"已选择待播: {filename}；当前播放: {playing}")
            elif self._active_mode is None and readable:
                self.status_var.set(f"已选择乐谱: {filename}")
            elif self._active_mode is None:
                self.status_var.set(f"乐谱不可播放: {filename}")
            self._update_playlist_action_states()
            self._update_navigation_buttons()

    def refresh_music_list(self):
        # 兼容旧接口，实际不再用
        self.all_music_files = self.get_all_music_files() or []
        self.filtered_music_files = self.all_music_files.copy() if self.all_music_files else []
        self.refresh_music_listbox()

    @staticmethod
    def _format_time_value(seconds):
        seconds = max(0, int(float(seconds)))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{sec:02d}"
        return f"{minutes}:{sec:02d}"

    def _read_score_for_filename(self, filename, show_error=False):
        if not filename:
            return None
        path = os.path.join(SHEET_MUSIC_DIR, filename)
        try:
            return load_score(path, valid_keys=self.key_controller.mapping.keys())
        except (ScoreValidationError, OSError) as exc:
            if show_error:
                messagebox.showerror("乐谱校验失败", str(exc))
            else:
                self._debug_log(f"[WARN] 无法读取乐谱 {filename}: {exc}")
            return None

    def _clear_song_info(self, filename="", invalid=False, reset_progress=True):
        self.music_info_vars['filename'].set(filename)
        self.music_info_vars['name'].set("")
        self.music_info_vars['author'].set("")
        self.music_info_vars['transcribedBy'].set("")
        placeholder = "--" if invalid else ""
        self.music_info_vars['duration'].set(placeholder)
        self.music_info_vars['bpm'].set(placeholder)
        self.music_info_vars['noteCount'].set(placeholder)
        if reset_progress:
            self._reset_progress(total_text="--" if invalid else "0:00")

    def _apply_song_info(self, filename, score, reset_progress=True):
        meta = summarize_meta(score["raw"])
        duration_sec = score["duration_ms"] / 1000.0
        self.music_info_vars['filename'].set(filename)
        self.music_info_vars['name'].set(meta.get('name', ''))
        self.music_info_vars['author'].set(meta.get('author', ''))
        self.music_info_vars['transcribedBy'].set(meta.get('transcribedBy', ''))
        self.music_info_vars['duration'].set(self._format_time_value(duration_sec))
        bpm = meta.get('bpm', 120)
        self.music_info_vars['bpm'].set(str(bpm) if bpm not in (None, "") else "--")
        self.music_info_vars['noteCount'].set(str(score["note_count"]))
        if reset_progress:
            self._reset_progress(total_sec=duration_sec)

    def update_song_info(self, filename, force=False, reset_progress=True):
        """展示选中曲谱；播放期间普通选择只记录为停止后的待显示项。"""
        if self._active_mode is not None and not force:
            if not self._programmatic_selection:
                if filename == self._playing_filename:
                    self._pending_selection_filename = None
                    self._pending_selection_changed = False
                else:
                    self._pending_selection_filename = filename
                    self._pending_selection_changed = True
            return True
        if not filename:
            self._clear_song_info(reset_progress=reset_progress)
            return False
        score = self._read_score_for_filename(filename)
        if score is None:
            self._clear_song_info(filename=filename, invalid=True, reset_progress=reset_progress)
            return False
        self._apply_song_info(filename, score, reset_progress=reset_progress)
        return True

    def _playlist_mutation_allowed(self):
        if self._active_mode is None:
            return True
        self.status_var.set("播放期间不能修改播放列表，请先停止（F7）")
        return False

    def add_selected_to_playlist(self):
        filename = self._selected_music_filename()
        if not filename:
            self.status_var.set("请先选择要加入播放列表的乐谱")
            return
        self.add_filename_to_playlist(filename)

    def remove_selected_from_playlist(self):
        filename = self._selected_music_filename()
        if not filename or not self._playlist_mutation_allowed():
            return
        self.remove_filename_from_playlist(filename)

    def move_selected_playlist_item(self, offset):
        filename = self._selected_music_filename()
        if not filename or not self._playlist_mutation_allowed():
            return
        visible_index = self.visible_music_files.index(filename)
        target_visible_index = visible_index + (1 if offset > 0 else -1)
        if target_visible_index < 0 or target_visible_index >= len(self.visible_music_files):
            self.status_var.set("已经到达播放列表边界")
            return
        target_filename = self.visible_music_files[target_visible_index]
        full_offset = (
            self.playlist_store.items.index(target_filename)
            - self.playlist_store.items.index(filename)
        )
        try:
            moved = self.playlist_store.move(filename, full_offset)
        except OSError as exc:
            messagebox.showerror("播放列表保存失败", str(exc))
            return
        if not moved:
            self.status_var.set("已经到达播放列表边界")
            return
        self.refresh_music_listbox()
        self.status_var.set(f"已调整播放顺序: {filename}")

    def _update_playlist_action_visibility(self):
        if not getattr(self, "add_playlist_frame", None):
            return
        if self.current_music_tab.get() == "播放列表":
            self.add_playlist_frame.pack_forget()
            self.manage_playlist_frame.pack(fill="x", pady=(8, 0))
        else:
            self.manage_playlist_frame.pack_forget()
            self.add_playlist_frame.pack(fill="x", pady=(8, 0))

    def _update_playlist_action_states(self):
        if not getattr(self, "add_to_playlist_btn", None):
            return
        locked = self._active_mode is not None
        filename = self._selected_music_filename()
        self.add_to_playlist_btn.config(
            state="disabled" if locked or not filename else "normal")
        in_playlist_view = self.current_music_tab.get() == "播放列表"
        visible_index = (
            self.visible_music_files.index(filename)
            if filename in self.visible_music_files else -1
        )
        self.playlist_up_btn.config(
            state="normal"
            if in_playlist_view and not locked and visible_index > 0 else "disabled")
        self.playlist_down_btn.config(
            state="normal"
            if in_playlist_view and not locked
            and 0 <= visible_index < len(self.visible_music_files) - 1
            else "disabled")
        self.playlist_remove_btn.config(
            state="normal"
            if in_playlist_view and not locked and filename in self.playlist_store.items
            else "disabled")

    def _player_threads_running(self):
        for player in (getattr(self, "player", None), getattr(self, "preview_player", None)):
            thread = getattr(player, "thread", None)
            if thread is not None and thread.is_alive():
                return True
        return False

    def _select_filename_if_visible(self, filename):
        if filename not in self.visible_music_files:
            return
        index = self.visible_music_files.index(filename)
        self._programmatic_selection = True
        self.music_listbox.selection_clear(0, tk.END)
        self.music_listbox.selection_set(index)
        self.music_listbox.activate(index)
        self.music_listbox.see(index)
        try:
            self.root.after_idle(
                lambda: setattr(self, "_programmatic_selection", False))
        except Exception:
            self._programmatic_selection = False

    def _create_playback_session(self, filename, mode):
        source_tab = self.current_music_tab.get()
        return PlaybackSession.create(
            self.visible_music_files,
            filename,
            mode=mode,
            auto_advance=source_tab == "播放列表",
            source_tab=source_tab,
        )

    def _begin_selected_playback(self, mode):
        if self._active_mode is not None:
            self.status_var.set("请先停止当前演奏或试听（F7）")
            return False
        if self._player_threads_running():
            self.status_var.set("播放器正在停止，请稍后重试")
            return False
        filename = self._selected_music_filename()
        if not filename:
            messagebox.showwarning("提示", "请先选择乐谱！")
            return False
        score = self._read_score_for_filename(filename, show_error=True)
        if score is None:
            return False
        try:
            session = self._create_playback_session(filename, mode)
        except ValueError:
            self.status_var.set("当前歌曲不在可见列表中，请重新选择")
            return False

        self._playback_session = session
        self._pending_selection_filename = None
        self._pending_selection_changed = False
        if not self._start_loaded_track(filename, score, mode):
            self._playback_session = None
            self._playing_filename = None
            self._set_playback_controls(None)
            return False
        return True

    def start_play(self):
        self._begin_selected_playback("game")

    def start_preview(self):
        """开始、暂停或继续当前选中 JSON 曲谱的本地试听。"""
        if self._active_mode == "preview" and self.preview_player.state != PlaybackState.STOPPED:
            self.toggle_play_pause()
            return
        self._begin_selected_playback("preview")

    def _apply_loaded_score(self, score):
        self.music_data = score["raw"]
        self.notes_by_time = score["notes_by_time"]
        self.sorted_times = score["sorted_times"]
        self.bpm = score["meta"].get('bpm', 120)
        if score["warnings"]:
            self._debug_log("[WARN] " + "；".join(score["warnings"][:5]))

    def _start_loaded_track(self, filename, score, mode, preserve_pending=False):
        self._apply_loaded_score(score)
        self._playing_filename = filename
        if not preserve_pending:
            self._pending_selection_filename = None
            self._pending_selection_changed = False
            self._select_filename_if_visible(filename)
        elif not self._pending_selection_changed:
            self._select_filename_if_visible(filename)
        self._apply_song_info(filename, score, reset_progress=True)

        if mode == "game":
            self.audio_preview.stop_all()
            if not self.check_and_set_game_window():
                return False
            self._prepare_game_input()
            self.player.speed = self.speed
            self.player.simulate = self.simulate
            self.player.miss_prob = self.miss_prob
            self.key_controller.set_debug(self.debug)
            title = self.music_info_vars['name'].get() or filename
            self.overlay.set_score(self.notes_by_time, self.sorted_times, title=title)
            self.overlay.show(self._game_hwnd)
            self._update_overlay_status()
            self._active_mode = "game"
            if not self.player.start(self.notes_by_time, self.sorted_times):
                self._active_mode = None
                return False
            self._set_playback_controls("game")
            self.status_var.set(
                f"演奏中: {filename}（F6/F8 切歌 / F11 暂停 / F7 停止）")
            return True

        self.audio_preview.stop_all()
        self._preview_generation += 1
        generation = self._preview_generation
        self._preview_loading = True
        self._active_mode = "preview_loading"
        self._set_playback_controls("preview_loading")
        self.status_var.set(f"正在准备试听音色: {filename}")

        if self.audio_preview.prepared:
            self._start_prepared_preview(generation)
            return True

        def progress(completed, total, index):
            self._run_on_ui(
                self._set_preview_prepare_progress,
                generation,
                completed,
                total,
            )

        def worker():
            try:
                self.audio_preview.prepare(progress_callback=progress)
            except Exception as exc:
                self._run_on_ui(self._preview_prepare_failed, generation, exc)
                return
            self._run_on_ui(self._start_prepared_preview, generation)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _set_preview_prepare_progress(self, generation, completed, total):
        if generation != self._preview_generation or not self._preview_loading:
            return
        filename = self._playing_filename or ""
        self.status_var.set(
            f"正在准备试听音色: {filename}（{completed}/{total}）")

    def _start_prepared_preview(self, generation):
        if generation != self._preview_generation or not self._preview_loading:
            return
        self._preview_loading = False
        self.preview_player.speed = self.speed
        self.preview_player.simulate = False
        self._active_mode = "preview"
        if not self.preview_player.start(self.notes_by_time, self.sorted_times):
            self._active_mode = None
            self._playback_session = None
            self._set_playback_controls(None)
            self.status_var.set("无法启动试听，请重试")
            return
        self._set_playback_controls("preview")
        self.status_var.set(
            f"本地试听中: {self._playing_filename}（F6/F8 切歌 / F11 暂停 / F7 停止）")

    def _preview_prepare_failed(self, generation, exc):
        if generation != self._preview_generation:
            return
        self._preview_loading = False
        self._active_mode = None
        self._playback_session = None
        self._playing_filename = None
        self._set_playback_controls(None)
        self.status_var.set("试听音色准备失败")
        messagebox.showerror("无法开始试听", str(exc))

    def _find_playable_session_target(self, offset):
        session = self._playback_session
        if session is None:
            return None
        for index in session.candidate_indexes(offset):
            filename = session.items[index]
            score = self._read_score_for_filename(filename)
            if score is not None:
                return index, filename, score
            self._debug_log(f"[WARN] 跳过不可播放的队列项: {filename}")
        return None

    def previous_track(self):
        self._change_track(-1)

    def next_track(self):
        self._change_track(1)

    def _change_track(self, offset):
        if self._active_mode is None:
            self._move_idle_selection(offset)
            return
        if self._active_mode == "transition":
            self.status_var.set("正在切换歌曲，请稍候")
            return
        target = self._find_playable_session_target(offset)
        if target is None:
            self.status_var.set("已经到达播放序列边界")
            return
        index, filename, score = target

        if self._active_mode == "preview_loading":
            self._playback_session.index = index
            self._apply_loaded_score(score)
            self._playing_filename = filename
            self._pending_selection_filename = None
            self._pending_selection_changed = False
            self._select_filename_if_visible(filename)
            self._apply_song_info(filename, score, reset_progress=True)
            self._update_navigation_buttons()
            self.status_var.set(f"正在准备试听音色: {filename}")
            return

        mode = self._playback_session.mode
        self._begin_track_transition(
            index, filename, score, mode, preserve_pending=False)

    def _move_idle_selection(self, offset):
        if not self.visible_music_files:
            self.status_var.set("当前列表没有可选择的歌曲")
            return
        selected = self.music_listbox.curselection()
        index = selected[0] if selected else 0
        target = index + (1 if offset > 0 else -1)
        if target < 0 or target >= len(self.visible_music_files):
            self.status_var.set("已经到达当前列表边界")
            return
        self.music_listbox.selection_clear(0, tk.END)
        self.music_listbox.selection_set(target)
        self.music_listbox.activate(target)
        self.music_listbox.see(target)
        filename = self.visible_music_files[target]
        self.update_song_info(filename)
        self.status_var.set(f"已选择乐谱: {filename}")
        self._update_navigation_buttons()

    def _begin_track_transition(
            self, index, filename, score, mode, preserve_pending):
        self._transition_generation += 1
        generation = self._transition_generation
        self._preview_generation += 1
        self._preview_loading = False
        self._playback_session.index = index
        self._active_mode = "transition"
        self._set_playback_controls("transition")
        self.status_var.set(f"正在切换到: {filename}")
        self.player.stop()
        self.preview_player.stop()
        if mode == "preview":
            self.audio_preview.stop_all()
        self._wait_for_track_transition(
            generation, filename, score, mode, preserve_pending)

    def _wait_for_track_transition(
            self, generation, filename, score, mode, preserve_pending):
        if generation != self._transition_generation or self._active_mode != "transition":
            return
        if self._player_threads_running():
            self.root.after(
                20,
                self._wait_for_track_transition,
                generation,
                filename,
                score,
                mode,
                preserve_pending,
            )
            return
        if not self._start_loaded_track(
                filename, score, mode, preserve_pending=preserve_pending):
            self.stop_play()
            self.status_var.set(f"无法播放队列歌曲: {filename}")

    def _update_navigation_buttons(self):
        if not getattr(self, "previous_btn", None):
            return
        if self._active_mode == "transition":
            previous_enabled = next_enabled = False
        elif self._active_mode is not None and self._playback_session is not None:
            previous_enabled = self._playback_session.can_move(-1)
            next_enabled = self._playback_session.can_move(1)
        else:
            selected = self.music_listbox.curselection()
            index = selected[0] if selected else -1
            previous_enabled = index > 0
            next_enabled = 0 <= index < len(self.visible_music_files) - 1
        self.previous_btn.config(state="normal" if previous_enabled else "disabled")
        self.next_btn.config(state="normal" if next_enabled else "disabled")

    def _set_playback_controls(self, mode):
        if mode is None:
            self.start_btn.config(state="normal")
            self.preview_btn.config(state="normal", text="本地试听")
            self.stop_btn.config(state="disabled")
        elif mode == "game":
            self.start_btn.config(state="disabled")
            self.preview_btn.config(state="disabled", text="本地试听")
            self.stop_btn.config(state="normal")
        elif mode == "preview_loading":
            self.start_btn.config(state="disabled")
            self.preview_btn.config(state="disabled", text="准备音色...")
            self.stop_btn.config(state="normal")
        elif mode == "preview":
            paused = self.preview_player.state == PlaybackState.PAUSED
            self.start_btn.config(state="disabled")
            self.preview_btn.config(
                state="normal",
                text="继续试听 (F11)" if paused else "暂停试听 (F11)",
            )
            self.stop_btn.config(state="normal")
        elif mode == "transition":
            self.start_btn.config(state="disabled")
            self.preview_btn.config(state="disabled", text="正在切歌...")
            self.stop_btn.config(state="normal")
        self._update_navigation_buttons()
        self._update_playlist_action_states()

    def _reset_progress(self, total_sec=None, total_text=None):
        self.elapsed_time_var.set("0:00")
        if total_text is not None:
            self.total_time_var.set(total_text)
        elif total_sec is not None:
            self.total_time_var.set(self._format_time_value(total_sec))
        else:
            self.total_time_var.set("0:00")
        self._progress_frac = 0.0
        self._elapsed_sec = 0.0
        self._current_note_info = (-1, None, [])
        if getattr(self, "progress_bar", None) is not None:
            self.progress_bar["value"] = 0
        if getattr(self, "progress_percent_var", None) is not None:
            self.progress_percent_var.set("0%")

    def stop_play(self):
        self._transition_generation += 1
        self._preview_generation += 1
        self._preview_loading = False
        if getattr(self, "player", None):
            self.player.stop()
        if getattr(self, "preview_player", None):
            self.preview_player.stop()
        if getattr(self, "audio_preview", None):
            self.audio_preview.stop_all()
        self._release_game_topmost()
        if getattr(self, 'overlay', None):
            self.overlay.hide()
            self._update_overlay_status()

        target = (
            self._pending_selection_filename
            if self._pending_selection_changed
            else self._playing_filename or self._selected_music_filename()
        )
        self._active_mode = None
        self._playback_session = None
        self._playing_filename = None
        self._pending_selection_filename = None
        self._pending_selection_changed = False
        if target:
            self._select_filename_if_visible(target)
            self.update_song_info(target, force=True, reset_progress=True)
        else:
            self._clear_song_info(reset_progress=True)
        self._set_playback_controls(None)
        self.status_var.set("已停止，点击游戏演奏或本地试听")

    def toggle_play_pause(self):
        """F11：暂停或继续当前的游戏演奏/本地试听。"""
        if self._active_mode == "game" and self.player.state == PlaybackState.PLAYING:
            self.player.pause()
            self.status_var.set("已暂停（F11 继续 / F7 停止）")
        elif self._active_mode == "game" and self.player.state == PlaybackState.PAUSED:
            self.player.resume()
            self.status_var.set("演奏中... F11 暂停 / F7 停止")
        elif self._active_mode == "preview" and self.preview_player.state == PlaybackState.PLAYING:
            self.preview_player.pause()
            self.audio_preview.stop_all()
            self._set_playback_controls("preview")
            self.status_var.set("试听已暂停（F11 继续 / F7 停止）")
        elif self._active_mode == "preview" and self.preview_player.state == PlaybackState.PAUSED:
            self.preview_player.resume()
            self._set_playback_controls("preview")
            self.status_var.set("本地试听中... F11 暂停 / F7 停止")

    # ---------- 播放内核回调 ----------
    def _run_on_ui(self, func, *args):
        try:
            self.root.after(0, lambda: func(*args))
        except Exception:
            pass

    def _on_status(self, msg):
        self._run_on_ui(self._apply_mode_status, "game", msg)

    def _on_preview_status(self, msg):
        if msg.startswith("演奏进度"):
            msg = "试听进度" + msg[len("演奏进度"):]
        elif msg == "演奏结束！":
            msg = "试听结束！"
        self._run_on_ui(self._apply_mode_status, "preview", msg)

    def _apply_mode_status(self, mode, msg):
        if self._active_mode == mode:
            self.status_var.set(msg)

    def _on_elapsed(self, sec):
        self._elapsed_sec = max(0.0, float(sec))
        def _apply():
            self.elapsed_time_var.set(self._format_time_value(sec))
        self._run_on_ui(_apply)

    def _on_total(self, sec):
        def _apply():
            self.total_time_var.set(self._format_time_value(sec))
        self._run_on_ui(_apply)

    def _on_progress(self, frac):
        # 仅在播放线程内记录进度值，真正的 UI 更新交给主线程的 _refresh_progress_ui。
        self._progress_frac = max(0.0, min(1.0, frac))

    def _on_note(self, idx, t_ms, notes):
        self._current_note_info = (idx, t_ms, list(notes or []))
        if idx >= 0:
            self._record_log(f"[NOTE] #{idx + 1} {t_ms}ms -> {','.join(notes or [])}", echo=False)

    def _on_finished(self):
        self._run_on_ui(self._handle_playback_finished)

    def _handle_playback_finished(self):
        if self._active_mode != "game":
            return
        self._handle_track_finished("game")

    def _on_preview_finished(self):
        self._run_on_ui(self._handle_preview_finished)

    def _handle_preview_finished(self):
        if self._active_mode != "preview":
            return
        self._handle_track_finished("preview")

    def _handle_track_finished(self, mode):
        session = self._playback_session
        if session is not None and session.auto_advance:
            target = self._find_playable_session_target(1)
            if target is not None:
                index, filename, score = target
                self._begin_track_transition(
                    index,
                    filename,
                    score,
                    mode,
                    preserve_pending=True,
                )
                return
        self._finalize_natural_playback(mode)

    def _finalize_natural_playback(self, mode):
        session = self._playback_session
        queue_finished = bool(session and session.auto_advance)
        last_playing = self._playing_filename
        pending_changed = self._pending_selection_changed
        pending = self._pending_selection_filename

        if mode == "game":
            self._release_game_topmost()
            if getattr(self, 'overlay', None):
                self.overlay.hide()
                self._update_overlay_status()
        self._active_mode = None
        self._playback_session = None
        self._set_playback_controls(None)
        if pending_changed:
            if pending:
                self.update_song_info(pending, force=True, reset_progress=True)
            else:
                self._clear_song_info(reset_progress=True)
        elif last_playing:
            # 保留自然结束后的 100% 进度，仅刷新歌曲信息本身。
            self.update_song_info(last_playing, force=True, reset_progress=False)

        self._playing_filename = None
        self._pending_selection_filename = None
        self._pending_selection_changed = False
        self._update_playlist_action_states()
        self._update_navigation_buttons()
        if queue_finished:
            label = "播放列表演奏结束" if mode == "game" else "播放列表试听结束"
        else:
            label = "演奏结束" if mode == "game" else "试听结束"
        self.status_var.set(f"{label}，可选择歌曲继续播放")

    def _on_preview_audio_error(self, exc):
        self._run_on_ui(self._handle_preview_audio_error, str(exc))

    def _handle_preview_audio_error(self, detail):
        if self._active_mode not in ("preview", "preview_loading"):
            return
        self.stop_play()
        self.status_var.set("本地试听失败")
        messagebox.showerror("本地试听失败", detail)

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
            if getattr(self, 'overlay', None) is not None:
                idx, _t_ms, notes = self._current_note_info
                self.overlay.update_playback(
                    progress=frac,
                    elapsed_sec=self._elapsed_sec,
                    current_idx=idx,
                    current_notes=notes,
                )
        except Exception:
            pass
        # 约 20fps 刷新，兼顾流畅度与开销
        self.root.after(50, self._refresh_progress_ui)

    def _debug_log(self, msg):
        # 调试模式日志：打印并保留最近若干条，供后续调试面板使用
        self._record_log(msg, echo=True)

    def _record_log(self, msg, echo=False):
        if echo:
            print(msg)
        self.debug_logs.append(msg)
        if len(self.debug_logs) > 200:
            self.debug_logs = self.debug_logs[-200:]
        self._run_on_ui(self._refresh_log_text)

    def toggle_overlay_lock(self):
        if not getattr(self, 'overlay', None):
            return
        locked = self.overlay.toggle_lock()
        self.config["overlay_locked"] = locked
        self._update_overlay_status()
        self.status_var.set("覆盖层已锁定并点击穿透" if locked else "覆盖层已解锁，可拖动位置")

    def _on_overlay_geometry_changed(self, geometry):
        self.config["overlay_geometry"] = geometry
        self._update_overlay_status()

    def _update_overlay_status(self):
        if not getattr(self, 'overlay', None):
            self.overlay_status_var.set("未初始化")
            return
        visible = "显示" if self.overlay.window.winfo_viewable() else "隐藏"
        mode = "点击穿透" if self.overlay.locked else "可拖动"
        self.overlay_status_var.set(f"{visible} / {mode} / {self.overlay.window.geometry()}")

    def _refresh_log_text(self):
        if not getattr(self, 'debug_text', None):
            return
        text = "\n".join(self.debug_logs[-50:])
        self.debug_text.configure(state="normal")
        self.debug_text.delete("1.0", tk.END)
        self.debug_text.insert(tk.END, text)
        self.debug_text.see(tk.END)
        self.debug_text.configure(state="disabled")

    def _refresh_diagnostics(self, schedule=True):
        self.admin_status_var.set("是" if is_admin() else "否")
        try:
            hwnd = self._game_hwnd or find_sky_game_window()
            self.game_window_var.set(describe_window(hwnd) if hwnd else "未检测到")
        except Exception as e:
            self.game_window_var.set(f"读取失败: {e}")
        self.foreground_window_var.set(describe_foreground_window())
        self._update_overlay_status()
        self._refresh_input_method_status()
        self._refresh_log_text()
        if schedule:
            self.root.after(1000, self._refresh_diagnostics)

    def _refresh_input_method_status(self):
        """刷新“键盘输入方式”状态提示：当前后端 + 驱动可用性。"""
        if not getattr(self, 'input_method_status_var', None) or not getattr(self, 'key_controller', None):
            return
        kc = self.key_controller
        driver_ok = kc.is_driver_available()
        eff_name = kc.effective_backend()
        eff_label = kc.get_backend_label(eff_name)
        parts = [f"当前使用: {eff_label}"]
        if kc.is_auto():
            parts.append("（自动模式）")
        if not driver_ok:
            parts.append("；Interception 驱动未安装，已回退到常规键盘")
        else:
            parts.append("；Interception 驱动已就绪")
        self.input_method_status_var.set("".join(parts))

    def _on_input_method_change(self, event=None):
        """下拉框切换输入方式，写入控制器并持久化到 config。"""
        label = self.input_method_var.get()
        name = self._input_label_to_name.get(label)
        if not name:
            return
        self.input_method = name
        self.key_controller.set_backend(name)
        self._debug_log(f"[INFO] 键盘输入方式切换为: {label}")
        self._refresh_input_method_status()

    def calibrate_driver_keyboard(self):
        """交互式校准驱动级键盘设备（后台线程，需用户按一次键）。"""
        if not self.key_controller.is_driver_available():
            messagebox.showwarning(
                "驱动未安装",
                "未检测到 Interception 驱动，无法使用驱动级键盘。\n"
                "请先安装 interception-driver（见 README 说明）。")
            return
        self._debug_log("[INFO] 开始校准驱动级键盘，请在 10 秒内按下任意键以识别设备...")

        def worker():
            try:
                self.key_controller.calibrate_driver_keyboard()
                self._debug_log("[INFO] 驱动级键盘设备校准完成")
                msg = "校准完成，已识别当前键盘设备。"
            except Exception as e:
                self._debug_log(f"[WARN] 驱动级键盘校准失败: {e}")
                msg = f"校准失败: {e}"
            try:
                self.root.after(0, lambda: messagebox.showinfo("校准结果", msg))
            except Exception:
                pass
            self._refresh_input_method_status()

        threading.Thread(target=worker, daemon=True).start()
        messagebox.showinfo(
            "请按键校准",
            "请在接下来的几秒内按下键盘上的任意一个键，\n程序将据此识别你的键盘设备。")

    def check_and_set_game_window(self):
        hwnd = find_sky_game_window()
        if hwnd:
            self._game_hwnd = hwnd
            self.game_window_var.set(describe_window(hwnd))
            if not self._bring_window_to_front(hwnd):
                try:
                    self.root.after(0, lambda: messagebox.showwarning(
                        "焦点切换失败",
                        "无法自动将游戏窗口置于前台（可能被系统限制）。\n请手动点击一下游戏窗口，再按 F5 / 开始演奏。"))
                except Exception:
                    pass
            return True
        messagebox.showwarning("未检测到游戏", "未找到进程名为 'Sky' 或 '光遇' 的游戏窗口，请先打开游戏！")
        self.game_window_var.set("未检测到")
        return False

    def _bring_window_to_front(self, hwnd):
        return bring_window_to_front(hwnd)

    def _prepare_game_input(self):
        hwnd = getattr(self, '_game_hwnd', None)
        if switch_to_english_input(hwnd):
            self._debug_log("[INFO] requested en-US keyboard layout")
        else:
            self._debug_log("[WARN] could not switch to en-US keyboard layout")

    def _release_game_topmost(self):
        """演奏结束后解除游戏窗口置顶，恢复正常桌面层级。"""
        release_topmost(getattr(self, '_game_hwnd', None))

    def load_music(self):
        selected = self._selected_music_filename()
        if not selected:
            messagebox.showwarning("提示", "请先选择乐谱！")
            return False
        score = self._read_score_for_filename(selected, show_error=True)
        if score is None:
            return False
        self._apply_loaded_score(score)
        return True

    # ---------- 扒谱（音频 → 乐谱） ----------
    def open_generate_dialog(self):
        """打开本地文件 / 网易云在线扒谱窗口。"""
        from transcription.dialog import TranscriptionDialog

        existing = getattr(self, "_transcription_dialog", None)
        if existing is not None and not existing.closed:
            existing.win.deiconify()
            existing.win.lift()
            existing.win.focus_force()
            return
        self._transcription_dialog = TranscriptionDialog(
            self.root,
            (),
            SHEET_MUSIC_DIR,
            accent=self.accent,
            on_saved=self._refresh_music_after_generation,
            auth_file=resource_path("netease_auth.json"),
        )

    def _refresh_music_after_generation(self, output_path=None):
        """保存草稿后立即刷新乐谱列表。"""
        selected_filename = os.path.basename(output_path) if output_path else None
        self.all_music_files = self.get_all_music_files() or []
        self.filtered_music_files = self.all_music_files.copy()
        self.search_var.set("")
        self.current_music_tab.set("全部曲谱")
        self.refresh_music_listbox()
        self.last_music_files = set(self.all_music_files)
        if selected_filename in self.visible_music_files:
            index = self.visible_music_files.index(selected_filename)
            self.music_listbox.selection_clear(0, tk.END)
            self.music_listbox.selection_set(index)
            self.music_listbox.activate(index)
            self.music_listbox.see(index)
            self.update_song_info(selected_filename)
            self.status_var.set(f"已生成并选择乐谱: {selected_filename}")

    def on_close(self):
        # 退出软件前确保处于"未播放"态并释放可能按住的按键
        if getattr(self, 'player', None):
            self.player.stop()
        if getattr(self, "preview_player", None):
            self.preview_player.stop()
        if getattr(self, "audio_preview", None):
            self.audio_preview.close()
        dialog = getattr(self, "_transcription_dialog", None)
        if dialog is not None:
            dialog.close()
        self._release_game_topmost()
        # 保存窗口大小和位置
        try:
            geo = self.root.geometry()
            size_pos = geo.split('+')
            size = size_pos[0].split('x')
            width, height = int(size[0]), int(size[1])
            x, y = int(size_pos[1]), int(size_pos[2])
            cfg = dict(getattr(self, "config", {}) or {})
            cfg.update({'width': width, 'height': height, 'x': x, 'y': y})
            cfg['input_method'] = getattr(self, 'input_method', 'auto')
            if getattr(self, 'overlay', None):
                cfg["overlay_geometry"] = self.overlay.window.geometry()
                cfg["overlay_locked"] = self.overlay.locked
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f)
        except Exception:
            pass
        self.root.destroy()

    def schedule_music_dir_watch(self):
        current_files = set(self.get_all_music_files())
        try:
            playlist_changed = self.playlist_store.refresh(current_files)
        except OSError as exc:
            playlist_changed = False
            self._debug_log(f"[WARN] 播放列表清理保存失败: {exc}")
        if current_files != self.last_music_files or playlist_changed:
            self.all_music_files = sorted(current_files, key=str.casefold)
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
        # 注册开始、切歌、播放/暂停、停止热键
        try:
            keyboard.add_hotkey(self.hotkeys['start'], lambda: self._run_on_ui(self.start_play))
            keyboard.add_hotkey(self.hotkeys['previous'], lambda: self._run_on_ui(self.previous_track))
            keyboard.add_hotkey(self.hotkeys['stop'], lambda: self._run_on_ui(self.stop_play))
            keyboard.add_hotkey(self.hotkeys['next'], lambda: self._run_on_ui(self.next_track))
            keyboard.add_hotkey(self.hotkeys['overlay_lock'], lambda: self._run_on_ui(self.toggle_overlay_lock))
            keyboard.add_hotkey(self.hotkeys['toggle_pause'], lambda: self._run_on_ui(self.toggle_play_pause))
            self.hotkey_status_var.set(
                f"已注册: {self.hotkeys['start']} 演奏 / "
                f"{self.hotkeys['previous']} 上一首 / {self.hotkeys['stop']} 停止 / "
                f"{self.hotkeys['next']} 下一首 / {self.hotkeys['toggle_pause']} 暂停 / "
                f"{self.hotkeys['overlay_lock']} 覆盖层"
            )
        except Exception as e:
            self.hotkey_status_var.set(f"注册失败: {e}")
            messagebox.showwarning("热键注册失败", f"全局热键注册失败，可能需要以管理员身份运行。\n详细信息：{e}")

    def on_music_tab_changed(self):
        # 分页切换时刷新乐谱列表
        self.refresh_music_listbox()

    def on_music_listbox_right_click(self, event):
        """右键菜单：收藏以及播放列表添加/移除。"""
        idx = self.music_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.visible_music_files or []):
            return
        self.music_listbox.selection_clear(0, tk.END)
        self.music_listbox.selection_set(idx)
        self.music_listbox.activate(idx)
        filename = self.visible_music_files[idx]
        self.update_song_info(filename)
        if self._active_mode is None:
            self.status_var.set(f"已选择乐谱: {filename}")
        menu = tk.Menu(self.music_listbox, tearoff=0)
        if filename in self.favorites:
            menu.add_command(label="取消收藏", command=lambda: self.toggle_favorite(filename))
        else:
            menu.add_command(label="收藏", command=lambda: self.toggle_favorite(filename))
        menu.add_separator()
        mutation_state = "normal" if self._active_mode is None else "disabled"
        if filename in self.playlist_store.items:
            menu.add_command(
                label="从播放列表移除",
                state=mutation_state,
                command=lambda: self.remove_filename_from_playlist(filename),
            )
        else:
            menu.add_command(
                label="加入播放列表",
                state=mutation_state,
                command=lambda: self.add_filename_to_playlist(filename),
            )
        menu.tk_popup(event.x_root, event.y_root)

    def add_filename_to_playlist(self, filename):
        if not self._playlist_mutation_allowed():
            return
        if self._read_score_for_filename(filename, show_error=True) is None:
            return
        try:
            added = self.playlist_store.add(filename)
        except OSError as exc:
            messagebox.showerror("播放列表保存失败", str(exc))
            return
        if added:
            self.status_var.set(f"已加入播放列表: {filename}")
        else:
            self.status_var.set(f"歌曲已在播放列表中: {filename}")
        self.refresh_music_listbox()

    def remove_filename_from_playlist(self, filename):
        if not self._playlist_mutation_allowed():
            return
        try:
            removed = self.playlist_store.remove(filename)
        except OSError as exc:
            messagebox.showerror("播放列表保存失败", str(exc))
            return
        if removed:
            self.status_var.set(f"已从播放列表移除: {filename}")
            self.refresh_music_listbox()

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

def run_transcriber_self_test(audio_path):
    """供 Windows 打包流水线验证 ONNX 模型、解码和 JSON 兼容性。"""
    import tempfile
    from transcription import TranscriptionOptions, export_song_json, transcribe_draft
    from transcription.netease import validate_netease_runtime

    validate_netease_runtime()

    result = transcribe_draft(
        audio_path,
        TranscriptionOptions(
            mode="polyphonic",
            source_key="C major",
            octave_shift=0,
            max_polyphony=3,
        ),
    )
    with tempfile.TemporaryDirectory(prefix="sky-self-test-") as temp_dir:
        output_path = os.path.join(temp_dir, "self-test.json")
        export_song_json(result, output_path, "self-test")
        load_score(output_path, valid_keys=note_to_key.keys())
    return True


def run_six_stem_model_self_test():
    """供打包流水线验证随包 CPU 运行时、模型文件与完整 SHA-256。"""
    from transcription.separation import DemucsStemSeparator

    DemucsStemSeparator().prepare()
    return True


if __name__ == "__main__":
    if "--six-stem-model-self-test" in sys.argv:
        try:
            run_six_stem_model_self_test()
            sys.exit(0)
        except Exception as exc:
            try:
                print(f"six-stem model self-test failed: {exc}", file=sys.stderr)
            finally:
                sys.exit(1)
    if "--transcriber-self-test" in sys.argv:
        try:
            arg_index = sys.argv.index("--transcriber-self-test")
            input_path = sys.argv[arg_index + 1]
            run_transcriber_self_test(input_path)
            sys.exit(0)
        except Exception as exc:
            try:
                print(f"transcriber self-test failed: {exc}", file=sys.stderr)
            finally:
                sys.exit(1)
    if relaunch_as_admin_if_needed():
        sys.exit(0)
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', font=('微软雅黑', 10))
    app = MusicGUI(root)
    app.bind_hotkeys()
    root.mainloop() 
