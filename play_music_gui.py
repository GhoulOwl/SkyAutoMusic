import os
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import webbrowser

from key_controller import KeyController, note_to_key
from player import MusicPlayer, PlaybackState
from score_loader import ScoreValidationError, load_score, read_json_with_fallback, summarize_meta
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
        win_w, win_h = 600, 480
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
        self.root.minsize(480, 360)
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
        self.default_hotkeys = {'start': 'F5', 'stop': 'F7', 'overlay_lock': 'F10'}
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
        self.music_info_vars = {
            'filename': tk.StringVar(),
            'path': tk.StringVar(),
            'name': tk.StringVar(),
            'author': tk.StringVar(),
            'transcribedBy': tk.StringVar(),
        }
        self.filtered_music_files = []  # 先初始化，防止后续方法引用时报错
        self.visible_music_files = []
        self.favorites = set()  # 收藏的乐谱文件名集合，可持久化
        self.favorite_file = resource_path('favorites.json')  # 用resource_path，兼容打包
        self.load_favorites()  # 启动时加载收藏
        self.debug_logs = []
        self._progress_frac = 0.0
        self._elapsed_sec = 0.0
        self._current_note_info = (-1, None, [])
        self._game_hwnd = None  # 当前已置顶/聚焦的游戏窗口句柄，stop 时解除置顶
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
        self.create_diagnostics_tab(parent=diagnostics_tab)

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
        ttk.Label(frame, text="覆盖层移动/锁定:").grid(row=3, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(frame, textvariable=self.hotkey_vars['overlay_lock'], font=("微软雅黑", 10, "bold"), foreground=self.accent).grid(row=3, column=1, sticky="w", padx=4, pady=4)

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
        self.visible_music_files = list(files)
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
            filename = self.visible_music_files[sel[0]]
            self.update_song_info(filename)
            self.status_var.set(f"已选择乐谱: {filename}")

    def refresh_music_list(self):
        # 兼容旧接口，实际不再用
        self.all_music_files = self.get_all_music_files() or []
        self.filtered_music_files = self.all_music_files.copy() if self.all_music_files else []
        self.refresh_music_listbox()

    def update_song_info(self, filename):
        # 页面信息区展示选中曲谱详细信息
        if not filename:
            self.music_info_vars['filename'].set("")
            self.music_info_vars['name'].set("")
            self.music_info_vars['author'].set("")
            self.music_info_vars['transcribedBy'].set("")
            return
        self.music_info_vars['filename'].set(filename)
        path = os.path.join(SHEET_MUSIC_DIR, filename)  # SHEET_MUSIC_DIR已用resource_path
        try:
            data = read_json_with_fallback(path)
            meta = summarize_meta(data)
        except Exception:
            self.music_info_vars['name'].set('')
            self.music_info_vars['author'].set('')
            self.music_info_vars['transcribedBy'].set('')
            return
        self.music_info_vars['name'].set(meta.get('name', ''))
        self.music_info_vars['author'].set(meta.get('author', ''))
        self.music_info_vars['transcribedBy'].set(meta.get('transcribedBy', ''))

    def start_play(self):
        # 三态约束：从头开始必须处于"未播放"态，否则异常（依据 1.md）
        if self.player.state != PlaybackState.STOPPED:
            self.status_var.set("请先停止当前演奏（F7）再重新开始")
            return
        if not self.check_and_set_game_window():
            return
        self._prepare_game_input()
        if not self.load_music():
            return
        # 把当前可调参数注入播放器
        self.player.speed = self.speed
        self.player.simulate = self.simulate
        self.player.miss_prob = self.miss_prob
        self.key_controller.set_debug(self.debug)
        # 从头开始：清零进度显示，避免残留上一次的进度
        self._progress_frac = 0.0
        self._elapsed_sec = 0.0
        self._current_note_info = (-1, None, [])
        if getattr(self, 'progress_bar', None) is not None:
            self.progress_bar['value'] = 0
        title = self.music_info_vars['name'].get() or self.music_info_vars['filename'].get()
        self.overlay.set_score(self.notes_by_time, self.sorted_times, title=title)
        self.overlay.show(self._game_hwnd)
        self._update_overlay_status()
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
        self._elapsed_sec = 0.0
        self._current_note_info = (-1, None, [])
        if getattr(self, 'progress_bar', None) is not None:
            self.progress_bar['value'] = 0
        if getattr(self, 'overlay', None):
            self.overlay.hide()
            self._update_overlay_status()
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
    def _run_on_ui(self, func, *args):
        try:
            self.root.after(0, lambda: func(*args))
        except Exception:
            pass

    def _on_status(self, msg):
        self._run_on_ui(self.status_var.set, msg)

    def _on_elapsed(self, sec):
        self._elapsed_sec = max(0.0, float(sec))
        def _apply():
            m, s = divmod(int(sec), 60)
            self.elapsed_time_var.set(f"{m}:{s:02d}")
        self._run_on_ui(_apply)

    def _on_total(self, sec):
        def _apply():
            m, s = divmod(int(sec), 60)
            self.total_time_var.set(f"{m}:{s:02d}")
        self._run_on_ui(_apply)

    def _on_progress(self, frac):
        # 仅在播放线程内记录进度值，真正的 UI 更新交给主线程的 _refresh_progress_ui。
        self._progress_frac = max(0.0, min(1.0, frac))

    def _on_note(self, idx, t_ms, notes):
        self._current_note_info = (idx, t_ms, list(notes or []))
        if idx >= 0:
            self._record_log(f"[NOTE] #{idx + 1} {t_ms}ms -> {','.join(notes or [])}", echo=False)

    def _on_finished(self):
        """演奏自然结束时由播放线程触发；转交主线程重置 UI 到"就绪"态。

        这样用户无需先点"停止"即可直接选择下一首并按"开始"。
        """
        self._run_on_ui(self._handle_playback_finished)

    def _handle_playback_finished(self):
        # 仅在确实由"播放中"自然结束时处理，避免与 stop_play 重复重置
        if str(self.start_btn.cget('state')) != 'disabled':
            return
        self._release_game_topmost()
        if getattr(self, 'overlay', None):
            self.overlay.hide()
            self._update_overlay_status()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("演奏结束，点击开始或按F5演奏下一首")

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
        sel = self.music_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先选择乐谱！")
            return False
        selected = self.visible_music_files[sel[0]]
        path = os.path.join(SHEET_MUSIC_DIR, selected)  # SHEET_MUSIC_DIR已用resource_path
        try:
            score = load_score(path, valid_keys=self.key_controller.mapping.keys())
        except ScoreValidationError as e:
            messagebox.showerror("乐谱校验失败", str(e))
            return False
        self.music_data = score["raw"]
        self.notes_by_time = score["notes_by_time"]
        self.sorted_times = score["sorted_times"]
        self.bpm = score["meta"].get('bpm', 120)
        if score["warnings"]:
            self._debug_log("[WARN] " + "；".join(score["warnings"][:5]))
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
            keyboard.add_hotkey(self.hotkeys['start'], lambda: self._run_on_ui(self.start_play))
            keyboard.add_hotkey('F11', lambda: self._run_on_ui(self.toggle_play_pause))
            keyboard.add_hotkey(self.hotkeys['stop'], lambda: self._run_on_ui(self.stop_play))
            keyboard.add_hotkey(self.hotkeys['overlay_lock'], lambda: self._run_on_ui(self.toggle_overlay_lock))
            self.hotkey_status_var.set(
                f"已注册: {self.hotkeys['start']} 开始 / F11 暂停 / {self.hotkeys['stop']} 停止 / {self.hotkeys['overlay_lock']} 覆盖层"
            )
        except Exception as e:
            self.hotkey_status_var.set(f"注册失败: {e}")
            messagebox.showwarning("热键注册失败", f"全局热键注册失败，可能需要以管理员身份运行。\n详细信息：{e}")

    def on_music_tab_changed(self):
        # 分页切换时刷新乐谱列表
        self.refresh_music_listbox()

    def on_music_listbox_right_click(self, event):
        """
        右键菜单：仅保留收藏/取消收藏
        """
        idx = self.music_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.visible_music_files or []):
            return
        self.music_listbox.selection_clear(0, tk.END)
        self.music_listbox.selection_set(idx)
        filename = self.visible_music_files[idx]
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
    if relaunch_as_admin_if_needed():
        sys.exit(0)
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', font=('微软雅黑', 10))
    app = MusicGUI(root)
    app.bind_hotkeys()
    root.mainloop() 
