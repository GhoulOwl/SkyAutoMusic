from __future__ import annotations

import os
import shutil
import tempfile
import threading
import tkinter as tk
from dataclasses import replace
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Callable, Dict, List, Optional, Sequence

from .arranger import NOTE_NAMES
from .backends import is_midi_file
from .models import (
    CancelledError,
    DEFAULT_ENABLED_STEMS,
    SourceMetadata,
    StemKind,
    TranscriptionOptions,
    TranscriptionResult,
)
from .netease import NetEaseClient, NetEaseSearchPage, NetEaseTrack
from .netease_auth import (
    CookieValidationResult,
    NetEaseCookieStore,
    parse_netscape_cookies,
    serialize_netscape_cookies,
    validate_cookie_account,
)
from .pipeline import (
    cleanup_result_artifacts,
    export_song_json,
    next_available_path,
    rearrange_draft,
    suggested_output_stem,
    transcribe_draft,
)
from .preview import PreviewPlayer, StemPreviewPlayer
from player import PlaybackState


MODE_LABELS = {
    "智能六轨融合": "stem_fusion",
    "复音高质量（Basic Pitch）": "polyphonic",
    "单旋律快速（pYIN）": "monophonic",
}
SENSITIVITY_LABELS = {"低": "low", "普通": "normal", "高": "high"}
QUANTIZE_LABELS = {"关闭": "off", "八分音符": "1/8", "十六分音符": "1/16"}
REPEAT_CLEANUP_LABELS = {"自动": "auto", "强力": "strong", "关闭": "off"}
FUSION_PROFILE_LABELS = {
    "人声优先": "vocal_first",
    "键盘优先": "keyboard_first",
    "平衡融合": "balanced",
}
STEM_LABELS = {
    "vocals": "人声",
    "drums": "鼓点",
    "piano": "钢琴/键盘",
    "bass": "贝斯",
    "guitar": "吉他",
    "instrumental": "去人声完整伴奏",
}


class TranscriptionDialog:
    """批量生成内存草稿，并允许试听、调参和确认保存。"""

    def __init__(
        self,
        parent: tk.Misc,
        files: Sequence[str],
        output_dir: str,
        accent: str = "#4F8CFF",
        on_saved: Optional[Callable[[str], None]] = None,
        auth_file: Optional[str] = None,
        netease_client: Optional[NetEaseClient] = None,
    ):
        self.parent = parent
        self.files = list(files)
        self.output_dir = output_dir
        self.accent = accent
        self.on_saved = on_saved or (lambda _path: None)
        self.results: Dict[str, TranscriptionResult] = {}
        self.errors: Dict[str, str] = {}
        self.saved_paths: Dict[str, str] = {}
        self.source_labels: Dict[str, str] = {
            path: os.path.basename(path) for path in self.files
        }
        self.online_paths = set()
        self.cancel_event = threading.Event()
        self.preview_player = PreviewPlayer(
            on_status=self._on_score_preview_status,
            on_finished=self._on_score_preview_finished,
            on_error=self._on_score_preview_error,
        )
        self.stem_preview_player = StemPreviewPlayer()
        self._preview_preparing = False
        self.busy = False
        self.closed = False
        self._workers_lock = threading.Lock()
        self._worker_count = 0
        self._temp_root = tempfile.TemporaryDirectory(prefix="sky-netease-drafts-")
        auth_path = auth_file or os.path.join(
            os.path.dirname(os.path.abspath(output_dir)), "netease_auth.json"
        )
        self.cookie_store = NetEaseCookieStore(auth_path)
        self.netease_client = netease_client or NetEaseClient()
        self.search_generation = 0
        self.search_page = 0
        self.search_limit = 20
        self.search_total = 0
        self.search_tracks: Dict[str, NetEaseTrack] = {}

        self.win = tk.Toplevel(parent)
        self.win.title("生成乐谱")
        self.win.transient(parent)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.mode_var = tk.StringVar(value="智能六轨融合")
        self.sensitivity_var = tk.StringVar(value="普通")
        self.key_var = tk.StringVar(value="自动")
        self.octave_var = tk.StringVar(value="自动")
        self.quantize_var = tk.StringVar(value="关闭")
        self.polyphony_var = tk.IntVar(value=3)
        self.repeat_cleanup_var = tk.StringVar(value="自动")
        self.fusion_profile_var = tk.StringVar(value="人声优先")
        self.use_drum_timing_var = tk.BooleanVar(value=True)
        self.stem_enabled_vars: Dict[StemKind, tk.BooleanVar] = {
            stem: tk.BooleanVar(value=stem in DEFAULT_ENABLED_STEMS)
            for stem in STEM_LABELS
            if stem != "drums"
        }
        self.stem_controls: Dict[StemKind, Dict[str, object]] = {}
        self.status_var = tk.StringVar(value="准备生成草稿…")
        self.detail_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="尚未生成草稿")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.local_summary_var = tk.StringVar(value="尚未选择本地文件")
        self.netease_cookie_status_var = tk.StringVar(value="未保存网易云 Cookie")
        self.netease_query_var = tk.StringVar()
        self.netease_page_var = tk.StringVar(value="第 1 页")
        self.netease_result_var = tk.StringVar(value="请输入歌曲名或歌手名")

        self._build_widgets()
        self._fit_toplevel(
            self.win,
            preferred_width=1000,
            preferred_height=940,
            minimum_width=800,
            minimum_height=680,
        )
        self._load_cookie_status()
        if self.files:
            self.local_summary_var.set(f"已选择 {len(self.files)} 个本地文件")
            self.win.after(100, self.generate_all)

    @staticmethod
    def _fit_toplevel(
        window: tk.Toplevel,
        *,
        preferred_width: int,
        preferred_height: int,
        minimum_width: int,
        minimum_height: int,
    ) -> None:
        """Choose a useful initial size without placing controls below the screen."""
        window.update_idletasks()
        screen_width = max(1, window.winfo_screenwidth())
        screen_height = max(1, window.winfo_screenheight())
        margin_x = min(80, max(20, screen_width // 20))
        margin_y = min(100, max(40, screen_height // 12))
        available_width = max(1, screen_width - margin_x)
        available_height = max(1, screen_height - margin_y)

        width = min(
            max(preferred_width, window.winfo_reqwidth()),
            available_width,
        )
        height = min(
            max(preferred_height, window.winfo_reqheight()),
            available_height,
        )
        window.minsize(
            min(minimum_width, available_width),
            min(minimum_height, available_height),
        )

        master = window.master
        try:
            master.update_idletasks()
            x = master.winfo_rootx() + max(0, (master.winfo_width() - width) // 2)
            y = master.winfo_rooty() + max(0, (master.winfo_height() - height) // 2)
        except (AttributeError, tk.TclError):
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
        x = max(0, min(x, screen_width - width))
        y = max(0, min(y, screen_height - height))
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _build_widgets(self) -> None:
        self._build_source_tabs()

        options = ttk.LabelFrame(self.win, text="转写与编配参数", padding=10)
        options.pack(fill="x", padx=12, pady=(12, 6))

        ttk.Label(options, text="模式").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.mode_box = ttk.Combobox(
            options,
            textvariable=self.mode_var,
            values=list(MODE_LABELS),
            state="readonly",
            width=22,
        )
        self.mode_box.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(options, text="灵敏度").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.sensitivity_box = ttk.Combobox(
            options,
            textvariable=self.sensitivity_var,
            values=list(SENSITIVITY_LABELS),
            state="readonly",
            width=8,
        )
        self.sensitivity_box.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        key_values = ["自动"]
        key_values.extend(f"{name} major" for name in NOTE_NAMES)
        key_values.extend(f"{name} minor" for name in NOTE_NAMES)
        ttk.Label(options, text="原曲调性").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        self.key_box = ttk.Combobox(
            options,
            textvariable=self.key_var,
            values=key_values,
            state="readonly",
            width=11,
        )
        self.key_box.grid(row=0, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(options, text="八度").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        self.octave_box = ttk.Combobox(
            options,
            textvariable=self.octave_var,
            values=["自动", "-2", "-1", "0", "+1", "+2"],
            state="readonly",
            width=8,
        )
        self.octave_box.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(options, text="节奏量化").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        self.quantize_box = ttk.Combobox(
            options,
            textvariable=self.quantize_var,
            values=list(QUANTIZE_LABELS),
            state="readonly",
            width=10,
        )
        self.quantize_box.grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(options, text="最大复音").grid(row=1, column=4, sticky="e", padx=4, pady=4)
        ttk.Spinbox(
            options,
            from_=1,
            to=5,
            textvariable=self.polyphony_var,
            width=6,
            state="readonly",
        ).grid(row=1, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(options, text="重复音清理").grid(
            row=2, column=0, sticky="e", padx=4, pady=4
        )
        self.repeat_cleanup_box = ttk.Combobox(
            options,
            textvariable=self.repeat_cleanup_var,
            values=list(REPEAT_CLEANUP_LABELS),
            state="readonly",
            width=8,
        )
        self.repeat_cleanup_box.grid(
            row=2, column=1, sticky="w", padx=4, pady=4
        )
        ttk.Label(options, text="融合预设").grid(
            row=2, column=2, sticky="e", padx=4, pady=4
        )
        self.fusion_profile_box = ttk.Combobox(
            options,
            textvariable=self.fusion_profile_var,
            values=list(FUSION_PROFILE_LABELS),
            state="readonly",
            width=10,
        )
        self.fusion_profile_box.grid(
            row=2, column=3, sticky="w", padx=4, pady=4
        )
        options.columnconfigure(6, weight=1)

        progress_frame = ttk.Frame(self.win)
        progress_frame.pack(fill="x", padx=12, pady=5)
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w")
        ttk.Progressbar(
            progress_frame,
            maximum=1.0,
            variable=self.progress_var,
            mode="determinate",
        ).pack(fill="x", pady=4)
        ttk.Label(
            progress_frame,
            textvariable=self.detail_var,
            foreground="#777",
        ).pack(anchor="w")

        content = ttk.Panedwindow(self.win, orient=tk.HORIZONTAL)
        content.pack(fill="both", expand=True, padx=12, pady=6)
        left = ttk.Frame(content, width=220)
        right = ttk.Frame(content)
        content.add(left, weight=1)
        content.add(right, weight=3)

        ttk.Label(left, text="草稿列表", font=("微软雅黑", 10, "bold")).pack(anchor="w")
        self.file_list = tk.Listbox(left, exportselection=False)
        self.file_list.pack(fill="both", expand=True, pady=(4, 0))
        self.file_list.bind("<<ListboxSelect>>", self._on_selection)
        for path in self.files:
            self.file_list.insert(tk.END, f"… {os.path.basename(path)}")

        ttk.Label(right, text="15键时间线（只读）", font=("微软雅黑", 10, "bold")).pack(anchor="w")
        self.timeline = tk.Canvas(
            right,
            background="#FFFFFF",
            highlightthickness=1,
            highlightbackground="#D9DEE7",
            height=300,
        )
        self.timeline.pack(fill="both", expand=True, pady=(4, 6))
        self.timeline.bind("<Configure>", lambda _event: self._draw_timeline())
        ttk.Label(
            right,
            textvariable=self.stats_var,
            justify="left",
            foreground="#555",
        ).pack(fill="x", anchor="w")
        self._build_stem_panel(right)

        actions = ttk.Frame(self.win)
        actions.pack(
            side="bottom",
            fill="x",
            padx=12,
            pady=(6, 12),
            before=content,
        )
        self.regenerate_btn = ttk.Button(
            actions, text="重新生成当前", command=self.regenerate_current, state="disabled"
        )
        self.regenerate_btn.pack(side="left", padx=(0, 6))
        self.preview_btn = ttk.Button(
            actions,
            text="光遇音色试听",
            command=self.preview_current,
            state="disabled",
        )
        self.preview_btn.pack(side="left", padx=6)
        self.stop_preview_btn = ttk.Button(
            actions, text="停止试听", command=self.stop_all_previews, state="normal"
        )
        self.stop_preview_btn.pack(side="left", padx=6)
        self.save_btn = ttk.Button(
            actions, text="保存当前", command=self.save_current, state="disabled"
        )
        self.save_btn.pack(side="left", padx=6)
        self.save_all_btn = ttk.Button(
            actions, text="保存全部", command=self.save_all, state="disabled"
        )
        self.save_all_btn.pack(side="left", padx=6)
        self.cancel_btn = ttk.Button(actions, text="取消", command=self.cancel)
        self.cancel_btn.pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="关闭", command=self.close).pack(side="right")

    def _build_stem_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="六轨分离（原声试听）", padding=6)
        frame.pack(fill="x", pady=(8, 0))
        for row, stem in enumerate(STEM_LABELS):
            if stem == "drums":
                toggle = ttk.Checkbutton(
                    frame,
                    text="鼓点用于节奏",
                    variable=self.use_drum_timing_var,
                )
            else:
                toggle = ttk.Checkbutton(
                    frame,
                    text=f"{STEM_LABELS[stem]}参与琴谱",
                    variable=self.stem_enabled_vars[stem],
                )
            toggle.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
            status = ttk.Label(frame, text="尚未生成", foreground="#667")
            status.grid(row=row, column=1, sticky="w", padx=4, pady=1)
            preview = ttk.Button(
                frame,
                text="试听原声",
                width=10,
                command=lambda value=stem: self.preview_stem(value),
                state="disabled",
            )
            preview.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=1)
            self.stem_controls[stem] = {
                "toggle": toggle,
                "status": status,
                "preview": preview,
            }
        frame.columnconfigure(1, weight=1)

    def _build_source_tabs(self) -> None:
        sources = ttk.Notebook(self.win)
        sources.pack(fill="x", padx=12, pady=(12, 6))
        local_tab = ttk.Frame(sources, padding=8)
        online_tab = ttk.Frame(sources, padding=8)
        online_tab.columnconfigure(0, weight=1)
        online_tab.rowconfigure(2, weight=1)
        sources.add(local_tab, text="本地文件")
        sources.add(online_tab, text="网易云在线")

        local_actions = ttk.Frame(local_tab)
        local_actions.pack(fill="x")
        self.local_choose_btn = ttk.Button(
            local_actions,
            text="选择音频 / MIDI",
            command=self.choose_local_files,
        )
        self.local_choose_btn.pack(side="left")
        self.local_generate_btn = ttk.Button(
            local_actions,
            text="生成本地草稿",
            command=self.generate_all,
            state="normal" if self.files else "disabled",
        )
        self.local_generate_btn.pack(side="left", padx=8)
        ttk.Label(
            local_actions,
            textvariable=self.local_summary_var,
            foreground="#666",
        ).pack(side="left", padx=8)

        cookie_row = ttk.Frame(online_tab)
        cookie_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(cookie_row, text="网易云 Cookie：").pack(side="left")
        ttk.Label(
            cookie_row,
            textvariable=self.netease_cookie_status_var,
            foreground="#666",
        ).pack(side="left", fill="x", expand=True)
        self.cookie_edit_btn = ttk.Button(
            cookie_row, text="编辑", command=self.open_cookie_editor
        )
        self.cookie_edit_btn.pack(side="right", padx=(6, 0))
        self.cookie_clear_btn = ttk.Button(
            cookie_row, text="清除", command=self.clear_cookie
        )
        self.cookie_clear_btn.pack(side="right", padx=(6, 0))
        self.cookie_validate_btn = ttk.Button(
            cookie_row, text="重新验证", command=self.validate_saved_cookie
        )
        self.cookie_validate_btn.pack(side="right", padx=(6, 0))

        search_row = ttk.Frame(online_tab)
        search_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.netease_search_entry = ttk.Entry(
            search_row, textvariable=self.netease_query_var
        )
        self.netease_search_entry.pack(side="left", fill="x", expand=True)
        self.netease_search_entry.bind("<Return>", lambda _event: self.search_netease(0))
        self.netease_search_btn = ttk.Button(
            search_row, text="搜索", command=lambda: self.search_netease(0)
        )
        self.netease_search_btn.pack(side="left", padx=(8, 0))

        result_frame = ttk.Frame(online_tab)
        result_frame.grid(row=2, column=0, sticky="nsew")
        columns = ("title", "artists", "album", "duration")
        tree_style = ttk.Style(self.win)
        tree_row_height = max(
            24,
            tkfont.nametofont("TkDefaultFont").metrics("linespace") + 6,
        )
        tree_style.configure("NetEase.Treeview", rowheight=tree_row_height)
        self.netease_tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            style="NetEase.Treeview",
            height=7,
            selectmode="browse",
        )
        headings = {
            "title": ("歌曲", 240),
            "artists": ("歌手", 180),
            "album": ("专辑", 210),
            "duration": ("时长", 65),
        }
        for column, (text, width) in headings.items():
            self.netease_tree.heading(column, text=text)
            self.netease_tree.column(
                column,
                width=width,
                minwidth=50,
                stretch=column != "duration",
                anchor="center" if column == "duration" else "w",
            )
        result_scroll = ttk.Scrollbar(
            result_frame, orient="vertical", command=self.netease_tree.yview
        )
        self.netease_tree.configure(yscrollcommand=result_scroll.set)
        self.netease_tree.grid(row=0, column=0, sticky="nsew")
        result_scroll.grid(row=0, column=1, sticky="ns")
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self.netease_tree.bind("<<TreeviewSelect>>", self._on_netease_selection)
        self.netease_tree.bind("<Double-1>", lambda _event: self.generate_online_draft())

        online_actions = ttk.Frame(online_tab)
        online_actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            online_actions,
            textvariable=self.netease_result_var,
            foreground="#666",
        ).pack(side="left", fill="x", expand=True)
        self.netease_prev_btn = ttk.Button(
            online_actions,
            text="上一页",
            command=self.search_previous_page,
            state="disabled",
        )
        self.netease_prev_btn.pack(side="left", padx=4)
        ttk.Label(online_actions, textvariable=self.netease_page_var).pack(side="left")
        self.netease_next_btn = ttk.Button(
            online_actions,
            text="下一页",
            command=self.search_next_page,
            state="disabled",
        )
        self.netease_next_btn.pack(side="left", padx=4)
        self.netease_generate_btn = ttk.Button(
            online_actions,
            text="生成在线草稿",
            command=self.generate_online_draft,
            state="disabled",
        )
        self.netease_generate_btn.pack(side="right", padx=(8, 0))

        def stabilize_online_layout() -> None:
            try:
                exists = sources.winfo_exists()
            except tk.TclError:
                return
            if not exists:
                return
            sources.update_idletasks()
            # The result row gets sacrificial space below Treeview's requested
            # height.  Windows themed Tk can otherwise round the Notebook
            # client area down at 125%/150% DPI and take pixels from the final
            # visible row or the pagination bar.
            online_tab.rowconfigure(
                2,
                minsize=self.netease_tree.winfo_reqheight() + tree_row_height,
            )
            online_tab.update_idletasks()
            sources.configure(
                height=max(
                    local_tab.winfo_reqheight(),
                    online_tab.winfo_reqheight(),
                )
                + 8
            )

        stabilize_online_layout()
        self.win.after_idle(stabilize_online_layout)

    def _start_worker(self, target: Callable, *args) -> None:
        with self._workers_lock:
            self._worker_count += 1

        def runner() -> None:
            try:
                target(*args)
            finally:
                cleanup = False
                with self._workers_lock:
                    self._worker_count -= 1
                    cleanup = self.closed and self._worker_count == 0
                if cleanup:
                    self._cleanup_temp_root()

        threading.Thread(target=runner, daemon=True).start()

    def _cleanup_temp_root(self) -> None:
        temp_root = getattr(self, "_temp_root", None)
        if temp_root is None:
            return
        self._temp_root = None
        try:
            temp_root.cleanup()
        except Exception:
            pass

    def choose_local_files(self) -> None:
        if self.busy:
            return
        files = filedialog.askopenfilenames(
            parent=self.win,
            title="选择要转写的音频或 MIDI 文件",
            filetypes=[
                ("支持的音乐文件", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.mid *.midi"),
                ("音频文件", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac"),
                ("MIDI 文件", "*.mid *.midi"),
                ("全部", "*.*"),
            ],
        )
        added = 0
        for path in files:
            if path in self.files:
                continue
            self.files.append(path)
            self.source_labels[path] = os.path.basename(path)
            self.file_list.insert(tk.END, f"… {os.path.basename(path)}")
            added += 1
        if added:
            local_count = sum(1 for path in self.files if path not in self.online_paths)
            self.local_summary_var.set(f"已选择 {local_count} 个本地文件")
            self.local_generate_btn.config(state="normal")

    def _load_cookie_status(self) -> None:
        try:
            text = self.cookie_store.load_text()
            if not text:
                self.netease_cookie_status_var.set("未保存网易云 Cookie")
                return
            status = self.cookie_store.load_validation()
            self.netease_cookie_status_var.set(status.message)
        except Exception as exc:
            self.netease_cookie_status_var.set(f"Cookie 无法使用：{exc}")

    def _load_saved_cookies(self):
        text = self.cookie_store.load_text()
        return parse_netscape_cookies(text) if text else []

    def open_cookie_editor(self) -> None:
        editor = tk.Toplevel(self.win)
        editor.title("网易云 cookies.txt")
        editor.transient(self.win)
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(1, weight=1)

        intro_label = ttk.Label(
            editor,
            text=(
                "粘贴 Netscape cookies.txt 的完整内容。程序只保留网易云域名 Cookie，"
                "并使用当前 Windows 用户的 DPAPI 加密保存。"
            ),
            justify="left",
        )
        intro_label.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        intro_label.bind(
            "<Configure>",
            lambda event: intro_label.configure(wraplength=max(120, event.width)),
        )

        text_frame = ttk.Frame(editor)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        text_widget = tk.Text(
            text_frame,
            wrap="none",
            font=("Consolas", 9),
            undo=True,
            width=80,
            height=18,
        )
        text_widget.grid(row=0, column=0, sticky="nsew")
        text_vscroll = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=text_widget.yview,
        )
        text_vscroll.grid(row=0, column=1, sticky="ns")
        text_hscroll = ttk.Scrollbar(
            text_frame,
            orient="horizontal",
            command=text_widget.xview,
        )
        text_hscroll.grid(row=1, column=0, sticky="ew")
        text_widget.configure(
            yscrollcommand=text_vscroll.set,
            xscrollcommand=text_hscroll.set,
        )
        try:
            existing = self.cookie_store.load_text() or ""
        except Exception as exc:
            existing = ""
            messagebox.showwarning("Cookie 读取失败", str(exc), parent=editor)
        if existing:
            text_widget.insert("1.0", existing)

        status_var = tk.StringVar(value="等待检查")
        status_label = ttk.Label(
            editor,
            textvariable=status_var,
            foreground="#666",
            justify="left",
        )
        status_label.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))
        status_label.bind(
            "<Configure>",
            lambda event: status_label.configure(wraplength=max(120, event.width)),
        )
        actions = ttk.Frame(editor)
        actions.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 12))
        save_btn = ttk.Button(actions, text="保存", state="disabled")
        save_btn.pack(side="right")
        ttk.Button(actions, text="取消", command=editor.destroy).pack(
            side="right", padx=(0, 8)
        )
        state = {
            "generation": 0,
            "after": None,
            "canonical": None,
            "validation": None,
        }

        def apply_validation(
            generation: int,
            canonical: str,
            validation: CookieValidationResult,
        ) -> None:
            if not editor.winfo_exists() or generation != state["generation"]:
                return
            state["canonical"] = canonical
            state["validation"] = validation
            status_var.set(validation.message)
            save_btn.config(
                state="normal" if validation.state in ("valid", "unverified") else "disabled"
            )

        def validate_worker(generation: int, canonical: str, cookies) -> None:
            validation = validate_cookie_account(cookies)
            self._after(apply_validation, generation, canonical, validation)

        def run_validation() -> None:
            state["after"] = None
            state["generation"] += 1
            generation = state["generation"]
            raw = text_widget.get("1.0", "end-1c")
            state["canonical"] = None
            state["validation"] = None
            save_btn.config(state="disabled")
            try:
                cookies = parse_netscape_cookies(raw)
                canonical = serialize_netscape_cookies(cookies)
            except Exception as exc:
                status_var.set(str(exc))
                return
            status_var.set("本地格式检查通过，正在验证网易云登录状态…")
            self._start_worker(validate_worker, generation, canonical, cookies)

        def schedule_validation(_event=None) -> None:
            if _event is not None:
                if not text_widget.edit_modified():
                    return
                text_widget.edit_modified(False)
            pending = state.get("after")
            if pending:
                try:
                    editor.after_cancel(pending)
                except Exception:
                    pass
            state["after"] = editor.after(600, run_validation)

        def save_cookie() -> None:
            canonical = state.get("canonical")
            validation = state.get("validation")
            if not canonical or validation is None:
                return
            try:
                self.cookie_store.save(canonical, validation)
            except Exception as exc:
                messagebox.showerror("Cookie 保存失败", str(exc), parent=editor)
                return
            self.netease_cookie_status_var.set(validation.message)
            editor.destroy()

        save_btn.config(command=save_cookie)
        text_widget.bind("<<Modified>>", schedule_validation)
        text_widget.edit_modified(False)
        self._fit_toplevel(
            editor,
            preferred_width=820,
            preferred_height=620,
            minimum_width=640,
            minimum_height=480,
        )
        schedule_validation()

    def validate_saved_cookie(self) -> None:
        try:
            cookies = self._load_saved_cookies()
        except Exception as exc:
            self.netease_cookie_status_var.set(f"Cookie 读取失败：{exc}")
            return
        if not cookies:
            self.netease_cookie_status_var.set("未保存网易云 Cookie")
            return
        self.netease_cookie_status_var.set("正在验证网易云登录状态…")

        def worker() -> None:
            result = validate_cookie_account(cookies)
            self._after(self._finish_saved_cookie_validation, result)

        self._start_worker(worker)

    def _finish_saved_cookie_validation(self, result: CookieValidationResult) -> None:
        self.netease_cookie_status_var.set(result.message)
        if result.state == "invalid":
            return
        try:
            text = self.cookie_store.load_text()
            if text:
                self.cookie_store.save(text, result)
        except Exception:
            pass

    def clear_cookie(self) -> None:
        if not messagebox.askyesno(
            "清除网易云 Cookie",
            "确定删除本机加密保存的网易云 Cookie 吗？",
            parent=self.win,
        ):
            return
        try:
            self.cookie_store.clear()
            self.netease_cookie_status_var.set("未保存网易云 Cookie")
        except Exception as exc:
            messagebox.showerror("清除失败", str(exc), parent=self.win)

    def search_netease(self, page: int = 0) -> None:
        if self.busy:
            return
        query = self.netease_query_var.get().strip()
        if not query:
            self.netease_result_var.set("请输入歌曲名或歌手名")
            return
        page = max(0, int(page))
        self.search_generation += 1
        generation = self.search_generation
        self.netease_result_var.set("正在搜索网易云…")
        self.netease_search_btn.config(state="disabled")
        self.netease_prev_btn.config(state="disabled")
        self.netease_next_btn.config(state="disabled")
        self.netease_generate_btn.config(state="disabled")
        try:
            cookies = self._load_saved_cookies()
        except Exception:
            cookies = []
            self.netease_cookie_status_var.set("Cookie 无法读取，本次按未登录搜索")

        def worker() -> None:
            try:
                result = self.netease_client.search(
                    query,
                    offset=page * self.search_limit,
                    limit=self.search_limit,
                    cookies=cookies,
                )
                error = None
            except Exception as exc:
                result = None
                error = str(exc)
            self._after(self._apply_search_result, generation, page, result, error)

        self._start_worker(worker)

    def _apply_search_result(
        self,
        generation: int,
        page: int,
        result: Optional[NetEaseSearchPage],
        error: Optional[str],
    ) -> None:
        if generation != self.search_generation:
            return
        self.netease_search_btn.config(state="disabled" if self.busy else "normal")
        if error or result is None:
            self.netease_result_var.set(error or "网易云搜索失败")
            self._set_busy(self.busy)
            return
        self.search_page = page
        self.search_total = result.total
        self.search_tracks = {track.song_id: track for track in result.items}
        self.netease_tree.delete(*self.netease_tree.get_children())
        for track in result.items:
            seconds = max(0, track.duration_ms // 1000)
            duration = f"{seconds // 60}:{seconds % 60:02d}"
            self.netease_tree.insert(
                "",
                "end",
                iid=track.song_id,
                values=(track.title, track.artist_text, track.album, duration),
            )
        page_count = max(1, (result.total + result.limit - 1) // result.limit)
        self.netease_page_var.set(f"第 {page + 1} / {page_count} 页")
        self.netease_result_var.set(f"找到 {result.total} 首，当前显示 {len(result.items)} 首")
        self.netease_prev_btn.config(state="normal" if page > 0 else "disabled")
        has_next = result.offset + len(result.items) < result.total
        self.netease_next_btn.config(state="normal" if has_next else "disabled")
        self.netease_generate_btn.config(state="disabled")
        self._set_busy(self.busy)

    def search_previous_page(self) -> None:
        if self.search_page > 0:
            self.search_netease(self.search_page - 1)

    def search_next_page(self) -> None:
        self.search_netease(self.search_page + 1)

    def _selected_netease_track(self) -> Optional[NetEaseTrack]:
        selected = self.netease_tree.selection()
        if not selected:
            return None
        return self.search_tracks.get(str(selected[0]))

    def _on_netease_selection(self, _event=None) -> None:
        self.netease_generate_btn.config(
            state="normal" if self._selected_netease_track() and not self.busy else "disabled"
        )

    def generate_online_draft(self) -> None:
        if self.busy:
            return
        track = self._selected_netease_track()
        if track is None:
            self.netease_result_var.set("请先选择一首搜索结果")
            return
        self.cancel_event = threading.Event()
        options = self._options()
        self._set_busy(True)
        self.progress_var.set(0.0)
        self.status_var.set(f"正在准备在线扒谱：{track.display_name}")
        temp_root = self._temp_root
        if temp_root is None:
            self._finish_online_job(None, track, None, "临时目录已关闭")
            return
        job_dir = tempfile.mkdtemp(prefix=f"{track.song_id}-", dir=temp_root.name)
        self._start_worker(self._online_worker, track, job_dir, options)

    def _online_worker(
        self,
        track: NetEaseTrack,
        job_dir: str,
        options: TranscriptionOptions,
    ) -> None:
        audio_path = None
        success = False
        try:
            def download_progress(stage: str, fraction: float, message: str) -> None:
                stage_start, stage_weight = {
                    "resolve": (0.00, 0.05),
                    "download": (0.05, 0.25),
                    "convert": (0.30, 0.10),
                }.get(stage, (0.0, 0.4))
                self._after(
                    self._apply_progress,
                    stage_start + stage_weight * fraction,
                    message,
                    track.display_name,
                )

            with self.cookie_store.materialize_cookiefile() as cookiefile:
                audio_path = self.netease_client.resolve_and_download(
                    track,
                    cookiefile=cookiefile,
                    temp_dir=job_dir,
                    cancel_event=self.cancel_event,
                    progress_cb=download_progress,
                )

            def transcribe_progress(stage: str, fraction: float, message: str) -> None:
                if options.mode == "stem_fusion":
                    stage_start, stage_weight = {
                        "separate": (0.40, 0.30),
                        "transcribe": (0.70, 0.22),
                        "arrange": (0.92, 0.08),
                    }.get(stage, (0.40, 0.60))
                else:
                    stage_start, stage_weight = {
                        "decode": (0.40, 0.05),
                        "transcribe": (0.45, 0.45),
                        "arrange": (0.90, 0.10),
                    }.get(stage, (0.40, 0.60))
                self._after(
                    self._apply_progress,
                    stage_start + stage_weight * fraction,
                    message,
                    track.display_name,
                )

            result = transcribe_draft(
                audio_path,
                options,
                self.cancel_event,
                transcribe_progress,
                job_dir,
            )
            result = replace(
                result,
                source_file=track.display_name,
                source=SourceMetadata(
                    platform="netease",
                    title=track.title,
                    artists=track.artists,
                    source_id=track.song_id,
                    webpage_url=track.webpage_url,
                    display_name=track.display_name,
                ),
            )
            success = True
            self._after(self._finish_online_job, audio_path, track, result, None)
        except CancelledError:
            self._after(self._finish_online_job, None, track, None, "已取消在线扒谱")
        except Exception as exc:
            self._after(self._finish_online_job, None, track, None, str(exc))
        finally:
            if not success:
                try:
                    shutil.rmtree(job_dir)
                except OSError:
                    pass

    def _finish_online_job(
        self,
        audio_path: Optional[str],
        track: NetEaseTrack,
        result: Optional[TranscriptionResult],
        error: Optional[str],
    ) -> None:
        if result is not None and audio_path:
            self.files.append(audio_path)
            self.online_paths.add(audio_path)
            self.source_labels[audio_path] = track.display_name
            self.file_list.insert(tk.END, f"… {track.display_name}")
            self._record_result(audio_path, result, None)
            index = len(self.files) - 1
            self.file_list.selection_clear(0, tk.END)
            self.file_list.selection_set(index)
            self.file_list.activate(index)
            self.file_list.see(index)
            self._show_path(audio_path)
            self.progress_var.set(1.0)
            self.status_var.set("在线草稿已生成，可调参、试听并确认保存")
        else:
            self.status_var.set(error or "在线扒谱失败")
            if error and error != "已取消在线扒谱":
                messagebox.showerror("在线扒谱失败", error, parent=self.win)
        self.detail_var.set("")
        self._set_busy(False)

    def _options(self) -> TranscriptionOptions:
        octave_text = self.octave_var.get()
        source_key = None if self.key_var.get() == "自动" else self.key_var.get()
        octave = None if octave_text == "自动" else int(octave_text)
        enabled_stems = tuple(
            stem
            for stem in DEFAULT_ENABLED_STEMS
            if self.stem_enabled_vars[stem].get()
        )
        return TranscriptionOptions(
            mode=MODE_LABELS[self.mode_var.get()],
            sensitivity=SENSITIVITY_LABELS[self.sensitivity_var.get()],
            source_key=source_key,
            octave_shift=octave,
            quantize=QUANTIZE_LABELS[self.quantize_var.get()],
            max_polyphony=int(self.polyphony_var.get()),
            repeat_cleanup=REPEAT_CLEANUP_LABELS[self.repeat_cleanup_var.get()],
            enabled_stems=enabled_stems,
            use_drum_timing=bool(self.use_drum_timing_var.get()),
            fusion_profile=FUSION_PROFILE_LABELS[self.fusion_profile_var.get()],
            instrumental_policy="smart_fill",
        )

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.regenerate_btn.config(state=state if self._current_result() else "disabled")
        self.preview_btn.config(state=state if self._current_result() else "disabled")
        self.save_btn.config(state=state if self._current_result() else "disabled")
        self.save_all_btn.config(state=state if self.results else "disabled")
        self.cancel_btn.config(state="normal" if busy else "disabled")
        self.local_choose_btn.config(state=state)
        local_pending = any(
            path not in self.results and path not in self.online_paths for path in self.files
        )
        self.local_generate_btn.config(
            state="normal" if not busy and local_pending else "disabled"
        )
        self.netease_search_btn.config(state=state)
        self.cookie_edit_btn.config(state=state)
        self.cookie_validate_btn.config(state=state)
        self.cookie_clear_btn.config(state=state)
        self.netease_generate_btn.config(
            state="normal"
            if not busy and self._selected_netease_track() is not None
            else "disabled"
        )
        if busy:
            self.netease_prev_btn.config(state="disabled")
            self.netease_next_btn.config(state="disabled")
        elif self.search_tracks:
            self.netease_prev_btn.config(
                state="normal" if self.search_page > 0 else "disabled"
            )
            has_next = (self.search_page + 1) * self.search_limit < self.search_total
            self.netease_next_btn.config(state="normal" if has_next else "disabled")
        self._update_stem_panel(self._current_result())

    def generate_all(self) -> None:
        if self.busy:
            return
        paths = [
            path
            for path in self.files
            if path not in self.online_paths and path not in self.results
        ]
        if not paths:
            self.status_var.set("没有待生成的本地文件")
            return
        self.cancel_event = threading.Event()
        options = self._options()
        self._set_busy(True)
        self.status_var.set("正在生成草稿…")
        self._start_worker(self._generate_worker, paths, options)

    def _generate_worker(
        self,
        paths: Sequence[str],
        options: TranscriptionOptions,
    ) -> None:
        total = len(paths)
        cancelled = False
        for index, path in enumerate(paths):
            if self.cancel_event.is_set():
                cancelled = True
                break

            def progress(stage: str, fraction: float, message: str) -> None:
                if options.mode == "stem_fusion":
                    stage_start, stage_weight = {
                        "separate": (0.00, 0.45),
                        "transcribe": (0.45, 0.45),
                        "arrange": (0.90, 0.10),
                    }.get(stage, (0.0, 1.0))
                else:
                    stage_start, stage_weight = {
                        "decode": (0.00, 0.10),
                        "transcribe": (0.10, 0.75),
                        "arrange": (0.85, 0.15),
                    }.get(stage, (0.0, 1.0))
                file_fraction = stage_start + stage_weight * fraction
                overall = (index + file_fraction) / max(1, total)
                self._after(self._apply_progress, overall, message, os.path.basename(path))

            try:
                temp_root = self._temp_root
                workspace = temp_root.name if temp_root is not None else None
                result = transcribe_draft(
                    path,
                    options,
                    self.cancel_event,
                    progress,
                    workspace,
                )
                self._after(self._record_result, path, result, None)
            except CancelledError:
                cancelled = True
                break
            except Exception as exc:
                self._after(self._record_result, path, None, str(exc))
        self._after(self._finish_batch, cancelled)

    def _after(self, callback: Callable, *args) -> None:
        if self.closed:
            return
        try:
            self.win.after(0, callback, *args)
        except Exception:
            pass

    def _apply_progress(self, fraction: float, message: str, detail: str) -> None:
        self.progress_var.set(max(0.0, min(1.0, fraction)))
        self.status_var.set(message)
        self.detail_var.set(detail)

    def _file_index(self, path: str) -> int:
        return self.files.index(path)

    def _record_result(
        self,
        path: str,
        result: Optional[TranscriptionResult],
        error: Optional[str],
    ) -> None:
        index = self._file_index(path)
        if result is not None:
            previous = self.results.get(path)
            self.results[path] = result
            if (
                previous is not None
                and previous.artifact_root
                and previous.artifact_root != result.artifact_root
            ):
                cleanup_result_artifacts(previous)
            self.errors.pop(path, None)
            self.file_list.delete(index)
            self.file_list.insert(
                index,
                f"✓ {self.source_labels.get(path, os.path.basename(path))} · "
                f"{len(result.song_notes)} 音",
            )
            if not self.file_list.curselection():
                self.file_list.selection_set(index)
                self.file_list.activate(index)
                self._show_path(path)
        else:
            self.errors[path] = error or "未知错误"
            self.file_list.delete(index)
            self.file_list.insert(
                index, f"✗ {self.source_labels.get(path, os.path.basename(path))}"
            )

    def _finish_batch(self, cancelled: bool) -> None:
        self.progress_var.set(1.0 if not cancelled else self.progress_var.get())
        if cancelled:
            self.status_var.set("已取消，未保存任何未确认草稿")
        else:
            self.status_var.set(
                f"草稿完成：成功 {len(self.results)}，失败 {len(self.errors)}"
            )
        self.detail_var.set("")
        self._set_busy(False)
        if not self.results and self.errors and not cancelled:
            preview = "\n".join(
                f"{os.path.basename(path)}：{error}"
                for path, error in list(self.errors.items())[:5]
            )
            messagebox.showerror("生成失败", preview, parent=self.win)

    def _selected_path(self) -> Optional[str]:
        selection = self.file_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        return self.files[index] if 0 <= index < len(self.files) else None

    def _current_result(self) -> Optional[TranscriptionResult]:
        path = self._selected_path()
        return self.results.get(path) if path else None

    def _on_selection(self, _event=None) -> None:
        path = self._selected_path()
        if path:
            self._show_path(path)
        self._set_busy(self.busy)

    def _show_path(self, path: str) -> None:
        result = self.results.get(path)
        if result is None:
            self.stats_var.set(self.errors.get(path, "草稿尚未完成"))
            self.timeline.delete("all")
            self._update_stem_panel(None)
            return
        if result.options.mode == "stem_fusion":
            enabled = set(result.options.enabled_stems)
            for stem, variable in self.stem_enabled_vars.items():
                variable.set(stem in enabled)
            self.use_drum_timing_var.set(result.options.use_drum_timing)
            profile_label = next(
                (
                    label
                    for label, value in FUSION_PROFILE_LABELS.items()
                    if value == result.options.fusion_profile
                ),
                "人声优先",
            )
            self.fusion_profile_var.set(profile_label)
        stats = result.stats
        warning = f"\n提示：{'；'.join(result.warnings)}" if result.warnings else ""
        self.stats_var.set(
            f"引擎：{result.engine}    检测调性：{result.detected_key}    "
            f"移调：{result.semitone_shift:+d} 半音    八度：{result.octave_shift:+d} 半音\n"
            f"BPM：{result.bpm:.1f}    音符：{len(result.song_notes)}    "
            f"和弦：{stats.get('chord_count', 0)}    折叠：{stats.get('folded_count', 0)}    "
            f"过滤：{stats.get('filtered_count', 0)}\n"
            f"参考节拍：{stats.get('timing_reference_bpm', result.bpm):.1f} BPM    "
            f"合并碎片：{stats.get('source_fragment_merged_count', 0)}    "
            f"抑制重复：{stats.get('mapped_repeat_suppressed_count', 0)}{warning}"
        )
        self._update_stem_panel(result)
        self._draw_timeline()

    def _update_stem_panel(
        self,
        result: Optional[TranscriptionResult],
    ) -> None:
        for stem, controls in self.stem_controls.items():
            status = controls["status"]
            preview = controls["preview"]
            toggle = controls["toggle"]
            if result is None or stem not in result.stems:
                status.config(text="尚未生成")
                preview.config(state="disabled")
                toggle.config(state="normal" if not self.busy else "disabled")
                continue
            stem_result = result.stems[stem]
            if stem == "drums":
                beat_count = int(stem_result.stats.get("beat_count", 0))
                text = f"{beat_count} 个节拍"
            else:
                text = f"{len(stem_result.events)} 个音符"
            if stem_result.warnings:
                detail = str(stem_result.warnings[0]).replace("\n", " ")
                text += f" · 提示：{detail[:32]}"
            status.config(text=text)
            preview.config(
                state="normal"
                if os.path.isfile(stem_result.audio_path) and not self.busy
                else "disabled"
            )
            toggle.config(state="normal" if not self.busy else "disabled")

    def _draw_timeline(self) -> None:
        canvas = self.timeline
        canvas.delete("all")
        result = self._current_result()
        if result is None or not result.song_notes:
            return
        width = max(100, canvas.winfo_width())
        height = max(160, canvas.winfo_height())
        left = 42
        right = 8
        row_height = height / 15.0
        times = [int(note["time"]) for note in result.song_notes]
        start = min(times)
        span = max(1, max(times) - start)

        for key_index in range(15):
            display_row = 14 - key_index
            y0 = display_row * row_height
            y1 = y0 + row_height
            fill = "#F7F9FC" if key_index % 2 else "#FFFFFF"
            canvas.create_rectangle(0, y0, width, y1, fill=fill, outline="")
            canvas.create_text(
                5, (y0 + y1) / 2, text=f"K{key_index}", anchor="w", fill="#667"
            )
            canvas.create_line(left, y1, width - right, y1, fill="#E6EAF0")

        plot_width = max(1, width - left - right)
        for note in result.song_notes:
            key = str(note["key"])
            try:
                key_index = int(key[4:])
            except Exception:
                continue
            display_row = 14 - key_index
            x = left + (int(note["time"]) - start) / span * plot_width
            y0 = display_row * row_height + 2
            y1 = (display_row + 1) * row_height - 2
            canvas.create_rectangle(
                x - 1.5, y0, x + 2.5, y1, fill=self.accent, outline=""
            )

    def regenerate_current(self) -> None:
        if self.busy:
            return
        path = self._selected_path()
        previous = self._current_result()
        if not path or previous is None:
            return
        options = self._options()
        self.cancel_event = threading.Event()
        self._set_busy(True)
        self.status_var.set("正在重新生成当前草稿…")

        def worker() -> None:
            def progress(stage: str, fraction: float, message: str) -> None:
                self._after(
                    self._apply_progress,
                    fraction,
                    message,
                    self.source_labels.get(path, os.path.basename(path)),
                )

            try:
                same_backend = (
                    is_midi_file(path)
                    or (
                        previous.options.mode == options.mode
                        and previous.options.sensitivity == options.sensitivity
                    )
                )
                if same_backend:
                    updated = rearrange_draft(
                        previous, options, self.cancel_event, progress
                    )
                else:
                    temp_root = self._temp_root
                    workspace = temp_root.name if temp_root is not None else None
                    updated = transcribe_draft(
                        path,
                        options,
                        self.cancel_event,
                        progress,
                        workspace,
                    )
                self._after(self._record_result, path, updated, None)
                self._after(self._finish_regenerate, None)
            except CancelledError:
                self._after(self._finish_regenerate, "已取消重新生成")
            except Exception as exc:
                self._after(self._finish_regenerate, str(exc))

        self._start_worker(worker)

    def _finish_regenerate(self, error: Optional[str]) -> None:
        self._set_busy(False)
        if error:
            self.status_var.set(error)
            if error != "已取消重新生成":
                messagebox.showerror("重新生成失败", error, parent=self.win)
        else:
            self.progress_var.set(1.0)
            self.status_var.set("当前草稿已更新")
            path = self._selected_path()
            if path:
                self._show_path(path)

    def preview_current(self) -> None:
        result = self._current_result()
        if result is None or self._preview_preparing:
            return
        self.stem_preview_player.stop()
        if self.preview_player.state in (
            PlaybackState.PLAYING,
            PlaybackState.PAUSED,
        ):
            state = self.preview_player.pause_or_resume()
            if state == PlaybackState.PAUSED:
                self.preview_btn.config(text="继续光遇音色试听")
                self.status_var.set("光遇音色试听已暂停")
            else:
                self.preview_btn.config(text="暂停光遇音色试听")
                self.status_var.set("正在使用光遇游戏音色试听")
            return
        if self.preview_player.prepared:
            self._start_score_preview(result)
            return

        self._preview_preparing = True
        self.preview_btn.config(state="disabled", text="准备光遇音色…")
        self.status_var.set("正在准备光遇游戏音色…")

        def worker() -> None:
            try:
                def progress(completed: int, total: int, _index: int) -> None:
                    self._after(
                        self.status_var.set,
                        f"正在准备光遇游戏音色… {completed}/{total}",
                    )

                self.preview_player.prepare(progress_callback=progress)
                self._after(self._finish_preview_prepare, result, None)
            except Exception as exc:
                self._after(self._finish_preview_prepare, result, str(exc))

        self._start_worker(worker)

    def _finish_preview_prepare(
        self,
        result: TranscriptionResult,
        error: Optional[str],
    ) -> None:
        self._preview_preparing = False
        if error:
            self.preview_btn.config(
                state="normal" if self._current_result() else "disabled",
                text="光遇音色试听",
            )
            self.status_var.set("光遇音色准备失败")
            messagebox.showerror("试听失败", error, parent=self.win)
            return
        self._start_score_preview(result)

    def _start_score_preview(self, result: TranscriptionResult) -> None:
        try:
            self.preview_player.play(result)
            self.preview_btn.config(text="暂停光遇音色试听")
            self.status_var.set("正在使用光遇游戏音色试听")
        except Exception as exc:
            self.preview_btn.config(text="光遇音色试听")
            messagebox.showerror("试听失败", str(exc), parent=self.win)
            self.status_var.set("试听失败")

    def preview_stem(self, stem: StemKind) -> None:
        result = self._current_result()
        if result is None or stem not in result.stems:
            return
        self.preview_player.stop()
        try:
            self.stem_preview_player.play(result.stems[stem].audio_path)
            self.preview_btn.config(text="光遇音色试听")
            self.status_var.set(f"正在试听{STEM_LABELS[stem]}分轨原声")
        except Exception as exc:
            messagebox.showerror("分轨试听失败", str(exc), parent=self.win)
            self.status_var.set("分轨试听失败")

    def stop_all_previews(self) -> None:
        self.preview_player.stop()
        self.stem_preview_player.stop()
        if not self.closed:
            self.preview_btn.config(
                text="光遇音色试听",
                state="normal" if self._current_result() and not self.busy else "disabled",
            )
            self.status_var.set("试听已停止")

    def _on_score_preview_status(self, message: str) -> None:
        self._after(self.status_var.set, message)

    def _on_score_preview_finished(self) -> None:
        self._after(self._finish_score_preview_ui)

    def _finish_score_preview_ui(self) -> None:
        self.preview_btn.config(
            text="光遇音色试听",
            state="normal" if self._current_result() and not self.busy else "disabled",
        )
        self.status_var.set("光遇音色试听结束")

    def _on_score_preview_error(self, exc: Exception) -> None:
        self._after(self._show_score_preview_error, str(exc))

    def _show_score_preview_error(self, detail: str) -> None:
        self.stop_all_previews()
        messagebox.showerror("光遇音色试听失败", detail, parent=self.win)

    def save_current(self) -> None:
        path = self._selected_path()
        result = self._current_result()
        if not path or result is None:
            return
        stem = suggested_output_stem(result)
        suggested = next_available_path(self.output_dir, stem)
        destination = filedialog.asksaveasfilename(
            parent=self.win,
            title="保存生成的乐谱",
            initialdir=self.output_dir,
            initialfile=os.path.basename(suggested),
            defaultextension=".json",
            filetypes=[("JSON 乐谱", "*.json")],
        )
        if not destination:
            return
        if os.path.exists(destination) and not messagebox.askyesno(
            "确认覆盖",
            f"{os.path.basename(destination)} 已存在，确定覆盖吗？",
            parent=self.win,
        ):
            return
        try:
            export_song_json(
                result,
                destination,
                result.source.title
                if result.source is not None
                and result.source.platform == "netease"
                and result.source.title
                else os.path.splitext(os.path.basename(destination))[0],
            )
            self.saved_paths[path] = destination
            self.on_saved(destination)
            self.status_var.set(f"已保存：{os.path.basename(destination)}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.win)

    def save_all(self) -> None:
        if not self.results:
            return
        saved: List[str] = []
        failures: List[str] = []
        for path in self.files:
            result = self.results.get(path)
            if result is None:
                continue
            stem = suggested_output_stem(result)
            destination = next_available_path(self.output_dir, stem)
            try:
                song_name = (
                    result.source.title
                    if result.source is not None
                    and result.source.platform == "netease"
                    and result.source.title
                    else stem
                )
                export_song_json(result, destination, song_name)
                self.saved_paths[path] = destination
                saved.append(destination)
            except Exception as exc:
                failures.append(f"{os.path.basename(path)}：{exc}")
        if saved:
            self.on_saved(saved[-1])
        if failures:
            messagebox.showerror(
                "部分保存失败", "\n".join(failures[:8]), parent=self.win
            )
        else:
            messagebox.showinfo(
                "保存完成",
                f"已保存 {len(saved)} 份乐谱到 Sheet Music 文件夹。",
                parent=self.win,
            )
        self.status_var.set(f"已保存 {len(saved)} 份乐谱")

    def cancel(self) -> None:
        if self.busy:
            self.cancel_event.set()
            self.status_var.set("正在取消，将在当前分块结束后停止…")

    def close(self) -> None:
        self.closed = True
        self.cancel_event.set()
        self.stem_preview_player.stop()
        self.preview_player.close()
        try:
            self.win.destroy()
        except Exception:
            pass
        with self._workers_lock:
            cleanup = self._worker_count == 0
        if cleanup:
            self._cleanup_temp_root()
