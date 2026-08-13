"""Compact V2 transcription dialog: source, arrangement controls, review, save."""
from __future__ import annotations

import os
import tempfile
import threading
import tkinter as tk
import webbrowser
from dataclasses import replace
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Sequence

from .arranger import NOTE_NAMES
from .backends import is_midi_file
from .models import CancelledError, SourceMetadata, TranscriptionOptions, TranscriptionResult
from .netease import NetEaseClient, NetEaseTrack
from .netease_auth import CookieValidationResult, NetEaseCookieStore, parse_netscape_cookies, serialize_netscape_cookies, validate_cookie_account
from .pipeline import cleanup_result_artifacts, export_song_json, next_available_path, rearrange_draft, refine_region, suggested_output_stem, transcribe_draft
from .preview import PreviewPlayer
from .quality import choose_quality_device, prepare_quality_model, resolve_quality_model


PRESET_LABELS = {"自动编配": "auto", "简洁": "simple", "标准": "standard", "丰满": "full"}
METER_LABELS = {"自动": "auto", "4/4": "4/4", "3/4": "3/4", "6/8": "6/8"}
ENGINE_LABELS = {"自动（当前为快速）": "auto", "快速（离线）": "fast", "高质量（MuScriptor）": "quality"}
QUALITY_MODEL_LABELS = {"自动选择": "auto", "Small（CPU）": "small", "Medium（GPU / Apple 芯片）": "medium"}


class TranscriptionDialog:
    """V2 UI.  Inputs are analysed once; arranging controls reuse the analysis draft."""

    def __init__(
        self,
        parent: tk.Misc,
        files: Sequence[str],
        output_dir: str,
        accent: str = "#4F8CFF",
        on_saved: Optional[Callable[[str], None]] = None,
        auth_file: Optional[str] = None,
        netease_client: Optional[NetEaseClient] = None,
        **_legacy_kwargs: object,
    ) -> None:
        self.parent, self.files, self.output_dir = parent, list(files), output_dir
        self.accent, self.on_saved = accent, on_saved or (lambda _path: None)
        self.results: Dict[str, TranscriptionResult] = {}
        self.errors: Dict[str, str] = {}
        self.source_labels: Dict[str, str] = {path: os.path.basename(path) for path in self.files}
        self.cancel_event = threading.Event()
        self.preview_player = PreviewPlayer(on_status=self._preview_status, on_finished=self._preview_finished, on_error=self._preview_error)
        self._preview_preparing = False
        self._temp_root = tempfile.TemporaryDirectory(prefix="sky-arrangement-drafts-")
        self.cookie_store = NetEaseCookieStore(auth_file or os.path.join(os.path.dirname(os.path.abspath(output_dir)), "netease_auth.json"))
        self.netease_client = netease_client or NetEaseClient()
        self.search_tracks: List[NetEaseTrack] = []
        self.busy = False
        self.closed = False
        self.refine_start_ms: Optional[int] = None
        self.refine_end_ms: Optional[int] = None
        self.refinement_candidate: Optional[tuple[str, TranscriptionResult]] = None

        self.win = tk.Toplevel(parent)
        self.win.title("生成乐谱")
        self.win.transient(parent)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.minsize(780, 620)
        self.win.geometry("980x820")

        self.preset_var = tk.StringVar(value="自动编配")
        self.key_var = tk.StringVar(value="自动")
        self.octave_var = tk.StringVar(value="自动")
        self.polyphony_var = tk.IntVar(value=4)
        self.bpm_var = tk.StringVar(value="自动")
        self.meter_var = tk.StringVar(value="自动")
        self.engine_var = tk.StringVar(value="快速（离线）")
        self.quality_model_var = tk.StringVar(value="自动选择")
        self.rights_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="选择音频后生成草稿")
        self.detail_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="尚未生成草稿")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.query_var = tk.StringVar()
        self.cookie_status_var = tk.StringVar(value=self.cookie_store.load_validation().message)
        self._build_widgets()
        if self.files:
            self.win.after(100, self.generate_all)

    def _build_widgets(self) -> None:
        notebook = ttk.Notebook(self.win)
        notebook.pack(fill="x", padx=12, pady=(12, 6))
        local = ttk.Frame(notebook, padding=10)
        online = ttk.Frame(notebook, padding=10)
        notebook.add(local, text="本地文件")
        notebook.add(online, text="网易云在线")
        ttk.Button(local, text="选择音频或 MIDI", command=self.choose_local_files).pack(side="left")
        self.local_summary = ttk.Label(local, text=self._local_summary())
        self.local_summary.pack(side="left", padx=10)

        ttk.Entry(online, textvariable=self.query_var, width=36).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(online, text="搜索", command=self.search_netease).grid(row=0, column=1, padx=3)
        ttk.Button(online, text="Cookie", command=self.edit_cookie).grid(row=0, column=2, padx=3)
        ttk.Label(online, textvariable=self.cookie_status_var, foreground="#666").grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))
        online.columnconfigure(0, weight=1)

        options = ttk.LabelFrame(self.win, text="钢琴编配", padding=10)
        options.pack(fill="x", padx=12, pady=6)
        ttk.Label(options, text="预设").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.preset_box = ttk.Combobox(options, textvariable=self.preset_var, values=list(PRESET_LABELS), state="readonly", width=10)
        self.preset_box.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        key_values = ["自动", *(f"{name} major" for name in NOTE_NAMES), *(f"{name} minor" for name in NOTE_NAMES)]
        ttk.Label(options, text="原曲调性").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.key_box = ttk.Combobox(options, textvariable=self.key_var, values=key_values, state="readonly", width=11)
        self.key_box.grid(row=0, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(options, text="旋律八度").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        self.octave_box = ttk.Combobox(options, textvariable=self.octave_var, values=["自动", "-1", "0", "+1"], state="readonly", width=7)
        self.octave_box.grid(row=0, column=5, sticky="w", padx=4, pady=4)
        ttk.Label(options, text="最大复音").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        self.polyphony_box = ttk.Spinbox(options, from_=2, to=5, textvariable=self.polyphony_var, width=6, state="readonly")
        self.polyphony_box.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(options, text="BPM 修正").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        self.bpm_box = ttk.Combobox(options, textvariable=self.bpm_var, values=["自动", *map(str, range(40, 221, 5))], width=8)
        self.bpm_box.grid(row=1, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(options, text="拍号修正").grid(row=1, column=4, sticky="e", padx=4, pady=4)
        self.meter_box = ttk.Combobox(options, textvariable=self.meter_var, values=list(METER_LABELS), state="readonly", width=7)
        self.meter_box.grid(row=1, column=5, sticky="w", padx=4, pady=4)
        ttk.Label(options, text="扒谱引擎").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        self.engine_box = ttk.Combobox(options, textvariable=self.engine_var, values=list(ENGINE_LABELS), state="readonly", width=18)
        self.engine_box.grid(row=2, column=1, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Label(options, text="高质量模型").grid(row=2, column=3, sticky="e", padx=4, pady=4)
        self.quality_model_box = ttk.Combobox(options, textvariable=self.quality_model_var, values=list(QUALITY_MODEL_LABELS), state="readonly", width=20)
        self.quality_model_box.grid(row=2, column=4, columnspan=2, sticky="w", padx=4, pady=4)
        self.rights_check = ttk.Checkbutton(options, text="我确认拥有该音频及生成乐谱所需权利（高质量模式）", variable=self.rights_var)
        self.rights_check.grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        self.model_setup_btn = ttk.Button(options, text="配置/下载高质量模型…", command=self.configure_quality_model)
        self.model_setup_btn.grid(row=3, column=4, columnspan=2, sticky="w", padx=4, pady=4)

        progress = ttk.Frame(self.win)
        progress.pack(fill="x", padx=12, pady=6)
        ttk.Label(progress, textvariable=self.status_var).pack(anchor="w")
        ttk.Progressbar(progress, maximum=1.0, variable=self.progress_var).pack(fill="x", pady=4)
        ttk.Label(progress, textvariable=self.detail_var, foreground="#666").pack(anchor="w")

        content = ttk.Panedwindow(self.win, orient=tk.HORIZONTAL)
        content.pack(fill="both", expand=True, padx=12, pady=6)
        left, right = ttk.Frame(content, width=240), ttk.Frame(content)
        content.add(left, weight=1); content.add(right, weight=3)
        ttk.Label(left, text="草稿列表").pack(anchor="w")
        self.file_list = tk.Listbox(left, exportselection=False)
        self.file_list.pack(fill="both", expand=True, pady=(4, 0))
        self.file_list.bind("<<ListboxSelect>>", self._on_selection)
        for path in self.files:
            self.file_list.insert(tk.END, f"… {self.source_labels[path]}")
        ttk.Label(right, text="15 键时间线（蓝色：旋律区；灰色：伴奏区）").pack(anchor="w")
        self.timeline = tk.Canvas(right, background="#FFFFFF", highlightthickness=1, highlightbackground="#D9DEE7", height=340)
        self.timeline.pack(fill="both", expand=True, pady=(4, 6))
        self.timeline.bind("<Configure>", lambda _event: self._draw_timeline())
        self.timeline.bind("<ButtonPress-1>", self._begin_refine_selection)
        self.timeline.bind("<B1-Motion>", self._update_refine_selection)
        self.timeline.bind("<ButtonRelease-1>", self._finish_refine_selection)
        ttk.Label(right, textvariable=self.stats_var, justify="left", foreground="#555").pack(fill="x", anchor="w")

        actions = ttk.Frame(self.win)
        actions.pack(fill="x", padx=12, pady=(6, 12))
        self.regenerate_btn = ttk.Button(actions, text="应用参数", command=self.regenerate_current, state="disabled")
        self.regenerate_btn.pack(side="left", padx=(0, 6))
        self.preview_btn = ttk.Button(actions, text="钢琴音色试听", command=self.preview_current, state="disabled")
        self.preview_btn.pack(side="left", padx=6)
        self.refine_btn = ttk.Button(actions, text="精修所选小节", command=self.refine_current, state="disabled")
        self.refine_btn.pack(side="left", padx=6)
        self.accept_refine_btn = ttk.Button(actions, text="接受精修", command=self.accept_refinement, state="disabled")
        self.accept_refine_btn.pack(side="left", padx=6)
        self.discard_refine_btn = ttk.Button(actions, text="放弃精修", command=self.discard_refinement, state="disabled")
        self.discard_refine_btn.pack(side="left", padx=6)
        ttk.Button(actions, text="停止试听", command=self.stop_preview).pack(side="left", padx=6)
        self.save_btn = ttk.Button(actions, text="保存当前", command=self.save_current, state="disabled")
        self.save_btn.pack(side="right", padx=6)
        self.save_all_btn = ttk.Button(actions, text="保存全部", command=self.save_all, state="disabled")
        self.save_all_btn.pack(side="right", padx=6)
        ttk.Button(actions, text="取消", command=self.cancel).pack(side="right", padx=6)

        self.online_list = tk.Listbox(online, height=7, exportselection=False)
        self.online_list.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Button(online, text="生成所选在线草稿", command=self.generate_online_draft).grid(row=3, column=0, sticky="w")

    def _local_summary(self) -> str:
        return "尚未选择本地文件" if not self.files else f"已选择 {len(self.files)} 个文件"

    def _options(self) -> TranscriptionOptions:
        octave = None if self.octave_var.get() == "自动" else int(self.octave_var.get())
        bpm = None if self.bpm_var.get() == "自动" else float(self.bpm_var.get())
        return TranscriptionOptions(
            mode="audio_arrangement", arrangement_preset=PRESET_LABELS[self.preset_var.get()],
            source_key=None if self.key_var.get() == "自动" else self.key_var.get(), melody_octave_shift=octave,
            max_polyphony=int(self.polyphony_var.get()), bpm_override=bpm, meter=METER_LABELS[self.meter_var.get()],
            engine=ENGINE_LABELS[self.engine_var.get()], quality_model=QUALITY_MODEL_LABELS[self.quality_model_var.get()],
            rights_confirmed=bool(self.rights_var.get()),
        )

    def choose_local_files(self) -> None:
        paths = filedialog.askopenfilenames(parent=self.win, title="选择音频或 MIDI", filetypes=[("支持的文件", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.mid *.midi"), ("所有文件", "*.*")])
        added = [path for path in paths if path not in self.files]
        if not added:
            return
        self.files.extend(added)
        for path in added:
            self.source_labels[path] = os.path.basename(path)
            self.file_list.insert(tk.END, f"… {self.source_labels[path]}")
        self.local_summary.config(text=self._local_summary())
        self.generate_all()

    def _start_worker(self, target: Callable[[], None]) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def _after(self, callback: Callable, *args) -> None:
        if not self.closed:
            self.win.after(0, callback, *args)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        current = self._current_result()
        self.regenerate_btn.config(state="disabled" if busy or current is None else "normal")
        self.preview_btn.config(state="disabled" if busy or current is None else "normal")
        can_refine = current is not None and current.quality_analysis is not None and self.refine_start_ms is not None and self.refine_end_ms is not None and self.refinement_candidate is None
        self.refine_btn.config(state="normal" if not busy and can_refine else "disabled")
        self.accept_refine_btn.config(state="normal" if not busy and self.refinement_candidate is not None else "disabled")
        self.discard_refine_btn.config(state="normal" if not busy and self.refinement_candidate is not None else "disabled")
        self.save_btn.config(state="disabled" if busy or current is None else "normal")
        self.save_all_btn.config(state="disabled" if busy or not self.results else "normal")
        for control in (self.preset_box, self.key_box, self.octave_box, self.polyphony_box, self.bpm_box, self.meter_box, self.engine_box, self.quality_model_box):
            control.config(state="disabled" if busy else "readonly")
        self.bpm_box.config(state="disabled" if busy else "normal")
        self.rights_check.config(state="disabled" if busy else "normal")
        self.model_setup_btn.config(state="disabled" if busy else "normal")

    def generate_all(self) -> None:
        if self.busy or not self.files:
            return
        self.cancel_event = threading.Event(); self._set_busy(True); self.progress_var.set(0.0)
        options = self._options()
        pending = list(self.files)
        def worker() -> None:
            for index, path in enumerate(pending):
                if self.cancel_event.is_set():
                    break
                def progress(stage: str, fraction: float, message: str, _index=index) -> None:
                    weights = {"decode": (0.0, .05), "separate": (.05, .45), "timing": (.50, .10), "melody": (.60, .15), "harmony": (.75, .10), "structure": (.85, .05), "quality": (.05, .85), "arrange": (.90, .10), "refine": (.05, .90)}
                    start, weight = weights.get(stage, (0.0, 1.0))
                    self._after(self._apply_progress, (_index + start + weight * fraction) / len(pending), message, self.source_labels.get(path, os.path.basename(path)))
                try:
                    result = transcribe_draft(path, options, self.cancel_event, progress, self._temp_root.name)
                    self._after(self._record_result, path, result, None)
                except CancelledError:
                    break
                except Exception as exc:
                    self._after(self._record_result, path, None, str(exc))
            self._after(self._finish_batch, self.cancel_event.is_set())
        self._start_worker(worker)

    def regenerate_current(self) -> None:
        if self.busy or not (path := self._selected_path()) or not (previous := self.results.get(path)):
            return
        options = self._options(); self.cancel_event = threading.Event(); self._set_busy(True)
        def worker() -> None:
            try:
                requires_analysis = options.bpm_override != previous.options.bpm_override or options.meter != previous.options.meter
                if previous.analysis is not None and not requires_analysis:
                    result = rearrange_draft(previous, options, self.cancel_event, lambda _s, f, m: self._after(self._apply_progress, f, m, self.source_labels[path]))
                else:
                    result = transcribe_draft(path, options, self.cancel_event, lambda _s, f, m: self._after(self._apply_progress, f, m, self.source_labels[path]), self._temp_root.name)
                self._after(self._record_result, path, result, None)
                self._after(self._finish_regenerate, None)
            except Exception as exc:
                self._after(self._finish_regenerate, str(exc))
        self._start_worker(worker)

    def _apply_progress(self, fraction: float, message: str, detail: str) -> None:
        self.progress_var.set(max(0.0, min(1.0, fraction))); self.status_var.set(message); self.detail_var.set(detail)

    def _file_index(self, path: str) -> int:
        return self.files.index(path)

    def _record_result(self, path: str, result: Optional[TranscriptionResult], error: Optional[str]) -> None:
        index = self._file_index(path)
        if result is None:
            self.errors[path] = error or "未知错误"; text = f"✗ {self.source_labels[path]}"
        else:
            previous = self.results.get(path)
            self.results[path] = result; self.errors.pop(path, None); text = f"✓ {self.source_labels[path]} · {len(result.song_notes)} 音"
            if previous and previous.artifact_root and previous.artifact_root != result.artifact_root:
                cleanup_result_artifacts(previous)
        self.file_list.delete(index); self.file_list.insert(index, text)
        if not self.file_list.curselection():
            self.file_list.selection_set(index); self.file_list.activate(index); self._show_path(path)

    def _finish_batch(self, cancelled: bool) -> None:
        self._set_busy(False); self.detail_var.set("")
        self.status_var.set("已取消" if cancelled else f"草稿完成：成功 {len(self.results)}，失败 {len(self.errors)}")
        if self.results: self.progress_var.set(1.0)

    def _finish_regenerate(self, error: Optional[str]) -> None:
        self._set_busy(False)
        if error:
            self.status_var.set(error)
            if error != "转写已取消": messagebox.showerror("重新生成失败", error, parent=self.win)
        else:
            self.progress_var.set(1.0); self.status_var.set("当前草稿已更新")
            if path := self._selected_path(): self._show_path(path)

    def _selected_path(self) -> Optional[str]:
        selection = self.file_list.curselection()
        return self.files[int(selection[0])] if selection and int(selection[0]) < len(self.files) else None

    def _current_result(self) -> Optional[TranscriptionResult]:
        path = self._selected_path()
        if path and self.refinement_candidate and self.refinement_candidate[0] == path:
            return self.refinement_candidate[1]
        return self.results.get(path) if path else None

    def _on_selection(self, _event=None) -> None:
        if path := self._selected_path(): self._show_path(path)
        self._set_busy(self.busy)

    def _show_path(self, path: str) -> None:
        result = self.results.get(path)
        if result is None:
            self.stats_var.set(self.errors.get(path, "草稿尚未完成")); self.timeline.delete("all"); return
        stats, analysis = result.stats, result.analysis
        warning = f"\n提示：{'；'.join(result.warnings)}" if result.warnings else ""
        confidence = ""
        if analysis:
            confidence = f"\n拍速 {analysis.tempo_map.confidence:.2f} · 旋律 {analysis.melody_confidence:.2f} · 和弦 {analysis.harmony_confidence:.2f} · 段落 {analysis.structure_confidence:.2f}"
            self.preset_var.set(next(label for label, value in PRESET_LABELS.items() if value == result.options.arrangement_preset))
            self.key_var.set(result.options.source_key or "自动")
            self.octave_var.set("自动" if result.options.melody_octave_shift is None else f"{result.options.melody_octave_shift:+d}".replace("+0", "0"))
            self.polyphony_var.set(result.options.max_polyphony)
        self.engine_var.set(next(label for label, value in ENGINE_LABELS.items() if value == result.options.engine))
        self.quality_model_var.set(next(label for label, value in QUALITY_MODEL_LABELS.items() if value == result.options.quality_model))
        self.stats_var.set(f"引擎：{result.engine}    调性：{result.detected_key}    BPM：{result.bpm:.1f}\n音符：{len(result.song_notes)}    落点：{stats.get('onset_count', 0)}    和弦：{stats.get('chord_count', 0)}    平均复音：{stats.get('average_polyphony', 0)}{confidence}{warning}")
        self._draw_timeline()

    def _draw_timeline(self) -> None:
        canvas = self.timeline; canvas.delete("all"); result = self._current_result()
        if not result or not result.song_notes: return
        width, height, left, right = max(100, canvas.winfo_width()), max(160, canvas.winfo_height()), 42, 8
        row = height / 15.0; times = [int(note["time"]) for note in result.song_notes]; start, span = min(times), max(1, max(times) - min(times))
        for key in range(15):
            display = 14 - key; y0, y1 = display * row, (display + 1) * row
            canvas.create_rectangle(0, y0, width, y1, fill="#F7F9FC" if key % 2 else "#FFF", outline="")
            canvas.create_text(5, (y0 + y1) / 2, text=f"K{key}", anchor="w", fill="#667")
        if result.analysis:
            for section in result.analysis.sections:
                x = left + (section.start_ms - start) / span * (width - left - right)
                canvas.create_line(x, 0, x, height, fill="#D6DCE8", dash=(2, 2))
        if result.quality_analysis:
            for bar in result.quality_analysis.bar_starts_ms:
                x = left + (bar - start) / span * (width - left - right)
                canvas.create_line(x, 0, x, height, fill="#D6DCE8", dash=(2, 2))
        if self.refine_start_ms is not None and self.refine_end_ms is not None:
            lo, hi = sorted((self.refine_start_ms, self.refine_end_ms))
            x0 = left + (lo - start) / span * (width - left - right); x1 = left + (hi - start) / span * (width - left - right)
            canvas.create_rectangle(x0, 0, x1, height, fill="#FFF4C2", outline="#E8B931", stipple="gray25")
        for note in result.song_notes:
            key = int(str(note["key"])[4:]); display = 14 - key; x = left + (int(note["time"]) - start) / span * (width - left - right)
            canvas.create_rectangle(x - 1.5, display * row + 2, x + 2.5, (display + 1) * row - 2, fill=self.accent if key >= 7 else "#98A6BA", outline="")

    def _timeline_time(self, event: tk.Event) -> Optional[int]:
        result = self._current_result()
        if not result or not result.song_notes:
            return None
        times = [int(note["time"]) for note in result.song_notes]; start, span = min(times), max(1, max(times) - min(times))
        width, left, right = max(100, self.timeline.winfo_width()), 42, 8
        ratio = max(0.0, min(1.0, (event.x - left) / max(1, width - left - right)))
        return int(round(start + span * ratio))

    def _begin_refine_selection(self, event: tk.Event) -> None:
        if self.busy or not (result := self._current_result()) or result.quality_analysis is None or self.refinement_candidate:
            return
        self.refine_start_ms = self._timeline_time(event); self.refine_end_ms = self.refine_start_ms; self._draw_timeline()

    def _update_refine_selection(self, event: tk.Event) -> None:
        if self.refine_start_ms is not None:
            self.refine_end_ms = self._timeline_time(event); self._draw_timeline()

    def _finish_refine_selection(self, event: tk.Event) -> None:
        if self.refine_start_ms is None:
            return
        self.refine_end_ms = self._timeline_time(event)
        if self.refine_end_ms is not None and abs(self.refine_end_ms - self.refine_start_ms) < 100:
            self.refine_start_ms = self.refine_end_ms = None
        self._set_busy(self.busy); self._draw_timeline()

    def configure_quality_model(self) -> None:
        if self.busy:
            return
        window = tk.Toplevel(self.win); window.title("配置高质量模型"); window.transient(self.win); window.resizable(False, False)
        requested = QUALITY_MODEL_LABELS[self.quality_model_var.get()]
        model_name = resolve_quality_model(requested, choose_quality_device())
        model_url = f"https://huggingface.co/MuScriptor/muscriptor-{model_name}"
        ttk.Label(window, text=(
            "打开模型页后，如显示“granted access”，说明许可已就绪；无需再次授权。\n"
            "否则先同意共享联系信息与非商用条件，再创建一个 Read Token 用于本次下载。\n"
            "Token 只在本次下载期间使用，不会保存到配置或乐谱中。"
        ), justify="left").pack(padx=14, pady=(14, 8))
        links = ttk.Frame(window); links.pack(pady=(0, 4))
        ttk.Button(links, text=f"打开 MuScriptor {model_name.title()} 条件页", command=lambda: webbrowser.open(model_url)).pack(side="left", padx=4)
        ttk.Button(links, text="创建 Read Token", command=lambda: webbrowser.open("https://huggingface.co/settings/tokens/new?tokenType=read")).pack(side="left", padx=4)
        token_var = tk.StringVar()
        row = ttk.Frame(window); row.pack(fill="x", padx=14, pady=6)
        ttk.Label(row, text="临时 HF Token（可留空）").pack(side="left")
        ttk.Entry(row, textvariable=token_var, show="•", width=32).pack(side="left", padx=(8, 0))
        button = ttk.Button(window, text="下载并验证"); button.pack(pady=(4, 14))
        def start() -> None:
            button.config(state="disabled", text="正在准备…")
            def worker() -> None:
                try:
                    name, device = prepare_quality_model(requested, token_var.get().strip() or None)
                    self._after(done, f"MuScriptor {name} 已在 {device} 就绪", None)
                except Exception as exc:
                    self._after(done, "", str(exc))
            self._start_worker(worker)
        def done(message: str, error: Optional[str]) -> None:
            if error:
                button.config(state="normal", text="下载并验证"); messagebox.showerror("高质量模型", error, parent=window); return
            self.status_var.set(message); window.destroy()
        button.config(command=start)

    def refine_current(self) -> None:
        path = self._selected_path(); current = self.results.get(path or "")
        if self.busy or not path or not current or current.quality_analysis is None or self.refine_start_ms is None or self.refine_end_ms is None:
            return
        self.cancel_event = threading.Event(); self._set_busy(True)
        start, end, options = self.refine_start_ms, self.refine_end_ms, self._options()
        def worker() -> None:
            try:
                candidate = refine_region(current, path, start, end, options, self.cancel_event, lambda _s, f, m: self._after(self._apply_progress, f, m, self.source_labels[path]))
                self._after(self._finish_refinement, path, candidate, None)
            except Exception as exc:
                self._after(self._finish_refinement, path, None, str(exc))
        self._start_worker(worker)

    def _finish_refinement(self, path: str, candidate: Optional[TranscriptionResult], error: Optional[str]) -> None:
        self._set_busy(False)
        if error or candidate is None:
            self.status_var.set(error or "片段精修失败")
            if error != "转写已取消": messagebox.showerror("片段精修失败", error or "未知错误", parent=self.win)
            return
        self.refinement_candidate = (path, candidate)
        self.status_var.set("已生成精修试听，确认后才会替换当前草稿")
        self._set_busy(False); self._show_path(path)

    def accept_refinement(self) -> None:
        if not self.refinement_candidate:
            return
        path, candidate = self.refinement_candidate; previous = self.results.get(path)
        self.results[path] = candidate; self.refinement_candidate = None
        if previous and previous.artifact_root and previous.artifact_root != candidate.artifact_root:
            cleanup_result_artifacts(previous)
        self.refine_start_ms = self.refine_end_ms = None
        self.status_var.set("已接受片段精修")
        self._set_busy(False); self._show_path(path)

    def discard_refinement(self) -> None:
        if not self.refinement_candidate:
            return
        self.refinement_candidate = None; self.refine_start_ms = self.refine_end_ms = None
        self.status_var.set("已放弃片段精修")
        self._set_busy(False)
        if path := self._selected_path(): self._show_path(path)

    def preview_current(self) -> None:
        result = self._current_result()
        if not result or self.busy: return
        if self.preview_player.prepared:
            try:
                self.preview_player.play(result); self.preview_btn.config(text="暂停钢琴音色试听"); self.status_var.set("正在试听")
            except Exception as exc: messagebox.showerror("试听失败", str(exc), parent=self.win)
            return
        self._preview_preparing = True; self.preview_btn.config(state="disabled", text="准备音色…")
        def worker() -> None:
            try:
                self.preview_player.prepare(); self._after(self._finish_preview_prepare, result, None)
            except Exception as exc: self._after(self._finish_preview_prepare, result, str(exc))
        self._start_worker(worker)

    def _finish_preview_prepare(self, result: TranscriptionResult, error: Optional[str]) -> None:
        self._preview_preparing = False
        if error:
            self.preview_btn.config(text="钢琴音色试听", state="normal"); messagebox.showerror("试听失败", error, parent=self.win); return
        self.preview_player.play(result); self.preview_btn.config(text="暂停钢琴音色试听")

    def stop_preview(self) -> None:
        self.preview_player.stop()
        if not self.closed: self.preview_btn.config(text="钢琴音色试听", state="normal" if self._current_result() and not self.busy else "disabled")

    def _preview_status(self, message: str) -> None:
        if not self.closed: self.status_var.set(message)
    def _preview_finished(self) -> None:
        if not self.closed: self.win.after(0, self.stop_preview)
    def _preview_error(self, exc: Exception) -> None:
        if not self.closed: self.win.after(0, lambda: messagebox.showerror("试听失败", str(exc), parent=self.win))

    def edit_cookie(self) -> None:
        window = tk.Toplevel(self.win); window.title("网易云 Cookie"); window.transient(self.win); window.geometry("700x420")
        text = tk.Text(window, wrap="none"); text.pack(fill="both", expand=True, padx=10, pady=10)
        try: text.insert("1.0", self.cookie_store.load_text() or "")
        except Exception: pass
        save_button = ttk.Button(window, text="保存并验证")
        save_button.pack(pady=(0, 10))

        def finish_save(canonical: str, validation: CookieValidationResult, error: Optional[str]) -> None:
            save_button.configure(text="保存并验证", state="normal")
            if error:
                messagebox.showerror("Cookie 验证失败", error, parent=window)
                return
            try:
                self.cookie_store.save(canonical, validation)
                self.cookie_status_var.set(self.cookie_store.load_validation().message)
                window.destroy()
            except Exception as exc: messagebox.showerror("Cookie 无效", str(exc), parent=window)

        def save() -> None:
            try:
                canonical = serialize_netscape_cookies(parse_netscape_cookies(text.get("1.0", "end")))
            except Exception as exc:
                messagebox.showerror("Cookie 无效", str(exc), parent=window)
                return
            save_button.configure(text="正在联网验证…", state="disabled")

            def worker() -> None:
                validation = validate_cookie_account(parse_netscape_cookies(canonical))
                self._after(finish_save, canonical, validation, None)

            self._start_worker(worker)

        save_button.configure(command=save)

    def search_netease(self) -> None:
        query = self.query_var.get().strip()
        if not query or self.busy: return
        self._set_busy(True); self.status_var.set("正在搜索网易云")
        def worker() -> None:
            try:
                cookies = parse_netscape_cookies(self.cookie_store.load_text() or "")
                page = self.netease_client.search(query, cookies=cookies)
                self._after(self._apply_search, list(page.items), None)
            except Exception as exc: self._after(self._apply_search, [], str(exc))
        self._start_worker(worker)

    def _apply_search(self, tracks: List[NetEaseTrack], error: Optional[str]) -> None:
        self._set_busy(False); self.online_list.delete(0, tk.END); self.search_tracks = tracks
        if error: self.status_var.set(error); return
        for track in tracks: self.online_list.insert(tk.END, f"{track.display_name} · {track.album}")
        self.status_var.set(f"找到 {len(tracks)} 首结果")

    def generate_online_draft(self) -> None:
        selection = self.online_list.curselection()
        if not selection or self.busy: return
        track = self.search_tracks[int(selection[0])]; self.cancel_event = threading.Event(); self._set_busy(True)
        def worker() -> None:
            try:
                with self.cookie_store.materialize_cookiefile() as cookiefile:
                    path = self.netease_client.resolve_and_download(track, cookiefile=cookiefile, temp_dir=self._temp_root.name, cancel_event=self.cancel_event, progress_cb=lambda _s, f, m: self._after(self._apply_progress, f * .1, m, track.display_name))
                options = self._options()
                result = transcribe_draft(path, options, self.cancel_event, lambda _s, f, m: self._after(self._apply_progress, .1 + .9 * f, m, track.display_name), self._temp_root.name)
                result.source = SourceMetadata(platform="netease", title=track.title, artists=track.artists, source_id=track.song_id, webpage_url=track.webpage_url, display_name=track.display_name)
                self._after(self._add_online_result, path, track, result, None)
            except Exception as exc: self._after(self._add_online_result, "", track, None, str(exc))
        self._start_worker(worker)

    def _add_online_result(self, path: str, track: NetEaseTrack, result: Optional[TranscriptionResult], error: Optional[str]) -> None:
        self._set_busy(False)
        if result is None: self.status_var.set(error or "在线草稿失败"); messagebox.showerror("在线草稿失败", error or "未知错误", parent=self.win); return
        self.files.append(path); self.source_labels[path] = track.display_name; self.file_list.insert(tk.END, f"✓ {track.display_name} · {len(result.song_notes)} 音"); self.results[path] = result
        index = len(self.files) - 1; self.file_list.selection_clear(0, tk.END); self.file_list.selection_set(index); self._show_path(path); self.progress_var.set(1.0)

    def save_current(self) -> None:
        if not (path := self._selected_path()) or not (result := self.results.get(path)): return
        stem = suggested_output_stem(result); output = next_available_path(self.output_dir, stem)
        try: export_song_json(result, output, self.source_labels.get(path, stem)); self.on_saved(output); self.status_var.set(f"已保存：{os.path.basename(output)}")
        except Exception as exc: messagebox.showerror("保存失败", str(exc), parent=self.win)

    def save_all(self) -> None:
        for path, result in list(self.results.items()):
            try:
                output = next_available_path(self.output_dir, suggested_output_stem(result)); export_song_json(result, output, self.source_labels.get(path, os.path.basename(path))); self.on_saved(output)
            except Exception as exc: messagebox.showerror("保存失败", f"{os.path.basename(path)}：{exc}", parent=self.win); return
        self.status_var.set(f"已保存 {len(self.results)} 份乐谱")

    def cancel(self) -> None:
        self.cancel_event.set(); self.status_var.set("正在取消…")

    def close(self) -> None:
        if self.closed: return
        self.closed = True; self.cancel_event.set(); self.preview_player.close()
        for result in self.results.values(): cleanup_result_artifacts(result)
        self._temp_root.cleanup(); self.win.destroy()
