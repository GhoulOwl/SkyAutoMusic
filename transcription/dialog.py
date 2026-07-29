from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Sequence

from .arranger import NOTE_NAMES
from .backends import is_midi_file
from .models import (
    CancelledError,
    TranscriptionOptions,
    TranscriptionResult,
)
from .pipeline import (
    export_song_json,
    next_available_path,
    rearrange_draft,
    transcribe_draft,
)
from .preview import PreviewPlayer


MODE_LABELS = {
    "复音高质量（Basic Pitch）": "polyphonic",
    "单旋律快速（pYIN）": "monophonic",
}
SENSITIVITY_LABELS = {"低": "low", "普通": "normal", "高": "high"}
QUANTIZE_LABELS = {"关闭": "off", "八分音符": "1/8", "十六分音符": "1/16"}


class TranscriptionDialog:
    """批量生成内存草稿，并允许试听、调参和确认保存。"""

    def __init__(
        self,
        parent: tk.Misc,
        files: Sequence[str],
        output_dir: str,
        accent: str = "#4F8CFF",
        on_saved: Optional[Callable[[], None]] = None,
    ):
        self.parent = parent
        self.files = list(files)
        self.output_dir = output_dir
        self.accent = accent
        self.on_saved = on_saved or (lambda: None)
        self.results: Dict[str, TranscriptionResult] = {}
        self.errors: Dict[str, str] = {}
        self.saved_paths: Dict[str, str] = {}
        self.cancel_event = threading.Event()
        self.preview_player = PreviewPlayer()
        self.busy = False
        self.closed = False

        self.win = tk.Toplevel(parent)
        self.win.title("生成乐谱")
        self.win.geometry("850x680")
        self.win.minsize(720, 580)
        self.win.transient(parent)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.mode_var = tk.StringVar(value="复音高质量（Basic Pitch）")
        self.sensitivity_var = tk.StringVar(value="普通")
        self.key_var = tk.StringVar(value="自动")
        self.octave_var = tk.StringVar(value="自动")
        self.quantize_var = tk.StringVar(value="关闭")
        self.polyphony_var = tk.IntVar(value=3)
        self.status_var = tk.StringVar(value="准备生成草稿…")
        self.detail_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="尚未生成草稿")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_widgets()
        self.win.after(100, self.generate_all)

    def _build_widgets(self) -> None:
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

        actions = ttk.Frame(self.win)
        actions.pack(fill="x", padx=12, pady=(6, 12))
        self.regenerate_btn = ttk.Button(
            actions, text="重新生成当前", command=self.regenerate_current, state="disabled"
        )
        self.regenerate_btn.pack(side="left", padx=(0, 6))
        self.preview_btn = ttk.Button(
            actions, text="本地试听", command=self.preview_current, state="disabled"
        )
        self.preview_btn.pack(side="left", padx=6)
        self.stop_preview_btn = ttk.Button(
            actions, text="停止试听", command=self.preview_player.stop, state="normal"
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

    def _options(self) -> TranscriptionOptions:
        octave_text = self.octave_var.get()
        source_key = None if self.key_var.get() == "自动" else self.key_var.get()
        octave = None if octave_text == "自动" else int(octave_text)
        return TranscriptionOptions(
            mode=MODE_LABELS[self.mode_var.get()],
            sensitivity=SENSITIVITY_LABELS[self.sensitivity_var.get()],
            source_key=source_key,
            octave_shift=octave,
            quantize=QUANTIZE_LABELS[self.quantize_var.get()],
            max_polyphony=int(self.polyphony_var.get()),
        )

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.regenerate_btn.config(state=state if self._current_result() else "disabled")
        self.preview_btn.config(state=state if self._current_result() else "disabled")
        self.save_btn.config(state=state if self._current_result() else "disabled")
        self.save_all_btn.config(state=state if self.results else "disabled")
        self.cancel_btn.config(state="normal" if busy else "disabled")

    def generate_all(self) -> None:
        if self.busy:
            return
        self.cancel_event = threading.Event()
        options = self._options()
        self._set_busy(True)
        self.status_var.set("正在生成草稿…")
        threading.Thread(
            target=self._generate_worker,
            args=(list(self.files), options),
            daemon=True,
        ).start()

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
                stage_start, stage_weight = {
                    "decode": (0.00, 0.10),
                    "transcribe": (0.10, 0.75),
                    "arrange": (0.85, 0.15),
                }.get(stage, (0.0, 1.0))
                file_fraction = stage_start + stage_weight * fraction
                overall = (index + file_fraction) / max(1, total)
                self._after(self._apply_progress, overall, message, os.path.basename(path))

            try:
                result = transcribe_draft(
                    path, options, self.cancel_event, progress
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
            self.results[path] = result
            self.errors.pop(path, None)
            self.file_list.delete(index)
            self.file_list.insert(
                index,
                f"✓ {os.path.basename(path)} · {len(result.song_notes)} 音",
            )
            if not self.file_list.curselection():
                self.file_list.selection_set(index)
                self.file_list.activate(index)
                self._show_path(path)
        else:
            self.errors[path] = error or "未知错误"
            self.file_list.delete(index)
            self.file_list.insert(index, f"✗ {os.path.basename(path)}")

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
            return
        stats = result.stats
        warning = f"\n提示：{'；'.join(result.warnings)}" if result.warnings else ""
        self.stats_var.set(
            f"引擎：{result.engine}    检测调性：{result.detected_key}    "
            f"移调：{result.semitone_shift:+d} 半音    八度：{result.octave_shift:+d} 半音\n"
            f"BPM：{result.bpm:.1f}    音符：{len(result.song_notes)}    "
            f"和弦：{stats.get('chord_count', 0)}    折叠：{stats.get('folded_count', 0)}    "
            f"过滤：{stats.get('filtered_count', 0)}{warning}"
        )
        self._draw_timeline()

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
                    os.path.basename(path),
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
                    updated = transcribe_draft(
                        path, options, self.cancel_event, progress
                    )
                self._after(self._record_result, path, updated, None)
                self._after(self._finish_regenerate, None)
            except CancelledError:
                self._after(self._finish_regenerate, "已取消重新生成")
            except Exception as exc:
                self._after(self._finish_regenerate, str(exc))

        threading.Thread(target=worker, daemon=True).start()

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
        if result is None:
            return
        try:
            self.status_var.set("正在生成本地试听…")
            self.preview_player.play(result)
            self.status_var.set("正在本地试听")
        except Exception as exc:
            messagebox.showerror("试听失败", str(exc), parent=self.win)
            self.status_var.set("试听失败")

    def save_current(self) -> None:
        path = self._selected_path()
        result = self._current_result()
        if not path or result is None:
            return
        stem = os.path.splitext(os.path.basename(path))[0]
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
                os.path.splitext(os.path.basename(destination))[0],
            )
            self.saved_paths[path] = destination
            self.on_saved()
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
            stem = os.path.splitext(os.path.basename(path))[0]
            destination = next_available_path(self.output_dir, stem)
            try:
                export_song_json(result, destination, stem)
                self.saved_paths[path] = destination
                saved.append(destination)
            except Exception as exc:
                failures.append(f"{os.path.basename(path)}：{exc}")
        if saved:
            self.on_saved()
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
        self.preview_player.stop()
        try:
            self.win.destroy()
        except Exception:
            pass
