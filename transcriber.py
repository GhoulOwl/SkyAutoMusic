"""音频 → 乐谱 JSON 转写器。

输入: mp3 / wav / flac / ogg
输出: 与现有 997 份乐谱完全兼容的 JSON
      顶层: [{ "name": str, "bpm": int, "songNotes": [{time: int_ms, key: "1Key0..14"}, ...] }]

策略（用户已确认）:
- 固定 C 大调映射: i = max(0, min(14, int(round(midi - 60))))
  - MIDI 60 (C4) → 1Key0
  - MIDI 74 (D5) → 1Key14
  - 超范围软限幅（<0 → 0, >14 → 14），保留节奏
- 保留原曲和弦: librosa.piptrack 在每个 onset 附近取多个稳定音高
  按音高升序全部写入 songNotes；同 time 多 key 由现有播放器自动并按
- 不引入音符时值: 节奏由播放器全局 NOTE_HOLD 决定

向后兼容:
- JSON 顶层用数组（与 Lycoris.json 等 997 份乐谱完全一致）
- 字段名沿用 Lycoris 的 `name`，并补 `bpm=120`（load_music 读不到时默认 120）
- 可选 `_transcribe_stats` 仅供 GUI 展示，load_music 不会读取
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

# numpy/scipy 仅在真正转写时用到，单元测试只测纯映射逻辑，所以放在函数内懒导入
np = None  # type: ignore
_librosa = None  # type: ignore


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


class TranscribeStats:
    """单次转写的统计信息，便于 UI 展示和单元测试断言。"""

    __slots__ = ("onset_count", "note_count", "clamped_low", "clamped_high", "duration_sec")

    def __init__(self):
        self.onset_count = 0
        self.note_count = 0
        self.clamped_low = 0
        self.clamped_high = 0
        self.duration_sec = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "onset_count": self.onset_count,
            "note_count": self.note_count,
            "clamped_low": self.clamped_low,
            "clamped_high": self.clamped_high,
            "duration_sec": round(self.duration_sec, 3),
        }


class Transcriber:
    """音频 → 乐谱 note 列表。

    设计目标:
    - transcribe() 输入输出纯粹，方便单元测试
    - run() 包裹 transcribe + JSON 落盘 + 进度回调，方便 GUI 集成
    - 不引入 librosa 之外的重量级依赖
    """

    NUM_KEYS = 15
    DEFAULT_SR = 22050
    # 每个 onset 之后取多长窗做音高估计（秒）。短窗能避开相邻 onset 干扰。
    ONSET_PITCH_WINDOW = 0.08
    # 同一 onset 最多写出几个 key（和弦保留上限，避免一拍按十几键）
    MAX_POLYPHONY = 3
    # 静音/能量门限（piptrack magnitude 归一化后），低于此视为无音
    PITCH_MAG_THRESHOLD = 0.05

    def __init__(self, sr: int = DEFAULT_SR, midi_root: int = 60):
        global _librosa, np
        if _librosa is None:
            try:
                import librosa as _l
                import numpy as _np
                _librosa = _l
                np = _np
            except ImportError as e:
                raise RuntimeError(
                    "librosa / numpy 未安装，无法转写。请先运行: pip install librosa numpy soundfile"
                ) from e
        self.sr = sr
        self.midi_root = midi_root  # C4 = 60

    # ---------- 核心: audio_path -> note list ----------

    def transcribe(self, audio_path: str) -> Tuple[List[Dict[str, Any]], TranscribeStats]:
        """返回 (notes, stats)。

        notes: 已按 time 升序、每条 {time: int_ms, key: "1Key0..14"}。
        stats: onset 数、note 数、上下限被夹掉的次数、音频时长。
        """
        y, sr = _librosa.load(audio_path, sr=self.sr, mono=True)
        stats = TranscribeStats()
        stats.duration_sec = float(len(y) / sr) if sr else 0.0

        if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
            return [], stats

        # 1) onset 检测
        onset_times = _librosa.onset.onset_detect(
            y=y, sr=sr, units="time", backtrack=False
        )
        if len(onset_times) == 0:
            return [], stats
        stats.onset_count = int(len(onset_times))

        # 2) 整段 piptrack → 取每个 onset 之后窗内的稳定音高
        pitches, magnitudes = _librosa.piptrack(y=y, sr=sr)
        # pitches/magnitudes shape: (n_freq_bins, n_frames)
        n_frames = pitches.shape[1]
        hop_length = 512  # librosa 默认
        frame_times = np.arange(n_frames) * hop_length / sr

        notes: List[Dict[str, Any]] = []
        window_sec = self.ONSET_PITCH_WINDOW
        for t in onset_times:
            frame_pitches = self._collect_pitches_in_window(
                pitches, magnitudes, frame_times, float(t), window_sec
            )
            if not frame_pitches:
                continue
            # 按能量取最强的 MAX_POLYPHONY 个音，再按音高升序输出（低→高）
            frame_pitches.sort(key=lambda pm: pm[1], reverse=True)
            top = frame_pitches[: self.MAX_POLYPHONY]
            top.sort(key=lambda pm: pm[0])
            t_ms = int(round(float(t) * 1000))
            for midi, _mag in top:
                key, clamped = self._midi_to_key(midi)
                if clamped < 0:
                    stats.clamped_low += 1
                elif clamped > 0:
                    stats.clamped_high += 1
                notes.append({"time": t_ms, "key": key})

        # 3) 按 time 排序
        notes.sort(key=lambda n: (n["time"], n["key"]))
        stats.note_count = len(notes)
        return notes, stats

    def _collect_pitches_in_window(
        self,
        pitches: np.ndarray,
        magnitudes: np.ndarray,
        frame_times: np.ndarray,
        onset_t: float,
        window_sec: float,
    ) -> List[Tuple[float, float]]:
        """收集 onset_t 之后 window_sec 内的音高，按半音级聚合后取能量最大的若干个。

        返回 [(midi, mag), ...]，半音级去重（避免同 onset 内多帧重复写同一键）。
        """
        mask = (frame_times >= onset_t) & (frame_times < onset_t + window_sec)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            # onset 极接近结尾时也兜一个 frame
            j = int(np.searchsorted(frame_times, onset_t))
            j = max(0, min(j, len(frame_times) - 1))
            idxs = np.array([j])

        # 按半音整数聚合：同一 onset 窗内可能多帧检测到同一音高（甚至 ±0.5 半音抖动），取能量最高的代表
        bins: Dict[int, float] = {}
        for i in idxs:
            col_p = pitches[:, i]
            col_m = magnitudes[:, i]
            valid = (col_p > 0) & (col_m > self.PITCH_MAG_THRESHOLD)
            if not np.any(valid):
                continue
            for p, m in zip(col_p[valid], col_m[valid]):
                midi = _librosa.hz_to_midi(p)
                midi_round = int(round(float(midi)))
                m_f = float(m)
                if m_f > bins.get(midi_round, 0.0):
                    bins[midi_round] = m_f
        # 输出 (midi, mag)；注意用浮点 midi 但聚合是按整数
        return [(float(k), v) for k, v in bins.items()]

    def _midi_to_key(self, midi: float) -> Tuple[str, int]:
        """固定 C 大调映射 + 软限幅。返回 (key_name, clamped)，clamped: -1/0/1。"""
        i = int(round(midi)) - self.midi_root
        if i < 0:
            return f"1Key0", -1
        if i >= self.NUM_KEYS:
            return f"1Key{self.NUM_KEYS - 1}", 1
        return f"1Key{i}", 0

    # ---------- 包裹: 落盘 ----------

    def transcribe_to_song(
        self,
        audio_path: str,
        song_name: Optional[str] = None,
        bpm: int = 120,
    ) -> Dict[str, Any]:
        notes, stats = self.transcribe(audio_path)
        if song_name is None:
            song_name = os.path.splitext(os.path.basename(audio_path))[0]
        return {
            "name": song_name,
            "bpm": int(bpm),
            "songNotes": notes,
            "_transcribe_stats": stats.to_dict(),  # 仅供 UI 展示；load_music 会忽略
        }

    def write_song_json(
        self,
        audio_path: str,
        output_dir: str,
        song_name: Optional[str] = None,
        bpm: int = 120,
    ) -> Tuple[str, Dict[str, Any]]:
        song = self.transcribe_to_song(audio_path, song_name=song_name, bpm=bpm)
        os.makedirs(output_dir, exist_ok=True)
        out_name = (song_name or os.path.splitext(os.path.basename(audio_path))[0]) + ".json"
        out_path = os.path.join(output_dir, out_name)
        # 顶层用数组（与 Lycoris.json 等所有 997 份乐谱一致）
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([song], f, ensure_ascii=False)
        return out_path, song

    # ---------- 多文件批处理（GUI 入口） ----------

    def run(
        self,
        files: Iterable[str],
        output_dir: str,
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
        bpm: int = 120,
    ) -> List[Dict[str, Any]]:
        """逐文件转写。

        progress_cb(filename, fraction, status_text) 由 GUI 提供，用于驱动进度条。
        返回每条结果: {input, output, ok, error, stats, song_name}
        """
        files = [f for f in files if is_audio_file(f)]
        results: List[Dict[str, Any]] = []
        total = len(files)
        if total == 0:
            if progress_cb:
                progress_cb("", 1.0, "未选择任何音频文件")
            return results
        for i, fp in enumerate(files):
            song_name = os.path.splitext(os.path.basename(fp))[0]
            if progress_cb:
                progress_cb(
                    fp,
                    i / total,
                    f"({i + 1}/{total}) 正在转写: {song_name}",
                )
            try:
                out_path, song = self.write_song_json(
                    fp, output_dir, song_name=song_name, bpm=bpm
                )
                results.append({
                    "input": fp,
                    "output": out_path,
                    "ok": True,
                    "error": None,
                    "stats": song.get("_transcribe_stats"),
                    "song_name": song_name,
                })
            except Exception as e:
                results.append({
                    "input": fp,
                    "output": None,
                    "ok": False,
                    "error": str(e),
                    "stats": None,
                    "song_name": song_name,
                })
        if progress_cb:
            ok = sum(1 for r in results if r["ok"])
            progress_cb("", 1.0, f"完成 {ok}/{total}")
        return results
