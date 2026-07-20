"""音频 → 乐谱 JSON 转写器。

输入: mp3 / wav / flac / ogg
输出: 与现有 997 份乐谱完全兼容的 JSON
      顶层: [{ "name": str, "bpm": int, "songNotes": [{time: int_ms, key: "1Key0..14"}, ...] }]

策略（用户已确认）:
- 智能八度折叠 + 自动移调:
  - 先收集全部检测到的 MIDI 音，在 [-6,6] 半音内搜索最优移调量 transpose，
    使无需折叠即落在 [0,14] 内的音符数最大化（平局偏向不移调）
  - 再做八度折叠: i = midi + transpose - root; <0 则 +12、>=15 则 -12
  - 超出键盘范围的音按八度折叠进 15 键，保留旋律轮廓与节奏
- HPSS 谐波/打击分离: onset 检测与音高估计均在 y_harmonic 上进行，
  聚焦旋律起音、减少打击乐误触发
- onset 检测: onset_strength envelope + backtrack=True + delta 门限，
  让 onset 回退到能量上升起点、过滤弱 onset，时间更准
- 保留原曲和弦: pYIN 提取主 F0，再在谐波 STFT 幅度谱上补足强基频峰
  （剔除主 F0 谐波），最多保留 MAX_POLYPHONY 个音，按音高升序写入 songNotes
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

    __slots__ = (
        "onset_count", "note_count", "clamped_low", "clamped_high",
        "duration_sec", "transpose",
    )

    def __init__(self):
        self.onset_count = 0
        self.note_count = 0
        self.clamped_low = 0
        self.clamped_high = 0
        self.duration_sec = 0.0
        # 自动移调量（半音），由八度折叠阶段计算；0 表示不移调
        self.transpose = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "onset_count": self.onset_count,
            "note_count": self.note_count,
            "clamped_low": self.clamped_low,
            "clamped_high": self.clamped_high,
            "duration_sec": round(self.duration_sec, 3),
            "transpose": self.transpose,
        }


class Transcriber:
    """音频 → 乐谱 note 列表。

    设计目标:
    - transcribe() 输入输出纯粹，方便单元测试
    - run() 包裹 transcribe + JSON 落盘 + 进度回调，方便 GUI 集成
    - 不引入 librosa 之外的重量级依赖
    """

    NUM_KEYS = 15
    # 提升采样率到 44100 以获得更高频率分辨率，改善音高估计精度
    DEFAULT_SR = 44100
    # 每个 onset 之后取多长窗做音高估计（秒）。0.13s 兼顾稳定性与相邻 onset 隔离
    ONSET_PITCH_WINDOW = 0.13
    # 同一 onset 最多写出几个 key（和弦保留上限，避免一拍按十几键）
    MAX_POLYPHONY = 3
    # 静音/能量门限（音高幅度归一化后），低于此视为无音
    PITCH_MAG_THRESHOLD = 0.10
    # onset_strength 峰值拾取门限：高于局部偏移量才视为有效 onset，过滤弱触发
    ONSET_DELTA = 0.07

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

        # HPSS: 分离谐波与打击乐。onset 检测与音高估计均在 y_harmonic 上进行，
        # 聚焦旋律起音、减少打击乐误触发
        y_harmonic, _y_percussive = _librosa.effects.hpss(y)
        hop_length = 512  # 与 piptrack / pYIN 时间分辨率保持一致

        # 1) onset 检测：onset_strength envelope + backtrack=True + delta 门限
        #    backtrack 让 onset 回退到能量上升起点；delta 过滤弱 onset
        onset_env = _librosa.onset.onset_strength(
            y=y_harmonic, sr=sr, hop_length=hop_length
        )
        onset_times = _librosa.onset.onset_detect(
            y=y_harmonic, sr=sr, onset_envelope=onset_env,
            hop_length=hop_length, units="time", backtrack=True,
            delta=self.ONSET_DELTA,
        )
        if len(onset_times) == 0:
            return [], stats
        stats.onset_count = int(len(onset_times))

        # 2) pYIN 主 F0 + STFT 强基频峰补足和弦（均在 y_harmonic 上）
        fmin = _librosa.note_to_hz("C2")
        fmax = _librosa.note_to_hz("C7")
        f0, voiced_flag, _voiced_probs = _librosa.pyin(
            y_harmonic, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=2048, hop_length=hop_length,
        )
        stft_mag = np.abs(_librosa.stft(y_harmonic, n_fft=2048, hop_length=hop_length))
        freqs = _librosa.fft_frequencies(sr=sr, n_fft=2048)
        # pyin 与 stft 帧数可能差 1，按较小者对齐
        n_frames = min(int(f0.shape[0]), int(stft_mag.shape[1]))
        f0 = f0[:n_frames]
        voiced_flag = voiced_flag[:n_frames]
        stft_mag = stft_mag[:, :n_frames]
        frame_times = _librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=hop_length
        )

        # 第一阶段: 收集每个 onset 的候选 midi（尚未映射）
        window_sec = self.ONSET_PITCH_WINDOW
        raw_events: List[Tuple[float, float]] = []  # (onset_t, midi)
        for t in onset_times:
            frame_pitches = self._collect_pitches_pyin(
                f0, voiced_flag, stft_mag, freqs,
                frame_times, float(t), window_sec,
            )
            for midi, _mag in frame_pitches:
                raw_events.append((float(t), midi))

        if not raw_events:
            return [], stats

        # 第二阶段: 智能八度折叠 + 自动移调
        transpose = self._compute_transpose([m for _t, m in raw_events])
        stats.transpose = transpose

        notes: List[Dict[str, Any]] = []
        for t, midi in raw_events:
            key, clamped = self._midi_to_key(midi, transpose)
            if clamped < 0:
                stats.clamped_low += 1
            elif clamped > 0:
                stats.clamped_high += 1
            notes.append({"time": int(round(t * 1000)), "key": key})

        # 3) 按 time 排序
        notes.sort(key=lambda n: (n["time"], n["key"]))
        stats.note_count = len(notes)
        return notes, stats

    def _collect_pitches_pyin(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        stft_mag: np.ndarray,
        freqs: np.ndarray,
        frame_times: np.ndarray,
        onset_t: float,
        window_sec: float,
    ) -> List[Tuple[float, float]]:
        """pYIN 主 F0 + STFT 强基频峰补足和弦。

        - 主 F0: onset 窗内 voiced f0 的中位数（最可靠）
        - 补足音: 窗内平均 STFT 幅度谱的局部峰，剔除主 F0 的整数倍谐波，
          按能量降序取至 MAX_POLYPHONY-1 个；主 F0 缺失时直接取最强峰补足
        返回 [(midi, mag), ...]，按音高升序、半音级去重。
        """
        mask = (frame_times >= onset_t) & (frame_times < onset_t + window_sec)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            # onset 极接近结尾时也兜一个 frame
            j = int(np.searchsorted(frame_times, onset_t))
            j = max(0, min(j, len(frame_times) - 1))
            idxs = np.array([j])

        # --- 主 F0: 窗内 voiced f0 中位数 ---
        window_voiced = f0[idxs][voiced_flag[idxs]]
        main_f0: Optional[float] = None
        if len(window_voiced) > 0:
            main_f0 = float(np.median(window_voiced))
        main_midi: Optional[float] = None
        if main_f0 and main_f0 > 0:
            main_midi = float(_librosa.hz_to_midi(main_f0))

        # --- STFT 峰: 窗内平均幅度谱局部极大 ---
        avg_mag = np.mean(stft_mag[:, idxs], axis=1)
        max_mag = float(np.max(avg_mag)) if avg_mag.size else 0.0
        candidates: List[Tuple[float, float]] = []  # (midi, mag_norm)
        if max_mag > 0:
            norm_mag = avg_mag / max_mag
            if norm_mag.shape[0] >= 3:
                peak_mask = (norm_mag[1:-1] > norm_mag[:-2]) & (
                    norm_mag[1:-1] > norm_mag[2:]
                )
                peak_idxs = np.where(peak_mask)[0] + 1
            else:
                peak_idxs = np.array([int(np.argmax(norm_mag))])
            for i in peak_idxs:
                m = float(norm_mag[i])
                if m < self.PITCH_MAG_THRESHOLD:
                    continue
                fr = float(freqs[i])
                if fr <= 0:
                    continue
                candidates.append((float(_librosa.hz_to_midi(fr)), m))

        # --- 剔除主 F0 的整数倍谐波 ---
        if main_f0 and main_f0 > 0:
            filtered: List[Tuple[float, float]] = []
            for midi, m in candidates:
                is_harmonic = False
                k = 2
                while k * main_f0 <= float(freqs[-1]):
                    if abs(midi - float(_librosa.hz_to_midi(k * main_f0))) < 0.5:
                        is_harmonic = True
                        break
                    k += 1
                if not is_harmonic:
                    filtered.append((midi, m))
            candidates = filtered

        # --- 选取额外音: 按能量降序、半音级去重，补足至 MAX_POLYPHONY ---
        candidates.sort(key=lambda pm: pm[1], reverse=True)
        seen = set()
        result: List[Tuple[float, float]] = []
        if main_midi is not None:
            result.append((main_midi, 1.0))
            seen.add(int(round(main_midi)))
        for midi, m in candidates:
            s = int(round(midi))
            if s in seen:
                continue
            seen.add(s)
            result.append((midi, m))
            if len(result) >= self.MAX_POLYPHONY:
                break
        result.sort(key=lambda pm: pm[0])  # 按音高升序
        return result

    def _compute_transpose(self, midis: List[float]) -> int:
        """在 [-6, 6] 半音内搜索最优移调量，使无需八度折叠即落在 [0, NUM_KEYS) 内的音符数最大化。

        平局时取绝对值最小者（偏向不移调）。
        """
        if not midis:
            return 0
        root = self.midi_root
        n = self.NUM_KEYS
        int_midis = [int(round(m)) for m in midis]
        best_t = 0
        best_score = -1
        for t in range(-6, 7):
            score = sum(1 for im in int_midis if 0 <= im + t - root < n)
            if score > best_score or (
                score == best_score and abs(t) < abs(best_t)
            ):
                best_score = score
                best_t = t
        return best_t

    def _midi_to_key(self, midi: float, transpose: int = 0) -> Tuple[str, int]:
        """C 大调映射 + 自动移调 + 八度折叠。

        返回 (key_name, clamped)，clamped: -1=原音低于键盘范围(上折)/0=原生在范围内/
        1=原音高于键盘范围(下折)。折叠后理论上必落在 [0, NUM_KEYS)，兜底夹边。
        """
        i = int(round(midi)) + transpose - self.midi_root
        clamped = 0
        if i < 0:
            clamped = -1
        elif i >= self.NUM_KEYS:
            clamped = 1
        # 八度折叠：把超出 [0, NUM_KEYS) 的音按 12 半音折进来
        while i < 0:
            i += 12
        while i >= self.NUM_KEYS:
            i -= 12
        # 兜底夹边（NUM_KEYS=15 时理论不会触发）
        if i < 0:
            i = 0
        elif i >= self.NUM_KEYS:
            i = self.NUM_KEYS - 1
        return f"1Key{i}", clamped

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
