"""Transcriber 纯单元测试，不依赖音频文件。

重点验证 C 调映射 + 八度折叠 + 自动移调逻辑。
直接调用内部方法 _midi_to_key / _compute_transpose / is_audio_file。
"""
import os
import sys
import unittest

# 允许从仓库根目录导入 transcriber（即使未安装 librosa 也能通过 _midi_to_key 测试）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcriber import is_audio_file, Transcriber  # noqa: E402


def _bare_transcriber() -> Transcriber:
    """绕过 __init__ 里的 librosa 存在性检查，仅用于测纯映射逻辑。"""
    t = Transcriber.__new__(Transcriber)
    t.NUM_KEYS = 15
    t.midi_root = 60
    t.sr = 44100
    return t


class TestMidiToKey(unittest.TestCase):
    """C 调映射 + 八度折叠：核心映射规则。"""

    def setUp(self):
        self.t = _bare_transcriber()

    # --- 原生范围内（不折叠） ---

    def test_root_is_1key0(self):
        # MIDI 60 (C4) → 1Key0
        key, clamped = self.t._midi_to_key(60)
        self.assertEqual(key, "1Key0")
        self.assertEqual(clamped, 0)

    def test_mid_range(self):
        # MIDI 70 (A4) → 1Key10
        key, clamped = self.t._midi_to_key(70)
        self.assertEqual(key, "1Key10")
        self.assertEqual(clamped, 0)

    def test_top_of_range(self):
        # MIDI 74 (D5) → 1Key14 (边界)
        key, clamped = self.t._midi_to_key(74)
        self.assertEqual(key, "1Key14")
        self.assertEqual(clamped, 0)

    def test_rounding(self):
        # MIDI 64.6 → i = 4.6 → round → 5
        key, clamped = self.t._midi_to_key(64.6)
        self.assertEqual(key, "1Key5")
        self.assertEqual(clamped, 0)

    def test_just_above_root(self):
        # MIDI 61 (C#4) → 1Key1
        key, clamped = self.t._midi_to_key(61)
        self.assertEqual(key, "1Key1")
        self.assertEqual(clamped, 0)

    # --- 超出范围：八度折叠 ---

    def test_below_range_folds_up(self):
        # MIDI 50 (D3) 比 C4 低 10 半音：i=-10 → +12 → 2 → 1Key2（上折）
        key, clamped = self.t._midi_to_key(50)
        self.assertEqual(key, "1Key2")
        self.assertEqual(clamped, -1)

    def test_above_range_folds_down(self):
        # MIDI 90 (F#6) 比 C4 高 30 半音：i=30 → -12 → 18 → -12 → 6 → 1Key6（下折）
        key, clamped = self.t._midi_to_key(90)
        self.assertEqual(key, "1Key6")
        self.assertEqual(clamped, 1)

    def test_octave_below_maps_to_same_key(self):
        # MIDI 48 (C3) 比 C4 低一个八度：i=-12 → +12 → 0 → 1Key0
        key, clamped = self.t._midi_to_key(48)
        self.assertEqual(key, "1Key0")
        self.assertEqual(clamped, -1)

    def test_two_octaves_above_folds_down(self):
        # MIDI 84 (C6) 比 C4 高 24 半音：i=24 → -12 → 12 → 1Key12
        key, clamped = self.t._midi_to_key(84)
        self.assertEqual(key, "1Key12")
        self.assertEqual(clamped, 1)

    # --- 自动移调 ---

    def test_transpose_shifts_mapping(self):
        # MIDI 50 + transpose=2: i = 50+2-60 = -8 → +12 → 4 → 1Key4（仍需上折）
        key, clamped = self.t._midi_to_key(50, transpose=2)
        self.assertEqual(key, "1Key4")
        self.assertEqual(clamped, -1)

    def test_transpose_brings_into_native_range(self):
        # MIDI 56 + transpose=4: i = 56+4-60 = 0 → 1Key0，原生范围（不折叠）
        key, clamped = self.t._midi_to_key(56, transpose=4)
        self.assertEqual(key, "1Key0")
        self.assertEqual(clamped, 0)


class TestComputeTranspose(unittest.TestCase):
    """自动移调量搜索：使原生范围内音符数最大化。"""

    def setUp(self):
        self.t = _bare_transcriber()

    def test_empty_returns_zero(self):
        self.assertEqual(self.t._compute_transpose([]), 0)

    def test_native_range_prefers_no_transpose(self):
        # 全部已在 [60,74] 内 → transpose=0 即可，平局偏向不移调
        self.assertEqual(self.t._compute_transpose([60, 61, 62, 70]), 0)

    def test_low_notes_pull_up(self):
        # 54/55/56 比 root 低 4-6 半音；仅 transpose=6 能把三者全部拉进原生范围
        self.assertEqual(self.t._compute_transpose([54, 55, 56]), 6)

    def test_high_notes_push_down(self):
        # 78/79/80 比 root 高 18-20 半音；仅 transpose=-6 能把三者全部降进原生范围
        self.assertEqual(self.t._compute_transpose([78, 79, 80]), -6)

    def test_tie_prefers_smaller_abs(self):
        # 单音 MIDI 60：多个 transpose 都原生，平局取 |t| 最小 → 0
        self.assertEqual(self.t._compute_transpose([60]), 0)


class TestIsAudioFile(unittest.TestCase):
    def test_recognized(self):
        for p in ["a.mp3", "b.WAV", "c.Flac", "d.ogg", "e.m4a", "f.AAC"]:
            self.assertTrue(is_audio_file(p), p)

    def test_rejected(self):
        for p in ["a.json", "b.txt", "c", "d.mp4"]:
            self.assertFalse(is_audio_file(p), p)


if __name__ == "__main__":
    unittest.main()
