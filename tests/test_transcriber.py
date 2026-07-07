"""Transcriber 纯单元测试，不依赖音频文件。

重点验证固定 C 调软限幅映射逻辑。
直接调用内部方法 _midi_to_key / is_audio_file。
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
    t.sr = 22050
    return t


class TestMidiToKey(unittest.TestCase):
    """固定 C 调 + 软限幅：核心映射规则。"""

    def setUp(self):
        self.t = _bare_transcriber()

    def test_below_range_clamps_to_0(self):
        # MIDI 50 (D3) 比 1Key0 还要低 10 个半音，应软限幅到 1Key0
        key, clamped = self.t._midi_to_key(50)
        self.assertEqual(key, "1Key0")
        self.assertEqual(clamped, -1)

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

    def test_above_range_clamps_to_14(self):
        # MIDI 90 (F#6) → 1Key14
        key, clamped = self.t._midi_to_key(90)
        self.assertEqual(key, "1Key14")
        self.assertEqual(clamped, 1)

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


class TestIsAudioFile(unittest.TestCase):
    def test_recognized(self):
        for p in ["a.mp3", "b.WAV", "c.Flac", "d.ogg", "e.m4a", "f.AAC"]:
            self.assertTrue(is_audio_file(p), p)

    def test_rejected(self):
        for p in ["a.json", "b.txt", "c", "d.mp4"]:
            self.assertFalse(is_audio_file(p), p)


if __name__ == "__main__":
    unittest.main()
