import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from player import PlaybackState
from transcription.comparison import ComparisonTransport, NativeTrackPlayer, comparison_range, prepare_comparison_audio
from transcription.models import TranscriptionError, TranscriptionResult
from transcription.preview import render_preview_wav


class Backend:
    def __init__(self, _path=None): self.calls = []
    def play(self, value): self.calls.append(('play', value))
    def pause(self): self.calls.append(('pause',))
    def resume(self): self.calls.append(('resume',))
    def stop(self): self.calls.append(('stop',))
    def close(self): self.calls.append(('close',))


class ComparisonTests(unittest.TestCase):
    def test_mode_switch_preserves_absolute_position_and_pause(self):
        clock = [0.0]
        transport = ComparisonTransport(clock=lambda: clock[0])
        old, new = Backend(), Backend()
        transport.load(old, 2000, 4000, 2200)
        clock[0] += .3
        transport.pause()
        position = transport.position_ms
        clock[0] += 5
        self.assertEqual(transport.position_ms, 2500)
        transport.load(new, 2000, 4000, position, paused=True)
        self.assertEqual(new.calls, [('play', 500), ('pause',)])
        self.assertEqual(transport.state, PlaybackState.PAUSED)
        self.assertEqual(old.calls[-2:], [('stop',), ('close',)])
        transport.resume(); clock[0] += .1
        self.assertEqual(transport.position_ms, 2600)
        transport.seek_ms(3000)
        self.assertEqual(new.calls[-1], ('play', 1000))
        transport.close()

    def test_native_track_closes_exactly_once_on_own_thread(self):
        backend = Backend()
        track = NativeTrackPlayer('unused.wav', backend_factory=lambda _path: backend)
        track.play(100)
        track.close(); track.close()
        track.thread.join(1)
        self.assertEqual(backend.calls, [('play', 100), ('close',)])
        self.assertFalse(track.thread.is_alive())

    def test_comparison_fails_clearly_when_source_is_missing(self):
        result = TranscriptionResult([], [{'time': 500, 'key': '1Key0'}], 'C major', 120, {}, [])
        with tempfile.TemporaryDirectory() as root, self.assertRaises(TranscriptionError):
            prepare_comparison_audio(result, root+'/missing.wav', 0, 2000, 'source', root)

    def test_default_range_is_current_bar_and_selection_is_honoured(self):
        result = TranscriptionResult([], [{'time': 500, 'key': '1Key0'}, {'time': 5500, 'key': '1Key1'}], 'C major', 120, {}, [])
        self.assertEqual(comparison_range(result, 2500), (2000, 4000))
        self.assertEqual(comparison_range(result, 2500, (3000, 1000)), (1000, 3000))

    def test_renderer_preserves_leading_rest_and_can_render_silence(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as root:
            sample = Path(root)/'sample.wav'
            sf.write(sample, np.ones(1000)*.1, 22050)
            class Library:
                def ensure_samples(self): return [str(sample)]*15
            path = Path(root)/'preview.wav'
            render_preview_wav([{'time': 500, 'key': '1Key0'}], str(path), library=Library(), start_ms=0, end_ms=1000)
            audio, sr = sf.read(path)
            self.assertTrue(np.all(audio[:round(sr*.5)] == 0))
            self.assertGreater(np.max(audio), 0)
            self.assertAlmostEqual(len(audio)/sr, 1, places=3)
            render_preview_wav([], str(path), library=Library(), start_ms=0, end_ms=1000)
            audio, _sr = sf.read(path)
            self.assertTrue(np.all(audio == 0))

    def test_pure_melody_uses_saved_roles_instead_of_reinterpreting_reviewed_notes(self):
        notes = [{'time': 100, 'key': '1Key3'}, {'time': 100, 'key': '1Key10'}]
        result = TranscriptionResult([], notes, 'C major', 120, {}, [], note_roles={'100:1Key3': 'melody', '100:1Key10': 'harmony'})
        rendered = []
        def render(chosen, path, **_kwargs):
            rendered.extend(chosen); Path(path).write_bytes(b'wave'); return path
        with tempfile.TemporaryDirectory() as root, patch('transcription.comparison.render_preview_wav', render):
            prepare_comparison_audio(result, 'missing.wav', 0, 1000, 'melody', root)
        self.assertEqual(rendered, [notes[0]])
