"""Source-time A/B transport and lazy, local comparison audio preparation."""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

from audio_preview import MciSamplePlayer, _PygameMixerRuntime
from player import PlaybackState
from .models import CancelledError, TranscriptionError
from .preview import render_preview_wav

MODES = {"原曲": "source", "分离人声": "vocal", "纯旋律": "melody", "完整编配": "arrangement"}


def comparison_range(result, playhead_ms=0, selection=None):
    if selection and selection[1] != selection[0]:
        return tuple(sorted(selection))
    analysis = result.quality_analysis or result.analysis
    if result.quality_analysis:
        bars = result.quality_analysis.bar_starts_ms
    elif result.analysis:
        bars = result.analysis.tempo_map.bar_starts_ms
    else:
        bars = list(range(0, max(1, max(int(n['time']) for n in result.song_notes))+2000, 2000))
    duration = int(analysis.duration_sec*1000) if analysis else max(int(n['time']) for n in result.song_notes)+1000
    bars = sorted(set([0, *(bar for bar in bars if 0 <= bar <= duration), duration]))
    index = min(len(bars)-2, max(0, bisect.bisect_right(bars, playhead_ms)-1))
    return bars[index], bars[index+1]


def prepare_comparison_audio(result, source_path, start_ms, end_ms, mode, root, library=None, cancel_event=None):
    if mode not in MODES.values() or end_ms <= start_ms:
        raise TranscriptionError("无效对照试听模式或片段")
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("试听准备已取消")
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    source_identity = None
    if mode in ("source", "vocal"):
        if not os.path.isfile(source_path):
            raise TranscriptionError("原始音频已不可用，仍可试听旋律和完整编配")
        stat = os.stat(source_path)
        source_identity = (os.path.abspath(source_path), stat.st_size, stat.st_mtime_ns)
    if mode == "melody":
        if result.quality_analysis:
            from .quality import arrange_quality_melody
            notes = arrange_quality_melody(result.quality_analysis, result.options, result.semitone_shift)
        else:
            notes = [n for n in result.song_notes if result.note_roles.get(f"{n['time']}:{n['key']}") == "melody"]
            if not result.note_roles:
                # V2 uses a packed integer role id; old drafts have no roles.
                # Re-arrange analysis to recover roles, rather than infer melody
                # by taking the highest rendered accompaniment note.
                if result.analysis:
                    from .arranger import arrange_analysis
                    all_notes, _stats, roles = arrange_analysis(result.analysis, result.options)
                    notes = [n for n in all_notes if roles.get((int(n['time']) << 4) | int(str(n['key'])[4:])) == 'melody']
                else:
                    from .benchmark import _melody
                    notes = _melody(result.song_notes)
    else:
        notes = result.song_notes
    identity = json.dumps([mode, start_ms, end_ms, source_identity, notes if mode in ('melody', 'arrangement') else None], sort_keys=True)
    path = root / (hashlib.sha256(identity.encode()).hexdigest() + '.wav')
    if path.is_file():
        return str(path)
    temporary = root / (uuid.uuid4().hex + '.wav')
    try:
        if mode in ('melody', 'arrangement'):
            render_preview_wav(notes, str(temporary), library=library, start_ms=start_ms, end_ms=end_ms)
        else:
            import librosa
            import numpy as np
            import soundfile as sf
            audio, sr = librosa.load(source_path, sr=22050, mono=True, offset=start_ms/1000, duration=(end_ms-start_ms)/1000)
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("试听准备已取消")
            if mode == 'vocal':
                from .separation import separate_audio
                sf.write(temporary, audio, sr)
                stem_root = root / uuid.uuid4().hex
                import shutil
                try:
                    stems = separate_audio(str(temporary), str(stem_root), cancel_event)
                    audio, sr = librosa.load(stems.vocal_path, sr=sr, mono=True)
                finally:
                    shutil.rmtree(stem_root, ignore_errors=True)
            length = max(1, round((end_ms-start_ms)/1000*sr))
            audio = np.pad(audio[:length], (0, max(0, length-len(audio))))
            fade = min(round(.005*sr), len(audio)//2)
            if fade:
                audio[:fade] *= np.linspace(0, 1, fade)
                audio[-fade:] *= np.linspace(1, 0, fade)
            sf.write(temporary, audio, sr)
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("试听准备已取消")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


class _MciTrack:
    def __init__(self, path):
        self.send = MciSamplePlayer._create_command_sender()
        self.alias = 'sky_compare_' + uuid.uuid4().hex
        if '"' in path:
            raise TranscriptionError('音频路径不能包含双引号')
        self.send(f'open "{os.path.abspath(path)}" type waveaudio alias {self.alias}')

    def play(self, offset_ms):
        self.send(f'stop {self.alias}')
        self.send(f'seek {self.alias} to {max(0, int(offset_ms))}')
        self.send(f'play {self.alias}')

    def pause(self):
        self.send(f'pause {self.alias}')

    def resume(self):
        self.send(f'resume {self.alias}')

    def stop(self):
        self.send(f'stop {self.alias}')

    def close(self):
        self.send(f'close {self.alias}')


class _PygameTrack:
    def __init__(self, path):
        self.pygame, self.block = _PygameMixerRuntime.acquire()
        try:
            import librosa
            import numpy as np
            audio, _sr = librosa.load(path, sr=48000, mono=True)
            pcm = np.clip(audio*32767, -32768, 32767).astype('<i2')
            self.pcm = np.repeat(pcm[:, None], 2, axis=1)
            self.channel = self.pygame.mixer.Channel(self.block[0])
            self.sound = None
        except Exception:
            _PygameMixerRuntime.release(self.block)
            raise

    def play(self, offset_ms):
        self.channel.stop()
        start = max(0, min(len(self.pcm)-1, int(offset_ms*48)))
        self.sound = self.pygame.mixer.Sound(buffer=self.pcm[start:].tobytes())
        self.channel.play(self.sound)

    def pause(self):
        self.channel.pause()

    def resume(self):
        self.channel.unpause()

    def stop(self):
        self.channel.stop()

    def close(self):
        self.channel.stop()
        _PygameMixerRuntime.release(self.block)


class NativeTrackPlayer:
    """MCI handles and pygame operations remain on their owning audio thread."""
    def __init__(self, path, backend_factory=None):
        self.requests = queue.Queue()
        self.closed = False
        self.thread = threading.Thread(target=self._worker, args=(path, backend_factory), daemon=True)
        self.ready = threading.Event()
        self.error = None
        self.thread.start()
        if not self.ready.wait(15):
            # A slow open must close itself if it eventually succeeds.
            self.closed = True
            raise TranscriptionError('打开对照音频超时')
        if self.error:
            raise self.error

    def _worker(self, path, factory):
        try:
            backend = (factory or (_MciTrack if sys.platform == 'win32' else _PygameTrack))(path)
        except Exception as exc:
            self.error = exc; self.ready.set(); return
        self.ready.set()
        disposed = False
        try:
            while not self.closed:
                name, args, done, errors = self.requests.get()
                try:
                    getattr(backend, name)(*args)
                    disposed = name == 'close'
                except Exception as exc:
                    errors.append(exc)
                finally:
                    done.set()
                if name == 'close':
                    return
        finally:
            if not disposed:
                backend.close()

    def _call(self, name, *args):
        if self.closed:
            return
        done, errors = threading.Event(), []
        self.requests.put((name, args, done, errors))
        if not done.wait(5):
            raise TranscriptionError('对照音频设备响应超时')
        if errors:
            raise errors[0]

    def play(self, offset_ms): self._call('play', offset_ms)
    def pause(self): self._call('pause')
    def resume(self): self._call('resume')
    def stop(self): self._call('stop')
    def close(self):
        if not self.closed:
            self._call('close')
            self.closed = True


class ComparisonTransport:
    def __init__(self, factory=NativeTrackPlayer, clock=time.perf_counter):
        self.factory, self.clock = factory, clock
        self.backend = None
        self.state = PlaybackState.STOPPED
        self.start_ms = self.end_ms = self.saved_ms = 0
        self.origin = 0

    @property
    def position_ms(self):
        value = self.saved_ms
        if self.state == PlaybackState.PLAYING:
            value += round((self.clock()-self.origin)*1000)
        return min(self.end_ms, max(self.start_ms, value))

    def load(self, backend, start_ms, end_ms, position_ms=None, paused=False):
        self.stop()
        if self.backend:
            self.backend.close()
        self.backend = backend
        self.start_ms, self.end_ms = start_ms, end_ms
        self.saved_ms = max(start_ms, min(end_ms, start_ms if position_ms is None else position_ms))
        self.backend.play(self.saved_ms-start_ms)
        self.origin = self.clock()
        self.state = PlaybackState.PLAYING
        if paused:
            self.pause()

    def pause(self):
        if self.state == PlaybackState.PLAYING:
            self.saved_ms = self.position_ms
            self.backend.pause()
            self.state = PlaybackState.PAUSED

    def resume(self):
        if self.state == PlaybackState.PAUSED:
            self.backend.resume()
            self.origin = self.clock()
            self.state = PlaybackState.PLAYING

    def seek_ms(self, value):
        if self.backend is None or self.state == PlaybackState.STOPPED:
            return None
        paused = self.state == PlaybackState.PAUSED
        self.saved_ms = max(self.start_ms, min(self.end_ms, int(value)))
        self.backend.play(self.saved_ms-self.start_ms)
        self.origin = self.clock()
        if paused:
            self.backend.pause()
        return self.saved_ms

    def stop(self):
        self.saved_ms = self.position_ms
        if self.backend:
            self.backend.stop()
        self.state = PlaybackState.STOPPED

    def close(self):
        self.stop()
        if self.backend:
            self.backend.close()
            self.backend = None
