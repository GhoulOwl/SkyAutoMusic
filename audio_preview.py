"""Local MP3 preview support for Sky JSON scores.

The preview controller intentionally implements the same ``press_keys`` /
``release_keys`` surface used by :class:`player.MusicPlayer`.  This lets the
existing timestamp scheduler drive audio samples without sending keyboard
events to the game.
"""

from __future__ import annotations

import ctypes
import os
import queue
import re
import shutil
import sys
import threading
import urllib.request
from pathlib import Path


SAMPLE_COUNT = 15
SAMPLE_BASE_URL = "https://sky-music.specy.app/assets/audio/sky/Piano"
_NOTE_PATTERN = re.compile(r"^[12]Key(\d+)$")


class AudioPreviewError(RuntimeError):
    """Raised when preview samples cannot be prepared or played."""


def note_to_sample_index(note):
    """Return the piano sample index for a Sky key, or ``None`` if unsupported."""

    match = _NOTE_PATTERN.fullmatch(str(note))
    if not match:
        return None
    index = int(match.group(1))
    return index if 0 <= index < SAMPLE_COUNT else None


def _default_asset_dirs():
    """Return read-only/bundled lookup locations in priority order."""

    relative = Path("assets") / "audio" / "sky" / "Piano"
    dirs = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        dirs.append(Path(bundle_root) / relative)
    dirs.append(Path(__file__).resolve().parent / relative)
    return dirs


def _default_cache_dir():
    relative = Path("assets") / "audio" / "sky" / "Piano"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / relative
    return Path(__file__).resolve().parent / relative


def _is_mp3_file(path):
    try:
        if Path(path).stat().st_size < 128:
            return False
        with open(path, "rb") as stream:
            header = stream.read(3)
        return header == b"ID3" or (
            len(header) >= 2
            and header[0] == 0xFF
            and (header[1] & 0xE0) == 0xE0
        )
    except OSError:
        return False


class PianoSampleLibrary:
    """Locate bundled samples and download only the missing files."""

    def __init__(
        self,
        cache_dir=None,
        search_dirs=None,
        base_url=SAMPLE_BASE_URL,
        opener=None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        raw_search_dirs = _default_asset_dirs() if search_dirs is None else search_dirs
        self.search_dirs = [Path(path) for path in raw_search_dirs]
        if self.cache_dir not in self.search_dirs:
            self.search_dirs.append(self.cache_dir)
        self.base_url = base_url.rstrip("/")
        self.opener = opener or urllib.request.urlopen
        self._lock = threading.Lock()

    def ensure_samples(self, progress_callback=None):
        """Return all 15 sample paths, downloading missing files atomically."""

        with self._lock:
            resolved = []
            for index in range(SAMPLE_COUNT):
                path = self._find_sample(index)
                if path is None:
                    path = self._download_sample(index)
                resolved.append(str(path.resolve()))
                if progress_callback:
                    progress_callback(index + 1, SAMPLE_COUNT, index)
            return resolved

    def _find_sample(self, index):
        filename = f"{index}.mp3"
        for directory in self.search_dirs:
            candidate = directory / filename
            if _is_mp3_file(candidate):
                return candidate
        return None

    def _download_sample(self, index):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / f"{index}.mp3"
        partial = target.with_suffix(".mp3.part")
        url = f"{self.base_url}/{index}.mp3"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "SkyAutoMusic/preview"},
        )
        try:
            with self.opener(request, timeout=20) as response:
                with open(partial, "wb") as output:
                    shutil.copyfileobj(response, output)
            if not _is_mp3_file(partial):
                raise AudioPreviewError(f"下载的音色文件无效: {url}")
            os.replace(partial, target)
            return target
        except Exception as exc:
            try:
                partial.unlink()
            except OSError:
                pass
            if isinstance(exc, AudioPreviewError):
                raise
            raise AudioPreviewError(f"无法下载音色 {index}.mp3: {exc}") from exc


class MciSamplePlayer:
    """Play one MP3 channel per pitch through the Windows MCI API."""

    def __init__(self, sample_paths, command_sender=None):
        if len(sample_paths) != SAMPLE_COUNT:
            raise AudioPreviewError(f"需要 {SAMPLE_COUNT} 个钢琴音色文件")
        if command_sender is None and sys.platform != "win32":
            raise AudioPreviewError("本地试听目前仅支持 Windows")

        self._lock = threading.RLock()
        self._closed = False
        self._aliases = {}
        self._command_sender = command_sender or self._create_command_sender()
        self._requests = queue.Queue()
        self._ready = threading.Event()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._audio_worker,
            args=(list(sample_paths),),
            daemon=True,
            name="SkyAudioPreview",
        )
        self._thread.start()
        if not self._ready.wait(timeout=15):
            self._closed = True
            raise AudioPreviewError("打开试听音频设备超时")
        if self._startup_error is not None:
            self._closed = True
            raise self._startup_error

    @staticmethod
    def _create_command_sender():
        winmm = ctypes.WinDLL("winmm")
        send_string = winmm.mciSendStringW
        send_string.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        send_string.restype = ctypes.c_uint
        get_error = winmm.mciGetErrorStringW
        get_error.argtypes = [ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint]
        get_error.restype = ctypes.c_bool

        def _send(command):
            code = send_string(command, None, 0, None)
            if code:
                buffer = ctypes.create_unicode_buffer(256)
                if get_error(code, buffer, len(buffer)):
                    detail = buffer.value
                else:
                    detail = f"MCI error {code}"
                raise AudioPreviewError(f"{detail}（{command}）")

        return _send

    def _audio_worker(self, sample_paths):
        try:
            self._open_samples(sample_paths)
        except Exception as exc:
            self._startup_error = exc
            self._close_aliases()
            self._ready.set()
            return

        self._ready.set()
        while True:
            action, payload, completed, result = self._requests.get()
            try:
                if action == "play":
                    self._play_on_worker(payload)
                elif action == "stop":
                    self._stop_on_worker()
                elif action == "close":
                    self._close_aliases()
                    return
            except Exception as exc:
                result.append(exc)
            finally:
                completed.set()

    def _open_samples(self, sample_paths):
        prefix = f"sky_preview_{id(self):x}"
        for index, raw_path in enumerate(sample_paths):
            path = os.path.abspath(raw_path)
            if '"' in path:
                raise AudioPreviewError(f"音色路径不能包含双引号: {path}")
            alias = f"{prefix}_{index}"
            self._command_sender(
                f'open "{path}" type mpegvideo alias {alias}'
            )
            self._aliases[index] = alias

    def play_indices(self, indices):
        self._submit("play", tuple(dict.fromkeys(indices)))

    def stop_all(self):
        with self._lock:
            if self._closed:
                return
        self._submit("stop")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._submit("close", allow_closed=True)
        self._thread.join(timeout=5)

    def _submit(self, action, payload=None, allow_closed=False):
        with self._lock:
            if self._closed and not allow_closed:
                raise AudioPreviewError("试听音频设备已关闭")
        completed = threading.Event()
        result = []
        self._requests.put((action, payload, completed, result))
        if not completed.wait(timeout=5):
            raise AudioPreviewError("试听音频设备响应超时")
        if result:
            raise result[0]

    def _play_on_worker(self, indices):
        for index in indices:
            alias = self._aliases.get(index)
            if alias is None:
                continue
            self._try_command(f"stop {alias}")
            self._command_sender(f"seek {alias} to start")
            self._command_sender(f"play {alias}")

    def _stop_on_worker(self):
        for alias in self._aliases.values():
            self._try_command(f"stop {alias}")
            self._try_command(f"seek {alias} to start")

    def _close_aliases(self):
        aliases = list(self._aliases.values())
        self._aliases.clear()
        for alias in aliases:
            self._try_command(f"stop {alias}")
            self._try_command(f"close {alias}")

    def _try_command(self, command):
        try:
            self._command_sender(command)
        except Exception:
            pass


class _PygameMixerRuntime:
    """Process-wide pygame mixer and private channel-block allocator.

    The main window and the transcription dialog can each own a preview
    controller.  Giving every controller its own channel block means stopping
    one preview does not stop the other one.
    """

    CHANNELS_PER_OWNER = 32
    _lock = threading.RLock()
    _pygame = None
    _owners = 0
    _next_channel = 0
    _free_blocks = []

    @classmethod
    def acquire(cls, pygame_module=None):
        with cls._lock:
            if cls._pygame is None:
                try:
                    if pygame_module is None:
                        import pygame
                    else:
                        pygame = pygame_module
                except Exception as exc:
                    raise AudioPreviewError(
                        "macOS 试听需要 pygame-ce，请先安装 macOS 依赖"
                    ) from exc
                try:
                    pygame.mixer.pre_init(
                        frequency=48000,
                        size=-16,
                        channels=2,
                        buffer=512,
                    )
                    pygame.mixer.init()
                except Exception as exc:
                    try:
                        pygame.mixer.quit()
                    except Exception:
                        pass
                    raise AudioPreviewError(f"无法初始化 macOS 音频设备: {exc}") from exc
                cls._pygame = pygame
                cls._owners = 0
                cls._next_channel = 0
                cls._free_blocks = []

            if cls._free_blocks:
                block = cls._free_blocks.pop()
            else:
                start = cls._next_channel
                block = tuple(range(start, start + cls.CHANNELS_PER_OWNER))
                cls._next_channel += cls.CHANNELS_PER_OWNER
                try:
                    cls._pygame.mixer.set_num_channels(cls._next_channel)
                except Exception as exc:
                    if cls._owners == 0:
                        try:
                            cls._pygame.mixer.quit()
                        except Exception:
                            pass
                        cls._pygame = None
                    raise AudioPreviewError(f"无法分配 macOS 试听声道: {exc}") from exc
            cls._owners += 1
            return cls._pygame, block

    @classmethod
    def release(cls, block):
        with cls._lock:
            if cls._pygame is None:
                return
            cls._free_blocks.append(tuple(block))
            cls._owners = max(0, cls._owners - 1)
            if cls._owners:
                return
            try:
                cls._pygame.mixer.quit()
            finally:
                cls._pygame = None
                cls._next_channel = 0
                cls._free_blocks = []


class PygameSamplePlayer:
    """Cross-platform sample player used by the macOS preview mode."""

    def __init__(self, sample_paths, pygame_module=None):
        if len(sample_paths) != SAMPLE_COUNT:
            raise AudioPreviewError(f"需要 {SAMPLE_COUNT} 个钢琴音色文件")
        self._lock = threading.RLock()
        self._closed = False
        self._pygame, self._channel_ids = _PygameMixerRuntime.acquire(pygame_module)
        self._channels = []
        try:
            self._sounds = [self._pygame.mixer.Sound(path) for path in sample_paths]
            self._channels = [
                self._pygame.mixer.Channel(channel_id)
                for channel_id in self._channel_ids
            ]
        except Exception as exc:
            _PygameMixerRuntime.release(self._channel_ids)
            self._closed = True
            raise AudioPreviewError(f"无法读取 macOS 钢琴音色: {exc}") from exc
        self._cursor = 0

    def play_indices(self, indices):
        with self._lock:
            if self._closed:
                raise AudioPreviewError("试听音频设备已关闭")
            for index in indices:
                if not 0 <= int(index) < len(self._sounds):
                    continue
                channel = self._channels[self._cursor]
                self._cursor = (self._cursor + 1) % len(self._channels)
                try:
                    channel.play(self._sounds[int(index)])
                except Exception as exc:
                    raise AudioPreviewError(f"播放 macOS 钢琴音色失败: {exc}") from exc

    def stop_all(self):
        with self._lock:
            if self._closed:
                return
            for channel in self._channels:
                try:
                    channel.stop()
                except Exception:
                    pass

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for channel in self._channels:
                try:
                    channel.stop()
                except Exception:
                    pass
            block = self._channel_ids
            self._channels = []
            self._channel_ids = ()
        _PygameMixerRuntime.release(block)


def create_sample_player(sample_paths):
    """Create the native preview backend for the current operating system."""

    if sys.platform == "win32":
        return MciSamplePlayer(sample_paths)
    if sys.platform == "darwin":
        return PygameSamplePlayer(sample_paths)
    raise AudioPreviewError("当前操作系统没有可用的试听后端")


class AudioPreviewController:
    """Keyboard-controller-compatible adapter backed by piano samples."""

    def __init__(self, library=None, backend_factory=None, on_error=None):
        self.library = library or PianoSampleLibrary()
        self.backend_factory = backend_factory or create_sample_player
        self.on_error = on_error
        self._backend = None
        self._lock = threading.RLock()
        self._error_reported = False
        self._closed = False

    @property
    def prepared(self):
        with self._lock:
            return self._backend is not None

    def prepare(self, progress_callback=None):
        with self._lock:
            if self._closed:
                raise AudioPreviewError("试听音频设备已关闭")
            if self._backend is not None:
                return
        paths = self.library.ensure_samples(progress_callback)
        backend = self.backend_factory(paths)
        with self._lock:
            if self._closed:
                backend.close()
                raise AudioPreviewError("试听音频设备已关闭")
            if self._backend is None:
                self._backend = backend
                self._error_reported = False
            else:
                backend.close()

    def press_keys(self, notes):
        indices = []
        seen = set()
        for note in notes:
            index = note_to_sample_index(note)
            if index is not None and index not in seen:
                seen.add(index)
                indices.append(index)
        if not indices:
            return
        with self._lock:
            backend = self._backend
        if backend is None:
            self._report_error(AudioPreviewError("试听音色尚未准备完成"))
            return
        try:
            backend.play_indices(indices)
        except Exception as exc:
            self._report_error(exc)

    def release_keys(self, notes):
        # Samples are one-shot sounds.  Releasing after MusicPlayer.NOTE_HOLD
        # must not truncate their natural piano decay.
        return None

    def stop_all(self):
        with self._lock:
            backend = self._backend
        if backend is not None:
            try:
                backend.stop_all()
            except Exception as exc:
                self._report_error(exc)

    def close(self):
        with self._lock:
            backend = self._backend
            self._backend = None
            self._closed = True
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass

    def _report_error(self, exc):
        with self._lock:
            if self._error_reported:
                return
            self._error_reported = True
        if self.on_error:
            self.on_error(exc)
