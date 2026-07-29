from __future__ import annotations

import glob
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from .models import CancelledError, ProgressCallback, TranscriptionError
from .netease_auth import NetscapeCookie, cookie_request_header


SEARCH_URL = "https://music.163.com/api/search/get/web"


@dataclass(frozen=True)
class NetEaseTrack:
    song_id: str
    title: str
    artists: Tuple[str, ...]
    album: str
    duration_ms: int

    @property
    def webpage_url(self) -> str:
        return f"https://music.163.com/song?id={self.song_id}"

    @property
    def artist_text(self) -> str:
        return " / ".join(self.artists)

    @property
    def display_name(self) -> str:
        artist = f" - {self.artist_text}" if self.artist_text else ""
        return f"{self.title}{artist}"


@dataclass(frozen=True)
class NetEaseSearchPage:
    items: Tuple[NetEaseTrack, ...]
    total: int
    offset: int
    limit: int


class _SilentLogger:
    def debug(self, _message) -> None:
        pass

    def info(self, _message) -> None:
        pass

    def warning(self, _message) -> None:
        pass

    def error(self, _message) -> None:
        pass


class NetEaseClient:
    def __init__(
        self,
        *,
        opener: Optional[Callable] = None,
        ydl_factory: Optional[Callable] = None,
        ffmpeg_resolver: Optional[Callable[[], str]] = None,
        timeout: float = 15.0,
    ):
        self._opener = opener or urllib.request.urlopen
        self._ydl_factory = ydl_factory
        self._ffmpeg_resolver = ffmpeg_resolver
        self.timeout = float(timeout)

    def search(
        self,
        query: str,
        *,
        offset: int = 0,
        limit: int = 20,
        cookies: Sequence[NetscapeCookie] = (),
    ) -> NetEaseSearchPage:
        keyword = (query or "").strip()
        if not keyword:
            raise TranscriptionError("请输入歌曲名或歌手名")
        offset = max(0, int(offset))
        limit = max(1, min(50, int(limit)))
        body = urllib.parse.urlencode(
            {"s": keyword, "type": "1", "offset": str(offset), "limit": str(limit)}
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://music.163.com/",
            "User-Agent": "Mozilla/5.0 SkyAutoMusic/1.0",
        }
        if cookies:
            headers["Cookie"] = cookie_request_header(cookies)
        request = urllib.request.Request(SEARCH_URL, data=body, headers=headers)
        try:
            with self._opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise TranscriptionError("网易云搜索连接失败，请检查网络后重试") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TranscriptionError("网易云搜索返回了无法解析的数据") from exc

        if payload.get("code") != 200:
            raise TranscriptionError("网易云搜索暂时不可用，请稍后重试")
        result = payload.get("result") or {}
        raw_songs = result.get("songs") or []
        items = []
        for raw in raw_songs:
            song_id = raw.get("id")
            title = raw.get("name")
            if song_id is None or not title:
                continue
            artists = tuple(
                str(artist.get("name"))
                for artist in (raw.get("artists") or [])
                if isinstance(artist, dict) and artist.get("name")
            )
            album_data = raw.get("album") or {}
            album = str(album_data.get("name") or "") if isinstance(album_data, dict) else ""
            try:
                duration_ms = max(0, int(raw.get("duration") or 0))
            except (TypeError, ValueError):
                duration_ms = 0
            items.append(
                NetEaseTrack(
                    song_id=str(song_id),
                    title=str(title),
                    artists=artists,
                    album=album,
                    duration_ms=duration_ms,
                )
            )
        try:
            total = max(0, int(result.get("songCount") or len(items)))
        except (TypeError, ValueError):
            total = len(items)
        return NetEaseSearchPage(tuple(items), total, offset, limit)

    def resolve_and_download(
        self,
        track: NetEaseTrack,
        *,
        cookiefile: Optional[str],
        temp_dir: str,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> str:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("在线扒谱已取消")
        os.makedirs(temp_dir, exist_ok=True)
        ffmpeg_exe = self._resolve_ffmpeg()
        ydl_factory = self._resolve_ydl_factory()

        def notify(stage: str, fraction: float, message: str) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("在线扒谱已取消")
            if progress_cb:
                progress_cb(stage, max(0.0, min(1.0, float(fraction))), message)

        def download_hook(status: Dict) -> None:
            state = status.get("status")
            if state == "downloading":
                total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
                downloaded = status.get("downloaded_bytes") or 0
                fraction = float(downloaded) / float(total) if total else 0.0
                notify("download", fraction, f"正在下载：{track.display_name}")
            elif state == "finished":
                notify("download", 1.0, "音频下载完成")

        def postprocessor_hook(status: Dict) -> None:
            state = status.get("status")
            if state == "started":
                notify("convert", 0.0, "正在转换为扒谱用 WAV")
            elif state == "finished":
                notify("convert", 1.0, "WAV 转换完成")

        output_template = os.path.join(temp_dir, f"{track.song_id}.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": {"default": output_template},
            "noplaylist": True,
            "overwrites": True,
            "quiet": True,
            "no_warnings": True,
            "logger": _SilentLogger(),
            "ffmpeg_location": ffmpeg_exe,
            "progress_hooks": [download_hook],
            "postprocessor_hooks": [postprocessor_hook],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
        }
        if cookiefile:
            options["cookiefile"] = cookiefile

        notify("resolve", 0.0, "正在解析网易云音频直链")
        try:
            with ydl_factory(options) as ydl:
                ydl.extract_info(track.webpage_url, download=True)
        except CancelledError:
            raise
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("在线扒谱已取消") from exc
            raise self._friendly_download_error(exc) from exc
        candidates = sorted(glob.glob(os.path.join(temp_dir, f"{track.song_id}*.wav")))
        if not candidates:
            raise TranscriptionError("音频已下载，但 FFmpeg 没有生成可扒谱的 WAV 文件")
        return candidates[0]

    def _resolve_ydl_factory(self) -> Callable:
        if self._ydl_factory is not None:
            return self._ydl_factory
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise TranscriptionError("缺少 yt-dlp，无法获取网易云音频") from exc
        return YoutubeDL

    def _resolve_ffmpeg(self) -> str:
        if self._ffmpeg_resolver is not None:
            path = self._ffmpeg_resolver()
        else:
            try:
                import imageio_ffmpeg
            except ImportError as exc:
                raise TranscriptionError("缺少内置 FFmpeg，无法转换网易云音频") from exc
            path = imageio_ffmpeg.get_ffmpeg_exe()
        if not path or not os.path.isfile(path):
            raise TranscriptionError("内置 FFmpeg 不可用，请重新安装完整依赖")
        return os.path.abspath(path)

    @staticmethod
    def _friendly_download_error(exc: Exception) -> TranscriptionError:
        detail = str(exc).lower()
        if any(word in detail for word in ("login", "cookie", "登录")):
            return TranscriptionError("网易云 Cookie 已失效，或该歌曲需要登录后播放")
        if any(word in detail for word in ("vip", "premium", "会员", "payment")):
            return TranscriptionError("当前网易云账号没有这首歌曲的播放权限")
        if any(word in detail for word in ("geo", "region", "country", "地区")):
            return TranscriptionError("该歌曲受地区限制，当前网络无法获取")
        if any(word in detail for word in ("unavailable", "removed", "not available", "下架")):
            return TranscriptionError("该歌曲已下架或暂时不可用")
        if any(word in detail for word in ("timeout", "timed out", "connection", "network")):
            return TranscriptionError("网易云音频下载失败，请检查网络后重试")
        return TranscriptionError("网易云音频获取失败，请更新 Cookie 或稍后重试")


def validate_netease_runtime() -> None:
    """Offline runtime check used by tests and the packaged executable."""

    try:
        from yt_dlp.extractor.neteasemusic import NetEaseMusicIE
    except ImportError as exc:
        raise RuntimeError("yt-dlp 网易云提取器未被打包") from exc
    if not NetEaseMusicIE.suitable("https://music.163.com/song?id=95670"):
        raise RuntimeError("yt-dlp 网易云单曲提取器不可用")
    NetEaseClient()._resolve_ffmpeg()
