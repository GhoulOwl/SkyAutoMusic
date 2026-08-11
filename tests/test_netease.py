import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.models import (  # noqa: E402
    CancelledError,
    SourceMetadata,
    TranscriptionOptions,
    TranscriptionResult,
)
from transcription.netease import (  # noqa: E402
    NetEaseClient,
    NetEaseTrack,
    validate_netease_runtime,
)
from transcription.netease_auth import (  # noqa: E402
    CookieFormatError,
    CookieValidationResult,
    NetEaseCookieStore,
    parse_netscape_cookies,
    serialize_netscape_cookies,
    validate_cookie_account,
)
from transcription.pipeline import (  # noqa: E402
    export_song_json,
    next_available_path,
    sanitize_filename_stem,
    suggested_output_stem,
)


def _cookies_text(*rows):
    return "# Netscape HTTP Cookie File\n" + "\n".join(rows) + "\n"


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class TestNetEaseCookies(unittest.TestCase):
    def test_parse_filters_other_domains_and_supports_httponly(self):
        text = _cookies_text(
            "#HttpOnly_.music.163.com\tTRUE\t/\tTRUE\t4102444800\tMUSIC_U\tsecret-token",
            ".music.163.com\tTRUE\t/\tFALSE\t0\t__csrf\tcsrf-value",
            ".example.com\tTRUE\t/\tFALSE\t0\tMUSIC_U\tmust-not-survive",
        )
        cookies = parse_netscape_cookies(text, now=1_700_000_000)
        self.assertEqual([cookie.name for cookie in cookies], ["MUSIC_U", "__csrf"])
        self.assertTrue(cookies[0].http_only)
        serialized = serialize_netscape_cookies(cookies)
        self.assertIn("#HttpOnly_.music.163.com", serialized)
        self.assertNotIn("example.com", serialized)

    def test_rejects_invalid_header_missing_and_expired_music_u(self):
        with self.assertRaises(CookieFormatError):
            parse_netscape_cookies("MUSIC_U=abc")
        with self.assertRaisesRegex(CookieFormatError, "MUSIC_U"):
            parse_netscape_cookies(
                _cookies_text(".music.163.com\tTRUE\t/\tFALSE\t0\t__csrf\tx")
            )
        with self.assertRaisesRegex(CookieFormatError, "过期"):
            parse_netscape_cookies(
                _cookies_text(
                    ".music.163.com\tTRUE\t/\tFALSE\t100\tMUSIC_U\texpired"
                ),
                now=200,
            )

    def test_live_validation_distinguishes_login_logout_and_network(self):
        cookies = parse_netscape_cookies(
            _cookies_text(".music.163.com\tTRUE\t/\tTRUE\t0\tMUSIC_U\ttoken")
        )

        def logged_in(request, timeout):
            self.assertIn("MUSIC_U=token", request.get_header("Cookie"))
            self.assertGreater(timeout, 0)
            return _Response(
                {"code": 200, "account": {"id": 7}, "profile": {"nickname": "测试用户"}}
            )

        valid = validate_cookie_account(cookies, opener=logged_in)
        self.assertEqual(valid.state, "valid")
        self.assertEqual(valid.nickname, "测试用户")

        invalid = validate_cookie_account(
            cookies,
            opener=lambda _request, timeout: _Response(
                {"code": 200, "account": None, "profile": None}
            ),
        )
        self.assertEqual(invalid.state, "invalid")

        def offline(_request, timeout):
            raise urllib.error.URLError("offline")

        unverified = validate_cookie_account(cookies, opener=offline)
        self.assertEqual(unverified.state, "unverified")

    def test_store_never_writes_plaintext_and_temp_cookie_is_removed(self):
        text = _cookies_text(
            ".music.163.com\tTRUE\t/\tTRUE\t0\tMUSIC_U\tsuper-secret-value"
        )

        def protect(data):
            return b"protected:" + data[::-1]

        def unprotect(data):
            self.assertTrue(data.startswith(b"protected:"))
            return data[len(b"protected:") :][::-1]

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "netease_auth.json")
            store = NetEaseCookieStore(path, protect=protect, unprotect=unprotect)
            store.save(text, CookieValidationResult("valid", "ok", "tester"))
            with open(path, "r", encoding="utf-8") as handle:
                persisted = handle.read()
            self.assertNotIn("super-secret-value", persisted)
            self.assertIn("ciphertext", persisted)
            with store.materialize_cookiefile() as cookiefile:
                self.assertTrue(os.path.isfile(cookiefile))
                with open(cookiefile, "r", encoding="utf-8") as handle:
                    self.assertIn("super-secret-value", handle.read())
                materialized = cookiefile
            self.assertFalse(os.path.exists(materialized))
            store.clear()
            self.assertFalse(os.path.exists(path))


class TestNetEaseClient(unittest.TestCase):
    def test_runtime_contains_extractor_and_bundled_ffmpeg(self):
        try:
            validate_netease_runtime()
        except RuntimeError as exc:
            self.skipTest(str(exc))

    def test_search_maps_paged_results_and_sends_cookie(self):
        captured = {}

        def opener(request, timeout):
            captured["data"] = request.data.decode("utf-8")
            captured["cookie"] = request.get_header("Cookie")
            captured["timeout"] = timeout
            return _Response(
                {
                    "code": 200,
                    "result": {
                        "songCount": 21,
                        "songs": [
                            {
                                "id": 123,
                                "name": "测试歌曲",
                                "duration": 185000,
                                "artists": [{"name": "歌手甲"}, {"name": "歌手乙"}],
                                "album": {"name": "测试专辑"},
                            }
                        ],
                    },
                }
            )

        cookies = parse_netscape_cookies(
            _cookies_text(".music.163.com\tTRUE\t/\tTRUE\t0\tMUSIC_U\ttoken")
        )
        page = NetEaseClient(opener=opener).search(
            "测试", offset=20, limit=20, cookies=cookies
        )
        self.assertEqual(page.total, 21)
        self.assertEqual(page.offset, 20)
        self.assertEqual(page.items[0].song_id, "123")
        self.assertEqual(page.items[0].artists, ("歌手甲", "歌手乙"))
        self.assertIn("offset=20", captured["data"])
        self.assertIn("MUSIC_U=token", captured["cookie"])

    def test_download_uses_ytdlp_ffmpeg_cookie_and_progress(self):
        captured = {}

        class FakeYDL:
            def __init__(self, options):
                captured["options"] = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, url, download):
                captured["url"] = url
                captured["download"] = download
                options = captured["options"]
                options["progress_hooks"][0](
                    {
                        "status": "downloading",
                        "downloaded_bytes": 50,
                        "total_bytes": 100,
                    }
                )
                options["progress_hooks"][0]({"status": "finished"})
                options["postprocessor_hooks"][0]({"status": "started"})
                output = options["outtmpl"]["default"].replace("%(ext)s", "wav")
                with open(output, "wb") as handle:
                    handle.write(b"RIFFfake")
                options["postprocessor_hooks"][0]({"status": "finished"})
                return {"id": "123", "url": "https://temporary.invalid/audio"}

        track = NetEaseTrack("123", "测试歌曲", ("歌手",), "专辑", 1000)
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            ffmpeg = os.path.join(directory, "ffmpeg.exe")
            with open(ffmpeg, "wb") as handle:
                handle.write(b"fake")
            cookiefile = os.path.join(directory, "cookies.txt")
            with open(cookiefile, "w", encoding="utf-8") as handle:
                handle.write("cookie")
            output = NetEaseClient(
                ydl_factory=FakeYDL,
                ffmpeg_resolver=lambda: ffmpeg,
            ).resolve_and_download(
                track,
                cookiefile=cookiefile,
                temp_dir=directory,
                progress_cb=lambda stage, fraction, message: progress.append(
                    (stage, fraction, message)
                ),
            )
            self.assertTrue(output.endswith(".wav"))
            self.assertTrue(os.path.isfile(output))

        self.assertEqual(captured["url"], track.webpage_url)
        self.assertTrue(captured["download"])
        self.assertEqual(captured["options"]["cookiefile"], cookiefile)
        self.assertEqual(captured["options"]["ffmpeg_location"], ffmpeg)
        self.assertEqual(
            captured["options"]["postprocessors"][0]["preferredcodec"], "wav"
        )
        self.assertTrue(any(stage == "download" for stage, _, _ in progress))
        self.assertTrue(any(stage == "convert" for stage, _, _ in progress))

    def test_cancelled_download_does_not_create_ydl(self):
        cancelled = threading.Event()
        cancelled.set()
        track = NetEaseTrack("1", "x", (), "", 0)
        with self.assertRaises(CancelledError):
            NetEaseClient(
                ydl_factory=lambda _options: self.fail("yt-dlp should not start"),
                ffmpeg_resolver=lambda: "unused",
            ).resolve_and_download(
                track,
                cookiefile=None,
                temp_dir=tempfile.gettempdir(),
                cancel_event=cancelled,
            )


class TestNetEaseExport(unittest.TestCase):
    def _result(self):
        return TranscriptionResult(
            events=[],
            song_notes=[{"time": 0, "key": "1Key0"}],
            detected_key="C major",
            bpm=120.0,
            stats={},
            warnings=[],
            engine="test",
            source_file="临时文件.wav",
            options=TranscriptionOptions(mode="audio_arrangement"),
            source=SourceMetadata(
                platform="netease",
                title='歌:名?',
                artists=("歌手/甲",),
                source_id="123",
                webpage_url="https://music.163.com/song?id=123",
                display_name="歌名 - 歌手",
            ),
        )

    def test_online_export_has_public_metadata_but_no_secret_or_direct_url(self):
        result = self._result()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "score.json")
            export_song_json(result, path)
            with open(path, "r", encoding="utf-8") as handle:
                song = json.load(handle)[0]
        self.assertEqual(song["name"], "歌:名?")
        self.assertEqual(song["author"], "歌手/甲")
        self.assertEqual(song["_transcribe"]["sourcePlatform"], "netease")
        self.assertEqual(song["_transcribe"]["sourceId"], "123")
        self.assertEqual(
            song["_transcribe"]["sourceUrl"],
            "https://music.163.com/song?id=123",
        )
        serialized = json.dumps(song, ensure_ascii=False)
        self.assertNotIn("MUSIC_U", serialized)
        self.assertNotIn("temporary.invalid", serialized)

    def test_online_stem_is_windows_safe_and_never_overwrites(self):
        stem = suggested_output_stem(self._result())
        self.assertNotRegex(stem, r'[<>:"/\\|?*]')
        self.assertIn("网易云123", stem)
        self.assertEqual(sanitize_filename_stem("CON"), "_CON")
        with tempfile.TemporaryDirectory() as directory:
            first = next_available_path(directory, stem)
            with open(first, "w", encoding="utf-8") as handle:
                handle.write("existing")
            second = next_available_path(directory, stem)
            self.assertNotEqual(first, second)
            self.assertTrue(second.endswith(" (1).json"))


if __name__ == "__main__":
    unittest.main()
