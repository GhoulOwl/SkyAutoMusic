"""Small, dependency-free description of the host platform capabilities.

The application still supports the original Windows game integration, but the
GUI can now run on macOS without importing any Windows-only modules.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformCapabilities:
    is_windows: bool
    is_macos: bool
    machine: str
    game_playback_supported: bool
    global_hotkeys_supported: bool
    overlay_supported: bool
    audio_backend: str
    credential_backend: str


def detect_platform() -> PlatformCapabilities:
    is_windows = sys.platform == "win32"
    is_macos = sys.platform == "darwin"
    return PlatformCapabilities(
        is_windows=is_windows,
        is_macos=is_macos,
        machine=platform.machine().lower(),
        game_playback_supported=is_windows,
        global_hotkeys_supported=is_windows,
        overlay_supported=is_windows,
        audio_backend="mci" if is_windows else ("pygame" if is_macos else "unsupported"),
        credential_backend="dpapi" if is_windows else ("keychain" if is_macos else "unsupported"),
    )


CAPABILITIES = detect_platform()

# Convenience flags are intentionally public so small modules do not need to
# construct their own platform policy.
IS_WINDOWS = CAPABILITIES.is_windows
IS_MACOS = CAPABILITIES.is_macos
GAME_PLAYBACK_SUPPORTED = CAPABILITIES.game_playback_supported
GLOBAL_HOTKEYS_SUPPORTED = CAPABILITIES.global_hotkeys_supported
OVERLAY_SUPPORTED = CAPABILITIES.overlay_supported
