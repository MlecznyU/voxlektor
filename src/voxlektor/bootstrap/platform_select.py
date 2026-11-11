"""Pick the platform adapter for the running system."""

from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.app.ports import PlatformInfo

_MIN_MACOS_MAJOR = 14


def _macos_major(version: str) -> int:
    """Major version of strings like "14", "14.5" or "15.1.2"; 0 when unparsable."""
    head = version.split(".", 1)[0]
    return int(head) if head.isascii() and head.isdigit() else 0


def select_platform(sys_platform: str, machine: str, macos_version: str) -> PlatformInfo:
    if (
        sys_platform == "darwin"
        and machine == "arm64"
        and _macos_major(macos_version) >= _MIN_MACOS_MAJOR
    ):
        return MacOSPlatform()
    return UnsupportedPlatform()
