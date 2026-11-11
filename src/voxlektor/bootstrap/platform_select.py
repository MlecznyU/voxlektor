"""Pick the platform adapter for the running system."""

from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.app.ports import PlatformInfo


def select_platform(sys_platform: str, machine: str) -> PlatformInfo:
    if sys_platform == "darwin" and machine == "arm64":
        return MacOSPlatform()
    return UnsupportedPlatform()
