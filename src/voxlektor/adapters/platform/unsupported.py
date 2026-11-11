"""Platform adapter for systems that are not supported yet."""

from pathlib import Path

from voxlektor.domain.errors import PlatformError


class UnsupportedPlatform:
    """Every call raises ``PlatformError``."""

    def config_dir(self) -> Path:
        raise PlatformError

    def runtime_dir(self) -> Path:
        raise PlatformError

    def socket_path(self) -> Path:
        raise PlatformError

    def log_path(self) -> Path:
        raise PlatformError

    def pid_path(self) -> Path:
        raise PlatformError
