"""macOS paths; directories are created with mode 0700."""

import os
from pathlib import Path

_HOME_ENV = "VOXLEKTOR_HOME"


class MacOSPlatform:
    """Config in ``~/.config/voxlektor``, runtime in ``~/Library/Application Support/voxlektor``.

    A relocation root (the ``home`` argument, else ``$VOXLEKTOR_HOME``) puts config in
    ``<root>/config`` and runtime files in ``<root>/run``.
    """

    def __init__(self, home: Path | None = None) -> None:
        if home is None and os.environ.get(_HOME_ENV):
            home = Path(os.environ[_HOME_ENV])
        self._root = home

    def config_dir(self) -> Path:
        if self._root is not None:
            return _private_dir(self._root / "config")
        return _private_dir(Path.home() / ".config" / "voxlektor")

    def runtime_dir(self) -> Path:
        if self._root is not None:
            return _private_dir(self._root / "run")
        return _private_dir(Path.home() / "Library" / "Application Support" / "voxlektor")

    def socket_path(self) -> Path:
        return self.runtime_dir() / "voxlektor.sock"

    def log_path(self) -> Path:
        return self.runtime_dir() / "voxlektor.log"

    def pid_path(self) -> Path:
        return self.runtime_dir() / "voxlektor.pid"


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path
