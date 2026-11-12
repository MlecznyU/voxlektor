"""Daemon autostart from the client side.

``ensure_daemon``: when ``ping()`` fails, a stale REGULAR file at the socket path is removed
(ruling R34: anything else there, such as a socket whose connect is refused, is left to the
daemon, which probes and removes it under its flock, so a client never races a starting
daemon). Then ``python -m voxlektor serve`` is spawned with an argv list (no shell), in a new
session, with stdin from ``/dev/null`` and stdout/stderr appended to the platform's log file.
``ping`` is retried every 0.1 s until it succeeds or ``wait_s`` has passed. The spawned process
is not watched (ruling R33): when an old daemon is still shutting down, the new ``serve`` exits
with "already running" after a few seconds, yet the old daemon's successor may still come up
within ``wait_s``.
"""

import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from voxlektor.app.ports import PlatformInfo
from voxlektor.domain.errors import ServiceUnavailable

_POLL_S = 0.1

# Replaced by tests (fake clock); production uses the real ones.
_sleep = time.sleep
_monotonic = time.monotonic


def ensure_daemon(platform: PlatformInfo, ping: Callable[[], bool], wait_s: float = 15.0) -> None:
    if ping():
        return
    _remove_stale_file(platform.socket_path())
    log_path = platform.log_path()
    with log_path.open("ab") as log:
        subprocess.Popen(  # fixed argv, never a shell
            [sys.executable, "-m", "voxlektor", "serve"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd="/",
        )
    deadline = _monotonic() + wait_s
    while not ping():
        if _monotonic() >= deadline:
            raise ServiceUnavailable(f"voxlektor did not start within {wait_s:g} s; see {log_path}")
        _sleep(_POLL_S)


def _remove_stale_file(path: Path) -> None:
    """Remove ``path`` only if it is a regular file (a daemon never leaves one behind)."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise ServiceUnavailable(f"cannot remove the stale file {path}: {exc}") from None
