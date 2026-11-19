"""The daemon's single-instance lock seen from the client: is it held, by whom, and stopping it.

A daemon holds an exclusive ``flock`` on ``<runtime dir>/voxlektor.lock`` (``LOCK_NAME``) for its
whole life (``entrypoints.daemon``). A daemon that holds it while its socket answers nothing (a
hung one, e.g. stuck in its final engine unload) blocks every new start.

``lock_held``: tries the lock without blocking and releases it at once; a missing file is not
held (this module never creates it).

``find_holder``: the openers of the lock file (``lsof -t``, without this process) and the pid in
the pid file name the holder: the pid-file pid when it has the lock file open, else the only
opener, else the pid-file pid, else unknown. ``verified`` only when that pid has the lock file
open AND its command line runs ``serve`` of voxlektor (``python -m voxlektor serve``, ``vxl
serve``, ``voxlektor serve``, read with ``ps``). Callers signal verified holders only.

``terminate``: SIGTERM, up to ``grace_s`` for the process to exit, then SIGKILL and up to
``_KILL_WAIT_S`` more. ``"gone"`` (it had already exited), ``"terminated"``, ``"killed"`` or
``"survived"`` (still there after SIGKILL, e.g. stuck in the kernel, or not ours to signal).
"""

import fcntl
import itertools
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LOCK_NAME = "voxlektor.lock"

type Outcome = Literal["gone", "terminated", "killed", "survived"]

_LSOF = "/usr/sbin/lsof"
_PS = "/bin/ps"
_PROGRAMS = frozenset({"voxlektor", "vxl"})
_TOOL_TIMEOUT_S = 5.0
_KILL_WAIT_S = 2.0
_POLL_S = 0.1

# Replaced by tests (fake clock and processes); production uses the real ones.
_sleep = time.sleep
_monotonic = time.monotonic
_kill = os.kill


@dataclass(frozen=True, slots=True)
class LockHolder:
    """The process holding the daemon lock: ``pid`` None when it cannot be told."""

    pid: int | None
    verified: bool


def lock_path(runtime_dir: Path) -> Path:
    """Where the daemon's single-instance lock lives."""
    return runtime_dir / LOCK_NAME


def lock_held(lock_path: Path) -> bool:
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)  # also releases the lock when it was taken here
    return False


def find_holder(lock_path: Path, pid_path: Path) -> LockHolder:
    openers = _openers(lock_path)
    recorded = _recorded_pid(pid_path)
    if recorded is not None and recorded in openers:
        pid: int | None = recorded
    elif len(openers) == 1:
        (pid,) = openers
    else:
        pid = recorded
    if pid is None:
        return LockHolder(None, verified=False)
    return LockHolder(pid, verified=pid in openers and _is_serve(pid))


def terminate(pid: int, grace_s: float = 3.0) -> Outcome:
    try:
        _kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "survived"
    if _exits_within(pid, grace_s):
        return "terminated"
    try:
        _kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"  # exited just after the grace
    except PermissionError:
        return "survived"
    return "killed" if _exits_within(pid, _KILL_WAIT_S) else "survived"


def _exits_within(pid: int, seconds: float) -> bool:
    deadline = _monotonic() + seconds
    while _alive(pid):
        if _monotonic() >= deadline:
            return False
        _sleep(_POLL_S)
    return True


def _alive(pid: int) -> bool:
    try:
        _kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, another user's
    return True


def _openers(lock_path: Path) -> frozenset[int]:
    out = _output([_LSOF, "-t", str(lock_path)]) or ""
    pids = {int(line) for line in out.split() if line.isdigit()}
    return frozenset(pids - {os.getpid()})


def _recorded_pid(pid_path: Path) -> int | None:
    try:
        text = pid_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return int(text) if text.isdigit() else None


def _is_serve(pid: int) -> bool:
    words = (_output([_PS, "-p", str(pid), "-o", "command="]) or "").split()
    return any(
        Path(word).name in _PROGRAMS and following == "serve"
        for word, following in itertools.pairwise(words)
    )


def _output(argv: list[str]) -> str | None:
    """The stdout of a fixed argv (never a shell); None when it fails, is missing or hangs."""
    try:
        done = subprocess.run(  # noqa: S603  # fixed tool paths, no shell
            argv, capture_output=True, text=True, timeout=_TOOL_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None
