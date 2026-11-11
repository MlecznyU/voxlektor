"""Daemon entry point: ``serve`` runs the service in the foreground until it is stopped.

Order:

1. Check the platform: unsupported -> message on stderr, exit 3.
2. Single instance: take an exclusive ``flock`` on ``<runtime dir>/voxlektor.lock`` and hold it
   for the daemon's whole lifetime (released by the kernel even on a crash). When another
   process holds it: if a daemon answers ``status`` with ok, print ``already running`` and exit
   1 at once; otherwise the holder is starting up or shutting down, so retry the lock every
   50 ms for up to 5 s (a shutting-down daemon frees it once its worker has finished), then
   print ``already running`` and exit 1.
3. With the lock held, probe the socket: any decodable reply (or a listener that does not
   answer) means something is alive -> ``already running``, exit 1. Only "no listener"
   (``ENOENT``, ``ECONNREFUSED``, not a socket) means stale: the file is removed.
4. Log to the platform's rotating log (1 MB x 2) and run the daemon (it writes the pid file once
   the socket listens and removes only its own socket and pid file when it stops). SIGTERM and
   SIGINT shut it down gracefully, like a ``shutdown`` request.

Exit codes: 0 clean stop, 1 already running, 2 the daemon failed (service unavailable),
3 unsupported platform.
"""

import asyncio
import contextlib
import enum
import errno
import fcntl
import logging
import logging.handlers
import os
import platform as host
import signal
import socket
import sys
import threading
import time
from collections.abc import Generator
from pathlib import Path

from voxlektor.app.commands import Shutdown
from voxlektor.app.effects import Reply
from voxlektor.bootstrap.daemon import Daemon, build_daemon
from voxlektor.bootstrap.platform_select import select_platform
from voxlektor.contracts import protocol as p
from voxlektor.contracts.codec import ProtocolError, decode_reply, encode_line
from voxlektor.domain.errors import PlatformError
from voxlektor.domain.settings import Settings

_log = logging.getLogger(__name__)

_EXIT_OK = 0
_EXIT_ALREADY_RUNNING = 1
_EXIT_UNAVAILABLE = 2
_EXIT_UNSUPPORTED = 3
_LOCK_NAME = "voxlektor.lock"
_LOCK_WAIT_S = 5.0
_LOCK_RETRY_S = 0.05
_LOG_BYTES = 1_000_000
_LOG_BACKUPS = 2
_PROBE_TIMEOUT_S = 1.0
_STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)
_NO_LISTENER = frozenset({errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK})


class _Probe(enum.Enum):
    NONE = enum.auto()  # nothing listens on the socket path: a leftover file at most
    READY = enum.auto()  # a daemon answered status with ok
    BUSY = enum.auto()  # something listens but is not ready (e.g. shutting down)


def serve(settings: Settings) -> int:
    platform = select_platform(sys.platform, host.machine(), host.mac_ver()[0])
    try:
        socket_path = platform.socket_path()
        log_path = platform.log_path()
        lock_path = platform.runtime_dir() / _LOCK_NAME
    except PlatformError as err:
        print(err.message, file=sys.stderr)
        return _EXIT_UNSUPPORTED
    with _single_instance(lock_path, socket_path) as owned:
        if not owned or _probe(socket_path) is not _Probe.NONE:
            print("already running")
            return _EXIT_ALREADY_RUNNING
        socket_path.unlink(missing_ok=True)  # stale: nobody listens on it
        with _rotating_log(log_path):
            _log.info("daemon started (pid %d, engine %s)", os.getpid(), settings.engine)
            try:
                asyncio.run(_run(build_daemon(settings, transport=True, platform=platform)))
            except Exception:
                _log.exception("daemon failed")
                return _EXIT_UNAVAILABLE
            _log.info("daemon stopped")
    return _EXIT_OK


async def _run(daemon: Daemon) -> None:
    loop = asyncio.get_running_loop()
    stopping: set[asyncio.Task[Reply]] = set()

    def finished(task: asyncio.Task[Reply]) -> None:
        stopping.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            _log.error("shutdown on a signal failed", exc_info=exc)

    def stop() -> None:
        task = loop.create_task(daemon.loop.handle(Shutdown()))
        stopping.add(task)
        task.add_done_callback(finished)

    handle_signals = threading.current_thread() is threading.main_thread()
    if handle_signals:
        for sig in _STOP_SIGNALS:
            loop.add_signal_handler(sig, stop)
    try:
        await daemon.run()
    finally:
        if handle_signals:
            for sig in _STOP_SIGNALS:
                loop.remove_signal_handler(sig)


@contextlib.contextmanager
def _single_instance(lock_path: Path, socket_path: Path) -> Generator[bool]:
    """Hold the daemon lock while the block runs; yields False when it could not be taken."""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        yield _take_lock(fd, socket_path)
    finally:
        os.close(fd)  # releases the lock


def _take_lock(fd: int, socket_path: Path) -> bool:
    deadline = time.monotonic() + _LOCK_WAIT_S
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            return True
        if _probe(socket_path) is _Probe.READY or time.monotonic() >= deadline:
            return False
        time.sleep(_LOCK_RETRY_S)  # the holder is starting up or shutting down


def _probe(path: Path) -> _Probe:
    """Ask whatever listens on ``path`` for its status."""
    request = encode_line(p.StatusRequest(v=p.PROTOCOL_VERSION, id="serve-probe"))
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_PROBE_TIMEOUT_S)
            sock.connect(str(path))
            sock.sendall(request)
            with sock.makefile("rb") as stream:
                line = stream.readline(p.MAX_FRAME)
    except OSError as exc:
        return _Probe.NONE if exc.errno in _NO_LISTENER else _Probe.BUSY
    try:
        reply = decode_reply(line)
    except ProtocolError:
        return _Probe.BUSY
    return _Probe.READY if reply.ok else _Probe.BUSY


@contextlib.contextmanager
def _rotating_log(path: Path) -> Generator[None]:
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=_LOG_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    previous = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)
        handler.close()
