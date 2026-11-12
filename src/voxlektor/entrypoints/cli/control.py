"""Control commands: playback, idle timeout, unload, stop, status, and ``serve``.

When nobody listens on the socket (``NotRunning``), ``status`` prints ``OFF`` and ``stop`` prints
``not running`` (exit 0). A daemon that is there but does not answer properly (any other
``ServiceUnavailable``: timeout, broken reply) is never reported as off: every command, these
two included, prints the message and exits 2. ``stop`` waits briefly until the daemon
has removed its socket, so a following command sees it gone. ``serve`` runs the daemon in the
foreground; ``entrypoints.daemon`` is imported only there (it loads asyncio and the runtime).
"""

import time
from pathlib import Path
from typing import Annotated

import typer

from voxlektor.bootstrap.client import build_client
from voxlektor.domain.errors import NotRunning
from voxlektor.entrypoints.cli.output import EXIT_OK, NOT_RUNNING, OFF, duration, status_lines

_STOP_WAIT_S = 5.0
_STOP_POLL_S = 0.1

# Replaced by tests (fake clock); production uses the real ones.
_sleep = time.sleep
_monotonic = time.monotonic


def _plain(cmd: str) -> int:
    build_client({}).send(cmd)
    return EXIT_OK


def pause() -> int:
    """Pause the reading."""
    return _plain("pause")


def resume() -> int:
    """Resume a paused reading."""
    return _plain("resume")


def skip() -> int:
    """Skip to the next chunk."""
    return _plain("skip")


def back() -> int:
    """Go back to the previous chunk."""
    return _plain("back")


def stop_reading() -> int:
    """Stop the reading; the engine stays warm."""
    return _plain("stop_reading")


def unload() -> int:
    """Unload the engine now; the daemon keeps running."""
    return _plain("unload")


def idle(
    timeout: Annotated[str, typer.Argument(metavar="DUR", help="0, 90s, 10m or 1h.")],
) -> int:
    """Set the idle timeout after which the engine is unloaded (0 = never)."""
    data = build_client({}).send("idle", timeout=timeout)
    seconds = data.get("idle_timeout_s")
    if isinstance(seconds, int):
        shown = "0 (never unload)" if seconds == 0 else duration(seconds)
        print(f"idle timeout: {shown}")
    return EXIT_OK


def status() -> int:
    """Show the daemon's state, or OFF."""
    ctx = build_client({})
    try:
        data = ctx.send("status")
    except NotRunning:
        print(OFF)
        return EXIT_OK
    for line in status_lines(data):
        print(line)
    return EXIT_OK


def stop() -> int:
    """Stop the daemon."""
    ctx = build_client({})
    try:
        ctx.send("shutdown")
    except NotRunning:
        print(NOT_RUNNING)
        return EXIT_OK
    print("stopped" if _gone(ctx.platform.socket_path()) else "stopping")
    return EXIT_OK


def serve(
    idle: Annotated[
        str | None, typer.Option("--idle", metavar="DUR", help="Idle timeout override.")
    ] = None,
) -> int:
    """Run the daemon in the foreground."""
    ctx = build_client({"idle_timeout": idle})
    from voxlektor.entrypoints import daemon  # noqa: PLC0415  # heavy: asyncio, runtime

    return daemon.serve(ctx.settings)


def _gone(path: Path) -> bool:
    """Wait until the daemon has removed ``path``; False when it is still there after a while."""
    deadline = _monotonic() + _STOP_WAIT_S
    while path.exists():
        if _monotonic() >= deadline:
            return False
        _sleep(_STOP_POLL_S)
    return True
