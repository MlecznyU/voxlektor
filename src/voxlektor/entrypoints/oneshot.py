"""One-shot reading in this process (``service.autostart = false``): the daemon's CoreLoop
without a socket, so there is no reading logic of its own.

``run`` speaks ``text`` with ``settings``, waits until the reading is over (status polled every
0.1 s on the loop), then shuts the loop down, which unloads the engine. Ctrl+C (SIGINT) stops
the reading first and exits 130 (interrupted), like any interrupted CLI command. Otherwise exit
0, or 2 when the reading failed (engine or audio, message on stderr). A rejected ``speak`` (for
example an unknown voice) raises its ``DomainError`` after cleanup.
"""

import asyncio
import contextlib
import signal
import sys
from collections.abc import Mapping

from voxlektor.app.commands import Shutdown, Speak, Status, StopReading
from voxlektor.bootstrap.daemon import Daemon, build_daemon
from voxlektor.domain.errors import DomainError
from voxlektor.domain.settings import Settings

_POLL_S = 0.1
_EXIT_OK = 0
_EXIT_FAILED = 2
_EXIT_INTERRUPTED = 130


def run(text: str, settings: Settings) -> int:
    return asyncio.run(_read(build_daemon(settings, transport=False), text))


async def _read(daemon: Daemon, text: str) -> int:
    loop = asyncio.get_running_loop()
    interrupted = asyncio.Event()
    serving = asyncio.create_task(daemon.run())
    loop.add_signal_handler(signal.SIGINT, interrupted.set)
    try:
        reply = await daemon.loop.handle(Speak(text, None, None, None, False))
        status: Mapping[str, object] = {}
        if reply.ok:
            status = await _until_idle(daemon, interrupted)
            if interrupted.is_set():
                await daemon.loop.handle(StopReading())
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        await daemon.loop.handle(Shutdown())
        await serving
    if not reply.ok:
        raise reply.error or DomainError("internal", "speak failed")
    if interrupted.is_set():
        return _EXIT_INTERRUPTED
    error = status.get("last_error")
    if isinstance(error, str):
        print(error, file=sys.stderr)
        return _EXIT_FAILED
    return _EXIT_OK


async def _until_idle(daemon: Daemon, interrupted: asyncio.Event) -> Mapping[str, object]:
    """Poll status until nothing is queued or playing, or Ctrl+C; returns the last status."""
    while True:
        status = (await daemon.loop.handle(Status())).data
        if status.get("queued") == 0 and status.get("playing") is False:
            return status
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(interrupted.wait(), _POLL_S)
        if interrupted.is_set():
            return status
