"""Control commands: playback, idle timeout, unload, stop, status, and ``serve``.

``status --json`` prints one JSON line for tools: ``{"running": false}`` when no daemon runs,
otherwise ``{"running": true, ...}`` with the protocol ``status`` fields. ``running`` means "a
daemon process is there", not "it answers"; see ``answering`` below. Every line also carries
``"protocol"``: the protocol version this CLI speaks (``PROTOCOL_VERSION``; a daemon that
answers has replied in it), so tools can gate on it (absent, from CLIs before 0.3.0: 1).

When nobody listens on the socket (``NotRunning``), ``status`` prints ``OFF`` and ``stop`` prints
``not running`` (exit 0). A daemon that is there but does not answer properly (any other
``ServiceUnavailable``: timeout, broken reply) is never reported as off: every command, these
two included, prints the message and exits 2. ``stop`` waits briefly until the daemon
has removed its socket, so a following command sees it gone.

A daemon that holds the lock but does not answer (no socket, or one that times out) within a
2 s grace (it may be starting up or finishing a stop): ``status --json`` prints
``{"running": true, "answering": false, "pid": N}`` (``pid`` null when it cannot be told; exit
0); plain ``status`` prints how to stop it and exits 2. ``stop`` gives it 5 s to answer or
release the lock (it may be finishing an ordinary stop), then SIGTERM, 3 s, SIGKILL
(``ClientContext.stop_hung``) and prints what it did; a holder that is not verifiably a voxlektor
daemon is never signalled (exit 2, its pid named). ``serve`` runs the daemon in the
foreground; ``entrypoints.daemon`` is imported only there (it loads asyncio and the runtime).
"""

import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from voxlektor.adapters.procs.lock_holder import LockHolder
from voxlektor.bootstrap.client import ClientContext, build_client, hung_message
from voxlektor.contracts.protocol import PROTOCOL_VERSION
from voxlektor.domain.errors import NotRunning, ServiceUnavailable
from voxlektor.entrypoints.cli.output import EXIT_OK, NOT_RUNNING, OFF, duration, status_lines

_HUNG_GRACE_S = 5.0  # `stop`: a daemon finishing an ordinary stop releases the lock meanwhile
_STATUS_GRACE_S = 2.0  # `status`: a daemon starting up or stopping is not called hung at once
_STOP_WAIT_S = 5.0
_STOP_POLL_S = 0.1

# Replaced by tests (fake clock); production uses the real ones.
_sleep = time.sleep
_monotonic = time.monotonic


type _Owner = Annotated[
    str | None, typer.Option("--owner", help="Act on this owner's reading only.")
]


class NavUnit(StrEnum):
    """How far ``skip`` and ``back`` move (``tag``: needs a daemon with the ``tag_nav`` cap)."""

    chunk = "chunk"
    tag = "tag"


type _Unit = Annotated[
    NavUnit,
    typer.Option("--unit", help="chunk, or tag: the whole tag run (a paragraph of a client)."),
]


def _plain(cmd: str, owner: str | None = None, unit: NavUnit = NavUnit.chunk) -> int:
    args: dict[str, object] = {} if owner is None else {"owner": owner}
    if unit is not NavUnit.chunk:
        args["unit"] = unit.value  # daemons before 0.4.0 ignore it: a plain chunk skip/back
    build_client({}).send(cmd, **args)
    return EXIT_OK


def pause(owner: _Owner = None) -> int:
    """Pause the reading."""
    return _plain("pause", owner)


def resume(owner: _Owner = None) -> int:
    """Resume a paused reading."""
    return _plain("resume", owner)


def skip(owner: _Owner = None, unit: _Unit = NavUnit.chunk) -> int:
    """Skip to the next chunk (``--unit tag``: to the next tag)."""
    return _plain("skip", owner, unit)


def back(owner: _Owner = None, unit: _Unit = NavUnit.chunk) -> int:
    """Go back to the previous chunk (``--unit tag``: to the start of the tag, or the previous)."""
    return _plain("back", owner, unit)


def stop_reading(owner: _Owner = None) -> int:
    """Stop the reading; the engine stays warm."""
    return _plain("stop_reading", owner)


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


def status(
    as_json: Annotated[
        bool, typer.Option("--json", help="One JSON line for tools: running + the status.")
    ] = False,
) -> int:
    """Show the daemon's state, or OFF."""
    ctx = build_client({})
    try:
        data = ctx.send("status")
    except ServiceUnavailable as err:
        holder = ctx.hung_daemon(_STATUS_GRACE_S)
        if holder is not None:
            if not as_json:
                raise ServiceUnavailable(hung_message(holder)) from None
            hung = {"running": True, "answering": False, "pid": holder.pid}
            print(json.dumps({**hung, "protocol": PROTOCOL_VERSION}))
            return EXIT_OK
        if not isinstance(err, NotRunning):
            raise
        print(json.dumps({"running": False, "protocol": PROTOCOL_VERSION}) if as_json else OFF)
        return EXIT_OK
    if as_json:
        print(json.dumps({"running": True, "protocol": PROTOCOL_VERSION, **data}))
        return EXIT_OK
    for line in status_lines(data):
        print(line)
    return EXIT_OK


def stop() -> int:
    """Stop the daemon."""
    ctx = build_client({})
    try:
        ctx.send("shutdown")
    except ServiceUnavailable as err:
        holder = ctx.hung_daemon(_HUNG_GRACE_S)
        if holder is not None:
            print(_stop_hung(ctx, holder))
            return EXIT_OK
        if not isinstance(err, NotRunning):
            raise
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


def _stop_hung(ctx: ClientContext, holder: LockHolder) -> str:
    """Signal a lock holder that does not answer; what was done, or ``ServiceUnavailable``."""
    outcome = ctx.stop_hung(holder)
    who = f"the daemon that did not answer (pid {holder.pid})"
    if outcome == "survived":
        raise ServiceUnavailable(f"{who} did not exit even after SIGKILL")
    if outcome == "gone":
        return f"{who} has exited"
    if outcome == "killed":
        return f"stopped {who}: SIGKILL after it ignored SIGTERM for 3 s"
    return f"stopped {who}: it exited on SIGTERM"


def _gone(path: Path) -> bool:
    """Wait until the daemon has removed ``path``; False when it is still there after a while."""
    deadline = _monotonic() + _STOP_WAIT_S
    while path.exists():
        if _monotonic() >= deadline:
            return False
        _sleep(_STOP_POLL_S)
    return True
