"""``vxl watch``: the daemon's events on stdout, one line each, flushed per line, until it ends.

With ``--json`` each line is one JSON object: the event frame's ``data`` plus ``"event"``; the
first is always ``snapshot`` (the status, plus ``"protocol"``: the protocol version the stream
speaks, ``PROTOCOL_VERSION``; a daemon's own ``protocol`` key would win). Without it, a
readable line per event: the name and its non-null fields as ``key=value``.

``--owner X`` keeps what concerns X: events whose ``owner`` is X, every ``floor`` event, ``state``,
and ``drained``/``error``/``paused``/``resumed`` with ``owner: null`` (daemon-wide). ``waiting``
and the tag events of other owners are dropped; the snapshot and unknown events always pass.

``--owner`` takes the protocol's owner names; the reserved ``cli`` (the plain CLI's own readings)
is rejected like ``vxl say --owner cli`` (exit 1).

No daemon: ``{"event":"daemon_off"}`` once, then a new attempt every second; ``--start``
autostarts the daemon first (as ``vxl say`` does). The daemon going away mid-stream (or a stream
that breaks, or a daemon that does not start) prints ``{"event":"disconnected"}`` and exits 2.
SIGTERM / SIGINT, and a closed stdout (``BrokenPipeError``), end it quietly with exit 0. Each
line is one write, then a flush. Blocking socket reads only (no asyncio).

Outdated daemon: a snapshot whose ``version`` differs from this CLI's (or has none), or a daemon
from before S1 that refuses ``subscribe``, is handled once per run with
``ClientContext.ensure(work=True)``, as ``vxl say`` does (``--start`` does it before connecting):
an idle daemon is replaced and the stream reconnects to the fresh one (nothing of the old stream
is printed); a busy one is kept and the snapshot line carries ``"daemon_outdated": true`` (a
busy pre-S1 daemon has no stream: ``disconnected``, exit 2). Replacing starts a fresh daemon, so
with ``service.autostart = false`` nothing is shut down: the snapshot line is just marked
(``--start`` still starts a missing daemon).
"""

import json
import os
import re
import signal
import sys
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from types import FrameType
from typing import Annotated, cast

import typer

import voxlektor
from voxlektor.adapters.transport.unix_stream import open_stream
from voxlektor.bootstrap.client import ClientContext, build_client
from voxlektor.contracts.protocol import OWNER_PATTERN, PROTOCOL_VERSION
from voxlektor.domain.errors import DomainError, NotRunning, ServiceUnavailable
from voxlektor.entrypoints.cli.output import EXIT_OK, EXIT_UNAVAILABLE

_RETRY_S = 1.0
_CONNECT_TIMEOUT_S = 5.0
_EVENT = "event"
_SNAPSHOT = "snapshot"
_DAEMON_OFF = "daemon_off"
_DISCONNECTED = "disconnected"
# Events that belong to one owner; of these, only the daemon-wide ones pass with ``owner: null``.
_OWNED = frozenset(
    {"waiting", "tag_started", "tag_finished", "paused", "resumed", "drained", "error"}
)
_DAEMON_WIDE = frozenset({"paused", "resumed", "drained", "error"})
_HUMAN = {_DAEMON_OFF: "daemon off; waiting for it", _DISCONNECTED: _DISCONNECTED}
_RESERVED_OWNER = "cli"  # the owner of requests without one; no client may name it

# Replaced by tests; production uses the real one.
_sleep = time.sleep

type _Emit = Callable[[Mapping[str, object]], None]


class _Terminated(BaseException):
    """SIGTERM arrived (raised from the handler, like ``KeyboardInterrupt`` for SIGINT)."""


def watch(
    as_json: Annotated[
        bool, typer.Option("--json", help="One JSON object per line (for tools).")
    ] = False,
    owner: Annotated[
        str | None, typer.Option("--owner", help="Only what concerns this owner.")
    ] = None,
    start: Annotated[
        bool, typer.Option("--start", help="Start the daemon if it is not running.")
    ] = False,
) -> int:
    """Print the daemon's events as they happen, until it goes away."""
    if owner is not None:
        _check_owner(owner)
    ctx = build_client({})
    emit = _json_line if as_json else _human_line
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        return _watch_or_disconnect(ctx, emit, owner, start=start)
    except BrokenPipeError:  # the reader of stdout went away: nobody left to tell
        _silence_stdout()
        return EXIT_OK
    except (KeyboardInterrupt, _Terminated):
        return EXIT_OK
    finally:
        signal.signal(signal.SIGTERM, previous)


def _silence_stdout() -> None:
    """Point stdout at ``/dev/null``, so the interpreter's exit-time flush cannot fail again."""
    try:
        fd = sys.stdout.fileno()
    except (OSError, ValueError):  # not a real file (``io.UnsupportedOperation``)
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    os.close(devnull)


def _check_owner(owner: str) -> None:
    if owner == _RESERVED_OWNER:
        raise DomainError("invalid_argument", f"owner {_RESERVED_OWNER!r} is reserved")
    if re.fullmatch(OWNER_PATTERN, owner) is None:
        raise DomainError(
            "invalid_argument",
            f"invalid owner {owner!r}: 1 to 64 of the characters A-Z a-z 0-9 . _ : -",
        )


def _watch_or_disconnect(ctx: ClientContext, emit: _Emit, owner: str | None, *, start: bool) -> int:
    try:
        return _watch(ctx, emit, owner, start=start)
    except ServiceUnavailable as err:  # a broken stream, or a daemon that did not start
        return _disconnected(emit, err)


def _watch(ctx: ClientContext, emit: _Emit, owner: str | None, *, start: bool) -> int:
    path = ctx.platform.socket_path()
    replace = ctx.settings.autostart  # replacing a daemon means starting a fresh one
    checked = start and replace  # the outdated-daemon check runs once per run
    if start:
        ctx.ensure(work=replace)
    announced = False
    while True:
        try:
            stream = open_stream(path, _CONNECT_TIMEOUT_S)
        except NotRunning:
            if not announced:
                emit({_EVENT: _DAEMON_OFF})
                announced = True
            _sleep(_RETRY_S)
            continue
        except ServiceUnavailable:
            # A daemon from before S1 refuses ``subscribe``: replace it (once) when idle.
            if checked or not replace:
                raise
            checked = True
            if not _replaced(ctx):
                raise
            continue  # follow the fresh daemon
        snapshot = next(stream, None)
        if snapshot is not None and _outdated(snapshot) and not checked and replace:
            checked = True
            if _replaced(ctx):
                _close(stream)
                continue  # follow the fresh daemon
        if snapshot is not None:
            emit({**snapshot, "daemon_outdated": True} if _outdated(snapshot) else snapshot)
        for event in stream:
            if owner is None or _concerns(event, owner):
                emit(event)
        return _disconnected(emit, None)


def _replaced(ctx: ClientContext) -> bool:
    """``ensure(work=True)``: True when an outdated daemon was replaced by a fresh one, False
    when it was busy and is kept (``daemon_outdated``)."""
    ctx.ensure(work=True)
    return not ctx.daemon_outdated


def _disconnected(emit: _Emit, err: ServiceUnavailable | None) -> int:
    if err is not None:  # a broken stream: the daemon is no longer usable
        print(err.message, file=sys.stderr)
    emit({_EVENT: _DISCONNECTED})
    return EXIT_UNAVAILABLE


def _outdated(snapshot: Mapping[str, object]) -> bool:
    return snapshot.get("version") != voxlektor.__version__


def _close(stream: Iterator[object]) -> None:
    """Close the old daemon's stream now rather than when it is collected (``open_stream``
    returns a generator that owns the socket)."""
    cast("Generator[object]", stream).close()


def _concerns(event: Mapping[str, object], owner: str) -> bool:
    name = event.get(_EVENT)
    if name not in _OWNED:  # snapshot, floor, state, and events this version does not know
        return True
    who = event.get("owner")
    return who == owner or (who is None and name in _DAEMON_WIDE)


def _write(line: str) -> None:
    """One whole line in one write, then flushed (a signal cannot split it)."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _json_line(event: Mapping[str, object]) -> None:
    if event.get(_EVENT) == _SNAPSHOT:  # tools gate on the protocol version (absent means 1)
        event = {"protocol": PROTOCOL_VERSION, **event}
    _write(json.dumps(event))


def _human_line(event: Mapping[str, object]) -> None:
    name = str(event.get(_EVENT))
    if name in _HUMAN:
        _write(_HUMAN[name])
        return
    fields = [
        f"{key}={_show(value)}"
        for key, value in event.items()
        if key != _EVENT and value is not None
    ]
    _write(" ".join([name, *fields]))


def _show(value: object) -> str:
    if isinstance(value, list):
        items = cast("list[object]", value)
        return ",".join(str(item) for item in items)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _terminate(signum: int, frame: FrameType | None) -> None:
    raise _Terminated
