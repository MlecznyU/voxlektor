"""``vxl say``: read text aloud through the daemon (autostarted), or in-process one-shot.

The text is the arguments joined with spaces, or the file given with ``-f`` (a regular file), or
stdin when it is not a terminal. Files and stdin are read bounded: more than ``MAX_FRAME`` bytes
is ``text too long`` (exit 1) before any client is built or daemon started. Text that
``split_text`` turns into nothing prints ``nothing to read`` and exits 0 without touching the
socket. Every request carries the effective voice, language, base language, rate and volume (flag >
env > config > defaults), so a changed config or ``VOXLEKTOR_*`` value applies to the next reading
without restarting the daemon. With ``service.autostart = false`` the text is read in this process
(``entrypoints.oneshot``, imported only on that path because it loads the engine stack).

Speaking is gated on setup: until ``vxl setup`` has completed (the store's marker), any text,
from any source and also in one-shot mode, exits 1 with ``voxlektor is not set up yet; run: vxl
setup``. The gate is in the CLI only; the daemon and the protocol do not check it.

``say`` starts work, so it replaces an idle daemon of another version (``ClientContext.ensure``
with ``work``); a busy one is kept and ``--json`` adds ``"daemon_outdated": true``.
"""

import json
import stat
import sys
from pathlib import Path
from typing import Annotated, TextIO

import typer

from voxlektor.bootstrap.client import build_client
from voxlektor.contracts.protocol import MAX_FRAME
from voxlektor.domain.errors import DomainError
from voxlektor.domain.text.chunking import split_text
from voxlektor.entrypoints.cli.output import EXIT_OK, NOTHING_TO_READ

_READ_LIMIT = MAX_FRAME + 1  # one byte more than any request can carry, to detect "too long"
_TOO_LONG = f"text too long: at most {MAX_FRAME} bytes fit in one request"


def say(  # noqa: PLR0913, PLR0917  # one parameter per frozen CLI option
    text: Annotated[
        list[str] | None,
        typer.Argument(help="Text to read; words are joined with spaces.", show_default=False),
    ] = None,
    voice: Annotated[str | None, typer.Option("--voice", "-v", help="Voice name.")] = None,
    lang: Annotated[str | None, typer.Option("--lang", "-l", help="auto, pl or en.")] = None,
    rate: Annotated[float | None, typer.Option("--rate", "-r", help="Rate, 0.5 to 3.0.")] = None,
    now: Annotated[bool, typer.Option("--now", help="Replace the current reading.")] = False,
    file: Annotated[
        Path | None, typer.Option("-f", metavar="FILE", help="Read the text from FILE.")
    ] = None,
    volume: Annotated[
        int | None, typer.Option("--volume", help="Volume, 0 to 100; 50 = even level.")
    ] = None,
    owner: Annotated[
        str | None, typer.Option("--owner", help="Owner of the reading (a named client).")
    ] = None,
    group: Annotated[
        str | None, typer.Option("--group", help="Reading group the text belongs to.")
    ] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Tag echoed back in the events.")] = None,
    preempt: Annotated[
        bool, typer.Option("--preempt", help="Take the floor from the current owner.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the daemon's reply data as one JSON line.")
    ] = False,
) -> int:
    """Read text aloud (the default command)."""
    content = _resolve_text(text or [], file)
    if not split_text(content):
        print(NOTHING_TO_READ)
        return EXIT_OK
    ctx = build_client({"voice": voice, "lang": lang, "rate": rate, "volume": volume})
    if not ctx.store.setup_completed():
        raise DomainError("not_set_up", "voxlektor is not set up yet; run: vxl setup")
    if not ctx.settings.autostart:
        from voxlektor.entrypoints import oneshot  # noqa: PLC0415  # heavy: engine stack

        return oneshot.run(content, ctx.settings)
    ctx.ensure(work=True)
    settings = ctx.settings
    extra: dict[str, object] = {"preempt": True} if preempt else {}
    for key, value in (("owner", owner), ("group", group), ("tag", tag)):
        if value is not None:
            extra[key] = value
    data = ctx.send(
        "speak",
        text=content,
        voice=settings.voice,
        lang=settings.lang,
        base_lang=settings.base_lang,
        rate=settings.rate,
        now=now,
        volume=settings.volume,
        **extra,
    )
    if as_json:
        print(json.dumps({**data, "daemon_outdated": True} if ctx.daemon_outdated else data))
    return EXIT_OK


def _resolve_text(words: list[str], file: Path | None) -> str:
    if file is not None:
        if words:
            raise typer.BadParameter("give either TEXT or -f FILE, not both")
        return _decode(_read_file(file), str(file))
    if words:
        text = " ".join(words)
        if len(text.encode("utf-8")) > MAX_FRAME:
            raise typer.BadParameter(_TOO_LONG)
        return text
    stdin = sys.stdin
    if stdin is None or stdin.isatty():
        raise typer.BadParameter("no text: give TEXT, -f FILE or pipe it on stdin")
    return _decode(_read_stdin(stdin), "stdin")


def _read_file(file: Path) -> bytes:
    """At most ``_READ_LIMIT`` bytes of a regular file (a FIFO or a device may block forever)."""
    try:
        if not stat.S_ISREG(file.stat().st_mode):
            raise typer.BadParameter(f"cannot read {file}: not a regular file")
        with file.open("rb") as fh:
            return fh.read(_READ_LIMIT)
    except OSError as exc:
        raise typer.BadParameter(f"cannot read {file}: {exc}") from None


def _read_stdin(stdin: TextIO) -> bytes:
    """At most ``_READ_LIMIT`` bytes of stdin (characters when it has no byte buffer)."""
    buffer = getattr(stdin, "buffer", None)
    if buffer is None:
        return stdin.read(_READ_LIMIT).encode("utf-8")
    raw: bytes = buffer.read(_READ_LIMIT)
    return raw


def _decode(raw: bytes, source: str) -> str:
    if len(raw) > MAX_FRAME:
        raise typer.BadParameter(_TOO_LONG)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(f"cannot read {source}: {exc}") from None
