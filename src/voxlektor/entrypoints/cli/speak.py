"""``vxl say``: read text aloud through the daemon (autostarted), or in-process one-shot.

The text is the arguments joined with spaces, or the file given with ``-f`` (a regular file), or
stdin when it is not a terminal. Files and stdin are read bounded: more than ``MAX_FRAME`` bytes
is ``text too long`` (exit 1) before any client is built or daemon started. Text that
``split_text`` turns into nothing prints ``nothing to read`` and exits 0 without touching the
socket. Every request carries the effective voice, language, base language and rate (flag > env >
config > defaults), so a changed config or ``VOXLEKTOR_*`` value applies to the next reading
without restarting the daemon. With ``service.autostart = false`` the text is read in this process
(``entrypoints.oneshot``, imported only on that path because it loads the engine stack).
"""

import stat
import sys
from pathlib import Path
from typing import Annotated, TextIO

import typer

from voxlektor.bootstrap.client import build_client
from voxlektor.contracts.protocol import MAX_FRAME
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
) -> int:
    """Read text aloud (the default command)."""
    content = _resolve_text(text or [], file)
    if not split_text(content):
        print(NOTHING_TO_READ)
        return EXIT_OK
    ctx = build_client({"voice": voice, "lang": lang, "rate": rate})
    if not ctx.settings.autostart:
        from voxlektor.entrypoints import oneshot  # noqa: PLC0415  # heavy: engine stack

        return oneshot.run(content, ctx.settings)
    ctx.ensure()
    settings = ctx.settings
    ctx.send(
        "speak",
        text=content,
        voice=settings.voice,
        lang=settings.lang,
        base_lang=settings.base_lang,
        rate=settings.rate,
        now=now,
    )
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
