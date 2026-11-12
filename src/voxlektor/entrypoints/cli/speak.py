"""``vxl say``: read text aloud through the daemon (autostarted), or in-process one-shot.

The text is the arguments joined with spaces, or the file given with ``-f``, or stdin when it is
not a terminal. Text that ``split_text`` turns into nothing prints ``nothing to read`` and exits
0 without touching the socket. The request carries the effective voice, language and rate
(flag > env > config > defaults), so a changed config applies without restarting the daemon.
With ``service.autostart = false`` the text is read in this process (``entrypoints.oneshot``,
imported only on that path because it loads the engine stack).
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

from voxlektor.bootstrap.client import build_client
from voxlektor.domain.text.chunking import split_text
from voxlektor.entrypoints.cli.output import EXIT_OK, NOTHING_TO_READ


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
        rate=settings.rate,
        now=now,
    )
    return EXIT_OK


def _resolve_text(words: list[str], file: Path | None) -> str:
    if file is not None:
        if words:
            raise typer.BadParameter("give either TEXT or -f FILE, not both")
        try:
            return file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise typer.BadParameter(f"cannot read {file}: {exc}") from None
    if words:
        return " ".join(words)
    stdin = sys.stdin
    if stdin is None or stdin.isatty():
        raise typer.BadParameter("no text: give TEXT, -f FILE or pipe it on stdin")
    return stdin.read()
