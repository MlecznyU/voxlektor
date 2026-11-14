"""``voxlektor`` / ``vxl``: the CLI entry point and its argument dispatch.

Dispatch (``main``): a first token that names a command runs that command; anything else is
text for ``say`` (``vxl -- pause`` reads "pause", so does ``vxl say pause``); no arguments read
stdin when it is not a terminal, otherwise print the help; ``vxl config`` alone prints the
config help. On the ``say`` path a word that starts with ``-`` but is no ``say`` option (``vxl
-5 degrees``) is rejected with a hint to use ``vxl -- <text>``.

Exit codes: 0 ok, 1 user error (bad input, a ``DomainError``), 2 service unavailable (also a
local ``OSError`` such as a full or read-only disk: message on stderr, no traceback),
3 unsupported platform (see ``output`` for how error replies map), 130 interrupted (Ctrl+C;
typer turns ``KeyboardInterrupt`` into this code). Usage errors are 1, not click's 2, which
here means "service unavailable".

The control path stays light: no numpy, MLX, audio libraries, asyncio or runtime. ``serve``
and one-shot mode import their heavy modules inside the command.
"""

import sys

import typer

# The ONLY use of typer internals in the project: typer 0.29 vendors click as ``typer._click``
# and raises its exceptions, whose common base (with ``show()``) is not re-exported. typer is
# pinned to 0.29.x in pyproject.toml; the CLI tests fail at once if this moves.
from typer._click.exceptions import ClickException

from voxlektor.contracts.codec import ProtocolError
from voxlektor.domain.errors import DomainError
from voxlektor.entrypoints.cli import config, control, output, setup, speak, voices

app = typer.Typer(
    name="vxl",
    help="Read text aloud with a local voice. Bare words are text: vxl Hello world.",
    add_completion=False,
    no_args_is_help=True,
)
app.command("say")(speak.say)
app.command("serve")(control.serve)
app.command("pause")(control.pause)
app.command("resume")(control.resume)
app.command("skip")(control.skip)
app.command("back")(control.back)
app.command("stop-reading")(control.stop_reading)
app.command("status")(control.status)
app.command("idle")(control.idle)
app.command("unload")(control.unload)
app.command("stop")(control.stop)
app.command("voices")(voices.voices)
app.command("setup")(setup.setup)
app.add_typer(config.config_app, name="config")

_COMMANDS = frozenset(
    {
        "say",
        "serve",
        "pause",
        "resume",
        "skip",
        "back",
        "stop-reading",
        "status",
        "idle",
        "unload",
        "stop",
        "voices",
        "setup",
        "config",
    }
)
_HELP = "--help"
_END_OF_OPTIONS = "--"
# The options of ``say`` (must match speak.say; the CLI tests use each of them).
_SAY_LONG = frozenset({"--voice", "--lang", "--rate", "--volume", "--now", _HELP})
_SAY_SHORT = frozenset({"-v", "-l", "-r", "-f"})


def main(argv: list[str] | None = None) -> int:
    args = _route(list(sys.argv[1:] if argv is None else argv))
    if args[0] == "say" and (stray := _stray_dash_word(args[1:])) is not None:
        print(
            f"unknown option {stray!r}; to read text that starts with '-', use: vxl -- <text>",
            file=sys.stderr,
        )
        return output.EXIT_USER
    return _run(args)


def _route(args: list[str]) -> list[str]:
    if not args:
        stdin = sys.stdin
        piped = stdin is not None and not stdin.isatty()
        return ["say"] if piped else [_HELP]
    if args == ["config"]:
        return ["config", _HELP]
    if args[0] in _COMMANDS or args[0] == _HELP:
        return args
    return ["say", *args]


def _stray_dash_word(words: list[str]) -> str | None:
    """The first word before ``--`` that looks like an option but is none of ``say``'s."""
    for word in words:
        if word == _END_OF_OPTIONS:
            return None
        if word == "-" or not word.startswith("-"):  # a lone "-" is text
            continue
        known = (
            word.split("=", 1)[0] in _SAY_LONG if word.startswith("--") else word[:2] in _SAY_SHORT
        )
        if not known:
            return word
    return None


def _run(args: list[str]) -> int:
    command = typer.main.get_command(app)
    try:
        result = command.main(args=args, prog_name="vxl", standalone_mode=False)
    except ClickException as err:  # usage errors and bad parameters
        err.show()
        return output.EXIT_USER
    except typer.Abort:  # end of input while prompting; nothing prompts today
        print("aborted", file=sys.stderr)
        return output.EXIT_USER
    except DomainError as err:
        return output.fail(err)
    except ProtocolError as err:  # a request the protocol cannot carry, e.g. too long
        print(err.message, file=sys.stderr)
        return output.EXIT_USER
    except OSError as err:  # local I/O: log file, voice install on a full or read-only disk, ...
        print(f"voxlektor: {err}", file=sys.stderr)
        return output.EXIT_UNAVAILABLE
    return result if isinstance(result, int) else output.EXIT_OK
