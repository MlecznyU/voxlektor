"""``vxl config voice NAME`` (validated against the local catalog) and ``vxl config show``."""

from typing import Annotated

import tomli_w
import typer

from voxlektor.bootstrap.client import build_client
from voxlektor.entrypoints.cli.output import EXIT_OK

config_app = typer.Typer(help="Show or change the configuration.", no_args_is_help=True)


@config_app.command("voice")
def voice(name: Annotated[str, typer.Argument(metavar="NAME", help="A voice name.")]) -> int:
    """Set the default voice."""
    ctx = build_client({})
    ctx.catalog().resolve(name)  # UnknownVoice lists the available voices
    ctx.store.write_value("voice", name)
    print(f"voice = {name}")
    return EXIT_OK


@config_app.command("show")
def show() -> int:
    """Show the effective settings (env > config file > defaults) in the config.toml layout."""
    ctx = build_client({})
    s = ctx.settings
    effective = {
        "engine": str(s.engine),
        "voice": str(s.voice),
        "lang": s.lang,
        "base_lang": s.base_lang,
        "rate": s.rate,
        "history": s.history,
        "service": {"idle_timeout": _toml_duration(s.idle_timeout_s), "autostart": s.autostart},
    }
    print(f"# effective settings; file: {ctx.config_path}")
    print(tomli_w.dumps(effective), end="")
    return EXIT_OK


def _toml_duration(seconds: int) -> str:
    """A duration the config file accepts: ``0`` or one unit (``1h``, ``10m``, ``90s``)."""
    if seconds == 0:
        return "0"
    for size, unit in ((3600, "h"), (60, "m")):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"
