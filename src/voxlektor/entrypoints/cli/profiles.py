"""``vxl profiles [--json]`` and ``vxl profiles rm NAME``: the settings profiles in config.toml.

A profile is an overlay over ``default`` (see ``domain.profiles``). The listing shows, for every
profile (``default`` first, then by name), the effective value of each profile key and its source
(``env | profile | default | builtin``; there are no flags here). ``--json`` prints one line for
tools: ``{"profiles": [{"name": NAME, "values": {KEY: VALUE}, "sources": {KEY: SOURCE}}]}``.
``rm`` removes a profile's table; ``default`` cannot be removed, a missing name is ``unknown
profile 'NAME'`` (exit 1).

An invalid value (in a profile, a global or a ``VOXLEKTOR_*`` variable) never fails the listing:
it is reported on stderr (``ignoring invalid …``) and the next layer's value is shown with that
layer's source (``bootstrap.client.build_tolerant_client``). Only a file that is no TOML exits 1.
"""

import json
import sys
from typing import Annotated

import typer

from voxlektor.bootstrap.client import build_client, build_tolerant_client
from voxlektor.domain.profiles import PROFILE_KEYS, Source
from voxlektor.domain.settings import Settings
from voxlektor.entrypoints.cli.output import EXIT_OK

profiles_app = typer.Typer(
    help="List the settings profiles, or remove one.", invoke_without_command=True
)


@profiles_app.callback()
def listing(
    ctx: typer.Context,
    as_json: Annotated[
        bool, typer.Option("--json", help="One JSON line for tools: values + sources.")
    ] = False,
) -> int:
    """List the profiles with their effective values and where each value comes from."""
    if ctx.invoked_subcommand is not None:
        return EXIT_OK
    _, profiles = build_tolerant_client(_warn)
    rows: list[tuple[str, dict[str, object], dict[str, Source]]] = []
    for name in profiles.names:
        _, settings, sources = profiles.resolve(name)
        rows.append((name, _values(settings), {key: sources[key] for key in PROFILE_KEYS}))
    if as_json:
        listed = [{"name": name, "values": v, "sources": s} for name, v, s in rows]
        print(json.dumps({"profiles": listed}))
        return EXIT_OK
    for name, values, sources in rows:
        print(name)
        for key, value in values.items():
            print(f"  {key} = {value}  ({sources[key]})")
    return EXIT_OK


@profiles_app.command("rm")
def rm(name: Annotated[str, typer.Argument(metavar="NAME", help="A profile name.")]) -> int:
    """Remove a profile (not ``default``)."""
    build_client({}).store.remove_profile(name)
    print(f"removed profile {name!r}")
    return EXIT_OK


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def _values(settings: Settings) -> dict[str, object]:
    """The profile keys of ``settings``, in ``PROFILE_KEYS`` order, as JSON values."""
    every: dict[str, object] = {
        "voice": str(settings.voice),
        "lang": settings.lang,
        "base_lang": settings.base_lang,
        "rate": settings.rate,
        "volume": settings.volume,
        "steps": settings.steps,
    }
    return {key: every[key] for key in PROFILE_KEYS}
