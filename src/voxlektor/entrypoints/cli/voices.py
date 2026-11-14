"""``vxl voices``: the local voice catalog (no protocol command exists for it).

``--json`` prints one line for tools: ``{"default": NAME, "voices": [NAME, ...]}``.
"""

import json
from typing import Annotated

import typer

from voxlektor.bootstrap.client import build_client
from voxlektor.entrypoints.cli.output import EXIT_OK


def voices(
    as_json: Annotated[
        bool, typer.Option("--json", help="One JSON line for tools: default + voices.")
    ] = False,
) -> int:
    """List the voices; * marks the default voice."""
    ctx = build_client({})
    names = ctx.catalog().list()
    if as_json:
        print(json.dumps({"default": ctx.settings.voice, "voices": names}))
        return EXIT_OK
    for name in names:
        marker = "*" if name == ctx.settings.voice else " "
        print(f"{marker} {name}")
    return EXIT_OK
