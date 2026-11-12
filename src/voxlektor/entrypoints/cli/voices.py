"""``vxl voices``: the local voice catalog (no protocol command exists for it)."""

from voxlektor.bootstrap.client import build_client
from voxlektor.entrypoints.cli.output import EXIT_OK


def voices() -> int:
    """List the voices; * marks the default voice."""
    ctx = build_client({})
    for name in ctx.catalog().list():
        marker = "*" if name == ctx.settings.voice else " "
        print(f"{marker} {name}")
    return EXIT_OK
