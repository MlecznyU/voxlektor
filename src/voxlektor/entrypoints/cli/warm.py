"""``vxl warm``: start the daemon if needed and ask it to load the engine; returns at once.

Exits 0 once the ``load`` is requested, not when the engine is warm; the usual idle timeout
follows. Owner-less (warming is daemon-wide). Gated on setup like ``say`` (exit 1 until ``vxl
setup`` has completed); with ``service.autostart = false`` there is no daemon to warm and it
does nothing (exit 0). ``warm`` starts work, so an idle daemon of another version is replaced
first (``ClientContext.ensure`` with ``work``); a busy older daemon that does not know ``load``
is reading already, so its engine is warm: that also exits 0.
"""

from voxlektor.bootstrap.client import build_client
from voxlektor.domain.errors import DomainError
from voxlektor.entrypoints.cli.output import EXIT_OK

_UNKNOWN_COMMAND = "unknown_command"


def warm() -> int:
    """Load the engine now, so the next reading starts quickly."""
    ctx = build_client({})
    if not ctx.store.setup_completed():
        raise DomainError("not_set_up", "voxlektor is not set up yet; run: vxl setup")
    if not ctx.settings.autostart:
        return EXIT_OK
    ctx.ensure(work=True)
    try:
        ctx.send("load")
    except DomainError as err:
        if not (ctx.daemon_outdated and err.code == _UNKNOWN_COMMAND):
            raise
    return EXIT_OK
