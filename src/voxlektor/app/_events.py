"""Private building blocks of ``Session``: the events it emits to clients (spec §4.4, §5.2).

``Tags`` follows every accepted tagged ``speak`` (a *run*) from its first start to its one end.
``Watch`` remembers what the clients were last told about the floor, pauses, waiting positions
and owners with work, and emits the differences. Every event's data carries ``owner`` and
``group`` (``None`` when not applicable).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from voxlektor.app.effects import Effect, Emit
from voxlektor.domain.floor import Floor, GroupKey


def emit(
    out: list[Effect], event: str, owner: str | None, group: str | None, **data: object
) -> None:
    out.append(Emit(event, {"owner": owner, "group": group, **data}))


@dataclass
class _Run:
    tag: str
    group: GroupKey
    started: bool = False
    owed: bool = True  # a tag_finished is owed when the run ends


class Tags:
    """Tag runs by speak id; a run lives while a chunk of its speak is in a group."""

    def __init__(self) -> None:
        self._runs: dict[int, _Run] = {}

    def open(self, speak: int, tag: str, group: GroupKey) -> None:
        self._runs[speak] = _Run(tag, group)

    def start(self, speak: int, group: GroupKey, out: list[Effect]) -> None:
        """A chunk of ``speak`` starts playing; only the run's first start is announced."""
        run = self._runs[speak]
        if not run.started:
            run.started = run.owed = True
            run.group = group
            emit(out, "tag_started", *group, tag=run.tag)

    def finish(self, speak: int, reason: str, out: list[Effect]) -> None:
        run = self._runs.pop(speak)
        if run.owed:
            emit(out, "tag_finished", *run.group, tag=run.tag, reason=reason)

    def back(self, speak: int, tag: str, group: GroupKey, out: list[Effect]) -> None:
        """``back`` re-queues a chunk of ``speak`` or the chunks after it.

        A started run ends with ``back`` and starts afresh when it plays again; a finished run
        comes back and owes an end only once it starts again; an unstarted run is untouched.
        """
        run = self._runs.get(speak)
        if run is None:
            self._runs[speak] = _Run(tag, group, owed=False)
        elif run.started:
            emit(out, "tag_finished", *run.group, tag=run.tag, reason="back")
            run.started = run.owed = False


class Watch:
    """What the clients were last told; ``sync`` emits whatever changed since."""

    def __init__(self) -> None:
        self._holder: GroupKey | None = None
        self._held = False
        self._paused: set[str] = set()
        self._positions: dict[GroupKey, int] = {}
        self._owners: set[str] = set()

    def floor(self, holder: GroupKey | None, out: list[Effect]) -> None:
        """Announce a new holder; the floor only empties when its group goes (``forget``)."""
        if holder is not None and holder != self._holder:
            self._holder = holder
            emit(out, "floor", *holder)

    def forget(self, key: GroupKey) -> None:
        """Group ``key`` is gone: a new group of the same name is news again."""
        if key == self._holder:
            self._holder = None
        self._positions.pop(key, None)

    def sync(self, floor: Floor, keys: Iterable[GroupKey], out: list[Effect]) -> None:
        owners = {key[0] for key in keys}
        if floor.held != self._held:
            self._held = floor.held
            emit(out, "paused" if floor.held else "resumed", None, None)
        paused = {owner for owner in owners if floor.paused(owner)}
        for owner in sorted(self._paused - paused):
            emit(out, "resumed", owner, None)
        for owner in sorted(paused - self._paused):
            emit(out, "paused", owner, None)
        self._paused = paused
        # ``waiting`` means queued behind ANOTHER owner (spec §5.2): a group behind its own
        # owner's group is neither announced nor tracked; the position is the play-order one.
        holder = floor.holder
        positions = {
            k: p
            for k in floor.play_order()
            if (p := floor.position(k)) is not None and (holder is None or k[0] != holder[0])
        }
        for key, position in positions.items():
            if self._positions.get(key) != position:
                emit(out, "waiting", *key, position=position)
        self._positions = positions
        for owner in sorted(self._owners - owners):
            emit(out, "drained", owner, None)
        if self._owners and not owners:
            emit(out, "drained", None, None)
        self._owners = owners
