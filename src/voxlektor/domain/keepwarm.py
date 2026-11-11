"""Keep-warm lifecycle of the engine: a pure state machine.

Time never comes from a clock; it arrives inside events (``Idle(now)``, ``Wake(now)``).
``transition`` returns the next state and the actions the caller must perform.

Events that a state does not act on are ignored explicitly: the state is unchanged and no
actions are produced.

Flags: ``unload_after_load`` means "unload right after the current or next load completes";
``reload_after_unload`` means "start a load when the current unload completes".
``idle_since`` is the time of the last ``Idle`` while WARM (also when the timeout is 0), so a
later ``SetTimeout`` can reschedule from the original idle start.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import assert_never


class Life(StrEnum):
    COLD = "cold"
    LOADING = "loading"
    WARM = "warm"
    UNLOADING = "unloading"


@dataclass(frozen=True, slots=True)
class NeedEngine:
    pass


@dataclass(frozen=True, slots=True)
class Loaded:
    pass


@dataclass(frozen=True, slots=True)
class LoadFailed:
    msg: str


@dataclass(frozen=True, slots=True)
class Idle:
    now: float


@dataclass(frozen=True, slots=True)
class Busy:
    pass


@dataclass(frozen=True, slots=True)
class UnloadRequested:
    pass


@dataclass(frozen=True, slots=True)
class Unloaded:
    pass


@dataclass(frozen=True, slots=True)
class Wake:
    now: float


@dataclass(frozen=True, slots=True)
class SetTimeout:
    seconds: int


type Event = (
    NeedEngine | Loaded | LoadFailed | Idle | Busy | UnloadRequested | Unloaded | Wake | SetTimeout
)


@dataclass(frozen=True, slots=True)
class StartLoad:
    pass


@dataclass(frozen=True, slots=True)
class StartUnload:
    pass


@dataclass(frozen=True, slots=True)
class ScheduleWake:
    at: float


@dataclass(frozen=True, slots=True)
class CancelWake:
    pass


@dataclass(frozen=True, slots=True)
class FlushPending:
    pass


type Action = StartLoad | StartUnload | ScheduleWake | CancelWake | FlushPending


@dataclass(frozen=True)
class KeepWarm:
    life: Life
    timeout_s: int
    deadline: float | None
    unload_after_load: bool
    reload_after_unload: bool = False
    idle_since: float | None = None


type _Result = tuple[KeepWarm, list[Action]]


def _unchanged(s: KeepWarm) -> _Result:
    return s, []


def _with_timeout(s: KeepWarm, seconds: int) -> _Result:
    return replace(s, timeout_s=max(0, seconds)), []


def _cold(s: KeepWarm, e: Event) -> _Result:
    match e:
        case NeedEngine():
            loading = replace(s, life=Life.LOADING)
            return loading, [StartLoad()]
        case SetTimeout(seconds):
            return _with_timeout(s, seconds)
        case Loaded() | LoadFailed() | Idle() | Busy() | UnloadRequested() | Unloaded() | Wake():
            return _unchanged(s)
        case _:
            assert_never(e)


def _loading(s: KeepWarm, e: Event) -> _Result:
    match e:
        case Loaded() if s.unload_after_load:
            unloading = replace(s, life=Life.UNLOADING, unload_after_load=False)
            return unloading, [FlushPending(), StartUnload()]
        case Loaded():
            return replace(s, life=Life.WARM), [FlushPending()]
        case UnloadRequested():
            return replace(s, unload_after_load=True), []
        case LoadFailed():
            return replace(s, life=Life.COLD, unload_after_load=False), []
        case SetTimeout(seconds):
            return _with_timeout(s, seconds)
        case NeedEngine() | Idle() | Busy() | Unloaded() | Wake():
            return _unchanged(s)
        case _:
            assert_never(e)


def _warm_idle(s: KeepWarm, now: float) -> _Result:
    if s.timeout_s == 0:
        return replace(s, deadline=None, idle_since=now), []
    deadline = now + s.timeout_s
    return replace(s, deadline=deadline, idle_since=now), [ScheduleWake(deadline)]


def _warm_busy(s: KeepWarm) -> _Result:
    if s.deadline is None:
        return replace(s, idle_since=None), []
    return replace(s, deadline=None, idle_since=None), [CancelWake()]


def _warm_leave(s: KeepWarm) -> KeepWarm:
    return replace(s, life=Life.UNLOADING, deadline=None, idle_since=None)


def _warm_unload(s: KeepWarm) -> _Result:
    if s.deadline is None:
        return _warm_leave(s), [StartUnload()]
    return _warm_leave(s), [CancelWake(), StartUnload()]


def _warm_set_timeout(s: KeepWarm, seconds: int) -> _Result:
    timeout = max(0, seconds)
    if s.idle_since is None:
        return replace(s, timeout_s=timeout), []
    if timeout == 0:
        cancel: list[Action] = [] if s.deadline is None else [CancelWake()]
        return replace(s, timeout_s=0, deadline=None), cancel
    deadline = s.idle_since + timeout
    return replace(s, timeout_s=timeout, deadline=deadline), [ScheduleWake(deadline)]


def _warm(s: KeepWarm, e: Event) -> _Result:
    match e:
        case Idle(now):
            return _warm_idle(s, now)
        case Busy():
            return _warm_busy(s)
        case Wake(now) if s.deadline is not None and now >= s.deadline:
            return _warm_leave(s), [StartUnload()]
        case UnloadRequested():
            return _warm_unload(s)
        case SetTimeout(seconds):
            return _warm_set_timeout(s, seconds)
        case NeedEngine() | Loaded() | LoadFailed() | Unloaded() | Wake():
            return _unchanged(s)
        case _:
            assert_never(e)


def _unloading(s: KeepWarm, e: Event) -> _Result:
    match e:
        case Unloaded() if s.reload_after_unload:
            loading = replace(s, life=Life.LOADING, reload_after_unload=False)
            return loading, [StartLoad()]
        case Unloaded():
            return replace(s, life=Life.COLD, unload_after_load=False), []
        case NeedEngine():
            return replace(s, reload_after_unload=True), []
        case UnloadRequested() if s.reload_after_unload:
            return replace(s, unload_after_load=True), []
        case SetTimeout(seconds):
            return _with_timeout(s, seconds)
        case Loaded() | LoadFailed() | Idle() | Busy() | UnloadRequested() | Wake():
            return _unchanged(s)
        case _:
            assert_never(e)


def transition(s: KeepWarm, e: Event) -> tuple[KeepWarm, list[Action]]:
    """Return the next state and the actions to perform for event ``e`` in state ``s``."""
    match s.life:
        case Life.COLD:
            return _cold(s, e)
        case Life.LOADING:
            return _loading(s, e)
        case Life.WARM:
            return _warm(s, e)
        case Life.UNLOADING:
            return _unloading(s, e)
        case _:
            assert_never(s.life)
