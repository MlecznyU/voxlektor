"""Application commands: what a client can ask the Session to do.

Independent of the wire protocol version; ``adapters.transport.mapping`` translates between them.
"""

from dataclasses import dataclass

from voxlektor.domain.types import LangPref


@dataclass(frozen=True, slots=True)
class Speak:
    """Read ``text``; ``None`` fields fall back to the settings; ``now`` replaces the reading."""

    text: str
    voice: str | None
    lang: LangPref | None
    rate: float | None
    now: bool


@dataclass(frozen=True, slots=True)
class Pause:
    pass


@dataclass(frozen=True, slots=True)
class Resume:
    pass


@dataclass(frozen=True, slots=True)
class Skip:
    pass


@dataclass(frozen=True, slots=True)
class Back:
    pass


@dataclass(frozen=True, slots=True)
class StopReading:
    pass


@dataclass(frozen=True, slots=True)
class Status:
    pass


@dataclass(frozen=True, slots=True)
class SetIdle:
    """Change the idle timeout live; ``timeout`` is a duration (``"0"``, ``"90s"``, ``"10m"``)."""

    timeout: str


@dataclass(frozen=True, slots=True)
class Unload:
    pass


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass


type Command = (
    Speak | Pause | Resume | Skip | Back | StopReading | Status | SetIdle | Unload | Shutdown
)
