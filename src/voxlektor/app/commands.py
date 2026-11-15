"""Application commands: what a client can ask the Session to do.

Independent of the wire protocol version; ``adapters.transport.mapping`` translates between them.
"""

from dataclasses import dataclass, field

from voxlektor.domain.types import Lang, LangPref


@dataclass(frozen=True, slots=True)
class Speak:
    """Read ``text``; ``None`` fields fall back to the settings; ``now`` replaces the reading.

    ``base_lang`` is the language ``auto`` detection falls back to on a tie (owner-approved P1);
    it is keyword-only so that the positional order of the other fields stays unchanged.
    ``volume`` (0..100, 50 = the even level; Etap 1.1) is keyword-only for the same reason.
    """

    text: str
    voice: str | None
    lang: LangPref | None
    base_lang: Lang | None = field(default=None, kw_only=True)
    volume: int | None = field(default=None, kw_only=True)
    rate: float | None
    now: bool
    owner: str | None = field(default=None, kw_only=True)
    group: str | None = field(default=None, kw_only=True)
    tag: str | None = field(default=None, kw_only=True)
    preempt: bool = field(default=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class Pause:
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class Resume:
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class Skip:
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class Back:
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class StopReading:
    owner: str | None = None


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
    """Stop the daemon; with ``if_idle`` refused (``busy``) while any owner has a reading."""

    if_idle: bool = False


@dataclass(frozen=True, slots=True)
class Load:
    """Warm the engine without reading (daemon-wide, owner-less); then the usual idle timeout."""


type Command = (
    Speak | Pause | Resume | Skip | Back | StopReading | Status | SetIdle | Unload | Shutdown | Load
)
