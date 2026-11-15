"""Effects: what the Session asks the runtime to do, plus the Session's answers.

Effects carry plain data only (ids, chunks, voice refs); audio never passes through here. The
runtime (``runtime.loop``) and the test simulator (``testing.simulator``) execute them.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import DomainError
from voxlektor.domain.reading import Chunk
from voxlektor.domain.types import ChunkId, EngineId, Epoch


@dataclass(frozen=True, slots=True)
class LoadEngine:
    """Load the engine on the engine thread, then call ``Session.on_loaded``/``on_load_failed``."""

    engine_id: EngineId


@dataclass(frozen=True, slots=True)
class UnloadEngine:
    """Unload the engine (and drop its voice prompts), then call ``Session.on_unloaded``."""


@dataclass(frozen=True, slots=True)
class PrepareVoice:
    """Build or load the voice prompt, keep it by ``ref.name``, then ``on_prompt_ready``.

    ``fresh`` bypasses the prompt cache and rebuilds the prompt from the reference audio. On
    failure the runtime calls ``on_prompt_failed(ref.name, msg)``.
    """

    ref: VoiceRef
    fresh: bool = False


@dataclass(frozen=True, slots=True)
class Generate:
    """Synthesize ``chunk`` with the prompt of ``voice``.

    The runtime stores the generated audio by ``chunk.id`` on the loop thread immediately
    before calling ``on_generated`` (which may answer ``DropAudio`` for it right away).
    """

    chunk: Chunk
    voice: str


@dataclass(frozen=True, slots=True)
class CancelGeneration:
    """Skip the queued ``Generate`` jobs of ``epochs``; they report nothing.

    The Session names exactly the epochs it has abandoned. A synthesis that is already running
    is never interrupted; its result arrives and the Session drops it as stale.
    """

    epochs: tuple[Epoch, ...]


@dataclass(frozen=True, slots=True)
class StartPlayback:
    """Hand the buffered audio of ``chunk`` to the output (the buffer entry is consumed)."""

    chunk: ChunkId


@dataclass(frozen=True, slots=True)
class PausePlayback:
    pass


@dataclass(frozen=True, slots=True)
class ResumePlayback:
    pass


@dataclass(frozen=True, slots=True)
class AbortPlayback:
    """Stop the chunk that is playing; its late callbacks are ignored by the Session."""


@dataclass(frozen=True, slots=True)
class DropAudio:
    """Forget the buffered audio of ``chunks`` (stale or no longer needed).

    The audio was stored on the loop thread just before ``on_generated``, so a ``DropAudio``
    returned by that call always finds it.
    """

    chunks: tuple[ChunkId, ...]


@dataclass(frozen=True, slots=True)
class ScheduleWake:
    """Call ``Session.on_wake`` at clock time ``at`` (immediately when it is in the past)."""

    at: float


@dataclass(frozen=True, slots=True)
class CancelWake:
    """Cancel the pending wake(s); a stale wake that still fires is harmless."""


@dataclass(frozen=True, slots=True)
class Exit:
    """Stop the service once the effects before it have been carried out."""


class Cue(StrEnum):
    """Short sounds that tell the listener what happened without a screen."""

    RETRY = "retry"  # a failure is being retried
    STOPPED = "stopped"  # the reading stopped because of an error


@dataclass(frozen=True, slots=True)
class PlayCue:
    """Play ``cue`` at ``volume`` (0..100, 50 = the even level; the reading it belongs to)."""

    cue: Cue
    volume: int = 50


@dataclass(frozen=True, slots=True)
class Emit:
    """Publish the server event ``event`` with ``data`` to the subscribed clients.

    Dropped when nothing can publish (a runtime without an event hub).
    """

    event: str
    data: Mapping[str, object]


type Effect = (
    LoadEngine
    | UnloadEngine
    | PrepareVoice
    | Generate
    | CancelGeneration
    | StartPlayback
    | PausePlayback
    | ResumePlayback
    | AbortPlayback
    | DropAudio
    | ScheduleWake
    | CancelWake
    | Exit
    | PlayCue
    | Emit
)


@dataclass(frozen=True)
class Reply:
    """The answer to one command; ``error`` is set exactly when ``ok`` is false."""

    ok: bool
    data: Mapping[str, object]
    error: DomainError | None


@dataclass
class Outcome:
    reply: Reply | None
    effects: list[Effect]
