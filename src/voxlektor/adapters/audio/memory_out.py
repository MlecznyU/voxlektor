"""Device-less audio adapters: an AudioOutput and a CuePlayer that only record.

``MemoryOutput`` completes chunks on demand or immediately. Production adapters (selected by
``VOXLEKTOR_OUTPUT=memory``); also used by tests. This module never imports an audio library.
"""

from collections.abc import Callable

from voxlektor.app.effects import Cue
from voxlektor.domain.types import ChunkId


class MemoryCuePlayer:
    """Records the cues it is asked to play (``played``) and their volumes (``calls``)."""

    def __init__(self) -> None:
        self.played: list[Cue] = []
        self.calls: list[tuple[Cue, int]] = []

    def play(self, cue: Cue, volume: int = 50) -> None:
        self.played.append(cue)
        self.calls.append((cue, volume))


class MemoryOutput:
    def __init__(self, auto_complete: bool = True) -> None:
        self._auto_complete = auto_complete
        self._on_done: Callable[[ChunkId], None] | None = None
        self._on_failed: Callable[[ChunkId, str], None] | None = None
        self.started: list[ChunkId] = []
        self.aborted = 0
        self.paused = False

    def set_callbacks(
        self,
        on_done: Callable[[ChunkId], None],
        on_failed: Callable[[ChunkId, str], None],
    ) -> None:
        if self._on_done is not None:
            raise RuntimeError("callbacks already set")
        self._on_done = on_done
        self._on_failed = on_failed

    def start(self, chunk: ChunkId, samples: object, sample_rate: int) -> None:
        del samples, sample_rate
        self.started.append(chunk)
        if self._auto_complete:
            self.complete(chunk)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def abort(self) -> None:
        self.aborted += 1

    def complete(self, chunk: ChunkId) -> None:
        """Report ``chunk`` as played to the end."""
        if self._on_done is None:
            raise RuntimeError("callbacks not set")
        self._on_done(chunk)

    def fail(self, chunk: ChunkId, msg: str) -> None:
        """Report that playing ``chunk`` failed."""
        if self._on_failed is None:
            raise RuntimeError("callbacks not set")
        self._on_failed(chunk, msg)
