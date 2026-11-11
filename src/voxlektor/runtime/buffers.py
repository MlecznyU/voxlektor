"""AudioBuffers: generated audio by chunk id, kept outside the Session (numpy stays out of it).

Used only on the loop thread. The bound is a runtime guard for the ahead window (Review Focus 3):
with ``ahead = 2`` at most two chunks are buffered, plus one stale result that arrives and is
dropped right away; going past ``max_items`` means a bug, so ``put`` raises.
"""

from collections.abc import Iterable

from voxlektor.contracts.engine import AudioChunk
from voxlektor.domain.types import ChunkId


class AudioBuffers:
    def __init__(self, max_items: int = 8) -> None:
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        self._max_items = max_items
        self._items: dict[ChunkId, AudioChunk] = {}

    def put(self, chunk_id: ChunkId, audio: AudioChunk) -> None:
        if chunk_id in self._items:
            raise RuntimeError(f"chunk {chunk_id} is already buffered")
        if len(self._items) >= self._max_items:
            raise RuntimeError(f"audio buffers would hold more than {self._max_items} chunks")
        self._items[chunk_id] = audio

    def take(self, chunk_id: ChunkId) -> AudioChunk:
        """Remove and return the audio of ``chunk_id``; ``KeyError`` when it is not buffered."""
        return self._items.pop(chunk_id)

    def drop(self, ids: Iterable[ChunkId]) -> None:
        """Forget ``ids``; ids that are not buffered are ignored."""
        for chunk_id in ids:
            self._items.pop(chunk_id, None)

    def __len__(self) -> int:
        return len(self._items)
