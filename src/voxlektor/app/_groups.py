"""Private building blocks of ``Session``: per-group reading state and per-chunk bookkeeping.

Plain data only; every instance is owned and mutated by the one ``Session`` that made it.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from voxlektor.domain.floor import GroupKey
from voxlektor.domain.reading import Chunk, UnderrunTracker
from voxlektor.domain.types import ChunkId, Epoch

CLI = "cli"  # the reserved owner of requests without an owner


class Gen(Enum):
    IN_FLIGHT = auto()
    READY = auto()
    DISCARDED = auto()  # skipped while in flight; forgotten when its result arrives
    RETRY = auto()  # failed; Generate again once WARM and the prompt is ready


@dataclass(frozen=True)
class Item:
    chunk: Chunk
    voice: str
    group: GroupKey
    tag: str | None = None
    counted: bool = False  # already counted in the reading metrics
    speak: int | None = None  # the tag run (accepted tagged speak) this chunk belongs to


@dataclass
class Group:
    """One group's reading state: its epoch, its age and its unplayed chunks in order."""

    epoch: Epoch
    age: float
    chunks: deque[ChunkId] = field(default_factory=deque[ChunkId])  # [0] is the head


@dataclass
class Metrics:
    """Numbers about the latest reading (reset when a reading starts from idle)."""

    started_at: float
    first_audio_s: float | None = None
    chunks: int = 0
    gen_seconds: float = 0.0
    audio_seconds: float = 0.0
    underruns: UnderrunTracker = field(default_factory=UnderrunTracker)


def history_add(history: deque[ChunkId], chunk: ChunkId, cap: int) -> list[ChunkId]:
    """Append ``chunk``; return the oldest entries evicted beyond ``cap``."""
    history.append(chunk)
    evicted: list[ChunkId] = []
    while len(history) > cap:
        evicted.append(history.popleft())
    return evicted
