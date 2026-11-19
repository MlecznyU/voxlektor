"""Pause policy: the silence before a chunk starts, from the structure of the text and queue.

Engine edge silence is trimmed, so every pause between chunks comes from here. Pauses the
model reads inside one chunk are its prosody and are not touched.
"""

from dataclasses import dataclass

from voxlektor.domain.reading import Chunk

SENTENCE = 0.25  # the next chunk goes on in the same paragraph
PARAGRAPH = 0.5  # a paragraph of the text ended, or a speak ended (same group)
GROUP = 0.8  # a group ended, or another owner or group takes the floor
INTERRUPT = 0.25  # after skip, back, preempt or a yield restart: a short breath


@dataclass(frozen=True)
class Boundary:
    """What lies between the chunk that played last and the one about to start.

    ``prev`` is the chunk that actually played last (``None`` after idle); groups are
    ``(owner, group)``. ``auto_groups`` is true when both groups are the automatic groups of
    their owner (plain CLI use), so a change between them of the same owner is only the end
    of a speak. ``interrupted`` is true when the person moved the reading (skip, back) or
    another reading took over (preempt, yield restart).
    """

    prev: Chunk | None
    prev_group: tuple[str, str] | None
    next_group: tuple[str, str]
    auto_groups: bool = False
    interrupted: bool = False


def pause_before(b: Boundary) -> float:
    """Seconds of silence before the next chunk (a minimum gap; elapsed silence counts)."""
    if b.prev is None:
        return 0.0
    if b.interrupted:
        return INTERRUPT
    if b.prev_group != b.next_group:
        same_owner = b.prev_group is not None and b.prev_group[0] == b.next_group[0]
        return PARAGRAPH if b.auto_groups and same_owner else GROUP
    return PARAGRAPH if b.prev.ends == "paragraph" else SENTENCE
