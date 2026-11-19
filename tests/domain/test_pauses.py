from typing import Literal

import pytest

from voxlektor.domain import pauses
from voxlektor.domain.pauses import Boundary, pause_before
from voxlektor.domain.reading import Chunk
from voxlektor.domain.types import ChunkId, Epoch

ANNA_A = ("anna", "a")
ANNA_B = ("anna", "b")
BOB_A = ("bob", "a")


def _chunk(ends: Literal["sentence", "paragraph"]) -> Chunk:
    return Chunk(ChunkId(1), "text", "en", 1.0, Epoch(0), ends=ends)


def test_pause_constants() -> None:
    assert pauses.SENTENCE == 0.25
    assert pauses.PARAGRAPH == 0.5
    assert pauses.GROUP == 0.8
    assert pauses.INTERRUPT == 0.25


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        # first chunk after idle, whatever else is true
        (Boundary(None, None, ANNA_A), 0.0),
        (Boundary(None, None, ANNA_A, interrupted=True), 0.0),
        # sentence end inside one paragraph of the same group
        (Boundary(_chunk("sentence"), ANNA_A, ANNA_A), 0.25),
        # text paragraph end or end of a speak in the same group
        (Boundary(_chunk("paragraph"), ANNA_A, ANNA_A), 0.5),
        # end of a group of the same owner, another owner takes the floor
        (Boundary(_chunk("paragraph"), ANNA_A, ANNA_B), 0.8),
        (Boundary(_chunk("sentence"), ANNA_A, ANNA_B), 0.8),
        (Boundary(_chunk("paragraph"), ANNA_A, BOB_A), 0.8),
        (Boundary(_chunk("paragraph"), ANNA_A, BOB_A, auto_groups=True), 0.8),
        (Boundary(_chunk("paragraph"), None, ANNA_A), 0.8),
        # after skip, back, preempt or a yield restart
        (Boundary(_chunk("sentence"), ANNA_A, ANNA_A, interrupted=True), 0.25),
        (Boundary(_chunk("paragraph"), ANNA_A, ANNA_A, interrupted=True), 0.25),
        (Boundary(_chunk("paragraph"), ANNA_A, BOB_A, interrupted=True), 0.25),
    ],
)
def test_pause_table(boundary: Boundary, expected: float) -> None:
    assert pause_before(boundary) == expected


def test_auto_groups_same_owner_is_speak_end() -> None:
    boundary = Boundary(_chunk("paragraph"), ANNA_A, ANNA_B, auto_groups=True)
    assert pause_before(boundary) == pauses.PARAGRAPH
    explicit = Boundary(_chunk("paragraph"), ANNA_A, ANNA_B, auto_groups=False)
    assert pause_before(explicit) == pauses.GROUP


def test_boundary_defaults() -> None:
    boundary = Boundary(None, None, ANNA_A)
    assert boundary.auto_groups is False
    assert boundary.interrupted is False
