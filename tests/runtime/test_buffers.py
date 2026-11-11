import numpy as np
import pytest

from voxlektor.contracts.engine import AudioChunk
from voxlektor.domain.types import ChunkId
from voxlektor.runtime.buffers import AudioBuffers


def _audio(n: int = 4) -> AudioChunk:
    return AudioChunk(np.zeros(n, dtype=np.float32), 24000)


def test_put_take_round_trip() -> None:
    buffers = AudioBuffers()
    audio = _audio()
    buffers.put(ChunkId(1), audio)
    assert len(buffers) == 1
    assert buffers.take(ChunkId(1)) is audio
    assert len(buffers) == 0


def test_take_of_a_missing_chunk_raises() -> None:
    with pytest.raises(KeyError):
        AudioBuffers().take(ChunkId(7))


def test_drop_forgets_present_ids_and_ignores_missing_ones() -> None:
    buffers = AudioBuffers()
    buffers.put(ChunkId(1), _audio())
    buffers.put(ChunkId(2), _audio())
    buffers.drop((ChunkId(1), ChunkId(9)))
    assert len(buffers) == 1
    assert buffers.take(ChunkId(2)).samples.shape == (4,)


def test_exceeding_max_items_raises() -> None:
    buffers = AudioBuffers(max_items=2)
    buffers.put(ChunkId(1), _audio())
    buffers.put(ChunkId(2), _audio())
    with pytest.raises(RuntimeError, match="more than 2"):
        buffers.put(ChunkId(3), _audio())
    assert len(buffers) == 2


def test_a_chunk_id_is_stored_once() -> None:
    buffers = AudioBuffers()
    buffers.put(ChunkId(1), _audio())
    with pytest.raises(RuntimeError, match="already buffered"):
        buffers.put(ChunkId(1), _audio())


def test_max_items_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_items"):
        AudioBuffers(max_items=0)
