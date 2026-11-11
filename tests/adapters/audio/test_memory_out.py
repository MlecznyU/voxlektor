import pytest

from voxlektor.adapters.audio.memory_out import MemoryOutput
from voxlektor.domain.types import ChunkId
from voxlektor.testing.fakes import FakeClock
from voxlektor.testing.fakes import MemoryOutput as ReExported

type _Wired = tuple[MemoryOutput, list[ChunkId], list[tuple[ChunkId, str]]]


def _wired(auto_complete: bool = True) -> _Wired:
    out = MemoryOutput(auto_complete=auto_complete)
    done: list[ChunkId] = []
    failed: list[tuple[ChunkId, str]] = []
    out.set_callbacks(done.append, lambda chunk, msg: failed.append((chunk, msg)))
    return out, done, failed


def test_fakes_reexports_the_production_adapter() -> None:
    assert ReExported is MemoryOutput


def test_auto_complete_calls_on_done_synchronously() -> None:
    out, done, _ = _wired()
    out.start(ChunkId(1), object(), 24000)
    assert out.started == [ChunkId(1)]
    assert done == [ChunkId(1)]


def test_manual_complete_and_fail() -> None:
    out, done, failed = _wired(auto_complete=False)
    out.start(ChunkId(1), object(), 24000)
    out.start(ChunkId(2), object(), 24000)
    assert done == []
    out.complete(ChunkId(1))
    out.fail(ChunkId(2), "device lost")
    assert done == [ChunkId(1)]
    assert failed == [(ChunkId(2), "device lost")]


def test_pause_resume_abort_are_recorded() -> None:
    out, _, _ = _wired(auto_complete=False)
    assert not out.paused
    out.pause()
    assert out.paused
    out.resume()
    assert not out.paused
    assert out.aborted == 0
    out.abort()
    assert out.aborted == 1


def test_start_before_callbacks_raises() -> None:
    out = MemoryOutput()
    with pytest.raises(RuntimeError):
        out.start(ChunkId(1), object(), 24000)


def test_callbacks_can_be_set_only_once() -> None:
    out, _, _ = _wired()
    with pytest.raises(RuntimeError):
        out.set_callbacks(lambda _chunk: None, lambda _chunk, _msg: None)


def test_pause_survives_abort() -> None:
    out, _, _ = _wired(auto_complete=False)
    out.pause()
    out.abort()
    assert out.paused


def test_complete_before_callbacks_raises() -> None:
    out = MemoryOutput(auto_complete=False)
    with pytest.raises(RuntimeError):
        out.complete(ChunkId(1))
    with pytest.raises(RuntimeError):
        out.fail(ChunkId(1), "x")


def test_fake_clock_advances() -> None:
    clock = FakeClock()
    assert clock.now() == 0.0
    clock.advance(2.5)
    assert clock.now() == 2.5
    assert FakeClock(10.0).now() == 10.0
