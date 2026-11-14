"""Device-free tests of SoundDeviceOutput: sounddevice.OutputStream is replaced by a fake."""

from collections.abc import Callable, Iterator
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pytest
import sounddevice as sd

from tests.adapters.audio.conftest import Reinits
from voxlektor.adapters.audio import sounddevice_out
from voxlektor.adapters.audio.sounddevice_out import SoundDeviceOutput
from voxlektor.domain.types import ChunkId
from voxlektor.testing.fakes import FakeClock

type _Callback = Callable[[npt.NDArray[np.float32], int, object, sd.CallbackFlags], None]


class FakeStream:
    instances: ClassVar[list["FakeStream"]] = []
    fail_start: ClassVar[bool] = False

    def __init__(
        self,
        *,
        samplerate: float,
        callback: _Callback,
        finished_callback: Callable[[], None],
        **ignored: object,  # channels, dtype, latency
    ) -> None:
        self.samplerate = samplerate
        self.callback = callback
        self.finished_callback = finished_callback
        self.started = False
        self.aborted = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self) -> None:
        if FakeStream.fail_start:
            raise sd.PortAudioError("no device")
        self.started = True

    def abort(self, ignore_errors: bool = True) -> None:
        self.aborted = True

    def close(self, ignore_errors: bool = True) -> None:
        assert not self.closed, "stream closed twice"
        self.closed = True

    def pull(self, frames: int, fill: float = 7.0) -> npt.NDArray[np.float32]:
        outdata = np.full((frames, 1), fill, dtype=np.float32)
        self.callback(outdata, frames, None, sd.CallbackFlags())
        return outdata


@pytest.fixture(autouse=True)
def fake_stream(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    FakeStream.instances = []
    FakeStream.fail_start = False
    monkeypatch.setattr(sd, "OutputStream", FakeStream)
    yield
    FakeStream.instances = []
    FakeStream.fail_start = False


class _Events:
    def __init__(self) -> None:
        self.done: list[ChunkId] = []
        self.failed: list[tuple[ChunkId, str]] = []


def _output() -> tuple[SoundDeviceOutput, _Events]:
    out = SoundDeviceOutput()
    events = _Events()
    out.set_callbacks(events.done.append, lambda c, m: events.failed.append((c, m)))
    return out, events


def _samples(n: int) -> npt.NDArray[np.float32]:
    return np.arange(1, n + 1, dtype=np.float32)


def test_callback_copies_frames_and_zero_pads_last_block() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(6), 24000)
    stream = FakeStream.instances[0]
    assert stream.started
    assert stream.samplerate == 24000
    assert stream.pull(4)[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    last = np.full((4, 1), 7.0, dtype=np.float32)
    with pytest.raises(sd.CallbackStop):
        stream.callback(last, 4, None, sd.CallbackFlags())
    assert last[:, 0].tolist() == [5.0, 6.0, 0.0, 0.0]
    stream.finished_callback()
    assert events.done == [ChunkId(1)]
    assert events.failed == []


def test_exact_block_boundary_then_stop_on_next_block() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(4), 24000)
    stream = FakeStream.instances[0]
    assert stream.pull(4)[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(sd.CallbackStop):
        stream.pull(4)
    stream.finished_callback()
    assert events.done == [ChunkId(1)]


def test_silence_while_paused_and_resume_continues() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(4), 24000)
    stream = FakeStream.instances[0]
    out.pause()
    assert stream.pull(4)[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    out.resume()
    assert stream.pull(2)[:, 0].tolist() == [1.0, 2.0]
    assert events.done == []


def test_pause_persists_across_abort_and_start() -> None:
    out, _ = _output()
    out.pause()
    out.abort()
    out.start(ChunkId(1), _samples(4), 24000)
    stream = FakeStream.instances[0]
    assert stream.pull(2)[:, 0].tolist() == [0.0, 0.0]
    out.resume()
    assert stream.pull(2)[:, 0].tolist() == [1.0, 2.0]


def test_finished_twice_reports_once() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(2), 24000)
    stream = FakeStream.instances[0]
    with pytest.raises(sd.CallbackStop):
        stream.pull(4)
    stream.finished_callback()
    stream.finished_callback()
    assert events.done == [ChunkId(1)]


def test_early_stop_is_reported_as_failure() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(10), 24000)
    stream = FakeStream.instances[0]
    stream.pull(4)
    stream.finished_callback()
    assert events.done == []
    assert events.failed == [(ChunkId(1), "stream stopped before end")]


def test_callback_error_aborts_and_reports_failure() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(4), 24000)
    stream = FakeStream.instances[0]
    bad = np.zeros(4, dtype=np.float32)  # wrong shape: the callback expects (frames, 1)
    with pytest.raises(sd.CallbackAbort):
        stream.callback(bad, 4, None, sd.CallbackFlags())
    stream.finished_callback()
    assert events.done == []
    assert len(events.failed) == 1
    assert events.failed[0][0] == ChunkId(1)
    assert events.failed[0][1].startswith("IndexError")


def test_cancelled_playback_reports_nothing() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(4), 24000)
    stream = FakeStream.instances[0]
    out.abort()
    with pytest.raises(sd.CallbackAbort):
        stream.pull(4)
    stream.finished_callback()
    assert events.done == []
    assert events.failed == []
    assert stream.aborted


def test_start_failure_reports_and_closes_stream() -> None:
    out, events = _output()
    FakeStream.fail_start = True
    out.start(ChunkId(3), _samples(4), 24000)
    assert events.failed == [(ChunkId(3), "no device")]
    assert FakeStream.instances[0].closed
    out.abort()  # nothing left to close; must not close twice


def test_start_failure_without_callbacks_is_silent() -> None:
    out = SoundDeviceOutput()
    FakeStream.fail_start = True
    out.start(ChunkId(3), _samples(4), 24000)
    assert FakeStream.instances[0].closed


def test_finished_without_callbacks_is_silent() -> None:
    out = SoundDeviceOutput()
    out.start(ChunkId(1), _samples(2), 24000)
    stream = FakeStream.instances[0]
    with pytest.raises(sd.CallbackStop):
        stream.pull(4)
    stream.finished_callback()


def test_callbacks_can_be_set_only_once() -> None:
    out, _ = _output()
    with pytest.raises(RuntimeError):
        out.set_callbacks(lambda _c: None, lambda _c, _m: None)


def test_every_stream_is_closed_after_natural_end_and_next_start() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(2), 24000)
    first = FakeStream.instances[0]
    with pytest.raises(sd.CallbackStop):
        first.pull(4)
    first.finished_callback()
    assert events.done == [ChunkId(1)]
    assert not first.closed  # kept open: PortAudio forbids closing from its own callback
    out.start(ChunkId(2), _samples(2), 24000)
    assert first.closed
    out.abort()
    assert [s.closed for s in FakeStream.instances] == [True, True]


def test_back_to_back_start_closes_the_previous_stream() -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(8), 24000)
    out.start(ChunkId(2), _samples(8), 24000)
    first, second = FakeStream.instances
    assert first.aborted
    assert first.closed
    assert not second.closed
    first.finished_callback()  # late finished of the replaced stream: ignored
    assert events.done == []
    assert events.failed == []
    out.abort()
    assert second.closed


def test_start_after_abort_opens_a_fresh_stream() -> None:
    out, _ = _output()
    out.start(ChunkId(1), _samples(8), 24000)
    out.abort()
    out.abort()  # idempotent
    out.start(ChunkId(2), _samples(8), 24000)
    first, second = FakeStream.instances
    assert first.closed
    assert second.started
    assert not second.closed
    out.abort()


# ---------------------------------------------------------------- following the default device


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, reinits: Reinits) -> FakeClock:
    """A fake clock for the idle check; every re-initialisation asserts that no stream is open."""
    fake = FakeClock(100.0)
    monkeypatch.setattr(sounddevice_out, "_monotonic", fake.now)

    def no_stream_open() -> None:
        still_open = [s for s in FakeStream.instances if not s.closed]
        assert still_open == [], "PortAudio re-initialised while a stream was open"

    reinits.check = no_stream_open
    return fake


def _play_to_end(out: SoundDeviceOutput, chunk: int) -> FakeStream:
    out.start(ChunkId(chunk), _samples(2), 24000)
    stream = FakeStream.instances[-1]
    with pytest.raises(sd.CallbackStop):
        stream.pull(4)
    stream.finished_callback()
    return stream


def test_a_fresh_output_does_not_re_initialise(clock: FakeClock, reinits: Reinits) -> None:
    out, _ = _output()
    out.start(ChunkId(1), _samples(2), 24000)
    assert reinits.count == 0


def test_a_reading_that_starts_from_idle_re_initialises_first(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    first = _play_to_end(out, 1)
    assert not first.closed  # the finished stream is kept open until the next start
    clock.advance(2.0)  # the output's idle threshold (_IDLE_S)
    out.start(ChunkId(2), _samples(2), 24000)
    assert reinits.count == 1  # and, by the clock fixture, with no stream open
    assert first.closed
    assert events.done == [ChunkId(1)]
    assert FakeStream.instances[-1].started


def test_chunks_of_one_reading_never_re_initialise(clock: FakeClock, reinits: Reinits) -> None:
    out, _ = _output()
    for chunk in range(5):
        _play_to_end(out, chunk)
        clock.advance(0.05)
    assert reinits.count == 0


def test_a_long_pause_while_playing_is_not_idle(clock: FakeClock, reinits: Reinits) -> None:
    out, _ = _output()
    out.start(ChunkId(1), _samples(8), 24000)
    out.pause()
    clock.advance(60.0)
    out.start(ChunkId(2), _samples(8), 24000)  # replaces a chunk that was playing (paused)
    assert reinits.count == 0


def test_a_playback_failure_re_initialises_at_the_next_abort(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(10), 24000)
    stream = FakeStream.instances[0]
    stream.pull(4)
    stream.finished_callback()  # stopped early: the device went away
    assert events.failed == [(ChunkId(1), "stream stopped before end")]
    assert reinits.count == 0  # never from the PortAudio thread
    out.abort()
    assert reinits.count == 1
    out.start(ChunkId(2), _samples(2), 24000)
    assert reinits.count == 1  # done once


def test_a_failed_start_re_initialises_before_the_next_start(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    FakeStream.fail_start = True
    out.start(ChunkId(1), _samples(4), 24000)
    assert events.failed == [(ChunkId(1), "no device")]
    FakeStream.fail_start = False
    out.start(ChunkId(1), _samples(4), 24000)  # the retry, at once
    assert reinits.count == 1
    assert FakeStream.instances[-1].started


def test_a_failed_refresh_in_start_is_a_playback_failure_and_the_next_start_recovers(
    clock: FakeClock, reinits: Reinits
) -> None:
    """R1: no exception escapes start(); the output stays quiet (not "playing") and retries."""
    out, events = _output()
    clock.advance(10.0)
    reinits.fail_initialize = 1
    out.start(ChunkId(1), _samples(2), 24000)  # must not raise
    assert events.failed == [(ChunkId(1), "Error initializing PortAudio")]
    assert FakeStream.instances == []  # nothing opened on an uninitialised library
    out.start(ChunkId(2), _samples(2), 24000)  # the next speak: initialise only, then play
    assert (reinits.terminates, reinits.initializes, reinits.count) == (1, 2, 1)
    assert FakeStream.instances[-1].started
    out.abort()


def test_a_failed_refresh_in_abort_is_retried_by_the_next_start(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    out.start(ChunkId(1), _samples(10), 24000)
    stream = FakeStream.instances[0]
    stream.pull(4)
    stream.finished_callback()  # failed: a refresh is due
    reinits.fail_terminate = 1
    out.abort()  # must not raise
    assert reinits.count == 0
    out.start(ChunkId(2), _samples(2), 24000)
    assert reinits.count == 1
    assert len(events.failed) == 1
    out.abort()


def test_another_output_still_playing_postpones_the_re_initialisation(
    clock: FakeClock, reinits: Reinits
) -> None:
    playing, _ = _output()
    playing.start(ChunkId(1), _samples(8), 24000)
    out, _ = _output()
    clock.advance(10.0)
    reinits.check = None  # the other output's stream is open on purpose
    out.start(ChunkId(2), _samples(2), 24000)
    assert reinits.count == 0
    playing.abort()


def test_another_output_releases_its_finished_stream_for_the_re_initialisation(
    clock: FakeClock, reinits: Reinits
) -> None:
    other, _ = _output()
    kept = _play_to_end(other, 1)
    out, _ = _output()
    clock.advance(10.0)
    out.start(ChunkId(2), _samples(2), 24000)
    assert reinits.count == 1
    assert kept.closed
    other.abort()  # its stream is gone already: nothing is closed twice
