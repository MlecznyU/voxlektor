"""Device-free tests of SoundDeviceOutput: the stream factory is replaced by a fake stream.

The fake runs at ``RATE`` = 10 Hz, so one frame is 0.1 s: a lead of 0.4 s is four silent frames
and the idle close (2 s with nothing queued) comes after twenty.
"""

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

RATE = 10
C1, C2, C3 = ChunkId(1), ChunkId(2), ChunkId(3)


class FakeStream:
    """Stands in for ``sd.OutputStream``; ``pull`` plays the part of the PortAudio thread.

    It also enforces CoreAudio's deadlock rule (seen in the field hang): a stream must never be
    stopped from outside while its own callback is stopping it. A callback that raises
    ``CallbackStop``/``CallbackAbort`` makes PortAudio call ``AudioOutputUnitStop`` on the IO
    thread, which holds the IO-context mutex and waits for the unit's mutex; ``stop``/``abort``/
    ``close`` from another thread hold the unit's mutex and wait for the IO-context mutex. So:
    after the callback raised, ``stop``/``abort``/``close`` before ``finish()`` is a deadlock, and
    ``stop``/``abort``/``close`` of a running stream first run one IO cycle "at the same moment"
    (``pull(1)``), which must not raise either.
    """

    instances: ClassVar[list["FakeStream"]] = []
    fail_open: ClassVar[bool] = False
    fail_start: ClassVar[bool] = False

    def __init__(
        self, sample_rate: int, callback: _Callback, finished_callback: Callable[[], None]
    ) -> None:
        if FakeStream.fail_open:
            raise sd.PortAudioError("cannot open")
        self.sample_rate = sample_rate
        self.callback = callback
        self.finished_callback = finished_callback
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.stopping_itself = False  # the callback raised; CoreAudio is still stopping it
        self._finished = False
        FakeStream.instances.append(self)

    def start(self) -> None:
        if FakeStream.fail_start:
            raise sd.PortAudioError("no device")
        self.started = True

    def stop(self, ignore_errors: bool = True) -> None:
        self._from_outside("stop")
        self.stopped = True
        self.finish()

    def abort(self, ignore_errors: bool = True) -> None:
        self._from_outside("abort")
        self.aborted = True
        self.finish()

    def close(self, ignore_errors: bool = True) -> None:
        assert not self.closed, "stream closed twice"
        self._from_outside("close")
        self.closed = True
        self.finish()

    def finish(self) -> None:
        """PortAudio's finished callback: once, after the stream stopped for any reason."""
        self.stopping_itself = False
        if self.started and not self._finished:
            self._finished = True
            self.finished_callback()

    def pull(self, frames: int, outdata: npt.NDArray[np.float32] | None = None) -> list[float]:
        if outdata is None:
            outdata = np.full((frames, 1), 7.0, dtype=np.float32)
        try:
            self.callback(outdata, frames, None, sd.CallbackFlags())
        except (sd.CallbackStop, sd.CallbackAbort):
            self.stopping_itself = True
            raise
        return [float(x) for x in outdata.reshape(-1)]

    def _from_outside(self, what: str) -> None:
        assert not self.stopping_itself, f"deadlock: {what} while the callback stops the stream"
        if self.started and not self._finished:
            try:
                self.pull(1)  # an IO cycle in flight at this very moment
            except (sd.CallbackStop, sd.CallbackAbort):
                raise AssertionError(f"deadlock: {what} raced the callback stopping it") from None


@pytest.fixture(autouse=True)
def fake_stream(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    FakeStream.instances = []
    FakeStream.fail_open = False
    FakeStream.fail_start = False
    monkeypatch.setattr(sounddevice_out, "_open_stream", FakeStream)
    yield
    FakeStream.instances = []


class _Events:
    def __init__(self) -> None:
        self.done: list[ChunkId] = []
        self.failed: list[tuple[ChunkId, str]] = []


def _output() -> tuple[SoundDeviceOutput, _Events]:
    out = SoundDeviceOutput()
    events = _Events()
    out.set_callbacks(events.done.append, lambda c, m: events.failed.append((c, m)))
    return out, events


def _samples(n: int, first: int = 1) -> npt.NDArray[np.float32]:
    return np.arange(first, first + n, dtype=np.float32)


def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _stream() -> FakeStream:
    return FakeStream.instances[-1]


# ---------------------------------------------------------------- one stream, lead gaps


def test_first_chunk_plays_at_once_and_is_done_with_its_last_sample() -> None:
    out, events = _output()
    out.start(C1, _samples(6), RATE, lead_s=0.8)  # nothing played before: the gap is long over
    stream = _stream()
    assert stream.started
    assert stream.sample_rate == RATE
    assert stream.pull(4) == [1.0, 2.0, 3.0, 4.0]
    assert events.done == []
    assert stream.pull(4) == [5.0, 6.0, 0.0, 0.0]
    assert events.done == [C1]
    assert not stream.stopped  # the stream stays open after the chunk
    assert events.failed == []


def test_lead_silence_then_samples_on_the_same_stream() -> None:
    out, events = _output()
    out.start(C1, _samples(2), RATE)
    stream = _stream()
    assert stream.pull(2) == [1.0, 2.0]
    out.start(C2, _samples(3, first=11), RATE, lead_s=0.4)
    assert stream.pull(8) == [*_zeros(4), 11.0, 12.0, 13.0, 0.0]
    assert events.done == [C1, C2]
    assert FakeStream.instances == [stream]


def test_lead_is_minimum_gap_after_stall() -> None:
    out, events = _output()
    out.start(C1, _samples(2), RATE)
    stream = _stream()
    stream.pull(2)
    assert stream.pull(3) == _zeros(3)  # a generation stall: 0.3 s already silent
    out.start(C2, _samples(2, first=11), RATE, lead_s=0.5)
    assert stream.pull(4) == [0.0, 0.0, 11.0, 12.0]  # only the missing 0.2 s
    assert stream.pull(6) == _zeros(6)  # a stall longer than the next lead
    out.start(C3, _samples(2, first=21), RATE, lead_s=0.25)
    assert stream.pull(2) == [21.0, 22.0]
    assert events.done == [C1, C2, C3]


def test_queued_chunks_play_in_order_with_their_leads_within_one_block() -> None:
    out, events = _output()
    out.start(C1, _samples(2), RATE)
    out.start(C2, _samples(2, first=11), RATE, lead_s=0.3)
    stream = _stream()
    assert stream.pull(8) == [1.0, 2.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0]
    assert events.done == [C1, C2]


def test_on_done_order_follows_the_last_sample_of_each_chunk() -> None:
    out, events = _output()
    out.start(C1, _samples(3), RATE)
    out.start(C2, _samples(3, first=11), RATE)
    stream = _stream()
    stream.pull(2)
    assert events.done == []
    stream.pull(2)  # C1's last sample and C2's first
    assert events.done == [C1]
    stream.pull(2)
    assert events.done == [C1, C2]


def test_an_empty_chunk_is_done_at_the_next_block() -> None:
    out, events = _output()
    out.start(C1, _samples(0), RATE)
    assert _stream().pull(2) == _zeros(2)
    assert events.done == [C1]


def test_idle_close_after_two_seconds_with_nothing_queued() -> None:
    out, events = _output()
    out.start(C1, _samples(2), RATE)
    first = _stream()
    first.pull(2)
    assert first.pull(19) == _zeros(19)
    with pytest.raises(sd.CallbackStop):
        first.pull(1)  # 2 s of silence: the stream stops itself
    first.finish()
    assert not first.closed  # PortAudio forbids closing from its own callback
    assert events.failed == []
    out.start(C2, _samples(2, first=11), RATE, lead_s=0.8)
    second = _stream()
    assert first.closed
    assert second is not first
    assert second.started
    assert second.pull(2) == [11.0, 12.0]  # the gap is already longer than the lead
    assert events.done == [C1, C2]


def test_pause_keeps_position_and_does_not_count_toward_the_gap() -> None:
    out, events = _output()
    out.start(C1, _samples(4), RATE)
    stream = _stream()
    assert stream.pull(2) == [1.0, 2.0]
    out.pause()
    assert stream.pull(30) == _zeros(30)  # no idle close while a chunk is paused
    out.resume()
    assert stream.pull(2) == [3.0, 4.0]
    assert events.done == [C1]
    out.pause()
    assert stream.pull(30) == _zeros(30)  # paused between chunks: no gap, no idle close
    out.start(C2, _samples(1, first=11), RATE, lead_s=0.3)
    assert stream.pull(1) == [0.0]
    out.resume()
    assert stream.pull(4) == [0.0, 0.0, 0.0, 11.0]  # the whole lead is still to come
    assert events.done == [C1, C2]


def test_pause_persists_across_abort_and_start() -> None:
    out, _ = _output()
    out.pause()
    out.abort()
    out.start(C1, _samples(4), RATE)
    stream = _stream()
    assert stream.pull(2) == _zeros(2)
    out.resume()
    assert stream.pull(2) == [1.0, 2.0]


def test_abort_clears_the_queue_silently_and_closes_the_stream() -> None:
    out, events = _output()
    out.start(C1, _samples(8), RATE)
    out.start(C2, _samples(8), RATE)
    first = _stream()
    first.pull(2)
    out.abort()
    assert first.aborted
    assert first.closed
    assert first.pull(2) == _zeros(2)  # a late block: silence; only the closer stops it
    out.abort()  # idempotent
    assert events.done == []
    assert events.failed == []
    out.start(C3, _samples(1, first=21), RATE, lead_s=0.2)
    second = _stream()
    assert second is not first
    assert second.pull(3) == [0.0, 0.0, 21.0]  # a cut counts as the last sample: full lead
    assert events.done == [C3]


def test_time_since_a_cut_counts_toward_the_next_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    # After skip/back the next chunk is often generated first: that wait is silence already.
    clock = FakeClock()
    monkeypatch.setattr(sounddevice_out, "_monotonic", clock.now)
    out, _ = _output()
    out.start(C1, _samples(4), RATE)
    _stream().pull(2)
    out.abort()
    clock.advance(0.3)
    out.start(C2, _samples(1, first=11), RATE, lead_s=0.4)
    assert _stream().pull(2) == [0.0, 11.0]  # 0.3 s already passed: 0.1 s left


def test_time_paused_after_a_cut_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sounddevice_out, "_monotonic", clock.now)
    out, _ = _output()
    out.start(C1, _samples(4), RATE)
    out.abort()
    out.pause()
    clock.advance(5.0)
    out.start(C2, _samples(1, first=11), RATE, lead_s=0.2)
    out.resume()
    assert _stream().pull(3) == [0.0, 0.0, 11.0]


def test_time_after_a_release_counts_too(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sounddevice_out, "_monotonic", clock.now)
    out, _ = _output()
    out.start(C1, _samples(1), RATE)
    _stream().pull(2)  # C1 done, then 0.1 s of gap
    assert out.release_idle_stream()  # carries the 0.1 s over
    clock.advance(0.2)
    out.start(C2, _samples(1, first=11), RATE, lead_s=0.5)
    assert _stream().pull(3) == [0.0, 0.0, 11.0]  # 0.1 + 0.2 s passed: 0.2 s left


# ---------------------------------------------------------------- never stopped twice at once


def test_abort_never_races_the_callback_into_stopping_the_stream_too() -> None:
    # The field hang: abort() marked the stream ended, and a callback running at that moment
    # raised CallbackAbort, so the IO thread and the loop thread stopped it at once (deadlock).
    out, events = _output()
    out.start(C1, _samples(8), RATE)
    stream = _stream()
    stream.pull(2)
    out.abort()  # FakeStream.abort runs one IO cycle at the same moment
    assert stream.aborted
    assert stream.closed
    assert events.done == []
    assert events.failed == []


def test_a_drain_never_races_the_callback_into_stopping_the_stream_too() -> None:
    out, _ = _output()
    out.start(C1, _samples(2), RATE)
    stream = _stream()
    stream.pull(3)
    assert out.release_idle_stream()  # stop() runs one IO cycle at the same moment
    assert stream.stopped
    assert stream.closed


def _stopping_itself(out: SoundDeviceOutput) -> FakeStream:
    """Play one chunk, then idle until the callback stops its stream (finished not yet come)."""
    out.start(C1, _samples(2), RATE)
    stream = _stream()
    stream.pull(2)
    stream.pull(19)
    with pytest.raises(sd.CallbackStop):
        stream.pull(1)
    assert stream.stopping_itself
    return stream


def test_a_stream_stopping_itself_is_left_alone_until_it_has_finished() -> None:
    out, events = _output()
    first = _stopping_itself(out)
    out.abort()  # must not stop or close it while CoreAudio still stops it
    assert not first.aborted
    assert not first.closed
    out.start(C2, _samples(2, first=11), RATE)  # a new stream meanwhile
    second = _stream()
    assert second is not first
    assert second.pull(2) == [11.0, 12.0]
    first.finish()
    out.abort()  # now the finished stream is closed as well
    assert first.closed
    assert second.closed
    assert events.done == [C1, C2]
    assert events.failed == []


def test_a_stream_stopping_itself_postpones_a_re_initialisation() -> None:
    out, _ = _output()
    first = _stopping_itself(out)
    assert not out.release_idle_stream()  # Pa_Terminate would close it: a deadlock
    assert not first.closed
    first.finish()
    assert out.release_idle_stream()
    assert first.closed


def test_a_stream_whose_finish_does_not_come_is_logged_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(sounddevice_out, "_monotonic", clock.now)
    out, _ = _output()
    first = _stopping_itself(out)
    assert not out.release_idle_stream()  # set aside now
    clock.advance(2.9)
    assert not out.release_idle_stream()
    assert "did not finish" not in caplog.text
    clock.advance(0.2)
    assert not out.release_idle_stream()
    assert not out.release_idle_stream()
    assert caplog.text.count("did not finish") == 1  # device refreshes wait for it
    first.finish()
    assert out.release_idle_stream()


def test_a_callback_error_is_closed_only_after_the_stream_finished() -> None:
    out, events = _output()
    out.start(C1, _samples(4), RATE)
    stream = _stream()
    with pytest.raises(sd.CallbackAbort):
        stream.pull(4, np.zeros(4, dtype=np.float32))  # wrong shape: the callback fails
    out.abort()
    assert not stream.closed
    stream.finish()
    out.abort()
    assert stream.closed
    assert events.failed == []  # aborted before the finish: the queue was dropped silently


def test_a_chunk_finished_before_a_callback_error_is_still_reported_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, events = _output()
    out.start(C1, _samples(2), RATE)
    stream = _stream()

    def broken_clock() -> float:
        raise RuntimeError("clock broke")  # fails after C1's last sample, in the same block

    monkeypatch.setattr(sounddevice_out, "_monotonic", broken_clock)
    with pytest.raises(sd.CallbackAbort):
        stream.pull(4)
    assert events.done == [C1]  # its last sample was handed over before the error


def test_exit_waits_briefly_for_a_stream_stopping_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def register(func: Callable[..., object], *args: object) -> None:
        hooks.append((func, args))

    monkeypatch.setattr(sounddevice_out.atexit, "register", register)
    monkeypatch.setattr(sounddevice_out, "_EXIT_WAIT_S", 0.05)
    out, _ = _output()
    first = _stopping_itself(out)
    ((hook, args),) = hooks
    hook(*args)  # never finishes: gives up after the short wait, leaves it alone
    assert not first.closed
    first.finish()
    hook(*args)
    assert first.closed


def test_rate_change_reopens() -> None:
    out, events = _output()
    out.start(C1, _samples(2), RATE)
    first = _stream()
    assert first.pull(3) == [1.0, 2.0, 0.0]  # 0.1 s of the gap on the old stream
    out.start(C2, _samples(2, first=11), 2 * RATE, lead_s=0.2)
    second = _stream()
    assert first.stopped  # drained, not cut: C1's tail is still in the device buffer
    assert not first.aborted
    assert first.closed
    assert second.sample_rate == 2 * RATE
    assert second.pull(5) == [0.0, 0.0, 11.0, 12.0, 0.0]  # 0.1 s carried over, 0.1 s added
    assert events.done == [C1, C2]
    assert events.failed == []


def test_rate_change_with_chunks_still_queued_fails_them() -> None:
    out, events = _output()
    out.start(C1, _samples(8), RATE)
    _stream().pull(2)
    out.start(C2, _samples(2), 2 * RATE)
    assert events.failed == [(C1, "output reopened at another sample rate")]
    assert _stream().pull(2) == [1.0, 2.0]
    assert events.done == [C2]


def test_open_failure_reported() -> None:
    out, events = _output()
    FakeStream.fail_open = True
    out.start(C1, _samples(4), RATE)
    assert events.failed == [(C1, "cannot open")]
    assert FakeStream.instances == []
    FakeStream.fail_open = False
    out.start(C2, _samples(2), RATE)
    assert _stream().pull(2) == [1.0, 2.0]
    assert events.done == [C2]


def test_start_failure_reports_and_closes_stream() -> None:
    out, events = _output()
    FakeStream.fail_start = True
    out.start(C3, _samples(4), RATE)
    assert events.failed == [(C3, "no device")]
    assert FakeStream.instances[0].closed
    out.abort()  # nothing left to close; must not close twice


def test_start_failure_without_callbacks_is_silent() -> None:
    out = SoundDeviceOutput()
    FakeStream.fail_start = True
    out.start(C3, _samples(4), RATE)
    assert FakeStream.instances[0].closed


def test_a_stream_that_stops_early_fails_every_queued_chunk() -> None:
    out, events = _output()
    out.start(C1, _samples(10), RATE)
    out.start(C2, _samples(4), RATE)
    first = _stream()
    first.pull(4)
    first.finish()  # the device went away
    assert events.done == []
    assert events.failed == [
        (C1, "stream stopped before end"),
        (C2, "stream stopped before end"),
    ]
    out.start(C3, _samples(2), RATE)  # a stopped stream is never reused
    assert first.closed
    assert _stream() is not first
    assert _stream().pull(2) == [1.0, 2.0]


def test_finished_twice_reports_once() -> None:
    out, events = _output()
    out.start(C1, _samples(10), RATE)
    stream = _stream()
    stream.finished_callback()
    stream.finished_callback()
    assert events.failed == [(C1, "stream stopped before end")]


def test_callback_error_aborts_and_reports_failure() -> None:
    out, events = _output()
    out.start(C1, _samples(4), RATE)
    stream = _stream()
    bad = np.zeros(4, dtype=np.float32)  # wrong shape: the callback expects (frames, 1)
    with pytest.raises(sd.CallbackAbort):
        stream.pull(4, bad)
    stream.finish()
    assert events.done == []
    assert len(events.failed) == 1
    assert events.failed[0][0] == C1
    assert events.failed[0][1].startswith("IndexError")


def test_finished_without_callbacks_is_silent() -> None:
    out = SoundDeviceOutput()
    out.start(C1, _samples(2), RATE)
    stream = _stream()
    stream.pull(4)
    stream.finish()


def test_callbacks_can_be_set_only_once() -> None:
    out, _ = _output()
    with pytest.raises(RuntimeError):
        out.set_callbacks(lambda _c: None, lambda _c, _m: None)


def test_release_idle_stream_refuses_while_queued_and_drains_otherwise() -> None:
    out, events = _output()
    out.start(C1, _samples(2), RATE)
    stream = _stream()
    assert not out.release_idle_stream()
    assert not stream.closed
    assert stream.pull(3) == [1.0, 2.0, 0.0]
    assert out.release_idle_stream()
    assert stream.stopped
    assert stream.closed
    assert out.release_idle_stream()  # nothing open: nothing closed twice
    out.start(C2, _samples(1, first=11), RATE, lead_s=0.3)
    assert _stream().pull(3) == [0.0, 0.0, 11.0]  # the 0.1 s heard before still counts
    assert events.done == [C1, C2]


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
    out.start(ChunkId(chunk), _samples(2), RATE)
    stream = _stream()
    stream.pull(2)
    return stream


def test_a_fresh_output_does_not_re_initialise(clock: FakeClock, reinits: Reinits) -> None:
    out, _ = _output()
    out.start(C1, _samples(2), RATE)
    assert reinits.count == 0


def test_a_reading_that_starts_from_idle_re_initialises_first(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    first = _play_to_end(out, 1)
    assert not first.closed  # the stream stays open between chunks
    clock.advance(2.0)  # the output's idle threshold (_IDLE_S)
    out.start(C2, _samples(2), RATE)
    assert reinits.count == 1  # and, by the clock fixture, with no stream open
    assert first.closed
    assert events.done == [C1]
    assert _stream().started


def test_chunks_of_one_reading_never_re_initialise(clock: FakeClock, reinits: Reinits) -> None:
    out, _ = _output()
    for chunk in range(5):
        _play_to_end(out, chunk)
        clock.advance(0.05)
    assert reinits.count == 0
    assert len(FakeStream.instances) == 1


def test_a_long_pause_while_playing_is_not_idle(clock: FakeClock, reinits: Reinits) -> None:
    out, _ = _output()
    out.start(C1, _samples(8), RATE)
    out.pause()
    clock.advance(60.0)
    out.start(C2, _samples(8), RATE)  # queued behind a chunk that is playing (paused)
    assert reinits.count == 0
    assert len(FakeStream.instances) == 1


def test_a_playback_failure_re_initialises_at_the_next_abort(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    out.start(C1, _samples(10), RATE)
    stream = _stream()
    stream.pull(4)
    stream.finish()  # stopped early: the device went away
    assert events.failed == [(C1, "stream stopped before end")]
    assert reinits.count == 0  # never from the PortAudio thread
    out.abort()
    assert reinits.count == 1
    out.start(C2, _samples(2), RATE)
    assert reinits.count == 1  # done once


def test_a_failed_start_re_initialises_before_the_next_start(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    FakeStream.fail_start = True
    out.start(C1, _samples(4), RATE)
    assert events.failed == [(C1, "no device")]
    FakeStream.fail_start = False
    out.start(C1, _samples(4), RATE)  # the retry, at once
    assert reinits.count == 1
    assert _stream().started


def test_a_failed_refresh_in_start_is_a_playback_failure_and_the_next_start_recovers(
    clock: FakeClock, reinits: Reinits
) -> None:
    """R1: no exception escapes start(); the output stays quiet (not "playing") and retries."""
    out, events = _output()
    clock.advance(10.0)
    reinits.fail_initialize = 1
    out.start(C1, _samples(2), RATE)  # must not raise
    assert events.failed == [(C1, "Error initializing PortAudio")]
    assert FakeStream.instances == []  # nothing opened on an uninitialised library
    out.start(C2, _samples(2), RATE)  # the next speak: initialise only, then play
    assert (reinits.terminates, reinits.initializes, reinits.count) == (1, 2, 1)
    assert _stream().started
    out.abort()


def test_a_failed_refresh_in_abort_is_retried_by_the_next_start(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, events = _output()
    out.start(C1, _samples(10), RATE)
    stream = _stream()
    stream.pull(4)
    stream.finish()  # failed: a refresh is due
    reinits.fail_terminate = 1
    out.abort()  # must not raise
    assert reinits.count == 0
    out.start(C2, _samples(2), RATE)
    assert reinits.count == 1
    assert len(events.failed) == 1
    out.abort()


def test_a_rate_change_within_a_reading_does_not_re_initialise(
    clock: FakeClock, reinits: Reinits
) -> None:
    out, _ = _output()
    _play_to_end(out, 1)
    out.start(C2, _samples(2), 2 * RATE)
    assert reinits.count == 0
    assert len(FakeStream.instances) == 2


def test_another_output_still_playing_postpones_the_re_initialisation(
    clock: FakeClock, reinits: Reinits
) -> None:
    playing, _ = _output()
    playing.start(C1, _samples(8), RATE)
    out, _ = _output()
    clock.advance(10.0)
    reinits.check = None  # the other output's stream is open on purpose
    out.start(C2, _samples(2), RATE)
    assert reinits.count == 0
    playing.abort()


def test_another_output_releases_its_finished_stream_for_the_re_initialisation(
    clock: FakeClock, reinits: Reinits
) -> None:
    other, _ = _output()
    kept = _play_to_end(other, 1)
    out, _ = _output()
    clock.advance(10.0)
    out.start(C2, _samples(2), RATE)
    assert reinits.count == 1
    assert kept.closed
    other.abort()  # its stream is gone already: nothing is closed twice


# ---------------------------------------------------------------- the real stream's parameters


def test_the_device_stream_asks_for_a_deadline_longer_than_any_gil_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The callback needs the GIL; with ``latency="high"`` CoreAudio gave it 4.6 ms per buffer
    while ``mx.clear_cache()`` on the engine thread holds the GIL for 6 to 34 ms, so buffers were
    late and crackled. The stream asks for 0.2 s (about 87 ms of headroom per callback)."""
    opened: list[dict[str, object]] = []

    def output_stream(**kwargs: object) -> object:
        opened.append(kwargs)
        return object()

    def callback(
        outdata: npt.NDArray[np.float32], frames: int, time: object, status: sd.CallbackFlags
    ) -> None:
        pass

    monkeypatch.setattr(sounddevice_out.sd, "OutputStream", output_stream)
    sounddevice_out._sounddevice_stream(24000, callback, lambda: None)  # pyright: ignore[reportPrivateUsage]
    assert opened[0]["latency"] == pytest.approx(0.2)
    assert "blocksize" not in opened[0]  # PortAudio sizes the host buffer from the latency
