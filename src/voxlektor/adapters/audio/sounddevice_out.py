"""AudioOutput on one continuous PortAudio callback stream; lead gaps; pause = silence.

One stream stays open while chunks keep coming. It holds a FIFO of chunks, each with its lead
(the pause before it). The callback writes the missing part of the lead as silence, then the
samples, and reports ``on_done`` as soon as the chunk's last sample has been handed to the
device (written into PortAudio's buffer); between chunks it writes silence. No new stream per
chunk, so there is no reopen gap at a boundary.

Lead = minimum gap: the stream counts the silent frames written since the previous chunk's last
sample (``gap``; frames written while paused do not count) and adds only what the lead still
misses, so a generation stall counts toward the pause. A fresh stream after an idle close or a
failure starts with the gap "long over"; after ``abort()`` it starts at zero (a cut counts as a
last sample, the next chunk gets its whole lead); a stream reopened at another sample rate or
released for a device refresh carries the gap over. The time between the cut (or the close)
and the next open counts as gap too, unless the output is paused at that open: after skip/back
the next chunk is often generated first, and that wait is silence already (no double lead).

Stream lifetime: the stream is (re)opened by ``start()`` when there is none, when it stopped,
when the output has been idle for ``_IDLE_S``, or when the sample rate changes (the old stream is
drained with ``stop()``, so the previous chunk's tail still plays; chunks still queued on it are
reported failed). The callback stops the stream itself after ``_IDLE_S`` of unpaused silence with
nothing queued (``CallbackStop``); the stopped stream is closed by the next ``start()``, by
``abort()`` or by ``release_idle_stream`` (PortAudio forbids closing a stream from its own
callbacks). ``abort()`` cuts and closes the stream at once and drops the queue silently. The
paused flag belongs to the output, not to a chunk: it survives ``abort()`` and ``start()`` (a
start while paused writes silence until ``resume()``); paused silence never closes the stream.

Never stopped from two threads at once (``_close``; both at once deadlock inside CoreAudio): a
stream that ``_close`` took over is never stopped by its callback (it writes silence until the
loop thread has cut it), and a stream whose callback is stopping it (idle, or a callback error)
is not touched until its finished callback came: it is set aside and closed by a later
``start``/``abort``/``release_idle_stream``, and a device refresh waits for it. One that has not
finished ``_STUCK_S`` (3 s) after it was set aside is logged once as a warning.

Threads: ``start/pause/resume/abort/release_idle_stream`` run on the loop thread, the stream
callback and the finished callback on the PortAudio thread (the adapter's own). Shared state
sits behind one lock, never held while a stream is stopped, aborted or closed (that waits for
the PortAudio thread). ``on_done``/``on_failed`` are invoked as given, outside the lock (the
runtime wraps them with ``call_soon_threadsafe``). Never reached by the CLI control path.

Following the default output device (``portaudio.refresh_devices``): only when a stream is
(re)opened. ``start()`` re-initialises PortAudio, after closing its own stream, when the output
has been idle for ``_IDLE_S`` (the first chunk of a reading that starts from idle; chunks of one
reading follow each other at once), and after any playback failure (``mark_stale``), so the next
speak works when the old device vanished. ``abort()`` does the same after a failure, so a cue
that follows plays on the new device.

Buffering (``_LATENCY_S``): the stream is opened with 0.2 s of latency, so the Python callback
has a deadline far longer than any GIL hold by another thread. The price: what was handed over
still plays, so a pause is heard up to about 0.2 s late and ``on_done`` comes up to that much
before the last sample is heard (an abort cuts at once).

Tests replace ``_open_stream`` (the stream factory) and ``_monotonic`` (the clock).
"""

import atexit
import contextlib
import logging
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from voxlektor.adapters.audio import portaudio
from voxlektor.domain.types import ChunkId

_IDLE_S = 2.0  # quiet this long = idle: the stream closes, the next start refreshes the devices
# The callback runs Python, so it first waits for the GIL; other threads hold it for tens of ms
# (``mx.clear_cache()`` after each synthesis: 6 to 34 ms, measured). With ``latency="high"``
# CoreAudio called back every 0.6 ms with 4.6 ms to spare, so such a hold made a buffer late
# (a crackle). 0.2 s gives 167 ms buffers, each handed over about 87 ms before it is played.
_LATENCY_S = 0.2
_LONG_AGO = 1 << 62  # gap frames of a fresh stream: any lead is already over
_EXIT_WAIT_S = 0.5  # at exit: how long a stream stopping itself may take to finish
_REOPENED = "output reopened at another sample rate"
# A stream stopping itself normally finishes within one IO cycle; this long without its finished
# callback is logged once (device refreshes, which wait for it, are blocked meanwhile).
_STUCK_S = 3.0

_log = logging.getLogger(__name__)

type _Callback = Callable[[npt.NDArray[np.float32], int, object, sd.CallbackFlags], None]


class _Stream(Protocol):
    def start(self) -> None: ...

    def stop(self, ignore_errors: bool = True) -> None: ...

    def abort(self, ignore_errors: bool = True) -> None: ...

    def close(self, ignore_errors: bool = True) -> None: ...


def _sounddevice_stream(
    sample_rate: int, callback: _Callback, finished_callback: Callable[[], None]
) -> _Stream:
    return sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        latency=_LATENCY_S,
        callback=callback,
        finished_callback=finished_callback,
    )


# Replaced by tests (fake stream, fake clock).
_open_stream: Callable[[int, _Callback, Callable[[], None]], _Stream] = _sounddevice_stream
_monotonic = time.monotonic


class _Entry:
    """One queued chunk: its buffer, read position and lead (in frames)."""

    __slots__ = ("chunk", "lead", "pos", "samples")

    def __init__(self, chunk: ChunkId, samples: npt.NDArray[np.float32], lead: int) -> None:
        self.chunk = chunk
        self.samples = samples
        self.lead = lead
        self.pos = 0


class _Open:
    """One opened stream: its rate, FIFO, gap counter and outcome."""

    def __init__(self, sample_rate: int, gap: int) -> None:
        self.sample_rate = sample_rate
        self.idle_frames = round(_IDLE_S * sample_rate)
        self.queue: deque[_Entry] = deque()
        self.gap = gap  # unpaused silent frames written since the last chunk's last sample
        self.stream: _Stream | None = None
        self.live = True  # False once the stream stopped (or is stopping)
        self.ended = False  # closed by us or finished: no more reports
        self.stopping = False  # the callback stops the stream itself (CallbackStop/Abort)
        self.finished = False  # PortAudio's finished callback came: nobody is stopping it
        self.error: str | None = None
        self.aside_at = 0.0  # when ``_close`` set it aside (stopping itself, not finished)
        self.warned = False  # the "did not finish" warning was logged


class SoundDeviceOutput:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._open: _Open | None = None
        self._carried_gap_s = float("inf")  # the gap a newly opened stream starts with
        self._carried_at = 0.0  # when that gap was taken over (time since then is silence too)
        self._on_done: Callable[[ChunkId], None] | None = None
        self._on_failed: Callable[[ChunkId, str], None] | None = None
        self._quiet_since: float | None = _monotonic()  # None while chunks are queued
        self._unclosed: list[_Open] = []  # stopping themselves: closed once they have finished
        portaudio.register(self)
        # A stream still running while the interpreter exits crashes it (PortAudio thread). Runs
        # before sounddevice's own exit handler (registered at its import; atexit is LIFO).
        atexit.register(_call_if_alive, weakref.WeakMethod(self._cut_at_exit))

    def _cut_at_exit(self) -> None:
        # Pa_Terminate (sounddevice's exit handler) closes what is left, so give a stream that is
        # stopping itself a moment to finish first (one IO cycle, normally milliseconds).
        self._close(cut=True)
        deadline = time.monotonic() + _EXIT_WAIT_S
        while self._has_unclosed(finished=False) and time.monotonic() < deadline:
            time.sleep(0.005)
        self._close(cut=True)

    def release_idle_stream(self) -> bool:
        """``portaudio.StreamOwner``: drain and close the stream unless chunks are queued.

        False while a stream is still stopping itself: ``Pa_Terminate`` would close it, which is
        the deadlock described in ``_close``.
        """
        with self._lock:
            if self._open is not None and self._open.queue:
                return False
        self._close(cut=False)
        return not self._has_unclosed()

    def set_callbacks(
        self,
        on_done: Callable[[ChunkId], None],
        on_failed: Callable[[ChunkId, str], None],
    ) -> None:
        if self._on_done is not None:
            raise RuntimeError("callbacks already set")
        self._on_done = on_done
        self._on_failed = on_failed

    def start(
        self, chunk: ChunkId, samples: object, sample_rate: int, *, lead_s: float = 0.0
    ) -> None:
        buffer = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        entry = _Entry(chunk, buffer, max(0, round(lead_s * sample_rate)))
        with self._lock:
            quiet_since = self._quiet_since
            idle = quiet_since is not None and _monotonic() - quiet_since >= _IDLE_S
            current = self._open
            if (
                current is not None
                and current.live
                and current.sample_rate == sample_rate
                and not idle
            ):
                current.queue.append(entry)
                self._quiet_since = None
                return
        self._report_failed(self._close(cut=False), _REOPENED)
        self._open_with(entry, sample_rate, idle=idle)

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def abort(self) -> None:
        """Drop every queued chunk silently (no callback) and close the stream at once.

        After a playback failure this is also where PortAudio is re-initialised (no stream of
        ours is open any more), so a cue that follows already uses the current default device.
        """
        self._close(cut=True)
        with self._lock:
            self._carried_gap_s = 0.0
            self._carried_at = _monotonic()
        # A failed refresh stays due: the next start() retries it and reports a failure there.
        with contextlib.suppress(sd.PortAudioError):
            portaudio.refresh_devices(idle=False)

    # ------------------------------------------------------------ loop thread

    def _open_with(self, entry: _Entry, sample_rate: int, *, idle: bool) -> None:
        with self._lock:
            gap_s = self._carried_gap_s
            if gap_s != float("inf") and not self._paused:
                gap_s += max(0.0, _monotonic() - self._carried_at)  # silence while closed
            gap = _LONG_AGO if gap_s == float("inf") else round(gap_s * sample_rate)
            opened = _Open(sample_rate, gap)
            opened.queue.append(entry)
            self._quiet_since = None

        def callback(
            outdata: npt.NDArray[np.float32], frames: int, time: object, status: sd.CallbackFlags
        ) -> None:
            self._fill(opened, outdata, frames)

        def finished() -> None:
            self._finished(opened)

        stream: _Stream | None = None
        try:
            # A failed refresh is a playback failure like a failed open: reported, retried later.
            portaudio.refresh_devices(idle=idle)
            stream = _open_stream(sample_rate, callback, finished)
            with self._lock:
                opened.stream = stream
                self._open = opened
            stream.start()
        except sd.PortAudioError as exc:
            with self._lock:
                opened.ended = True
                opened.live = False
                opened.queue.clear()
                if self._open is opened:
                    self._open = None
                self._carried_gap_s = float("inf")
                self._quiet_since = _monotonic()
            if stream is not None:
                stream.close()
            portaudio.mark_stale()  # e.g. the default device is gone: re-scan before the next try
            self._report_failed([entry.chunk], str(exc))

    def _close(self, *, cut: bool) -> list[ChunkId]:
        """Close the stream (drained, or cut when ``cut``); returns the chunks still queued.

        Never while the callback stops the stream itself. Then PortAudio's IO thread is inside
        ``AudioOutputUnitStop`` holding CoreAudio's IO-context mutex and waiting for the unit's
        mutex, while ``stop``/``abort``/``close`` here hold the unit's mutex and wait for the
        IO-context mutex: a deadlock in CoreAudio, forever, with no Python frame and no GIL
        involved (the field hang: a daemon deaf to its socket and to SIGTERM). The decision is
        taken under the lock on both sides: a stream marked ``ended`` here is never stopped by
        its callback (it writes silence), and a stream whose callback chose to stop it is left
        alone until its finished callback came; it is closed by a later ``_close``.
        """
        with self._lock:
            finished = [o for o in self._unclosed if o.finished]
            self._unclosed = [o for o in self._unclosed if not o.finished]
            now = _monotonic()
            stuck = [o for o in self._unclosed if not o.warned and now - o.aside_at >= _STUCK_S]
            for o in stuck:
                o.warned = True
            opened, self._open = self._open, None
            pending: list[ChunkId] = []
            if opened is not None:
                pending = [] if opened.ended else [e.chunk for e in opened.queue]
                opened.queue.clear()
                opened.ended = True
                if opened.live:
                    self._carried_gap_s = opened.gap / opened.sample_rate
                    self._carried_at = now
                else:
                    self._carried_gap_s = float("inf")
                opened.live = False
                if pending:
                    self._quiet_since = _monotonic()
                if opened.stopping and not opened.finished:
                    opened.aside_at = now
                    self._unclosed.append(opened)
                    opened = None
        for o in stuck:
            _log.warning(
                "an output stream that stopped itself did not finish within %.0f s (%.1f s); "
                "device refreshes wait for it",
                _STUCK_S,
                now - o.aside_at,
            )
        for done in finished:
            if done.stream is not None:
                done.stream.close()
        stream = None if opened is None else opened.stream
        if stream is not None:
            if cut:
                stream.abort()
            else:
                stream.stop()
            stream.close()
        return pending

    def _has_unclosed(self, *, finished: bool = True) -> bool:
        """Any stream set aside (``finished=False``: only those not finished yet)."""
        with self._lock:
            return any(finished or not o.finished for o in self._unclosed)

    def _report_failed(self, chunks: list[ChunkId], msg: str) -> None:
        if self._on_failed is not None:
            for chunk in chunks:
                self._on_failed(chunk, msg)

    # ------------------------------------------------------------ PortAudio thread

    def _fill(self, opened: _Open, outdata: npt.NDArray[np.float32], frames: int) -> None:
        # No allocation beyond slicing and the list of chunks done, no logging; the application
        # is only called (on_done) after the lock is released.
        # The stream is stopped either from here (raising) or from the loop thread (``_close``),
        # never both: see ``_close``. Both decide under the lock.
        done: list[ChunkId] = []
        failure: Exception | None = None
        with self._lock:
            if opened.ended:
                outdata.fill(0.0)  # being closed by the loop thread: it alone stops the stream
                return
            try:
                stop = self._write(opened, outdata, frames, done)
            except Exception as exc:
                opened.error = f"{type(exc).__name__}: {exc}"
                failure, stop = exc, True
            if stop:
                opened.stopping = True
                opened.live = False  # no start() may queue on a stopping stream
        # Chunks whose last sample was handed over are done, even when this block then failed.
        if self._on_done is not None:
            for chunk in done:
                self._on_done(chunk)
        if failure is not None:
            raise sd.CallbackAbort from failure
        if stop:
            raise sd.CallbackStop

    def _write(
        self, opened: _Open, outdata: npt.NDArray[np.float32], frames: int, done: list[ChunkId]
    ) -> bool:
        """Fill one block (lock held); True when the stream should stop (idle)."""
        i = 0
        while i < frames:
            if self._paused:
                outdata[i:].fill(0.0)  # paused: position kept, no gap counted
                return False
            if not opened.queue:
                outdata[i:].fill(0.0)
                opened.gap += frames - i
                return opened.gap >= opened.idle_frames  # idle: the stream stops itself
            head = opened.queue[0]
            missing = head.lead - opened.gap if head.pos == 0 else 0
            if missing > 0:
                n = min(missing, frames - i)
                outdata[i : i + n].fill(0.0)
                opened.gap += n
                i += n
                continue
            n = min(len(head.samples) - head.pos, frames - i)
            outdata[i : i + n, 0] = head.samples[head.pos : head.pos + n]
            head.pos += n
            i += n
            if head.pos == len(head.samples):
                opened.queue.popleft()
                done.append(head.chunk)
                opened.gap = 0
                if not opened.queue:
                    self._quiet_since = _monotonic()
        return False

    def _finished(self, opened: _Open) -> None:
        # After the stream has stopped (drained, idle, aborted or failed). The stream is not
        # closed here; start()/abort()/release_idle_stream() close it from the loop thread.
        with self._lock:
            opened.live = False
            opened.finished = True
            if opened.ended:
                return
            opened.ended = True
            pending = [e.chunk for e in opened.queue]
            opened.queue.clear()
            if pending:
                self._quiet_since = _monotonic()
            error = opened.error
        if error is not None or pending:
            portaudio.mark_stale()  # the device may have vanished: re-scan before the next stream
        self._report_failed(pending, error or "stream stopped before end")


def _call_if_alive(method: "weakref.WeakMethod[Callable[[], object]]") -> None:
    bound = method()
    if bound is not None:
        bound()
