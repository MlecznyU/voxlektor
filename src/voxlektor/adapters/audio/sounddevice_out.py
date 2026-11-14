"""AudioOutput on a PortAudio callback stream; pause = silence.

Threads: ``start/pause/resume/abort`` run on the loop thread, the stream callback and the
finished callback on the PortAudio thread. Shared state sits behind one lock; the callbacks
``on_done``/``on_failed`` are invoked as given (the runtime wraps them with
``call_soon_threadsafe``). Never reached by the CLI control path.

Stream lifetime: the stream of the last chunk stays open after it finished (PortAudio forbids
closing a stream from its own finished callback) and is closed by the next ``start()`` or by
``abort()``. The paused flag belongs to the output, not to a chunk: it survives ``abort()``
and ``start()`` (a start while paused writes silence until ``resume()``).

Following the default output device (``portaudio.refresh_devices``): ``start()`` re-initialises
PortAudio, after closing its own last stream, when the output has been idle for ``_IDLE_S`` (the
first chunk of a reading that starts from idle; chunks of one reading follow each other at once),
and after any playback failure (``mark_stale``), so the next speak works when the old device
vanished. ``abort()`` does the same after a failure, so a cue that follows plays on the new device.
"""

import contextlib
import threading
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from voxlektor.adapters.audio import portaudio
from voxlektor.domain.types import ChunkId

_IDLE_S = 2.0  # quiet this long before a start = a reading that starts from idle

# Replaced by tests (fake clock).
_monotonic = time.monotonic


class _Playback:
    """One chunk being played: its buffer, read position and outcome."""

    def __init__(self, chunk: ChunkId, samples: npt.NDArray[np.float32]) -> None:
        self.chunk = chunk
        self.samples = samples
        self.pos = 0
        self.cancelled = False
        self.error: str | None = None


class SoundDeviceOutput:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._playback: _Playback | None = None
        self._stream: sd.OutputStream | None = None
        self._on_done: Callable[[ChunkId], None] | None = None
        self._on_failed: Callable[[ChunkId, str], None] | None = None
        self._quiet_since: float | None = _monotonic()  # None while a chunk plays
        portaudio.register(self)

    def release_idle_stream(self) -> bool:
        """``portaudio.StreamOwner``: close the last stream unless a chunk still plays."""
        with self._lock:
            if self._playback is not None:
                return False
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()
        return True

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
        self._stop()
        with self._lock:
            quiet_since, self._quiet_since = self._quiet_since, None
        idle = quiet_since is not None and _monotonic() - quiet_since >= _IDLE_S
        buffer = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        playback = _Playback(chunk, buffer)

        def callback(
            outdata: npt.NDArray[np.float32], frames: int, time: object, status: sd.CallbackFlags
        ) -> None:
            self._fill(playback, outdata, frames)

        def finished() -> None:
            self._finished(playback)

        stream: sd.OutputStream | None = None
        try:
            # A failed refresh is a playback failure like a failed open: reported, retried later.
            portaudio.refresh_devices(idle=idle)
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                latency="high",
                callback=callback,
                finished_callback=finished,
            )
            with self._lock:
                self._playback = playback
                self._stream = stream
            stream.start()
        except sd.PortAudioError as exc:
            with self._lock:
                playback.cancelled = True
                self._playback = None
                self._stream = None
                self._quiet_since = _monotonic()
            if stream is not None:
                stream.close()
            portaudio.mark_stale()  # e.g. the default device is gone: re-scan before the next try
            if self._on_failed is not None:
                self._on_failed(chunk, str(exc))

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def abort(self) -> None:
        """Stop the current chunk silently (no callback) and close the last stream.

        After a playback failure this is also where PortAudio is re-initialised (no stream of
        ours is open any more), so a cue that follows already uses the current default device.
        """
        self._stop()
        # A failed refresh stays due: the next start() retries it and reports a failure there.
        with contextlib.suppress(sd.PortAudioError):
            portaudio.refresh_devices(idle=False)

    def _stop(self) -> None:
        with self._lock:
            playback = self._playback
            stream = self._stream
            self._playback = None
            self._stream = None
            if playback is not None:
                playback.cancelled = True
                self._quiet_since = _monotonic()
        if stream is not None:
            stream.abort()
            stream.close()

    def _fill(self, playback: _Playback, outdata: npt.NDArray[np.float32], frames: int) -> None:
        # PortAudio thread: no allocation beyond slicing, no logging, no application calls.
        try:
            with self._lock:
                if playback.cancelled:
                    raise sd.CallbackAbort
                if self._paused:
                    outdata.fill(0.0)
                    return
                remaining = len(playback.samples) - playback.pos
                n = min(frames, remaining)
                if n > 0:
                    outdata[:n, 0] = playback.samples[playback.pos : playback.pos + n]
                    playback.pos += n
                if n < frames:
                    outdata[n:].fill(0.0)
                    raise sd.CallbackStop
        except (sd.CallbackAbort, sd.CallbackStop):
            raise
        except Exception as exc:
            playback.error = f"{type(exc).__name__}: {exc}"
            raise sd.CallbackAbort from exc

    def _finished(self, playback: _Playback) -> None:
        # PortAudio thread, after the stream has stopped (drained or aborted). The stream is
        # not closed here; start()/abort() close it from the loop thread.
        with self._lock:
            if playback.cancelled:
                return
            playback.cancelled = True
            if self._playback is playback:
                self._playback = None
                self._quiet_since = _monotonic()
            error = playback.error
            if error is None and playback.pos < len(playback.samples):
                error = "stream stopped before end"
        if error is not None:
            portaudio.mark_stale()  # the device may have vanished: re-scan before the next stream
            if self._on_failed is not None:
                self._on_failed(playback.chunk, error)
        elif self._on_done is not None:
            self._on_done(playback.chunk)
