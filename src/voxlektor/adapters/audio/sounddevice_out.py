"""AudioOutput on a PortAudio callback stream; pause = silence.

Threads: ``start/pause/resume/abort`` run on the loop thread, the stream callback and the
finished callback on the PortAudio thread. Shared state sits behind one lock; the callbacks
``on_done``/``on_failed`` are invoked as given (the runtime wraps them with
``call_soon_threadsafe``). Never reached by the CLI control path.

Stream lifetime: the stream of the last chunk stays open after it finished (PortAudio forbids
closing a stream from its own finished callback) and is closed by the next ``start()`` or by
``abort()``. The paused flag belongs to the output, not to a chunk: it survives ``abort()``
and ``start()`` (a start while paused writes silence until ``resume()``).
"""

import threading
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from voxlektor.domain.types import ChunkId


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
        self.abort()
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
            if stream is not None:
                stream.close()
            if self._on_failed is not None:
                self._on_failed(chunk, str(exc))

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def abort(self) -> None:
        """Stop the current chunk silently (no callback) and close the last stream."""
        with self._lock:
            playback = self._playback
            stream = self._stream
            self._playback = None
            self._stream = None
            if playback is not None:
                playback.cancelled = True
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
            error = playback.error
            if error is None and playback.pos < len(playback.samples):
                error = "stream stopped before end"
        if error is not None:
            if self._on_failed is not None:
                self._on_failed(playback.chunk, error)
        elif self._on_done is not None:
            self._on_done(playback.chunk)
