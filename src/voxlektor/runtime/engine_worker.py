"""EngineWorker: the one thread that calls the Engine SPI.

Jobs run strictly in submission order (FIFO) on a single daemon thread, so ``load``, every
``prepare_voice``/``synthesize`` and ``unload`` happen on the same thread and MLX arrays never
leave it. Voice prompts are kept here, by voice name, for the current engine load.

``on_result``/``on_error`` are called ON THE WORKER THREAD; the caller (``runtime.loop``) passes
callbacks that hop back onto the loop with ``call_soon_threadsafe``. A callback that raises is
logged and never stops the worker. The result of a ``Generate`` is ``(audio, gen_seconds)``:
the joined, time-stretched audio (then, when a ``level`` function is given, normalised and set to
the chunk's volume) and how long it took; every other job yields ``None``.

Epochs and cooperative cancellation: the loop thread calls ``cancel_epochs(epochs)`` for each
``CancelGeneration`` effect, naming exactly the epochs the Session has abandoned. The cancelled
set is a frozenset that the loop thread replaces (under the lock) and the worker thread only
reads. ``cancel_before(epoch)`` (a lower bound) remains for callers that cancel everything
older. A queued ``Generate`` whose chunk epoch is cancelled is skipped when its turn comes and
reports nothing: the Session has already forgotten it. A synthesis that is running is never
interrupted (OmniVoice cannot be), so at most one chunk is wasted.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import assert_never

from voxlektor.app.effects import Generate, LoadEngine, PrepareVoice, UnloadEngine
from voxlektor.app.ports import EngineProvider, VoiceRepository
from voxlektor.app.voice_prompts import get_or_build_prompt
from voxlektor.contracts.engine import AudioChunk, Engine, VoicePrompt
from voxlektor.domain.reading import Chunk
from voxlektor.domain.types import Epoch

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_log = logging.getLogger(__name__)

type _Job = LoadEngine | UnloadEngine | PrepareVoice | Generate
type _Result = tuple[AudioChunk, float] | None


@dataclass(frozen=True)
class _Task:
    job: _Job
    on_result: Callable[[_Result], None]
    on_error: Callable[[str], None]


class EngineWorker:
    def __init__(
        self,
        provider: EngineProvider,
        repo: VoiceRepository,
        stretch: Callable[[npt.NDArray[np.float32], float], npt.NDArray[np.float32]],
        level: Callable[[npt.NDArray[np.float32], int, int], npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self._provider = provider
        self._repo = repo
        self._stretch = stretch
        self._level = level
        self._engine: Engine | None = None  # touched only on the worker thread
        self._prompts: dict[str, VoicePrompt] = {}  # likewise
        self._min_epoch = 0  # written on the loop thread, read on the worker thread
        self._cancelled: frozenset[Epoch] = frozenset()  # likewise; swapped, never mutated
        self._tasks: queue.SimpleQueue[_Task | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="voxlektor-engine", daemon=True)
        self._thread.start()

    def submit(
        self,
        job: LoadEngine | UnloadEngine | PrepareVoice | Generate,
        on_result: Callable[[tuple[AudioChunk, float] | None], None],
        on_error: Callable[[str], None],
    ) -> None:
        """Queue ``job``; one of the callbacks is called when it has run (none if skipped)."""
        with self._lock:
            if self._closed:
                raise RuntimeError("the engine worker is closed")
            self._tasks.put(_Task(job, on_result, on_error))

    def cancel_before(self, epoch: Epoch) -> None:
        """Skip queued ``Generate`` jobs of chunks older than ``epoch`` (never lowers it)."""
        self._min_epoch = max(self._min_epoch, epoch)

    def cancel_epochs(self, epochs: Iterable[Epoch]) -> None:
        """Skip queued ``Generate`` jobs of chunks of ``epochs`` (added to those cancelled)."""
        with self._lock:  # one writer at a time; the worker reads the frozenset it swaps in
            self._cancelled = self._cancelled | frozenset(epochs)

    def close(self) -> None:
        """Run the jobs already queued, unload a loaded engine, then join the thread."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._tasks.put(None)
        self._thread.join()

    # ------------------------------------------------------------ worker thread

    def _run(self) -> None:
        while (task := self._tasks.get()) is not None:
            self._serve(task)
        if self._engine is not None:
            try:
                self._unload()
            except Exception:
                _log.exception("unloading the engine at close failed")

    def _serve(self, task: _Task) -> None:
        job = task.job
        if isinstance(job, Generate) and self._stale(job.chunk.epoch):
            _log.debug("skipped stale chunk %s (epoch %s)", job.chunk.id, job.chunk.epoch)
            return
        try:
            result = self._execute(job)
        except Exception as exc:  # any engine failure is reported, never raised
            _log.warning("engine job %s failed: %s", type(job).__name__, exc)
            self._call(task.on_error, str(exc) or type(exc).__name__)
        else:
            self._call(task.on_result, result)

    def _stale(self, epoch: Epoch) -> bool:
        return epoch < self._min_epoch or epoch in self._cancelled

    @staticmethod
    def _call[T](callback: Callable[[T], None], value: T) -> None:
        try:
            callback(value)
        except Exception:
            _log.exception("engine worker callback failed")

    def _execute(self, job: _Job) -> _Result:
        match job:
            case LoadEngine(engine_id):
                if self._engine is not None:
                    self._unload()
                engine = self._provider(engine_id)
                engine.load()
                self._engine = engine
            case UnloadEngine():
                if self._engine is not None:
                    self._unload()
            case PrepareVoice(ref, fresh):
                engine = self._loaded()
                self._prompts[ref.name] = get_or_build_prompt(engine, self._repo, ref, fresh=fresh)
            case Generate(chunk, voice):
                return self._generate(chunk, voice)
            case _:
                assert_never(job)
        return None

    def _unload(self) -> None:
        engine, self._engine = self._engine, None
        self._prompts.clear()
        if engine is not None:
            engine.unload()

    def _loaded(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("engine not loaded")
        return self._engine

    def _generate(self, chunk: Chunk, voice: str) -> tuple[AudioChunk, float]:
        import numpy as np  # numpy stays off the module import path

        engine = self._loaded()
        prompt = self._prompts.get(voice)
        if prompt is None:
            raise RuntimeError(f"no voice prompt for {voice!r}")
        started = time.perf_counter()
        pieces = [piece.samples for piece in engine.synthesize(chunk.text, chunk.lang, prompt)]
        joined = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        samples = self._stretch(joined.astype(np.float32, copy=False), chunk.rate)
        sample_rate = engine.spec.sample_rate
        if self._level is not None:
            samples = self._level(samples, sample_rate, chunk.volume)
        return AudioChunk(samples, sample_rate), time.perf_counter() - started
