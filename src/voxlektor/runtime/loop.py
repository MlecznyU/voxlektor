"""CoreLoop: the asyncio side of the daemon; the ONLY caller of the Session.

Threads:

* Loop thread (asyncio): ``handle`` (commands from the transport), ``execute`` (the effects
  the Session returns), wake timers and every Session callback.
* Engine thread (``EngineWorker``): runs engine jobs and calls back on its own thread.
* PortAudio thread (the output): reports the end or failure of a chunk on its own thread.

Every callback from another thread is posted with ``loop.call_soon_threadsafe`` and only then
reaches the Session, on the loop thread; a callback that comes after the loop has closed is
dropped. Generated audio goes into ``AudioBuffers`` on the loop thread immediately before
``Session.on_generated`` (the contract in ``app.effects.Generate``). Output callbacks posted
from the loop thread itself (``MemoryOutput`` completes synchronously) are likewise deferred, so
the Session is never re-entered while effects are being executed.

Output failures: an exception from the output (``start``/``pause``/``resume``/``abort``) is not
a bug of ours; it is logged and reported to the Session as a playback failure of the chunk last
started (``on_play_failed``, ignored when that chunk is no longer playing), so the daemon keeps
serving and the next ``speak`` works.

Epochs (cooperative cancellation): the loop does not guess the Session's epochs. The Session
names the epochs it abandons in ``CancelGeneration``, and the loop hands them to
``EngineWorker.cancel_epochs``, so queued generations of an abandoned reading are skipped.

Events: ``Emit`` is published to the ``EventHub`` given as ``events`` (on the loop thread, in
effect order); without a hub it is dropped.

Stopping: ``Exit`` (after ``Shutdown``) resolves ``run_until_stopped``, which cancels the wake
timers and aborts the output. The worker belongs to whoever built it: the owner closes it after
``run_until_stopped`` returns (``bootstrap.daemon.Daemon.run`` first removes the socket, then
closes the worker, which finishes the jobs already queued, such as the ``UnloadEngine`` before
``Exit``). An exception while running the Session or an effect is a bug: it is logged and
``run_until_stopped`` raises it. Once stopping, commands get an ``internal`` error and late
callbacks are ignored.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import assert_never

from voxlektor.app.commands import Command
from voxlektor.app.effects import (
    AbortPlayback,
    CancelGeneration,
    CancelWake,
    DropAudio,
    Effect,
    Emit,
    Exit,
    Generate,
    LoadEngine,
    PausePlayback,
    PlayCue,
    PrepareVoice,
    Reply,
    ResumePlayback,
    ScheduleWake,
    StartPlayback,
    UnloadEngine,
)
from voxlektor.app.ports import AudioOutput, Clock, CuePlayer
from voxlektor.app.session import Session
from voxlektor.contracts.engine import AudioChunk
from voxlektor.domain.errors import DomainError
from voxlektor.domain.types import ChunkId
from voxlektor.runtime.buffers import AudioBuffers
from voxlektor.runtime.engine_worker import EngineWorker
from voxlektor.runtime.events import EventHub

_log = logging.getLogger(__name__)

type _Call = Callable[[], list[Effect]]
type _Job = LoadEngine | UnloadEngine | PrepareVoice | Generate


class _Bound:
    """The event loop a CoreLoop serves and the future that resolves when it stops."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.stopped: asyncio.Future[None] = loop.create_future()


class CoreLoop:
    def __init__(  # noqa: PLR0913, PLR0917  # the frozen constructor names every collaborator
        self,
        session: Session,
        worker: EngineWorker,
        output: AudioOutput,
        buffers: AudioBuffers,
        clock: Clock,
        cues: CuePlayer,
        *,
        events: EventHub | None = None,
    ) -> None:
        self._session = session
        self._worker = worker
        self._output = output
        self._buffers = buffers
        self._clock = clock
        self._cues = cues
        self._events = events
        self._bound: _Bound | None = None
        self._stopping = False
        self._wakes: set[asyncio.TimerHandle] = set()
        self._last_started: ChunkId | None = None
        output.set_callbacks(self._on_play_done, self._on_play_failed)

    # ------------------------------------------------------------ loop thread

    async def handle(self, cmd: Command) -> Reply:
        """Run one command through the Session and carry out its effects."""
        self._bind()
        if self._stopping:
            error = DomainError("internal", "the service is shutting down")
            return Reply(ok=False, data={}, error=error)
        try:
            outcome = self._session.handle(cmd)
            self.execute(outcome.effects)
        except Exception as exc:
            self._crash(exc)
            raise
        if outcome.reply is None:
            raise RuntimeError(f"the Session gave no reply to {cmd!r}")
        return outcome.reply

    async def run_until_stopped(self) -> None:
        """Wait for ``Exit`` (or a crash); then cancel the timers and stop the sound.

        The worker is not closed here: its owner closes it afterwards.
        """
        bound = self._bind()
        try:
            await bound.stopped
        finally:
            self._stopping = True
            self._cancel_wakes()
            try:
                self._output.abort()
            except Exception:
                _log.exception("aborting the output at stop failed")

    def execute(self, effects: list[Effect]) -> None:
        """Carry out ``effects`` in order (loop thread only)."""
        bound = self._bind()
        for effect in effects:
            self._execute(effect, bound)

    def _bind(self) -> _Bound:
        loop = asyncio.get_running_loop()  # RuntimeError outside the loop thread
        if self._bound is None:
            self._bound = _Bound(loop)
        elif self._bound.loop is not loop:
            raise RuntimeError("this CoreLoop is bound to another event loop")
        return self._bound

    def _execute(self, effect: Effect, bound: _Bound) -> None:
        match effect:
            case LoadEngine() | UnloadEngine() | PrepareVoice() | Generate():
                self._submit(effect)
            case CancelGeneration(epochs):
                self._worker.cancel_epochs(epochs)
            case StartPlayback() | PausePlayback() | ResumePlayback() | AbortPlayback():
                self._control_output(self._output_action(effect))
            case DropAudio(chunks):
                self._buffers.drop(chunks)
            case ScheduleWake(at):
                self._schedule_wake(at, bound.loop)
            case CancelWake():
                self._cancel_wakes()
            case PlayCue(cue, volume):
                self._cues.play(cue, volume)
            case Emit(event, data):
                if self._events is not None:
                    self._events.publish(event, data)
            case Exit():
                self._stop(bound)
            case _:
                assert_never(effect)

    def _output_action(
        self, effect: StartPlayback | PausePlayback | ResumePlayback | AbortPlayback
    ) -> Callable[[], None]:
        match effect:
            case StartPlayback(chunk):
                audio = self._buffers.take(chunk)
                self._last_started = chunk
                return lambda: self._output.start(chunk, audio.samples, audio.sample_rate)
            case PausePlayback():
                return self._output.pause
            case ResumePlayback():
                return self._output.resume
            case AbortPlayback():
                return self._output.abort

    def _control_output(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:
            _log.exception("the audio output failed")
            chunk = self._last_started
            if chunk is not None:
                msg = f"audio output failed: {exc}"
                self._post(lambda: self._session.on_play_failed(chunk, msg))

    def _deliver(self, call: _Call) -> None:
        """Run a Session callback and its effects; ignored once stopping."""
        if self._stopping:
            return
        try:
            self.execute(call())
        except Exception as exc:
            self._crash(exc)

    def _stop(self, bound: _Bound) -> None:
        self._stopping = True
        if not bound.stopped.done():
            bound.stopped.set_result(None)

    def _crash(self, exc: Exception) -> None:
        _log.error("the core loop failed; stopping", exc_info=exc)
        self._stopping = True
        bound = self._bound
        if bound is not None and not bound.stopped.done():
            bound.stopped.set_exception(exc)

    # ------------------------------------------------------------ wakes

    def _schedule_wake(self, at: float, loop: asyncio.AbstractEventLoop) -> None:
        delay = max(0.0, at - self._clock.now())  # a wake in the past fires right away

        def fire() -> None:
            self._wakes.discard(handle)
            self._deliver(self._session.on_wake)

        handle = loop.call_at(loop.time() + delay, fire)
        self._wakes.add(handle)

    def _cancel_wakes(self) -> None:
        for handle in self._wakes:
            handle.cancel()
        self._wakes.clear()

    # ------------------------------------------------------------ engine jobs

    def _submit(self, job: _Job) -> None:
        self._worker.submit(
            job,
            lambda result: self._post(lambda: self._job_done(job, result)),
            lambda msg: self._post(lambda: self._job_failed(job, msg)),
        )

    def _job_done(self, job: _Job, result: tuple[AudioChunk, float] | None) -> list[Effect]:
        match job:
            case LoadEngine():
                return self._session.on_loaded()
            case UnloadEngine():
                return self._session.on_unloaded()
            case PrepareVoice(ref):
                return self._session.on_prompt_ready(ref.name)
            case Generate(chunk):
                if result is None:
                    raise TypeError("Generate returned no audio")
                audio, gen_seconds = result
                self._buffers.put(chunk.id, audio)  # stored right before on_generated
                seconds = len(audio.samples) / audio.sample_rate
                return self._session.on_generated(chunk.epoch, chunk.id, seconds, gen_seconds)
            case _:
                assert_never(job)

    def _job_failed(self, job: _Job, msg: str) -> list[Effect]:
        match job:
            case LoadEngine():
                return self._session.on_load_failed(msg)
            case UnloadEngine():
                # The worker has dropped the engine either way, so it counts as unloaded.
                _log.warning("engine unload failed: %s", msg)
                return self._session.on_unloaded()
            case PrepareVoice(ref):
                return self._session.on_prompt_failed(ref.name, msg)
            case Generate(chunk):
                return self._session.on_gen_failed(chunk.epoch, chunk.id, msg)
            case _:
                assert_never(job)

    # ------------------------------------------------------------ any thread

    def _post(self, call: _Call) -> None:
        """Hand ``call`` to the loop thread; safe from any thread."""
        bound = self._bound
        if bound is None:
            _log.warning("a callback arrived before the loop started; dropped")
            return
        try:
            bound.loop.call_soon_threadsafe(self._deliver, call)
        except RuntimeError:  # the loop is closed: the service has stopped
            _log.debug("a callback arrived after the loop closed; dropped")

    def _on_play_done(self, chunk: ChunkId) -> None:
        self._post(lambda: self._session.on_play_done(chunk))

    def _on_play_failed(self, chunk: ChunkId, msg: str) -> None:
        self._post(lambda: self._session.on_play_failed(chunk, msg))
