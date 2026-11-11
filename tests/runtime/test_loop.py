"""CoreLoop with the real engine worker thread, FakeEngine and MemoryOutput.

No sleeps and no polling: ``_Watched`` resolves futures from inside the Session callbacks, so a
test wakes up exactly when the state it waits for is reached (or times out).
"""

import asyncio
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from voxlektor.adapters.audio.memory_out import MemoryCuePlayer, MemoryOutput
from voxlektor.app.commands import (
    Pause,
    Resume,
    Shutdown,
    Skip,
    Speak,
    Status,
    StopReading,
    Unload,
)
from voxlektor.app.effects import CancelWake, Cue, Effect, ScheduleWake, UnloadEngine
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import AudioChunk, Engine, VoicePrompt, VoiceRef
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId, EngineId, Epoch
from voxlektor.runtime.buffers import AudioBuffers
from voxlektor.runtime.engine_worker import EngineWorker
from voxlektor.runtime.loop import CoreLoop
from voxlektor.testing.fakes import FakeEngine, InMemoryVoiceRepository

TIMEOUT = 10.0
VOICE = "v0-default"
REF = VoiceRef(VOICE, Path("/voices/v0-default/ref.wav"), "reference transcript")
SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the river bank while the morning sun "
    "rises slowly above the distant hills and the birds sing their songs to the waking world "
    "around them all."
)
THREE_CHUNKS = " ".join([SENTENCE] * 3)


class _Monotonic:
    def now(self) -> float:
        return time.monotonic()


class _Watched(Session):
    """A Session that wakes waiting tests after every callback; it also records threads."""

    def __init__(self, settings: Settings, catalog: VoiceCatalog) -> None:
        super().__init__(settings, catalog, _Monotonic())
        self._waiters: list[tuple[Callable[[], bool], asyncio.Future[None]]] = []
        self.callback_threads: set[int] = set()
        self.wakes = 0
        self.buffered_at_generated: list[int] = []
        self.buffers: AudioBuffers | None = None

    def status(self) -> dict[str, Any]:
        reply = self.handle(Status()).reply
        assert reply is not None
        return dict(reply.data)

    async def until(self, predicate: Callable[[], bool]) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append((predicate, future))
        self._check()
        await asyncio.wait_for(future, TIMEOUT)

    def _check(self) -> None:
        for predicate, future in list(self._waiters):
            if not future.done() and predicate():
                future.set_result(None)
                self._waiters.remove((predicate, future))

    def _after(self, effects: list[Effect]) -> list[Effect]:
        self.callback_threads.add(threading.get_ident())
        # Check once the returned effects have run (the loop runs them before this callback).
        asyncio.get_running_loop().call_soon(self._check)
        return effects

    def on_loaded(self) -> list[Effect]:
        return self._after(super().on_loaded())

    def on_load_failed(self, msg: str) -> list[Effect]:
        return self._after(super().on_load_failed(msg))

    def on_prompt_ready(self, voice: str) -> list[Effect]:
        return self._after(super().on_prompt_ready(voice))

    def on_prompt_failed(self, voice: str, msg: str) -> list[Effect]:
        return self._after(super().on_prompt_failed(voice, msg))

    def on_generated(
        self, epoch: Epoch, chunk: ChunkId, audio_seconds: float, gen_seconds: float
    ) -> list[Effect]:
        if self.buffers is not None:
            self.buffered_at_generated.append(len(self.buffers))
        return self._after(super().on_generated(epoch, chunk, audio_seconds, gen_seconds))

    def on_gen_failed(self, epoch: Epoch, chunk: ChunkId, msg: str) -> list[Effect]:
        return self._after(super().on_gen_failed(epoch, chunk, msg))

    def on_play_done(self, chunk: ChunkId) -> list[Effect]:
        return self._after(super().on_play_done(chunk))

    def on_play_failed(self, chunk: ChunkId, msg: str) -> list[Effect]:
        return self._after(super().on_play_failed(chunk, msg))

    def on_wake(self) -> list[Effect]:
        self.wakes += 1
        return self._after(super().on_wake())

    def on_unloaded(self) -> list[Effect]:
        return self._after(super().on_unloaded())


class _PeakBuffers(AudioBuffers):
    def __init__(self, max_items: int) -> None:
        super().__init__(max_items)
        self.peak = 0

    def put(self, chunk_id: ChunkId, audio: AudioChunk) -> None:
        super().put(chunk_id, audio)
        self.peak = max(self.peak, len(self))


class _Provider:
    def __init__(self, make: Callable[[], FakeEngine]) -> None:
        self._make = make
        self.engines: list[FakeEngine] = []

    def __call__(self, engine_id: str) -> Engine:
        if engine_id != "fake":
            raise ValueError(f"unknown engine {engine_id!r}")
        self.engines.append(self._make())
        return self.engines[-1]


@dataclass
class _Rig:
    core: CoreLoop
    session: _Watched
    output: MemoryOutput
    cues: MemoryCuePlayer
    buffers: AudioBuffers
    worker: EngineWorker
    provider: _Provider
    loop_thread: list[int] = field(default_factory=list[int])

    def status(self) -> dict[str, Any]:
        return self.session.status()

    def worker_alive(self) -> bool:
        try:
            self.worker.submit(UnloadEngine(), lambda result: None, lambda msg: None)
        except RuntimeError:  # closed (and joined)
            return False
        return True

    def idle(self, played: int) -> Callable[[], bool]:
        def check() -> bool:
            status = self.status()
            done = not status["queued"] and not status["playing"]
            return done and len(self.output.started) == played

        return check


def _rig(
    *,
    settings: Settings | None = None,
    make: Callable[[], FakeEngine] = FakeEngine,
    buffers: AudioBuffers | None = None,
    auto_complete: bool = True,
    output: MemoryOutput | None = None,
) -> _Rig:
    settings = settings or Settings(engine=EngineId("fake"))
    repo = InMemoryVoiceRepository([REF])
    session = _Watched(settings, VoiceCatalog(repo))
    provider = _Provider(make)
    worker = EngineWorker(provider, repo, lambda samples, rate: samples)
    output = MemoryOutput(auto_complete=auto_complete) if output is None else output
    cues = MemoryCuePlayer()
    buffers = AudioBuffers() if buffers is None else buffers  # empty buffers are falsy
    session.buffers = buffers
    core = CoreLoop(session, worker, output, buffers, _Monotonic(), cues)
    return _Rig(core, session, output, cues, buffers, worker, provider)


@pytest.fixture
def no_leaked_threads() -> Iterator[None]:
    before = set(threading.enumerate())
    yield
    leaked = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    assert not leaked, leaked


type _Scenario = Callable[[_Rig], Coroutine[Any, Any, None]]


def _run(rig: _Rig, scenario: _Scenario) -> None:
    """Run ``scenario`` next to ``run_until_stopped``; shut down cleanly at the end."""

    async def main() -> None:
        rig.loop_thread.append(threading.get_ident())
        runner = asyncio.create_task(rig.core.run_until_stopped())
        try:
            await scenario(rig)
        finally:
            if not runner.done():
                await rig.core.handle(Shutdown())
            try:
                await asyncio.wait_for(runner, TIMEOUT)
            finally:
                await asyncio.to_thread(rig.worker.close)  # the owner closes the worker

    asyncio.run(main())


def _speak(text: str, *, now: bool = False) -> Speak:
    return Speak(text, None, "en", None, now)


# ------------------------------------------------------------------ reading


@pytest.mark.usefixtures("no_leaked_threads")
def test_three_chunk_speak_plays_three_chunks_and_ends_idle() -> None:
    rig = _rig()

    async def scenario(rig: _Rig) -> None:
        reply = await rig.core.handle(_speak(THREE_CHUNKS))
        assert reply.ok
        assert reply.data["chunks"] == 3
        await rig.session.until(rig.idle(played=3))
        status = rig.status()
        assert status["state"] == "warm"
        assert status["queued"] == 0
        assert status["playing"] is False
        assert status["chunks"] == 3

    _run(rig, scenario)
    assert len(set(rig.output.started)) == 3
    assert rig.session.callback_threads == set(rig.loop_thread)
    assert rig.session.buffered_at_generated == [1, 1, 1]  # stored before on_generated
    assert len(rig.buffers) == 0
    assert rig.cues.played == []


@pytest.mark.usefixtures("no_leaked_threads")
def test_buffers_stay_within_the_ahead_window_for_a_long_reading() -> None:
    """Review Focus 3, threaded: the window fills to ahead (2) and never goes beyond it.

    Playback is slow on purpose: each chunk is completed (from another thread) only once the
    worker has filled the window, i.e. 2 chunks are buffered (fewer only for the last chunks).
    No stale result can arrive in this reading (no stop, skip, back or --now), so even the
    transient "+1" of a stale result dropped right away cannot happen: ``max_items=2`` makes
    any third buffered chunk an immediate failure.
    """
    buffers = _PeakBuffers(max_items=2)
    rig = _rig(
        make=lambda: FakeEngine(seconds_per_char=0.0005), buffers=buffers, auto_complete=False
    )
    text = (SENTENCE + " ") * (100_000 // (len(SENTENCE) + 1) + 1)
    assert len(text) >= 100_000

    async def scenario(rig: _Rig) -> None:
        reply = await rig.core.handle(_speak(text))
        chunks = reply.data["chunks"]
        assert isinstance(chunks, int)
        assert chunks > 300
        for played in range(1, chunks + 1):
            window = min(2, chunks - played)  # chunks after the playing one, at most 2

            def filled(played: int = played, window: int = window) -> bool:
                playing = len(rig.output.started) == played and rig.status()["playing"]
                return playing and len(rig.buffers) == window

            await rig.session.until(filled)
            await asyncio.to_thread(rig.output.complete, rig.output.started[-1])
        await rig.session.until(rig.idle(played=chunks))

    _run(rig, scenario)
    assert buffers.peak == 2
    assert len(buffers) == 0


@pytest.mark.usefixtures("no_leaked_threads")
def test_output_callbacks_from_another_thread_reach_the_session_on_the_loop() -> None:
    rig = _rig(auto_complete=False)

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(lambda: rig.status()["playing"] is True)
        chunk = rig.output.started[0]
        other = threading.Thread(target=rig.output.complete, args=(chunk,))
        other.start()
        other.join()
        await rig.session.until(rig.idle(played=1))
        assert rig.status()["playing"] is False

    _run(rig, scenario)
    assert rig.session.callback_threads == set(rig.loop_thread)


@pytest.mark.usefixtures("no_leaked_threads")
def test_pause_resume_skip_and_stop_reach_the_output_and_the_buffers() -> None:
    rig = _rig(auto_complete=False)

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(THREE_CHUNKS))
        await rig.session.until(lambda: len(rig.output.started) == 1 and len(rig.buffers) == 2)
        await rig.core.handle(Pause())
        assert rig.output.paused
        await rig.core.handle(Resume())
        assert not rig.output.paused
        await rig.core.handle(Skip())  # aborts the playing chunk, the next one starts
        assert rig.output.aborted == 1
        assert len(rig.output.started) == 2
        await rig.core.handle(StopReading())  # aborts again and drops the buffered audio
        assert rig.output.aborted == 2
        assert len(rig.buffers) == 0
        assert rig.status()["queued"] == 0

    _run(rig, scenario)


@pytest.mark.usefixtures("no_leaked_threads")
def test_play_failure_stops_the_reading_with_the_stopped_cue() -> None:
    rig = _rig(auto_complete=False)

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(THREE_CHUNKS))
        await rig.session.until(lambda: len(rig.output.started) == 1)
        rig.output.fail(rig.output.started[0], "device lost")
        await rig.session.until(lambda: rig.status()["last_error"] == "device lost")

    _run(rig, scenario)
    assert rig.cues.played == [Cue.STOPPED]
    assert len(rig.buffers) == 0


# ------------------------------------------------------------------ engine failures


@pytest.mark.usefixtures("no_leaked_threads")
def test_generation_failures_retry_with_cues_then_stop() -> None:
    rig = _rig(make=lambda: FakeEngine(fail_on="BOOM"))

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak("This sentence goes BOOM right away."))
        await rig.session.until(lambda: rig.status()["last_error"] is not None)
        assert rig.status()["last_error"] == "fake engine failure on 'BOOM'"

    _run(rig, scenario)
    assert rig.cues.played == [Cue.RETRY, Cue.RETRY, Cue.STOPPED]
    assert len(rig.provider.engines) == 2  # the second failure reloaded the engine


@pytest.mark.usefixtures("no_leaked_threads")
def test_prompt_failure_is_retried_fresh_and_the_reading_plays() -> None:
    rig = _rig(make=lambda: FakeEngine(fail_prepare=1))

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(rig.idle(played=1))

    _run(rig, scenario)
    assert rig.cues.played == [Cue.RETRY]
    assert rig.session.status()["last_error"] is None


@pytest.mark.usefixtures("no_leaked_threads")
def test_load_failure_is_reported() -> None:
    rig = _rig(settings=Settings(engine=EngineId("nope")))

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(lambda: rig.status()["last_error"] is not None)
        status = rig.status()
        assert status["last_error"] == "unknown engine 'nope'"
        assert status["state"] == "cold"

    _run(rig, scenario)
    assert rig.cues.played == [Cue.STOPPED]


@pytest.mark.usefixtures("no_leaked_threads")
def test_an_unload_failure_still_ends_cold() -> None:
    class _BadUnload(FakeEngine):
        def unload(self) -> None:
            super().unload()
            raise RuntimeError("unload went wrong")

    rig = _rig(make=_BadUnload)

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(rig.idle(played=1))
        await rig.core.handle(Unload())
        await rig.session.until(lambda: rig.status()["state"] == "cold")

    _run(rig, scenario)


# ------------------------------------------------------------------ wakes and keep-warm


@pytest.mark.usefixtures("no_leaked_threads")
def test_wake_fires_through_call_at_and_cancel_wake_cancels_it() -> None:
    rig = _rig()

    async def scenario(rig: _Rig) -> None:
        now = time.monotonic()
        rig.core.execute([ScheduleWake(now + 0.05), CancelWake(), ScheduleWake(now + 0.1)])
        await rig.session.until(lambda: rig.session.wakes >= 1)
        assert time.monotonic() - now >= 0.095  # the cancelled 0.05 s wake never fired
        assert rig.session.wakes == 1
        started = time.monotonic()
        rig.core.execute([ScheduleWake(started - 5.0)])  # in the past: fires right away
        await rig.session.until(lambda: rig.session.wakes == 2)
        rig.core.execute([ScheduleWake(started + 0.05)])
        await rig.session.until(lambda: rig.session.wakes == 3)
        assert time.monotonic() - started >= 0.045

    _run(rig, scenario)


@pytest.mark.usefixtures("no_leaked_threads")
def test_idle_engine_unloads_when_the_idle_timer_fires() -> None:
    # Settings.idle_timeout_s is whole seconds, so 1 s is the shortest real idle timeout.
    rig = _rig(settings=Settings(engine=EngineId("fake"), idle_timeout_s=1))

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(rig.idle(played=1))
        idle_at = time.monotonic()
        assert rig.status()["state"] == "warm"
        await rig.session.until(lambda: rig.status()["state"] == "cold")
        assert time.monotonic() - idle_at >= 0.9
        assert rig.session.wakes >= 1

    _run(rig, scenario)
    assert rig.provider.engines[0].unload_calls == 1


# ------------------------------------------------------------------ epochs (R31)


class _GatedEngine(FakeEngine):
    """The first synthesis blocks until ``gate`` is set; ``entered`` says it has started."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        self.entered.set()
        assert self.gate.wait(TIMEOUT)
        return super().synthesize(text, lang, prompt)


def _texts(engine: FakeEngine) -> list[str]:
    return [text for text, _ in engine.synth_calls]


@pytest.mark.usefixtures("no_leaked_threads")
@pytest.mark.parametrize("how", ["stop", "now"])
def test_queued_generates_of_an_abandoned_reading_are_not_synthesized(how: str) -> None:
    # ahead = 3: chunk 0 is being synthesized (blocked), chunks 1 and 2 wait in the queue.
    rig = _rig(settings=Settings(engine=EngineId("fake"), ahead=3), make=_GatedEngine)
    sentences = [f"Sentence number {n} of the old reading. " + SENTENCE for n in range(3)]
    fresh = "A completely new sentence that replaces the old reading right now."

    async def scenario(rig: _Rig) -> None:
        assert (await rig.core.handle(_speak(" ".join(sentences)))).data["chunks"] == 3
        await rig.session.until(lambda: bool(rig.provider.engines))
        engine = rig.provider.engines[0]
        assert isinstance(engine, _GatedEngine)
        assert await asyncio.to_thread(engine.entered.wait, TIMEOUT)
        if how == "stop":
            await rig.core.handle(StopReading())
        await rig.core.handle(_speak(fresh, now=how == "now"))
        engine.gate.set()
        await rig.session.until(rig.idle(played=1))

    _run(rig, scenario)
    texts = _texts(rig.provider.engines[0])
    assert len(texts) == 2  # the running chunk 0 (wasted) and the new reading
    assert texts[0].startswith("Sentence number 0")
    assert texts[1] == fresh
    assert rig.status()["last_error"] is None


# ------------------------------------------------------------------ output failures


@pytest.mark.usefixtures("no_leaked_threads")
def test_an_output_whose_abort_raises_does_not_stop_the_daemon() -> None:
    class _BadAbort(MemoryOutput):
        def abort(self) -> None:
            super().abort()
            raise RuntimeError("abort exploded")

    rig = _rig(output=_BadAbort(auto_complete=False))

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(THREE_CHUNKS))
        await rig.session.until(lambda: len(rig.output.started) == 1)
        assert (await rig.core.handle(StopReading())).ok
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(lambda: len(rig.output.started) == 2)
        rig.output.complete(rig.output.started[-1])
        await rig.session.until(rig.idle(played=2))

    _run(rig, scenario)  # Shutdown's abort raises too; the loop still stops cleanly


@pytest.mark.usefixtures("no_leaked_threads")
def test_an_output_whose_start_fails_is_a_playback_failure() -> None:
    class _FailsOnce(MemoryOutput):
        def __init__(self) -> None:
            super().__init__(auto_complete=True)
            self.failures = 1

        def start(self, chunk: ChunkId, samples: object, sample_rate: int) -> None:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("no device")
            super().start(chunk, samples, sample_rate)

    rig = _rig(output=_FailsOnce())

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(lambda: rig.status()["last_error"] is not None)
        assert rig.status()["last_error"] == "audio output failed: no device"
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(rig.idle(played=1))

    _run(rig, scenario)
    assert rig.cues.played == [Cue.STOPPED]


# ------------------------------------------------------------------ stopping


@pytest.mark.usefixtures("no_leaked_threads")
def test_shutdown_unloads_before_the_worker_stops() -> None:
    """``_run`` closes the worker after ``run_until_stopped``; the queued unload runs first."""
    rig = _rig(auto_complete=False)

    async def scenario(rig: _Rig) -> None:
        await rig.core.handle(_speak(THREE_CHUNKS))
        await rig.session.until(lambda: len(rig.output.started) == 1)
        reply = await rig.core.handle(Shutdown())
        assert reply.ok
        late = await rig.core.handle(Status())
        assert not late.ok
        assert late.error is not None
        assert late.error.code == "internal"
        assert "shutting down" in late.error.message

    _run(rig, scenario)
    assert rig.provider.engines[0].unload_calls == 1
    assert not rig.worker_alive()


def test_a_runtime_bug_stops_the_loop_with_the_error() -> None:
    # Chunk 1 keeps playing, so chunks 2 and 3 (ahead = 2) overflow a one-item buffer.
    rig = _rig(buffers=AudioBuffers(max_items=1), auto_complete=False)

    async def main() -> None:
        runner = asyncio.create_task(rig.core.run_until_stopped())
        await rig.core.handle(_speak(THREE_CHUNKS))
        with pytest.raises(RuntimeError, match="more than 1"):
            await asyncio.wait_for(runner, TIMEOUT)
        late = await rig.core.handle(Status())
        assert not late.ok

    try:
        asyncio.run(main())
    finally:
        rig.worker.close()
    assert not rig.worker_alive()


def test_callbacks_after_the_loop_closed_are_dropped() -> None:
    rig = _rig(auto_complete=False)

    async def main() -> None:
        runner = asyncio.create_task(rig.core.run_until_stopped())
        await rig.core.handle(_speak(SENTENCE))
        await rig.session.until(lambda: len(rig.output.started) == 1)
        await rig.core.handle(Shutdown())
        await runner

    try:
        asyncio.run(main())
    finally:
        rig.worker.close()
    rig.output.complete(rig.output.started[0])  # the loop is closed: nothing happens


def test_callbacks_before_the_loop_started_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    rig = _rig(auto_complete=False)
    try:
        rig.output.complete(ChunkId(0))  # no loop yet: nothing reaches the Session
    finally:
        rig.worker.close()
    assert "before the loop started" in caplog.text


def test_execute_needs_a_running_loop() -> None:
    rig = _rig()
    try:
        with pytest.raises(RuntimeError):
            rig.core.execute([CancelWake()])
    finally:
        rig.worker.close()


def test_a_core_loop_serves_one_event_loop_only() -> None:
    rig = _rig()

    async def touch() -> None:
        await rig.core.handle(Status())

    try:
        asyncio.run(touch())
        with pytest.raises(RuntimeError, match="another event loop"):
            asyncio.run(touch())
    finally:
        rig.worker.close()
