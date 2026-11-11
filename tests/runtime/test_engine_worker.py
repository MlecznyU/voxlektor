import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from voxlektor.app.effects import Generate, LoadEngine, PrepareVoice, UnloadEngine
from voxlektor.contracts.engine import AudioChunk, Engine, VoicePrompt, VoiceRef
from voxlektor.contracts.voicefmt import prompt_cache_name
from voxlektor.domain.reading import Chunk
from voxlektor.domain.types import ChunkId, EngineId, Epoch
from voxlektor.runtime.engine_worker import EngineWorker
from voxlektor.testing.fakes import FakeEngine, InMemoryVoiceRepository

REF = VoiceRef("v1", Path("/voices/v1/ref.wav"), "reference transcript")
KEY = prompt_cache_name("fake", "fake0001", 1)
TIMEOUT = 5.0
LOAD = LoadEngine(EngineId("fake"))
PREPARE = PrepareVoice(REF)

type Samples = npt.NDArray[np.float32]
type Job = LoadEngine | UnloadEngine | PrepareVoice | Generate
type JobResult = tuple[AudioChunk, float] | None
type _Event = tuple[str, int, JobResult | str]


def _gen(cid: int, text: str = "hello there", rate: float = 1.0) -> Generate:
    return Generate(Chunk(ChunkId(cid), text, "en", rate, Epoch(0)), "v1")


class _Recorder:
    """Collects (label, thread id, result or error message) from the worker callbacks."""

    def __init__(self) -> None:
        self.events: queue.Queue[_Event] = queue.Queue()

    def submit(self, worker: EngineWorker, label: str, job: Job) -> None:
        worker.submit(job, self._ok(label), self._err(label))

    def _ok(self, label: str) -> Callable[[JobResult], None]:
        return lambda result: self.events.put((label, threading.get_ident(), result))

    def _err(self, label: str) -> Callable[[str], None]:
        return lambda msg: self.events.put((f"{label}!", threading.get_ident(), msg))

    def next(self) -> _Event:
        return self.events.get(timeout=TIMEOUT)

    def take(self, n: int) -> list[_Event]:
        return [self.next() for _ in range(n)]


class _Stretch:
    """Records the rates; rate 2.0 halves the samples so the call is visible in the result."""

    def __init__(self) -> None:
        self.rates: list[float] = []
        self.threads: set[int] = set()

    def __call__(self, samples: Samples, rate: float) -> Samples:
        self.rates.append(rate)
        self.threads.add(threading.get_ident())
        return samples[: len(samples) // 2] if rate == 2.0 else samples


class _Provider:
    def __init__(self, make: Callable[[], FakeEngine] = FakeEngine) -> None:
        self._make = make
        self.engines: list[FakeEngine] = []

    def __call__(self, engine_id: str) -> Engine:
        if engine_id != "fake":
            raise ValueError(f"unknown engine {engine_id!r}")
        engine = self._make()
        self.engines.append(engine)
        return engine


@pytest.fixture
def provider() -> _Provider:
    return _Provider()


@pytest.fixture
def repo() -> InMemoryVoiceRepository:
    return InMemoryVoiceRepository([REF])


@pytest.fixture
def stretch() -> _Stretch:
    return _Stretch()


@pytest.fixture
def worker(
    provider: _Provider, repo: InMemoryVoiceRepository, stretch: _Stretch
) -> Iterator[EngineWorker]:
    w = EngineWorker(provider, repo, stretch)
    yield w
    w.close()


def test_jobs_run_in_order_on_one_worker_thread(
    worker: EngineWorker, stretch: _Stretch, provider: _Provider
) -> None:
    rec = _Recorder()
    jobs: list[tuple[str, Job]] = [
        ("load", LOAD),
        ("prepare", PREPARE),
        ("gen0", _gen(0)),
        ("gen1", _gen(1)),
        ("gen2", _gen(2)),
        ("unload", UnloadEngine()),
    ]
    for label, job in jobs:
        rec.submit(worker, label, job)
    events = rec.take(len(jobs))
    assert [label for label, _, _ in events] == [label for label, _ in jobs]
    threads = {ident for _, ident, _ in events}
    assert len(threads) == 1
    assert threading.get_ident() not in threads
    assert stretch.threads == threads
    assert provider.engines[0].unload_calls == 1


def test_generate_joins_pieces_and_stretches_by_the_chunk_rate(
    worker: EngineWorker, stretch: _Stretch
) -> None:
    rec = _Recorder()
    text = "x" * 30  # 30 chars * 0.05 s = 1.5 s: two pieces from FakeEngine
    for label, job in [("load", LOAD), ("prepare", PREPARE)]:
        rec.submit(worker, label, job)
    rec.submit(worker, "plain", _gen(0, text))
    rec.submit(worker, "fast", _gen(1, text, rate=2.0))
    results = [result for _, _, result in rec.take(4)][2:]
    plain, fast = results
    assert isinstance(plain, tuple)
    assert isinstance(fast, tuple)
    (audio, gen_seconds), (fast_audio, _) = plain, fast
    assert audio.sample_rate == 24000
    assert audio.samples.dtype == np.float32
    assert len(audio.samples) == 36000
    assert len(fast_audio.samples) == 18000
    assert gen_seconds >= 0.0
    assert stretch.rates == [1.0, 2.0]


class _GatedEngine(FakeEngine):
    """Blocks the first synthesis until ``gate`` is set; ``entered`` says it started."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        self.entered.set()
        assert self.gate.wait(TIMEOUT)
        return super().synthesize(text, lang, prompt)


def test_queued_generates_older_than_the_epoch_are_skipped_silently(
    repo: InMemoryVoiceRepository,
) -> None:
    provider = _Provider(_GatedEngine)
    worker = EngineWorker(provider, repo, _Stretch())
    rec = _Recorder()

    def gen(cid: int, epoch: int) -> Generate:
        return Generate(Chunk(ChunkId(cid), f"chunk {cid}", "en", 1.0, Epoch(epoch)), "v1")

    try:
        for label, job in [("load", LOAD), ("prepare", PREPARE), ("g0", gen(0, 0))]:
            rec.submit(worker, label, job)
        rec.take(2)
        engine = provider.engines[0]
        assert isinstance(engine, _GatedEngine)
        assert engine.entered.wait(TIMEOUT)  # g0 is running and cannot be interrupted
        rec.submit(worker, "g1", gen(1, 0))
        rec.submit(worker, "g2", gen(2, 0))
        rec.submit(worker, "g3", gen(3, 1))
        worker.cancel_before(Epoch(1))
        worker.cancel_before(Epoch(0))  # never lowers it
        engine.gate.set()
        assert [label for label, _, _ in rec.take(2)] == ["g0", "g3"]
    finally:
        worker.close()
    assert [text for text, _ in engine.synth_calls] == ["chunk 0", "chunk 3"]
    assert rec.events.empty()  # the skipped jobs reported nothing


def test_errors_go_to_on_error_and_the_worker_keeps_going(worker: EngineWorker) -> None:
    rec = _Recorder()
    rec.submit(worker, "early", _gen(0))  # no engine yet
    rec.submit(worker, "bad", LoadEngine(EngineId("nope")))
    rec.submit(worker, "load", LOAD)
    rec.submit(worker, "noprompt", _gen(1))  # loaded, but no prompt for v1
    events = rec.take(4)
    assert [label for label, _, _ in events] == ["early!", "bad!", "load", "noprompt!"]
    assert events[0][2] == "engine not loaded"
    assert events[1][2] == "unknown engine 'nope'"
    assert events[3][2] == "no voice prompt for 'v1'"


def test_engine_failures_carry_the_engine_message(repo: InMemoryVoiceRepository) -> None:
    provider = _Provider(lambda: FakeEngine(fail_on="BOOM", fail_prepare=1))
    worker = EngineWorker(provider, repo, _Stretch())
    rec = _Recorder()
    try:
        for label, job in [("load", LOAD), ("p1", PREPARE), ("p2", PREPARE)]:
            rec.submit(worker, label, job)
        rec.submit(worker, "boom", _gen(0, "a BOOM here"))
        events = rec.take(4)
    finally:
        worker.close()
    assert [label for label, _, _ in events] == ["load", "p1!", "p2", "boom!"]
    assert events[1][2] == "fake prepare failure"
    assert events[3][2] == "fake engine failure on 'BOOM'"


def test_a_message_less_exception_is_named_by_its_type(repo: InMemoryVoiceRepository) -> None:
    class _Silent(FakeEngine):
        def load(self) -> None:
            raise RuntimeError

    worker = EngineWorker(_Provider(_Silent), repo, _Stretch())
    rec = _Recorder()
    try:
        rec.submit(worker, "load", LOAD)
        assert rec.next()[2] == "RuntimeError"
    finally:
        worker.close()


def test_fresh_prepare_bypasses_the_prompt_cache(
    worker: EngineWorker, repo: InMemoryVoiceRepository, provider: _Provider
) -> None:
    rec = _Recorder()
    rec.submit(worker, "load", LOAD)
    rec.submit(worker, "cached", PREPARE)
    rec.submit(worker, "again", PREPARE)
    rec.submit(worker, "fresh", PrepareVoice(REF, fresh=True))
    assert [label for label, _, _ in rec.take(4)] == ["load", "cached", "again", "fresh"]
    assert provider.engines[0].prepare_calls == 2  # the first build and the fresh one
    assert ("v1", KEY) in repo.cache


def test_unload_forgets_prompts_and_a_new_load_needs_them_again(
    worker: EngineWorker, provider: _Provider
) -> None:
    rec = _Recorder()
    for label, job in [
        ("load", LOAD),
        ("prepare", PREPARE),
        ("unload", UnloadEngine()),
        ("unload-again", UnloadEngine()),  # nothing loaded: a no-op
        ("reload", LOAD),
        ("gen", _gen(0)),
    ]:
        rec.submit(worker, label, job)
    events = rec.take(6)
    assert events[-1][0] == "gen!"
    assert events[-1][2] == "no voice prompt for 'v1'"
    assert len(provider.engines) == 2


def test_a_second_load_replaces_the_loaded_engine(
    worker: EngineWorker, provider: _Provider
) -> None:
    rec = _Recorder()
    rec.submit(worker, "load", LOAD)
    rec.submit(worker, "load2", LOAD)
    rec.take(2)
    first, second = provider.engines
    assert first.unload_calls == 1
    assert second.unload_calls == 0


def test_a_raising_callback_does_not_stop_the_worker(worker: EngineWorker) -> None:
    rec = _Recorder()

    def explode(result: JobResult) -> None:
        raise RuntimeError("callback bug")

    worker.submit(LOAD, explode, lambda msg: None)
    rec.submit(worker, "unload", UnloadEngine())
    assert rec.next()[0] == "unload"


def test_close_unloads_on_the_worker_thread_and_joins(
    repo: InMemoryVoiceRepository, provider: _Provider
) -> None:
    worker = EngineWorker(provider, repo, _Stretch())
    rec = _Recorder()
    rec.submit(worker, "load", LOAD)
    load_thread = rec.next()[1]
    unload_threads: list[int] = []
    engine = provider.engines[0]
    original = engine.unload

    def unload() -> None:
        unload_threads.append(threading.get_ident())
        original()

    engine.unload = unload
    worker.close()
    assert unload_threads == [load_thread]
    assert not any(t.ident == load_thread for t in threading.enumerate())
    worker.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        rec.submit(worker, "late", LOAD)


def test_an_unload_failure_at_close_is_logged(
    repo: InMemoryVoiceRepository, caplog: pytest.LogCaptureFixture
) -> None:
    class _BadUnload(FakeEngine):
        def unload(self) -> None:
            raise RuntimeError("cannot unload")

    worker = EngineWorker(_Provider(_BadUnload), repo, _Stretch())
    rec = _Recorder()
    rec.submit(worker, "load", LOAD)
    rec.next()
    worker.close()
    assert "unloading the engine at close failed" in caplog.text


def test_close_waits_for_queued_jobs(repo: InMemoryVoiceRepository, provider: _Provider) -> None:
    worker = EngineWorker(provider, repo, _Stretch())
    rec = _Recorder()
    rec.submit(worker, "load", LOAD)
    rec.submit(worker, "unload", UnloadEngine())
    worker.close()
    assert [label for label, _, _ in rec.take(2)] == ["load", "unload"]
