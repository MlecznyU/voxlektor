"""Session behaviour, driven through the deterministic Simulator (no threads, no sleeps)."""

from pathlib import Path

import pytest

from voxlektor.app.commands import (
    Back,
    Command,
    Pause,
    Resume,
    SetIdle,
    Shutdown,
    Skip,
    Speak,
    Status,
    StopReading,
    Unload,
)
from voxlektor.app.effects import (
    AbortPlayback,
    CancelWake,
    Cue,
    DropAudio,
    Effect,
    Exit,
    Generate,
    LoadEngine,
    PausePlayback,
    PrepareVoice,
    ResumePlayback,
    ScheduleWake,
    StartPlayback,
    UnloadEngine,
)
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.reading import Chunk
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId, EngineId, Epoch, Lang, LangPref
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import SimulationError, Simulator

VOICES = ("m1-calm", "v0-default")
STATUS_KEYS = {
    "state",
    "engine",
    "voice",
    "idle_timeout_s",
    "idle_remaining_s",
    "queued",
    "playing",
    "paused",
    "underruns",
    "first_audio_s",
    "chunks",
    "gen_seconds",
    "audio_seconds",
    "last_error",
    "caps",
}
CAPS = [
    "back",
    "idle",
    "pause",
    "resume",
    "shutdown",
    "skip",
    "speak",
    "status",
    "stop_reading",
    "unload",
]
FAKE_FAILURE = "fake engine failure on 'BOOM'"


def _catalog() -> VoiceCatalog:
    refs = [VoiceRef(n, Path(f"/voices/{n}/ref.wav"), f"reference {n}") for n in VOICES]
    return VoiceCatalog(InMemoryVoiceRepository(refs))


def _sim(
    settings: Settings | None = None,
    *,
    auto_complete: bool = True,
    fail_on: str | None = None,
    engine: FakeEngine | None = None,
) -> Simulator:
    clock = FakeClock()
    session = Session(settings or Settings(engine=EngineId("fake")), _catalog(), clock)
    fake = engine or FakeEngine(sample_rate=100, fail_on=fail_on)
    return Simulator(session, fake, MemoryOutput(auto_complete=auto_complete), clock)


def _sentence(tag: str) -> str:
    """About 210 characters: two of them never share a chunk."""
    return f"{tag} {'word ' * 40}end."


def _text(*tags: str) -> str:
    return " ".join(_sentence(tag) for tag in tags)


def _speak(
    text: str,
    *,
    voice: str | None = None,
    now: bool = False,
    lang: LangPref | None = "en",
    rate: float | None = None,
) -> Speak:
    return Speak(text, voice, lang, rate, now)


def _count(sim: Simulator, kind: type[Effect]) -> int:
    return sum(isinstance(e, kind) for e in sim.effects)


def _of[E: Effect](sim: Simulator, kind: type[E]) -> list[E]:
    return [e for e in sim.effects if isinstance(e, kind)]


def _played_tags(sim: Simulator) -> list[str]:
    return [sim.chunks[c].text.split()[0] for c in sim.output.started]


# ---------------------------------------------------------------- keep-warm


def test_cold_speak_loads_once_then_warm() -> None:
    sim = _sim()
    assert sim.status()["state"] == "cold"
    reply = sim.handle(_speak(_text("a1", "a2")))
    assert reply.ok
    assert reply.data == {"chunks": 2}
    assert reply.error is None
    assert _of(sim, LoadEngine) == [LoadEngine(EngineId("fake"))]
    assert sim.status()["state"] == "warm"
    assert _played_tags(sim) == ["a1", "a2"]
    sim.handle(_speak(_text("b1")))
    assert _count(sim, LoadEngine) == 1
    assert sim.engine.load_calls == 1
    assert _played_tags(sim) == ["a1", "a2", "b1"]


def test_voice_prompt_is_prepared_once_per_voice_per_load() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    sim.handle(_speak(_text("b")))
    assert _count(sim, PrepareVoice) == 1
    assert sim.engine.prepare_calls == 1
    sim.handle(_speak(_text("c"), voice="m1-calm"))
    assert [e.ref.name for e in _of(sim, PrepareVoice)] == ["v0-default", "m1-calm"]
    sim.handle(Unload())
    sim.handle(_speak(_text("d")))
    assert [e.ref.name for e in _of(sim, PrepareVoice)] == ["v0-default", "m1-calm", "v0-default"]
    assert [g.voice for g in _of(sim, Generate)] == [
        "v0-default",
        "v0-default",
        "m1-calm",
        "v0-default",
    ]


def test_idle_unload_fires_at_timeout() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    assert _of(sim, ScheduleWake) == [ScheduleWake(600.0)]
    sim.advance(599)
    assert sim.status()["state"] == "warm"
    assert sim.status()["idle_remaining_s"] == pytest.approx(1.0)
    sim.advance(2)
    assert sim.status()["state"] == "cold"
    assert sim.engine.unload_calls == 1
    assert _count(sim, UnloadEngine) == 1


def test_paused_reading_never_unloads() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    assert sim.playing is not None
    sim.handle(Pause())
    assert sim.output.paused
    sim.advance(10_000)
    status = sim.status()
    assert status["state"] == "warm"
    assert status["paused"] is True
    assert status["idle_remaining_s"] is None
    assert _count(sim, UnloadEngine) == 0


def test_timeout_zero_never_unloads() -> None:
    sim = _sim(Settings(idle_timeout_s=0))
    sim.handle(_speak(_text("a")))
    sim.advance(100_000)
    status = sim.status()
    assert status["state"] == "warm"
    assert status["idle_timeout_s"] == 0
    assert status["idle_remaining_s"] is None
    assert _count(sim, ScheduleWake) == 0


def test_set_idle_applies_live() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    sim.advance(30)
    reply = sim.handle(SetIdle("1m"))
    assert reply.ok
    assert reply.data == {"idle_timeout_s": 60}
    assert sim.status()["idle_timeout_s"] == 60
    assert sim.status()["idle_remaining_s"] == pytest.approx(30.0)
    sim.advance(31)
    assert sim.status()["state"] == "cold"


def test_set_idle_in_the_past_unloads_immediately() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    sim.advance(120)
    sim.handle(SetIdle("1m"))
    assert sim.status()["state"] == "cold"


def test_set_idle_rejects_bad_duration() -> None:
    sim = _sim()
    reply = sim.handle(SetIdle("ten minutes"))
    assert not reply.ok
    assert reply.error is not None
    assert reply.error.code == "invalid_duration"
    assert sim.status()["idle_timeout_s"] == 600


def test_unload_during_reading_stops_reading_and_goes_cold() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    playing = sim.playing
    assert playing is not None
    reply = sim.handle(Unload())
    assert reply.ok
    status = sim.status()
    assert status["state"] == "cold"
    assert status["queued"] == 0
    assert status["playing"] is False
    assert sim.output.aborted == 1
    assert sim.audio == {}
    # A late "done" for the aborted chunk is ignored.
    sim.complete(playing)
    assert sim.status()["state"] == "cold"
    assert sim.output.started == [playing]


def test_unload_when_cold_is_harmless() -> None:
    sim = _sim()
    assert sim.handle(Unload()).ok
    assert sim.effects == []


def test_shutdown_stops_unloads_and_exits() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    reply = sim.handle(Shutdown())
    assert reply.ok
    assert isinstance(sim.effects[-1], Exit)
    assert sim.exited
    assert sim.status()["state"] == "cold"
    assert sim.engine.unload_calls == 1


def test_speak_during_loading_is_queued_and_plays_after_load() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a")))
    assert sim.status()["state"] == "loading"
    sim.handle(_speak(_text("b")))
    assert sim.status()["queued"] == 2
    assert sim.output.started == []
    sim.run_until_idle()
    assert sim.status()["state"] == "warm"
    assert _played_tags(sim) == ["a", "b"]
    assert _count(sim, LoadEngine) == 1


def test_unload_during_loading_ends_cold_after_load() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a")))
    sim.handle(Unload())
    assert sim.status()["state"] == "loading"
    sim.run_until_idle()
    assert sim.status()["state"] == "cold"
    assert sim.output.started == []
    assert sim.engine.load_calls == 1
    assert sim.engine.unload_calls == 1


def test_speak_after_unload_during_loading_still_plays() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a")))
    sim.handle(Unload())
    sim.handle(_speak(_text("b")))
    sim.run_until_idle()
    assert _played_tags(sim) == ["b"]
    assert sim.status()["state"] == "warm"
    assert sim.engine.load_calls == 2


def test_speak_during_unloading_reloads() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    sim.auto_run = False
    sim.handle(Unload())
    assert sim.status()["state"] == "unloading"
    sim.handle(_speak(_text("b")))
    sim.run_until_idle()
    assert sim.status()["state"] == "warm"
    assert _played_tags(sim) == ["a", "b"]
    assert sim.engine.load_calls == 2
    assert _count(sim, PrepareVoice) == 2


def test_stop_during_loading_schedules_idle_unload_after_load() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a")))
    sim.handle(StopReading())
    sim.run_until_idle()
    assert sim.status()["state"] == "warm"
    assert sim.output.started == []
    assert _of(sim, ScheduleWake) == [ScheduleWake(600.0)]
    sim.advance(601)
    assert sim.status()["state"] == "unloading"
    sim.run_until_idle()
    assert sim.status()["state"] == "cold"


def test_wake_while_busy_does_not_unload() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a")))
    sim.complete()
    assert _of(sim, ScheduleWake) == [ScheduleWake(600.0)]
    sim.handle(_speak(_text("b")))
    assert CancelWake() in sim.effects
    sim.advance(700)
    assert sim.status()["state"] == "warm"


def test_load_failure_stops_reading_and_records_error() -> None:
    class BrokenEngine(FakeEngine):
        def load(self) -> None:
            raise RuntimeError("no model")

    sim = _sim(engine=BrokenEngine(sample_rate=100))
    sim.handle(_speak(_text("a")))
    status = sim.status()
    assert status["state"] == "cold"
    assert status["last_error"] == "no model"
    assert status["queued"] == 0
    assert sim.output.started == []
    assert sim.cues == [Cue.STOPPED]


def test_load_failure_without_a_reading_plays_no_cue() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a")))
    sim.handle(StopReading())
    sim.apply(sim.session.on_load_failed("no model"))
    assert sim.status()["last_error"] == "no model"
    assert sim.status()["state"] == "cold"
    assert sim.cues == []


# ---------------------------------------------------------------- failures (Review Focus 5)


def test_generation_failure_stops_reading_and_session_stays_usable() -> None:
    sim = _sim(fail_on="BOOM")
    sim.handle(_speak(_text("a", "BOOM", "c")))
    status = sim.status()
    assert status["last_error"] == FAKE_FAILURE
    assert _played_tags(sim) == ["a"]
    assert status["queued"] == 0
    assert status["playing"] is False
    assert sim.cues == [Cue.RETRY, Cue.RETRY, Cue.STOPPED]
    assert _count(sim, LoadEngine) == 2  # the second attempt reloaded the engine
    assert sum(g.chunk.text.startswith("BOOM") for g in _of(sim, Generate)) == 3
    sim.handle(_speak(_text("fine")))
    assert _played_tags(sim) == ["a", "fine"]
    assert sim.status()["state"] == "warm"


def test_new_reading_from_idle_clears_last_error() -> None:
    """C5: status.last_error describes the latest reading only."""
    sim = _sim(fail_on="BOOM")
    sim.handle(_speak(_text("BOOM")))
    assert sim.status()["last_error"] == FAKE_FAILURE
    sim.handle(_speak("   "))  # an empty plan starts no reading
    assert sim.status()["last_error"] == FAKE_FAILURE
    sim.handle(_speak(_text("fine")))
    assert sim.status()["last_error"] is None


@pytest.mark.parametrize(
    ("base_lang", "expected"), [(None, "pl"), ("en", "en"), ("pl", "pl")], ids=str
)
def test_request_base_lang_breaks_auto_detection_ties(
    base_lang: Lang | None, expected: Lang
) -> None:
    """P1: a request's base_lang wins over Settings.base_lang ("pl" here)."""
    sim = _sim()
    sim.handle(Speak("npm run dev", None, "auto", None, False, base_lang=base_lang))
    assert [g.chunk.lang for g in _of(sim, Generate)] == [expected]


def test_generation_failing_once_is_retried() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_on="BOOM", fail_times=1))
    sim.handle(_speak(_text("a", "BOOM", "c")))
    assert _played_tags(sim) == ["a", "BOOM", "c"]
    assert sim.cues == [Cue.RETRY]
    assert _count(sim, LoadEngine) == 1
    assert sim.status()["last_error"] is None


def test_generation_failing_twice_reloads_the_engine_then_plays() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_on="BOOM", fail_times=2))
    sim.handle(_speak(_text("a", "BOOM", "c")))
    assert _played_tags(sim) == ["a", "BOOM", "c"]
    assert sim.cues == [Cue.RETRY, Cue.RETRY]
    assert _count(sim, LoadEngine) == 2
    assert _count(sim, UnloadEngine) == 1
    assert _count(sim, PrepareVoice) == 2  # prompts are prepared again after the reload
    assert sim.status()["last_error"] is None
    assert sim.status()["state"] == "warm"


def test_skip_a_chunk_waiting_for_its_retry() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_on="BOOM", fail_times=2))
    sim.auto_run = False
    sim.handle(_speak(_text("BOOM", "b")))
    while len(sim.cues) < 2:
        sim.step()
    assert sim.status()["state"] == "unloading"  # BOOM waits for the reload
    sim.handle(Skip())
    sim.run_until_idle()
    assert _played_tags(sim) == ["b"]
    assert sim.cues == [Cue.RETRY, Cue.RETRY]


def test_two_chunks_failing_during_one_reload_reload_once() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_on="BOOM", fail_times=4))
    sim.handle(_speak(_text("BOOM1", "BOOM2")))
    assert _played_tags(sim) == ["BOOM1", "BOOM2"]
    assert sim.cues == [Cue.RETRY] * 4
    assert _count(sim, LoadEngine) == 2


def test_prompt_failing_once_is_rebuilt_fresh() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_prepare=1))
    sim.handle(_speak(_text("a")))
    assert _played_tags(sim) == ["a"]
    assert sim.cues == [Cue.RETRY]
    assert [p.fresh for p in _of(sim, PrepareVoice)] == [False, True]


def test_prompt_failing_twice_stops_and_a_later_speak_retries() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_prepare=2))
    sim.handle(_speak(_text("a")))
    status = sim.status()
    assert status["last_error"] == "fake prepare failure"
    assert status["queued"] == 0
    assert sim.output.started == []
    assert sim.cues == [Cue.RETRY, Cue.STOPPED]
    sim.handle(_speak(_text("b")))
    assert _played_tags(sim) == ["b"]
    assert [p.fresh for p in _of(sim, PrepareVoice)] == [False, True, False]


def test_prompt_failure_counter_resets_with_a_new_reading() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a")))
    sim.step()  # load: PrepareVoice is queued
    sim.apply(sim.session.on_prompt_failed("v0-default", "bad ref"))
    sim.run_until_idle()
    assert sim.cues == [Cue.RETRY]
    sim.handle(Unload())
    sim.handle(_speak(_text("b")))
    sim.step()  # unload
    sim.step()  # load again: PrepareVoice is queued
    assert sim.status()["state"] == "warm"
    sim.apply(sim.session.on_prompt_failed("v0-default", "bad ref"))
    assert sim.cues == [Cue.RETRY, Cue.RETRY]
    sim.run_until_idle()
    assert sim.status()["last_error"] is None


def test_stale_prompt_failures_play_no_cue() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    assert sim.session.on_prompt_failed("m1-calm", "never requested") == []
    assert sim.session.on_prompt_failed("v0-default", "already ready") == []
    sim.auto_run = False
    sim.handle(_speak(_text("b"), voice="m1-calm"))  # PrepareVoice(m1-calm) is queued
    sim.handle(StopReading())
    # The voice was requested in this load, but no reading needs it any more.
    assert sim.session.on_prompt_failed("m1-calm", "late") == []
    assert sim.cues == []
    assert sim.status()["last_error"] is None
    sim.handle(_speak(_text("c"), voice="m1-calm"))  # asks for the prompt again
    sim.run_until_idle()
    assert [p.ref.name for p in _of(sim, PrepareVoice)] == ["v0-default", "m1-calm", "m1-calm"]
    assert _played_tags(sim) == ["a", "c"]


def test_prompt_failure_for_a_replaced_reading_is_ignored() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_prepare=1))
    sim.auto_run = False
    sim.handle(_speak(_text("m"), voice="m1-calm", now=True))
    sim.step()  # load: PrepareVoice(m1-calm) is queued
    sim.handle(_speak(_text("n1", "n2", "n3"), now=True))
    sim.run_until_idle()  # the m1-calm prepare fails: no reading needs that voice any more
    assert sim.session.on_prompt_failed("m1-calm", "late again") == []
    assert sim.cues == []
    assert sim.status()["last_error"] is None
    assert [p.fresh for p in _of(sim, PrepareVoice)] == [False, False]


def test_prompt_failure_after_stop_and_another_voice_is_ignored() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_prepare=1))
    sim.auto_run = False
    sim.handle(_speak(_text("m"), voice="m1-calm"))
    sim.step()  # load: PrepareVoice(m1-calm) is queued
    sim.handle(StopReading())
    sim.handle(_speak(_text("n1", "n2")))
    sim.run_until_idle()
    assert sim.cues == []
    assert sim.status()["last_error"] is None
    assert _played_tags(sim) == ["n1", "n2"]


def test_a_ready_chunk_plays_during_a_reload() -> None:
    engine = FakeEngine(sample_rate=100, fail_on="BOOM", fail_times=2)
    sim = _sim(auto_complete=False, engine=engine)
    sim.auto_run = False
    sim.handle(_speak(_text("a", "b", "BOOM")))
    while len(sim.cues) < 2:
        sim.step()
    assert sim.status()["state"] == "unloading"
    sim.complete()  # a ends; b is ready and must not wait for the reload
    assert _played_tags(sim) == ["a", "b"]
    assert sim.status()["state"] == "unloading"
    sim.run_until_idle()
    sim.complete()
    assert _played_tags(sim) == ["a", "b", "BOOM"]


def test_every_engine_load_gets_the_full_prompt_ladder() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_on="BOOM", fail_times=2))
    sim.auto_run = False
    sim.handle(_speak(_text("BOOM", "b")))
    sim.step()  # load: PrepareVoice is queued
    sim.apply(sim.session.on_prompt_failed("v0-default", "bad ref"))  # retried fresh
    while _count(sim, LoadEngine) < 2:  # the generation ladder reloads the engine
        sim.step()
    sim.step()  # the second load: PrepareVoice is queued again
    assert sim.status()["state"] == "warm"
    sim.apply(sim.session.on_prompt_failed("v0-default", "bad ref"))
    assert Cue.STOPPED not in sim.cues
    sim.run_until_idle()
    assert sim.cues == [Cue.RETRY] * 4
    assert _played_tags(sim) == ["BOOM", "b"]


def test_back_requeues_under_fresh_ids() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    sim.complete()
    sim.handle(Back())
    while sim.playing is not None:
        sim.complete()
    assert _played_tags(sim) == ["a", "b", "a", "b", "c"]
    epochs: dict[ChunkId, set[Epoch]] = {}
    for g in _of(sim, Generate):
        epochs.setdefault(g.chunk.id, set()).add(g.chunk.epoch)
    assert all(len(e) == 1 for e in epochs.values())
    assert len(set(sim.output.started)) == 5


def test_regenerated_chunks_are_counted_once() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    before = sim.status()
    sim.handle(Back())  # restarts a: a and b are generated again
    after = sim.status()
    assert after["chunks"] == before["chunks"] == 2
    assert after["audio_seconds"] == before["audio_seconds"]


def test_speak_rejects_rate_out_of_range() -> None:
    sim = _sim()
    for rate in (0.49, 3.01, 0.0):
        reply = sim.handle(_speak(_text("a"), rate=rate))
        assert not reply.ok
        assert reply.error is not None
        assert reply.error.code == "invalid_setting"
    assert sim.effects == []
    assert sim.handle(_speak(_text("a"), rate=0.5)).ok
    assert sim.handle(_speak(_text("b"), rate=3.0)).ok


def test_replies_do_not_share_data() -> None:
    sim = _sim()
    first = sim.handle(Pause())
    second = sim.handle(Resume())
    assert first.data == second.data == {}
    assert first.data is not second.data


def test_zero_history_keeps_nothing_to_go_back_to() -> None:
    sim = _sim(Settings(engine=EngineId("fake"), history=0))
    sim.handle(_speak(_text("a", "b")))
    sim.handle(Back())
    assert _played_tags(sim) == ["a", "b"]


def test_playback_failure_stops_reading_and_records_error() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    sim.fail_playback("device gone")
    status = sim.status()
    assert status["last_error"] == "device gone"
    assert status["queued"] == 0
    assert status["playing"] is False
    assert sim.audio == {}
    assert sim.status()["idle_remaining_s"] == pytest.approx(600.0)
    assert sim.cues == [Cue.STOPPED]
    sim.handle(_speak(_text("c")))
    assert _played_tags(sim) == ["a", "c"]


# ---------------------------------------------------------------- stale callbacks


def test_stale_play_callbacks_are_ignored() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    first = sim.playing
    assert first is not None
    session = sim.session
    assert session.on_play_done(ChunkId(first + 100)) == []
    assert session.on_play_failed(ChunkId(first + 100), "late") == []
    sim.handle(_speak(_text("x"), now=True))
    assert session.on_play_done(first) == []
    assert session.on_play_failed(first, "late") == []
    assert sim.status()["last_error"] is None


def test_stale_generation_callbacks_are_ignored_or_dropped() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    session = sim.session
    old = _of(sim, Generate)[0].chunk
    sim.handle(_speak(_text("x"), now=True))
    assert session.on_gen_failed(old.epoch, old.id, "late") == []
    assert session.on_generated(old.epoch, old.id, 1.0, 0.5) == [DropAudio((old.id,))]
    current = _of(sim, Generate)[-1].chunk
    assert session.on_gen_failed(current.epoch, ChunkId(999), "unknown chunk") == []
    assert session.on_generated(current.epoch, ChunkId(999), 1.0, 0.5) == [
        DropAudio((ChunkId(999),))
    ]
    assert session.on_prompt_ready("m1-calm") == []
    assert sim.status()["last_error"] is None


def test_duplicate_generated_result_is_dropped() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    ready = _of(sim, Generate)[1].chunk
    assert ready.id in sim.audio
    effects = sim.session.on_generated(ready.epoch, ready.id, 1.0, 0.5)
    assert effects == [DropAudio((ready.id,))]
    assert sim.session.on_gen_failed(ready.epoch, ready.id, "late") == []


def test_skipped_chunk_in_flight_is_dropped_when_it_arrives() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c", "d")))
    sim.auto_run = False
    sim.complete()  # a done; b starts; d is requested but not generated yet
    sim.handle(Skip())  # b skipped; c (ready) starts
    sim.handle(Skip())  # c skipped; d (in flight) becomes the head
    assert sim.status()["underruns"] == 0
    sim.handle(Skip())  # d skipped while in flight
    assert sim.status()["queued"] == 0
    assert sim.generated_unplayed == 1  # d still counts until its result arrives
    sim.run_until_idle()
    assert _played_tags(sim) == ["a", "b", "c"]
    late = _of(sim, DropAudio)[-1]
    assert [sim.chunks[c].text.split()[0] for c in late.chunks] == ["d"]
    assert sim.generated_unplayed == 0


# ---------------------------------------------------------------- ahead window (Review Focus 3)


def test_long_text_keeps_at_most_two_generated_unplayed() -> None:
    sim = _sim(auto_complete=False)
    text = "Sentence number one goes here and it is long enough. " * 2000
    assert len(text) >= 100_000
    reply = sim.handle(_speak(text))
    chunks = reply.data["chunks"]
    assert isinstance(chunks, int)
    assert chunks > 300
    steps = 0
    while sim.playing is not None:
        assert sim.generated_unplayed <= 2
        sim.complete()
        steps += 1
    assert steps == chunks
    assert sim.max_generated_unplayed == 2
    assert sim.status()["queued"] == 0


def test_long_text_with_auto_complete_stays_within_window() -> None:
    sim = _sim()
    sim.handle(_speak("Short sentence that is long enough to stand alone, really. " * 1700))
    assert sim.max_generated_unplayed <= 2
    assert sim.status()["state"] == "warm"
    assert sim.status()["queued"] == 0


# ---------------------------------------------------------------- now / skip / back


def test_now_aborts_old_reading_and_drops_late_results() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a1", "a2", "a3", "a4")))
    sim.auto_run = False
    sim.complete()  # a2 starts, a4 is requested but not generated yet
    old_epoch = sim.started[-1][1]
    reply = sim.handle(_speak(_text("n1", "n2"), now=True))
    assert reply.data == {"chunks": 2}
    assert sim.output.aborted == 1
    drop = _of(sim, DropAudio)[-1]
    assert [sim.chunks[c].text.split()[0] for c in drop.chunks] == ["a3"]
    sim.run_until_idle()  # the late a4 arrives first and is dropped
    late = _of(sim, DropAudio)[-1]
    assert [sim.chunks[c].text.split()[0] for c in late.chunks] == ["a4"]
    sim.complete()
    sim.complete()
    assert _played_tags(sim) == ["a1", "a2", "n1", "n2"]
    assert all(epoch > old_epoch for _, epoch in sim.started[2:])
    assert sim.status()["queued"] == 0


def test_now_with_empty_text_changes_nothing() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a")))
    before = list(sim.effects)
    reply = sim.handle(_speak("  \n ", now=True))
    assert reply.ok
    assert reply.data == {"chunks": 0}
    assert sim.effects == before
    assert sim.playing is not None


def test_now_while_paused_starts_unpaused() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a")))
    sim.handle(Pause())
    sim.handle(_speak(_text("n"), now=True))
    assert ResumePlayback() in sim.effects
    assert sim.status()["paused"] is False
    assert _played_tags(sim) == ["a", "n"]


def test_skip_moves_to_the_next_chunk() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    assert sim.handle(Skip()).ok
    assert sim.output.aborted == 1
    assert _played_tags(sim) == ["a", "b"]
    sim.handle(Skip())
    sim.handle(Skip())
    assert _played_tags(sim) == ["a", "b", "c"]
    assert sim.status()["queued"] == 0
    assert sim.status()["idle_remaining_s"] == pytest.approx(600.0)
    assert sim.handle(Skip()).ok


def test_skip_a_waiting_head_drops_its_audio() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    sim.handle(Pause())
    sim.complete()  # a ends while paused: b is ready but waits
    assert sim.playing is None
    sim.handle(Skip())
    assert sim.chunks[_of(sim, DropAudio)[-1].chunks[0]].text.startswith("b ")
    sim.handle(Resume())
    assert _played_tags(sim) == ["a", "c"]


def test_skip_a_head_not_generated_yet() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a", "b")))
    sim.handle(Skip())
    sim.run_until_idle()
    assert _played_tags(sim) == ["b"]


def test_back_replays_previous_and_current() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    sim.complete()
    assert _played_tags(sim) == ["a", "b"]
    assert sim.handle(Back()).ok
    assert sim.output.aborted == 1
    while sim.playing is not None:
        sim.complete()
    assert _played_tags(sim) == ["a", "b", "a", "b", "c"]


def test_back_at_the_start_restarts_current() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    sim.handle(Back())
    while sim.playing is not None:
        sim.complete()
    assert _played_tags(sim) == ["a", "a", "b"]


def test_back_when_idle_replays_last_chunk() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a", "b")))
    sim.handle(Back())
    assert _played_tags(sim) == ["a", "b", "b"]
    assert sim.status()["state"] == "warm"


def test_back_with_nothing_is_noop() -> None:
    sim = _sim()
    assert sim.handle(Back()).ok
    assert sim.effects == []


def test_back_while_head_waits_keeps_order() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c", "d")))
    sim.handle(Pause())
    sim.complete()  # a done while paused; b waits (ready)
    sim.handle(Back())  # [a, b] re-queued; ready audio dropped
    sim.handle(Resume())
    while sim.playing is not None:
        sim.complete()
    assert _played_tags(sim) == ["a", "a", "b", "c", "d"]


def test_history_is_bounded() -> None:
    sim = _sim(Settings(history=1))
    sim.handle(_speak(_text("a", "b", "c")))
    sim.handle(Back())
    sim.handle(Back())
    assert _played_tags(sim) == ["a", "b", "c", "c", "c"]


# ---------------------------------------------------------------- pause / resume / stop


def test_pause_and_resume_map_to_output() -> None:
    sim = _sim(auto_complete=False)
    assert sim.handle(Pause()).ok  # idle: nothing to pause
    assert sim.status()["paused"] is False
    assert sim.handle(Resume()).ok  # not paused: nothing to resume
    sim.handle(_speak(_text("a", "b")))
    sim.handle(Pause())
    sim.handle(Pause())
    assert _count(sim, PausePlayback) == 1
    assert sim.status()["paused"] is True
    sim.handle(Resume())
    assert _count(sim, ResumePlayback) == 1
    assert sim.status()["paused"] is False
    assert not sim.output.paused


def test_resume_starts_a_ready_head() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    sim.handle(Pause())
    sim.complete()
    assert sim.status()["playing"] is False
    assert sim.status()["queued"] == 1
    sim.handle(Resume())
    assert _played_tags(sim) == ["a", "b"]
    assert sim.status()["underruns"] == 0


def test_stop_reading_clears_queue_and_lifts_pause() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b", "c")))
    sim.handle(Pause())
    assert sim.handle(StopReading()).ok
    assert AbortPlayback() in sim.effects
    assert sim.effects[-2:] == [ResumePlayback(), ScheduleWake(600.0)]
    status = sim.status()
    assert status["paused"] is False
    assert status["queued"] == 0
    assert sim.audio == {}
    sim.handle(_speak(_text("d")))
    assert _played_tags(sim) == ["a", "d"]


def test_stop_reading_when_idle_has_no_effects() -> None:
    sim = _sim()
    assert sim.handle(StopReading()).ok
    assert sim.effects == []


# ---------------------------------------------------------------- settings, voices, status


def test_speak_uses_settings_defaults_and_overrides() -> None:
    settings = Settings(engine=EngineId("fake"), lang="pl", rate=1.5)
    sim = _sim(settings)
    sim.handle(Speak(_text("a"), None, None, None, False))
    first = _of(sim, Generate)[0]
    assert (first.voice, first.chunk.lang, first.chunk.rate) == ("v0-default", "pl", 1.5)
    sim.handle(Speak(_text("b"), "m1-calm", "en", 1.25, False))
    second = _of(sim, Generate)[1]
    assert (second.voice, second.chunk.lang, second.chunk.rate) == ("m1-calm", "en", 1.25)


def test_unknown_voice_is_an_error() -> None:
    sim = _sim()
    reply = sim.handle(_speak(_text("a"), voice="nobody"))
    assert not reply.ok
    assert reply.error is not None
    assert reply.error.code == "unknown_voice"
    assert sim.effects == []


def test_empty_speak_replies_zero_chunks_without_effects() -> None:
    session = Session(Settings(), _catalog(), FakeClock())
    outcome = session.handle(_speak(" \n\n "))
    assert outcome.reply is not None
    assert outcome.reply.ok
    assert outcome.reply.data == {"chunks": 0}
    assert outcome.effects == []


def test_status_contains_every_key_and_caps() -> None:
    sim = _sim()
    status = sim.status()
    assert set(status) == STATUS_KEYS
    assert status["caps"] == CAPS
    assert status == {
        "state": "cold",
        "engine": "fake",
        "voice": "v0-default",
        "idle_timeout_s": 600,
        "idle_remaining_s": None,
        "queued": 0,
        "playing": False,
        "paused": False,
        "underruns": 0,
        "first_audio_s": None,
        "chunks": 0,
        "gen_seconds": 0.0,
        "audio_seconds": 0.0,
        "last_error": None,
        "caps": CAPS,
    }


def test_status_reports_the_reading_voice_and_metrics() -> None:
    sim = _sim(auto_complete=False)
    sim.auto_run = False
    sim.handle(_speak(_text("a", "b"), voice="m1-calm"))
    sim.advance(1.5)
    sim.run_until_idle()
    status = sim.status()
    assert status["voice"] == "m1-calm"
    assert status["playing"] is True
    assert status["first_audio_s"] == pytest.approx(1.5)
    assert status["chunks"] == 2
    audio = status["audio_seconds"]
    assert isinstance(audio, float)
    assert audio > 0
    assert status["gen_seconds"] == pytest.approx(0.0)


def test_underrun_is_counted_when_playback_runs_dry() -> None:
    sim = _sim(auto_complete=False)
    sim.auto_run = False
    sim.handle(_speak(_text("a", "b")))
    while sim.playing is None:
        sim.step()
    sim.complete()  # b has been requested but not generated
    assert sim.status()["underruns"] == 1
    sim.run_until_idle()
    assert _played_tags(sim) == ["a", "b"]


def test_new_reading_resets_metrics() -> None:
    sim = _sim(auto_complete=False)
    sim.auto_run = False
    sim.handle(_speak(_text("a", "b")))
    while sim.playing is None:
        sim.step()
    sim.complete()
    sim.run_until_idle()
    sim.complete()
    assert sim.status()["underruns"] == 1
    sim.handle(_speak(_text("c")))
    sim.run_until_idle()
    status = sim.status()
    assert status["underruns"] == 0
    assert status["chunks"] == 1


# ---------------------------------------------------------------- simulator self-checks


def test_simulator_rejects_a_second_concurrent_playback() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    second = _of(sim, Generate)[1].chunk.id
    with pytest.raises(SimulationError, match="already playing"):
        sim.apply([StartPlayback(second)])


def test_simulator_rejects_playback_without_audio() -> None:
    sim = _sim()
    with pytest.raises(SimulationError, match="no audio"):
        sim.apply([StartPlayback(ChunkId(42))])


def test_simulator_rejects_generate_while_not_warm() -> None:
    sim = _sim()
    sim.auto_run = False
    sim.handle(_speak(_text("a")))
    assert sim.status()["state"] == "loading"
    chunk = Chunk(ChunkId(1), "text", "en", 1.0, Epoch(0))
    with pytest.raises(SimulationError, match="not warm"):
        sim.apply([Generate(chunk, "v0-default")])


def test_simulator_rejects_generate_without_a_prompt() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    chunk = Chunk(ChunkId(99), "text", "en", 1.0, Epoch(0))
    with pytest.raises(SimulationError, match="without a prompt"):
        sim.apply([Generate(chunk, "m1-calm")])


def test_simulator_rejects_playback_while_output_is_paused() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a", "b")))
    sim.handle(Pause())
    sim.complete()
    waiting = _of(sim, Generate)[1].chunk.id
    with pytest.raises(SimulationError, match="paused"):
        sim.apply([StartPlayback(waiting)])


def test_simulator_needs_a_playing_chunk_to_complete() -> None:
    sim = _sim()
    with pytest.raises(SimulationError, match="nothing is playing"):
        sim.complete()


def test_simulator_reports_a_late_failure_for_another_chunk() -> None:
    sim = _sim(auto_complete=False)
    sim.handle(_speak(_text("a")))
    playing = sim.playing
    sim.fail_playback("late", chunk=ChunkId(999))
    assert sim.playing == playing
    assert sim.status()["last_error"] is None


def test_simulator_fires_wakes_scheduled_in_the_past() -> None:
    sim = _sim()
    sim.handle(_speak(_text("a")))
    sim.clock.advance(1000)
    sim.apply([ScheduleWake(10.0)])
    assert sim.wakes == []
    assert sim.status()["state"] == "cold"


def test_commands_are_plain_values() -> None:
    commands: list[Command] = [
        _speak("x"),
        Pause(),
        Resume(),
        Skip(),
        Back(),
        StopReading(),
        Status(),
        SetIdle("1m"),
        Unload(),
        Shutdown(),
    ]
    assert len(set(commands)) == len(commands)
