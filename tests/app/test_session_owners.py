"""Multi-owner Session: floor, grace, preempt and per-owner control (spec §4), via the Simulator."""

from pathlib import Path

from voxlektor.app.commands import Back, Pause, Resume, Skip, Speak, StopReading, Unload
from voxlektor.app.effects import (
    AbortPlayback,
    CancelGeneration,
    Cue,
    DropAudio,
    Effect,
    Generate,
    ResumePlayback,
    ScheduleWake,
)
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import Simulator

FAKE_FAILURE = "fake engine failure on 'BOOM'"


def _sim(
    *,
    grace: int = 8,
    ahead: int = 2,
    history: int = 20,
    engine: FakeEngine | None = None,
) -> Simulator:
    refs = [VoiceRef(n, Path(f"/voices/{n}/ref.wav"), n) for n in ("m1-calm", "v0-default")]
    clock = FakeClock()
    settings = Settings(engine=EngineId("fake"), floor_grace_s=grace, ahead=ahead, history=history)
    session = Session(settings, VoiceCatalog(InMemoryVoiceRepository(refs)), clock)
    fake = engine or FakeEngine(sample_rate=100, fail_on="BOOM")
    return Simulator(session, fake, MemoryOutput(auto_complete=False), clock)


def _text(*tags: str) -> str:
    return "\n\n".join(f"{tag} {'word ' * 18}end." for tag in tags)


def _say(  # noqa: PLR0913  # test helper mirroring the Speak fields
    sim: Simulator,
    owner: str | None,
    *tags: str,
    group: str | None = None,
    now: bool = False,
    preempt: bool = False,
    tag: str | None = None,
) -> dict[str, object]:
    text = _text(*tags)
    cmd = Speak(text, None, "en", None, now, owner=owner, group=group, preempt=preempt, tag=tag)
    reply = sim.handle(cmd)
    assert reply.ok, reply.error
    sim.clock.advance(1)  # distinct ages
    return dict(reply.data)


def _played(sim: Simulator) -> list[str]:
    return [sim.chunks[c].text.split()[0] for c in sim.output.started]


def _playing(sim: Simulator) -> str | None:
    return None if sim.playing is None else sim.chunks[sim.playing].text.split()[0]


def _finish(sim: Simulator) -> None:
    while sim.playing is not None:
        sim.complete()


def _of[E: Effect](sim: Simulator, kind: type[E]) -> list[E]:
    return [e for e in sim.effects if isinstance(e, kind)]


def _generated(sim: Simulator) -> list[str]:
    return [g.chunk.text.split()[0] for g in _of(sim, Generate)]


# ---------------------------------------------------------------- floor rule


def test_second_owner_waits_for_group_end() -> None:
    sim = _sim()
    assert _say(sim, "a", "a1", "a2") == {"chunks": 2, "group": "g1"}
    assert _say(sim, "b", "b1") == {"chunks": 1, "group": "g2"}
    assert _played(sim) == ["a1"]
    sim.complete()
    assert _played(sim) == ["a1", "a2"]
    sim.complete()
    assert _played(sim) == ["a1", "a2", "b1"]
    _finish(sim)
    assert sim.status()["queued"] == 0


def test_ahead_generates_next_owner_before_boundary() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    _say(sim, "b", "b1")
    sim.complete()
    assert _playing(sim) == "a2"
    assert "b1" in _generated(sim)
    sim.complete()
    assert _playing(sim) == "b1"
    assert sim.status()["underruns"] == 0


def test_append_to_named_group_keeps_it_one_group() -> None:
    sim = _sim()
    assert _say(sim, "a", "a1", group="r1") == {"chunks": 1, "group": "r1"}
    _say(sim, "b", "b1")
    assert _say(sim, "a", "a2", group="r1") == {"chunks": 1, "group": "r1"}
    _finish(sim)
    assert _played(sim) == ["a1", "a2", "b1"]


def test_auto_group_names_skip_names_in_use() -> None:
    sim = _sim()
    assert _say(sim, "a", "a1", group="g1")["group"] == "g1"
    assert _say(sim, "a", "a2")["group"] == "g2"


def test_now_replaces_only_same_owner() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    _say(sim, "b", "b1")
    drops_before = len(_of(sim, DropAudio))
    _say(sim, "a", "n1", now=True)
    assert _playing(sim) == "n1"
    dropped = [c for d in _of(sim, DropAudio)[drops_before:] for c in d.chunks]
    assert all(not sim.chunks[c].text.startswith("b1") for c in dropped)
    _finish(sim)
    assert _played(sim) == ["a1", "n1", "b1"]


def test_now_from_waiting_owner_keeps_place() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    _say(sim, "c", "c1")
    _say(sim, "b", "bn", now=True)
    assert sim.output.aborted == 0
    assert _playing(sim) == "a1"
    _finish(sim)
    assert _played(sim) == ["a1", "bn", "c1"]


def test_preempt_takes_floor_and_restarts_interrupted_chunk() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    _say(sim, "b", "b1", preempt=True)
    assert sim.output.aborted == 1
    assert _playing(sim) == "b1"
    assert sim.status()["floor"] == {"owner": "b", "group": "g2"}
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "a1", "a2"]


def test_preempt_before_the_holder_plays_keeps_its_audio_in_flight() -> None:
    sim = _sim()
    sim.auto_run = False
    _say(sim, "a", "a1")
    sim.step()  # load
    sim.step()  # prompt: a1 is queued
    _say(sim, "b", "b1", preempt=True)
    assert sim.output.aborted == 0
    sim.run_until_idle()
    _finish(sim)
    assert _played(sim) == ["b1", "a1"]
    assert _generated(sim).count("a1") == 1  # the in-flight a1 was kept, not regenerated


def test_audio_behind_an_interrupted_chunk_does_not_block_its_restart() -> None:
    sim = _sim(ahead=1)
    sim.auto_run = False
    _say(sim, "a", "a1", "a2")
    sim.step()  # load
    sim.step()  # prompt: a1 queued
    sim.step()  # a1 ready and playing: a2 queued
    _say(sim, "b", "b1", preempt=True)  # a1 aborted; a2 still in flight, kept
    sim.run_until_idle()
    assert _playing(sim) == "b1"
    sim.complete()  # a gets the floor back: a1 must be generated although a2 is ready
    sim.run_until_idle()
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "a1", "a2"]


def test_preempt_by_the_holder_changes_nothing_audible() -> None:
    sim = _sim()
    _say(sim, "a", "a1", group="r")
    _say(sim, "a", "a2", group="r", preempt=True)
    assert sim.output.aborted == 0
    _finish(sim)
    assert _played(sim) == ["a1", "a2"]


def test_now_and_preempt_from_a_waiting_owner_take_the_floor() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    _say(sim, "b", "bn", now=True, preempt=True)
    assert _playing(sim) == "bn"
    _finish(sim)
    assert _played(sim) == ["a1", "bn", "a1"]


# ---------------------------------------------------------------- pause, grace, hold


def test_owner_pause_yields_after_grace() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    _say(sim, "b", "b1")
    assert sim.handle(Pause(owner="a")).ok
    assert sim.status()["paused"] is True
    assert ScheduleWake(10.0) in sim.effects  # b arrived at 1, a paused at 2: max(2, 1) + 8
    sim.advance(7.5)
    assert _playing(sim) == "a1"
    sim.advance(1)
    assert _playing(sim) == "b1"
    assert sim.status()["paused"] is False
    assert sim.status()["floor"] == {"owner": "b", "group": "g2"}
    assert sim.handle(Resume(owner="a")).ok
    assert _playing(sim) == "b1"  # resuming waits for the boundary
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "a1", "a2"]


def test_owner_pause_without_waiter_never_yields() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    sim.handle(Pause(owner="a"))
    sim.advance(60)
    assert _playing(sim) == "a1"
    assert sim.output.paused
    assert sim.status()["floor"] == {"owner": "a", "group": "g1"}
    sim.handle(Resume(owner="a"))
    assert not sim.output.paused
    _finish(sim)
    assert _played(sim) == ["a1"]


def test_zero_grace_yields_at_once() -> None:
    sim = _sim(grace=0)
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause(owner="a"))
    assert _playing(sim) == "b1"
    assert sim.status()["paused"] is False


def test_grace_starts_when_a_paused_waiter_resumes() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause(owner="b"))  # t=2
    sim.clock.advance(1)
    sim.handle(Pause(owner="a"))  # t=3: nobody eligible waits
    sim.advance(97)
    sim.handle(Resume(owner="b"))  # t=100: b is eligible from now on
    assert _playing(sim) == "a1"
    assert ScheduleWake(108.0) in sim.effects
    sim.advance(7.5)
    assert _playing(sim) == "a1"
    sim.advance(1)
    assert _playing(sim) == "b1"


def test_grace_wake_firing_early_is_rearmed() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause(owner="a"))  # deadline max(2, 1) + 8 = 10
    sim.clock.advance(7.999)
    effects = sim.session.on_wake()  # the timer fires a moment early
    assert ScheduleWake(10.0) in effects
    assert _playing(sim) == "a1"


def test_preempt_from_a_paused_owner_lifts_its_pause() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause(owner="b"))
    _say(sim, "b", "b2", preempt=True)
    assert sim.status()["floor"] == {"owner": "b", "group": "g3"}
    assert sim.status()["paused"] is False
    assert _playing(sim) == "b2"


def test_paused_waiting_owner_is_passed_over_and_not_generated() -> None:
    sim = _sim(ahead=1)
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    _say(sim, "c", "c1")
    sim.handle(Pause(owner="c"))  # c generated nothing yet: the window holds b1
    sim.complete()
    sim.complete()
    assert _played(sim) == ["a1", "b1"]
    assert "c1" not in _generated(sim)
    assert sim.status()["queued"] == 1
    sim.handle(Resume(owner="c"))
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "c1"]


def test_pause_lifts_when_the_owner_has_nothing_left() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause(owner="b"))
    sim.handle(StopReading(owner="b"))
    _say(sim, "b", "b2")  # a fresh reading of b is not paused
    _finish(sim)
    assert _played(sim) == ["a1", "b2"]


def test_pause_of_an_owner_with_nothing_is_a_noop() -> None:
    sim = _sim()
    sim.handle(Pause(owner="a"))
    _say(sim, "a", "a1")
    assert sim.status()["paused"] is False
    assert _playing(sim) == "a1"


def test_ownerless_pause_holds_everyone() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause())
    sim.advance(60)
    assert _played(sim) == ["a1"]
    assert sim.output.paused
    sim.complete()  # a1 ends while held: nobody else starts speaking
    sim.advance(60)
    assert _played(sim) == ["a1"]
    assert sim.status()["floor"] is None
    sim.handle(Resume())
    assert _played(sim) == ["a1", "b1"]


def test_owner_resume_lifts_hold_only_for_holder() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause())
    resumes = len(_of(sim, ResumePlayback))
    sim.handle(Resume(owner="b"))
    assert sim.status()["paused"] is True
    assert len(_of(sim, ResumePlayback)) == resumes
    sim.handle(Resume(owner="a"))
    assert sim.status()["paused"] is False
    assert len(_of(sim, ResumePlayback)) == resumes + 1


def test_owner_pause_survives_the_ownerless_resume() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    sim.handle(Pause(owner="a"))
    sim.handle(Pause())
    sim.handle(Resume())
    assert sim.status()["paused"] is True


def test_cli_now_lifts_the_hold_for_another_holder() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    sim.handle(Pause())
    _say(sim, None, "n1", now=True)
    assert sim.status()["paused"] is False
    _finish(sim)
    assert _played(sim) == ["a1", "n1"]


# ---------------------------------------------------------------- skip, back, stop


def test_skip_back_of_waiting_owner_are_noops() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    _say(sim, "b", "b1")
    before = list(sim.effects)
    assert sim.handle(Skip(owner="b")).ok
    assert sim.handle(Back(owner="b")).ok
    assert sim.handle(Skip(owner="z")).ok
    assert sim.effects == before


def test_skip_of_the_holder_stays_in_its_group() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Skip(owner="a"))
    assert _playing(sim) == "b1"


def test_ownerless_skip_acts_on_the_holder() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    sim.handle(Skip())
    assert _playing(sim) == "a2"


def test_back_stays_in_owner_history() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1", "b2")
    sim.complete()
    assert _playing(sim) == "b1"
    sim.handle(Back(owner="b"))  # b has no history: b1 restarts, a1 is not replayed
    assert _playing(sim) == "b1"
    sim.complete()
    sim.handle(Back(owner="b"))  # b1 and b2 again
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "b1", "b2", "b1", "b2"]


def test_back_reaches_the_owners_previous_group() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    _say(sim, "a", "a2")
    sim.complete()
    sim.complete()
    assert _playing(sim) == "a2"
    sim.handle(Back(owner="a"))  # a1 (a's previous chunk) and a2, never b1
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "a2", "a1", "a2"]


def test_back_reepochs_the_owners_later_groups() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "a", "a2")
    sim.handle(Back(owner="a"))
    epochs = {sim.chunks[c].epoch for c in sim.output.started[1:]}
    _finish(sim)
    epochs |= {sim.chunks[c].epoch for c in sim.output.started[1:]}
    assert len(epochs) == 1
    assert _played(sim) == ["a1", "a1", "a2"]


def test_ownerless_back_after_the_end_replays_the_last_owner() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    _finish(sim)
    sim.handle(Back())
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "b1"]


def test_ownerless_back_while_held_without_holder_requeues_last_owner() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "a", "a2")
    sim.handle(Pause())
    sim.complete()  # a1 ends while held: a2 waits without the floor
    sim.handle(Back())  # a1 goes back in front of a2
    sim.handle(Resume())
    _finish(sim)
    assert _played(sim) == ["a1", "a1", "a2"]


def test_stop_waiting_owner_removes_its_groups() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    sim.complete()
    _say(sim, "b", "b1")  # generated ahead while a2 plays
    assert "b1" in _generated(sim)
    sim.handle(StopReading(owner="b"))
    assert sim.output.aborted == 0
    assert [sim.chunks[c].text.split()[0] for c in _of(sim, DropAudio)[-1].chunks] == ["b1"]
    _finish(sim)
    assert _played(sim) == ["a1", "a2"]
    assert sim.status()["queued"] == 0


def test_stop_holder_passes_the_floor_on_at_once() -> None:
    sim = _sim()
    _say(sim, "a", "a1", "a2")
    _say(sim, "b", "b1")
    sim.handle(StopReading(owner="a"))
    assert _playing(sim) == "b1"
    sim.handle(StopReading(owner="a"))  # nothing left: a no-op
    _finish(sim)
    assert _played(sim) == ["a1", "b1"]


def test_stop_one_owner_cancels_only_its_epochs() -> None:
    sim = _sim(ahead=3)
    sim.auto_run = False
    _say(sim, "a", "a1", "a2")
    sim.step()  # load
    sim.step()  # prompt: a1 and a2 are queued
    _say(sim, "b", "b1")  # b1 is queued behind them
    a_epoch = _of(sim, Generate)[0].chunk.epoch
    b_epoch = _of(sim, Generate)[-1].chunk.epoch
    assert a_epoch != b_epoch
    sim.handle(StopReading(owner="a"))
    assert _of(sim, CancelGeneration) == [CancelGeneration((a_epoch,))]
    sim.run_until_idle()
    assert _played(sim) == ["b1"]


def test_ownerless_stop_cancels_every_owner_in_one_effect() -> None:
    sim = _sim(ahead=3)
    sim.auto_run = False
    _say(sim, "a", "a1")
    sim.step()
    sim.step()
    _say(sim, "b", "b1")
    epochs = tuple(g.chunk.epoch for g in _of(sim, Generate))
    sim.handle(StopReading())
    assert _of(sim, CancelGeneration) == [CancelGeneration(epochs)]
    sim.run_until_idle()
    assert sim.status()["queued"] == 0


# ---------------------------------------------------------------- failures


def test_chunk_failure_stops_only_that_owner() -> None:
    sim = _sim()
    sim.auto_run = False
    _say(sim, "a", "BOOM")
    _say(sim, "b", "b1")
    sim.run_until_idle()
    _finish(sim)
    assert _played(sim) == ["b1"]
    status = sim.status()
    assert status["last_error"] == FAKE_FAILURE
    assert sim.cues == [Cue.RETRY, Cue.RETRY, Cue.STOPPED]


def test_retry_of_a_passed_over_owner_waits_until_it_resumes() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_on="BOOM", fail_times=1))
    sim.auto_run = False
    _say(sim, "a", "a1")
    sim.step()  # load
    sim.step()  # prompt: a1 queued
    _say(sim, "b", "BOOM")  # queued behind a1
    sim.handle(Pause(owner="b"))
    sim.run_until_idle()  # BOOM fails once: its retry waits, b is passed over
    assert _generated(sim).count("BOOM") == 1
    sim.handle(Resume(owner="b"))
    sim.run_until_idle()
    _finish(sim)
    assert _played(sim) == ["a1", "BOOM"]
    assert sim.cues == [Cue.RETRY]


def test_prompt_failure_stops_the_owners_that_need_the_voice() -> None:
    sim = _sim(engine=FakeEngine(sample_rate=100, fail_prepare=2))
    _say(sim, "a", "a1")
    assert sim.status()["last_error"] == "fake prepare failure"
    assert sim.status()["queued"] == 0
    _say(sim, "b", "b1")
    _finish(sim)
    assert _played(sim) == ["b1"]
    assert sim.status()["last_error"] is None


def test_owner_error_is_cleared_when_that_owner_starts_again() -> None:
    sim = _sim()
    _say(sim, "b", "b1", "b2")
    _say(sim, "a", "BOOM")  # generated ahead while b1 plays: it fails its ladder at once
    assert sim.status()["last_error"] == FAKE_FAILURE
    _say(sim, "c", "c1")  # another owner starting keeps a's error
    assert sim.status()["last_error"] == FAKE_FAILURE
    _say(sim, "a", "a2")  # a starts from nothing while b reads: a's error goes
    assert sim.status()["last_error"] is None


def test_load_failure_stops_everyone() -> None:
    class BrokenEngine(FakeEngine):
        def load(self) -> None:
            raise RuntimeError("no model")

    sim = _sim(engine=BrokenEngine(sample_rate=100))
    sim.auto_run = False
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.run_until_idle()
    status = sim.status()
    assert status["last_error"] == "no model"
    assert status["queued"] == 0
    assert status["floor"] is None
    assert sim.cues == [Cue.STOPPED]


def test_playback_failure_stops_everyone() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.fail_playback("device gone")
    assert sim.status()["queued"] == 0
    assert sim.status()["last_error"] == "device gone"


# ---------------------------------------------------------------- requests and status


def test_owner_cli_is_reserved() -> None:
    sim = _sim()
    commands = [
        Speak(_text("x"), None, "en", None, False, owner="cli"),
        Pause(owner="cli"),
        Resume(owner="cli"),
        Skip(owner="cli"),
        Back(owner="cli"),
        StopReading(owner="cli"),
    ]
    for cmd in commands:
        reply = sim.handle(cmd)
        assert not reply.ok
        assert reply.error is not None
        assert (reply.error.code, reply.error.message) == (
            "invalid_argument",
            "owner 'cli' is reserved",
        )
    assert sim.effects == []


def test_ownerless_speak_belongs_to_cli_and_queues() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    assert _say(sim, None, "c1") == {"chunks": 1, "group": "g2"}
    assert sim.status()["floor"] == {"owner": "a", "group": "g1"}
    _finish(sim)
    assert _played(sim) == ["a1", "c1"]


def test_status_floor_and_waiting() -> None:
    sim = _sim()
    status = sim.status()
    assert (status["floor"], status["waiting"]) == (None, 0)
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    _say(sim, "b", "b2", group="x")
    status = sim.status()
    assert status["floor"] == {"owner": "a", "group": "g1"}
    assert status["waiting"] == 2
    assert status["queued"] == 2


def test_unload_stops_every_owner() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause(owner="a"))
    sim.handle(Unload())
    sim.run_until_idle()
    status = sim.status()
    assert (status["queued"], status["paused"], status["state"]) == (0, False, "cold")
    assert AbortPlayback() in sim.effects
