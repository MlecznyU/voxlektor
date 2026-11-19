"""Events and tags from the Session (spec §4.4, §5.2), via the Simulator."""

import re
from pathlib import Path
from typing import cast

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from voxlektor.app.commands import Back, Pause, Resume, Skip, Speak, StopReading, Unload
from voxlektor.app.effects import DropAudio, Generate
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId, EngineId
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import Simulator

FAKE_FAILURE = "fake engine failure on 'BOOM'"
EVENT_NAMES = {
    "state",
    "floor",
    "waiting",
    "tag_started",
    "tag_finished",
    "paused",
    "resumed",
    "drained",
    "error",
}

type Ev = tuple[str, dict[str, object]]


def _sim(*, grace: int = 8, ahead: int = 2) -> Simulator:
    refs = [VoiceRef(n, Path(f"/voices/{n}/ref.wav"), n) for n in ("m1-calm", "v0-default")]
    clock = FakeClock()
    config = Settings(engine=EngineId("fake"), floor_grace_s=grace, ahead=ahead)
    session = Session(config, VoiceCatalog(InMemoryVoiceRepository(refs)), clock)
    engine = FakeEngine(sample_rate=100, fail_on="BOOM")
    return Simulator(session, engine, MemoryOutput(auto_complete=False), clock)


def _text(*words: str) -> str:
    return "\n\n".join(f"{word} {'word ' * 18}end." for word in words)


def _say(  # noqa: PLR0913  # test helper mirroring the Speak fields
    sim: Simulator,
    owner: str | None,
    *words: str,
    tag: str | None = None,
    group: str | None = None,
    now: bool = False,
    preempt: bool = False,
) -> None:
    cmd = Speak(
        _text(*words), None, "en", None, now, owner=owner, group=group, tag=tag, preempt=preempt
    )
    reply = sim.handle(cmd)
    assert reply.ok, reply.error
    sim.clock.advance(1)  # distinct ages


def _events(sim: Simulator, *names: str) -> list[Ev]:
    return [(e.event, dict(e.data)) for e in sim.events if not names or e.event in names]


def _tags(sim: Simulator) -> list[tuple[str, object, object]]:
    """(event, tag, reason) of every tag event, in order."""
    return [
        (name, data["tag"], data.get("reason"))
        for name, data in _events(sim, "tag_started", "tag_finished")
    ]


def _finish(sim: Simulator) -> None:
    while sim.playing is not None:
        sim.complete()


# ---------------------------------------------------------------- tags


def test_tag_done() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1", group="r1")
    _finish(sim)
    assert _tags(sim) == [("tag_started", "t1", None), ("tag_finished", "t1", "done")]
    started, finished = _events(sim, "tag_started", "tag_finished")
    assert started[1] == {"owner": "a", "group": "r1", "tag": "t1"}
    assert finished[1] == {"owner": "a", "group": "r1", "tag": "t1", "reason": "done"}


def test_tag_skipped() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1")
    _say(sim, "a", "q1", tag="t2")
    sim.handle(Skip(owner="a"))  # p1: t1 still has p2
    assert _tags(sim) == [("tag_started", "t1", None)]
    sim.handle(Skip())  # p2: the last chunk of t1
    assert _tags(sim)[1:] == [("tag_finished", "t1", "skipped"), ("tag_started", "t2", None)]


def test_tag_stopped() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1")
    sim.handle(StopReading(owner="a"))
    _say(sim, "b", "q1", tag="t2")
    sim.handle(Unload())
    assert _tags(sim) == [
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "stopped"),
        ("tag_started", "t2", None),
        ("tag_finished", "t2", "stopped"),
    ]


def test_tag_replaced() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1")
    _say(sim, "a", "q1", tag="t2")
    _say(sim, "a", "n1", tag="t3", now=True)
    assert _tags(sim) == [
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "replaced"),
        ("tag_finished", "t2", "replaced"),
        ("tag_started", "t3", None),
    ]


def test_tag_back() -> None:
    sim = _sim()
    _say(sim, "a", "p1", tag="t1")
    _say(sim, "a", "q1", "q2", tag="t2")
    sim.complete()  # p1: t1 done; q1 plays
    sim.complete()  # q1; q2 plays
    sim.handle(Back(owner="a"))  # q1 again: t2 started and unfinished -> back, then afresh
    assert _tags(sim)[2:] == [
        ("tag_started", "t2", None),
        ("tag_finished", "t2", "back"),
        ("tag_started", "t2", None),
    ]
    sim.handle(Pause(owner="a"))
    sim.handle(Back(owner="a"))  # p1 again: t2 is interrupted, t1 had finished
    sim.handle(Resume(owner="a"))
    _finish(sim)
    assert _tags(sim)[5:] == [
        ("tag_finished", "t2", "back"),
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "done"),
        ("tag_started", "t2", None),
        ("tag_finished", "t2", "done"),
    ]


def test_back_before_a_tag_starts_leaves_it_alone() -> None:
    sim = _sim()
    _say(sim, "a", "p1", tag="t1", group="r1")
    _say(sim, "a", "p2", tag="t2", group="r1")
    sim.handle(Pause(owner="a"))
    sim.complete()  # p1 ends while paused: t1 done, t2 not started
    sim.handle(Back(owner="a"))
    sim.handle(Resume(owner="a"))
    _finish(sim)
    assert _tags(sim) == [
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "done"),
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "done"),
        ("tag_started", "t2", None),
        ("tag_finished", "t2", "done"),
    ]


def test_back_from_idle_replays_a_finished_tag() -> None:
    sim = _sim()
    _say(sim, None, "p1", tag="t1")
    _finish(sim)
    sim.handle(Back())
    _finish(sim)
    assert _tags(sim) == [
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "done"),
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "done"),
    ]


def test_back_reopened_tag_stopped_before_it_plays_owes_nothing() -> None:
    sim = _sim()
    _say(sim, "a", "p1", tag="t1")
    _say(sim, "a", "q1")
    sim.complete()  # p1: t1 done
    sim.handle(Pause())
    sim.handle(Back())  # p1 queued again, paused
    sim.handle(StopReading())
    assert _tags(sim) == [("tag_started", "t1", None), ("tag_finished", "t1", "done")]


def test_tag_error() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "BOOM", tag="t1")
    _say(sim, "b", "q1", tag="t2")
    _finish(sim)
    assert ("tag_finished", "t1", "error") in _tags(sim)
    assert ("tag_finished", "t2", "done") in _tags(sim)
    assert _events(sim, "error") == [
        ("error", {"owner": "a", "group": None, "code": "engine_failed", "message": FAKE_FAILURE})
    ]


def test_daemon_wide_failure_errors_every_tag() -> None:
    sim = _sim()
    _say(sim, "a", "p1", tag="t1")
    _say(sim, "b", "q1", tag="t2")
    sim.fail_playback("device gone")
    assert _tags(sim)[1:] == [("tag_finished", "t1", "error"), ("tag_finished", "t2", "error")]
    assert _events(sim, "error") == [
        ("error", {"owner": None, "group": None, "code": "audio_failed", "message": "device gone"})
    ]


def test_unstarted_tag_still_finishes() -> None:
    sim = _sim()
    _say(sim, "a", "p1", tag="t1")
    _say(sim, "b", "q1", tag="t2")
    sim.handle(StopReading(owner="b"))
    assert _tags(sim) == [("tag_started", "t1", None), ("tag_finished", "t2", "stopped")]


def test_empty_plan_emits_no_tag() -> None:
    sim = _sim()
    assert sim.handle(Speak("  ", None, "en", None, False, owner="a", tag="t0")).ok
    assert sim.events == []


def test_yield_and_preempt_restarts_do_not_restart_the_tag() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1")
    _say(sim, "b", "q1", preempt=True)  # a is interrupted; p1 restarts later
    sim.complete()  # q1
    sim.handle(Pause(owner="a"))
    _say(sim, "c", "r1")
    sim.advance(10)  # a yields to c after its grace
    sim.handle(Resume(owner="a"))
    _finish(sim)
    assert _tags(sim) == [("tag_started", "t1", None), ("tag_finished", "t1", "done")]
    floors = [(d["owner"], d["group"]) for _, d in _events(sim, "floor")]
    assert [owner for owner, _ in floors] == ["a", "b", "a", "c", "a"]


# ---------------------------------------------------------------- floor, waiting, drained


def test_waiting_position_updates() -> None:
    sim = _sim()
    _say(sim, "a", "p1", group="r1")
    _say(sim, "b", "q1", group="s1")
    _say(sim, "c", "r1", group="u1")
    assert _events(sim, "floor", "waiting") == [
        ("floor", {"owner": "a", "group": "r1"}),
        ("waiting", {"owner": "b", "group": "s1", "position": 1}),
        ("waiting", {"owner": "c", "group": "u1", "position": 2}),
    ]
    sim.complete()
    assert _events(sim, "floor", "waiting")[3:] == [
        ("floor", {"owner": "b", "group": "s1"}),
        ("waiting", {"owner": "c", "group": "u1", "position": 1}),
    ]
    sim.handle(Pause(owner="c"))  # passed over: no position, no new waiting event
    sim.handle(Resume(owner="c"))
    assert _events(sim, "waiting")[-1] == (
        "waiting",
        {"owner": "c", "group": "u1", "position": 1},
    )
    assert len(_events(sim, "waiting")) == 4


def test_no_waiting_behind_the_owners_own_group() -> None:
    sim = _sim()
    _say(sim, "a", "p1", group="r1")
    _say(sim, "a", "p2", group="r2")
    _say(sim, "b", "q1", group="s1")
    assert _events(sim, "waiting") == [("waiting", {"owner": "b", "group": "s1", "position": 2})]
    sim.complete()
    assert _events(sim, "waiting")[1:] == [
        ("waiting", {"owner": "b", "group": "s1", "position": 1}),
    ]
    _say(sim, "c", "u1", group="v1", preempt=True)  # r2 is now behind another owner
    assert _events(sim, "waiting")[2:] == [
        ("waiting", {"owner": "a", "group": "r2", "position": 1}),
        ("waiting", {"owner": "b", "group": "s1", "position": 2}),
    ]


def test_preempt_lifts_the_owners_pause_with_an_event() -> None:
    sim = _sim()
    _say(sim, "a", "p1")
    _say(sim, "b", "q1")
    sim.handle(Pause(owner="b"))
    _say(sim, "b", "q2", preempt=True)
    assert _events(sim, "paused", "resumed") == [
        ("paused", {"owner": "b", "group": None}),
        ("resumed", {"owner": "b", "group": None}),
    ]


def test_regrouped_holder_gets_the_floor_event_again() -> None:
    sim = _sim()
    _say(sim, "a", "p1", group="r1")
    _say(sim, "a", "p2", group="r1", now=True)
    assert _events(sim, "floor") == [("floor", {"owner": "a", "group": "r1"})] * 2


def test_drained_per_owner_while_other_waits() -> None:
    sim = _sim()
    _say(sim, "a", "p1")
    _say(sim, "b", "q1")
    sim.complete()
    assert _events(sim, "drained") == [("drained", {"owner": "a", "group": None})]
    sim.complete()
    assert _events(sim, "drained")[1:] == [
        ("drained", {"owner": "b", "group": None}),
        ("drained", {"owner": None, "group": None}),
    ]


def test_drained_daemon_wide() -> None:
    sim = _sim()
    _say(sim, None, "p1")
    _say(sim, None, "p2")
    _finish(sim)
    assert _events(sim, "drained") == [
        ("drained", {"owner": "cli", "group": None}),
        ("drained", {"owner": None, "group": None}),
    ]
    _say(sim, "a", "q1", now=True)  # nothing replaced: no drained in between
    sim.handle(StopReading())
    assert len(_events(sim, "drained")) == 4


def test_now_replacing_the_only_reading_is_not_a_drain() -> None:
    sim = _sim()
    _say(sim, "a", "p1")
    _say(sim, "a", "n1", now=True)
    assert _events(sim, "drained") == []


# ---------------------------------------------------------------- state, pauses


def test_state_events() -> None:
    sim = _sim()
    _say(sim, "a", "p1")
    _finish(sim)
    sim.handle(Unload())
    states = [d["state"] for _, d in _events(sim, "state")]
    assert states == ["loading", "warm", "unloading", "cold"]
    assert all(d["owner"] is None and d["group"] is None for _, d in _events(sim, "state"))


def test_paused_and_resumed_events() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2")
    sim.handle(Pause(owner="a"))
    sim.handle(Pause(owner="a"))  # already paused: no event
    sim.handle(Resume(owner="a"))
    sim.handle(Pause())
    sim.handle(Resume())
    assert _events(sim, "paused", "resumed") == [
        ("paused", {"owner": "a", "group": None}),
        ("resumed", {"owner": "a", "group": None}),
        ("paused", {"owner": None, "group": None}),
        ("resumed", {"owner": None, "group": None}),
    ]


def test_resumed_on_now_lifts_pause() -> None:
    sim = _sim()
    _say(sim, "a", "p1")
    sim.handle(Pause(owner="a"))
    _say(sim, "a", "n1", now=True)
    assert _events(sim, "paused", "resumed")[-1] == ("resumed", {"owner": "a", "group": None})
    sim.handle(Pause())
    _say(sim, None, "c1", now=True)  # a cli now lifts the hold
    assert _events(sim, "resumed")[-1] == ("resumed", {"owner": None, "group": None})
    sim.handle(Pause(owner="a"))
    sim.handle(StopReading(owner="a"))
    assert _events(sim, "resumed")[-1] == ("resumed", {"owner": "a", "group": None})
    sim.handle(Pause())
    _finish(sim)  # the daemon going idle lifts the hold
    assert _events(sim, "resumed")[-1] == ("resumed", {"owner": None, "group": None})


def test_owner_resume_lifts_hold_only_for_holder() -> None:
    sim = _sim()
    _say(sim, "a", "a1")
    _say(sim, "b", "b1")
    sim.handle(Pause())
    sim.handle(Resume(owner="b"))
    assert _events(sim, "resumed") == []
    sim.handle(Resume(owner="a"))
    assert _events(sim, "resumed") == [("resumed", {"owner": None, "group": None})]


# ---------------------------------------------------------------- status


def test_status_tag() -> None:
    sim = _sim()
    assert sim.status()["tag"] is None
    _say(sim, "a", "p1", tag="t1", group="r1")
    _say(sim, "a", "q1", tag="t2", group="r1")
    assert sim.status()["tag"] == "t1"  # playing
    sim.handle(Pause(owner="a"))
    sim.handle(Skip(owner="a"))  # q1 is the floor group's next chunk, not playing
    assert sim.status()["playing"] is False
    assert sim.status()["tag"] == "t2"
    _say(sim, "a", "u1", now=True)
    assert sim.status()["tag"] is None


def test_status_groups_order() -> None:
    sim = _sim()
    assert sim.status()["groups"] == []
    _say(sim, "a", "p1", group="r1")
    _say(sim, "b", "q1", group="s1")
    _say(sim, "c", "u1", group="v1")
    _say(sim, "d", "w1", group="x1")
    sim.handle(Pause(owner="b"))
    assert sim.status()["groups"] == [
        {"owner": "a", "group": "r1", "position": 0, "paused": False},
        {"owner": "c", "group": "v1", "position": 1, "paused": False},
        {"owner": "d", "group": "x1", "position": 2, "paused": False},
        {"owner": "b", "group": "s1", "position": None, "paused": True},
    ]


# ---------------------------------------------------------------- tag navigation (0.4.0)


def _played(sim: Simulator) -> list[str]:
    return [sim.chunks[c].text.split()[0] for c in sim.output.started]


def _generated(sim: Simulator) -> list[ChunkId]:
    return [e.chunk.id for e in sim.effects if isinstance(e, Generate)]


def test_tag_skip_drops_the_run_and_plays_the_ready_next_tag() -> None:
    sim = _sim(ahead=4)
    _say(sim, "a", "p1", "p2", "p3", tag="t1", group="r")
    _say(sim, "a", "q1", "q2", tag="t2", group="r")
    generated = _generated(sim)
    sim.handle(Skip(owner="a", unit="tag"))
    assert sim.playing in generated  # q1's audio was ready: no new generation
    assert len(_generated(sim)) == len(generated)
    assert _tags(sim) == [
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "skipped"),
        ("tag_started", "t2", None),
    ]
    _finish(sim)
    assert _played(sim) == ["p1", "q1", "q2"]


def test_tag_skip_of_an_untagged_chunk_skips_one_chunk() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2")
    sim.handle(Skip(owner="a", unit="tag"))
    _finish(sim)
    assert _played(sim) == ["p1", "p2"]


def test_tag_skip_of_the_last_run_hands_the_floor_on() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1")
    _say(sim, "b", "q1")
    sim.handle(Skip(owner="a", unit="tag"))
    _finish(sim)
    assert _played(sim) == ["p1", "q1"]


def test_tag_back_restarts_the_run_and_keeps_the_ready_audio() -> None:
    sim = _sim(ahead=4)
    _say(sim, "a", "p1", "p2", "p3", tag="t1", group="r")
    _say(sim, "a", "q1", tag="t2", group="r")
    sim.complete()  # p1; p2 plays, p3 and q1 are ready
    before = len(sim.effects)
    sim.handle(Back(owner="a", unit="tag"))
    after = sim.effects[before:]
    assert not [e for e in after if isinstance(e, DropAudio)]
    assert len([e for e in after if isinstance(e, Generate)]) == 2  # p1 and p2 again
    _finish(sim)
    assert _played(sim) == ["p1", "p2", "p1", "p2", "p3", "q1"]
    assert _tags(sim) == [
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "back"),
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "done"),
        ("tag_started", "t2", None),
        ("tag_finished", "t2", "done"),
    ]


def test_tag_back_at_the_start_of_a_run_replays_the_previous_run() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1", group="r")
    _say(sim, "a", "q1", "q2", tag="t2", group="r")
    sim.complete()
    sim.complete()  # q1 plays: nothing of t2 has finished yet
    sim.handle(Back(owner="a", unit="tag"))
    _finish(sim)
    assert _played(sim) == ["p1", "p2", "q1", "p1", "p2", "q1", "q2"]
    assert _tags(sim)[2:] == [
        ("tag_started", "t2", None),
        ("tag_finished", "t2", "back"),
        ("tag_started", "t1", None),
        ("tag_finished", "t1", "done"),
        ("tag_started", "t2", None),
        ("tag_finished", "t2", "done"),
    ]


def test_tag_back_of_a_first_run_restarts_it() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1")
    sim.handle(Back(owner="a", unit="tag"))  # no history: p1 again
    _finish(sim)
    assert _played(sim) == ["p1", "p1", "p2"]


def test_tag_back_from_idle_replays_the_last_run() -> None:
    sim = _sim()
    _say(sim, "a", "p1", "p2", tag="t1")
    _say(sim, "a", "q1", "q2", tag="t2")
    _finish(sim)
    sim.handle(Back(owner="a", unit="tag"))  # a holds nothing: no-op
    sim.handle(Back(unit="tag"))  # the last owner's last run
    _finish(sim)
    assert _played(sim) == ["p1", "p2", "q1", "q2", "q1", "q2"]


def test_tag_back_moves_one_chunk_around_untagged_chunks() -> None:
    sim = _sim()
    _say(sim, "a", "u1", "u2", group="r")
    _say(sim, "a", "q1", tag="t2", group="r")
    sim.complete()
    sim.complete()  # q1 plays; the run before it is untagged: only u2 comes back
    sim.handle(Back(owner="a", unit="tag"))
    sim.complete()  # u2
    sim.handle(Back(owner="a", unit="tag"))  # q1's head is tagged, u2 untagged: u2 again
    _finish(sim)
    sim.handle(Back(unit="tag"))  # idle, last played q1 (tagged): t2 only
    _finish(sim)
    assert _played(sim) == ["u1", "u2", "q1", "u2", "q1", "u2", "q1", "q1"]


def test_tag_back_of_an_untagged_head_goes_back_one_chunk() -> None:
    sim = _sim()
    _say(sim, "a", "u1", "u2")
    sim.complete()
    sim.handle(Back(owner="a", unit="tag"))
    _finish(sim)
    assert _played(sim) == ["u1", "u2", "u1", "u2"]


# ---------------------------------------------------------------- property


OWNERS = st.sampled_from([None, "a", "b"])
WORDS = st.sampled_from([("p",), ("p", "q"), ("p", "q", "r"), ("p", "q", "r"), ("p", "BOOM"), ()])
OPS = st.one_of(
    st.tuples(st.just("speak"), OWNERS, WORDS, st.booleans(), st.booleans(), st.booleans()),
    st.tuples(
        st.sampled_from(
            ["stop", "skip", "back", "back", "pause", "resume", "skip_tag", "back_tag", "back_tag"]
        ),
        OWNERS,
    ),
    st.tuples(st.sampled_from(["complete"] * 6 + ["fail_play", "advance", "advance"])),
)


def _run(sim: Simulator, op: tuple[object, ...], index: int) -> str | None:
    """Apply ``op``; return the tag of an accepted tagged speak."""
    name = op[0]
    if name == "speak":
        _, owner, words, now, preempt, tagged = op
        tag = f"t{index}" if tagged else None
        text = _text(*cast("tuple[str, ...]", words)) if words else "   "
        cmd = Speak(
            text, None, "en", None, bool(now), owner=owner_of(owner), tag=tag, preempt=bool(preempt)
        )
        reply = sim.handle(cmd)
        return tag if reply.data.get("chunks") else None
    owner = owner_of(op[1]) if len(op) > 1 else None
    commands = {"stop": StopReading, "skip": Skip, "back": Back, "pause": Pause, "resume": Resume}
    if name in commands:
        sim.handle(commands[str(name)](owner))
    elif name == "skip_tag":
        sim.handle(Skip(owner, unit="tag"))
    elif name == "back_tag":
        sim.handle(Back(owner, unit="tag"))
    elif name == "advance":
        sim.advance(5)
    elif sim.playing is not None:
        if name == "complete":
            sim.complete()
        else:
            sim.fail_playback("device gone")
    return None


def owner_of(value: object) -> str | None:
    return None if value is None else str(value)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(OPS, max_size=40))
def test_tags_end_exactly_once(ops: list[tuple[object, ...]]) -> None:
    sim = _sim()
    accepted: list[str] = []
    for index, op in enumerate(ops):
        tag = _run(sim, op, index)  # a SimulationError is an AssertionError: it fails the test
        if tag is not None:
            accepted.append(tag)
        sim.clock.advance(0.5)
    sim.handle(StopReading())
    for event in sim.events:
        assert event.event in EVENT_NAMES
        assert "owner" in event.data
        assert "group" in event.data
    backs = any(op[0] in {"back", "back_tag"} for op in ops)
    for tag in accepted:
        marks = "".join(
            "S" if name == "tag_started" else "F"
            for name, data in _events(sim, "tag_started", "tag_finished")
            if data["tag"] == tag
        )
        assert re.fullmatch(r"S?F(SF)*", marks), (tag, marks)
        if not backs:
            assert marks in {"F", "SF"}, (tag, marks)
    assert sim.status()["groups"] == []
