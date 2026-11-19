"""Pauses between chunks, start caps and steps (S2 spec §2, §4.2, §6a), via the Simulator."""

from pathlib import Path

from voxlektor.app.commands import Back, Pause, Resume, Skip, Speak
from voxlektor.app.effects import Generate
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.pauses import GROUP, INTERRUPT, PARAGRAPH, SENTENCE
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import Simulator


def _sim(*, grace: int = 8, ahead: int = 2, steps: int = 16) -> Simulator:
    refs = [VoiceRef(n, Path(f"/voices/{n}/ref.wav"), n) for n in ("m1-calm", "v0-default")]
    clock = FakeClock()
    settings = Settings(engine=EngineId("fake"), floor_grace_s=grace, ahead=ahead, steps=steps)
    session = Session(settings, VoiceCatalog(InMemoryVoiceRepository(refs)), clock)
    return Simulator(session, FakeEngine(sample_rate=100), MemoryOutput(auto_complete=False), clock)


def _sentence(word: str) -> str:
    return f"{word} {'word ' * 18}end."  # about 100 characters: one sentence per capped chunk


def _para(*words: str) -> str:
    return " ".join(_sentence(w) for w in words)


def _say(  # noqa: PLR0913  # test helper mirroring the Speak fields
    sim: Simulator,
    owner: str | None,
    *paragraphs: str,
    group: str | None = None,
    now: bool = False,
    preempt: bool = False,
    steps: int | None = None,
) -> None:
    text = "\n\n".join(paragraphs)
    cmd = Speak(text, None, "en", None, now, owner=owner, group=group, preempt=preempt, steps=steps)
    reply = sim.handle(cmd)
    assert reply.ok, reply.error
    sim.clock.advance(1)


def _finish(sim: Simulator) -> None:
    while sim.playing is not None:
        sim.complete()


def _played(sim: Simulator) -> list[str]:
    return [sim.chunks[c].text.split()[0] for c in sim.output.started]


def _leads(sim: Simulator) -> list[float]:
    return sim.output.leads


def _generated(sim: Simulator) -> list[Generate]:
    return [e for e in sim.effects if isinstance(e, Generate)]


# ---------------------------------------------------------------- pause choice


def test_lead_for_single_speak_paragraphs() -> None:
    sim = _sim()
    _say(sim, None, _para("s1", "s2", "s3"), _para("p2"))
    _finish(sim)
    assert _played(sim) == ["s1", "s2", "s3", "p2"]
    assert _leads(sim) == [0.0, SENTENCE, SENTENCE, PARAGRAPH]


def test_lead_between_speaks() -> None:
    sim = _sim()
    _say(sim, "a", _para("a1"), group="r1")
    _say(sim, "a", _para("a2"), group="r1")  # the next paragraph of one reply
    _say(sim, "a", _para("a3"), group="r2")  # another reply: a group end
    _finish(sim)
    assert _played(sim) == ["a1", "a2", "a3"]
    assert _leads(sim) == [0.0, PARAGRAPH, GROUP]


def test_lead_owner_change() -> None:
    sim = _sim()
    _say(sim, "a", _para("a1"), group="r1")
    _say(sim, "b", _para("b1"), group="r1")
    _finish(sim)
    assert _played(sim) == ["a1", "b1"]
    assert _leads(sim) == [0.0, GROUP]


def test_lead_auto_groups_cli() -> None:
    sim = _sim()
    _say(sim, None, _para("c1"))
    _say(sim, None, _para("c2"))  # plain CLI: every speak gets its own automatic group
    _say(sim, "a", _para("a1"))
    _say(sim, "a", _para("a2"))
    _say(sim, "a", _para("a3"), group="named")
    _finish(sim)
    assert _played(sim) == ["c1", "c2", "a1", "a2", "a3"]
    assert _leads(sim) == [0.0, PARAGRAPH, GROUP, PARAGRAPH, GROUP]


def test_lead_zero_after_idle() -> None:
    sim = _sim()
    _say(sim, "a", _para("a1"))
    _finish(sim)
    _say(sim, "a", _para("a2"))
    _finish(sim)
    assert _leads(sim) == [0.0, 0.0]


def test_skip_during_gap_uses_interrupt_pause() -> None:
    sim = _sim()
    sim.auto_run = False
    _say(sim, None, _para("x1"), _para("x2"), _para("x3"))
    while sim.playing is None:
        sim.step()
    sim.complete()  # x2 is still being generated: a gap
    assert sim.playing is None
    assert sim.handle(Skip()).ok
    sim.run_until_idle()
    assert _played(sim) == ["x1", "x3"]
    assert _leads(sim) == [0.0, INTERRUPT]


def test_skip_while_playing_uses_interrupt_pause() -> None:
    sim = _sim()
    _say(sim, None, _para("x1"), _para("x2"), _para("x3"))
    assert sim.handle(Skip()).ok
    _finish(sim)
    assert _played(sim) == ["x1", "x2", "x3"]
    assert _leads(sim) == [0.0, INTERRUPT, PARAGRAPH]


def test_back_uses_interrupt_pause() -> None:
    sim = _sim()
    _say(sim, None, _para("x1"), _para("x2"))
    sim.complete()
    assert sim.handle(Back()).ok
    _finish(sim)
    assert _played(sim) == ["x1", "x2", "x1", "x2"]
    assert _leads(sim) == [0.0, PARAGRAPH, INTERRUPT, PARAGRAPH]


def test_back_from_idle_starts_without_lead() -> None:
    sim = _sim()
    _say(sim, None, _para("x1"))
    _finish(sim)
    assert sim.handle(Back()).ok
    _finish(sim)
    assert _played(sim) == ["x1", "x1"]
    assert _leads(sim) == [0.0, 0.0]


def test_preempt_uses_interrupt_pause() -> None:
    sim = _sim()
    _say(sim, "a", _para("a1"), _para("a2"))
    _say(sim, "b", _para("b1"), preempt=True)
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "a1", "a2"]
    assert _leads(sim) == [0.0, INTERRUPT, GROUP, PARAGRAPH]


def test_yield_uses_interrupt_pause() -> None:
    sim = _sim()
    _say(sim, "a", _para("a1"))
    _say(sim, "b", _para("b1"))
    assert sim.handle(Pause(owner="a")).ok
    sim.advance(10)  # the grace runs out: a yields to b
    _finish(sim)
    assert _played(sim) == ["a1", "b1"]
    assert sim.handle(Resume(owner="a")).ok
    _finish(sim)
    assert _played(sim) == ["a1", "b1", "a1"]  # a's restart: another owner had the floor
    assert _leads(sim) == [0.0, INTERRUPT, GROUP]


def test_now_replacing_the_whole_reading_starts_without_lead() -> None:
    sim = _sim()
    _say(sim, None, _para("x1"), _para("x2"))
    _say(sim, None, _para("y1"), now=True)
    _finish(sim)
    assert _played(sim) == ["x1", "y1"]
    assert _leads(sim) == [0.0, 0.0]


def test_now_beside_another_owner_uses_interrupt_pause() -> None:
    sim = _sim()
    _say(sim, "a", _para("a1"), _para("a2"))
    _say(sim, "b", _para("b1"))
    _say(sim, "a", _para("n1"), now=True)
    _finish(sim)
    assert _played(sim) == ["a1", "n1", "b1"]
    assert _leads(sim) == [0.0, INTERRUPT, GROUP]


# ---------------------------------------------------------------- start caps


def test_caps_only_from_idle() -> None:
    sim = _sim()
    _say(sim, "a", _para("s1", "s2", "s3", "s4"), group="r")
    _say(sim, "a", _para("t1", "t2", "t3", "t4"), group="r")  # appended: normal sizes
    _finish(sim)
    texts = [g.chunk.text for g in _generated(sim)]
    assert [t.split()[0] for t in texts] == ["s1", "s2", "s3", "t1", "t4"]
    assert texts[0] == _sentence("s1")
    assert texts[1] == _sentence("s2")
    assert texts[3] == _para("t1", "t2", "t3")


def test_caps_again_after_idle() -> None:
    sim = _sim()
    _say(sim, "a", _para("s1", "s2", "s3"))
    _finish(sim)
    _say(sim, "a", _para("t1", "t2", "t3"))
    _finish(sim)
    firsts = [g.chunk.text.split()[0] for g in _generated(sim)]
    assert firsts == ["s1", "s2", "s3", "t1", "t2", "t3"]


def test_now_replacing_the_whole_reading_is_capped() -> None:
    sim = _sim()
    _say(sim, None, _para("x1"), _para("x2"))
    _say(sim, None, _para("y1", "y2", "y3"), now=True)
    _finish(sim)
    assert _played(sim) == ["x1", "y1", "y2", "y3"]


def test_chunks_carry_how_their_text_goes_on() -> None:
    sim = _sim()
    _say(sim, "a", _para("s1", "s2", "s3"), group="r")
    _say(sim, "a", _para("t1"), group="r")
    _finish(sim)
    chunks = [g.chunk for g in _generated(sim)]
    assert [c.ends for c in chunks] == ["sentence", "sentence", "paragraph", "paragraph"]


# ---------------------------------------------------------------- steps


def test_steps_reach_chunks() -> None:
    sim = _sim(steps=12)
    _say(sim, None, _para("d1"))
    _say(sim, None, _para("e1"), steps=8)
    _finish(sim)
    assert [g.chunk.steps for g in _generated(sim)] == [12, 8]


def test_steps_out_of_range_are_rejected() -> None:
    sim = _sim()
    for steps in (3, 33):
        reply = sim.handle(Speak("hi", None, "en", None, False, steps=steps))
        assert not reply.ok
        assert reply.error is not None
        assert reply.error.code == "invalid_setting"
        assert reply.error.message == "steps must be between 4 and 32"
    assert sim.output.started == []
