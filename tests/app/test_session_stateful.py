"""Hypothesis state machine: random interleavings of commands, worker steps, time and callbacks.

Every invariant reads the real Session (through ``Status``) and the real Simulator state; the
machine itself only remembers which epochs a stop/unload/--now/failure made stale.
"""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from voxlektor.app.commands import (
    Back,
    Command,
    Pause,
    Resume,
    SetIdle,
    Skip,
    Speak,
    StopReading,
    Unload,
)
from voxlektor.app.effects import Cue, Generate, PlayCue, PrepareVoice, Reply
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import AudioChunk, VoicePrompt, VoiceRef
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId, EngineId
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import Simulator

_LONG = "word " * 40


def _sentence(tag: str) -> str:
    return f"{tag} {_LONG}end."


TEXTS = st.sampled_from(
    [
        _sentence("one"),
        f"{_sentence('a')} {_sentence('b')} {_sentence('c')}",
        f"{_sentence('x')} {_sentence('BOOM')} {_sentence('y')}",
        "BOOM",
        "Short.",
        "   ",
    ]
)
VOICES = st.sampled_from([None, "v0-default", "m1-calm"])
DURATIONS = st.sampled_from(["0", "5s", "30s", "1m", "10m", "bad"])


class _FlakyEngine(FakeEngine):
    """Text with "BOOM" always fails; ``synth_failures``/``prepare_failures`` fail the next calls.

    Every failure message is unique, so each reading stopped on an error shows as a new
    ``last_error``.
    """

    def __init__(self) -> None:
        super().__init__(sample_rate=50)
        self.synth_failures = 0
        self.prepare_failures = 0
        self.raised = 0

    def _fail(self, what: str) -> NoReturn:
        self.raised += 1
        raise RuntimeError(f"{what} failure #{self.raised}")

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        if "BOOM" in text:
            self._fail("synth")
        if self.synth_failures > 0:
            self.synth_failures -= 1
            self._fail("synth")
        return super().synthesize(text, lang, prompt)

    def prepare_voice(self, ref: VoiceRef) -> VoicePrompt:
        if self.prepare_failures > 0:
            self.prepare_failures -= 1
            self._fail("prepare")
        return super().prepare_voice(ref)


class SessionMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        refs = [VoiceRef(n, Path(f"/voices/{n}/ref.wav"), n) for n in ("m1-calm", "v0-default")]
        clock = FakeClock()
        session = Session(
            Settings(engine=EngineId("fake"), idle_timeout_s=60),
            VoiceCatalog(InMemoryVoiceRepository(refs)),
            clock,
        )
        self.engine = _FlakyEngine()
        self.sim = Simulator(session, self.engine, MemoryOutput(auto_complete=False), clock)
        self.sim.auto_run = False
        # Chunks generated with an epoch below the floor belong to a stopped/replaced reading.
        self.epoch_floor = 0
        self.checked_starts = 0
        self.finished_any = False  # a chunk played to the end (an ownerless back can target it)
        self.error_stops = 0  # readings stopped on an error, seen as a new last_error
        self.play_failures = 0
        # A superset of the voices the current reading uses (reset when a reading starts).
        self.reading_voices: set[str] = set()
        self.spoken_voices: set[str] = set()
        self.checked_replies = 0
        self.retries = 0  # retries the Session really performed, from its failure replies

    def _note_error(self, before: object) -> None:
        if self.sim.status()["last_error"] != before:
            # A reading stopped on an error: everything generated so far is stale now.
            self.error_stops += 1
            self._invalidate_seen_epochs()

    def _invalidate_seen_epochs(self) -> None:
        self._check_new_starts()  # starts so far happened before the invalidation
        epochs = [e.chunk.epoch for e in self.sim.effects if isinstance(e, Generate)]
        self.epoch_floor = max(epochs, default=-1) + 1

    def _check_new_starts(self) -> None:
        for _, epoch in self.sim.started[self.checked_starts :]:
            assert epoch >= self.epoch_floor
        self.checked_starts = len(self.sim.started)

    def _send(self, cmd: Command, *, invalidates: bool = False) -> Reply:
        if invalidates:
            self._invalidate_seen_epochs()
        return self.sim.handle(cmd)

    # ------------------------------------------------------------ commands

    def _reading_idle(self) -> bool:
        status = self.sim.status()
        return status["queued"] == 0 and status["playing"] is False

    def _note_voice(self, voice: str | None, *, replaces: bool) -> None:
        name = voice if voice is not None else "v0-default"
        self.spoken_voices.add(name)
        if replaces:
            self.reading_voices = {name}
        else:
            self.reading_voices.add(name)

    @rule(text=TEXTS, voice=VOICES)
    def speak(self, text: str, voice: str | None) -> None:
        idle = self._reading_idle()
        reply = self._send(Speak(text, voice, "en", None, False))
        assert reply.ok
        if reply.data["chunks"] != 0:
            self._note_voice(voice, replaces=idle)

    @rule(text=TEXTS, voice=VOICES)
    def speak_now(self, text: str, voice: str | None) -> None:
        # An empty --now changes nothing, so it must not raise the floor either.
        invalidates = bool(text.strip())
        assert self._send(Speak(text, voice, "en", None, True), invalidates=invalidates).ok
        if invalidates:
            self._note_voice(voice, replaces=True)

    @rule()
    def pause(self) -> None:
        assert self._send(Pause()).ok

    @rule()
    def resume(self) -> None:
        assert self._send(Resume()).ok

    @rule()
    def skip(self) -> None:
        assert self._send(Skip()).ok

    @rule()
    def back(self) -> None:
        # Back re-queues under a new epoch: older audio is stale once something was re-queued.
        # An ownerless back acts on the floor holder's owner, else on the owner of the chunk
        # that finished last; with neither (e.g. the hold kept the next group from taking the
        # floor and nothing has finished yet) it is a no-op and invalidates nothing.
        self._check_new_starts()
        acts = self.sim.status()["floor"] is not None or self.finished_any
        epochs = [e.chunk.epoch for e in self.sim.effects if isinstance(e, Generate)]
        floor = max(epochs, default=-1) + 1
        effects_before = len(self.sim.effects)
        assert self._send(Back()).ok
        if not acts:
            assert self.sim.effects[effects_before:] == []
            return
        status = self.sim.status()
        if status["queued"] != 0 or status["playing"]:
            self.epoch_floor = floor
            # Back may re-queue chunks of earlier readings, in any voice spoken so far.
            self.reading_voices |= self.spoken_voices

    @rule()
    def stop(self) -> None:
        assert self._send(StopReading(), invalidates=True).ok

    @rule()
    def unload(self) -> None:
        assert self._send(Unload(), invalidates=True).ok

    @rule(duration=DURATIONS)
    def set_idle(self, duration: str) -> None:
        assert self._send(SetIdle(duration)).ok == (duration != "bad")

    # ------------------------------------------------------------ world

    @rule(seconds=st.sampled_from([1, 5, 30, 59, 61, 600]))
    def advance_time(self, seconds: int) -> None:
        self.sim.advance(seconds)

    @rule(jobs=st.integers(min_value=1, max_value=4))
    def worker_steps(self, jobs: int) -> None:
        for _ in range(jobs):
            before = self.sim.status()["last_error"]
            self.sim.step()
            self._note_error(before)

    @rule()
    def drain_worker(self) -> None:
        # Without new commands at most one reading can stop on an error here.
        before = self.sim.status()["last_error"]
        self.sim.run_until_idle()
        self._note_error(before)

    @rule(synth=st.integers(0, 3), prepare=st.integers(0, 2))
    def transient_failures(self, synth: int, prepare: int) -> None:
        self.engine.synth_failures = synth
        self.engine.prepare_failures = prepare

    @rule()
    def complete_play(self) -> None:
        if self.sim.playing is not None:
            self.sim.complete()
            self.finished_any = True

    @rule()
    def fail_play(self) -> None:
        if self.sim.playing is not None:
            self._check_new_starts()
            before = self.sim.status()["last_error"]
            self.play_failures += 1
            self.sim.fail_playback(f"device error #{self.play_failures}")
            self._note_error(before)

    @rule(chunk=st.integers(min_value=0, max_value=30))
    def late_play_callbacks(self, chunk: int) -> None:
        """A done/failed for a chunk that is not playing (e.g. after an abort) is ignored."""
        if self.sim.playing != chunk:
            before = self.sim.status()
            assert self.sim.session.on_play_done(ChunkId(chunk)) == []
            assert self.sim.session.on_play_failed(ChunkId(chunk), "late") == []
            assert self.sim.status() == before

    # ------------------------------------------------------------ invariants

    @invariant()
    def at_most_one_chunk_playing(self) -> None:
        assert self.sim.status()["playing"] == (self.sim.playing is not None)
        assert len(self.sim.output.started) == len(self.sim.started)

    @invariant()
    def generate_only_while_warm(self) -> None:
        assert all(state == "warm" for state, _ in self.sim.generate_states)

    @invariant()
    def no_stale_chunk_started(self) -> None:
        self._check_new_starts()

    @invariant()
    def cold_only_without_a_reading(self) -> None:
        # Unload stops the reading first, so COLD with a reading is never allowed.
        status = self.sim.status()
        if status["state"] == "cold":
            assert status["queued"] == 0
            assert status["playing"] is False
            assert status["paused"] is False

    @invariant()
    def idle_remaining_is_none_while_busy(self) -> None:
        status = self.sim.status()
        busy = status["queued"] != 0 or status["playing"] or status["paused"]
        if busy:
            assert status["idle_remaining_s"] is None

    @invariant()
    def ahead_window_holds(self) -> None:
        assert self.sim.generated_unplayed <= 2

    @invariant()
    def paused_matches_the_output(self) -> None:
        assert self.sim.status()["paused"] == self.sim.output.paused

    @invariant()
    def stopped_cue_iff_a_reading_stopped_on_error(self) -> None:
        assert self.sim.cues.count(Cue.STOPPED) == self.error_stops

    @invariant()
    def never_more_attempts_than_the_ladder(self) -> None:
        attempts = Counter(e.chunk.id for e in self.sim.effects if isinstance(e, Generate))
        assert max(attempts.values(), default=0) <= 3  # first try + two retries

    @invariant()
    def retry_cues_match_real_retries(self) -> None:
        """Each RETRY answers one live failure; no other Session answer ever plays it."""
        for reply in self.sim.failure_replies[self.checked_replies :]:
            retry_cues = reply.effects.count(PlayCue(Cue.RETRY))
            assert retry_cues <= 1
            if retry_cues == 0:
                continue
            assert PlayCue(Cue.STOPPED) not in reply.effects
            if reply.chunk is None:
                # A prompt retry: a fresh rebuild for a voice the current reading still uses.
                assert reply.voice in self.reading_voices
                fresh = [e for e in reply.effects if isinstance(e, PrepareVoice) and e.fresh]
                assert [e.ref.name for e in fresh] == [reply.voice]
            else:
                # A generation retry: the failed chunk belongs to the newest epoch.
                earlier = self.sim.effects[: reply.effects_before]
                newest = max(e.chunk.epoch for e in earlier if isinstance(e, Generate))
                assert reply.chunk.epoch == newest
            self.retries += 1
        self.checked_replies = len(self.sim.failure_replies)
        assert self.sim.cues.count(Cue.RETRY) == self.retries


TestSessionMachine = SessionMachine.TestCase
TestSessionMachine.settings = settings(
    max_examples=400,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
