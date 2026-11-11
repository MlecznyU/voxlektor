"""Deterministic Session driver: an immediate engine worker, a device-less output, a fake clock.

No threads and no sleeps. Effects are executed in order: engine effects go to a FIFO worker
queue (like ``runtime.engine_worker``), output effects go to ``MemoryOutput``, and output
callbacks are queued and delivered after the current batch (like ``call_soon_threadsafe``).

With ``auto_run`` (the default) the worker queue is drained after every action; set it to
False to hold engine jobs back and release them with ``step()`` or ``run_until_idle()``.

The simulator also checks the Session: ``SimulationError`` is raised when

* a chunk starts while another plays, or while the output is paused;
* a chunk starts without generated audio;
* ``Generate`` is emitted while the engine is not WARM;
* ``Generate`` runs for a voice that has no prompt in the current engine load;
* ``Status`` produces effects.

It records what tests need to check the rest: every effect, the cues played, the epoch of each
started chunk, the Session state at each ``Generate`` and every engine failure together with
the Session's answer (``failure_replies``).
"""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import assert_never

import numpy as np

from voxlektor.app.commands import Command, Status
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
    PlayCue,
    PrepareVoice,
    Reply,
    ResumePlayback,
    ScheduleWake,
    StartPlayback,
    UnloadEngine,
)
from voxlektor.app.session import Session
from voxlektor.contracts.engine import AudioChunk, VoicePrompt, VoiceRef
from voxlektor.domain.reading import Chunk
from voxlektor.domain.types import ChunkId, Epoch
from voxlektor.testing.fakes import FakeClock, FakeEngine, MemoryOutput

type _Job = LoadEngine | UnloadEngine | PrepareVoice | Generate


class SimulationError(AssertionError):
    """The Session asked for something the runtime could never do."""


@dataclass(frozen=True)
class FailureReply:
    """An engine failure reported to the Session and the effects the Session answered with.

    ``chunk`` is None for a prompt failure (``on_prompt_failed``) and the failed chunk for a
    generation failure (``on_gen_failed``). ``effects_before`` is ``len(Simulator.effects)`` at
    the moment of the failure, so a check can see which ``Generate`` effects came before it.
    """

    voice: str
    chunk: Chunk | None
    effects_before: int
    effects: tuple[Effect, ...]


class Simulator:
    def __init__(
        self, session: Session, engine: FakeEngine, output: MemoryOutput, clock: FakeClock
    ) -> None:
        self.session = session
        self.engine = engine
        self.output = output
        self.clock = clock
        self.auto_run = True
        self.effects: list[Effect] = []  # every effect executed, in order
        self.wakes: list[float] = []  # pending wake times (several may be outstanding)
        self.audio: dict[ChunkId, AudioChunk] = {}  # generated and not started yet
        self.chunks: dict[ChunkId, Chunk] = {}  # every chunk ever sent to Generate
        self.prompts: dict[str, VoicePrompt] = {}
        self.playing: ChunkId | None = None
        self.started: list[tuple[ChunkId, Epoch]] = []  # chunk and the epoch of its audio
        self.generate_states: list[tuple[str, Generate]] = []  # Session state at each Generate
        self.cues: list[Cue] = []  # every cue played, in order
        self.failure_replies: list[FailureReply] = []  # engine failures and the Session's answers
        self.max_generated_unplayed = 0
        self.exited = False
        self._audio_epoch: dict[ChunkId, Epoch] = {}
        self._jobs: deque[_Job] = deque()
        self._callbacks: deque[Callable[[], list[Effect]]] = deque()
        output.set_callbacks(self._on_done, self._on_failed)

    # ------------------------------------------------------------ driving

    def handle(self, cmd: Command) -> Reply:
        outcome = self.session.handle(cmd)
        self.apply(outcome.effects)
        if outcome.reply is None:
            raise SimulationError(f"{cmd!r} produced no reply")
        return outcome.reply

    def status(self) -> Mapping[str, object]:
        outcome = self.session.handle(Status())
        if outcome.effects or outcome.reply is None:
            raise SimulationError(f"Status must only reply, got {outcome!r}")
        return outcome.reply.data

    def apply(self, effects: list[Effect]) -> None:
        """Execute ``effects`` as the runtime would, then deliver whatever follows."""
        self._apply(effects)
        self._settle()

    def step(self) -> bool:
        """Run one engine job (and what follows from it); False when the queue is empty."""
        if not self._jobs:
            return False
        self._run_job(self._jobs.popleft())
        self._settle()
        return True

    def run_until_idle(self) -> None:
        self._settle()
        while self.step():
            pass

    def advance(self, seconds: float) -> None:
        """Move the clock forward, firing every wake that falls due on the way, in order."""
        end = self.clock.now() + seconds
        while due := [at for at in self.wakes if at <= end]:
            at = min(due)
            self.wakes.remove(at)
            self.clock.advance(max(0.0, at - self.clock.now()))
            self._apply(self.session.on_wake())
            self._settle()
        self.clock.advance(max(0.0, end - self.clock.now()))
        self._settle()

    def complete(self, chunk: ChunkId | None = None) -> None:
        """Report ``chunk`` (default: the playing one) as played to the end."""
        self.output.complete(self._target(chunk))
        self._settle()

    def fail_playback(self, msg: str, chunk: ChunkId | None = None) -> None:
        self.output.fail(self._target(chunk), msg)
        self._settle()

    @property
    def generated_unplayed(self) -> int:
        """Chunks of the newest epoch that are generated or being generated, but not started."""
        epochs = [self._audio_epoch[cid] for cid in self.audio]
        epochs += [job.chunk.epoch for job in self._jobs if isinstance(job, Generate)]
        return epochs.count(max(epochs)) if epochs else 0

    # ------------------------------------------------------------ internals

    def _target(self, chunk: ChunkId | None) -> ChunkId:
        target = self.playing if chunk is None else chunk
        if target is None:
            raise SimulationError("nothing is playing")
        return target

    def _on_done(self, chunk: ChunkId) -> None:
        if chunk == self.playing:
            self.playing = None
        self._callbacks.append(lambda: self.session.on_play_done(chunk))

    def _on_failed(self, chunk: ChunkId, msg: str) -> None:
        if chunk == self.playing:
            self.playing = None
        self._callbacks.append(lambda: self.session.on_play_failed(chunk, msg))

    def _settle(self) -> None:
        while True:
            if self._callbacks:
                self._apply(self._callbacks.popleft()())
            elif self._fire_due_wake():
                pass
            elif self.auto_run and self._jobs:
                self._run_job(self._jobs.popleft())
            else:
                return

    def _fire_due_wake(self) -> bool:
        due = [at for at in self.wakes if at <= self.clock.now()]
        if not due:
            return False
        self.wakes.remove(min(due))
        self._apply(self.session.on_wake())
        return True

    def _apply(self, effects: list[Effect]) -> None:
        for effect in effects:
            self.effects.append(effect)
            self._execute(effect)
            self.max_generated_unplayed = max(self.max_generated_unplayed, self.generated_unplayed)

    def _execute(self, effect: Effect) -> None:
        match effect:
            case LoadEngine() | UnloadEngine() | PrepareVoice():
                self._jobs.append(effect)
            case Generate(chunk):
                self._submit_generate(effect, chunk)
            case StartPlayback(chunk):
                self._start(chunk)
            case PausePlayback() | ResumePlayback() | AbortPlayback():
                self._control_output(effect)
            case DropAudio(chunks):
                for cid in chunks:
                    self.audio.pop(cid, None)
                    self._audio_epoch.pop(cid, None)
            case ScheduleWake(at):
                self.wakes.append(at)
            case CancelWake():
                self.wakes.clear()
            case Exit() | PlayCue():
                self._record(effect)
            case _:
                assert_never(effect)

    def _record(self, effect: Exit | PlayCue) -> None:
        match effect:
            case Exit():
                self.exited = True
            case PlayCue(cue):
                self.cues.append(cue)

    def _control_output(self, effect: PausePlayback | ResumePlayback | AbortPlayback) -> None:
        match effect:
            case PausePlayback():
                self.output.pause()
            case ResumePlayback():
                self.output.resume()
            case AbortPlayback():
                self.output.abort()
                self.playing = None

    def _submit_generate(self, effect: Generate, chunk: Chunk) -> None:
        state = str(self.status()["state"])
        self.generate_states.append((state, effect))
        if state != "warm":
            raise SimulationError(f"Generate while the engine is {state} (not warm)")
        self.chunks[chunk.id] = chunk
        self._jobs.append(effect)

    def _start(self, chunk: ChunkId) -> None:
        if self.playing is not None:
            raise SimulationError(f"chunk {chunk} started while {self.playing} is already playing")
        if self.output.paused:
            raise SimulationError(f"chunk {chunk} started while the output is paused")
        audio = self.audio.pop(chunk, None)
        if audio is None:
            raise SimulationError(f"chunk {chunk} started with no audio")
        self.playing = chunk
        self.started.append((chunk, self._audio_epoch.pop(chunk)))
        self.output.start(chunk, audio.samples, audio.sample_rate)

    def _run_job(self, job: _Job) -> None:
        match job:
            case LoadEngine():
                effects = self._load()
            case UnloadEngine():
                self.engine.unload()
                self.prompts.clear()
                effects = self.session.on_unloaded()
            case PrepareVoice(ref):
                effects = self._prepare(ref)
            case Generate(chunk, voice):
                effects = self._generate(chunk, voice)
            case _:
                assert_never(job)
        self._apply(effects)

    def _load(self) -> list[Effect]:
        try:
            self.engine.load()
        except Exception as exc:  # any engine failure becomes on_load_failed, as in the runtime
            return self.session.on_load_failed(str(exc))
        return self.session.on_loaded()

    def _prepare(self, ref: VoiceRef) -> list[Effect]:
        # The simulator keeps no prompt cache, so a fresh prepare is the same call.
        try:
            self.prompts[ref.name] = self.engine.prepare_voice(ref)
        except Exception as exc:  # any engine failure becomes on_prompt_failed, as in the runtime
            effects = self.session.on_prompt_failed(ref.name, str(exc))
            return self._answered(FailureReply(ref.name, None, len(self.effects), tuple(effects)))
        return self.session.on_prompt_ready(ref.name)

    def _answered(self, reply: FailureReply) -> list[Effect]:
        self.failure_replies.append(reply)
        return list(reply.effects)

    def _generate(self, chunk: Chunk, voice: str) -> list[Effect]:
        prompt = self.prompts.get(voice)
        if prompt is None:
            raise SimulationError(f"Generate for voice {voice!r} without a prompt")
        try:
            pieces = list(self.engine.synthesize(chunk.text, chunk.lang, prompt))
        except Exception as exc:  # any engine failure becomes on_gen_failed, as in the runtime
            effects = self.session.on_gen_failed(chunk.epoch, chunk.id, str(exc))
            return self._answered(FailureReply(voice, chunk, len(self.effects), tuple(effects)))
        rate = self.engine.spec.sample_rate
        samples = np.concatenate([piece.samples for piece in pieces])
        self.audio[chunk.id] = AudioChunk(samples, rate)
        self._audio_epoch[chunk.id] = chunk.epoch
        return self.session.on_generated(chunk.epoch, chunk.id, len(samples) / rate, 0.0)
