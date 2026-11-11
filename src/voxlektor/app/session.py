"""Session: the single writer. ALL mutable state of the service lives here.

Synchronous and single-threaded: in production only the asyncio loop thread calls it (commands
and the callbacks that the engine worker and the audio output hand back). Every entry point
returns the effects the runtime must carry out; the Session never performs I/O itself.

Model of one reading:

* ``_queue`` (``PlaybackQueue[ChunkId]``) holds the playback order. Its ``current`` item is the
  *head*: the chunk that plays next, or is playing (``_playing``).
* Every chunk not yet played is either pending (``_pending``, not requested yet, in playback
  order) or ahead (``_ahead``: ``Generate`` emitted, or waiting for a retry). The ahead window
  bounds ``len(_ahead)``, so at most ``settings.ahead`` chunks are generated and unplayed.
* ``_epoch`` grows on stop, unload, ``--now``, failures and back; results of older epochs are
  dropped. Back re-queues chunks under fresh ids, so a chunk id never spans two epochs.
  A chunk skipped while in flight stays ahead as *discarded* until its result arrives, so it
  keeps counting against the window.

Retry ladder: a prompt failure for a voice that a chunk still to be generated uses is retried
once per engine load and reading with a fresh prompt (``PrepareVoice(fresh)``); a prompt
failure that no live chunk needs is dropped silently;
a generation failure is retried once as is, then once after an engine reload; the next failure
stops the reading. Every retry plays ``Cue.RETRY``; a reading stopped by an error (also a
playback or load failure) plays ``Cue.STOPPED`` and records ``last_error``.

Metrics (``chunks``, ``gen_seconds``, ``audio_seconds``) count each chunk of the latest reading
once, at its first successful generation; regenerations after Back or a retry are not counted.
"""

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import assert_never

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
    Outcome,
    PausePlayback,
    PlayCue,
    PrepareVoice,
    Reply,
    ResumePlayback,
    ScheduleWake,
    StartPlayback,
    UnloadEngine,
)
from voxlektor.app.ports import Clock
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain import keepwarm as kw
from voxlektor.domain.errors import InvalidDuration, InvalidSetting, UnknownVoice
from voxlektor.domain.playback import PlaybackQueue
from voxlektor.domain.reading import AheadWindow, Chunk, UnderrunTracker, plan_reading
from voxlektor.domain.settings import Settings, parse_duration
from voxlektor.domain.types import ChunkId, Epoch

# Names of the commands this Session supports, spelled as on the wire (protocol v1), sorted.
_CAPS = (
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
)
_RATE_MIN = 0.5  # the same bounds as the ``rate`` setting
_RATE_MAX = 3.0
_RELOAD_ON_FAILURE = 2  # the generation failure that reloads the engine before retrying
_STOP_ON_FAILURE = 3  # the generation failure that stops the reading


def _ok(**data: object) -> Reply:
    return Reply(ok=True, data=data, error=None)  # a fresh dict for every reply


class _Gen(Enum):
    IN_FLIGHT = auto()
    READY = auto()
    DISCARDED = auto()
    RETRY = auto()  # failed; Generate again once WARM and the prompt is ready


@dataclass(frozen=True)
class _Item:
    chunk: Chunk
    voice: str
    counted: bool = False  # already counted in the reading metrics


@dataclass
class _Metrics:
    """Numbers about the latest reading (reset when a reading starts from idle)."""

    started_at: float
    first_audio_s: float | None = None
    chunks: int = 0
    gen_seconds: float = 0.0
    audio_seconds: float = 0.0
    underruns: UnderrunTracker = field(default_factory=UnderrunTracker)


class Session:
    def __init__(self, settings: Settings, catalog: VoiceCatalog, clock: Clock) -> None:
        self._settings = settings
        self._catalog = catalog
        self._clock = clock
        self._kw = kw.KeepWarm(kw.Life.COLD, settings.idle_timeout_s, None, False)
        self._queue: PlaybackQueue[ChunkId] = PlaybackQueue(history=settings.history)
        self._window = AheadWindow(settings.ahead)
        self._epoch = Epoch(0)
        self._next_id = 0
        self._items: dict[ChunkId, _Item] = {}  # queued, current and history chunks
        self._refs: dict[str, VoiceRef] = {}
        self._pending: deque[Chunk] = deque()
        self._ahead: dict[ChunkId, _Gen] = {}
        self._playing: ChunkId | None = None
        self._requested: set[str] = set()  # PrepareVoice emitted during this engine load
        self._prompts: set[str] = set()  # prompts ready during this engine load
        self._gen_failures: dict[ChunkId, int] = {}  # per chunk, reset on success
        self._prompt_failures: dict[str, int] = {}  # per voice; reset per reading and per load
        self._metrics = _Metrics(started_at=clock.now())
        self._last_error: str | None = None

    # ------------------------------------------------------------ commands

    def handle(self, cmd: Command) -> Outcome:
        out: list[Effect] = []
        match cmd:
            case Status():
                return Outcome(Reply(ok=True, data=self._status(), error=None), [])
            case Speak():
                reply = self._speak(cmd, out)
            case Pause():
                reply = self._pause(out)
            case Resume():
                self._lift_pause(out)
                reply = _ok()
            case Skip():
                reply = self._skip(out)
            case Back():
                reply = self._back(out)
            case StopReading():
                self._stop_reading(out)
                reply = _ok()
            case SetIdle(timeout):
                reply = self._set_idle(timeout, out)
            case Unload() | Shutdown():
                self._stop_reading(out)
                self._feed(kw.UnloadRequested(), out)
                reply = _ok()
            case _:
                assert_never(cmd)
        self._settle(out)
        if isinstance(cmd, Shutdown):
            out.append(Exit())
        return Outcome(reply, out)

    def _speak(self, cmd: Speak, out: list[Effect]) -> Reply:
        voice = cmd.voice if cmd.voice is not None else self._settings.voice
        try:
            ref = self._catalog.resolve(voice)
        except UnknownVoice as err:
            return Reply(ok=False, data={}, error=err)
        rate = cmd.rate if cmd.rate is not None else self._settings.rate
        if not _RATE_MIN <= rate <= _RATE_MAX:
            message = f"rate must be between {_RATE_MIN} and {_RATE_MAX}"
            return Reply(ok=False, data={}, error=InvalidSetting(message))
        lang = cmd.lang if cmd.lang is not None else self._settings.lang
        epoch = Epoch(self._epoch + 1) if cmd.now else self._epoch
        plan = plan_reading(cmd.text, lang, self._settings.base_lang, rate, epoch, self._next_id)
        if not plan:
            return _ok(chunks=0)
        if cmd.now:
            self._stop_reading(out)  # bumps the epoch to the one the plan was made with
        if not self._active():
            self._reset_reading()
        self._refs[voice] = ref
        self._next_id += len(plan)
        for chunk in plan:
            self._items[chunk.id] = _Item(chunk, voice)
            self._pending.append(chunk)
            self._queue.push(chunk.id)
        return _ok(chunks=len(plan))

    def _pause(self, out: list[Effect]) -> Reply:
        if self._active() and not self._queue.paused:
            self._queue.pause()
            out.append(PausePlayback())
        return _ok()

    def _skip(self, out: list[Effect]) -> Reply:
        head = self._queue.current
        if head is None:
            return _ok()
        state = self._ahead.get(head)
        if self._playing == head:
            out.append(AbortPlayback())
            self._playing = None
        elif state is _Gen.IN_FLIGHT:
            self._ahead[head] = _Gen.DISCARDED  # its result is dropped when it arrives
        elif state is None:
            self._pending.popleft()  # a head that is neither ahead nor playing is pending[0]
        else:
            if state is _Gen.READY:
                out.append(DropAudio((head,)))
            del self._ahead[head]  # READY or waiting for a retry
        self._gen_failures.pop(head, None)
        self._queue.skip()
        del self._items[head]
        return _ok()

    def _back(self, out: list[Effect]) -> Reply:
        was_active = self._active()
        if not self._queue.back():
            return _ok()
        if self._playing is not None:
            out.append(AbortPlayback())
            self._playing = None
        self._drop_ready(out)
        self._ahead.clear()
        self._gen_failures.clear()
        self._epoch = Epoch(self._epoch + 1)
        if not was_active:
            self._reset_reading()
        self._pending.clear()
        for cid in self._queue.clear():  # the re-queued chunks first, in playback order
            item = self._items.pop(cid)
            new_id = ChunkId(self._next_id)
            self._next_id += 1
            chunk = replace(item.chunk, id=new_id, epoch=self._epoch)
            self._items[new_id] = _Item(chunk, item.voice, counted=item.counted and was_active)
            self._pending.append(chunk)
            self._queue.push(new_id)
        return _ok()

    def _set_idle(self, timeout: str, out: list[Effect]) -> Reply:
        try:
            seconds = parse_duration(timeout)
        except InvalidDuration as err:
            return Reply(ok=False, data={}, error=err)
        self._settings = replace(self._settings, idle_timeout_s=seconds)
        self._feed(kw.SetTimeout(seconds), out)
        return _ok(idle_timeout_s=seconds)

    def _status(self) -> dict[str, object]:
        busy = self._active()
        deadline = self._kw.deadline
        remaining = None if busy or deadline is None else max(0.0, deadline - self._clock.now())
        head = self._queue.current
        waiting = 1 if head is not None and head != self._playing else 0
        voice = self._settings.voice if head is None else self._items[head].voice
        metrics = self._metrics
        return {
            "state": self._kw.life.value,
            "engine": self._settings.engine,
            "voice": voice,
            "idle_timeout_s": self._kw.timeout_s,
            "idle_remaining_s": remaining,
            "queued": self._queue.queued + waiting,
            "playing": self._playing is not None,
            "paused": self._queue.paused,
            "underruns": metrics.underruns.count,
            "first_audio_s": metrics.first_audio_s,
            "chunks": metrics.chunks,
            "gen_seconds": metrics.gen_seconds,
            "audio_seconds": metrics.audio_seconds,
            "last_error": self._last_error,
            "caps": list(_CAPS),
        }

    # ------------------------------------------------------------ callbacks

    def on_loaded(self) -> list[Effect]:
        out: list[Effect] = []
        self._feed(kw.Loaded(), out)
        self._settle(out)
        return out

    def on_load_failed(self, msg: str) -> list[Effect]:
        out: list[Effect] = []
        self._feed(kw.LoadFailed(msg), out)
        self._fail(msg, out)
        return out

    def on_prompt_ready(self, voice: str) -> list[Effect]:
        out: list[Effect] = []
        if voice in self._requested:  # otherwise it belongs to an earlier engine load
            self._prompts.add(voice)
        self._settle(out)
        return out

    def on_prompt_failed(self, voice: str, msg: str) -> list[Effect]:
        out: list[Effect] = []
        if voice not in self._requested or voice in self._prompts:
            self._settle(out)  # an earlier engine load, or already superseded
            return out
        if not self._needs_prompt(voice):
            self._requested.discard(voice)  # no live chunk uses it; a later Speak asks again
            self._settle(out)
            return out
        failures = self._prompt_failures.get(voice, 0) + 1
        self._prompt_failures[voice] = failures
        if failures == 1:
            out.append(PlayCue(Cue.RETRY))
            out.append(PrepareVoice(self._refs[voice], fresh=True))
            self._settle(out)
        else:
            self._requested.discard(voice)
            self._fail(msg, out)
        return out

    def on_generated(
        self, epoch: Epoch, chunk: ChunkId, audio_seconds: float, gen_seconds: float
    ) -> list[Effect]:
        out: list[Effect] = []
        if self._take_result(epoch, chunk):
            self._ahead[chunk] = _Gen.READY
            self._gen_failures.pop(chunk, None)
            item = self._items[chunk]
            if not item.counted:
                self._items[chunk] = replace(item, counted=True)
                self._metrics.chunks += 1
                self._metrics.gen_seconds += gen_seconds
                self._metrics.audio_seconds += audio_seconds
        else:
            out.append(DropAudio((chunk,)))
        self._settle(out)
        return out

    def on_gen_failed(self, epoch: Epoch, chunk: ChunkId, msg: str) -> list[Effect]:
        out: list[Effect] = []
        if not self._take_result(epoch, chunk):
            self._settle(out)
            return out
        failures = self._gen_failures.get(chunk, 0) + 1
        if failures >= _STOP_ON_FAILURE:
            self._fail(msg, out)
            return out
        self._gen_failures[chunk] = failures
        self._ahead[chunk] = _Gen.RETRY
        out.append(PlayCue(Cue.RETRY))
        if failures == _RELOAD_ON_FAILURE and self._kw.life is kw.Life.WARM:
            self._feed(kw.UnloadRequested(), out)  # a reload already under way is enough
            self._feed(kw.NeedEngine(), out)
        self._settle(out)
        return out

    def on_play_done(self, chunk: ChunkId) -> list[Effect]:
        if chunk != self._playing:
            return []  # e.g. the end of a chunk that was aborted a moment ago
        out: list[Effect] = []
        self._playing = None
        self._finish_head(chunk)
        head = self._queue.start_next()
        ran_dry = head is not None and self._ahead.get(head) is not _Gen.READY
        if ran_dry and not self._queue.paused:
            self._metrics.underruns.on_queue_empty(more_expected=True)
        self._settle(out)
        return out

    def on_play_failed(self, chunk: ChunkId, msg: str) -> list[Effect]:
        if chunk != self._playing:
            return []
        out: list[Effect] = []
        self._playing = None  # the output has stopped already: nothing to abort
        self._fail(msg, out)
        return out

    def on_wake(self) -> list[Effect]:
        out: list[Effect] = []
        self._feed(kw.Wake(self._clock.now()), out)
        self._settle(out)
        return out

    def on_unloaded(self) -> list[Effect]:
        out: list[Effect] = []
        self._feed(kw.Unloaded(), out)
        self._settle(out)
        return out

    # ------------------------------------------------------------ internals

    def _active(self) -> bool:
        return self._queue.current is not None or self._queue.queued > 0

    def _reset_reading(self) -> None:
        self._metrics = _Metrics(started_at=self._clock.now())
        self._gen_failures.clear()
        self._prompt_failures.clear()

    def _take_result(self, epoch: Epoch, chunk: ChunkId) -> bool:
        """True when a result for ``chunk`` is wanted; forget a discarded chunk either way."""
        state = self._ahead.get(chunk) if epoch == self._epoch else None
        if state is _Gen.DISCARDED:
            del self._ahead[chunk]
        return state is _Gen.IN_FLIGHT

    def _fail(self, msg: str, out: list[Effect]) -> None:
        self._last_error = msg
        stopped = self._active()
        self._stop_reading(out)
        if stopped:
            out.append(PlayCue(Cue.STOPPED))
        self._settle(out)

    def _stop_reading(self, out: list[Effect]) -> None:
        if self._playing is not None:
            out.append(AbortPlayback())
            self._playing = None
        self._drop_ready(out)
        for cid in self._queue.clear():
            del self._items[cid]
        self._pending.clear()
        self._ahead.clear()
        self._gen_failures.clear()
        self._epoch = Epoch(self._epoch + 1)
        self._lift_pause(out)

    def _drop_ready(self, out: list[Effect]) -> None:
        ready = tuple(cid for cid, state in self._ahead.items() if state is _Gen.READY)
        if ready:
            out.append(DropAudio(ready))

    def _lift_pause(self, out: list[Effect]) -> None:
        if self._queue.paused:
            self._queue.resume()
            out.append(ResumePlayback())

    def _finish_head(self, finished: ChunkId) -> None:
        before = self._queue.history
        self._queue.finish_current()
        kept = set(self._queue.history)
        for cid in (*before, finished):
            if cid not in kept:
                del self._items[cid]  # evicted from the bounded history

    def _feed(self, event: kw.Event, out: list[Effect]) -> None:
        self._kw, actions = kw.transition(self._kw, event)
        for action in actions:
            match action:
                case kw.StartLoad():
                    self._forget_prompts()
                    out.append(LoadEngine(self._settings.engine))
                case kw.StartUnload():
                    self._forget_prompts()
                    out.append(UnloadEngine())
                case kw.ScheduleWake(at):
                    out.append(ScheduleWake(at))
                case kw.CancelWake():
                    out.append(CancelWake())
                case kw.FlushPending():
                    pass  # _settle runs after every entry point and starts the queued work
                case _:
                    assert_never(action)

    def _forget_prompts(self) -> None:
        """A new engine load (R22): no prompts, and every voice gets the full prompt ladder."""
        self._prompts.clear()
        self._requested.clear()
        self._prompt_failures.clear()

    def _settle(self, out: list[Effect]) -> None:
        """Bring keep-warm, playback and generation in line with the current reading."""
        self._queue.start_next()
        if not self._active():
            self._lift_pause(out)  # a reading that ended while paused leaves no pause behind
            if self._kw.life is kw.Life.WARM and self._kw.idle_since is None:
                self._feed(kw.Idle(self._clock.now()), out)
            return
        if self._kw.life is not kw.Life.WARM:
            self._start_playback(out)  # ready audio stays valid while the engine reloads
            self._feed(kw.NeedEngine(), out)
            return
        self._feed(kw.Busy(), out)
        self._start_playback(out)
        self._generate(out)

    def _start_playback(self, out: list[Effect]) -> None:
        head = self._queue.current
        if head is None or self._playing is not None or self._queue.paused:
            return
        if self._ahead.get(head) is not _Gen.READY:
            return
        del self._ahead[head]
        self._playing = head
        out.append(StartPlayback(head))
        if self._metrics.first_audio_s is None:
            self._metrics.first_audio_s = self._clock.now() - self._metrics.started_at

    def _generate(self, out: list[Effect]) -> None:
        for cid in [c for c, state in self._ahead.items() if state is _Gen.RETRY]:
            item = self._items[cid]
            if not self._prompt_ready(item.voice, out):
                return
            self._ahead[cid] = _Gen.IN_FLIGHT
            out.append(Generate(item.chunk, item.voice))
        while True:
            chunk = self._window.next_to_generate(self._pending, len(self._ahead))
            if chunk is None or not self._prompt_ready(self._items[chunk.id].voice, out):
                return
            self._pending.popleft()
            self._ahead[chunk.id] = _Gen.IN_FLIGHT
            out.append(Generate(chunk, self._items[chunk.id].voice))

    def _needs_prompt(self, voice: str) -> bool:
        """True when a chunk still to be generated (pending or waiting to retry) uses ``voice``."""
        waiting = [cid for cid, state in self._ahead.items() if state is _Gen.RETRY]
        waiting += [chunk.id for chunk in self._pending]
        return any(self._items[cid].voice == voice for cid in waiting)

    def _prompt_ready(self, voice: str, out: list[Effect]) -> bool:
        """True when ``voice`` can be used now; otherwise make sure its prompt is requested."""
        if voice in self._prompts:
            return True
        if voice not in self._requested:
            self._requested.add(voice)
            out.append(PrepareVoice(self._refs[voice]))
        return False
