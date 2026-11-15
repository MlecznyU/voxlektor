"""Session: the single writer. ALL mutable state of the service lives here.

Synchronous and single-threaded: in production only the asyncio loop thread calls it (commands
and the callbacks that the engine worker and the audio output hand back). Every entry point
returns the effects the runtime must carry out; the Session never performs I/O itself.

Model of the reading queue (spec §4):

* Several *owners* (clients) share the daemon. Requests without an owner belong to the reserved
  owner ``cli``. Each ``speak`` goes into a *group* ``(owner, group)``, read without
  interruption; a ``speak`` without a group gets a fresh one (``g1``, ``g2``, ...).
* ``_floor`` (``domain.floor.Floor``) decides which group holds the floor: the oldest waiting
  group whose owner is not paused, handed over only at group boundaries, after the grace of a
  paused holder, or by ``preempt``. Only the holder's head chunk (``_Group.chunks[0]``) plays.
* Every chunk not played yet is in its group's ``chunks``; it is either *pending* (not ahead and
  not playing) or *ahead* (``_ahead``: ``Generate`` emitted, ready, or waiting for a retry).
  The ahead window walks the unplayed chunks in ``Floor.play_order()``: a pending chunk is
  requested while fewer than ``settings.ahead`` chunks before it are ahead (discarded ones always
  count), so the next group starts generating while the current one plays, and audio of a
  yielded or preempted group (or behind an interrupted chunk) never blocks what plays first.
  Chunks of passed-over (paused) owners are not generated.
* Epochs: every group carries one; all groups of an owner share it, because every change is
  owner-wide (stop, ``now``, back, failures). A new epoch is a fresh global integer; a stopped
  or backed owner's epoch is dead, and dead epochs with generations outstanding go out in one
  ``CancelGeneration``. Back re-queues chunks under fresh ids, so a chunk id never spans two
  epochs. A chunk skipped while in flight stays ahead as *discarded* until its result arrives,
  so it keeps counting against the window.
* Yield or preempt interrupts a group: its playing chunk is aborted and becomes pending again at
  the front of its group, and its ready audio is dropped (both are regenerated later).
* Pauses: an owner pause is a flag in the floor (it lifts when that owner has nothing left, on
  its ``resume``, ``stop_reading``, ``speak now`` or ``preempt``); the ownerless pause is the
  daemon-wide *hold* (no hand-over, lifted by an ownerless ``resume``/``stop_reading``/``speak
  now``, a ``resume`` from the holder's owner, or the daemon going idle). The output is paused
  while the hold is on or the holder's owner is paused. The grace of a paused holder counts from
  when a waiter last became eligible (``Floor``), so it restarts on a resume or a hold release.
* History is per owner: ``back`` stays within the owner's own chunks.
* Events for clients (spec §5.2) are ``Emit`` effects: ``state`` on every keep-warm life change,
  ``floor`` when a group gets the floor, ``tag_started``/``tag_finished`` per tag run
  (``_events.Tags``, spec §4.4), ``error`` when a failure stops a reading; ``paused``,
  ``resumed``, ``waiting`` and ``drained`` are the differences ``_events.Watch`` finds once per
  ``_settle``, after the hand-over.

Retry ladder: a prompt failure for a voice that a chunk still to be generated uses is retried
once per engine load and reading with a fresh prompt (``PrepareVoice(fresh)``); a prompt
failure that no live chunk needs is dropped silently;
a generation failure is retried once as is, then once after an engine reload; the next failure
stops the reading. Every retry plays ``Cue.RETRY``; a reading stopped by an error plays
``Cue.STOPPED`` and records the error. A chunk or prompt failure stops only the owners it
belongs to; a load or playback failure stops everyone. Errors are kept per owner; status shows
the latest. A generation retry cue plays at the volume of the failed chunk; every other cue at
the volume of the latest accepted ``Speak`` (``_cue_volume``; ``settings.volume`` before the
first one).

Metrics (``chunks``, ``gen_seconds``, ``audio_seconds``) count each chunk of the latest reading
once, at its first successful generation; regenerations after Back or a retry are not counted.
A reading starts when a ``speak`` arrives while the whole daemon is idle.
"""

from collections import deque
from dataclasses import replace
from typing import assert_never

from voxlektor.app._events import Tags, Watch, emit
from voxlektor.app._groups import CLI as _CLI
from voxlektor.app._groups import Gen, Group, Item, Metrics, history_add
from voxlektor.app.commands import (
    Back,
    Command,
    Load,
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
    CancelGeneration,
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
from voxlektor.domain.errors import DomainError, InvalidDuration, InvalidSetting, UnknownVoice
from voxlektor.domain.floor import Floor, GroupKey
from voxlektor.domain.reading import plan_reading
from voxlektor.domain.settings import Settings, parse_duration
from voxlektor.domain.types import ChunkId, Epoch

# Names of the commands this Session supports, spelled as on the wire (protocol v1), sorted.
_CAPS = (
    "back",
    "idle",
    "load",
    "pause",
    "resume",
    "shutdown",
    "skip",
    "speak",
    "status",
    "stop_reading",
    "subscribe",  # served by the transport (an event stream), advertised here with the rest
    "unload",
)
_RATE_MIN = 0.5  # the same bounds as the ``rate`` setting
_RATE_MAX = 3.0
_VOLUME_MIN = 0  # the same bounds as the ``volume`` setting
_VOLUME_MAX = 100
_RELOAD_ON_FAILURE = 2  # the generation failure that reloads the engine before retrying
_STOP_ON_FAILURE = 3  # the generation failure that stops the reading

type _Owners = frozenset[str] | None  # None: every owner


def _ok(**data: object) -> Reply:
    return Reply(ok=True, data=data, error=None)  # a fresh dict for every reply


_BUSY = DomainError("busy", "the daemon is reading")


def _reserved() -> Reply:
    error = DomainError("invalid_argument", "owner 'cli' is reserved")
    return Reply(ok=False, data={}, error=error)


def _out_of_range(rate: float, volume: int) -> InvalidSetting | None:
    if not _RATE_MIN <= rate <= _RATE_MAX:
        return InvalidSetting(f"rate must be between {_RATE_MIN} and {_RATE_MAX}")
    if not _VOLUME_MIN <= volume <= _VOLUME_MAX:
        return InvalidSetting(f"volume must be between {_VOLUME_MIN} and {_VOLUME_MAX}")
    return None


def _hits(owners: _Owners, owner: str) -> bool:
    return owners is None or owner in owners


class Session:
    """``version`` (the package version, from bootstrap) is reported as status ``version``."""

    def __init__(
        self, settings: Settings, catalog: VoiceCatalog, clock: Clock, *, version: str = "0"
    ) -> None:
        self._version = version
        self._settings = settings
        self._catalog = catalog
        self._clock = clock
        self._kw = kw.KeepWarm(kw.Life.COLD, settings.idle_timeout_s, None, False)
        self._floor = Floor(settings.floor_grace_s)
        self._groups: dict[GroupKey, Group] = {}
        self._histories: dict[str, deque[ChunkId]] = {}  # per owner, oldest first
        self._last_owner: str | None = None  # the owner of the chunk that finished last
        self._epoch = Epoch(0)  # the newest epoch handed out (or to hand out next)
        self._epoch_used = False
        self._next_id = 0
        self._next_group = 1
        self._items: dict[ChunkId, Item] = {}  # queued, history and discarded chunks
        self._refs: dict[str, VoiceRef] = {}
        self._ahead: dict[ChunkId, Gen] = {}
        self._playing: ChunkId | None = None
        self._paused = False  # the output is paused
        self._grace_wake: float | None = None  # the grace deadline a wake is scheduled for
        self._requested: set[str] = set()  # PrepareVoice emitted during this engine load
        self._prompts: set[str] = set()  # prompts ready during this engine load
        self._gen_failures: dict[ChunkId, int] = {}  # per chunk, reset on success
        self._prompt_failures: dict[str, int] = {}  # per voice; reset per reading and per load
        self._metrics = Metrics(started_at=clock.now())
        self._errors: dict[str | None, str] = {}  # per owner (None: daemon-wide), latest last
        self._cue_volume = settings.volume  # the volume of the latest accepted Speak
        self._tags = Tags()
        self._watch = Watch()
        self._next_speak = 0  # the id of the next tag run

    # ------------------------------------------------------------ commands

    def handle(self, cmd: Command) -> Outcome:  # noqa: PLR0912  # one arm per command
        out: list[Effect] = []
        if not isinstance(cmd, Status | SetIdle | Unload | Shutdown | Load) and cmd.owner == _CLI:
            return Outcome(_reserved(), [])
        match cmd:
            case Status():
                return Outcome(Reply(ok=True, data=self._status(), error=None), [])
            case Shutdown(if_idle=True) if self._groups:
                # Atomic in the single writer: another session's reading is never dropped.
                return Outcome(Reply(ok=False, data={}, error=_BUSY), [])
            case Load():
                self._load(out)
                reply = _ok()
            case Speak():
                reply = self._speak(cmd, out)
            case Pause(owner):
                self._pause(owner)
                reply = _ok()
            case Resume(owner):
                self._resume(owner)
                reply = _ok()
            case Skip(owner):
                self._skip(owner, out)
                reply = _ok()
            case Back(owner):
                self._back(owner, out)
                reply = _ok()
            case StopReading(owner):
                self._kill(None if owner is None else frozenset({owner}), "stopped", out)
                reply = _ok()
            case SetIdle(timeout):
                reply = self._set_idle(timeout, out)
            case Unload() | Shutdown():
                self._kill(None, "stopped", out)
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
        volume = cmd.volume if cmd.volume is not None else self._settings.volume
        error = _out_of_range(rate, volume)
        if error is not None:
            return Reply(ok=False, data={}, error=error)
        owner = cmd.owner or _CLI
        mine = self._floor.groups_of(owner)
        fresh = cmd.now or not mine
        epoch = self._peek_epoch() if fresh else self._groups[mine[0]].epoch
        lang = cmd.lang if cmd.lang is not None else self._settings.lang
        base_lang = cmd.base_lang or self._settings.base_lang
        plan = plan_reading(cmd.text, lang, base_lang, rate, epoch, self._next_id, volume=volume)
        if not plan:
            return _ok(chunks=0)
        if fresh:
            self._epoch, self._epoch_used = epoch, True
        self._cue_volume = volume
        self._refs[voice] = ref
        self._next_id += len(plan)
        key = self._place(cmd, owner, epoch, out)
        group = self._groups[key]
        speak = None
        if cmd.tag is not None:
            speak, self._next_speak = self._next_speak, self._next_speak + 1
            self._tags.open(speak, cmd.tag, key)
        for chunk in plan:
            self._items[chunk.id] = Item(chunk, voice, key, cmd.tag, speak=speak)
            group.chunks.append(chunk.id)
        return _ok(chunks=len(plan), group=key[1])

    def _place(self, cmd: Speak, owner: str, epoch: Epoch, out: list[Effect]) -> GroupKey:
        """Find or create the group of an accepted ``speak``; replace and preempt as asked."""
        key = (owner, cmd.group or self._auto_group(owner))
        if cmd.group is None:
            self._next_group += 1
        now = self._clock.now()
        held_floor = self._holds(owner)
        mine = self._floor.groups_of(owner)
        age = min((self._groups[k].age for k in mine), default=now) if cmd.now else now
        if cmd.now:
            self._kill(frozenset({owner}), "replaced", out)
            if owner == _CLI:
                # Ruling: a ``cli`` ``speak now`` (the ownerless user at the keyboard) also
                # lifts the daemon-wide hold, as an ownerless ``resume`` would.
                self._floor.release(now)
        if not self._groups:
            self._reset_reading()
        elif not self._floor.groups_of(owner):
            self._errors.pop(owner, None)  # this owner starts again: its error is history
        if key not in self._groups:
            self._groups[key] = Group(epoch, age)
            self._floor.add(key, now, age=age)
        if cmd.preempt or (cmd.now and held_floor):
            self._floor.resume_owner(owner, now)  # like ``now``: taking the floor lifts a pause
            interrupted = self._floor.preempt(key, now)
            if interrupted is not None:
                self._interrupt(interrupted, out)
            self._watch.floor(self._floor.holder, out)
        return key

    def _load(self, out: list[Effect]) -> None:
        """Warm the engine without a reading; a warm, idle engine restarts its idle timeout."""
        self._feed(kw.NeedEngine(), out)
        if self._kw.life is kw.Life.WARM and not self._groups:
            self._feed(kw.Idle(self._clock.now()), out)

    def _auto_group(self, owner: str) -> str:
        while (owner, f"g{self._next_group}") in self._groups:
            self._next_group += 1
        return f"g{self._next_group}"

    def _pause(self, owner: str | None) -> None:
        if owner is None:
            if self._groups:
                self._floor.hold()
        elif self._floor.groups_of(owner):
            self._floor.pause_owner(owner, self._clock.now())

    def _resume(self, owner: str | None) -> None:
        now = self._clock.now()
        if owner is None or self._holds(owner):
            self._floor.release(now)
        if owner is not None:
            self._floor.resume_owner(owner, now)

    def _skip(self, owner: str | None, out: list[Effect]) -> None:
        key = self._floor.holder
        if key is None or (owner is not None and key[0] != owner):
            return
        group = self._groups[key]
        head = group.chunks.popleft()
        self._tag_ends(self._items[head], "skipped", out)
        state = self._ahead.get(head)
        if self._playing == head:
            out.append(AbortPlayback())
            self._playing = None
        if state is Gen.IN_FLIGHT:
            self._ahead[head] = Gen.DISCARDED  # its result is dropped when it arrives
        else:
            if state is Gen.READY:
                out.append(DropAudio((head,)))
            self._ahead.pop(head, None)
            del self._items[head]
        self._gen_failures.pop(head, None)
        if not group.chunks:
            self._remove_group(key)

    def _back(self, owner: str | None, out: list[Effect]) -> None:
        holder = self._floor.holder
        if owner is None:
            target = holder[0] if holder is not None else self._last_owner
        else:
            target = owner if self._holds(owner) else None
        if target is None:
            return
        keys = self._floor.groups_of(target)
        head_key = holder if holder is not None and holder[0] == target else None
        head_key = head_key or (keys[0] if keys else None)
        history = self._histories.get(target, deque[ChunkId]())
        if head_key is None and not history:
            return
        was_active = bool(self._groups)
        self._drop_work(frozenset({target}), out)
        epoch = self._peek_epoch()
        self._epoch, self._epoch_used = epoch, True
        if not was_active:
            self._reset_reading()
        if head_key is None:
            head_key = (target, self._auto_group(target))
            self._next_group += 1
            now = self._clock.now()
            self._groups[head_key] = Group(epoch, now)
            self._floor.add(head_key, now)
        if history:
            self._groups[head_key].chunks.appendleft(history.pop())
        for cid in self._groups[head_key].chunks:
            item = self._items[cid]
            if item.speak is not None and item.tag is not None:
                self._tags.back(item.speak, item.tag, head_key, out)
        for key in (head_key, *(k for k in keys if k != head_key)):
            group = self._groups[key]
            group.epoch = epoch
            group.chunks = deque(self._requeue(cid, key, was_active) for cid in group.chunks)

    def _requeue(self, cid: ChunkId, key: GroupKey, was_active: bool) -> ChunkId:
        """``cid`` again under a fresh id and the epoch of group ``key``."""
        item = self._items.pop(cid)
        new_id = ChunkId(self._next_id)
        self._next_id += 1
        chunk = replace(item.chunk, id=new_id, epoch=self._groups[key].epoch)
        counted = item.counted and was_active
        self._items[new_id] = replace(item, chunk=chunk, group=key, counted=counted)
        return new_id

    def _set_idle(self, timeout: str, out: list[Effect]) -> Reply:
        try:
            seconds = parse_duration(timeout)
        except InvalidDuration as err:
            return Reply(ok=False, data={}, error=err)
        self._settings = replace(self._settings, idle_timeout_s=seconds)
        self._feed(kw.SetTimeout(seconds), out)
        return _ok(idle_timeout_s=seconds)

    def _status(self) -> dict[str, object]:
        busy = bool(self._groups)
        deadline = self._kw.deadline
        remaining = None if busy or deadline is None else max(0.0, deadline - self._clock.now())
        head = self._head()
        voice = self._settings.voice if head is None else self._items[head].voice
        holder = self._floor.holder
        unplayed = sum(len(g.chunks) for g in self._groups.values())
        metrics = self._metrics
        return {
            "state": self._kw.life.value,
            "engine": self._settings.engine,
            "voice": voice,
            "idle_timeout_s": self._kw.timeout_s,
            "idle_remaining_s": remaining,
            "queued": unplayed - (self._playing is not None),
            "playing": self._playing is not None,
            "paused": self._paused,
            "underruns": metrics.underruns.count,
            "first_audio_s": metrics.first_audio_s,
            "chunks": metrics.chunks,
            "gen_seconds": metrics.gen_seconds,
            "audio_seconds": metrics.audio_seconds,
            "last_error": next(reversed(self._errors.values()), None),
            "caps": list(_CAPS),
            "floor": None if holder is None else {"owner": holder[0], "group": holder[1]},
            "waiting": len(self._groups) - (holder is not None),
            "groups": self._group_rows(),
            "tag": None if head is None else self._items[head].tag,
            "version": self._version,
        }

    def _group_rows(self) -> list[dict[str, object]]:
        """Every group in play order (0 = the holder); passed-over groups last, position None."""
        order = self._floor.play_order()
        passed = sorted((k for k in self._groups if k not in order), key=self._age)
        holder = self._floor.holder
        return [
            {
                "owner": key[0],
                "group": key[1],
                "position": 0 if key == holder else self._floor.position(key),
                "paused": self._floor.paused(key[0]),
            }
            for key in (*order, *passed)
        ]

    def _age(self, key: GroupKey) -> float:
        return self._groups[key].age

    # ------------------------------------------------------------ callbacks

    def on_loaded(self) -> list[Effect]:
        out: list[Effect] = []
        self._feed(kw.Loaded(), out)
        self._settle(out)
        return out

    def on_load_failed(self, msg: str) -> list[Effect]:
        out: list[Effect] = []
        self._feed(kw.LoadFailed(msg), out)
        self._fail(msg, None, "engine_failed", out)
        return out

    def on_prompt_ready(self, voice: str) -> list[Effect]:
        out: list[Effect] = []
        if voice in self._requested:  # otherwise it belongs to an earlier engine load
            self._prompts.add(voice)
        self._settle(out)
        return out

    def on_prompt_failed(self, voice: str, msg: str) -> list[Effect]:
        out: list[Effect] = []
        owners = self._owners_needing(voice)
        if voice not in self._requested or voice in self._prompts:
            self._settle(out)  # an earlier engine load, or already superseded
            return out
        if not owners:
            self._requested.discard(voice)  # no live chunk uses it; a later Speak asks again
            self._settle(out)
            return out
        failures = self._prompt_failures.get(voice, 0) + 1
        self._prompt_failures[voice] = failures
        if failures == 1:
            out.append(PlayCue(Cue.RETRY, self._cue_volume))
            out.append(PrepareVoice(self._refs[voice], fresh=True))
            self._settle(out)
        else:
            self._requested.discard(voice)
            del self._prompt_failures[voice]
            # Ruling: the prompt is shared per voice, so its failure stops every owner with a
            # chunk still to be generated in that voice, not only the owner that asked first.
            self._fail(msg, owners, "engine_failed", out)
        return out

    def on_generated(
        self, epoch: Epoch, chunk: ChunkId, audio_seconds: float, gen_seconds: float
    ) -> list[Effect]:
        out: list[Effect] = []
        if self._take_result(epoch, chunk):
            self._ahead[chunk] = Gen.READY
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
        item = self._items[chunk]
        failures = self._gen_failures.get(chunk, 0) + 1
        if failures >= _STOP_ON_FAILURE:
            self._fail(msg, frozenset({item.group[0]}), "engine_failed", out)
            return out
        self._gen_failures[chunk] = failures
        self._ahead[chunk] = Gen.RETRY
        out.append(PlayCue(Cue.RETRY, item.chunk.volume))
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
        key = self._items[chunk].group
        group = self._groups[key]
        group.chunks.popleft()
        self._tag_ends(self._items[chunk], "done", out)
        self._last_owner = key[0]
        history = self._histories.setdefault(key[0], deque())
        for evicted in history_add(history, chunk, self._settings.history):
            del self._items[evicted]
        if not group.chunks:
            self._remove_group(key)
        self._hand_over(out)
        head = self._head()
        ran_dry = head is not None and self._ahead.get(head) is not Gen.READY
        if ran_dry and not self._wants_pause():
            self._metrics.underruns.on_queue_empty(more_expected=True)
        self._settle(out)
        return out

    def on_play_failed(self, chunk: ChunkId, msg: str) -> list[Effect]:
        if chunk != self._playing:
            return []
        out: list[Effect] = []
        self._playing = None  # the output has stopped already: nothing to abort
        self._fail(msg, None, "audio_failed", out)
        return out

    def on_wake(self) -> list[Effect]:
        out: list[Effect] = []
        self._grace_wake = None  # a wake that fired early is re-armed by ``_settle``
        self._floor.on_time(self._clock.now())
        self._feed(kw.Wake(self._clock.now()), out)
        self._settle(out)
        return out

    def on_unloaded(self) -> list[Effect]:
        out: list[Effect] = []
        self._feed(kw.Unloaded(), out)
        self._settle(out)
        return out

    # ------------------------------------------------------------ floor and groups

    def _holds(self, owner: str) -> bool:
        holder = self._floor.holder
        return holder is not None and holder[0] == owner

    def _head(self) -> ChunkId | None:
        holder = self._floor.holder
        return None if holder is None else self._groups[holder].chunks[0]

    def _peek_epoch(self) -> Epoch:
        """A fresh epoch: higher than every epoch handed out (or the unused current one)."""
        return Epoch(self._epoch + 1) if self._epoch_used else self._epoch

    def _hand_over(self, out: list[Effect]) -> None:
        before = self._floor.holder
        if self._floor.hand_over(self._clock.now()) != before and before in self._groups:
            self._interrupt(before, out)  # it yielded after its grace
        self._watch.floor(self._floor.holder, out)

    def _interrupt(self, key: GroupKey, out: list[Effect]) -> None:
        """``key`` lost the floor: its chunk restarts later and its ready audio is dropped."""
        if self._playing is not None and self._items[self._playing].group == key:
            out.append(AbortPlayback())
            self._playing = None
        ready = tuple(c for c in self._groups[key].chunks if self._ahead.get(c) is Gen.READY)
        if ready:
            out.append(DropAudio(ready))
        for cid in ready:
            del self._ahead[cid]

    def _remove_group(self, key: GroupKey) -> None:
        del self._groups[key]
        self._floor.remove(key)
        self._watch.forget(key)
        if not self._floor.groups_of(key[0]):
            # an owner with nothing left is not paused
            self._floor.resume_owner(key[0], self._clock.now())

    def _drop_work(self, owners: _Owners, out: list[Effect]) -> None:
        """Abort, drop and cancel everything in progress of ``owners``' groups (dead epoch)."""
        playing = self._playing
        if playing is not None and _hits(owners, self._items[playing].group[0]):
            out.append(AbortPlayback())
            self._playing = None
        doomed = [c for c in self._ahead if _hits(owners, self._items[c].group[0])]
        ready = tuple(c for c in doomed if self._ahead[c] is Gen.READY)
        if ready:
            out.append(DropAudio(ready))
        busy = {Gen.IN_FLIGHT, Gen.DISCARDED}
        dead = {self._items[c].chunk.epoch for c in doomed if self._ahead[c] in busy}
        if dead:
            out.append(CancelGeneration(tuple(sorted(dead))))
        for cid in doomed:
            if self._ahead.pop(cid) is Gen.DISCARDED:
                del self._items[cid]
            self._gen_failures.pop(cid, None)

    def _kill(self, owners: _Owners, reason: str, out: list[Effect]) -> None:
        """Stop every group of ``owners`` (None: of everyone, and lift the hold).

        Every tag run of those groups ends with ``reason``.
        """
        self._drop_work(owners, out)
        keys = [k for k in self._groups if _hits(owners, k[0])]
        items = [self._items[c] for k in keys for c in self._groups[k].chunks]
        for speak in sorted({i.speak for i in items if i.speak is not None}):
            self._tags.finish(speak, reason, out)
        for key in keys:
            for cid in self._groups[key].chunks:
                del self._items[cid]
            self._remove_group(key)
        if owners is None:
            self._floor.release(self._clock.now())

    def _tag_ends(self, item: Item, reason: str, out: list[Effect]) -> None:
        """``item`` left its group (played or skipped); its tag run ends if it was the last."""
        speak = item.speak
        if speak is None:
            return
        chunks = (c for g in self._groups.values() for c in g.chunks)
        if all(self._items[c].speak != speak for c in chunks):
            self._tags.finish(speak, reason, out)

    def _owners_needing(self, voice: str) -> frozenset[str]:
        """The owners with a chunk still to be generated (pending or to retry) in ``voice``."""
        waiting = [c for c, state in self._ahead.items() if state is Gen.RETRY]
        waiting += [c for g in self._groups.values() for c in g.chunks if self._is_pending(c)]
        return frozenset(self._items[c].group[0] for c in waiting if self._items[c].voice == voice)

    def _is_pending(self, cid: ChunkId) -> bool:
        return cid not in self._ahead and cid != self._playing

    # ------------------------------------------------------------ internals

    def _reset_reading(self) -> None:
        self._metrics = Metrics(started_at=self._clock.now())
        self._gen_failures.clear()
        self._prompt_failures.clear()
        self._errors.clear()  # status reports the failure of the latest reading only

    def _take_result(self, epoch: Epoch, chunk: ChunkId) -> bool:
        """True when a result for ``chunk`` is wanted; forget a discarded chunk either way."""
        state = self._ahead.get(chunk)
        if state is None or self._items[chunk].chunk.epoch != epoch:
            return False
        if state is Gen.DISCARDED:
            del self._ahead[chunk]
            del self._items[chunk]
        return state is Gen.IN_FLIGHT

    def _fail(self, msg: str, owners: _Owners, code: str, out: list[Effect]) -> None:
        stopped = sorted({k[0] for k in self._groups if _hits(owners, k[0])})
        for owner in [None] if owners is None else sorted(owners):
            self._errors.pop(owner, None)
            self._errors[owner] = msg
        if stopped:  # a failure stopped a reading: tell the owners it stopped
            for owner in [None] if owners is None else stopped:
                emit(out, "error", owner, None, code=code, message=msg)
        self._kill(owners, "error", out)
        if stopped:
            out.append(PlayCue(Cue.STOPPED, self._cue_volume))
        self._settle(out)

    def _feed(self, event: kw.Event, out: list[Effect]) -> None:
        life = self._kw.life
        self._kw, actions = kw.transition(self._kw, event)
        if self._kw.life is not life:
            emit(out, "state", None, None, state=self._kw.life.value)
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
                    self._grace_wake = None  # it cancels the grace wake too
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

    def _wants_pause(self) -> bool:
        holder = self._floor.holder
        owner_paused = holder is not None and self._floor.paused(holder[0])
        return bool(self._groups) and (self._floor.held or owner_paused)

    def _sync_pause(self, out: list[Effect]) -> None:
        wanted = self._wants_pause()
        if wanted != self._paused:
            self._paused = wanted
            out.append(PausePlayback() if wanted else ResumePlayback())

    def _settle(self, out: list[Effect]) -> None:
        """Bring the floor, keep-warm, playback and generation in line with the groups."""
        self._floor.on_time(self._clock.now())
        self._hand_over(out)
        if not self._groups:
            self._floor.release(self._clock.now())  # the daemon going idle lifts the hold
        self._watch.sync(self._floor, self._groups, out)
        if not self._groups:
            self._sync_pause(out)  # a reading that ended while paused leaves no pause behind
            if self._kw.life is kw.Life.WARM and self._kw.idle_since is None:
                self._feed(kw.Idle(self._clock.now()), out)
            return
        self._sync_pause(out)
        if self._kw.life is not kw.Life.WARM:
            self._start_playback(out)  # ready audio stays valid while the engine reloads
            self._feed(kw.NeedEngine(), out)
        else:
            self._feed(kw.Busy(), out)
            self._start_playback(out)
            self._generate(out)
        deadline = self._floor.grace_deadline()
        if deadline is not None and deadline != self._grace_wake:
            self._grace_wake = deadline
            out.append(ScheduleWake(deadline))

    def _start_playback(self, out: list[Effect]) -> None:
        head = self._head()
        if head is None or self._playing is not None or self._paused:
            return
        if self._ahead.get(head) is not Gen.READY:
            return
        del self._ahead[head]
        self._playing = head
        out.append(StartPlayback(head))
        item = self._items[head]
        if item.speak is not None:
            self._tags.start(item.speak, item.group, out)
        if self._metrics.first_audio_s is None:
            self._metrics.first_audio_s = self._clock.now() - self._metrics.started_at

    def _generate(self, out: list[Effect]) -> None:
        order = self._floor.play_order()
        for cid in [c for c, state in self._ahead.items() if state is Gen.RETRY]:
            item = self._items[cid]
            if item.group not in order:
                continue
            if not self._prompt_ready(item.voice, out):
                return
            self._ahead[cid] = Gen.IN_FLIGHT
            out.append(Generate(item.chunk, item.voice))
        used = sum(state is Gen.DISCARDED for state in self._ahead.values())
        for key in order:
            for cid in self._groups[key].chunks:
                if cid == self._playing:
                    continue
                if cid in self._ahead:
                    used += 1  # ahead of everything after it in the play order
                    continue
                item = self._items[cid]
                if used >= self._settings.ahead or not self._prompt_ready(item.voice, out):
                    return
                used += 1
                self._ahead[cid] = Gen.IN_FLIGHT
                out.append(Generate(item.chunk, item.voice))

    def _prompt_ready(self, voice: str, out: list[Effect]) -> bool:
        """True when ``voice`` can be used now; otherwise make sure its prompt is requested."""
        if voice in self._prompts:
            return True
        if voice not in self._requested:
            self._requested.add(voice)
            out.append(PrepareVoice(self._refs[voice]))
        return False
