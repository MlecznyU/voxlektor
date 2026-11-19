"""Wiring of the service: the only place that picks concrete adapters for the daemon.

``build_daemon`` connects Session, EngineWorker (engines from the registry), AudioBuffers, the
audio output, the cue player and a monotonic clock into a ``CoreLoop``. With ``transport`` the
returned ``Daemon`` also serves protocol v1 on the platform's socket and owns the runtime files:
the pid file is written once the socket listens (so its presence means "ready").

Stopping order (``Daemon.run``): when the loop stops, the server is closed and the socket and
pid file are removed first, so a successor can start at once; then the cue stream is closed and
the engine worker is closed (it finishes its queued jobs, such as the final unload; this blocks
the loop thread at the very end, when nothing else is served). Only our own files are removed:
the socket inode recorded after bind, and the pid file inode we wrote, still holding our pid.
A successor that has already replaced them keeps its files.

Events: one ``EventHub`` feeds ``CoreLoop`` (publisher) and the socket server (one bounded queue
of ``_SUBSCRIBER_LIMIT`` events per ``subscribe`` connection). The ``snapshot`` is the Session's
status read directly, the one Session call outside ``CoreLoop``: ``Status`` has no effects and
the server reads it on the loop thread in the same step as it subscribes, so it is consistent
with the events that follow (``CoreLoop.handle`` is a coroutine and the snapshot is not).

Outputs: an injected ``output``/``cues`` wins; otherwise ``VOXLEKTOR_OUTPUT=memory`` selects the
device-less ``MemoryOutput`` and ``MemoryCuePlayer`` (test runs), else sounddevice. The
sounddevice adapters are imported only when selected, so memory mode never loads PortAudio.
"""

import logging
import os
import platform as host
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import msgspec

import voxlektor
from voxlektor.adapters.audio.dsp import fade_edges, time_stretch, trim_edges
from voxlektor.adapters.audio.dsp import level as dsp_level
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer, MemoryOutput
from voxlektor.adapters.storage.voices_fs import FsVoiceRepository
from voxlektor.adapters.transport.mapping import error_reply, to_command, to_wire_reply
from voxlektor.adapters.transport.unix_server import Subscription, serve_unix
from voxlektor.app.commands import Status
from voxlektor.app.ports import AudioOutput, CuePlayer, PlatformInfo
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.bootstrap.platform_select import select_platform
from voxlektor.bootstrap.registry import load_engine
from voxlektor.contracts.codec import ProtocolError, decode_request, encode_line
from voxlektor.contracts.protocol import ErrorCode, Reply, ReplyError
from voxlektor.domain.settings import Settings
from voxlektor.runtime.buffers import AudioBuffers
from voxlektor.runtime.engine_worker import EngineWorker
from voxlektor.runtime.events import BoundedSubscription, EventHub
from voxlektor.runtime.loop import CoreLoop

_log = logging.getLogger(__name__)

_OUTPUT_ENV = "VOXLEKTOR_OUTPUT"
_VOICES_DIR = "voices"
_MAX_ECHO_ID = 128  # chars of a request id echoed in an error reply
_MAX_ERROR_MESSAGE = 1024  # chars of an error message (it may quote a command name or a value)
_SUBSCRIBER_LIMIT = 1000  # events waiting per subscriber before its connection is closed

type _FileId = tuple[int, int]  # (st_dev, st_ino)


class Daemon(Protocol):
    """A wired service: ``loop`` serves commands; ``run()`` serves until ``Exit``."""

    loop: CoreLoop

    async def run(self) -> None: ...


class _MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class _Daemon:
    def __init__(  # noqa: PLR0913, PLR0917  # private wiring of every collaborator
        self,
        loop: CoreLoop,
        worker: EngineWorker,
        close_cues: Callable[[], None] | None,
        socket_path: Path | None,
        pid_path: Path | None,
        subscribe: Callable[[], Subscription],
    ) -> None:
        self.loop = loop
        self._worker = worker
        self._close_cues = close_cues
        self._socket_path = socket_path
        self._pid_path = pid_path
        self._subscribe = subscribe
        self._socket_id: _FileId | None = None
        self._pid_id: _FileId | None = None

    async def run(self) -> None:
        server = None
        try:
            if self._socket_path is not None:
                server = await serve_unix(
                    self._socket_path,
                    self._handle_line,
                    on_ready=self._on_ready,
                    subscribe=self._subscribe,
                )
            await self.loop.run_until_stopped()
        finally:
            steps: list[tuple[str, Callable[[], None]]] = []
            if server is not None:
                steps.append(("closing the server", server.close))
            steps.append(("removing the runtime files", self._remove_runtime_files))
            if self._close_cues is not None:
                steps.append(("closing the cue player", self._close_cues))
            steps.append(("closing the engine worker", self._worker.close))
            _run_all(steps)

    async def _handle_line(self, line: bytes) -> bytes:
        """One request line in, one reply line out; every failure becomes an error reply."""
        req_id: str | None = None
        try:
            req = decode_request(line)
            req_id = req.id
            reply = await self.loop.handle(to_command(req))
            return _encode(to_wire_reply(req.id, reply))
        except ProtocolError as err:
            return _encode(error_reply(err))
        except Exception as exc:  # a bug; CoreLoop has logged it and is stopping
            internal = ProtocolError(ErrorCode.internal, f"internal error: {exc}", req_id)
            return _encode(error_reply(internal))

    def _on_ready(self) -> None:
        if self._socket_path is not None:
            self._socket_id = _file_id(self._socket_path)
        if self._pid_path is not None:
            tmp = self._pid_path.with_name(self._pid_path.name + ".tmp")
            tmp.write_text(f"{os.getpid()}\n", encoding="ascii")
            os.replace(tmp, self._pid_path)
            self._pid_id = _file_id(self._pid_path)

    def _remove_runtime_files(self) -> None:
        if self._socket_path is not None and self._socket_id is not None:
            _unlink_if_same(self._socket_path, self._socket_id)
        if (
            self._pid_path is not None
            and self._pid_id is not None
            and _holds_our_pid(self._pid_path)
        ):
            _unlink_if_same(self._pid_path, self._pid_id)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _clipped(reply: Reply) -> Reply:
    """An error reply whose echoed id and message are short (they may echo client input)."""
    rid = None if reply.id is None else _clip(reply.id, _MAX_ECHO_ID)
    error = reply.error
    if error is not None:
        error = ReplyError(error.code, _clip(error.message, _MAX_ERROR_MESSAGE))
    return msgspec.structs.replace(reply, id=rid, error=error)


def _encode(reply: Reply) -> bytes:
    """Encode a reply; an error reply never exceeds ``MAX_FRAME``, whatever the client sent.

    Error replies are clipped first (an id or a command name of ~1 MiB is echoed otherwise); an
    ok reply that is too large (its id is) becomes a clipped ``bad_frame`` error reply.
    """
    if not reply.ok:
        reply = _clipped(reply)
    try:
        return encode_line(reply)
    except ProtocolError as err:
        return encode_line(_clipped(error_reply(ProtocolError(err.code, err.message, reply.id))))


def _run_all(steps: list[tuple[str, Callable[[], None]]]) -> None:
    """Run every cleanup step, even when an earlier one fails; failures are logged."""
    for what, step in steps:
        try:
            step()
        except Exception:
            _log.exception("%s failed", what)


def _file_id(path: Path) -> _FileId:
    st = path.lstat()
    return st.st_dev, st.st_ino


def _unlink_if_same(path: Path, file_id: _FileId) -> None:
    try:
        if _file_id(path) == file_id:
            path.unlink()
    except FileNotFoundError:
        pass  # already gone


def _holds_our_pid(path: Path) -> bool:
    try:
        return path.read_text(encoding="ascii").strip() == str(os.getpid())
    except (OSError, UnicodeDecodeError):
        return False


def build_daemon(
    settings: Settings,
    *,
    transport: bool,
    platform: PlatformInfo | None = None,
    output: AudioOutput | None = None,
    cues: CuePlayer | None = None,
) -> Daemon:
    if platform is None:
        platform = select_platform(sys.platform, host.machine(), host.mac_ver()[0])
    repo = FsVoiceRepository(platform.config_dir() / _VOICES_DIR)
    catalog = VoiceCatalog(repo)
    catalog.ensure_bundled()
    socket_path = platform.socket_path() if transport else None
    pid_path = platform.pid_path() if transport else None
    memory = os.environ.get(_OUTPUT_ENV) == "memory"
    if output is None:
        output = MemoryOutput() if memory else _device_output()
    close_cues: Callable[[], None] | None = None
    if cues is None:
        if memory:
            cues = MemoryCuePlayer()
        else:
            cues, close_cues = _device_cues()
    clock = _MonotonicClock()
    session = Session(settings, catalog, clock, version=voxlektor.__version__)
    hub = EventHub()

    def status() -> Mapping[str, object]:
        # Read-only: Status has no effects, and this runs on the loop thread between events.
        reply = session.handle(Status()).reply
        return {} if reply is None else reply.data

    # Starts the engine thread; every chunk: trim edge silence, time_stretch (unless the engine
    # speaks the rate itself), fade the edges, then level (normalise, volume, limit).
    worker = EngineWorker(
        load_engine, repo, time_stretch, level=dsp_level, trim=trim_edges, fade=fade_edges
    )
    try:
        core = CoreLoop(session, worker, output, AudioBuffers(), clock, cues, events=hub)
    except BaseException:
        worker.close()
        raise
    return _Daemon(
        core,
        worker,
        close_cues,
        socket_path,
        pid_path,
        lambda: BoundedSubscription(hub, _SUBSCRIBER_LIMIT, status),
    )


# The device adapters load PortAudio when imported, so they are imported only when selected.


def _device_output() -> AudioOutput:
    from voxlektor.adapters.audio.sounddevice_out import SoundDeviceOutput  # noqa: PLC0415

    return SoundDeviceOutput()


def _device_cues() -> tuple[CuePlayer, Callable[[], None]]:
    """The sounddevice cue player and the function that closes its last stream."""
    from voxlektor.adapters.audio.cues import SoundDeviceCuePlayer  # noqa: PLC0415

    player = SoundDeviceCuePlayer()
    return player, player.close
