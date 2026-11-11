import asyncio
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.fixtures.sockets import TIMEOUT, request, wait_for_file
from voxlektor.adapters.audio import cues as cues_module
from voxlektor.adapters.audio import sounddevice_out
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer, MemoryOutput
from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.app.commands import Command, Shutdown, Status
from voxlektor.app.effects import Reply
from voxlektor.bootstrap import daemon as daemon_module
from voxlektor.bootstrap.daemon import Daemon, build_daemon
from voxlektor.contracts import protocol as p
from voxlektor.contracts.engine import AudioChunk, Engine, VoicePrompt
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import ChunkId, EngineId
from voxlektor.testing.fakes import FakeEngine

FAKE = Settings(engine=EngineId("fake"))
SPEAK = b'{"v":1,"id":"s","cmd":"speak","text":"Hello there.","lang":"en"}\n'
SHUTDOWN = b'{"v":1,"id":"bye","cmd":"shutdown"}\n'
STATUS = b'{"v":1,"id":"st","cmd":"status"}\n'


class _Forbidden:
    def __init__(self) -> None:
        raise AssertionError("a device adapter was built")


def _stop(built: Daemon) -> None:
    async def main() -> None:
        runner = asyncio.create_task(built.run())
        assert (await built.loop.handle(Shutdown())).ok
        await asyncio.wait_for(runner, TIMEOUT)

    asyncio.run(main())


class _Serving:
    """A built daemon running in its own thread with its own event loop."""

    def __init__(self, built: Daemon) -> None:
        self.errors: list[BaseException] = []
        self.thread = threading.Thread(target=self._serve, args=(built,))
        self.thread.start()

    def _serve(self, built: Daemon) -> None:
        try:
            asyncio.run(built.run())
        except BaseException as exc:
            self.errors.append(exc)

    def join(self) -> None:
        self.thread.join(TIMEOUT)
        assert not self.thread.is_alive()


@pytest.fixture
def no_leaked_threads() -> Iterator[None]:
    before = set(threading.enumerate())
    yield
    assert [t for t in threading.enumerate() if t not in before and t.is_alive()] == []


# ------------------------------------------------------------------ adapters


def test_voxlektor_output_memory_selects_device_less_adapters(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXLEKTOR_OUTPUT", "memory")
    monkeypatch.setattr(sounddevice_out, "SoundDeviceOutput", _Forbidden)
    monkeypatch.setattr(cues_module, "SoundDeviceCuePlayer", _Forbidden)
    built = build_daemon(FAKE, transport=False, platform=MacOSPlatform(short_tmp))
    _stop(built)


def test_memory_mode_wiring_does_not_load_portaudio() -> None:
    code = (
        "import sys, voxlektor.bootstrap.daemon; "
        "assert 'sounddevice' not in sys.modules, 'sounddevice imported'"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_without_injection_or_env_the_device_adapters_are_used(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[str] = []

    class _DeviceCues(MemoryCuePlayer):
        def close(self) -> None:
            made.append("cues closed")

    def device_output() -> MemoryOutput:
        made.append("output")
        return MemoryOutput()

    def device_cues() -> _DeviceCues:
        made.append("cues")
        return _DeviceCues()

    monkeypatch.delenv("VOXLEKTOR_OUTPUT", raising=False)
    monkeypatch.setattr(sounddevice_out, "SoundDeviceOutput", device_output)
    monkeypatch.setattr(cues_module, "SoundDeviceCuePlayer", device_cues)
    built = build_daemon(FAKE, transport=False, platform=MacOSPlatform(short_tmp))
    _stop(built)
    assert made == ["output", "cues", "cues closed"]  # the cue stream is closed at shutdown


def test_default_platform_installs_bundled_voices_under_the_config_dir(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXLEKTOR_HOME", str(short_tmp))
    monkeypatch.setenv("VOXLEKTOR_OUTPUT", "memory")
    built = build_daemon(FAKE, transport=False)

    async def main() -> None:
        runner = asyncio.create_task(built.run())
        status = await built.loop.handle(Status())
        assert status.data["state"] == "cold"
        await built.loop.handle(Shutdown())
        await asyncio.wait_for(runner, TIMEOUT)

    asyncio.run(main())
    assert (short_tmp / "config" / "voices" / "v0-default" / "voice.toml").is_file()
    assert not (short_tmp / "run" / "voxlektor.sock").exists()


# ------------------------------------------------------------------ failures


@pytest.mark.usefixtures("no_leaked_threads")
def test_a_socket_that_cannot_be_served_closes_the_worker(short_tmp: Path) -> None:
    class _NoDir(MacOSPlatform):
        def socket_path(self) -> Path:
            return short_tmp / "missing" / "s.sock"

    built = build_daemon(FAKE, transport=True, platform=_NoDir(short_tmp), output=MemoryOutput())
    with pytest.raises(FileNotFoundError):
        asyncio.run(built.run())


@pytest.mark.usefixtures("no_leaked_threads")
def test_a_failing_core_loop_closes_the_engine_thread(short_tmp: Path) -> None:
    class _NoCallbacks(MemoryOutput):
        def set_callbacks(
            self,
            on_done: Callable[[ChunkId], None],
            on_failed: Callable[[ChunkId, str], None],
        ) -> None:
            raise RuntimeError("callbacks refused")

    platform = MacOSPlatform(short_tmp)
    with pytest.raises(RuntimeError, match="callbacks refused"):
        build_daemon(FAKE, transport=False, platform=platform, output=_NoCallbacks())


@pytest.mark.usefixtures("no_leaked_threads")
def test_an_exception_while_handling_a_request_becomes_an_internal_error_reply(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = MacOSPlatform(short_tmp)
    built = build_daemon(FAKE, transport=True, platform=platform, output=MemoryOutput())
    original = built.loop.handle

    async def broken(cmd: Command) -> Reply:
        if isinstance(cmd, Status):
            raise RuntimeError("status is broken")
        return await original(cmd)

    monkeypatch.setattr(built.loop, "handle", broken)
    serving = _Serving(built)
    try:
        wait_for_file(platform.pid_path())
        reply = request(platform.socket_path(), STATUS)
    finally:
        request(platform.socket_path(), SHUTDOWN)
        serving.join()
    assert reply.id == "st"
    assert reply.error is not None
    assert reply.error.code is p.ErrorCode.internal
    assert "status is broken" in reply.error.message
    assert serving.errors == []


# ------------------------------------------------------------------ runtime files (R30)


class _GatedEngine(FakeEngine):
    """Synthesis blocks until ``gate`` is set: a slow engine job that is still queued."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        self.entered.set()
        assert self.gate.wait(TIMEOUT)
        return super().synthesize(text, lang, prompt)


@pytest.mark.usefixtures("no_leaked_threads")
def test_a_successor_keeps_its_socket_and_pid_when_the_old_daemon_exits(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gated = _GatedEngine()
    engines: list[FakeEngine] = []

    def provider(engine_id: str) -> Engine:
        engines.append(FakeEngine() if engines else gated)
        return engines[-1]

    monkeypatch.setattr(daemon_module, "load_engine", provider)
    platform = MacOSPlatform(short_tmp)
    sock, pid = platform.socket_path(), platform.pid_path()

    def start() -> _Serving:
        output = MemoryOutput(auto_complete=False)
        built = build_daemon(FAKE, transport=True, platform=platform, output=output)
        return _Serving(built)

    old = start()
    try:
        wait_for_file(pid)
        assert request(sock, SPEAK).ok
        assert gated.entered.wait(TIMEOUT)  # the old daemon's engine thread is busy
        assert request(sock, SHUTDOWN).ok
        # The old daemon removes its socket and pid file before it waits for its worker.
        wait_for_file(sock, present=False)
        wait_for_file(pid, present=False)
        new = start()
        wait_for_file(pid)
    finally:
        gated.gate.set()
    old.join()  # the old daemon finishes its slow job and exits after the successor started
    assert old.errors == []
    assert sock.exists()
    assert pid.read_text().strip() == str(os.getpid())
    assert request(sock, STATUS).ok  # the successor still serves
    assert request(sock, SHUTDOWN).ok
    new.join()
    assert not sock.exists()
    assert not pid.exists()


@pytest.mark.usefixtures("no_leaked_threads")
def test_the_daemon_never_removes_runtime_files_it_does_not_own(short_tmp: Path) -> None:
    platform = MacOSPlatform(short_tmp)
    sock, pid = platform.socket_path(), platform.pid_path()
    serving = _Serving(build_daemon(FAKE, transport=True, platform=platform, output=MemoryOutput()))
    wait_for_file(pid)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(TIMEOUT)
        conn.connect(str(sock))
        # Someone else takes over both paths (new inodes; the pid file even holds our pid).
        sock.unlink()
        sock.write_text("successor")
        pid.unlink()
        pid.write_text(f"{os.getpid()}\n")
        conn.sendall(SHUTDOWN)
        assert conn.recv(4096).startswith(b'{"v":1,"id":"bye","ok":true')
    serving.join()
    assert sock.read_text() == "successor"
    assert pid.exists()


@pytest.mark.usefixtures("no_leaked_threads")
def test_every_cleanup_step_runs_even_when_removing_the_files_fails(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    closed: list[str] = []

    class _DeviceCues(MemoryCuePlayer):
        def close(self) -> None:
            closed.append("cues")

    def refuse(path: Path, file_id: object) -> None:
        raise PermissionError(f"cannot inspect {path.name}")

    monkeypatch.delenv("VOXLEKTOR_OUTPUT", raising=False)
    monkeypatch.setattr(sounddevice_out, "SoundDeviceOutput", MemoryOutput)
    monkeypatch.setattr(cues_module, "SoundDeviceCuePlayer", _DeviceCues)
    monkeypatch.setattr(daemon_module, "_unlink_if_same", refuse)
    platform = MacOSPlatform(short_tmp)
    serving = _Serving(build_daemon(FAKE, transport=True, platform=platform))
    wait_for_file(platform.pid_path())
    assert request(platform.socket_path(), SHUTDOWN).ok
    serving.join()  # the engine thread was closed too (no_leaked_threads)
    assert serving.errors == []
    assert closed == ["cues"]
    assert "removing the runtime files failed" in caplog.text
    assert "cannot inspect voxlektor.sock" in caplog.text


@pytest.mark.usefixtures("no_leaked_threads")
def test_a_pid_file_rewritten_with_another_pid_is_left_alone(short_tmp: Path) -> None:
    platform = MacOSPlatform(short_tmp)
    pid = platform.pid_path()
    serving = _Serving(build_daemon(FAKE, transport=True, platform=platform, output=MemoryOutput()))
    wait_for_file(pid)
    with pid.open("w") as handle:  # same inode, other content
        handle.write("99999\n")
    assert request(platform.socket_path(), SHUTDOWN).ok
    serving.join()
    assert pid.read_text() == "99999\n"
    assert not platform.socket_path().exists()
