"""Daemon autostart: stale file recovery, the spawn argv and the 0.1 s ping loop."""

import asyncio
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.fixtures.sockets import TIMEOUT, request
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer, MemoryOutput
from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.adapters.procs import spawn
from voxlektor.adapters.procs.spawn import ensure_daemon
from voxlektor.bootstrap.daemon import build_daemon
from voxlektor.domain.errors import ServiceUnavailable
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId
from voxlektor.testing.fakes import FakeClock

STATUS = b'{"v":1,"id":"ping","cmd":"status"}\n'
SHUTDOWN = b'{"v":1,"id":"bye","cmd":"shutdown"}\n'


@pytest.fixture
def platform(short_tmp: Path) -> MacOSPlatform:
    return MacOSPlatform(short_tmp)


def _ping(platform: MacOSPlatform) -> bool:
    try:
        return request(platform.socket_path(), STATUS).ok
    except OSError:
        return False


class _Spawned:
    """What the fake ``Popen`` was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the ping loop's clock and sleep; returns the recorded sleeps."""
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(spawn, "_sleep", sleep)
    monkeypatch.setattr(spawn, "_monotonic", clock.now)
    return sleeps


def _fake_popen(monkeypatch: pytest.MonkeyPatch, on_spawn: object = None) -> _Spawned:
    spawned = _Spawned()

    def popen(args: list[str], **kwargs: object) -> _Spawned:
        spawned.calls.append((args, kwargs))
        if callable(on_spawn):
            on_spawn()
        return spawned

    monkeypatch.setattr(spawn.subprocess, "Popen", popen)
    return spawned


def test_running_daemon_is_left_alone(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = _fake_popen(monkeypatch)
    ensure_daemon(platform, lambda: True)
    assert spawned.calls == []


def test_stale_file_is_replaced_by_a_live_daemon_socket(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review Focus 2: a leftover regular file at the socket path does not block autostart."""
    sock_path = platform.socket_path()
    sock_path.write_text("left over by a crash")
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    def start_daemon() -> None:
        assert not sock_path.exists()  # removed before the spawn
        built = build_daemon(
            Settings(engine=EngineId("fake")),
            transport=True,
            platform=platform,
            output=MemoryOutput(),
            cues=MemoryCuePlayer(),
        )

        def serve() -> None:
            try:
                asyncio.run(built.run())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=serve, name="fake-spawned-daemon")
        thread.start()
        threads.append(thread)

    spawned = _fake_popen(monkeypatch, start_daemon)
    try:
        ensure_daemon(platform, lambda: _ping(platform))
        assert stat.S_ISSOCK(sock_path.lstat().st_mode)
        assert _ping(platform)
    finally:
        for thread in threads:
            request(sock_path, SHUTDOWN)
            thread.join(TIMEOUT)
            assert not thread.is_alive()
    assert errors == []
    ((args, kwargs),) = spawned.calls
    assert args == [sys.executable, "-m", "voxlektor", "serve"]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    log = kwargs["stdout"]
    assert kwargs["stderr"] is log
    assert getattr(log, "name", None) == str(platform.log_path())
    assert "shell" not in kwargs


def test_refused_socket_is_left_to_the_daemon(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    """Ruling R34: only the daemon, under its flock, removes a stale socket."""
    sock_path = platform.socket_path()
    seen: list[bool] = []
    _fake_popen(monkeypatch, lambda: seen.append(sock_path.exists()))
    pings = iter([False, False, True])
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(sock_path))  # bound, never listening: nobody serves it
        ensure_daemon(platform, lambda: next(pings))
    assert seen == [True]  # still there when serve is spawned
    assert stat.S_ISSOCK(sock_path.lstat().st_mode)
    assert fake_clock == [0.1]


def test_directory_at_socket_path_is_left_alone(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    platform.socket_path().mkdir()
    spawned = _fake_popen(monkeypatch)
    pings = iter([False, True])
    ensure_daemon(platform, lambda: next(pings))
    assert platform.socket_path().is_dir()
    assert len(spawned.calls) == 1


def test_live_listener_is_never_removed(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    sock_path = platform.socket_path()
    _fake_popen(monkeypatch)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as live:
        live.bind(str(sock_path))
        live.listen(1)
        with pytest.raises(ServiceUnavailable):
            ensure_daemon(platform, lambda: False, wait_s=0.3)
        assert stat.S_ISSOCK(sock_path.lstat().st_mode)


class _FixedPaths:
    """A PlatformInfo whose paths are used as given (MacOSPlatform re-applies 0700)."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def config_dir(self) -> Path:
        return self._root

    def runtime_dir(self) -> Path:
        return self._root

    def socket_path(self) -> Path:
        return self._root / "voxlektor.sock"

    def log_path(self) -> Path:
        return self._root / "voxlektor.log"

    def pid_path(self) -> Path:
        return self._root / "voxlektor.pid"


def test_unremovable_stale_file_is_service_unavailable(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    paths = _FixedPaths(short_tmp / "ro")
    paths.runtime_dir().mkdir()
    paths.socket_path().write_text("left over by a crash")
    spawned = _fake_popen(monkeypatch)
    paths.runtime_dir().chmod(0o500)  # the file cannot be unlinked from a read-only directory
    try:
        with pytest.raises(ServiceUnavailable, match="cannot remove"):
            ensure_daemon(paths, lambda: False)
    finally:
        paths.runtime_dir().chmod(0o700)
    assert spawned.calls == []


def test_gives_up_after_wait_s_pinging_every_tenth_of_a_second(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    _fake_popen(monkeypatch)
    with pytest.raises(ServiceUnavailable, match=r"did not start within 1 s"):
        ensure_daemon(platform, lambda: False, wait_s=1.0)
    assert fake_clock
    assert set(fake_clock) == {0.1}
    assert 0.99 <= sum(fake_clock) <= 1.11  # up to wait_s, plus float rounding of 0.1 steps


def test_serve_exiting_early_twice_is_spawned_again(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    """C3: autostart right after a slow shutdown; the third serve comes up."""
    exited: list[int] = []  # spawn index of every child that has exited

    def popen(args: list[str], **kwargs: object) -> _Spawned:
        child = _Spawned()
        child.calls.append((args, kwargs))
        index = len(spawned)
        spawned.append(child)
        if index < 2:  # the first two have exited ("already running") by the next poll
            child.returncode = 1
            exited.append(index)
        return child

    spawned: list[_Spawned] = []
    monkeypatch.setattr(spawn.subprocess, "Popen", popen)
    pings = iter([False] * 10 + [True])  # the third serve answers after ~1 s
    ensure_daemon(platform, lambda: next(pings))
    assert len(spawned) == 3
    assert exited == [0, 1]
    assert all(c.calls[0][0] == [sys.executable, "-m", "voxlektor", "serve"] for c in spawned)


def test_spawns_are_bounded_to_three(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    spawned = _fake_popen(monkeypatch)
    spawned.returncode = 1  # every serve exits at once
    with pytest.raises(ServiceUnavailable, match="did not start"):
        ensure_daemon(platform, lambda: False, wait_s=2.0)
    assert len(spawned.calls) == 3


def test_running_child_is_not_spawned_again(
    platform: MacOSPlatform, monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    spawned = _fake_popen(monkeypatch)  # returncode None: still starting
    pings = iter([False] * 30 + [True])
    ensure_daemon(platform, lambda: next(pings))
    assert len(spawned.calls) == 1
