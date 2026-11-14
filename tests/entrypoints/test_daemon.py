"""``serve``: platform check, single instance, stale socket, log file, signals and shutdown.

``serve`` runs in the test's main thread (so its signal handlers are real); a helper thread
plays the client once the pid file appears.
"""

import fcntl
import json
import logging
import logging.handlers
import os
import queue
import signal
import socket
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.fixtures.sockets import TIMEOUT, request, wait_for_file
from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.app.ports import PlatformInfo
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId
from voxlektor.entrypoints import daemon as entry
from voxlektor.entrypoints.daemon import serve

FAKE = Settings(engine=EngineId("fake"))
SHUTDOWN_LINE = b'{"v":1,"id":"bye","cmd":"shutdown"}\n'


@pytest.fixture
def home(short_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("VOXLEKTOR_HOME", str(short_tmp))
    monkeypatch.setenv("VOXLEKTOR_OUTPUT", "memory")
    yield short_tmp / "run"
    handlers = logging.getLogger().handlers
    assert not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers)


def _client(run_dir: Path, action: Callable[[Path], None]) -> threading.Thread:
    """Start a thread that waits until the daemon is ready, then runs ``action``."""
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def main() -> None:
        wait_for_file(run_dir / "voxlektor.pid")
        action(run_dir / "voxlektor.sock")

    thread = threading.Thread(target=main)
    thread.start()
    return thread


def test_unsupported_platform_exits_3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unsupported(*args: str) -> PlatformInfo:
        return UnsupportedPlatform()

    monkeypatch.setattr(entry, "select_platform", unsupported)
    assert serve(FAKE) == 3
    assert "platform not supported yet" in capsys.readouterr().err


@pytest.mark.parametrize(
    "answer",
    [
        b'{"v":1,"id":"ID","ok":true,"data":{"state":"WARM"}}\n',
        b'{"v":1,"id":"ID","ok":false,"error":{"code":"internal","message":"bye"}}\n',
        b'{"v":1,"id":"someone-else","ok":true,"data":{}}\n',
        b"garbage\n",
    ],
    ids=["ready", "shutting-down", "foreign-reply", "undecodable"],
)
def test_anything_answering_on_the_socket_means_already_running(
    home: Path, capsys: pytest.CaptureFixture[str], answer: bytes
) -> None:
    home.mkdir(mode=0o700, parents=True)
    path = home / "voxlektor.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        listener.listen(1)

        def reply() -> None:
            conn, _ = listener.accept()
            with conn:
                probe_id = json.loads(conn.recv(4096))["id"]  # echoed like a real daemon
                conn.sendall(answer.replace(b'"ID"', json.dumps(probe_id).encode()))

        other = threading.Thread(target=reply)
        other.start()
        assert serve(FAKE) == 1
        other.join(TIMEOUT)
    assert "already running" in capsys.readouterr().out
    assert path.exists()  # the other process's socket is left alone


def _hold_lock(run_dir: Path) -> int:
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(run_dir / "voxlektor.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_a_held_lock_that_is_never_released_means_already_running(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(entry, "_LOCK_WAIT_S", 0.2)
    fd = _hold_lock(home)
    try:
        assert serve(FAKE) == 1
    finally:
        os.close(fd)
    assert "already running" in capsys.readouterr().out


def test_serve_waits_for_a_daemon_that_is_shutting_down(home: Path) -> None:
    """The old daemon holds the lock and answers "shutting down"; serve waits, then runs."""
    fd = _hold_lock(home)
    path = home / "voxlektor.sock"
    codes: queue.Queue[int] = queue.Queue()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as old:
        old.bind(str(path))
        old.listen(1)
        old.settimeout(TIMEOUT)
        waiting = threading.Thread(target=lambda: codes.put(serve(FAKE)))
        waiting.start()
        conn, _ = old.accept()  # serve found the lock taken and probes the old daemon
        with conn:
            probe_id = json.dumps(json.loads(conn.recv(4096))["id"]).encode()
            conn.sendall(
                b'{"v":1,"id":' + probe_id + b',"ok":false,'
                b'"error":{"code":"internal","message":"the service is shutting down"}}\n'
            )
    path.unlink()  # the old daemon removes its socket ...
    client = _client(home, lambda sock: request(sock, SHUTDOWN_LINE) and None)
    os.close(fd)  # ... and exits, releasing the lock
    assert codes.get(timeout=TIMEOUT) == 0
    waiting.join(TIMEOUT)
    client.join(TIMEOUT)


def test_two_concurrent_serves_run_exactly_one_daemon(home: Path) -> None:
    home.mkdir(mode=0o700, parents=True)
    codes: queue.Queue[int] = queue.Queue()
    threads = [threading.Thread(target=lambda: codes.put(serve(FAKE))) for _ in range(2)]
    for thread in threads:
        thread.start()
    # The loser waits for the lock until the winner answers status, then gives up.
    assert codes.get(timeout=TIMEOUT) == 1
    wait_for_file(home / "voxlektor.pid")
    assert request(home / "voxlektor.sock", SHUTDOWN_LINE).ok
    assert codes.get(timeout=TIMEOUT) == 0
    for thread in threads:
        thread.join(TIMEOUT)
        assert not thread.is_alive()


def test_stale_socket_is_replaced_and_shutdown_cleans_up(home: Path) -> None:
    home.mkdir(mode=0o700, parents=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(home / "voxlektor.sock"))  # a real socket inode nobody listens on
    replies: list[bool] = []

    def shutdown(path: Path) -> None:
        replies.append(request(path, b'{"v":1,"id":"st","cmd":"status"}\n').ok)
        replies.append(request(path, b'{"v":1,"id":"bye","cmd":"shutdown"}\n').ok)

    client = _client(home, shutdown)
    assert serve(FAKE) == 0
    client.join(TIMEOUT)
    assert replies == [True, True]
    assert not (home / "voxlektor.sock").exists()
    assert not (home / "voxlektor.pid").exists()
    log = (home / "voxlektor.log").read_text()
    assert "daemon started" in log
    assert "daemon stopped" in log


def test_sigterm_shuts_down_gracefully(home: Path) -> None:
    client = _client(home, lambda path: os.kill(os.getpid(), signal.SIGTERM))
    assert serve(FAKE) == 0
    client.join(TIMEOUT)
    assert not (home / "voxlektor.sock").exists()
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_a_failing_daemon_is_logged_and_exits_2(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("wiring failed")

    monkeypatch.setattr(entry, "build_daemon", broken)
    assert serve(FAKE) == 2  # service unavailable
    assert "wiring failed" in (home / "voxlektor.log").read_text()
