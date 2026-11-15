"""The real CLI in a subprocess against a daemon it autostarts (FakeEngine, memory output).

Teardown stops every ``voxlektor serve`` process started with this test's ``VOXLEKTOR_HOME``,
found by its environment (so also one spawned before its pid file existed), and asserts that
none remains and the pid file is gone.
"""

import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pytest

import voxlektor
from tests.fixtures.sockets import wait_for_exit, wait_for_file

pytestmark = pytest.mark.e2e

CLI_TIMEOUT = 60.0
_PGREP = "/usr/bin/pgrep"
_PS = "/bin/ps"


@dataclass
class _Cli:
    env: dict[str, str]
    home: Path

    @property
    def run_dir(self) -> Path:
        return self.home / "run"

    @property
    def pid_path(self) -> Path:
        return self.run_dir / "voxlektor.pid"

    @property
    def socket_path(self) -> Path:
        return self.run_dir / "voxlektor.sock"

    def __call__(self, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "voxlektor", *args],
            env=self.env,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
            check=False,
        )


def _serve_pids(home: Path) -> list[int]:
    """Pids of ``voxlektor serve`` processes whose environment has this VOXLEKTOR_HOME."""
    found = subprocess.run(  # noqa: S603
        [_PGREP, "-f", "voxlektor serve"], capture_output=True, text=True, check=False
    )
    marker = f"VOXLEKTOR_HOME={home}"
    pids: list[int] = []
    for pid in (int(word) for word in found.stdout.split()):
        shown = subprocess.run(  # noqa: S603
            [_PS, "-E", "-ww", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if marker in shown.stdout.split():
            pids.append(pid)
    return pids


def _stop_all(home: Path) -> None:
    for pid in _serve_pids(home):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        wait_for_exit(pid, CLI_TIMEOUT)


@pytest.fixture
def vxl(short_tmp: Path, setup_completed: Path) -> Iterator[_Cli]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("VOXLEKTOR_")}
    env |= {
        "VOXLEKTOR_HOME": str(short_tmp),
        "VOXLEKTOR_ENGINE": "fake",
        "VOXLEKTOR_OUTPUT": "memory",
    }
    cli = _Cli(env, short_tmp)
    try:
        yield cli
    finally:
        _stop_all(short_tmp)
        assert _serve_pids(short_tmp) == []
        assert not cli.pid_path.exists()


def _stale_regular_file(path: Path) -> None:
    path.write_text("left over by a crash")


def _stale_refused_socket(path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(path))  # closed without listening: connect is refused


@pytest.mark.parametrize(
    "leave_stale", [_stale_regular_file, _stale_refused_socket], ids=["file", "socket"]
)
def test_speak_autostarts_then_status_stop_status(
    vxl: _Cli, leave_stale: Callable[[Path], None]
) -> None:
    """Review Focus 2 end to end: whatever a crash left at the socket path, autostart works."""
    vxl.run_dir.mkdir(mode=0o700, parents=True)
    leave_stale(vxl.socket_path)
    status_before = vxl("status")
    assert status_before.returncode == 0, status_before.stderr
    assert status_before.stdout.strip() == "OFF"
    spoken = vxl("test")
    assert spoken.returncode == 0, spoken.stderr
    assert vxl.pid_path.exists()
    status = vxl("status")
    assert status.returncode == 0, status.stderr
    assert "WARM" in status.stdout
    assert "engine: fake" in status.stdout
    stop = vxl("stop")
    assert stop.returncode == 0, stop.stderr
    assert stop.stdout.strip() == "stopped"
    assert not vxl.socket_path.exists()
    off = vxl("status")
    assert off.returncode == 0, off.stderr
    assert off.stdout.strip() == "OFF"
    wait_for_file(vxl.pid_path, CLI_TIMEOUT, present=False)


def test_empty_input_never_starts_a_daemon(vxl: _Cli) -> None:
    empty = vxl("")
    assert empty.returncode == 0
    assert empty.stdout.strip() == "nothing to read"
    piped = vxl(stdin="\n")
    assert piped.returncode == 0
    assert piped.stdout.strip() == "nothing to read"
    assert not vxl.socket_path.exists()
    assert not vxl.pid_path.exists()
    assert _serve_pids(vxl.home) == []


def test_setup_gate_then_flag_mode_setup_then_speaking(vxl: _Cli) -> None:
    """A home that was never set up: speaking is refused until `vxl setup --yes` ran."""
    (vxl.home / "config" / "config.toml").write_text("config_version = 1\n")
    refused = vxl("test")
    assert refused.returncode == 1
    assert "voxlektor is not set up yet; run: vxl setup" in refused.stderr
    no_tty = vxl("setup")
    assert no_tty.returncode == 1
    assert "vxl setup --yes" in no_tty.stderr
    done = vxl("setup", "--yes", "--engine", "fake", "--volume", "40")
    assert done.returncode == 0, done.stderr
    assert "volume = 40" in done.stdout
    assert not vxl.pid_path.exists()  # flag mode starts no daemon
    spoken = vxl("test")
    assert spoken.returncode == 0, spoken.stderr
    assert vxl("stop").returncode == 0


def test_control_without_daemon_exits_2_and_stop_is_quiet(vxl: _Cli) -> None:
    pause = vxl("pause")
    assert pause.returncode == 2
    assert "not running" in pause.stderr
    stop = vxl("stop")
    assert stop.returncode == 0
    assert stop.stdout.strip() == "not running"


class _JsonLines:
    """Reads one JSON object per line from a pipe, each within a deadline (no blocking reads)."""

    def __init__(self, pipe: IO[bytes]) -> None:
        self._fd = pipe.fileno()
        self._buffer = b""

    def next(self, timeout: float = CLI_TIMEOUT) -> dict[str, object] | None:
        """The next object; ``None`` at the end of the output."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            ready, _, _ = select.select([self._fd], [], [], max(deadline - time.monotonic(), 0))
            if not ready:
                raise TimeoutError(f"no line within {timeout} s; got {self._buffer!r}")
            piece = os.read(self._fd, 65536)
            if not piece:
                return None
            self._buffer += piece
        line, _, self._buffer = self._buffer.partition(b"\n")
        return json.loads(line)


def test_watch_follows_a_reading_until_the_daemon_stops(vxl: _Cli) -> None:
    """The real ``vxl watch --json --start`` against the daemon it starts (memory output)."""
    watch = subprocess.Popen(
        [sys.executable, "-m", "voxlektor", "watch", "--json", "--start"],
        env=vxl.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert watch.stdout is not None
        lines = _JsonLines(watch.stdout)
        snapshot = lines.next()
        assert snapshot is not None
        assert snapshot["event"] == "snapshot"
        assert snapshot["version"] == voxlektor.__version__
        assert "daemon_outdated" not in snapshot
        spoken = vxl("say", "--owner", "s1", "--tag", "t0", "Hello there, this is a test.")
        assert spoken.returncode == 0, spoken.stderr
        seen: list[dict[str, object]] = []
        while not seen or seen[-1] != {"event": "drained", "owner": None, "group": None}:
            event = lines.next()
            assert event is not None, seen
            seen.append(event)
        names = [(e["event"], e.get("tag"), e.get("reason")) for e in seen]
        assert ("tag_started", "t0", None) in names
        assert ("tag_finished", "t0", "done") in names
        assert {"event": "floor", "owner": "s1", "group": "g1"} in seen
        assert {"event": "drained", "owner": "s1", "group": None} in seen
        assert vxl("stop").returncode == 0
        rest: list[dict[str, object]] = []
        while (event := lines.next()) is not None:
            rest.append(event)
        assert rest[-1] == {"event": "disconnected"}
        assert watch.wait(CLI_TIMEOUT) == 2
    finally:
        if watch.poll() is None:
            watch.kill()
            watch.wait(CLI_TIMEOUT)
        for pipe in (watch.stdout, watch.stderr):
            if pipe is not None:
                pipe.close()
