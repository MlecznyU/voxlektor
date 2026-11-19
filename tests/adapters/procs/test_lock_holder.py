"""The daemon lock seen from the client: held or not, its holder, and stopping a silent holder."""

import fcntl
import os
import signal
from collections.abc import Iterator
from pathlib import Path

import pytest

from voxlektor.adapters.procs import lock_holder
from voxlektor.adapters.procs.lock_holder import (
    LOCK_NAME,
    LockHolder,
    find_holder,
    lock_held,
    terminate,
)
from voxlektor.testing.fakes import FakeClock

DAEMON = 4242
OTHER = 5151
ME = os.getpid()


@pytest.fixture
def lock(short_tmp: Path) -> Path:
    return short_tmp / LOCK_NAME


@pytest.fixture
def pid_path(short_tmp: Path) -> Path:
    return short_tmp / "voxlektor.pid"


@pytest.fixture
def held(lock: Path) -> Iterator[int]:
    """The lock held by another open file (flock is per open file, also within one process)."""
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield fd
    finally:
        os.close(fd)


def test_lock_name_is_the_daemons() -> None:
    assert LOCK_NAME == "voxlektor.lock"


def test_missing_lock_file_is_not_held_and_is_not_created(lock: Path) -> None:
    assert lock_held(lock) is False
    assert not lock.exists()


def test_free_lock_is_not_held_and_is_left_free(lock: Path) -> None:
    lock.write_text("")
    assert lock_held(lock) is False
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released again: no BlockingIOError
    finally:
        os.close(fd)


def test_lock_held_by_another_open_file_is_held(lock: Path, held: int) -> None:
    assert lock_held(lock) is True


class _Ps:
    """Fake ``lsof -t`` and ``ps -o command=`` answers; records every argv."""

    def __init__(self, openers: list[int], commands: dict[int, str]) -> None:
        self.openers = openers
        self.commands = commands
        self.argvs: list[list[str]] = []

    def output(self, argv: list[str]) -> str | None:
        self.argvs.append(argv)
        if argv[0].endswith("lsof"):
            return "".join(f"{pid}\n" for pid in self.openers)
        pid = int(argv[argv.index("-p") + 1])
        return self.commands.get(pid)


def _ps(
    monkeypatch: pytest.MonkeyPatch, openers: list[int], commands: dict[int, str] | None = None
) -> _Ps:
    fake = _Ps(openers, commands or {})
    monkeypatch.setattr(lock_holder, "_output", fake.output)
    return fake


SERVE = "/usr/bin/python3.12 -m voxlektor serve"


@pytest.mark.parametrize(
    "command",
    [
        SERVE,
        "/Users/me/.local/share/uv/tools/voxlektor/bin/python /Users/me/.local/bin/vxl serve",
        "/Users/me/.local/bin/voxlektor serve --idle 10m",
    ],
)
def test_the_only_opener_running_serve_is_a_verified_holder(
    lock: Path, pid_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    fake = _ps(monkeypatch, [ME, DAEMON], {DAEMON: command})
    assert find_holder(lock, pid_path) == LockHolder(DAEMON, verified=True)
    assert fake.argvs[0] == ["/usr/sbin/lsof", "-t", str(lock)]
    assert fake.argvs[1] == ["/bin/ps", "-p", str(DAEMON), "-o", "command="]


@pytest.mark.parametrize(
    "command",
    [None, "/usr/bin/python3 -m voxlektor say hi", "/bin/zsh", "/usr/bin/vim voxlektor.lock"],
)
def test_an_opener_that_is_no_voxlektor_serve_is_not_verified(
    lock: Path, pid_path: Path, monkeypatch: pytest.MonkeyPatch, command: str | None
) -> None:
    _ps(monkeypatch, [OTHER], {} if command is None else {OTHER: command})
    assert find_holder(lock, pid_path) == LockHolder(OTHER, verified=False)


def test_pid_file_picks_the_holder_among_several_openers(
    lock: Path, pid_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_path.write_text(f"{DAEMON}\n")
    _ps(monkeypatch, [OTHER, DAEMON], {DAEMON: SERVE, OTHER: SERVE})
    assert find_holder(lock, pid_path) == LockHolder(DAEMON, verified=True)


def test_pid_file_pid_without_the_lock_open_is_not_verified(
    lock: Path, pid_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_path.write_text(f"{DAEMON}\n")
    _ps(monkeypatch, [], {DAEMON: SERVE})  # e.g. lsof failed: never trust the pid file alone
    assert find_holder(lock, pid_path) == LockHolder(DAEMON, verified=False)


def test_several_openers_without_a_pid_file_are_ambiguous(
    lock: Path, pid_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ps(monkeypatch, [OTHER, DAEMON], {DAEMON: SERVE, OTHER: SERVE})
    assert find_holder(lock, pid_path) == LockHolder(None, verified=False)


@pytest.mark.parametrize("pid_text", ["", "not a pid\n", "-3\n"])
def test_unreadable_pid_file_and_no_openers_is_an_unknown_holder(
    lock: Path, pid_path: Path, monkeypatch: pytest.MonkeyPatch, pid_text: str
) -> None:
    pid_path.write_text(pid_text)
    _ps(monkeypatch, [])
    assert find_holder(lock, pid_path) == LockHolder(None, verified=False)


def test_lsof_noise_is_ignored(lock: Path, pid_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ps(monkeypatch, [], {DAEMON: SERVE})

    def output(argv: list[str]) -> str | None:
        return f"junk\n{DAEMON}\n\n" if argv[0].endswith("lsof") else fake.output(argv)

    monkeypatch.setattr(lock_holder, "_output", output)
    assert find_holder(lock, pid_path) == LockHolder(DAEMON, verified=True)


def test_output_runs_a_fixed_argv_and_returns_none_on_failure() -> None:
    assert lock_holder._output(["/bin/echo", "hi"]) == "hi\n"  # pyright: ignore[reportPrivateUsage]
    assert lock_holder._output(["/usr/bin/false"]) is None  # pyright: ignore[reportPrivateUsage]
    assert lock_holder._output(["/no/such/tool"]) is None  # pyright: ignore[reportPrivateUsage]


def test_output_gives_up_on_a_tool_that_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lock_holder, "_TOOL_TIMEOUT_S", 0.05)
    assert lock_holder._output(["/bin/sleep", "5"]) is None  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------- terminate


class _Process:
    """A fake process: exits on the signals in ``dies_on``, ignores the others."""

    def __init__(self, dies_on: set[int], *, gone: bool = False) -> None:
        self.dies_on = dies_on
        self.alive = not gone
        self.signals: list[int] = []

    def kill(self, pid: int, sig: int) -> None:
        assert pid == DAEMON
        if sig != 0:
            self.signals.append(sig)
        if not self.alive:
            raise ProcessLookupError
        if sig in self.dies_on:
            self.alive = False


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(lock_holder, "_monotonic", fake.now)
    monkeypatch.setattr(lock_holder, "_sleep", fake.advance)
    return fake


def _process(monkeypatch: pytest.MonkeyPatch, process: _Process) -> _Process:
    monkeypatch.setattr(lock_holder, "_kill", process.kill)
    return process


def test_holder_that_exits_on_sigterm_is_terminated(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    process = _process(monkeypatch, _Process({signal.SIGTERM}))
    assert terminate(DAEMON, grace_s=3.0) == "terminated"
    assert process.signals == [signal.SIGTERM]
    assert clock.now() < 3.0


def test_holder_that_ignores_sigterm_is_killed_after_the_grace(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    process = _process(monkeypatch, _Process({signal.SIGKILL}))
    assert terminate(DAEMON, grace_s=3.0) == "killed"
    assert process.signals == [signal.SIGTERM, signal.SIGKILL]
    assert clock.now() >= 3.0


def test_holder_that_survives_sigkill_is_reported(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    process = _process(monkeypatch, _Process(set()))
    assert terminate(DAEMON, grace_s=3.0) == "survived"
    assert process.signals == [signal.SIGTERM, signal.SIGKILL]


def test_holder_already_gone_is_reported(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    process = _process(monkeypatch, _Process(set(), gone=True))
    assert terminate(DAEMON, grace_s=3.0) == "gone"
    assert process.signals == [signal.SIGTERM]


@pytest.mark.parametrize(
    ("on_sigkill", "outcome"),
    [(ProcessLookupError, "terminated"), (PermissionError, "survived")],
)
def test_sigkill_that_finds_the_holder_gone_or_foreign(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    on_sigkill: type[OSError],
    outcome: str,
) -> None:
    """The pid exited right after the grace, or was reused by another user's process."""

    def kill(pid: int, sig: int) -> None:
        if sig == signal.SIGKILL:
            raise on_sigkill
        if sig == 0 and clock.now() >= 1.0:
            raise PermissionError  # exists but is not ours: still alive

    monkeypatch.setattr(lock_holder, "_kill", kill)
    assert terminate(DAEMON, grace_s=3.0) == outcome


def test_holder_of_another_user_counts_as_alive(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    def kill(pid: int, sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(lock_holder, "_kill", kill)
    assert terminate(DAEMON, grace_s=3.0) == "survived"
