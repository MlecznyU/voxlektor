"""``vxl watch``: the daemon's event stream on stdout, filtered by owner, until it goes away."""

import io
import json
import os
import signal
from collections.abc import Callable
from pathlib import Path

import pytest

import voxlektor
from tests.adapters.transport.test_unix_stream import EventServer, frame, stream_of
from voxlektor.app.ports import PlatformInfo
from voxlektor.bootstrap import client
from voxlektor.domain.errors import ServiceUnavailable
from voxlektor.entrypoints.cli import watch
from voxlektor.entrypoints.cli.app import main

VERSION = voxlektor.__version__
SNAPSHOT: dict[str, object] = {
    "state": "WARM",
    "playing": False,
    "owner": None,
    "group": None,
    "version": VERSION,
}
# The --json snapshot line: the snapshot plus the protocol version, for tools to gate on.
SNAPSHOT_LINE: dict[str, object] = {"event": "snapshot", "protocol": 1, **SNAPSHOT}


@pytest.fixture
def sock(fresh_home: Path) -> Path:
    return client.build_client({}).platform.socket_path()


def _json_lines(out: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in out.splitlines()]


def _run(server: EventServer, *args: str) -> int:
    try:
        return main(["watch", *args])
    finally:
        server.join()


def test_watch_prints_snapshot_first(sock: Path, capsys: pytest.CaptureFixture[str]) -> None:
    server = EventServer(
        sock,
        stream_of(
            frame("snapshot", SNAPSHOT),
            frame("floor", {"owner": "a", "group": "g1"}),
        ),
    )
    _run(server, "--json")
    (sent,) = server.received
    assert sent["cmd"] == "subscribe"
    lines = _json_lines(capsys.readouterr().out)
    assert lines[0] == SNAPSHOT_LINE
    assert lines[1] == {"event": "floor", "owner": "a", "group": "g1"}


def test_watch_json_snapshot_keeps_a_protocol_the_daemon_sent(
    sock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = EventServer(sock, stream_of(frame("snapshot", {**SNAPSHOT, "protocol": 7})))
    _run(server, "--json")
    assert _json_lines(capsys.readouterr().out)[0]["protocol"] == 7


def test_watch_disconnected_exit_2(sock: Path, capsys: pytest.CaptureFixture[str]) -> None:
    server = EventServer(sock, stream_of(frame("snapshot", SNAPSHOT)))
    assert _run(server, "--json") == 2
    assert _json_lines(capsys.readouterr().out) == [
        SNAPSHOT_LINE,
        {"event": "disconnected"},
    ]


class _FlushLog(io.StringIO):
    """Records what had been written each time ``flush`` was called."""

    def __init__(self) -> None:
        super().__init__()
        self.flushed: list[str] = []

    def flush(self) -> None:
        self.flushed.append(self.getvalue())


def test_watch_flushes_per_line(sock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = _FlushLog()
    monkeypatch.setattr("sys.stdout", log)
    server = EventServer(
        sock,
        stream_of(frame("snapshot", SNAPSHOT), frame("state", {"state": "UNLOADING"})),
    )
    _run(server, "--json")
    lines = log.getvalue().splitlines(keepends=True)
    assert len(lines) == 3
    for count in range(1, 4):
        assert "".join(lines[:count]) in log.flushed


def test_watch_waits_for_daemon(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sleeps: list[float] = []
    servers: list[EventServer] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:  # off for two attempts: daemon_off is still printed once
            servers.append(EventServer(sock, stream_of(frame("snapshot", SNAPSHOT))))

    monkeypatch.setattr(watch, "_sleep", sleep)
    try:
        assert main(["watch", "--json"]) == 2
    finally:
        for server in servers:
            server.join()
    assert sleeps == [1.0, 1.0]
    assert _json_lines(capsys.readouterr().out) == [
        {"event": "daemon_off"},
        SNAPSHOT_LINE,
        {"event": "disconnected"},
    ]


def test_watch_start_starts_daemon(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    servers: list[EventServer] = []

    def ensure(platform: PlatformInfo, ping: object) -> None:
        assert platform.socket_path() == sock
        servers.append(EventServer(sock, stream_of(frame("snapshot", SNAPSHOT))))

    monkeypatch.setattr(client, "ensure_daemon", ensure)

    def no_retry(seconds: float) -> None:
        pytest.fail("no retry expected")

    monkeypatch.setattr(watch, "_sleep", no_retry)
    try:
        assert main(["watch", "--json", "--start"]) == 2
    finally:
        for server in servers:
            server.join()
    assert len(servers) == 1
    assert _json_lines(capsys.readouterr().out)[0] == SNAPSHOT_LINE


def test_watch_without_start_never_starts_daemon(
    sock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ensure(platform: PlatformInfo, ping: object) -> None:
        pytest.fail("must not start")

    monkeypatch.setattr(client, "ensure_daemon", ensure)
    server = EventServer(sock, stream_of(frame("snapshot", SNAPSHOT)))
    assert _run(server, "--json") == 2


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_watch_sigterm_exit_0(
    sock: Path, capsys: pytest.CaptureFixture[str], signum: signal.Signals
) -> None:
    before = signal.getsignal(signal.SIGTERM)
    server = EventServer(
        sock,
        stream_of(frame("snapshot", SNAPSHOT)),
        then=lambda: os.kill(os.getpid(), signum),
        hold=True,
    )
    assert _run(server, "--json") == 0
    assert _json_lines(capsys.readouterr().out) == [SNAPSHOT_LINE]
    assert signal.getsignal(signal.SIGTERM) == before


OWNER_STREAM: list[tuple[str, dict[str, object]]] = [
    ("error", {"owner": "a", "group": "g", "code": "internal", "message": "mine"}),
    ("error", {"owner": "b", "group": "h", "code": "internal", "message": "theirs"}),
    ("error", {"owner": None, "group": None, "code": "internal", "message": "daemon"}),
    ("floor", {"owner": "b", "group": "h"}),
    ("state", {"state": "LOADING", "owner": None, "group": None}),
    ("waiting", {"owner": "a", "group": "g", "position": 1}),
    ("waiting", {"owner": "b", "group": "h", "position": 1}),
    ("tag_started", {"owner": "a", "group": "g", "tag": "t1"}),
    ("tag_finished", {"owner": "b", "group": "h", "tag": "t2", "reason": "done"}),
    ("paused", {"owner": None, "group": None}),
    ("paused", {"owner": "b", "group": None}),
    ("resumed", {"owner": None, "group": None}),
    ("resumed", {"owner": "a", "group": None}),
    ("drained", {"owner": "b", "group": None}),
    ("drained", {"owner": "a", "group": None}),
    ("drained", {"owner": None, "group": None}),
]


def test_watch_owner_filters_errors(sock: Path, capsys: pytest.CaptureFixture[str]) -> None:
    server = EventServer(
        sock,
        stream_of(
            frame("snapshot", {**SNAPSHOT, "owner": "b"}),
            *(frame(name, data) for name, data in OWNER_STREAM),
        ),
    )
    _run(server, "--json", "--owner", "a")
    shown = [(line["event"], line.get("owner")) for line in _json_lines(capsys.readouterr().out)]
    assert shown == [
        ("snapshot", "b"),  # the snapshot always passes
        ("error", "a"),
        ("error", None),
        ("floor", "b"),
        ("state", None),
        ("waiting", "a"),
        ("tag_started", "a"),
        ("paused", None),
        ("resumed", None),
        ("resumed", "a"),
        ("drained", "a"),
        ("drained", None),
        ("disconnected", None),
    ]


def test_watch_without_owner_shows_everything(
    sock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = EventServer(
        sock,
        stream_of(frame("snapshot", SNAPSHOT), *(frame(n, d) for n, d in OWNER_STREAM)),
    )
    _run(server, "--json")
    assert len(_json_lines(capsys.readouterr().out)) == len(OWNER_STREAM) + 2


def test_watch_ignores_unknown_events(sock: Path, capsys: pytest.CaptureFixture[str]) -> None:
    server = EventServer(
        sock,
        stream_of(
            frame("snapshot", SNAPSHOT),
            frame("brand_new", {"owner": "z", "group": None, "x": 1}),
        ),
    )
    _run(server, "--json", "--owner", "a")
    lines = _json_lines(capsys.readouterr().out)
    assert lines[1] == {"event": "brand_new", "owner": "z", "group": None, "x": 1}


def test_watch_human_lines(sock: Path, capsys: pytest.CaptureFixture[str]) -> None:
    server = EventServer(
        sock,
        stream_of(
            frame(
                "snapshot",
                {
                    "state": "WARM",
                    "caps": ["status", "subscribe"],
                    "owner": None,
                    "m": {"a": 1},
                    "version": VERSION,
                },
            ),
            frame("waiting", {"owner": "a", "group": "g", "position": 2}),
            frame("tag_finished", {"owner": "a", "group": None, "tag": "t", "reason": "done"}),
        ),
    )
    assert _run(server) == 2
    assert capsys.readouterr().out.splitlines() == [
        f'snapshot state=WARM caps=status,subscribe m={{"a": 1}} version={VERSION}',
        "waiting owner=a group=g position=2",
        "tag_finished owner=a tag=t reason=done",
        "disconnected",
    ]


def test_watch_human_daemon_off(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    servers: list[EventServer] = []

    def sleep(seconds: float) -> None:
        servers.append(EventServer(sock, stream_of(frame("snapshot", {"version": VERSION}))))

    monkeypatch.setattr(watch, "_sleep", sleep)
    try:
        main(["watch"])
    finally:
        for server in servers:
            server.join()
    assert capsys.readouterr().out.splitlines() == [
        "daemon off; waiting for it",
        f"snapshot version={VERSION}",
        "disconnected",
    ]


def test_watch_broken_stream_is_disconnected(
    sock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = EventServer(sock, stream_of(frame("snapshot", SNAPSHOT), b"garbage\n"))
    assert _run(server, "--json") == 2
    captured = capsys.readouterr()
    assert _json_lines(captured.out)[-1] == {"event": "disconnected"}
    assert "invalid event" in captured.err


def _refuse(req: dict[str, object]) -> list[bytes]:
    """A daemon from before S1: ``subscribe`` is an error reply."""
    error = {"code": "unsupported", "message": "unsupported command 'subscribe'"}
    reply = {"v": 1, "id": req["id"], "ok": False, "error": error}
    return [json.dumps(reply).encode() + b"\n"]


@pytest.mark.parametrize(
    ("args", "out"),
    [(["--json"], '{"event": "disconnected"}\n'), ([], "disconnected\n")],
    ids=["json", "human"],
)
def test_watch_refused_subscribe_exit_2(
    sock: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    out: str,
) -> None:
    """A busy pre-S1 daemon is kept (``ensure`` marks it outdated): nothing to stream."""
    calls = _fake_ensure(monkeypatch, lambda ctx: ctx._outdated.append(True))  # pyright: ignore[reportPrivateUsage]
    server = EventServer(sock, _refuse)
    assert _run(server, *args) == 2
    captured = capsys.readouterr()
    assert calls == [True]
    assert captured.out == out
    assert "unsupported command 'subscribe'" in captured.err


def test_watch_replaces_idle_pre_s1_daemon(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    servers = [EventServer(sock, _refuse)]

    def replace(ctx: client.ClientContext) -> None:
        servers[-1].join()
        sock.unlink()
        servers.append(EventServer(sock, stream_of(frame("snapshot", SNAPSHOT))))

    calls = _fake_ensure(monkeypatch, replace)
    try:
        assert main(["watch", "--json"]) == 2
    finally:
        for server in servers:
            server.join()
    assert calls == [True]
    assert _json_lines(capsys.readouterr().out) == [
        SNAPSHOT_LINE,
        {"event": "disconnected"},
    ]


def test_watch_refused_again_after_replacement_exit_2(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    servers = [EventServer(sock, _refuse)]

    def replace(ctx: client.ClientContext) -> None:
        servers[-1].join()
        sock.unlink()
        servers.append(EventServer(sock, _refuse))

    calls = _fake_ensure(monkeypatch, replace)
    try:
        assert main(["watch", "--json"]) == 2
    finally:
        for server in servers:
            server.join()
    assert calls == [True]
    assert len(servers) == 2
    assert _json_lines(capsys.readouterr().out) == [{"event": "disconnected"}]


def test_watch_broken_snapshot_is_disconnected(
    sock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = EventServer(sock, stream_of(b"garbage\n"))
    assert _run(server, "--json") == 2
    captured = capsys.readouterr()
    assert _json_lines(captured.out) == [{"event": "disconnected"}]
    assert "invalid event" in captured.err


def test_watch_stream_ending_before_the_snapshot_is_disconnected(
    sock: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = EventServer(sock, stream_of())
    assert _run(server, "--json") == 2
    assert _json_lines(capsys.readouterr().out) == [{"event": "disconnected"}]


# ------------------------------------------------------------------ outdated daemon

OLD = {**SNAPSHOT, "version": "0.0.1-old"}


def _fake_ensure(
    monkeypatch: pytest.MonkeyPatch, on_ensure: Callable[[client.ClientContext], None]
) -> list[bool]:
    """Replace ``ClientContext.ensure``; returns the ``work`` flag of every call."""
    calls: list[bool] = []

    def ensure(self: client.ClientContext, *, work: bool = False) -> None:
        calls.append(work)
        on_ensure(self)

    monkeypatch.setattr(client.ClientContext, "ensure", ensure)
    return calls


@pytest.mark.parametrize("start", [False, True], ids=["connected", "start"])
def test_watch_restarts_an_idle_outdated_daemon_and_reconnects(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], start: bool
) -> None:
    """Idle: the old daemon goes and a fresh one starts; nothing of the old one is printed.

    Without ``--start`` the check runs on the snapshot (one stream from the old daemon, then a
    reconnect); with it, ``ensure(work=True)`` runs before the first connection.
    """
    servers: list[EventServer] = []
    if not start:
        servers.append(EventServer(sock, stream_of(frame("snapshot", OLD), frame("state", {}))))

    def replace(ctx: client.ClientContext) -> None:
        if servers:
            servers[-1].join()
            sock.unlink()
        servers.append(EventServer(sock, stream_of(frame("snapshot", SNAPSHOT))))

    calls = _fake_ensure(monkeypatch, replace)
    try:
        assert main(["watch", "--json", *(["--start"] if start else [])]) == 2
    finally:
        for server in servers:
            server.join()
    assert calls == [True]
    assert len(servers) == (1 if start else 2)
    assert _json_lines(capsys.readouterr().out) == [
        SNAPSHOT_LINE,
        {"event": "disconnected"},
    ]


def test_watch_marks_a_busy_outdated_daemon(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def keep(ctx: client.ClientContext) -> None:  # busy: ensure keeps it and says so
        ctx._outdated.append(True)  # pyright: ignore[reportPrivateUsage]

    calls = _fake_ensure(monkeypatch, keep)
    server = EventServer(
        sock, stream_of(frame("snapshot", OLD), frame("floor", {"owner": "a", "group": "g1"}))
    )
    assert _run(server, "--json") == 2
    assert calls == [True]
    assert _json_lines(capsys.readouterr().out) == [
        {"event": "snapshot", "protocol": 1, **OLD, "daemon_outdated": True},
        {"event": "floor", "owner": "a", "group": "g1"},
        {"event": "disconnected"},
    ]


def test_watch_start_with_a_busy_outdated_daemon_checks_once(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_ensure(monkeypatch, lambda ctx: ctx._outdated.append(True))  # pyright: ignore[reportPrivateUsage]
    server = EventServer(sock, stream_of(frame("snapshot", OLD)))
    assert _run(server, "--json", "--start") == 2
    assert calls == [True]
    assert _json_lines(capsys.readouterr().out)[0] == {
        "event": "snapshot",
        "protocol": 1,
        **OLD,
        "daemon_outdated": True,
    }


def test_watch_current_daemon_is_not_checked(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_ensure(monkeypatch, lambda ctx: None)
    server = EventServer(sock, stream_of(frame("snapshot", SNAPSHOT)))
    assert _run(server, "--json") == 2
    assert calls == []
    assert "daemon_outdated" not in _json_lines(capsys.readouterr().out)[0]


def test_watch_autostart_off_keeps_an_outdated_daemon(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Replacing would start a fresh daemon: with ``autostart = false`` nothing is shut down."""
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "false")
    calls = _fake_ensure(monkeypatch, lambda ctx: pytest.fail("must not replace"))
    server = EventServer(sock, stream_of(frame("snapshot", OLD)))
    assert _run(server, "--json") == 2
    assert calls == []
    assert _json_lines(capsys.readouterr().out) == [
        {"event": "snapshot", "protocol": 1, **OLD, "daemon_outdated": True},
        {"event": "disconnected"},
    ]


def test_watch_start_with_autostart_off_does_not_replace(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--start`` still starts a missing daemon, but an outdated one is only marked."""
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "false")
    calls = _fake_ensure(monkeypatch, lambda ctx: None)
    server = EventServer(sock, stream_of(frame("snapshot", OLD)))
    assert _run(server, "--json", "--start") == 2
    assert calls == [False]
    assert _json_lines(capsys.readouterr().out)[0] == {
        "event": "snapshot",
        "protocol": 1,
        **OLD,
        "daemon_outdated": True,
    }


def test_watch_autostart_off_with_a_pre_s1_daemon_exit_2(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "false")
    calls = _fake_ensure(monkeypatch, lambda ctx: pytest.fail("must not replace"))
    server = EventServer(sock, _refuse)
    assert _run(server, "--json") == 2
    assert calls == []
    assert _json_lines(capsys.readouterr().out) == [{"event": "disconnected"}]


@pytest.mark.parametrize("case", ["outdated", "pre-s1", "start"])
def test_watch_failed_ensure_is_disconnected(
    sock: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    def fail(ctx: client.ClientContext) -> None:
        raise ServiceUnavailable("voxlektor did not start in time")

    _fake_ensure(monkeypatch, fail)
    if case == "start":
        assert main(["watch", "--json", "--start"]) == 2
    else:
        lines = stream_of(frame("snapshot", OLD)) if case == "outdated" else _refuse
        assert _run(EventServer(sock, lines), "--json") == 2
    captured = capsys.readouterr()
    assert _json_lines(captured.out) == [{"event": "disconnected"}]
    assert "did not start in time" in captured.err


# ------------------------------------------------------------------ stdout


class _WriteLog(io.StringIO):
    """Records every ``write`` call."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    def write(self, s: str) -> int:
        self.writes.append(s)
        return super().write(s)


@pytest.mark.parametrize("args", [["--json"], []], ids=["json", "human"])
def test_watch_writes_each_line_whole(
    sock: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    log = _WriteLog()
    monkeypatch.setattr("sys.stdout", log)
    server = EventServer(
        sock, stream_of(frame("snapshot", SNAPSHOT), frame("state", {"state": "UNLOADING"}))
    )
    _run(server, *args)
    assert len(log.writes) == 3
    assert all(s.endswith("\n") and s.count("\n") == 1 for s in log.writes)


class _ClosedPipe(io.StringIO):
    """A stdout whose reader went away; ``fileno`` is a real descriptor when ``fd`` is given."""

    def __init__(self, fd: int | None) -> None:
        super().__init__()
        self._fd = fd

    def write(self, s: str) -> int:
        raise BrokenPipeError

    def fileno(self) -> int:
        if self._fd is None:
            raise io.UnsupportedOperation("fileno")
        return self._fd


@pytest.mark.parametrize("real_fd", [False, True], ids=["no-fd", "fd"])
def test_watch_broken_pipe_exit_0(
    sock: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    real_fd: bool,
) -> None:
    target = tmp_path / "out"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT)
    try:
        monkeypatch.setattr("sys.stdout", _ClosedPipe(fd if real_fd else None))
        server = EventServer(sock, stream_of(frame("snapshot", SNAPSHOT)))
        assert _run(server, "--json") == 0
        os.write(fd, b"late")  # the exit-time flush: goes to /dev/null once silenced
    finally:
        os.close(fd)
    assert capsys.readouterr().err == ""
    assert target.read_bytes() == (b"" if real_fd else b"late")


# ------------------------------------------------------------------ --owner validation


@pytest.mark.parametrize(
    ("owner", "message"),
    [("cli", "owner 'cli' is reserved"), ("a b", "invalid owner 'a b'"), ("x" * 65, "invalid")],
    ids=["reserved", "space", "too-long"],
)
def test_watch_rejects_a_bad_owner(
    sock: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    owner: str,
    message: str,
) -> None:
    def no_stream(path: Path, timeout: float) -> object:
        pytest.fail("must not connect")

    monkeypatch.setattr(watch, "open_stream", no_stream)
    assert main(["watch", "--json", "--owner", owner]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
