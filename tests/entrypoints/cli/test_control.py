"""Control commands: playback, idle, unload, stop, status and serve."""

import json
from pathlib import Path

import pytest

from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.domain.errors import NotRunning, ServiceUnavailable
from voxlektor.domain.settings import Settings
from voxlektor.entrypoints import daemon
from voxlektor.entrypoints.cli import control
from voxlektor.entrypoints.cli.app import main
from voxlektor.testing.fakes import FakeClock

STATUS = {
    "state": "WARM",
    "engine": "fake",
    "voice": "v0-default",
    "idle_timeout_s": 600,
    "idle_remaining_s": 590.5,
    "queued": 0,
    "playing": False,
    "paused": False,
    "underruns": 0,
    "first_audio_s": 0.42,
    "chunks": 3,
    "gen_seconds": 1.5,
    "audio_seconds": 9.0,
    "last_error": None,
    "caps": ["status"],
}


@pytest.mark.parametrize(
    ("command", "wire_cmd"),
    [
        ("pause", "pause"),
        ("resume", "resume"),
        ("skip", "skip"),
        ("back", "back"),
        ("stop-reading", "stop_reading"),
        ("unload", "unload"),
    ],
)
def test_plain_commands_send_their_wire_command(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], command: str, wire_cmd: str
) -> None:
    assert main([command]) == 0
    assert wire.requests == [{"cmd": wire_cmd}]
    assert wire.ensured == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("given", "seconds", "shown"),
    [("10m", 600, "10m"), ("90s", 90, "1m30s"), ("0", 0, "0 (never unload)")],
)
def test_idle_sets_the_timeout_and_shows_it(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], given: str, seconds: int, shown: str
) -> None:
    wire.replies["idle"] = {"ok": True, "data": {"idle_timeout_s": seconds}}
    assert main(["idle", given]) == 0
    assert wire.requests == [{"cmd": "idle", "timeout": given}]
    assert capsys.readouterr().out.strip() == f"idle timeout: {shown}"


def test_idle_reply_without_a_timeout_prints_nothing(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["idle", "10m"]) == 0
    assert capsys.readouterr().out == ""


def test_invalid_idle_duration_is_a_user_error(wire: FakeWire) -> None:
    wire.fail("idle", "invalid_duration", "invalid duration '5x'; use 0, 90s, 10m or 1h")
    assert main(["idle", "5x"]) == 1


def test_control_without_daemon_exits_2(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    wire.replies["pause"] = ServiceUnavailable()
    assert main(["pause"]) == 2
    assert capsys.readouterr().err.strip() == "voxlektor is not running"


def test_stop_when_not_running(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    wire.replies["shutdown"] = NotRunning()
    assert main(["stop"]) == 0
    assert capsys.readouterr().out.strip() == "not running"


@pytest.mark.parametrize("command", ["stop", "status"])
def test_unresponsive_daemon_is_never_reported_as_off(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    """L1: a daemon that is there but does not answer is unavailable (2), not off."""
    unresponsive = ServiceUnavailable("voxlektor did not answer within 5 s")
    wire.replies["shutdown"] = unresponsive
    wire.replies["status"] = unresponsive
    assert main([command]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "did not answer" in captured.err


def test_stop_sends_shutdown(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stop"]) == 0
    assert wire.cmds == ["shutdown"]
    assert capsys.readouterr().out.strip() == "stopped"


def test_stop_waits_until_the_socket_is_gone(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sock = home / "run" / "voxlektor.sock"
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.write_text("")
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            sock.unlink()  # the daemon removes its socket once it stops serving

    monkeypatch.setattr(control, "_sleep", sleep)
    assert main(["stop"]) == 0
    assert sleeps == [0.1, 0.1, 0.1]
    assert capsys.readouterr().out.strip() == "stopped"


def test_stop_gives_up_waiting_after_a_while(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sock = home / "run" / "voxlektor.sock"
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.write_text("")
    clock = FakeClock()
    monkeypatch.setattr(control, "_monotonic", clock.now)
    monkeypatch.setattr(control, "_sleep", clock.advance)
    assert main(["stop"]) == 0
    assert capsys.readouterr().out.strip() == "stopping"


def test_status_when_not_running_is_off(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    wire.replies["status"] = NotRunning()
    assert main(["status"]) == 0
    assert capsys.readouterr().out.strip() == "OFF"


def test_status_lines(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    wire.replies["status"] = {"ok": True, "data": STATUS}
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "state: WARM"
    assert "engine: fake" in out
    assert "voice: v0-default" in out
    assert "idle: unloads in 9m50s" in out
    assert "RAM" not in out


def test_status_error_reply_is_reported(wire: FakeWire) -> None:
    wire.fail("status", "internal", "the service is shutting down")
    assert main(["status"]) == 2


def test_status_json_when_not_running(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    wire.replies["status"] = NotRunning()
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"running": False}


def test_status_json_is_the_daemon_status_plus_running(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    wire.replies["status"] = {"ok": True, "data": STATUS}
    assert main(["status", "--json"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1  # one line, for tools
    assert json.loads(out) == {"running": True, **STATUS}


def test_status_json_of_an_unresponsive_daemon_exits_2_without_json(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    wire.fail("status", "internal", "the service is shutting down")
    assert main(["status", "--json"]) == 2
    assert capsys.readouterr().out == ""


def test_serve_passes_the_idle_override(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Settings] = []

    def fake_serve(settings: Settings) -> int:
        seen.append(settings)
        return 0

    monkeypatch.setattr(daemon, "serve", fake_serve)
    assert main(["serve", "--idle", "90s"]) == 0
    assert main(["serve"]) == 0
    assert [s.idle_timeout_s for s in seen] == [90, 600]
    assert wire.requests == []


def test_serve_returns_the_daemon_exit_code(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch
) -> None:
    def already_running(settings: Settings) -> int:
        return 1

    monkeypatch.setattr(daemon, "serve", already_running)
    assert main(["serve"]) == 1


def test_serve_with_an_invalid_idle_exits_1(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["serve", "--idle", "5x"]) == 1
    assert "invalid duration" in capsys.readouterr().err
