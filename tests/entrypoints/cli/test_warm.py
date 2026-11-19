"""``vxl warm``: start the daemon if needed and ask it to load the engine, without waiting."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import voxlektor
from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.app.ports import PlatformInfo
from voxlektor.bootstrap import client
from voxlektor.domain.errors import ServiceUnavailable
from voxlektor.entrypoints.cli.app import main


def test_warm_starts_daemon(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["warm"]) == 0
    assert wire.ensured == 1
    assert wire.requests == [{"cmd": "load"}]
    assert capsys.readouterr().out == ""


def test_warm_with_a_profile_loads_the_engine(wire: FakeWire, home: Path) -> None:
    config = home / "config" / "config.toml"
    config.write_text(config.read_text() + "[profiles.claude]\nrate = 1.25\n")
    assert main(["warm", "--profile", "claude"]) == 0
    assert wire.requests == [{"cmd": "load"}]


def test_warm_with_an_unknown_profile_exits_1(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["warm", "--profile", "nobody"]) == 1
    assert "unknown profile 'nobody'" in capsys.readouterr().err
    assert wire.requests == []
    assert wire.ensured == 0


def test_warm_restarts_an_outdated_idle_daemon(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses: list[dict[str, object]] = [
        {"ok": True, "data": {"version": "0.0.1", "queued": 0}},
        {"ok": True, "data": {"version": voxlektor.__version__}},
    ]
    pings: list[bool] = []

    def ensure(platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        wire.replies["status"] = statuses.pop(0)
        pings.append(ping())

    monkeypatch.setattr(client, "ensure_daemon", ensure)
    assert main(["warm"]) == 0
    assert pings == [True, True]
    assert wire.requests == [
        {"cmd": "status"},
        {"cmd": "shutdown", "if_idle": True},
        {"cmd": "status"},
        {"cmd": "load"},
    ]


def test_warm_before_setup_exits_1(
    fresh_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeWire()
    monkeypatch.setattr(client, "request", fake.request)
    monkeypatch.setattr(client, "ensure_daemon", fake.ensure)
    assert main(["warm"]) == 1
    assert "not set up" in capsys.readouterr().err
    assert fake.requests == []
    assert fake.ensured == 0


def test_warm_with_autostart_off_starts_nothing(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "false")
    assert main(["warm"]) == 0
    assert wire.requests == []
    assert wire.ensured == 0
    assert capsys.readouterr().out == ""


def test_warm_on_a_busy_outdated_daemon_without_load_succeeds(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An older daemon that is reading for another session: its engine is warm already.
    wire.replies["status"] = {"ok": True, "data": {"version": "0.0.1", "playing": True}}
    wire.fail("load", "unknown_command", "unknown command 'load'")

    def ensure(platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        assert ping()

    monkeypatch.setattr(client, "ensure_daemon", ensure)
    assert main(["warm"]) == 0
    assert wire.cmds == ["status", "load"]
    assert capsys.readouterr().err == ""


def test_warm_without_load_on_a_current_daemon_fails(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    wire.fail("load", "unknown_command", "unknown command 'load'")
    assert main(["warm"]) == 2  # the daemon cannot serve this client, as for every command
    assert "unknown command 'load'" in capsys.readouterr().err


def test_warm_daemon_that_does_not_start_exits_2(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def ensure(platform: PlatformInfo, ping: Callable[[], bool]) -> None:
        raise ServiceUnavailable("voxlektor did not start within 15 s")

    monkeypatch.setattr(client, "ensure_daemon", ensure)
    assert main(["warm"]) == 2
    assert "did not start" in capsys.readouterr().err
    assert json.dumps(wire.cmds) == "[]"
