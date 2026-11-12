"""The speak path: where the text comes from, autostart, and one-shot mode."""

from pathlib import Path

import pytest

from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.domain.settings import Settings
from voxlektor.entrypoints import oneshot
from voxlektor.entrypoints.cli.app import main


def test_file_text_is_read(wire: FakeWire, home: Path) -> None:
    notes = home / "notes.txt"
    notes.write_text("First line.\nSecond line.\n", encoding="utf-8")
    assert main(["-f", str(notes)]) == 0
    ((req),) = wire.requests
    assert req["text"] == "First line.\nSecond line.\n"


def test_empty_file_is_nothing_to_read(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = home / "empty.txt"
    notes.write_text("\n\n")
    assert main(["say", "-f", str(notes)]) == 0
    assert capsys.readouterr().out.strip() == "nothing to read"
    assert wire.requests == []


def test_missing_file_exits_1(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-f", str(home / "nope.txt")]) == 1
    assert "cannot read" in capsys.readouterr().err
    assert wire.requests == []


def test_undecodable_file_exits_1(wire: FakeWire, home: Path) -> None:
    notes = home / "bad.txt"
    notes.write_bytes(b"\xff\xfe\xfa")
    assert main(["-f", str(notes)]) == 1
    assert wire.requests == []


def test_file_and_text_together_exit_1(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = home / "notes.txt"
    notes.write_text("x")
    assert main(["-f", str(notes), "hello"]) == 1
    assert "not both" in capsys.readouterr().err
    assert wire.requests == []


def test_say_without_text_on_a_terminal_exits_1(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["say"]) == 1
    assert "no text" in capsys.readouterr().err
    assert wire.requests == []


def test_autostart_off_reads_in_this_process(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "false")
    seen: list[tuple[str, Settings]] = []

    def fake_run(text: str, settings: Settings) -> int:
        seen.append((text, settings))
        return 0

    monkeypatch.setattr(oneshot, "run", fake_run)
    assert main(["-v", "m1-calm", "-r", "1.5", "hello", "there"]) == 0
    ((text, settings),) = seen
    assert text == "hello there"
    assert settings.voice == "m1-calm"
    assert settings.rate == 1.5
    assert settings.autostart is False
    assert wire.requests == []
    assert wire.ensured == 0


def test_one_shot_exit_code_is_passed_on(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "0")

    def failing_run(text: str, settings: Settings) -> int:
        return 2

    monkeypatch.setattr(oneshot, "run", failing_run)
    assert main(["hello"]) == 2
