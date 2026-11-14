"""The speak path: where the text comes from, autostart, and one-shot mode."""

import io
import os
import sys
from pathlib import Path

import pytest

from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.contracts.protocol import MAX_FRAME
from voxlektor.domain.settings import Settings
from voxlektor.entrypoints import oneshot
from voxlektor.entrypoints.cli import speak
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


@pytest.fixture
def no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the speak path builds a client (or autostarts a daemon)."""

    def forbidden(flags: object) -> None:
        raise AssertionError("build_client must not run for rejected text")

    monkeypatch.setattr(speak, "build_client", forbidden)


@pytest.mark.usefixtures("no_client")
def test_too_long_file_exits_1_before_the_client(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = home / "long.txt"
    notes.write_bytes(b"a" * (MAX_FRAME + 1))
    assert main(["-f", str(notes)]) == 1
    assert "text too long" in capsys.readouterr().err
    assert wire.requests == []
    assert wire.ensured == 0


def test_file_of_exactly_max_frame_bytes_is_not_too_long(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = home / "edge.txt"
    notes.write_bytes(b"a" * MAX_FRAME)
    main(["-f", str(notes)])
    assert "text too long" not in capsys.readouterr().err


@pytest.mark.usefixtures("no_client")
def test_fifo_is_refused(wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fifo = home / "pipe"
    os.mkfifo(fifo)
    assert main(["-f", str(fifo)]) == 1
    assert "not a regular file" in capsys.readouterr().err
    assert wire.requests == []


@pytest.mark.usefixtures("no_client")
def test_directory_is_refused(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["-f", str(home)]) == 1
    assert "not a regular file" in capsys.readouterr().err


@pytest.mark.usefixtures("no_client")
def test_too_long_stdin_exits_1_before_the_client(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("a" * (MAX_FRAME + 1)))
    assert main([]) == 1
    assert "text too long" in capsys.readouterr().err
    assert wire.ensured == 0


@pytest.mark.usefixtures("no_client")
def test_too_long_binary_stdin_is_read_bounded(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = io.BytesIO(b"a" * (3 * MAX_FRAME))
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(raw, encoding="utf-8"))
    assert main([]) == 1
    assert "text too long" in capsys.readouterr().err
    assert raw.tell() <= MAX_FRAME + 1 + io.DEFAULT_BUFFER_SIZE  # never slurped whole


def test_binary_stdin_is_decoded(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO("Zażółć gęślą jaźń.".encode())
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(raw, encoding="utf-8"))
    assert main([]) == 0
    ((req),) = wire.requests
    assert req["text"] == "Zażółć gęślą jaźń."


@pytest.mark.usefixtures("no_client")
def test_undecodable_stdin_exits_1(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"\xff\xfe"), encoding="utf-8"))
    assert main([]) == 1
    assert "cannot read stdin" in capsys.readouterr().err


@pytest.mark.usefixtures("no_client")
def test_too_long_words_exit_1_before_the_client(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["say", "a" * (MAX_FRAME + 1)]) == 1
    assert "text too long" in capsys.readouterr().err


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


@pytest.mark.parametrize(
    "argv", [["say", "--volume", "20", "hi"], ["--volume", "20", "hi"], ["--volume=20", "hi"]]
)
def test_volume_flag_goes_into_the_request(wire: FakeWire, argv: list[str]) -> None:
    assert main(argv) == 0
    ((req),) = wire.requests
    assert req["volume"] == 20


def test_every_request_carries_the_effective_volume(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["hi"]) == 0
    assert wire.requests[-1]["volume"] == 50
    monkeypatch.setenv("VOXLEKTOR_VOLUME", "70")
    assert main(["hi"]) == 0
    assert wire.requests[-1]["volume"] == 70
    assert main(["--volume", "10", "hi"]) == 0
    assert wire.requests[-1]["volume"] == 10


@pytest.mark.parametrize("value", ["200", "-1", "loud"])
def test_invalid_volume_exits_1_before_any_request(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    assert main([f"--volume={value}", "hi"]) == 1
    assert wire.requests == []
    if value != "loud":
        assert "volume must be an integer between 0 and 100" in capsys.readouterr().err


def test_volume_reaches_one_shot_mode(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "false")
    seen: list[Settings] = []

    def fake_run(text: str, settings: Settings) -> int:
        seen.append(settings)
        return 0

    monkeypatch.setattr(oneshot, "run", fake_run)
    assert main(["--volume", "80", "hello"]) == 0
    assert [s.volume for s in seen] == [80]
