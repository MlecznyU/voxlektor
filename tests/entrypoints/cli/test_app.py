"""Dispatch of ``vxl`` arguments and the mapping of failures to exit codes."""

import io
import sys
from pathlib import Path

import pytest

from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.adapters.transport import unix_client
from voxlektor.app.ports import PlatformInfo
from voxlektor.bootstrap import client
from voxlektor.domain.errors import ServiceUnavailable
from voxlektor.entrypoints.cli.app import main


def _speak(wire: FakeWire) -> dict[str, object]:
    (req,) = wire.requests
    assert req["cmd"] == "speak"
    return req


def test_a_command_name_runs_that_command(wire: FakeWire) -> None:
    """Review Focus 4: text equal to a subcommand is the subcommand."""
    assert main(["pause"]) == 0
    assert wire.cmds == ["pause"]
    assert wire.ensured == 0


@pytest.mark.parametrize("argv", [["say", "pause"], ["--", "pause"]], ids=["say", "dashdash"])
def test_say_or_double_dash_reads_a_command_name_aloud(wire: FakeWire, argv: list[str]) -> None:
    assert main(argv) == 0
    assert _speak(wire)["text"] == "pause"
    assert wire.ensured == 1


def test_bare_words_are_text_for_say(wire: FakeWire) -> None:
    assert main(["hello", "world"]) == 0
    assert _speak(wire)["text"] == "hello world"


def test_speak_flags_go_into_the_request(wire: FakeWire) -> None:
    assert main(["-v", "m1-calm", "-r", "1.25", "hej"]) == 0
    req = _speak(wire)
    assert req["text"] == "hej"
    assert req["voice"] == "m1-calm"
    assert req["rate"] == 1.25
    assert req["lang"] == "auto"
    assert req["now"] is False


def test_long_flags_lang_and_now(wire: FakeWire) -> None:
    assert main(["say", "--now", "--lang", "en", "--voice", "f2-narrator", "hi"]) == 0
    req = _speak(wire)
    assert req["now"] is True
    assert req["lang"] == "en"
    assert req["voice"] == "f2-narrator"


def test_effective_settings_fill_what_flags_leave_out(
    wire: FakeWire, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXLEKTOR_VOICE", "f1-warm")
    monkeypatch.setenv("VOXLEKTOR_RATE", "1.5")
    assert main(["-r", "1.25", "hi"]) == 0
    req = _speak(wire)
    assert req["voice"] == "f1-warm"
    assert req["rate"] == 1.25


@pytest.mark.parametrize("argv", [[""], ["   "], ["say", "\n"]], ids=repr)
def test_empty_text_is_nothing_to_read_without_socket(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    """Review Focus 1: ``vxl ""`` never touches the socket."""
    assert main(argv) == 0
    assert capsys.readouterr().out.strip() == "nothing to read"
    assert wire.requests == []
    assert wire.ensured == 0


def test_piped_newline_is_nothing_to_read(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review Focus 1: ``echo | vxl``."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == "nothing to read"
    assert wire.requests == []


def test_piped_text_is_read(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Hello from a pipe.\n"))
    assert main([]) == 0
    assert _speak(wire)["text"] == "Hello from a pipe.\n"


def test_piped_text_with_flags_only(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Piped."))
    assert main(["-v", "m1-calm"]) == 0
    req = _speak(wire)
    assert req["text"] == "Piped."
    assert req["voice"] == "m1-calm"


@pytest.mark.parametrize("argv", [[], ["--help"]], ids=["no-args-tty", "help"])
def test_help(wire: FakeWire, capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Usage" in out
    assert "say" in out
    assert "stop-reading" in out
    assert wire.requests == []


def test_argv_defaults_to_sys_argv(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vxl", "skip"])
    assert main() == 0
    assert wire.cmds == ["skip"]


@pytest.mark.parametrize(
    "argv",
    [["-r", "fast", "hi"], ["idle"], ["config", "nope"], ["pause", "--bogus"]],
    ids=["bad-float", "missing-argument", "unknown-subcommand", "unknown-command-option"],
)
def test_usage_errors_exit_1(wire: FakeWire, argv: list[str]) -> None:
    assert main(argv) == 1
    assert wire.requests == []


@pytest.mark.parametrize(
    "argv",
    [["-5", "degrees"], ["--bogus", "hi"], ["say", "hi", "-x"], ["--voices=x", "hi"]],
    ids=repr,
)
def test_text_starting_with_a_dash_suggests_double_dash(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert f"unknown option {argv[0] if argv[0] != 'say' else argv[2]!r}" in err
    assert "vxl -- <text>" in err
    assert wire.requests == []


def test_double_dash_reads_dash_words(wire: FakeWire) -> None:
    assert main(["--", "-5", "degrees"]) == 0
    assert _speak(wire)["text"] == "-5 degrees"


def test_every_say_option_form_is_accepted(wire: FakeWire, home: Path) -> None:
    notes = home / "n.txt"
    notes.write_text("from a file")
    assert main(["--rate=1.5", "--voice", "m1-calm", "--lang=en", "--now", "-", "x"]) == 0
    req = _speak(wire)
    assert (req["rate"], req["voice"], req["lang"], req["now"], req["text"]) == (
        1.5,
        "m1-calm",
        "en",
        True,
        "- x",
    )
    assert main(["-f", str(notes), "-l", "pl", "-r1.25", "-vm1-calm"]) == 0
    last = wire.requests[-1]
    assert (last["text"], last["lang"], last["rate"], last["voice"]) == (
        "from a file",
        "pl",
        1.25,
        "m1-calm",
    )


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["-r", "9", "hi"], "rate must be between"),
        (["-l", "de", "hi"], "lang must be one of"),
    ],
    ids=["rate", "lang"],
)
def test_invalid_settings_exit_1_before_any_request(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    assert main(argv) == 1
    assert message in capsys.readouterr().err
    assert wire.requests == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("invalid_argument", 1),
        ("unknown_voice", 1),
        ("unknown_engine", 1),
        ("invalid_duration", 1),
        ("invalid_setting", 1),
        ("bad_frame", 1),
        ("unsupported", 1),
        ("bad_version", 2),
        ("unknown_command", 2),
        ("internal", 2),
        ("engine_failed", 2),
        ("audio_failed", 2),
    ],
)
def test_error_replies_map_to_exit_codes(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], code: str, expected: int
) -> None:
    wire.fail("skip", code, f"{code} happened")
    assert main(["skip"]) == expected
    assert f"{code} happened" in capsys.readouterr().err


def test_unknown_voice_reply_lists_the_voices(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    wire.fail("speak", "unknown_voice", "unknown voice 'nobody'; available: m1-calm, v0-default")
    assert main(["-v", "nobody", "hi"]) == 1
    assert "available: m1-calm, v0-default" in capsys.readouterr().err


def test_service_unavailable_exits_2(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    wire.replies["back"] = ServiceUnavailable()
    assert main(["back"]) == 2
    assert "not running" in capsys.readouterr().err


def test_oversized_text_exits_1(wire: FakeWire, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "request", unix_client.request)
    assert main(["say", "a " * (1 << 20)]) == 1


def test_end_of_input_aborts_with_1(wire: FakeWire, capsys: pytest.CaptureFixture[str]) -> None:
    wire.replies["pause"] = EOFError()
    assert main(["pause"]) == 1
    assert "aborted" in capsys.readouterr().err


def test_ctrl_c_exits_130_like_any_interrupted_command(wire: FakeWire) -> None:
    wire.replies["pause"] = KeyboardInterrupt()
    assert main(["pause"]) == 130


def test_unsupported_platform_exits_3(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(*args: str) -> PlatformInfo:
        return UnsupportedPlatform()

    monkeypatch.setattr(client, "select_platform", unsupported)
    assert main(["hello"]) == 3
    assert "platform not supported yet" in capsys.readouterr().err
