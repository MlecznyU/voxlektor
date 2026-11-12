"""``vxl config voice NAME`` and ``vxl config show``."""

import tomllib
from pathlib import Path

import pytest

from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.entrypoints.cli.app import main


def _config(home: Path) -> Path:
    return home / "config" / "config.toml"


def test_config_voice_persists(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "voice", "f2-narrator"]) == 0
    assert capsys.readouterr().out.strip() == "voice = f2-narrator"
    data = tomllib.loads(_config(home).read_text())
    assert data["voice"] == "f2-narrator"
    assert data["config_version"] == 1
    assert main(["hello"]) == 0
    assert wire.requests[-1]["voice"] == "f2-narrator"
    assert wire.requests[-1]["cmd"] == "speak"


def test_config_voice_unknown_lists_names_and_writes_nothing(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "voice", "nobody"]) == 1
    err = capsys.readouterr().err
    assert "unknown voice 'nobody'" in err
    assert "available: f1-warm, f2-narrator" in err
    assert "v0-default" in err
    assert not _config(home).exists()


def test_config_show_prints_the_effective_settings(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(home).parent.mkdir(parents=True, exist_ok=True)
    _config(home).write_text('voice = "m1-calm"\n[service]\nidle_timeout = "90s"\n')
    monkeypatch.setenv("VOXLEKTOR_BASE_LANG", "en")
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == f"# effective settings; file: {_config(home)}"
    assert lines[1:] == [
        'engine = "omnivoice"',
        'voice = "m1-calm"',
        'lang = "auto"',
        'base_lang = "en"',
        "rate = 1.0",
        "history = 20",
        "",
        "[service]",
        'idle_timeout = "90s"',
        "autostart = true",
    ]
    assert wire.requests == []
    # The output is a config file the settings store accepts as is.
    _config(home).write_text(out)
    monkeypatch.delenv("VOXLEKTOR_BASE_LANG")
    assert main(["config", "show"]) == 0
    assert capsys.readouterr().out == out


@pytest.mark.parametrize(
    ("seconds", "shown"), [("0", '"0"'), ("3600", '"1h"'), ("600", '"10m"'), ("90", '"90s"')]
)
def test_config_show_idle_timeout_in_a_form_the_file_accepts(
    wire: FakeWire,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    seconds: str,
    shown: str,
) -> None:
    monkeypatch.setenv("VOXLEKTOR_IDLE_TIMEOUT", seconds)
    assert main(["config", "show"]) == 0
    assert f"idle_timeout = {shown}" in capsys.readouterr().out.splitlines()


def test_config_without_subcommand_prints_help(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "voice" in out
    assert "show" in out


def test_config_show_reports_a_broken_file(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(home).parent.mkdir(parents=True, exist_ok=True)
    _config(home).write_text("rate = [")
    assert main(["config", "show"]) == 1
    assert str(_config(home)) in capsys.readouterr().err
