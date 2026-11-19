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
    assert data["profiles"] == {"default": {"voice": "f2-narrator"}}
    assert data["config_version"] == 2
    assert main(["hello"]) == 0
    assert wire.requests[-1]["voice"] == "f2-narrator"
    assert wire.requests[-1]["cmd"] == "speak"


def test_config_voice_unknown_lists_names_and_writes_nothing(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _config(home).read_text()  # the completed-setup config of the `home` fixture
    assert main(["config", "voice", "nobody"]) == 1
    err = capsys.readouterr().err
    assert "unknown voice 'nobody'" in err
    assert "available: f1-warm, f2-narrator" in err
    assert "v0-default" in err
    assert _config(home).read_text() == before


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
        "volume = 50",
        "steps = 12",
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


def test_config_show_of_a_profile_overlays_default(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(home).write_text(
        "config_version = 2\n[profiles.default]\nrate = 1.5\nvolume = 30\n"
        '[profiles.claude]\nrate = 1.25\nvoice = "m1-calm"\n'
    )
    monkeypatch.setenv("VOXLEKTOR_STEPS", "8")
    assert main(["config", "show", "--profile", "claude"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"# effective settings of profile claude; file: {_config(home)}"
    assert 'voice = "m1-calm"' in lines
    assert "rate = 1.25" in lines
    assert "volume = 30" in lines
    assert "steps = 8" in lines
    assert main(["config", "show"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"# effective settings; file: {_config(home)}"
    assert "rate = 1.5" in lines


def test_config_show_of_an_unknown_profile_exits_1(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "show", "--profile", "nobody"]) == 1
    assert "unknown profile 'nobody'" in capsys.readouterr().err


def test_config_show_includes_volume(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["config", "show"]) == 0
    assert "volume = 50" in capsys.readouterr().out.splitlines()
    monkeypatch.setenv("VOXLEKTOR_VOLUME", "80")
    assert main(["config", "show"]) == 0
    assert "volume = 80" in capsys.readouterr().out.splitlines()


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


def test_config_show_skips_invalid_values_with_a_warning(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(home).write_text(
        "config_version = 2\nhistory = 0\n[profiles.default]\nrate = 9.0\nvolume = 30\n"
        "[profiles.claude]\nvolume = 500\n"
    )
    monkeypatch.setenv("VOXLEKTOR_RATE", "fast")
    assert main(["config", "show", "--profile", "claude"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert (lines[5], lines[6], lines[8]) == ("rate = 1.0", "volume = 30", "history = 20")
    assert captured.err.splitlines() == [
        "ignoring invalid rate in config.toml: rate must be between 0.5 and 3.0",
        "ignoring invalid history in config.toml: history must be a positive integer",
        "ignoring invalid volume in profile 'claude' of config.toml: "
        "volume must be an integer between 0 and 100",
        "ignoring invalid VOXLEKTOR_RATE in the environment: "
        "VOXLEKTOR_RATE: invalid value 'fast' for rate",
    ]


def test_config_show_reports_a_broken_file(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(home).parent.mkdir(parents=True, exist_ok=True)
    _config(home).write_text("rate = [")
    assert main(["config", "show"]) == 1
    assert str(_config(home)) in capsys.readouterr().err
