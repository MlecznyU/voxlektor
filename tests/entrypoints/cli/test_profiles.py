"""``vxl profiles [--json]`` and ``vxl profiles rm NAME``."""

import json
import tomllib
from pathlib import Path

import pytest

from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.entrypoints.cli.app import main

_TOML = (
    "config_version = 2\n"
    "setup_completed = true\n"
    'engine = "omnivoice"\n'
    "[profiles.default]\n"
    'voice = "f2-narrator"\n'
    "rate = 1.5\n"
    "[profiles.zz-audiobook]\n"
    "steps = 24\n"
    "[profiles.claude]\n"
    "rate = 1.25\n"
    "volume = 70\n"
)


def _config(home: Path) -> Path:
    path = home / "config" / "config.toml"
    path.write_text(_TOML)
    return path


def test_profiles_json_lists_every_profile_with_values_and_sources(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(home)
    monkeypatch.setenv("VOXLEKTOR_LANG", "en")
    assert main(["profiles", "--json"]) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1
    data = json.loads(out)
    assert [p["name"] for p in data["profiles"]] == ["default", "claude", "zz-audiobook"]
    default, claude, _ = data["profiles"]
    assert default["values"] == {
        "voice": "f2-narrator",
        "lang": "en",
        "base_lang": "pl",
        "rate": 1.5,
        "volume": 50,
        "steps": 12,
    }
    assert default["sources"] == {
        "voice": "default",
        "lang": "env",
        "base_lang": "builtin",
        "rate": "default",
        "volume": "builtin",
        "steps": "builtin",
    }
    assert claude["values"]["rate"] == 1.25
    assert claude["values"]["voice"] == "f2-narrator"
    assert claude["sources"]["rate"] == "profile"
    assert claude["sources"]["volume"] == "profile"
    assert claude["sources"]["voice"] == "default"
    assert wire.requests == []


def test_profiles_without_a_file_lists_the_implicit_default(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profiles", "--json"]) == 0
    (only,) = json.loads(capsys.readouterr().out)["profiles"]
    assert only["name"] == "default"
    assert set(only["sources"].values()) == {"builtin"}


def test_profiles_prints_one_block_per_profile(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(home)
    assert main(["profiles"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:7] == [
        "default",
        "  voice = f2-narrator  (default)",
        "  lang = auto  (builtin)",
        "  base_lang = pl  (builtin)",
        "  rate = 1.5  (default)",
        "  volume = 50  (builtin)",
        "  steps = 12  (builtin)",
    ]
    assert lines[7] == "claude"
    assert "  rate = 1.25  (profile)" in lines
    assert lines[14] == "zz-audiobook"
    assert lines[-1] == "  steps = 24  (profile)"


def test_profiles_rm_removes_the_profile(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(home)
    assert main(["profiles", "rm", "claude"]) == 0
    assert capsys.readouterr().out.strip() == "removed profile 'claude'"
    data = tomllib.loads(path.read_text())
    assert set(data["profiles"]) == {"default", "zz-audiobook"}
    assert main(["say", "--profile", "claude", "hi"]) == 1
    assert "unknown profile 'claude'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("default", "profile 'default' cannot be removed"),
        ("nobody", "unknown profile 'nobody'"),
        ("Bad Name", "unknown profile 'Bad Name'"),
    ],
)
def test_profiles_rm_refuses_default_and_unknown_names(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str], name: str, message: str
) -> None:
    path = _config(home)
    assert main(["profiles", "rm", name]) == 1
    assert message in capsys.readouterr().err
    assert path.read_text() == _TOML


@pytest.mark.parametrize("as_json", [True, False])
def test_profiles_report_a_broken_value_and_list_the_rest(
    wire: FakeWire,
    home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    as_json: bool,
) -> None:
    (home / "config" / "config.toml").write_text(
        "[profiles.default]\nrate = 1.5\n[profiles.claude]\nrate = 9.0\nvolume = 70\n"
    )
    monkeypatch.setenv("VOXLEKTOR_STEPS", "many")
    assert main(["profiles", *(["--json"] if as_json else [])]) == 0
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "ignoring invalid rate in profile 'claude' of config.toml: "
        "rate must be between 0.5 and 3.0",
        "ignoring invalid VOXLEKTOR_STEPS in the environment: "
        "VOXLEKTOR_STEPS: invalid value 'many' for steps",
    ]
    if as_json:
        _, claude = json.loads(captured.out)["profiles"]
        assert claude["values"]["rate"] == 1.5
        assert claude["sources"]["rate"] == "default"
        assert claude["values"]["volume"] == 70
        assert claude["sources"]["steps"] == "builtin"
    else:
        assert "  rate = 1.5  (default)" in captured.out.splitlines()[8:]


def test_profiles_report_a_file_that_is_no_toml(
    wire: FakeWire, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (home / "config" / "config.toml").write_text("rate = [")
    assert main(["profiles", "--json"]) == 1
    assert capsys.readouterr().out == ""
