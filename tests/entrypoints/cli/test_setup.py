"""``vxl setup`` (interactive and ``--yes``) and the setup gate in front of speaking."""

import io
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

import voxlektor
from tests.entrypoints.cli.conftest import FakeWire, Tty
from voxlektor.adapters.engines.omnivoice_model import MODEL
from voxlektor.adapters.models import hf_cache
from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.adapters.terminal import menus
from voxlektor.app.ports import PlatformInfo
from voxlektor.app.setup import SetupCancelled
from voxlektor.bootstrap import client
from voxlektor.domain.errors import NotRunning
from voxlektor.domain.settings import Settings
from voxlektor.entrypoints import oneshot
from voxlektor.entrypoints.cli import setup as setup_cli
from voxlektor.entrypoints.cli.app import main
from voxlektor.testing.fakes import ENTER, FAKE_MODEL, FakeModelStore, ScriptedPrompter

NOT_SET_UP = "voxlektor is not set up yet; run: vxl setup"


def _config(home: Path) -> Path:
    return home / "config" / "config.toml"


def _data(home: Path) -> dict[str, object]:
    return tomllib.loads(_config(home).read_text())


class _Hub:
    """``snapshot_download`` without a network: ``cached`` repos are present."""

    def __init__(self) -> None:
        self.cached: set[str] = set()
        self.downloads: list[str] = []

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        tqdm_class: type[object] | None = None,
    ) -> str:
        if local_files_only:
            if repo_id not in self.cached:
                raise LocalEntryNotFoundError("not cached")
            return "/cache"
        self.downloads.append(repo_id)
        make: Any = tqdm_class  # duck-typed, as the library sees it
        size = 1_700_000_000  # OmniVoice's known size (the floor of the reported total)
        bar = make(desc="Reconstructing", total=size, initial=0)
        bar.update(size // 2)
        bar.update(size // 2)
        self.cached.add(repo_id)
        return "/cache"


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> _Hub:
    fake = _Hub()
    monkeypatch.setattr(hf_cache, "snapshot_download", fake.snapshot_download)
    return fake


@pytest.fixture
def fresh_wire(fresh_home: Path, monkeypatch: pytest.MonkeyPatch) -> FakeWire:
    """The fake wire in a home where setup has never run."""
    fake = FakeWire()
    monkeypatch.setattr(client, "request", fake.request)
    monkeypatch.setattr(client, "ensure_daemon", fake.ensure)
    return fake


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_cli, "_has_terminal", lambda: True)


def _script(monkeypatch: pytest.MonkeyPatch, answers: Sequence[object]) -> ScriptedPrompter:
    prompter = ScriptedPrompter(answers)
    monkeypatch.setattr(menus, "QuestionaryPrompter", lambda: prompter)
    return prompter


# ---------------------------------------------------------------- the gate


@pytest.mark.parametrize("source", ["text", "say", "stdin", "file"])
def test_speaking_before_setup_exits_1(
    fresh_wire: FakeWire,
    fresh_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
) -> None:
    notes = fresh_home / "notes.txt"
    notes.write_text("from a file")
    argv = {"text": ["hello"], "say": ["say", "hello"], "stdin": [], "file": ["-f", str(notes)]}
    if source == "stdin":
        monkeypatch.setattr(sys, "stdin", io.StringIO("from stdin"))
    assert main(argv[source]) == 1
    assert NOT_SET_UP in capsys.readouterr().err
    assert fresh_wire.requests == []
    assert fresh_wire.ensured == 0


def test_one_shot_mode_is_gated_too(
    fresh_wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXLEKTOR_AUTOSTART", "false")

    def must_not_run(text: str, settings: Settings) -> int:
        raise AssertionError("one-shot ran before setup")

    monkeypatch.setattr(oneshot, "run", must_not_run)
    assert main(["hello"]) == 1
    assert NOT_SET_UP in capsys.readouterr().err


def test_mvp_config_without_marker_is_gated(
    fresh_wire: FakeWire, fresh_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(fresh_home).parent.mkdir(parents=True)
    _config(fresh_home).write_text('config_version = 1\nvoice = "f1-warm"\n[service]\nx = 1\n')
    assert main(["hello"]) == 1
    assert NOT_SET_UP in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv", [["status"], ["stop"], ["voices"], ["config", "show"], ["pause"], ["idle", "5m"]]
)
def test_control_commands_are_not_gated(fresh_wire: FakeWire, argv: list[str]) -> None:
    assert main(argv) == 0


# ---------------------------------------------------------------- refusals and errors


def test_without_a_terminal_and_without_yes_it_refuses(
    fresh_wire: FakeWire, fresh_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup"]) == 1  # pytest's stdout is no terminal
    assert (
        "vxl setup needs a terminal; for tools use: vxl setup --yes [options]"
        in capsys.readouterr().err
    )
    assert not _config(fresh_home).exists()


def test_has_terminal_needs_both_stdin_and_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", Tty())
    monkeypatch.setattr(sys, "stdout", Tty())
    assert setup_cli._has_terminal() is True  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert setup_cli._has_terminal() is False  # pyright: ignore[reportPrivateUsage]


def test_setting_options_without_yes_exit_1(
    fresh_wire: FakeWire, terminal: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup", "--voice", "f1-warm"]) == 1
    assert "need --yes" in capsys.readouterr().err


def test_unsupported_platform_exits_3_with_the_issues_url(
    fresh_wire: FakeWire, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unsupported(*args: str) -> PlatformInfo:
        return UnsupportedPlatform()

    monkeypatch.setattr(client, "select_platform", unsupported)
    assert main(["setup", "--yes"]) == 3
    err = capsys.readouterr().err
    assert "platform not supported yet" in err
    assert "https://github.com/MlecznyU/voxlektor/issues" in err
    assert "https://github.com/MlecznyU/voxlektor/pulls" in err


# ---------------------------------------------------------------- flag mode


def test_flag_mode_end_to_end_with_the_hidden_fake_engine(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "setup",
        "--yes",
        "--engine",
        "fake",
        "--voice",
        "f1-warm",
        "--base-lang",
        "en",
        "--rate",
        "1.5",
        "--volume",
        "70",
        "--idle",
        "1h",
        "--autostart",  # --no-autostart would make the "hello" below read in this process
    ]
    assert main(argv) == 0
    data = _data(fresh_home)
    assert data == {
        "config_version": 1,
        "engine": "fake",
        "voice": "f1-warm",
        "base_lang": "en",
        "rate": 1.5,
        "volume": 70,
        "setup_completed": True,
        "service": {"idle_timeout": "1h", "autostart": True},
    }
    out = capsys.readouterr().out
    assert "volume = 70" in out
    assert f"config: {_config(fresh_home)}" in out
    assert fresh_wire.cmds == ["shutdown"]  # changed: a running daemon is stopped
    assert hub.downloads == []
    assert main(["hello"]) == 0  # the gate is open now
    assert fresh_wire.cmds[-1] == "speak"


def test_flag_mode_keeps_unknown_keys_and_tables(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub
) -> None:
    _config(fresh_home).parent.mkdir(parents=True)
    _config(fresh_home).write_text('foo = 1\nengine = "fake"\n[service]\nzed = 2\n[extra]\na = 1\n')
    assert main(["setup", "--yes"]) == 0
    data = _data(fresh_home)
    assert data["foo"] == 1
    assert data["extra"] == {"a": 1}
    assert data["service"] == {"zed": 2, "idle_timeout": "10m", "autostart": True}
    assert data["setup_completed"] is True


def test_flag_mode_without_changes_leaves_the_daemon_alone(fresh_wire: FakeWire, hub: _Hub) -> None:
    hub.cached.add("mlx-community/OmniVoice-bf16")
    assert main(["setup", "--yes"]) == 0
    assert fresh_wire.cmds == []


def test_flag_mode_download_needs_the_license_flag(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup", "--yes"]) == 1
    assert "model download needs --accept-model-license" in capsys.readouterr().err
    assert not _config(fresh_home).exists()
    assert hub.downloads == []


def test_flag_mode_downloads_with_the_license_flag_and_progress_on_stderr(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup", "--yes", "--accept-model-license", "--engine", "omnivoice"]) == 0
    assert hub.downloads == ["mlx-community/OmniVoice-bf16"]
    err = capsys.readouterr().err
    assert "downloading the model: 50%" in err
    assert "downloading the model: 100%" in err
    assert _data(fresh_home)["setup_completed"] is True


def test_stderr_progress_prints_one_line_per_step(capsys: pytest.CaptureFixture[str]) -> None:
    inner = FakeModelStore(steps=[(0, 0), (5, 100), (9, 100), (55, 100), (100, 100)])
    store = setup_cli._StderrProgress(inner)  # pyright: ignore[reportPrivateUsage]
    seen: list[tuple[int, int]] = []
    assert store.is_present(FAKE_MODEL) is False
    store.download(FAKE_MODEL, lambda done, total: seen.append((done, total)))
    assert len(seen) == 5  # everything is forwarded
    lines = capsys.readouterr().err.splitlines()
    assert lines == [
        "downloading the model: 0%",
        "downloading the model: 50%",
        "downloading the model: 100%",
    ]


def test_stderr_progress_is_redrawn_when_the_total_grows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inner = FakeModelStore(steps=[(10, 10), (10, 100), (5, 100), (55, 100), (55, 200), (200, 200)])
    setup_cli._StderrProgress(inner).download(FAKE_MODEL, lambda done, total: None)  # pyright: ignore[reportPrivateUsage]
    assert capsys.readouterr().err.splitlines() == [
        "downloading the model: 100%",
        "downloading the model: 10%",
        "downloading the model: 50%",
        "downloading the model: 20%",
        "downloading the model: 100%",
    ]


def test_interrupted_download_exits_2_and_marks_nothing(
    fresh_wire: FakeWire, fresh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(repo_id: str, **kwargs: object) -> str:
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        raise ConnectionError("offline")

    monkeypatch.setattr(hf_cache, "snapshot_download", offline)
    assert main(["setup", "--yes", "--accept-model-license"]) == 2
    assert not _config(fresh_home).exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--voice", "nobody"],
        ["--engine", "nope"],
        ["--base-lang", "de"],
        ["--rate", "9"],
        ["--volume", "101"],
        ["--idle", "soon"],
    ],
)
def test_flag_mode_rejects_invalid_values(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, argv: list[str]
) -> None:
    assert main(["setup", "--yes", "--engine", "fake", *argv]) == 1
    assert not _config(fresh_home).exists()


# ---------------------------------------------------------------- --check


def _check(capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["setup", "--check"])
    out = capsys.readouterr().out
    assert out.count("\n") == 1  # one JSON line, for tools
    return code, json.loads(out)


def test_check_before_setup_exits_1_and_reports_why(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    code, data = _check(capsys)
    assert code == 1
    assert data["ready"] is False
    assert data["setup_completed"] is False
    assert data["version"] == voxlektor.__version__
    assert data["config_path"] == str(_config(fresh_home))
    assert data["engine"] == "omnivoice"
    assert data["model"] == {
        "present": False,
        "repo": "mlx-community/OmniVoice-bf16",
        "revision": "8fb0b754cad788aaefec690cd55c207e8a628f85",
        "size_bytes": MODEL.size_bytes,
        "license_name": "CC-BY-NC-4.0",
        "license_url": "https://huggingface.co/k2-fsa/OmniVoice",
        "commercial_use": False,
    }
    assert data["settings"]["volume"] == 50
    assert fresh_wire.requests == []  # read-only: no daemon, nothing written
    assert not _config(fresh_home).exists()
    assert hub.downloads == []


def test_check_after_setup_with_the_model_present_exits_0(
    fresh_wire: FakeWire, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    hub.cached.add("mlx-community/OmniVoice-bf16")
    assert main(["setup", "--yes", "--voice", "m1-calm", "--volume", "70"]) == 0
    capsys.readouterr()
    code, data = _check(capsys)
    assert code == 0
    assert data["ready"] is True
    assert data["setup_completed"] is True
    assert data["model"]["present"] is True
    assert data["settings"] == {
        "voice": "m1-calm",
        "lang": "auto",
        "base_lang": "pl",
        "rate": 1.0,
        "volume": 70,
        "idle_timeout_s": 600,
        "autostart": True,
    }


def test_check_with_setup_done_but_model_missing_is_not_ready(
    fresh_wire: FakeWire, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    hub.cached.add("mlx-community/OmniVoice-bf16")
    assert main(["setup", "--yes"]) == 0
    hub.cached.clear()  # e.g. the HF cache was cleaned
    capsys.readouterr()
    code, data = _check(capsys)
    assert code == 1
    assert data["setup_completed"] is True
    assert data["model"]["present"] is False
    assert data["ready"] is False


def test_check_with_an_unknown_engine_reports_no_model(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(fresh_home).parent.mkdir(parents=True)
    _config(fresh_home).write_text('engine = "nope"\nsetup_completed = true\n')
    code, data = _check(capsys)
    assert code == 1
    assert data["engine"] == "nope"
    assert data["model"] is None
    assert data["ready"] is False


def test_check_tolerates_an_invalid_config_value(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(fresh_home).parent.mkdir(parents=True)
    _config(fresh_home).write_text("volume = 500\n")
    code = main(["setup", "--check"])
    captured = capsys.readouterr()
    assert code == 1
    assert json.loads(captured.out)["settings"]["volume"] == 50
    assert "ignoring invalid" in captured.err


@pytest.mark.parametrize("extra", [["--yes"], ["--volume", "70"], ["--accept-model-license"]])
def test_check_combined_with_other_options_exits_1(
    fresh_wire: FakeWire, hub: _Hub, extra: list[str]
) -> None:
    assert main(["setup", "--check", *extra]) == 1


def test_check_needs_no_terminal(
    fresh_wire: FakeWire,
    hub: _Hub,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code, _ = _check(capsys)
    assert code == 1  # not ready, but answered: no "needs a terminal" refusal


def test_stopping_when_no_daemon_runs_is_quiet(
    fresh_wire: FakeWire, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh_wire.replies["shutdown"] = NotRunning()
    assert main(["setup", "--yes", "--engine", "fake"]) == 0
    assert "stopped the running service" not in capsys.readouterr().err


def test_stopping_waits_until_the_daemon_is_gone(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket = fresh_home / "run" / "voxlektor.sock"
    socket.parent.mkdir(parents=True)
    socket.write_text("")
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds
        if clock[0] >= 0.3:
            socket.unlink(missing_ok=True)

    monkeypatch.setattr(setup_cli, "_sleep", sleep)
    monkeypatch.setattr(setup_cli, "_monotonic", lambda: clock[0])
    assert main(["setup", "--yes", "--engine", "fake"]) == 0
    assert not socket.exists()
    assert 0.3 <= clock[0] < 1.0


# ---------------------------------------------------------------- interactive


@pytest.mark.usefixtures("terminal")
def test_interactive_setup_end_to_end(
    fresh_wire: FakeWire,
    fresh_home: Path,
    hub: _Hub,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompter = _script(
        monkeypatch,
        [ENTER, True, "f1-warm", True, "en", ENTER, ENTER, 60, True, ENTER, ENTER],
    )
    assert main(["setup"]) == 0
    assert hub.downloads == ["mlx-community/OmniVoice-bf16"]
    assert "downloading the model: 100%" in prompter.said
    data = _data(fresh_home)
    assert data["voice"] == "f1-warm"
    assert data["volume"] == 60
    assert data["setup_completed"] is True
    speaks = [req for req in fresh_wire.requests if req["cmd"] == "speak"]
    # samples: voice, rate, volume; then the test sentence after the restart
    assert [req["volume"] for req in speaks] == [50, 50, 60, 60]
    assert all(req["now"] is True for req in speaks)
    assert speaks[-1]["text"] == "All set. This is how I will read."
    shutdown = fresh_wire.cmds.index("shutdown")
    assert fresh_wire.cmds[shutdown + 1 :][0] == "speak"  # the test sentence comes after
    assert f"config: {_config(fresh_home)}" in capsys.readouterr().out


def test_interactive_without_autostart_reads_no_test_sentence(
    fresh_wire: FakeWire, hub: _Hub, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub.cached.add("mlx-community/OmniVoice-bf16")
    _script(monkeypatch, [ENTER] * 9 + [False])
    assert main(["setup"]) == 0
    speaks = [req for req in fresh_wire.requests if req["cmd"] == "speak"]
    assert len(speaks) == 3  # the samples only


@pytest.mark.usefixtures("terminal")
def test_interactive_setup_stops_the_sample_daemon_even_without_changes(
    fresh_wire: FakeWire,
    fresh_home: Path,
    hub: _Hub,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With ``autostart = false`` the daemon started for the samples must not stay."""
    hub.cached.add("mlx-community/OmniVoice-bf16")
    _config(fresh_home).parent.mkdir(parents=True)
    _config(fresh_home).write_text("[service]\nautostart = false\n")
    _script(monkeypatch, [ENTER] * 10)  # Enter everywhere: nothing changes
    assert main(["setup"]) == 0
    assert fresh_wire.cmds[-1] == "shutdown"  # and no test sentence after it
    assert "stopped the running service" in capsys.readouterr().err


# ---------------------------------------------------------------- repairing a broken config


def test_flag_mode_repairs_invalid_config_values(
    fresh_wire: FakeWire, fresh_home: Path, hub: _Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(fresh_home).parent.mkdir(parents=True)
    _config(fresh_home).write_text('rate = 9.0\nvolume = 30\n[service]\nidle_timeout = "soon"\n')
    assert main(["setup", "--yes", "--engine", "fake"]) == 0
    err = capsys.readouterr().err
    assert "ignoring invalid rate in config.toml: rate must be between 0.5 and 3.0" in err
    assert "ignoring invalid idle_timeout in config.toml: invalid duration 'soon'" in err
    data = _data(fresh_home)
    assert (data["rate"], data["volume"]) == (1.0, 30)  # the default, and the valid value kept
    assert data["service"] == {"idle_timeout": "10m", "autostart": True}
    assert data["setup_completed"] is True


@pytest.mark.usefixtures("terminal")
def test_interactive_setup_starts_despite_an_invalid_env_value(
    fresh_wire: FakeWire,
    fresh_home: Path,
    hub: _Hub,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hub.cached.add("mlx-community/OmniVoice-bf16")
    monkeypatch.setenv("VOXLEKTOR_VOLUME", "loud")
    _script(monkeypatch, [ENTER] * 10)
    assert main(["setup"]) == 0
    err = capsys.readouterr().err
    assert "ignoring invalid VOXLEKTOR_VOLUME in the environment: " in err
    assert _data(fresh_home)["volume"] == 50


@pytest.mark.usefixtures("terminal")
def test_esc_in_a_menu_exits_1_and_writes_nothing(
    fresh_wire: FakeWire,
    fresh_home: Path,
    hub: _Hub,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hub.cached.add("mlx-community/OmniVoice-bf16")
    _script(monkeypatch, [ENTER, SetupCancelled()])
    assert main(["setup"]) == 1
    assert "setup cancelled" in capsys.readouterr().err
    assert not _config(fresh_home).exists()


def test_ctrl_c_in_a_menu_exits_130_and_writes_nothing(
    fresh_wire: FakeWire,
    fresh_home: Path,
    hub: _Hub,
    terminal: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _script(monkeypatch, [KeyboardInterrupt()])
    assert main(["setup"]) == 130
    assert not _config(fresh_home).exists()


def test_declining_the_model_license_exits_1_and_writes_nothing(
    fresh_wire: FakeWire,
    fresh_home: Path,
    hub: _Hub,
    terminal: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompter = _script(monkeypatch, [ENTER, ENTER])  # Enter on the license: the default is no
    assert main(["setup"]) == 1
    assert any("non-commercial" in line for line in prompter.said)
    assert hub.downloads == []
    assert not _config(fresh_home).exists()
