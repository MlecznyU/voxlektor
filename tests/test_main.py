"""``python -m voxlektor`` runs the CLI and exits with its code."""

import runpy
import sys

import pytest


def test_module_entry_point_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["voxlektor", "--help"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("voxlektor", run_name="__main__")
    assert caught.value.code == 0
    assert "Usage" in capsys.readouterr().out


def test_importing_the_module_runs_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    namespace = runpy.run_module("voxlektor.__main__", run_name="voxlektor_main_import")
    assert callable(namespace["main"])
    assert capsys.readouterr().out == ""
