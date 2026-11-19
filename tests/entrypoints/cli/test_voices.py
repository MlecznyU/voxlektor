"""``vxl voices`` lists the local catalog (there is no protocol command for it)."""

import json

import pytest

from tests.entrypoints.cli.conftest import FakeWire
from voxlektor.entrypoints.cli.app import main


def test_lists_bundled_voices_and_marks_the_default(
    wire: FakeWire, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["voices"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "* v0-default" in lines
    assert "  m1-calm" in lines
    assert len(lines) == 8
    assert wire.requests == []


def test_marker_follows_the_configured_voice(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXLEKTOR_VOICE", "f2-narrator")
    assert main(["voices"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "* f2-narrator" in lines
    assert "  v0-default" in lines


def test_voices_json(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXLEKTOR_VOICE", "m1-calm")
    assert main(["voices", "--json"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    data = json.loads(out)
    assert data["default"] == "m1-calm"
    assert len(data["voices"]) == 8
    assert data["voices"] == sorted(data["voices"])
    assert "v0-default" in data["voices"]
    assert wire.requests == []
