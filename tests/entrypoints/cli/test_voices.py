"""``vxl voices`` lists the local catalog (there is no protocol command for it)."""

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
    assert len(lines) == 7
    assert wire.requests == []


def test_marker_follows_the_configured_voice(
    wire: FakeWire, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXLEKTOR_VOICE", "f2-narrator")
    assert main(["voices"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "* f2-narrator" in lines
    assert "  v0-default" in lines
