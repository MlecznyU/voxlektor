"""One-shot mode in process: the daemon's CoreLoop without a socket, memory output, fake engine."""

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from voxlektor.app.ports import AudioOutput, CuePlayer, PlatformInfo
from voxlektor.bootstrap import daemon as wiring
from voxlektor.contracts.engine import Engine
from voxlektor.domain.errors import UnknownVoice
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId, VoiceName
from voxlektor.entrypoints import oneshot
from voxlektor.testing.fakes import FakeEngine, MemoryOutput

FAKE = Settings(engine=EngineId("fake"))


@pytest.fixture
def home(short_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith("VOXLEKTOR_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("VOXLEKTOR_HOME", str(short_tmp))
    monkeypatch.setenv("VOXLEKTOR_OUTPUT", "memory")
    return short_tmp


class Wired:
    """Captures what ``build_daemon`` wires for ``oneshot.run`` and lets a test pick parts."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, output: MemoryOutput | None) -> None:
        self.engine = FakeEngine(sample_rate=100)
        self.output = output
        real = wiring.build_daemon

        def provider(engine_id: str) -> Engine:
            assert engine_id == "fake"
            return self.engine

        def build(
            settings: Settings,
            *,
            transport: bool,
            platform: PlatformInfo | None = None,
            output: AudioOutput | None = None,
            cues: CuePlayer | None = None,
        ) -> wiring.Daemon:
            assert transport is False
            return real(settings, transport=transport, platform=platform, output=self.output)

        monkeypatch.setattr(wiring, "load_engine", provider)
        monkeypatch.setattr(oneshot, "build_daemon", build)


@pytest.mark.usefixtures("home")
def test_one_shot_reads_and_exits_0() -> None:
    assert oneshot.run("Hello world. This is a one-shot reading.", FAKE) == 0


@pytest.mark.usefixtures("home")
def test_one_shot_plays_every_chunk_and_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    wired = Wired(monkeypatch, MemoryOutput(auto_complete=True))
    assert oneshot.run("Hello world.", FAKE) == 0
    assert wired.output is not None
    assert len(wired.output.started) == 1
    assert wired.engine.load_calls == 1
    assert wired.engine.unload_calls == 1


@pytest.mark.usefixtures("home")
def test_failed_reading_exits_2_with_the_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wired = Wired(monkeypatch, MemoryOutput(auto_complete=True))
    wired.engine = FakeEngine(sample_rate=100, fail_on="BOOM")
    assert oneshot.run("BOOM goes the engine.", FAKE) == 2
    assert "fake engine failure" in capsys.readouterr().err


@pytest.mark.usefixtures("home")
def test_rejected_speak_raises_after_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    wired = Wired(monkeypatch, MemoryOutput(auto_complete=True))
    with pytest.raises(UnknownVoice):
        oneshot.run("Hello.", Settings(engine=EngineId("fake"), voice=VoiceName("nobody")))
    assert wired.engine.load_calls == 0


@pytest.mark.usefixtures("home")
def test_ctrl_c_stops_the_reading_unloads_and_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    output = MemoryOutput(auto_complete=False)  # the first chunk plays until interrupted
    wired = Wired(monkeypatch, output)

    def press_ctrl_c() -> None:
        deadline = time.monotonic() + 10.0
        while not output.started and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGINT)

    presser = threading.Thread(target=press_ctrl_c, daemon=True)
    presser.start()
    assert oneshot.run("Hello world. This reading is interrupted.", FAKE) == 130
    presser.join(timeout=5.0)
    assert output.started  # it was playing when Ctrl+C arrived
    assert output.aborted >= 1  # StopReading aborted the playback
    assert wired.engine.unload_calls == 1
