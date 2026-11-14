import os
import threading
import time

import numpy as np
import pytest
import sounddevice as sd

from voxlektor.adapters.audio import sounddevice_out
from voxlektor.adapters.audio.sounddevice_out import SoundDeviceOutput
from voxlektor.domain.types import ChunkId

pytestmark = pytest.mark.audio

_SR = 24000


def _sine(seconds: float) -> np.ndarray:
    t = np.arange(int(_SR * seconds), dtype=np.float64) / _SR
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def test_plays_half_second_sine_and_fires_on_done() -> None:
    out = SoundDeviceOutput()
    done = threading.Event()
    got: list[ChunkId] = []
    failed: list[str] = []

    def on_done(chunk: ChunkId) -> None:
        got.append(chunk)
        done.set()

    out.set_callbacks(on_done, lambda _chunk, msg: failed.append(msg))
    out.start(ChunkId(7), _sine(0.5), _SR)
    assert done.wait(timeout=2.0), "on_done did not fire within 2 s"
    assert got == [ChunkId(7)]
    assert failed == []


def test_abort_suppresses_on_done() -> None:
    out = SoundDeviceOutput()
    done = threading.Event()
    out.set_callbacks(lambda _chunk: done.set(), lambda _chunk, _msg: done.set())
    out.start(ChunkId(1), _sine(0.5), _SR)
    out.abort()
    assert not done.wait(timeout=1.0)


def test_back_to_back_start_plays_only_the_last_chunk() -> None:
    out = SoundDeviceOutput()
    done: list[ChunkId] = []
    finished = threading.Event()

    def on_done(chunk: ChunkId) -> None:
        done.append(chunk)
        finished.set()

    out.set_callbacks(on_done, lambda _chunk, _msg: finished.set())
    out.start(ChunkId(1), _sine(0.4), _SR)
    out.start(ChunkId(2), _sine(0.2), _SR)
    assert finished.wait(timeout=2.0)
    out.abort()
    assert done == [ChunkId(2)]


@pytest.fixture
def reinits() -> None:
    """Overrides the device-free stand-in of tests/adapters/audio/conftest.py: real PortAudio."""


def test_follows_a_default_device_switch_between_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    """C1, the device switch scenario (manual when a wait is given).

    Run ``AUDIO_SWITCH_WAIT_S=15 uv run pytest -m audio -s -k device_switch``: a first tone plays,
    then the test waits; switch the macOS default output meanwhile (Control Centre > Sound, or
    plug in headphones). The second tone must come out of the NEW device. Without the wait it
    still proves that a real PortAudio re-initialisation between two readings keeps playback
    working and happens exactly once.
    """
    initialize = vars(sd)["_initialize"]
    calls: list[None] = []

    def counting_initialize() -> None:
        calls.append(None)
        initialize()

    monkeypatch.setattr(sd, "_initialize", counting_initialize)
    out = SoundDeviceOutput()
    done = threading.Event()
    out.set_callbacks(lambda _chunk: done.set(), lambda _chunk, msg: pytest.fail(msg))
    out.start(ChunkId(1), _sine(0.3), _SR)
    assert done.wait(timeout=2.0)
    wait_s = float(os.environ.get("AUDIO_SWITCH_WAIT_S", "0"))
    if wait_s:
        print(f"\nswitch the default output device now ({wait_s:g} s)...", flush=True)
        time.sleep(wait_s)
    offset = 10.0  # the output has been quiet for longer than its idle threshold
    monkeypatch.setattr(sounddevice_out, "_monotonic", lambda: time.monotonic() + offset)
    done.clear()
    out.start(ChunkId(2), _sine(0.3), _SR)
    assert done.wait(timeout=2.0), "the reading after the switch did not play"
    assert len(calls) == 1
    out.abort()


def test_pause_holds_playback_until_resume() -> None:
    out = SoundDeviceOutput()
    done = threading.Event()
    out.set_callbacks(lambda _chunk: done.set(), lambda _chunk, _msg: done.set())
    out.start(ChunkId(1), _sine(0.3), _SR)
    out.pause()
    assert not done.wait(timeout=0.8)
    out.resume()
    assert done.wait(timeout=2.0)
