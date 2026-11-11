import threading

import numpy as np
import pytest

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


def test_pause_holds_playback_until_resume() -> None:
    out = SoundDeviceOutput()
    done = threading.Event()
    out.set_callbacks(lambda _chunk: done.set(), lambda _chunk, _msg: done.set())
    out.start(ChunkId(1), _sine(0.3), _SR)
    out.pause()
    assert not done.wait(timeout=0.8)
    out.resume()
    assert done.wait(timeout=2.0)
