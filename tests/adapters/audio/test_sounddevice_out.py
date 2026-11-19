import importlib
import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest
import sounddevice as sd

from voxlektor.adapters.audio import sounddevice_out
from voxlektor.adapters.audio.cues import SoundDeviceCuePlayer
from voxlektor.adapters.audio.sounddevice_out import SoundDeviceOutput
from voxlektor.app.effects import Cue
from voxlektor.domain.types import ChunkId

pytestmark = pytest.mark.audio

_SR = 24000


class _CallbackTime(Protocol):
    """The ``time`` argument of a PortAudio stream callback (seconds on the stream clock)."""

    currentTime: float  # PortAudio's field names
    outputBufferDacTime: float


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
    out.abort()  # the stream stays open between chunks: never leave it running at exit
    assert got == [ChunkId(7)]
    assert failed == []


def test_abort_suppresses_on_done() -> None:
    out = SoundDeviceOutput()
    done = threading.Event()
    out.set_callbacks(lambda _chunk: done.set(), lambda _chunk, _msg: done.set())
    out.start(ChunkId(1), _sine(0.5), _SR)
    out.abort()
    assert not done.wait(timeout=1.0)


def test_chunks_play_in_order_on_one_stream_with_their_lead() -> None:
    """Smoke test of the continuous stream: two chunks, a 0.5 s lead, no reopen in between."""
    out = SoundDeviceOutput()
    done: list[tuple[ChunkId, float]] = []
    failed: list[str] = []
    finished = threading.Event()

    def on_done(chunk: ChunkId) -> None:
        done.append((chunk, time.monotonic()))
        if chunk == ChunkId(2):
            finished.set()

    out.set_callbacks(on_done, lambda _chunk, msg: failed.append(msg))
    started = time.monotonic()
    out.start(ChunkId(1), _sine(0.3), _SR)
    out.start(ChunkId(2), _sine(0.3), _SR, lead_s=0.5)
    assert finished.wait(timeout=3.0), "the second chunk did not finish within 3 s"
    out.abort()
    assert [chunk for chunk, _ in done] == [ChunkId(1), ChunkId(2)]
    assert failed == []
    assert done[1][1] - done[0][1] >= 0.75  # lead (0.5 s) + the second chunk (0.3 s), less jitter
    assert done[1][1] - started < 2.0


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


def test_every_callback_has_a_deadline_longer_than_a_gil_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The callback runs Python, so it waits for the GIL; another thread may hold it for tens of
    milliseconds (``mx.clear_cache()`` took up to 34 ms). Each buffer must therefore be handed
    over well before the device plays it: measured headroom (DAC time minus callback time)."""
    headroom: list[float] = []
    real = sounddevice_out._open_stream  # pyright: ignore[reportPrivateUsage]

    def measuring(
        sample_rate: int,
        callback: sounddevice_out._Callback,  # pyright: ignore[reportPrivateUsage]
        finished: Callable[[], None],
    ) -> sounddevice_out._Stream:  # pyright: ignore[reportPrivateUsage]
        def timed(
            outdata: npt.NDArray[np.float32], frames: int, time: object, status: sd.CallbackFlags
        ) -> None:
            info = cast("_CallbackTime", time)
            headroom.append(info.outputBufferDacTime - info.currentTime)
            callback(outdata, frames, time, status)

        return real(sample_rate, timed, finished)

    monkeypatch.setattr(sounddevice_out, "_open_stream", measuring)
    out = SoundDeviceOutput()
    done = threading.Event()
    out.set_callbacks(lambda _chunk: done.set(), lambda _chunk, msg: pytest.fail(msg))
    out.start(ChunkId(1), _sine(0.6), _SR)
    assert done.wait(timeout=3.0)
    out.abort()
    assert len(headroom) >= 2
    assert min(headroom) >= 0.06


def test_pause_holds_playback_until_resume() -> None:
    out = SoundDeviceOutput()
    done = threading.Event()
    out.set_callbacks(lambda _chunk: done.set(), lambda _chunk, _msg: done.set())
    out.start(ChunkId(1), _sine(0.3), _SR)
    out.pause()
    assert not done.wait(timeout=0.8)
    out.resume()
    assert done.wait(timeout=2.0)
    out.abort()


_REPO = Path(__file__).resolve().parents[3]


def stress(module: str = sounddevice_out.__name__, iterations: int = 300, seed: int = 1) -> None:
    """Child-process body of the stress test below (a deadlock cannot be interrupted in-process).

    Many start/abort/drain/release cycles plus cues on the real device, while another thread
    holds the GIL in bursts (one C call, like ``mx.clear_cache()``). Tiny buffers and a tiny idle
    threshold keep a callback in flight, and the stream stopping itself, as often as possible.
    """
    target = importlib.import_module(module)
    names = vars(target)
    names["_LATENCY_S"] = 0.005  # many tiny buffers: a callback is nearly always in flight
    names["_IDLE_S"] = 0.03  # the stream stops itself after every chunk left alone
    output_class = cast("type[SoundDeviceOutput]", target.SoundDeviceOutput)
    stop = threading.Event()

    def hold_gil() -> None:
        while not stop.is_set():
            sum(range(300_000))  # one C call: the GIL is held throughout (a few ms)
            time.sleep(0.002)

    threading.Thread(target=hold_gil, daemon=True).start()
    rng = random.Random(seed)  # noqa: S311
    out = output_class()
    out.set_callbacks(lambda _chunk: None, lambda _chunk, _msg: None)
    cues = SoundDeviceCuePlayer()
    for i in range(iterations):
        rate = rng.choice([_SR, _SR, 16000])  # another rate: the old stream is drained
        out.start(ChunkId(i + 1), _sine(rng.uniform(0.005, 0.03)), rate)
        if rng.random() < 0.3:
            cues.play(Cue.RETRY, 5)  # a 0.12 s tone; the next one closes it, maybe at its end
        time.sleep(rng.uniform(0.0, 0.06))
        action = rng.choice(["abort", "release", "none"])
        if action == "abort":
            out.abort()
        elif action == "release":
            out.release_idle_stream()
    out.abort()
    cues.close()
    stop.set()
    print("stress ok", flush=True)


def test_start_and_abort_never_hang_while_another_thread_holds_the_gil() -> None:
    """The field hang: abort/stop racing the callback's own stop deadlocked inside CoreAudio."""
    code = f"from {__name__} import stress; stress()"
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            cwd=_REPO,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("start/abort hung (killed after 120 s)")
    assert result.returncode == 0, result.stderr
    assert "stress ok" in result.stdout
