"""Cue players: the memory recorder and the sounddevice player (device-free, with a fake stream)."""

import io
import logging
from collections.abc import Callable, Iterator
from importlib import resources
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pytest
import sounddevice as sd
import soundfile as sf

from voxlektor.adapters.audio import cues
from voxlektor.adapters.audio.cues import SoundDeviceCuePlayer
from voxlektor.adapters.audio.memory_out import MemoryCuePlayer
from voxlektor.app.effects import Cue
from voxlektor.app.ports import CuePlayer

type _Callback = Callable[[npt.NDArray[np.float32], int, object, sd.CallbackFlags], None]


class FakeStream:
    instances: ClassVar[list["FakeStream"]] = []
    fail_open: ClassVar[bool] = False

    def __init__(self, *, samplerate: float, callback: _Callback, **ignored: object) -> None:
        if FakeStream.fail_open:
            raise sd.PortAudioError("no device")
        self.samplerate = samplerate
        self.callback = callback
        self.started = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def abort(self, ignore_errors: bool = True) -> None:
        pass

    def close(self, ignore_errors: bool = True) -> None:
        assert not self.closed, "stream closed twice"
        self.closed = True

    def drain(self, frames: int = 512) -> npt.NDArray[np.float32]:
        """Pull blocks until the callback stops; return everything it wrote."""
        blocks: list[npt.NDArray[np.float32]] = []
        while True:
            outdata = np.full((frames, 1), 7.0, dtype=np.float32)
            try:
                self.callback(outdata, frames, None, sd.CallbackFlags())
            except sd.CallbackStop:
                blocks.append(outdata[:, 0].copy())
                return np.concatenate(blocks)
            blocks.append(outdata[:, 0].copy())


@pytest.fixture(autouse=True)
def fake_stream(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    FakeStream.instances = []
    FakeStream.fail_open = False
    monkeypatch.setattr(sd, "OutputStream", FakeStream)
    yield
    FakeStream.instances = []
    FakeStream.fail_open = False


def test_memory_player_records_cues_in_order() -> None:
    player = MemoryCuePlayer()
    port: CuePlayer = player
    port.play(Cue.RETRY)
    port.play(Cue.STOPPED)
    assert player.played == [Cue.RETRY, Cue.STOPPED]


def test_retry_is_a_short_soft_tone() -> None:
    player: CuePlayer = SoundDeviceCuePlayer()
    player.play(Cue.RETRY)
    stream = FakeStream.instances[0]
    assert stream.started
    rate = int(stream.samplerate)
    audio = stream.drain()
    nonzero = np.flatnonzero(audio)
    duration = (nonzero[-1] + 1) / rate
    assert 0.08 <= duration <= 0.2
    assert 0.02 < float(np.max(np.abs(audio))) <= 0.25  # audible but quiet
    assert abs(float(audio[0])) < 0.01  # the envelope fades in ...
    assert abs(float(audio[nonzero[-1]])) < 0.01  # ... and out, without clicks
    assert audio[nonzero[-1] + 1 :].tolist() == [0.0] * (len(audio) - nonzero[-1] - 1)


def test_stopped_plays_the_bundled_clip() -> None:
    player = SoundDeviceCuePlayer()
    player.play(Cue.STOPPED)
    stream = FakeStream.instances[0]
    assert stream.samplerate == 24000
    duration = len(stream.drain()) / stream.samplerate
    assert 2.0 < duration < 10.0  # "Reading stopped because of an error. Run vxl status ..."


def test_the_bundled_clip_is_24khz_mono_16bit_package_data() -> None:
    clip = resources.files("voxlektor.bundled_cues").joinpath("stopped_en.wav")
    info = sf.info(io.BytesIO(clip.read_bytes()))
    assert (info.samplerate, info.channels, info.subtype) == (24000, 1, "PCM_16")
    assert 2.0 < info.duration < 10.0


def test_a_new_cue_closes_the_previous_stream() -> None:
    player = SoundDeviceCuePlayer()
    player.play(Cue.RETRY)
    player.play(Cue.RETRY)
    first, second = FakeStream.instances
    assert first.closed
    assert not second.closed
    assert second.started


def test_close_closes_the_last_stream_once_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    player = SoundDeviceCuePlayer()
    player.close()  # nothing played yet
    player.play(Cue.RETRY)
    player.close()
    player.close()
    assert FakeStream.instances[0].closed
    player.play(Cue.RETRY)

    def broken(ignore_errors: bool = True) -> None:
        raise sd.PortAudioError("close failed")

    monkeypatch.setattr(FakeStream.instances[1], "close", broken)
    with caplog.at_level(logging.ERROR, logger=cues.__name__):
        player.close()
    assert "closing the cue stream failed" in caplog.text


def test_device_errors_are_logged_never_raised(caplog: pytest.LogCaptureFixture) -> None:
    player = SoundDeviceCuePlayer()
    FakeStream.fail_open = True
    with caplog.at_level(logging.ERROR, logger=cues.__name__):
        player.play(Cue.RETRY)
    assert "cue retry failed" in caplog.text


def test_a_missing_clip_is_logged_and_its_cue_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cues, "_STOPPED_CLIP", "missing.wav")
    with caplog.at_level(logging.WARNING, logger=cues.__name__):
        player = SoundDeviceCuePlayer()
        player.play(Cue.STOPPED)
    assert FakeStream.instances == []
    assert "cannot load cue clip" in caplog.text
    assert "no sound for cue stopped" in caplog.text
