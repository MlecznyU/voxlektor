"""Cue players: short sounds that tell the listener what happened without a screen.

``SoundDeviceCuePlayer`` opens its own short, non-blocking stream for every cue, so CoreAudio
mixes it with the reading playback and it sounds even while the reading is paused. RETRY is a
generated soft tone; STOPPED is the bundled clip ``bundled_cues/stopped_en.wav`` (made once by
``scripts/make_cues.py``). Failures are logged and never raised into the loop.

Threads: ``play``/``close`` run on the loop thread; the stream callback (PortAudio thread) only
reads its own buffer. The stream of the last cue stays open until the next cue or ``close()``
(PortAudio forbids closing a stream from its own callback).

The device-less ``MemoryCuePlayer`` lives in ``memory_out`` (no audio library imports).
"""

import io
import logging
from importlib import resources

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf

from voxlektor.app.effects import Cue

_log = logging.getLogger(__name__)

_CUES_PACKAGE = "voxlektor.bundled_cues"
_STOPPED_CLIP = "stopped_en.wav"
_TONE_RATE = 24000
_TONE_HZ = 660.0
_TONE_SECONDS = 0.12
_TONE_PEAK = 0.12

type _Samples = npt.NDArray[np.float32]
type _Sound = tuple[_Samples, int]


class SoundDeviceCuePlayer:
    def __init__(self) -> None:
        self._sounds: dict[Cue, _Sound] = {Cue.RETRY: (_retry_tone(), _TONE_RATE)}
        clip = _load_clip(_STOPPED_CLIP)
        if clip is not None:
            self._sounds[Cue.STOPPED] = clip
        self._stream: sd.OutputStream | None = None

    def play(self, cue: Cue) -> None:
        try:
            self._play(cue)
        except Exception:
            _log.exception("cue %s failed", cue)

    def close(self) -> None:
        """Close the stream of the last cue (at shutdown); never raises."""
        previous, self._stream = self._stream, None
        if previous is None:
            return
        try:
            previous.close()
        except Exception:
            _log.exception("closing the cue stream failed")

    def _play(self, cue: Cue) -> None:
        sound = self._sounds.get(cue)
        if sound is None:
            _log.warning("no sound for cue %s", cue)
            return
        samples, rate = sound
        self.close()
        position = 0

        def callback(
            outdata: _Samples, frames: int, time: object, status: sd.CallbackFlags
        ) -> None:
            # PortAudio thread: copy frames only; no allocation, no logging.
            nonlocal position
            n = min(frames, len(samples) - position)
            outdata[:n, 0] = samples[position : position + n]
            position += n
            if n < frames:
                outdata[n:].fill(0.0)
                raise sd.CallbackStop

        stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32", callback=callback)
        self._stream = stream
        stream.start()


def _retry_tone() -> _Samples:
    """A short sine with a Hann envelope: soft, low and click-free."""
    n = round(_TONE_SECONDS * _TONE_RATE)
    t = np.arange(n, dtype=np.float64) / _TONE_RATE
    envelope = np.hanning(n)
    tone = _TONE_PEAK * envelope * np.sin(2.0 * np.pi * _TONE_HZ * t)
    return tone.astype(np.float32)


def _load_clip(name: str) -> _Sound | None:
    try:
        data = resources.files(_CUES_PACKAGE).joinpath(name).read_bytes()
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        _log.exception("cannot load cue clip %s", name)
        return None
    mono = samples.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(mono), rate
