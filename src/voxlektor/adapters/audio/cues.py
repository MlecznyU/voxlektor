"""Cue players: short sounds that tell the listener what happened without a screen.

``SoundDeviceCuePlayer`` opens its own short, non-blocking stream for every cue, so CoreAudio
mixes it with the reading playback and it sounds even while the reading is paused. RETRY is a
generated soft tone; STOPPED is the bundled clip ``bundled_cues/stopped_en.wav`` (made once by
``scripts/make_cues.py``). Failures are logged and never raised into the loop.

Loudness: every cue is normalised once at load to the speech target (``dsp.level`` at volume 50);
at play it goes through ``dsp.level`` at the reading's volume (gain + the same limiter) and then
6 dB down, so cues sit 6 dB under speech (-26 dBFS at volume 50) and never above -7 dBFS. Volume 0
plays nothing. The prepared samples of the last volume are kept per cue.

Threads: ``play``/``close`` run on the loop thread; the stream callback (PortAudio thread) only
reads its own buffer. The stream of the last cue stays open until the next cue or ``close()``
(PortAudio forbids closing a stream from its own callback), or until ``portaudio.refresh_devices``
asks for it once it has stopped (``release_idle_stream``): the cue player never holds a stream
across a PortAudio re-initialisation, and a cue still sounding postpones the re-initialisation.
A stream is never closed while its callback stops it (``_CueStream``: a CoreAudio deadlock); it
is set aside until its finished callback came.

The device-less ``MemoryCuePlayer`` lives in ``memory_out`` (no audio library imports).
"""

import io
import logging
import threading
from importlib import resources

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf

from voxlektor.adapters.audio import portaudio
from voxlektor.adapters.audio.dsp import level
from voxlektor.app.effects import Cue

_log = logging.getLogger(__name__)

_CUES_PACKAGE = "voxlektor.bundled_cues"
_STOPPED_CLIP = "stopped_en.wav"
_TONE_RATE = 24000
_TONE_HZ = 660.0
_TONE_SECONDS = 0.12
_TONE_PEAK = 0.12
_CUE_UNDER_SPEECH = 10.0 ** (-6.0 / 20.0)  # cues sit 6 dB under speech at every volume
# As the reading stream (``sounddevice_out._LATENCY_S``): the Python callback waits for the GIL,
# which other threads hold for tens of ms; 0.2 s of buffer keeps such a hold from a dropout.
_LATENCY_S = 0.2

type _Samples = npt.NDArray[np.float32]
type _Sound = tuple[_Samples, int]


class _CueStream:
    """One cue's stream, stopped by its callback at the end or closed by the player, never both.

    Both at once deadlock inside CoreAudio: the callback's ``CallbackStop`` makes PortAudio
    call ``AudioOutputUnitStop`` on the IO thread (holding the IO-context mutex, waiting for
    the unit's), ``close`` on the loop thread holds the unit's mutex and waits for the
    IO-context mutex. So both sides decide under ``_lock`` (never held across a PortAudio call):
    once the player took the stream over, its callback writes silence and never stops it; once
    the callback chose to stop it, the player waits for the finished callback before closing.
    """

    def __init__(self, samples: _Samples, rate: int) -> None:
        self._samples = samples
        self._position = 0
        self._lock = threading.Lock()
        self._taken_over = False  # the player closes it: the callback must not stop it
        self._stopping = False  # the callback stops it itself
        self._finished = False  # PortAudio's finished callback came
        self.stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            latency=_LATENCY_S,
            callback=self._callback,
            finished_callback=self._on_finished,
        )

    def has_finished(self) -> bool:
        with self._lock:
            return self._finished

    def take_over(self) -> bool:
        """Loop thread: True when the player may close the stream now (and the callback will
        never stop it); False while the callback is stopping it (ask again later)."""
        with self._lock:
            if self._stopping and not self._finished:
                return False
            self._taken_over = True
            return True

    def _callback(self, outdata: _Samples, frames: int, time: object, status: object) -> None:
        # PortAudio thread: copy frames only; no allocation, no logging.
        with self._lock:
            if self._taken_over:
                outdata.fill(0.0)
                return
            samples, position = self._samples, self._position
            n = min(frames, len(samples) - position)
            outdata[:n, 0] = samples[position : position + n]
            self._position = position + n
            if n == frames:
                return
            outdata[n:].fill(0.0)
            self._stopping = True
        raise sd.CallbackStop

    def _on_finished(self) -> None:
        with self._lock:
            self._finished = True


class SoundDeviceCuePlayer:
    def __init__(self) -> None:
        self._sounds: dict[Cue, _Sound] = {Cue.RETRY: (_retry_tone(), _TONE_RATE)}
        clip = _load_clip(_STOPPED_CLIP)
        if clip is not None:
            self._sounds[Cue.STOPPED] = clip
        # Normalised once, at load, to the speech target (volume 50).
        self._sounds = {c: (level(s, rate, 50), rate) for c, (s, rate) in self._sounds.items()}
        self._ready: dict[Cue, tuple[int, _Sound]] = {}  # the last volume asked for, per cue
        self._stream: _CueStream | None = None
        self._unclosed: list[_CueStream] = []  # stopping themselves: closed once finished
        portaudio.register(self)

    def release_idle_stream(self) -> bool:
        """``portaudio.StreamOwner``: close the last cue's stream once it has stopped playing."""
        if self._stream is not None and not self._stream.has_finished():
            return False  # still sounding, or still stopping itself
        self.close()
        return not self._unclosed

    def play(self, cue: Cue, volume: int = 50) -> None:
        try:
            self._play(cue, volume)
        except Exception:
            _log.exception("cue %s failed", cue)

    def close(self) -> None:
        """Close the stream of the last cue (at shutdown); never raises.

        A stream whose callback is stopping it is set aside and closed by a later ``close``
        once it has finished (see ``_CueStream``).
        """
        previous, self._stream = self._stream, None
        if previous is not None:
            self._unclosed.append(previous)
        unclosed, self._unclosed = self._unclosed, []
        for cue_stream in unclosed:
            if cue_stream.take_over():
                self._close_stream(cue_stream.stream)
            else:
                self._unclosed.append(cue_stream)

    @staticmethod
    def _close_stream(stream: sd.OutputStream) -> None:
        try:
            stream.close()
        except Exception:
            _log.exception("closing the cue stream failed")

    def _prepared(self, cue: Cue, volume: int) -> _Sound | None:
        """The cue at ``volume``: ``level`` (gain + limiter), then 6 dB under speech; cached."""
        cached = self._ready.get(cue)
        if cached is not None and cached[0] == volume:
            return cached[1]
        sound = self._sounds.get(cue)
        if sound is None:
            return None
        samples, rate = sound
        leveled = (level(samples, rate, volume) * _CUE_UNDER_SPEECH).astype(np.float32)
        self._ready[cue] = (volume, (leveled, rate))
        return leveled, rate

    def _play(self, cue: Cue, volume: int) -> None:
        if volume == 0:
            return  # silence: no stream at all
        sound = self._prepared(cue, volume)
        if sound is None:
            _log.warning("no sound for cue %s", cue)
            return
        samples, rate = sound
        self.close()
        portaudio.refresh_devices(idle=False)  # after a playback failure: the current device
        cue_stream = _CueStream(samples, rate)
        try:
            cue_stream.stream.start()
        except BaseException:
            self._close_stream(cue_stream.stream)  # never started: no callback runs
            raise
        self._stream = cue_stream


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
