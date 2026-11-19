"""Signal processing helpers for playback: time stretch, loudness normalisation, gain, limiter.

``level`` runs per chunk on the engine thread right after ``time_stretch``:

1. **Normalise**: measure the speech level as the RMS of the 20 ms frames above -50 dBFS (pauses
   and near-silence do not count) and move it to -20 dBFS. The normalisation gain is clamped to
   [-12, +18] dB, so a near-silent chunk is never blown up; a chunk with no frame above the gate
   keeps its level (0 dB).
2. **Gain**: the user's ``volume`` (0..100): ``(volume / 50) ** 2``; 50 = 0 dB, 0 = silence.
3. **Limit**: a look-ahead peak limiter (gain reduction, never waveshaping): ``out = g * x`` with
   a smooth gain ``g`` in (0, 1] that keeps every sample at or below the -1 dBFS ceiling (0.891).
   The gain needed per sample is ``min(1, ceiling / |x|)``, taken as a minimum over 1 ms blocks,
   then smoothed per block: a linear attack that starts 5 ms ahead (the gain is down before the
   peak arrives), a 10 ms hold (no ripple between pitch periods) and a linear release of at most
   +1.0 per 100 ms. Block gains are interpolated linearly between block edges; each edge takes
   the lower of its two blocks, so no sample gets more gain than it may. A chunk whose peak is
   below the ceiling passes bit-identical, and so does every sample further than look-ahead,
   hold and release from a loud one.

Non-finite samples (NaN, inf) from an engine become 0 before anything is measured.
Deterministic, numpy-only and vectorised (no state between chunks).
"""

import numpy as np
import numpy.typing as npt
import pytsmod

type _Samples = npt.NDArray[np.float32]

_FRAME_S = 0.020  # loudness measurement frame
_GATE_DBFS = -50.0  # frames quieter than this are ignored when measuring
_TARGET_DBFS = -20.0  # speech RMS at volume 50
_NORM_MIN_DB = -12.0
_NORM_MAX_DB = 18.0
_CEILING = 0.891  # -1 dBFS; the limiter output never exceeds it
# The float32 just below the ceiling: the cast to float32 can never round above 0.891.
_SAFE_CEILING = float(np.nextafter(np.float32(_CEILING), np.float32(0.0)))
_BLOCK_S = 0.001  # limiter gain resolution
_LOOKAHEAD_BLOCKS = 5  # 5 ms: attack from unity to any depth, finished before the peak
_HOLD_BLOCKS = 10  # 10 ms: longer than a pitch period, so the gain does not ripple
_RELEASE_BLOCKS = 100  # 100 ms: the gain rises by at most 1.0 over this span
_UNITY_VOLUME = 50
_TRIM_FRAME_S = 0.010  # trim measurement frame
_TRIM_THRESHOLD = 10.0 ** (-50.0 / 20.0)  # -50 dBFS peak
_TRIM_MARGIN_S = 0.020  # kept around the audible part so consonants are not cut


def time_stretch(samples: npt.NDArray[np.float32], rate: float) -> npt.NDArray[np.float32]:
    """Change speed without changing pitch (``rate`` 1.5 = 1.5x faster); 1.0 is the identity.

    Empty input (an all-silent chunk after ``trim_edges``) is returned unchanged.
    """
    if rate == 1.0 or len(samples) == 0:
        return samples
    stretched = pytsmod.wsola(samples, 1.0 / rate)
    return np.asarray(stretched, dtype=np.float32)


def trim_edges(samples: npt.NDArray[np.float32], sample_rate: int) -> npt.NDArray[np.float32]:
    """Cut leading and trailing near-silence, keeping a 20 ms margin; all-silent gives empty.

    A 10 ms frame is audible when its peak is above -50 dBFS (peak, not RMS, so a short quiet
    consonant is never dropped). Non-finite samples count as silence.
    """
    frame = max(1, round(sample_rate * _TRIM_FRAME_S))
    magnitude = np.nan_to_num(np.abs(samples), nan=0.0, posinf=0.0, neginf=0.0)
    count = -(-len(magnitude) // frame)
    padded = np.zeros(count * frame, dtype=magnitude.dtype)
    padded[: len(magnitude)] = magnitude
    audible = np.flatnonzero(padded.reshape(count, frame).max(axis=1) > _TRIM_THRESHOLD)
    if len(audible) == 0:
        return samples[:0]
    margin = round(sample_rate * _TRIM_MARGIN_S)
    start = max(0, int(audible[0]) * frame - margin)
    end = min(len(samples), (int(audible[-1]) + 1) * frame + margin)
    return samples[start:end]


def fade_edges(
    samples: npt.NDArray[np.float32], sample_rate: int, fade_s: float = 0.005
) -> npt.NDArray[np.float32]:
    """Linear fade in and out (default 5 ms) so cuts do not click; first and last sample are 0."""
    if len(samples) == 0:
        return samples
    length = min(max(1, round(sample_rate * fade_s)), max(1, len(samples) // 2))
    ramp = np.arange(length, dtype=np.float32) / np.float32(length)
    out = samples.astype(np.float32, copy=True)
    out[:length] *= ramp
    out[-length:] *= ramp[::-1]
    return out


def gain(volume: int) -> float:
    """The linear gain of ``volume`` (0..100): 0 silent, 25 about -12 dB, 50 unity, 100 x4."""
    return (volume / _UNITY_VOLUME) ** 2


def level(
    samples: npt.NDArray[np.float32], sample_rate: int, volume: int
) -> npt.NDArray[np.float32]:
    """Normalise to the speech target, apply ``gain(volume)``, then limit (see the module doc)."""
    if volume == 0 or len(samples) == 0:
        return np.zeros(len(samples), dtype=np.float32)
    x = np.nan_to_num(samples.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    scale = 10.0 ** (_normalisation_db(x, sample_rate) / 20.0) * gain(volume)
    return _limit(x * scale, sample_rate).astype(np.float32)


def _normalisation_db(x: npt.NDArray[np.float64], sample_rate: int) -> float:
    frame = max(1, round(sample_rate * _FRAME_S))
    count = len(x) // frame
    frames = x[: count * frame].reshape(count, frame) if count else x.reshape(1, len(x))
    power = np.mean(np.square(frames), axis=1)
    active = power[power > 10.0 ** (_GATE_DBFS / 10.0)]
    if len(active) == 0:
        return 0.0
    speech_db = 10.0 * float(np.log10(np.mean(active)))
    return min(max(_TARGET_DBFS - speech_db, _NORM_MIN_DB), _NORM_MAX_DB)


def _limit(x: npt.NDArray[np.float64], sample_rate: int) -> npt.NDArray[np.float64]:
    """Look-ahead peak limiter: ``x * g`` with a smooth gain ``g`` (see the module doc)."""
    magnitude = np.abs(x)
    if float(np.max(magnitude)) <= _SAFE_CEILING:
        return x
    block = max(1, round(sample_rate * _BLOCK_S))
    count = -(-len(x) // block)
    needed = np.ones(count * block)
    loud = magnitude > _SAFE_CEILING
    needed[: len(x)][loud] = _SAFE_CEILING / magnitude[loud]
    gains = _smooth(np.min(needed.reshape(count, block), axis=1))
    edges = np.empty(count + 1)
    edges[0], edges[-1] = gains[0], gains[-1]
    edges[1:-1] = np.minimum(gains[:-1], gains[1:])
    positions = np.arange(count + 1, dtype=np.float64) * block
    return x * np.interp(np.arange(len(x), dtype=np.float64), positions, edges)


def _smooth(needed: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-block gain, never above ``needed``: linear attack ahead, hold, linear release.

    Each ramp is a running minimum of ``needed[j] +- slope * distance``; scaled by the ramp
    length the slopes are integers, so blocks left alone come out exactly 1.0.
    """
    index = np.arange(len(needed), dtype=np.float64)
    reach = _LOOKAHEAD_BLOCKS
    attack = (np.minimum.accumulate((needed * reach + index)[::-1])[::-1] - index) / reach
    padded = np.concatenate([np.ones(_HOLD_BLOCKS), attack])
    held = np.min(np.lib.stride_tricks.sliding_window_view(padded, _HOLD_BLOCKS + 1), axis=1)
    reach = _RELEASE_BLOCKS
    return (np.minimum.accumulate(held * reach - index) + index) / reach
