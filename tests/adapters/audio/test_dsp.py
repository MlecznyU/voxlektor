import numpy as np

from voxlektor.adapters.audio.dsp import time_stretch

_SR = 16000


def _tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(_SR * seconds), dtype=np.float64) / _SR
    return (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def test_rate_one_is_identity() -> None:
    samples = _tone()
    out = time_stretch(samples, 1.0)
    assert out is samples


def test_faster_rate_shortens_audio() -> None:
    out = time_stretch(_tone(), 1.5)
    assert out.dtype == np.float32
    assert abs(len(out) - _SR / 1.5) <= 0.05 * _SR / 1.5


def test_slower_rate_lengthens_audio() -> None:
    out = time_stretch(_tone(), 0.75)
    assert out.dtype == np.float32
    assert abs(len(out) - _SR / 0.75) <= 0.05 * _SR / 0.75
