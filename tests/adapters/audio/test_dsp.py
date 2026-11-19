import numpy as np
import numpy.typing as npt
import pytest

from voxlektor.adapters.audio.dsp import fade_edges, gain, level, time_stretch, trim_edges

_SR = 16000
_LEVEL_SR = 24000
_CEILING = 0.891

type _Samples = npt.NDArray[np.float32]


def _tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(_SR * seconds), dtype=np.float64) / _SR
    return (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def _sine_at(rms_dbfs: float, seconds: float = 1.0) -> _Samples:
    """A 220 Hz sine whose RMS is ``rms_dbfs``."""
    t = np.arange(int(_LEVEL_SR * seconds), dtype=np.float64) / _LEVEL_SR
    amplitude = np.sqrt(2.0) * 10.0 ** (rms_dbfs / 20.0)
    return (amplitude * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def _rms_dbfs(samples: _Samples) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))))


@pytest.mark.parametrize(("volume", "expected"), [(0, 0.0), (25, 0.25), (50, 1.0), (100, 4.0)])
def test_gain(volume: int, expected: float) -> None:
    assert gain(volume) == pytest.approx(expected)


@pytest.mark.parametrize("source_db", [-35.0, -20.0, -10.0])
def test_level_reaches_target(source_db: float) -> None:
    out = level(_sine_at(source_db), _LEVEL_SR, 50)
    assert out.dtype == np.float32
    assert len(out) == _LEVEL_SR
    assert _rms_dbfs(out) == pytest.approx(-20.0, abs=0.5)


def test_level_volume_scale() -> None:
    at_50 = _rms_dbfs(level(_sine_at(-35.0), _LEVEL_SR, 50))
    at_25 = _rms_dbfs(level(_sine_at(-35.0), _LEVEL_SR, 25))
    assert at_50 - at_25 == pytest.approx(12.0, abs=0.5)
    silent = level(_sine_at(-35.0), _LEVEL_SR, 0)
    assert silent.dtype == np.float32
    assert not silent.any()


def test_level_ignores_quiet_frames_when_measuring() -> None:
    """Pauses between words do not count: half silence, half speech still lands on target."""
    speech = _sine_at(-30.0)
    padded = np.concatenate([np.zeros(_LEVEL_SR, dtype=np.float32), speech])
    out = level(padded, _LEVEL_SR, 50)
    assert _rms_dbfs(out[_LEVEL_SR:]) == pytest.approx(-20.0, abs=0.5)
    assert not out[:_LEVEL_SR].any()


def test_level_normalisation_gain_is_clamped() -> None:
    quiet = level(_sine_at(-45.0), _LEVEL_SR, 50)  # needs +25 dB, gets at most +18
    assert _rms_dbfs(quiet) == pytest.approx(-27.0, abs=0.5)
    loud = level(_sine_at(3.0), _LEVEL_SR, 25)  # needs -23 dB, gets at most -12
    assert _rms_dbfs(loud) == pytest.approx(3.0 - 12.0 - 12.04, abs=0.5)


def test_level_silence_stays_silent() -> None:
    zeros = np.zeros(_LEVEL_SR, dtype=np.float32)
    assert not level(zeros, _LEVEL_SR, 100).any()
    assert len(level(np.zeros(0, dtype=np.float32), _LEVEL_SR, 50)) == 0
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(_LEVEL_SR) * 10.0 ** (-70.0 / 20.0)).astype(np.float32)
    out = level(noise, _LEVEL_SR, 50)
    assert np.isfinite(out).all()
    applied_db = _rms_dbfs(out) - _rms_dbfs(noise)
    assert applied_db <= 18.0 + 1e-3
    short = level(_sine_at(-30.0, seconds=0.005), _LEVEL_SR, 50)  # shorter than one frame
    assert np.isfinite(short).all()
    assert len(short) == int(_LEVEL_SR * 0.005)


def test_limiter_ceiling() -> None:
    loud = level(_sine_at(0.0), _LEVEL_SR, 100)
    assert float(np.max(np.abs(loud))) <= _CEILING
    rng = np.random.default_rng(3)
    spikes = (rng.standard_normal(_LEVEL_SR) * 0.05).astype(np.float32)
    spikes[::997] = 1.0
    spiky = level(spikes, _LEVEL_SR, 100)
    assert float(np.max(np.abs(spiky))) <= _CEILING
    assert np.isfinite(spiky).all()


def _clamped_scale(norm_db: float, volume: int) -> float:
    """The exact scale ``level`` applies when the normalisation gain is clamped to ``norm_db``."""
    return 10.0 ** (norm_db / 20.0) * gain(volume)


def _applied_gain(source: _Samples, out: _Samples, scale: float) -> npt.NDArray[np.float64]:
    """``out / (source * scale)`` per sample; NaN where the input is too small to tell."""
    before = source.astype(np.float64) * scale
    measurable = np.abs(before) > 0.01
    ratio = np.divide(out.astype(np.float64), before, out=np.zeros_like(before), where=measurable)
    return np.where(measurable, ratio, np.nan)


@pytest.mark.parametrize(
    ("rms_dbfs", "norm_db"),
    [(-45.0, 18.0), (7.6, -12.0)],  # quiet: clamped up to +18 dB; loud: peak lands at ~0.85
)
def test_limiter_passes_a_signal_below_the_ceiling_bit_identical(
    rms_dbfs: float, norm_db: float
) -> None:
    source = _sine_at(rms_dbfs)
    expected = (source.astype(np.float64) * _clamped_scale(norm_db, 50)).astype(np.float32)
    assert 0.0 < float(np.max(np.abs(expected))) < _CEILING
    np.testing.assert_array_equal(level(source, _LEVEL_SR, 50), expected)


def test_limiter_keeps_the_shape_of_a_loud_sine() -> None:
    """Gain reduction, not waveshaping: out = g * in with a smooth g in (0, 1]."""
    source = _sine_at(0.0)  # needs -20 dB, clamped to -12; volume 100 then puts the peak at 1.41
    out = level(source, _LEVEL_SR, 100)
    g = _applied_gain(source, out, _clamped_scale(-12.0, 100))
    known = g[np.isfinite(g)]
    assert float(np.max(np.abs(out))) <= _CEILING
    assert (known > 0.0).all()
    assert (known <= 1.0 + 1e-6).all()
    assert float(known.min()) < 0.7  # the limiter is really working here
    steps = np.diff(g)
    assert float(np.max(np.abs(steps[np.isfinite(steps)]))) < 0.01
    # Steady loud tone: one constant gain across every period, so the sine stays a sine.
    steady = g[int(0.05 * _LEVEL_SR) : int(0.95 * _LEVEL_SR)]
    steady = steady[np.isfinite(steady)]
    assert float(np.ptp(steady)) < 1e-3


def test_limiter_gain_drops_before_the_peak_and_recovers_after_it() -> None:
    # Loud noise (-6 dBFS) needs -14 dB, so the normalisation is clamped to exactly -12 dB.
    rng = np.random.default_rng(11)
    samples = (rng.standard_normal(_LEVEL_SR) * 0.5).astype(np.float32)
    middle = _LEVEL_SR // 2
    one_ms = _LEVEL_SR // 1000
    samples[middle] = 8.0  # 2.0 after scaling: must come down to the ceiling
    samples[middle - one_ms] = 2.0  # 0.5 after scaling: below the ceiling on its own
    samples[middle + one_ms] = 2.0
    before = samples.astype(np.float64) * _clamped_scale(-12.0, 50)
    out = level(samples, _LEVEL_SR, 50)
    assert abs(float(out[middle])) <= _CEILING
    # 1 ms before the spike the gain is already down (look-ahead), and 1 ms after it still is.
    assert float(out[middle - one_ms]) < 0.9 * before[middle - one_ms]
    assert float(out[middle + one_ms]) < 0.9 * before[middle + one_ms]
    # Far from it (beyond look-ahead, hold and release) every sample is bit-identical.
    far = int(0.2 * _LEVEL_SR)
    untouched = np.r_[0 : middle - far, middle + far : _LEVEL_SR]
    assert float(np.max(np.abs(before[untouched]))) < _CEILING
    np.testing.assert_array_equal(out[untouched], before[untouched].astype(np.float32))


def test_level_maps_non_finite_samples_to_silence() -> None:
    source = _sine_at(-20.0)
    broken = source.copy()
    broken[100] = np.nan
    broken[200] = np.inf
    broken[300] = -np.inf
    out = level(broken, _LEVEL_SR, 50)
    assert np.isfinite(out).all()
    assert out[100] == 0.0
    assert out[200] == 0.0
    assert out[300] == 0.0
    assert _rms_dbfs(out) == pytest.approx(-20.0, abs=0.5)
    assert not level(np.full(_LEVEL_SR, np.nan, dtype=np.float32), _LEVEL_SR, 100).any()


def test_rate_one_is_identity() -> None:
    samples = _tone()
    out = time_stretch(samples, 1.0)
    assert out is samples


def test_faster_rate_shortens_audio() -> None:
    out = time_stretch(_tone(), 1.5)
    assert out.dtype == np.float32
    assert abs(len(out) - _SR / 1.5) <= 0.05 * _SR / 1.5


def test_empty_input_is_returned_unchanged_at_any_rate() -> None:
    # An all-silent chunk is trimmed to nothing before the stretch; pytsmod raises on empty.
    empty = np.zeros(0, dtype=np.float32)
    out = time_stretch(empty, 1.5)
    assert out is empty


def test_slower_rate_lengthens_audio() -> None:
    out = time_stretch(_tone(), 0.75)
    assert out.dtype == np.float32
    assert abs(len(out) - _SR / 0.75) <= 0.05 * _SR / 0.75


def _padded(lead_s: float, body: _Samples, tail_s: float) -> _Samples:
    lead = np.zeros(round(_SR * lead_s), dtype=np.float32)
    tail = np.zeros(round(_SR * tail_s), dtype=np.float32)
    return np.concatenate([lead, body, tail])


def test_trim_cuts_silence_to_tone_plus_margin() -> None:
    tone = _tone(0.5)
    out = trim_edges(_padded(1.0, tone, 1.0), _SR)
    assert out.dtype == np.float32
    margin = round(_SR * 0.02)
    assert len(tone) + 2 * margin - 0.011 * _SR <= len(out) <= len(tone) + 2 * margin + 0.011 * _SR
    assert np.array_equal(out[margin + 160 : -margin - 160], tone[160:-160])


def test_trim_all_silent_is_empty() -> None:
    out = trim_edges(np.full(_SR, 1e-5, dtype=np.float32), _SR)
    assert len(out) == 0
    assert out.dtype == np.float32


def test_trim_empty_input() -> None:
    assert len(trim_edges(np.zeros(0, dtype=np.float32), _SR)) == 0


def test_trim_without_silence_keeps_length() -> None:
    tone = _tone(0.5)
    assert len(trim_edges(tone, _SR)) == len(tone)


def test_trim_keeps_quiet_burst_near_the_edge() -> None:
    burst = (10.0 ** (-40.0 / 20.0) * _tone(0.03) / 0.3).astype(np.float32)
    signal = _padded(0.5, burst, 0.5)
    tone_end = np.concatenate([signal, _tone(0.3)])
    out = trim_edges(tone_end, _SR)
    assert np.max(np.abs(out[: round(_SR * 0.06)])) > 0.009
    only = trim_edges(signal, _SR)
    assert 0 < len(only) < len(signal)
    assert np.max(np.abs(only)) == pytest.approx(np.max(np.abs(burst)))


def test_trim_handles_non_finite_as_silence() -> None:
    samples = np.full(_SR, np.nan, dtype=np.float32)
    assert len(trim_edges(samples, _SR)) == 0


def test_fade_zeroes_edges_and_is_monotone() -> None:
    ones = np.ones(_SR, dtype=np.float32)
    out = fade_edges(ones, _SR)
    n = round(_SR * 0.005)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert out[-1] == 0.0
    assert np.all(np.diff(out[:n]) > 0)
    assert np.all(np.diff(out[-n:]) < 0)
    assert np.array_equal(out[n:-n], ones[n:-n])
    assert np.all(ones == 1.0)


def test_fade_short_input_and_empty() -> None:
    out = fade_edges(np.ones(3, dtype=np.float32), _SR)
    assert out[0] == 0.0
    assert out[-1] == 0.0
    one = fade_edges(np.ones(1, dtype=np.float32), _SR)
    assert one[0] == 0.0
    assert len(fade_edges(np.zeros(0, dtype=np.float32), _SR)) == 0
