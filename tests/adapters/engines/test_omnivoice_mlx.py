"""Real-model tests (``-m model``): Apple Silicon, ~3 GB RAM, the pinned revision is downloaded."""

import io
import re
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tests.fixtures.samples import BENCH_PARAGRAPHS_EN, BENCH_PARAGRAPHS_PL, PL_PARAGRAPH_110
from voxlektor.adapters.audio.dsp import trim_edges
from voxlektor.adapters.engines.omnivoice_mlx import OMNIVOICE_REVISION, OmniVoiceEngine
from voxlektor.adapters.engines.omnivoice_pace import (
    LANG_CORRECTION,
    speech_weight,
    target_seconds,
)
from voxlektor.adapters.storage.voices_fs import FsVoiceRepository
from voxlektor.app.voice_prompts import get_or_build_prompt
from voxlektor.contracts.engine import VoicePrompt
from voxlektor.testing.engine_contract import run_engine_contract

pytestmark = pytest.mark.model

VOICE = "v0-default"
SR = 24000
type Samples = npt.NDArray[np.float32]


def _bundled_repo(tmp_path: Path) -> FsVoiceRepository:
    repo = FsVoiceRepository(tmp_path / "voices")
    repo.install_bundled()
    return repo


def test_spec_pins_the_measured_revision() -> None:
    spec = OmniVoiceEngine().spec
    assert spec.id == "omnivoice"
    assert spec.revision == OMNIVOICE_REVISION
    assert len(OMNIVOICE_REVISION) == 40
    assert spec.sample_rate == 24000


def test_engine_contract_with_the_real_model(tmp_path: Path) -> None:
    engine = OmniVoiceEngine()
    try:
        run_engine_contract(engine, _bundled_repo(tmp_path).ref(VOICE))
    finally:
        engine.unload()


def test_polish_paragraph_latency_with_a_warm_cached_prompt(tmp_path: Path) -> None:
    assert len(PL_PARAGRAPH_110) == 110
    repo = _bundled_repo(tmp_path)
    ref = repo.ref(VOICE)
    engine = OmniVoiceEngine()
    engine.load()
    try:
        get_or_build_prompt(engine, repo, ref)  # builds and writes the cache
        prompt = get_or_build_prompt(engine, repo, ref)  # served from the cache
        list(engine.synthesize("Rozgrzewka.", "pl", prompt))  # warm up kernels
        started = time.perf_counter()
        chunks = list(engine.synthesize(PL_PARAGRAPH_110, "pl", prompt))
        elapsed = time.perf_counter() - started
    finally:
        engine.unload()
    seconds = sum(len(chunk.samples) for chunk in chunks) / engine.spec.sample_rate
    print(f"\nLATENCY 110-char PL paragraph: {elapsed:.2f}s for {seconds:.1f}s of audio")
    assert chunks


_GIB = 1 << 30
_CACHE_BOUND = 4 * _GIB  # without clearing, the cache grew past 20 GB over 20 generates
_CACHE_RUNS = 8


def test_mlx_cache_stays_bounded_over_many_syntheses(tmp_path: Path) -> None:
    import mlx.core as mx  # noqa: PLC0415  # only this test reads MLX's allocator directly

    repo = _bundled_repo(tmp_path)
    engine = OmniVoiceEngine()
    engine.load()
    sizes: list[int] = []
    try:
        prompt = get_or_build_prompt(engine, repo, repo.ref(VOICE))
        sizes.append(mx.get_cache_memory())  # after prepare_voice
        for _ in range(_CACHE_RUNS):
            list(engine.synthesize(PL_PARAGRAPH_110, "pl", prompt))
            sizes.append(mx.get_cache_memory())
        peak = mx.get_peak_memory()
    finally:
        engine.unload()
    shown = ", ".join(f"{size / _GIB:.2f}" for size in sizes)
    print(f"\nMLX CACHE GiB after prepare + {_CACHE_RUNS} runs: {shown}; peak {peak / _GIB:.2f}")
    assert max(sizes) < _CACHE_BOUND


# ------------------------------------------------------------------ native rate and steps

_PL_TEXT = BENCH_PARAGRAPHS_PL[1]
_EN_TEXT = BENCH_PARAGRAPHS_EN[1]
_VOWELS = {"pl": "aeiouyąęó", "en": "aeiouy"}
_RATE_TOLERANCE = 0.10  # trimmed speech length vs the requested duration
_KEPT = 0.8  # share of syllables that must survive (the envelope count is noisy, ~10 %)
_FRAME = SR // 100  # 10 ms


class _Voices:
    """One loaded engine and the bundled voices' prompts for the tests of this section."""

    def __init__(self, root: Path) -> None:
        self.repo = _bundled_repo(root)
        self.engine = OmniVoiceEngine()
        self.engine.load()
        self._prompts: dict[str, VoicePrompt] = {}

    def prompt(self, voice: str) -> VoicePrompt:
        if voice not in self._prompts:
            self._prompts[voice] = get_or_build_prompt(self.engine, self.repo, self.repo.ref(voice))
        return self._prompts[voice]

    def pace(self, voice: str) -> float:
        blob = self.engine.serialize_prompt(self.prompt(voice))
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            return float(archive["pace"])

    def speak(
        self, voice: str, text: str, lang: str, rate: float = 1.0, steps: int | None = None
    ) -> Samples:
        chunks = self.engine.synthesize(text, lang, self.prompt(voice), rate=rate, steps=steps)
        return np.concatenate([chunk.samples for chunk in chunks])


@pytest.fixture(scope="module")
def voices(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Voices]:
    loaded = _Voices(tmp_path_factory.mktemp("voices"))
    loaded.speak(VOICE, "Rozgrzewka.", "pl")  # warm up kernels
    yield loaded
    loaded.engine.unload()


def _seconds(samples: Samples) -> float:
    return len(samples) / SR


def _frame_db(samples: Samples) -> npt.NDArray[np.float64]:
    count = len(samples) // _FRAME
    frames = samples[: count * _FRAME].astype(np.float64).reshape(count, _FRAME)
    return 20.0 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)


def _syllables(samples: Samples) -> int:
    """Syllable nuclei: prominent peaks (>= 3 dB over both sides) of the smoothed envelope.

    Checked against the vowel groups of the sample texts (within ~10 %); good for ratios.
    """
    env = np.convolve(_frame_db(samples), np.ones(5) / 5, mode="same")
    floor = float(np.percentile(env, 95)) - 25.0
    count, last = 0, -100
    for i in range(1, len(env) - 1):
        if env[i] <= floor or env[i] < env[i - 1] or env[i] <= env[i + 1] or i - last < 8:
            continue
        if env[i] - env[max(0, i - 15) : i].min() >= 3 and env[i] - env[i + 1 : i + 16].min() >= 3:
            count += 1
            last = i
    return count


def _text_syllables(text: str, lang: str) -> int:
    return len(re.findall(f"[{_VOWELS[lang]}]+", text.lower()))


def test_native_rate_duration_within_tolerance(voices: _Voices) -> None:
    """The engine requests ``target_seconds`` and the model fills it: trimmed speech stays
    within tolerance of the request at 1.1 and 1.25, and gets shorter as the rate rises."""
    pace = voices.pace(VOICE)
    requested: dict[float, float] = {}
    trimmed: dict[float, float] = {}
    for rate in (1.0, 1.1, 1.25, 1.5):
        requested[rate] = target_seconds(_PL_TEXT, "pl", pace, "pl", rate)
        trimmed[rate] = _seconds(trim_edges(voices.speak(VOICE, _PL_TEXT, "pl", rate), SR))
    base = trimmed[1.0]
    shown = "\n".join(
        f"  {rate}: requested {requested[rate]:.2f}s, trimmed {trimmed[rate]:.2f}s "
        f"({trimmed[rate] / requested[rate]:.3f} of it), effective rate {base / trimmed[rate]:.3f}"
        for rate in trimmed
    )
    print(f"\nNATIVE RATE {VOICE} PL {len(_PL_TEXT)} chars (pace {pace:.2f} w/s):\n{shown}")
    for rate in (1.1, 1.25):
        assert abs(trimmed[rate] / requested[rate] - 1.0) <= _RATE_TOLERANCE, rate
    assert trimmed[1.0] > trimmed[1.1] > trimmed[1.25] > trimmed[1.5]


@pytest.mark.parametrize("voice", ["v0-default", "f2-narrator"])  # slowest and fastest pace
def test_native_rate_never_clips(voices: _Voices, voice: str) -> None:
    """Rate 1.5, an English chunk, a Polish voice, measured BEFORE trimming: the last word is
    spoken with full energy (trailing speech present, not swallowed) and no syllables are lost
    against the text and against rate 1.0 (envelope syllable count, ~10 % noise)."""
    raw = voices.speak(voice, _EN_TEXT, "en", 1.5)
    normal = trim_edges(voices.speak(voice, _EN_TEXT, "en", 1.0), SR)
    db = _frame_db(raw)
    peak = float(db.max())
    speech = np.flatnonzero(db > peak - 30.0)
    tail_ms = (len(db) - 1 - int(speech[-1])) * 10
    last_word_db = float(db[max(0, int(speech[-1]) - 30) : int(speech[-1]) + 1].max())
    fast, slow = _syllables(trim_edges(raw, SR)), _syllables(normal)
    expected = _text_syllables(_EN_TEXT, "en")
    print(
        f"\nNO CLIP {voice} EN at 1.5: raw {_seconds(raw):.2f}s, after the last speech frame "
        f"{tail_ms} ms, last 300 ms peak {last_word_db - peak:.1f} dB re max, last 40 ms "
        f"{float(db[-4:].max()) - peak:.1f} dB re max, syllables {fast} at 1.5 / {slow} at 1.0 "
        f"(text {expected})"
    )
    assert last_word_db > peak - 20.0  # the last word is spoken with energy, not swallowed
    assert fast >= _KEPT * expected  # no words dropped to fit the faster window
    assert fast >= _KEPT * slow


def test_steps_honoured(voices: _Voices) -> None:
    timings: dict[int, float] = {}
    for steps in (4, 8, 16):
        started = time.perf_counter()
        audio = voices.speak(VOICE, PL_PARAGRAPH_110, "pl", steps=steps)
        timings[steps] = time.perf_counter() - started
        assert audio.size > 0
    shown = ", ".join(f"{steps}: {secs:.2f}s" for steps, secs in timings.items())
    print(f"\nSTEPS timing (110-char PL chunk): {shown}")
    assert timings[4] < timings[8] < timings[16]
    assert timings[4] < 0.6 * timings[16]


def test_speaking_rate_per_voice_and_language(voices: _Voices) -> None:
    """Each voice's reference pace, the cps it gives per language, and the PL/EN correction.

    At equal speech-weight pace the syllable rates of both languages are compared; their ratio
    must match ``LANG_CORRECTION`` (which turns PL pace into EN pace).
    """
    ratios: list[float] = []
    lines: list[str] = []
    for voice in voices.repo.names():
        pace = voices.pace(voice)
        rates: dict[str, float] = {}
        cps: dict[str, float] = {}
        for lang, text in (("pl", _PL_TEXT), ("en", _EN_TEXT)):
            speech = trim_edges(voices.speak(voice, text, lang), SR)
            rates[lang] = _syllables(speech) / _seconds(speech)
            cps[lang] = len(text) / _seconds(speech)
        ratios.append(rates["en"] / rates["pl"])
        lines.append(
            f"{voice}: pace {pace:.2f} w/s, cps pl {cps['pl']:.2f} en {cps['en']:.2f}, "
            f"syl/s pl {rates['pl']:.2f} en {rates['en']:.2f}"
        )
    mean = statistics.mean(ratios)
    spread = statistics.pstdev(ratios)
    print("\nSPEAKING RATE\n" + "\n".join(lines))
    print(f"syllable rate EN/PL at equal pace: mean {mean:.3f}, sd {spread:.3f}")
    print(
        f"weight per char: pl {speech_weight(_PL_TEXT) / len(_PL_TEXT):.3f}, "
        f"en {speech_weight(_EN_TEXT) / len(_EN_TEXT):.3f}"
    )
    # Equal perceived pace needs EN pace = PL pace * (1 / measured ratio).
    assert abs(1.0 / mean - LANG_CORRECTION[("pl", "en")]) <= 0.1
