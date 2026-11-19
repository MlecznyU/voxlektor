from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from voxlektor.contracts.engine import AudioChunk, VoicePrompt, VoiceRef
from voxlektor.testing.engine_contract import run_engine_contract
from voxlektor.testing.fakes import FakeEngine

REF = VoiceRef("v1", Path("/voices/v1/ref.wav"), "reference transcript")


class _Silent(FakeEngine):
    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        return iter(())


class _WrongRate(FakeEngine):
    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        yield AudioChunk(np.zeros(100, np.float32), self.spec.sample_rate + 1)


class _Float64(FakeEngine):
    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        wide_dtype: str = "float64"  # a plain str keeps numpy from inferring float64 statically
        yield AudioChunk(np.zeros(100, wide_dtype), self.spec.sample_rate)


class _Stereo(FakeEngine):
    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        yield AudioChunk(np.zeros((100, 2), np.float32), self.spec.sample_rate)


class _LossyRoundTrip(FakeEngine):
    """Audio length follows the prompt text, and deserialization throws that text away."""

    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        yield AudioChunk(np.zeros(1000 * len(str(prompt.data)), np.float32), self.spec.sample_rate)

    def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt:
        return VoicePrompt(voice, self.PROMPT_FORMAT, "x")


class _RateIgnored(FakeEngine):
    """Declares RATE but speaks every rate at the same length."""

    def __init__(self) -> None:
        super().__init__(native_rate=True)

    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        return super().synthesize(text, lang, prompt, steps=steps)


class _RateWithFixedTail(FakeEngine):
    """Honours rate on the speech but adds a tail the rate does not shorten (as OmniVoice's
    ``TAIL_S``): at rate 1.5 the audio is clearly shorter, but not by a full factor 1.5."""

    def __init__(self) -> None:
        super().__init__(native_rate=True)

    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        speech = round(10_000 / rate)
        yield AudioChunk(np.zeros(speech + 4_000, np.float32), self.spec.sample_rate)


def test_contract_accepts_a_rate_engine_with_a_fixed_tail() -> None:
    run_engine_contract(_RateWithFixedTail(), REF)


class _NoOptions(FakeEngine):
    """Breaks the SPI: no ``rate``/``steps`` keywords."""

    def synthesize(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, text: str, lang: str, prompt: VoicePrompt
    ) -> Iterator[AudioChunk]:
        return super().synthesize(text, lang, prompt)


class _WrongPromptFormat(FakeEngine):
    def prepare_voice(self, ref: VoiceRef) -> VoicePrompt:
        return VoicePrompt(ref.name, self.PROMPT_FORMAT + 1, ref.ref_text)


class _NotIdempotent(FakeEngine):
    def load(self) -> None:
        if self.load_calls:
            raise RuntimeError("second load")
        super().load()


@pytest.mark.parametrize(
    ("engine", "message"),
    [
        (_Silent(), "at least one chunk"),
        (_WrongRate(), "sample rate"),
        (_Float64(), "float32"),
        (_Stereo(), "mono"),
        (_LossyRoundTrip(), "round trip"),
        (_WrongPromptFormat(), "prompt format"),
        (_RateIgnored(), "rate"),
    ],
)
def test_contract_rejects_broken_engines(engine: FakeEngine, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        run_engine_contract(engine, REF)


def test_contract_requires_the_rate_and_steps_keywords() -> None:
    with pytest.raises(TypeError, match=r"rate|steps"):
        run_engine_contract(_NoOptions(), REF)  # pyright: ignore[reportArgumentType]  # on purpose


def test_contract_propagates_a_failing_second_load() -> None:
    with pytest.raises(RuntimeError, match="second load"):
        run_engine_contract(_NotIdempotent(), REF)
