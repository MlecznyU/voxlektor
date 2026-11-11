from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from voxlektor.contracts.engine import AudioChunk, VoicePrompt, VoiceRef
from voxlektor.testing.engine_contract import run_engine_contract
from voxlektor.testing.fakes import FakeEngine

REF = VoiceRef("v1", Path("/voices/v1/ref.wav"), "reference transcript")


class _Silent(FakeEngine):
    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        return iter(())


class _WrongRate(FakeEngine):
    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        yield AudioChunk(np.zeros(100, np.float32), self.spec.sample_rate + 1)


class _Float64(FakeEngine):
    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        wide_dtype: str = "float64"  # a plain str keeps numpy from inferring float64 statically
        yield AudioChunk(np.zeros(100, wide_dtype), self.spec.sample_rate)


class _Stereo(FakeEngine):
    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        yield AudioChunk(np.zeros((100, 2), np.float32), self.spec.sample_rate)


class _LossyRoundTrip(FakeEngine):
    """Audio length follows the prompt text, and deserialization throws that text away."""

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        yield AudioChunk(np.zeros(1000 * len(str(prompt.data)), np.float32), self.spec.sample_rate)

    def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt:
        return VoicePrompt(voice, self.PROMPT_FORMAT, "x")


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
    ],
)
def test_contract_rejects_broken_engines(engine: FakeEngine, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        run_engine_contract(engine, REF)


def test_contract_propagates_a_failing_second_load() -> None:
    with pytest.raises(RuntimeError, match="second load"):
        run_engine_contract(_NotIdempotent(), REF)
