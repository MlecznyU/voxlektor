from pathlib import Path

import numpy as np
import pytest

from voxlektor.contracts.engine import ENGINE_API, Capability, EngineSpec, VoicePrompt, VoiceRef
from voxlektor.testing.engine_contract import run_engine_contract
from voxlektor.testing.fakes import FakeEngine

REF = VoiceRef("v1", Path("/voices/v1/ref.wav"), "reference transcript")


def test_engine_api_version() -> None:
    assert ENGINE_API == 2  # 2 (0.3.0): synthesize(..., *, rate, steps); EngineSpec.api


def test_fake_engine_passes_the_contract_suite() -> None:
    run_engine_contract(FakeEngine(), REF)


def test_spec_describes_the_fake() -> None:
    spec = FakeEngine(sample_rate=16000).spec
    assert spec == EngineSpec(
        id="fake",
        revision="fake0001",
        prompt_format=1,
        sample_rate=16000,
        langs=frozenset({"pl", "en"}),
        capabilities=frozenset({Capability.LIVE, Capability.CLONE}),
        api=2,
    )
    assert EngineSpec("x", "r", 1, 1, frozenset(), frozenset()).api == 1  # older engines


def test_rate_capability_value() -> None:
    assert Capability.RATE == "rate"


def test_fake_with_native_rate_passes_the_contract_suite() -> None:
    engine = FakeEngine(native_rate=True)
    assert Capability.RATE in engine.spec.capabilities
    run_engine_contract(engine, REF)


def test_native_rate_shortens_the_audio_and_options_are_recorded() -> None:
    engine = FakeEngine(sample_rate=1000, seconds_per_char=0.01, native_rate=True)
    engine.load()
    prompt = engine.prepare_voice(REF)
    plain = list(engine.synthesize("a" * 300, "en", prompt))
    fast = list(engine.synthesize("a" * 300, "en", prompt, rate=1.5, steps=8))
    assert sum(len(c.samples) for c in plain) == 3000
    assert sum(len(c.samples) for c in fast) == 2000
    assert engine.synth_options == [(1.0, None), (1.5, 8)]


def test_without_native_rate_the_rate_is_ignored() -> None:
    engine = FakeEngine(sample_rate=1000, seconds_per_char=0.01)
    engine.load()
    chunks = list(engine.synthesize("a" * 300, "en", engine.prepare_voice(REF), rate=1.5))
    assert sum(len(c.samples) for c in chunks) == 3000
    assert engine.synth_options == [(1.5, None)]


def test_synthesize_yields_float32_mono_sine_of_the_expected_length() -> None:
    engine = FakeEngine(sample_rate=1000, seconds_per_char=0.01)
    engine.load()
    chunks = list(engine.synthesize("a" * 250, "en", engine.prepare_voice(REF)))
    samples = np.concatenate([chunk.samples for chunk in chunks])
    assert len(chunks) == 3
    assert all(chunk.samples.dtype == np.float32 and chunk.samples.ndim == 1 for chunk in chunks)
    assert all(chunk.sample_rate == 1000 for chunk in chunks)
    assert len(samples) == 2500
    assert float(np.abs(samples).max()) > 0.0


def test_empty_text_still_yields_one_chunk() -> None:
    engine = FakeEngine()
    engine.load()
    chunks = list(engine.synthesize("", "en", engine.prepare_voice(REF)))
    assert len(chunks) == 1


def test_counters_track_calls() -> None:
    engine = FakeEngine()
    engine.load()
    engine.load()
    prompt = engine.prepare_voice(REF)
    list(engine.synthesize("one", "pl", prompt))
    list(engine.synthesize("two", "en", prompt))
    engine.unload()
    assert (engine.load_calls, engine.unload_calls, engine.prepare_calls) == (2, 1, 1)
    assert engine.synth_calls == [("one", "pl"), ("two", "en")]


def test_fail_on_raises_for_that_text_only() -> None:
    engine = FakeEngine(fail_on="BOOM")
    engine.load()
    prompt = engine.prepare_voice(REF)
    assert list(engine.synthesize("fine", "en", prompt))
    with pytest.raises(RuntimeError, match="BOOM"):
        list(engine.synthesize("this goes BOOM now", "en", prompt))


def test_delay_is_applied_per_synthesis() -> None:
    engine = FakeEngine(delay_s=0.01)
    engine.load()
    assert list(engine.synthesize("x", "en", engine.prepare_voice(REF)))


def test_calls_before_load_are_rejected() -> None:
    engine = FakeEngine()
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.prepare_voice(REF)
    engine.load()
    prompt = engine.prepare_voice(REF)
    engine.unload()
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.synthesize("x", "en", prompt)


def test_unsupported_language_is_rejected() -> None:
    engine = FakeEngine()
    engine.load()
    with pytest.raises(ValueError, match="language"):
        engine.synthesize("x", "de", engine.prepare_voice(REF))


def test_prompt_round_trip_keeps_voice_and_format() -> None:
    engine = FakeEngine()
    engine.load()
    prompt = engine.prepare_voice(REF)
    assert prompt == VoicePrompt("v1", 1, "reference transcript")
    assert engine.deserialize_prompt("v1", engine.serialize_prompt(prompt)) == prompt


@pytest.mark.parametrize("garbage", [b"", b"not a prompt", b"fake-prompt-v1:\xff\xfe"])
def test_deserialize_rejects_garbage(garbage: bytes) -> None:
    with pytest.raises(ValueError, match="prompt"):
        FakeEngine().deserialize_prompt("v1", garbage)


def test_serialize_rejects_foreign_prompt_data() -> None:
    with pytest.raises(TypeError):
        FakeEngine().serialize_prompt(VoicePrompt("v1", 1, 42))
