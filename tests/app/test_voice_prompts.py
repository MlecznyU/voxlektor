from pathlib import Path

from voxlektor.app.voice_prompts import get_or_build_prompt
from voxlektor.contracts.engine import VoicePrompt, VoiceRef
from voxlektor.contracts.voicefmt import prompt_cache_name
from voxlektor.testing.fakes import FakeEngine, InMemoryVoiceRepository

REF = VoiceRef("v1", Path("/voices/v1/ref.wav"), "reference transcript")
KEY = prompt_cache_name("fake", "fake0001", 1)


def _loaded_engine() -> FakeEngine:
    engine = FakeEngine()
    engine.load()
    return engine


def test_builds_once_then_hits_the_cache() -> None:
    engine, repo = _loaded_engine(), InMemoryVoiceRepository([REF])
    first = get_or_build_prompt(engine, repo, REF)
    second = get_or_build_prompt(engine, repo, REF)
    assert first == second == VoicePrompt("v1", 1, "reference transcript")
    assert engine.prepare_calls == 1
    assert repo.cache[("v1", KEY)] == engine.serialize_prompt(first)


def test_corrupt_cache_is_rebuilt_and_overwritten() -> None:
    engine, repo = _loaded_engine(), InMemoryVoiceRepository([REF])
    repo.write_cache("v1", KEY, b"garbage")
    prompt = get_or_build_prompt(engine, repo, REF)
    assert prompt == VoicePrompt("v1", 1, "reference transcript")
    assert engine.prepare_calls == 1
    assert repo.read_cache("v1", KEY) == engine.serialize_prompt(prompt)
    get_or_build_prompt(engine, repo, REF)
    assert engine.prepare_calls == 1


def test_cache_is_keyed_by_engine_revision_and_format() -> None:
    engine, repo = _loaded_engine(), InMemoryVoiceRepository([REF])
    get_or_build_prompt(engine, repo, REF)
    assert list(repo.cache) == [("v1", "fake-fake0001-p1.bin")]
