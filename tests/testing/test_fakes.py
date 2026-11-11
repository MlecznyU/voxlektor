from pathlib import Path

import pytest

from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import UnknownVoice
from voxlektor.testing.fakes import FakeEngine, InMemoryVoiceRepository

A = VoiceRef("a", Path("/v/a/ref.wav"), "text a")
B = VoiceRef("b", Path("/v/b/ref.wav"), "text b")


def test_names_are_sorted() -> None:
    assert InMemoryVoiceRepository([B, A]).names() == ["a", "b"]


def test_empty_repository_has_no_voices() -> None:
    assert InMemoryVoiceRepository().names() == []


def test_ref_returns_the_stored_voice() -> None:
    assert InMemoryVoiceRepository([A]).ref("a") == A


def test_unknown_voice_lists_available_names() -> None:
    with pytest.raises(UnknownVoice) as info:
        InMemoryVoiceRepository([A, B]).ref("zzz")
    assert info.value.available == ("a", "b")


def test_cache_round_trip_is_per_voice_and_key() -> None:
    repo = InMemoryVoiceRepository([A, B])
    assert repo.read_cache("a", "k.bin") is None
    repo.write_cache("a", "k.bin", b"one")
    repo.write_cache("a", "k.bin", b"two")
    assert repo.read_cache("a", "k.bin") == b"two"
    assert repo.read_cache("b", "k.bin") is None
    assert repo.read_cache("a", "other.bin") is None


def test_install_bundled_is_counted() -> None:
    repo = InMemoryVoiceRepository()
    repo.install_bundled()
    repo.install_bundled()
    assert repo.install_calls == 2


def _synth(engine: FakeEngine, text: str) -> int:
    prompt = engine.prepare_voice(A)
    return len(list(engine.synthesize(text, "en", prompt)))


def test_fail_on_without_fail_times_always_fails() -> None:
    engine = FakeEngine(fail_on="BOOM")
    engine.load()
    for _ in range(3):
        with pytest.raises(RuntimeError, match="fake engine failure on 'BOOM'"):
            _synth(engine, "a BOOM b")
    assert _synth(engine, "fine") == 1


def test_fail_times_fails_only_the_first_matching_syntheses() -> None:
    engine = FakeEngine(fail_on="BOOM", fail_times=2)
    engine.load()
    assert _synth(engine, "fine") == 1  # not matching: does not use up a failure
    for _ in range(2):
        with pytest.raises(RuntimeError, match="fake engine failure"):
            _synth(engine, "BOOM")
    assert _synth(engine, "BOOM") == 1


def test_fail_prepare_fails_the_first_prepares() -> None:
    engine = FakeEngine(fail_prepare=1)
    engine.load()
    with pytest.raises(RuntimeError, match="fake prepare failure"):
        engine.prepare_voice(A)
    assert engine.prepare_voice(A).voice == "a"
    assert engine.prepare_calls == 1
