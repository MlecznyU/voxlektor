from pathlib import Path

import pytest

from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import UnknownVoice
from voxlektor.testing.fakes import InMemoryVoiceRepository

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
