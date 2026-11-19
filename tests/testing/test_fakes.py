from collections.abc import Callable
from pathlib import Path

import pytest

from voxlektor.app.ports import Option
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import ServiceUnavailable, UnknownVoice
from voxlektor.testing.fakes import (
    ENTER,
    FAKE_MODEL,
    FakeEngine,
    FakeModelStore,
    FakeSamplePlayer,
    InMemorySettingsStore,
    InMemoryVoiceRepository,
    ScriptedPrompter,
)

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


def test_scripted_prompter_answers_in_order_then_presses_enter() -> None:
    prompter = ScriptedPrompter(["b", 2, True, ENTER])
    options = [Option("a", "A"), Option("b", "B")]
    assert prompter.choose("pick", options, "a") == "b"
    assert prompter.number("n", 0, 10, 5) == 2.0
    assert prompter.confirm("ok?", default=False) is True
    assert prompter.confirm("again?", default=False) is False  # ENTER
    assert prompter.choose("pick", options, "a") == "a"  # script exhausted
    prompter.say("hello")
    assert prompter.said == ["hello"]
    assert prompter.options == [options, options]
    assert [kind for kind, _, _ in prompter.asked] == [
        "choose",
        "number",
        "confirm",
        "confirm",
        "choose",
    ]
    assert repr(ENTER) == "ENTER"


@pytest.mark.parametrize(
    ("method", "answer"),
    [("choose", 1), ("number", True), ("number", "x"), ("confirm", "yes")],
)
def test_scripted_prompter_rejects_mistyped_answers(method: str, answer: object) -> None:
    prompter = ScriptedPrompter([answer])
    ask: dict[str, Callable[[], object]] = {
        "choose": lambda: prompter.choose("q", [], "d"),
        "number": lambda: prompter.number("q", 0, 1, 0),
        "confirm": lambda: prompter.confirm("q", default=True),
    }
    with pytest.raises(TypeError, match="scripted answer"):
        ask[method]()


def test_scripted_prompter_raises_scripted_exceptions() -> None:
    prompter = ScriptedPrompter([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        prompter.confirm("q", default=True)


def test_fake_model_store_downloads_and_fails() -> None:
    store = FakeModelStore()
    seen: list[tuple[int, int]] = []
    assert store.is_present(FAKE_MODEL) is False
    store.download(FAKE_MODEL, lambda done, total: seen.append((done, total)))
    assert seen == [(0, 10), (5, 10), (10, 10)]
    assert store.is_present(FAKE_MODEL) is True
    failing = FakeModelStore(steps=[], fail=ServiceUnavailable("offline"))
    with pytest.raises(ServiceUnavailable):
        failing.download(FAKE_MODEL, lambda done, total: None)
    assert failing.is_present(FAKE_MODEL) is False
    assert failing.downloads == ["fake"]


def test_in_memory_settings_store_records_writes() -> None:
    store = InMemorySettingsStore({"voice": "a"}, {"rate": 2.0})
    store.write_value("voice", "b")
    assert store.read_file() == {"voice": "b"}
    assert store.read_env() == {"rate": 2.0}
    assert store.writes == [("voice", "b")]
    assert store.setup_completed() is False
    store.mark_setup_completed()
    assert store.setup_completed() is True


def test_in_memory_settings_store_profiles() -> None:
    store = InMemorySettingsStore({"engine": "fake", "voice": "a"}, profiles={"c": {"rate": 1.1}})
    assert store.read_globals() == {"engine": "fake"}
    assert store.read_profiles() == {"default": {"voice": "a"}, "c": {"rate": 1.1}}
    store.write_profile_value("default", "lang", "pl")
    store.write_profile_value("n", "rate", "2")
    store.unset_profile_value("default", "voice")
    store.unset_profile_value("c", "rate")
    assert store.read_file() == {"engine": "fake", "lang": "pl"}
    assert store.read_profiles() == {"default": {"lang": "pl"}, "c": {}, "n": {"rate": "2"}}
    store.create_profile("default")
    store.create_profile("n")
    store.create_profile("e")
    assert store.profiles["e"] == {}
    assert store.profiles["n"] == {"rate": "2"}
    store.remove_profile("e")
    store.remove_profile("c")
    assert list(store.read_profiles()) == ["default", "n"]
    assert store.profile_writes == [("default", "lang", "pl"), ("n", "rate", "2")]
    assert store.writes == []


def test_fake_sample_player_records() -> None:
    player = FakeSamplePlayer()
    player.play("hi", "a", "en", 1.0, 50)
    assert player.played == [("hi", "a", "en", 1.0, 50)]
