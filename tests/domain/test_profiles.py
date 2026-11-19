import pytest

from voxlektor.domain.errors import InvalidSetting
from voxlektor.domain.profiles import (
    DEFAULT_PROFILE,
    GLOBAL_KEYS,
    PROFILE_KEYS,
    resolve,
    valid_profile_name,
)

PROFILES = {
    "default": {"voice": "d-voice", "rate": 1.1, "steps": 8},
    "calm": {"voice": "p-voice", "rate": 0.9},
}


def test_key_sets() -> None:
    assert PROFILE_KEYS == ("voice", "lang", "base_lang", "rate", "volume", "steps")
    assert "engine" in GLOBAL_KEYS
    assert "history" in GLOBAL_KEYS
    assert not set(PROFILE_KEYS) & set(GLOBAL_KEYS)
    assert DEFAULT_PROFILE == "default"


def test_builtins_only() -> None:
    settings, sources = resolve(profile="default", profiles={}, globals_={}, env={}, flags={})
    assert settings.voice == "v0-default"
    assert sources["voice"] == "builtin"
    assert sources["engine"] == "builtin"


def test_precedence_per_key() -> None:
    # flags > the selected profile's own keys > env > default > built-ins
    settings, sources = resolve(
        profile="calm",
        profiles=PROFILES,
        globals_={},
        env={"rate": 1.5, "volume": 70, "steps": 12},
        flags={"volume": 20},
    )
    assert settings.volume == 20
    assert sources["volume"] == "flag"
    assert settings.rate == 0.9  # the profile sets it: the env does not override it
    assert sources["rate"] == "profile"
    assert settings.voice == "p-voice"
    assert sources["voice"] == "profile"
    assert settings.steps == 12  # the profile does not set it: env beats default
    assert sources["steps"] == "env"
    assert sources["lang"] == "builtin"
    _, inherited = resolve(profile="calm", profiles=PROFILES, globals_={}, env={}, flags={})
    assert inherited["steps"] == "default"


def test_flag_beats_the_profile_value() -> None:
    settings, sources = resolve(
        profile="calm", profiles=PROFILES, globals_={}, env={"rate": 1.5}, flags={"rate": 2.0}
    )
    assert (settings.rate, sources["rate"]) == (2.0, "flag")


def test_for_the_default_profile_env_beats_its_values() -> None:
    settings, sources = resolve(
        profile="default", profiles=PROFILES, globals_={}, env={"rate": 1.5}, flags={}
    )
    assert (settings.rate, sources["rate"]) == (1.5, "env")
    assert (settings.voice, sources["voice"]) == ("d-voice", "default")


def test_default_profile_selected_reports_default() -> None:
    settings, sources = resolve(profile="default", profiles=PROFILES, globals_={}, env={}, flags={})
    assert settings.voice == "d-voice"
    assert sources["voice"] == "default"


def test_profile_without_table_entry_for_selected_default() -> None:
    _, sources = resolve(profile="default", profiles={"x": {}}, globals_={}, env={}, flags={})
    assert sources["voice"] == "builtin"


def test_unknown_profile() -> None:
    with pytest.raises(InvalidSetting, match="unknown profile 'nope'"):
        resolve(profile="nope", profiles=PROFILES, globals_={}, env={}, flags={})


def test_globals_never_from_profile_table() -> None:
    settings, sources = resolve(
        profile="calm",
        profiles={"calm": {"engine": "evil", "history": 99}},
        globals_={"history": 5},
        env={},
        flags={},
    )
    assert settings.history == 5
    assert sources["history"] == "default"
    assert settings.engine == "omnivoice"
    assert sources["engine"] == "builtin"


def test_globals_env_and_flags_override() -> None:
    settings, sources = resolve(
        profile="default",
        profiles={},
        globals_={"history": 5, "autostart": False},
        env={"history": 7, "idle_timeout": "1m"},
        flags={"history": 9},
    )
    assert settings.history == 9
    assert sources["history"] == "flag"
    assert settings.idle_timeout_s == 60
    assert sources["idle_timeout"] == "env"
    assert sources["autostart"] == "default"


def test_unrelated_env_keys_ignored() -> None:
    settings, _ = resolve(
        profile="default", profiles={}, globals_={}, env={"socket": "/x"}, flags={"other": 1}
    )
    assert settings.rate == 1.0


def test_invalid_value_raises() -> None:
    with pytest.raises(InvalidSetting):
        resolve(
            profile="default", profiles={"default": {"rate": 99}}, globals_={}, env={}, flags={}
        )


@pytest.mark.parametrize("name", ["default", "a", "0x", "a_b-c", "a" * 32])
def test_valid_names(name: str) -> None:
    assert valid_profile_name(name)


@pytest.mark.parametrize("name", ["", "A", "-a", "_a", "a" * 33, "a b", "ą", "a\n"])
def test_invalid_names(name: str) -> None:
    assert not valid_profile_name(name)
