import pytest
from hypothesis import given
from hypothesis import strategies as st

from voxlektor.domain.errors import (
    DomainError,
    InvalidDuration,
    InvalidSetting,
    PlatformError,
    UnknownVoice,
)
from voxlektor.domain.settings import Settings, merge_layers, parse_duration
from voxlektor.domain.types import EngineId, VoiceName


def test_defaults() -> None:
    settings = merge_layers()
    assert settings == Settings()
    assert (settings.engine, settings.voice, settings.lang) == ("omnivoice", "v0-default", "auto")
    assert (settings.rate, settings.idle_timeout_s, settings.autostart) == (1.0, 600, True)
    assert (settings.history, settings.ahead) == (20, 2)


def test_precedence_defaults_file_env_flags() -> None:
    file = {"voice": "from-file", "rate": 1.1}
    env = {"voice": "from-env", "rate": 1.2}
    flags = {"voice": "from-flags"}
    assert merge_layers().voice == "v0-default"
    assert merge_layers(file).voice == "from-file"
    assert merge_layers(file, env).voice == "from-env"
    merged = merge_layers(file, env, flags)
    assert merged.voice == "from-flags"
    assert merged.rate == 1.2


def test_all_keys_are_applied() -> None:
    settings = merge_layers(
        {
            "engine": "fake",
            "voice": "ola",
            "lang": "pl",
            "rate": 2,
            "history": 5,
            "idle_timeout": "1h",
            "autostart": False,
        }
    )
    assert settings == Settings(
        engine=EngineId("fake"),
        voice=VoiceName("ola"),
        lang="pl",
        rate=2.0,
        history=5,
        idle_timeout_s=3600,
        autostart=False,
    )


def test_idle_timeout_accepts_seconds_int() -> None:
    assert merge_layers({"idle_timeout": 0}).idle_timeout_s == 0
    assert merge_layers({"idle_timeout": 90}).idle_timeout_s == 90


@pytest.mark.parametrize(
    "layer",
    [
        {"rate": 5},
        {"rate": 0.4},
        {"rate": "fast"},
        {"rate": True},
        {"lang": "de"},
        {"history": 0},
        {"history": 1.5},
        {"history": True},
        {"engine": ""},
        {"voice": 3},
        {"idle_timeout": -1},
        {"idle_timeout": 1.5},
        {"idle_timeout": True},
        {"autostart": "yes"},
        {"nonsense": 1},
        {"engine": None},
        {"voice": ["a"]},
        {"lang": None},
        {"rate": None},
        {"rate": [1.0]},
        {"history": None},
        {"idle_timeout": None},
        {"idle_timeout": ["10m"]},
        {"autostart": None},
        {"autostart": 1},
    ],
)
def test_invalid_settings_raise(layer: dict[str, object]) -> None:
    with pytest.raises(InvalidSetting) as info:
        merge_layers(layer)
    assert info.value.code == "invalid_setting"


def test_invalid_duration_in_layer_raises() -> None:
    with pytest.raises(InvalidDuration):
        merge_layers({"idle_timeout": "soon"})


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("0", 0), ("90s", 90), ("10m", 600), ("1h", 3600), ("0s", 0), ("2h", 7200)],
)
def test_parse_duration(text: str, seconds: int) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["abc", "-5m", "10x", "", "10", "m", "1.5h", " 5m", "5 m"])
def test_parse_duration_rejects(text: str) -> None:
    with pytest.raises(InvalidDuration) as info:
        parse_duration(text)
    assert info.value.code == "invalid_duration"


@given(st.integers(min_value=0, max_value=10_000))
def test_parse_duration_minutes_property(n: int) -> None:
    assert parse_duration(f"{n}m") == 60 * n


def test_errors_carry_codes() -> None:
    unknown = UnknownVoice("zed", ("a", "b"))
    assert (unknown.code, unknown.name, unknown.available) == ("unknown_voice", "zed", ("a", "b"))
    platform = PlatformError()
    assert platform.code == "unsupported_platform"
    assert platform.message == "platform not supported yet"
    assert str(platform) == "platform not supported yet"
    assert isinstance(platform, DomainError)
