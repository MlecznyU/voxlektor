"""Layered settings: defaults < file < env < flags."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from voxlektor.domain.errors import InvalidDuration, InvalidSetting
from voxlektor.domain.types import EngineId, LangPref, VoiceName

_DURATION = re.compile(r"([0-9]+)([smh])", re.ASCII)
_UNITS = {"s": 1, "m": 60, "h": 3600}
_LANGS: tuple[LangPref, ...] = ("auto", "pl", "en")
_RATE_MIN = 0.5
_RATE_MAX = 3.0
_DEFAULT_ENGINE = EngineId("omnivoice")
_DEFAULT_VOICE = VoiceName("v0-default")


@dataclass(frozen=True, slots=True)
class Settings:
    engine: EngineId = _DEFAULT_ENGINE
    voice: VoiceName = _DEFAULT_VOICE
    lang: LangPref = "auto"
    rate: float = 1.0
    idle_timeout_s: int = 600
    autostart: bool = True
    history: int = 20
    ahead: int = 2


def parse_duration(s: str) -> int:
    """Parse ``"0"``, ``"90s"``, ``"10m"`` or ``"1h"`` into seconds."""
    if s == "0":
        return 0
    match = _DURATION.fullmatch(s)
    if match is None:
        raise InvalidDuration(f"invalid duration {s!r}; use 0, 90s, 10m or 1h")
    return int(match[1]) * _UNITS[match[2]]


def _text(key: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSetting(f"{key} must be a non-empty string")
    return value


def _positive_int(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidSetting(f"{key} must be a positive integer")
    return value


def _rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidSetting("rate must be a number")
    if not _RATE_MIN <= value <= _RATE_MAX:
        raise InvalidSetting(f"rate must be between {_RATE_MIN} and {_RATE_MAX}")
    return float(value)


def _lang(value: object) -> LangPref:
    for lang in _LANGS:
        if value == lang:
            return lang
    raise InvalidSetting("lang must be one of auto, pl, en")


def _idle_timeout(value: object) -> int:
    if isinstance(value, bool):
        raise InvalidSetting("idle_timeout must be a duration or seconds")
    if isinstance(value, str):
        return parse_duration(value)
    if isinstance(value, int) and value >= 0:
        return value
    raise InvalidSetting("idle_timeout must be a duration or non-negative seconds")


def _autostart(value: object) -> bool:
    if not isinstance(value, bool):
        raise InvalidSetting("autostart must be true or false")
    return value


def merge_layers(*layers: Mapping[str, object]) -> Settings:
    """Merge layers (later wins) over the defaults and validate the result."""
    merged: dict[str, object] = {}
    for layer in layers:
        merged.update(layer)
    base = Settings()
    engine, voice, lang = base.engine, base.voice, base.lang
    rate, idle_timeout_s = base.rate, base.idle_timeout_s
    autostart, history = base.autostart, base.history
    for key, value in merged.items():
        match key:
            case "engine":
                engine = EngineId(_text(key, value))
            case "voice":
                voice = VoiceName(_text(key, value))
            case "lang":
                lang = _lang(value)
            case "rate":
                rate = _rate(value)
            case "history":
                history = _positive_int(key, value)
            case "idle_timeout":
                idle_timeout_s = _idle_timeout(value)
            case "autostart":
                autostart = _autostart(value)
            case _:
                raise InvalidSetting(f"unknown setting {key!r}")
    return Settings(
        engine=engine,
        voice=voice,
        lang=lang,
        rate=rate,
        idle_timeout_s=idle_timeout_s,
        autostart=autostart,
        history=history,
    )
