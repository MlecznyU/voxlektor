"""SettingsStore backed by a TOML file and ``VOXLEKTOR_*`` environment variables."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w

from voxlektor.domain.errors import InvalidDuration, InvalidSetting
from voxlektor.domain.settings import merge_layers, parse_duration

_CONFIG_VERSION = 1
_SETUP_MARKER = "setup_completed"  # not a Settings field: written only by `vxl setup`
_TOP_KEYS = ("engine", "voice", "lang", "base_lang", "rate", "volume", "history")
_SERVICE_KEYS = ("idle_timeout", "autostart", "floor_grace")
_ENV_NAMES = {
    "engine": "VOXLEKTOR_ENGINE",
    "voice": "VOXLEKTOR_VOICE",
    "lang": "VOXLEKTOR_LANG",
    "base_lang": "VOXLEKTOR_BASE_LANG",
    "rate": "VOXLEKTOR_RATE",
    "volume": "VOXLEKTOR_VOLUME",
    "idle_timeout": "VOXLEKTOR_IDLE_TIMEOUT",
    "autostart": "VOXLEKTOR_AUTOSTART",
    "floor_grace": "VOXLEKTOR_FLOOR_GRACE",
}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(key: str, raw: str, source: str) -> str | int | float | bool:
    """Turn a string (env or CLI) into the typed value for ``key``."""
    try:
        match key:
            case "rate":
                return float(raw)
            case "history" | "volume":
                return int(raw)
            case "autostart":
                lowered = raw.strip().lower()
                if lowered in _TRUE:
                    return True
                if lowered in _FALSE:
                    return False
                raise ValueError(raw)
            case "idle_timeout" | "floor_grace":
                digits = raw.isascii() and raw.isdigit()
                if not digits:
                    parse_duration(raw)
                return int(raw) if digits else raw
            case "engine" | "voice" | "lang" | "base_lang":
                if key == "base_lang" and raw not in ("pl", "en"):
                    raise ValueError(raw)
                return raw
            case _:
                raise InvalidSetting(f"unknown setting {key!r}")
    except (ValueError, InvalidDuration):
        raise InvalidSetting(f"{source}: invalid value {raw!r} for {key}") from None


class TomlSettingsStore:
    def __init__(self, path: Path, environ: Mapping[str, str]) -> None:
        self._path = path
        self._environ = environ

    def _load(self) -> dict[str, Any]:
        try:
            with self._path.open("rb") as handle:
                return tomllib.load(handle)
        except FileNotFoundError:
            return {}
        except tomllib.TOMLDecodeError as error:
            raise InvalidSetting(f"{self._path}: {error}") from None

    def read_file(self) -> dict[str, object]:
        data = self._load()
        result: dict[str, object] = {key: data[key] for key in _TOP_KEYS if key in data}
        service = data.get("service", {})
        if not isinstance(service, dict):
            raise InvalidSetting(f"{self._path}: [service] must be a table")
        for key in _SERVICE_KEYS:
            if key in service:
                result[key] = service[key]  # pyright: ignore[reportUnknownMemberType]
        return result

    def read_env(self) -> dict[str, object]:
        return {
            key: _coerce(key, self._environ[name], name)
            for key, name in _ENV_NAMES.items()
            if name in self._environ
        }

    def write_value(self, key: str, value: str) -> None:
        typed = _coerce(key, value, "value")
        merge_layers({key: typed})
        data = self._load()
        data["config_version"] = _CONFIG_VERSION
        if key in _SERVICE_KEYS:
            service = data.setdefault("service", {})
            if not isinstance(service, dict):
                raise InvalidSetting(f"{self._path}: [service] must be a table")
            service[key] = typed  # pyright: ignore[reportUnknownVariableType]
        else:
            data[key] = typed
        self._save(data)

    def setup_completed(self) -> bool:
        """``vxl setup`` has finished once (top-level ``setup_completed = true``)."""
        return self._load().get(_SETUP_MARKER) is True

    def mark_setup_completed(self) -> None:
        """Record a finished setup; every other key and table is kept as it is."""
        data = self._load()
        data["config_version"] = _CONFIG_VERSION
        data[_SETUP_MARKER] = True
        self._save(data)

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_bytes(tomli_w.dumps(data).encode())
        os.replace(tmp, self._path)
