"""SettingsStore backed by a TOML file and ``VOXLEKTOR_*`` environment variables.

File layout v2: top-level globals (``engine``, ``history``, ``setup_completed``,
``config_version``, ``[service]``) and one ``[profiles.NAME]`` overlay table per profile.
A v1 file (profile keys at top level) reads as the ``default`` profile and is rewritten as
v2 on the next save; reading never rewrites. Unknown keys and tables are always kept.

A mixed file (v2, plus top-level profile keys an older CLI wrote into it) is logged once per
store as a warning on read: the top-level values count only where ``[profiles.default]`` lacks
the key, and the next save moves them there; a top-level value that ``[profiles.default]``
already overrides is dropped then, with a warning naming both values (never silently).
"""

import logging
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w

from voxlektor.domain.errors import InvalidDuration, InvalidSetting
from voxlektor.domain.settings import merge_layers, parse_duration

_CONFIG_VERSION = 2
_SETUP_MARKER = "setup_completed"  # not a Settings field: written only by `vxl setup`
_DEFAULT_PROFILE = "default"
_GLOBAL_KEYS = ("engine", "history")
_PROFILE_KEYS = ("voice", "lang", "base_lang", "rate", "volume", "steps")
_SERVICE_KEYS = ("idle_timeout", "autostart", "floor_grace")
_FILE_ORDER = ("engine", *_PROFILE_KEYS, "history", *_SERVICE_KEYS)  # v1 read_file order
_ENV_NAMES = {
    "engine": "VOXLEKTOR_ENGINE",
    "voice": "VOXLEKTOR_VOICE",
    "lang": "VOXLEKTOR_LANG",
    "base_lang": "VOXLEKTOR_BASE_LANG",
    "rate": "VOXLEKTOR_RATE",
    "volume": "VOXLEKTOR_VOLUME",
    "steps": "VOXLEKTOR_STEPS",
    "idle_timeout": "VOXLEKTOR_IDLE_TIMEOUT",
    "autostart": "VOXLEKTOR_AUTOSTART",
    "floor_grace": "VOXLEKTOR_FLOOR_GRACE",
}
_log = logging.getLogger(__name__)
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(key: str, raw: str, source: str) -> str | int | float | bool:
    """Turn a string (env or CLI) into the typed value for ``key``."""
    try:
        match key:
            case "rate":
                return float(raw)
            case "history" | "volume" | "steps":
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


def _is_v2(data: dict[str, Any]) -> bool:
    return data.get("config_version") == _CONFIG_VERSION or "profiles" in data


def _require_profile_key(key: str) -> None:
    if key not in _PROFILE_KEYS:
        raise InvalidSetting(f"{key!r} is not a profile setting")


class TomlSettingsStore:
    def __init__(self, path: Path, environ: Mapping[str, str]) -> None:
        self._path = path
        self._environ = environ
        self._warned_mixed = False

    def _load(self) -> dict[str, Any]:
        try:
            with self._path.open("rb") as handle:
                return tomllib.load(handle)
        except FileNotFoundError:
            return {}
        except tomllib.TOMLDecodeError as error:
            raise InvalidSetting(f"{self._path}: {error}") from None

    def read_file(self) -> dict[str, object]:
        """Merged view: the globals plus the ``default`` profile's keys."""
        data = self._load()
        merged = self._globals(data) | self._profiles(data)[_DEFAULT_PROFILE]
        return {key: merged[key] for key in _FILE_ORDER if key in merged}

    def read_globals(self) -> dict[str, object]:
        """``engine``, ``history`` and the ``[service]`` keys (flattened)."""
        return self._globals(self._load())

    def read_profiles(self) -> dict[str, dict[str, object]]:
        """Every profile's overlay (known keys only); ``default`` is always present."""
        return self._profiles(self._load())

    def _globals(self, data: dict[str, Any]) -> dict[str, object]:
        result: dict[str, object] = {key: data[key] for key in _GLOBAL_KEYS if key in data}
        service = data.get("service", {})
        if not isinstance(service, dict):
            raise InvalidSetting(f"{self._path}: [service] must be a table")
        for key in _SERVICE_KEYS:
            if key in service:
                result[key] = service[key]  # pyright: ignore[reportUnknownMemberType]
        return result

    def _tables(self, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """The raw ``[profiles]`` tables, validated; v1 top-level keys fold into ``default``."""
        raw: object = data.get("profiles", {})
        if not isinstance(raw, dict):
            raise InvalidSetting(f"{self._path}: [profiles] must be a table")
        tables: dict[str, dict[str, Any]] = {}
        for name, table in raw.items():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(table, dict):
                raise InvalidSetting(f"{self._path}: [profiles.{name}] must be a table")
            tables[str(name)] = dict(table)  # pyright: ignore[reportUnknownArgumentType]
        legacy = {key: data[key] for key in _PROFILE_KEYS if key in data}
        if legacy and _is_v2(data) and not self._warned_mixed:
            self._warned_mixed = True
            _log.warning(
                "%s: top-level %s next to [profiles] (written by an older voxlektor?); they "
                "count only where [profiles.default] lacks them and move there on the next save",
                self._path,
                ", ".join(sorted(legacy)),
            )
        tables[_DEFAULT_PROFILE] = legacy | tables.get(_DEFAULT_PROFILE, {})
        return tables

    def _profiles(self, data: dict[str, Any]) -> dict[str, dict[str, object]]:
        return {
            name: {key: table[key] for key in _PROFILE_KEYS if key in table}
            for name, table in self._tables(data).items()
        }

    def _migrated(self) -> dict[str, Any]:
        """The file as v2 data: v1 top-level profile keys moved into ``[profiles.default]``."""
        data = self._load()
        tables = self._tables(data)
        for key in _PROFILE_KEYS:
            if key in data and tables[_DEFAULT_PROFILE].get(key) != data[key]:
                _log.warning(
                    "%s: dropped top-level %s = %r; [profiles.default] has %s = %r",
                    self._path,
                    key,
                    data[key],
                    key,
                    tables[_DEFAULT_PROFILE][key],
                )
            data.pop(key, None)
        if not tables[_DEFAULT_PROFILE] and _DEFAULT_PROFILE not in data.get("profiles", {}):
            del tables[_DEFAULT_PROFILE]  # implicit: no empty [profiles.default] is written
        data["config_version"] = _CONFIG_VERSION
        if tables:
            data["profiles"] = tables
        return data

    def read_env(self) -> dict[str, object]:
        return {
            key: _coerce(key, self._environ[name], name)
            for key, name in _ENV_NAMES.items()
            if name in self._environ
        }

    def write_value(self, key: str, value: str) -> None:
        """Write one setting; a profile key goes into ``[profiles.default]``."""
        if key in _PROFILE_KEYS:
            self.write_profile_value(_DEFAULT_PROFILE, key, value)
            return
        typed = _coerce(key, value, "value")
        merge_layers({key: typed})
        data = self._migrated()
        if key in _SERVICE_KEYS:
            service = data.setdefault("service", {})
            if not isinstance(service, dict):
                raise InvalidSetting(f"{self._path}: [service] must be a table")
            service[key] = typed  # pyright: ignore[reportUnknownVariableType]
        else:
            data[key] = typed
        self._save(data)

    def write_profile_value(self, profile: str, key: str, value: str) -> None:
        """Set ``key`` in ``[profiles.PROFILE]`` (created when missing)."""
        _require_profile_key(key)
        typed = _coerce(key, value, "value")
        merge_layers({key: typed})
        data = self._migrated()
        data.setdefault("profiles", {}).setdefault(profile, {})[key] = typed
        self._save(data)

    def create_profile(self, profile: str) -> None:
        """Add an empty ``[profiles.PROFILE]``; nothing is written when it exists (or is
        ``default``, which always exists)."""
        if profile in self.read_profiles():
            return
        data = self._migrated()
        data.setdefault("profiles", {})[profile] = {}
        self._save(data)

    def unset_profile_value(self, profile: str, key: str) -> None:
        """Drop ``key`` from a profile; the profile itself stays."""
        _require_profile_key(key)
        data = self._migrated()
        tables: dict[str, dict[str, Any]] = data.get("profiles", {})
        if profile != _DEFAULT_PROFILE and profile not in tables:
            raise InvalidSetting(f"unknown profile {profile!r}")
        tables.get(profile, {}).pop(key, None)
        self._save(data)

    def remove_profile(self, profile: str) -> None:
        """Delete a whole profile; ``default`` cannot be removed."""
        if profile == _DEFAULT_PROFILE:
            raise InvalidSetting(f"profile {profile!r} cannot be removed")
        data = self._migrated()
        tables: dict[str, dict[str, Any]] = data.get("profiles", {})
        if profile not in tables:
            raise InvalidSetting(f"unknown profile {profile!r}")
        del tables[profile]
        self._save(data)

    def setup_completed(self) -> bool:
        """``vxl setup`` has finished once (top-level ``setup_completed = true``)."""
        return self._load().get(_SETUP_MARKER) is True

    def mark_setup_completed(self) -> None:
        """Record a finished setup; every other key and table is kept as it is."""
        data = self._migrated()
        data[_SETUP_MARKER] = True
        self._save(data)

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_bytes(tomli_w.dumps(data).encode())
        os.replace(tmp, self._path)
