"""Profile resolution: flags > the selected profile's own keys > env > default > built-ins.

The ``VOXLEKTOR_*`` env acts as an override of ``default``: it wins over the ``default``
profile (and the file's globals) but never over a key the selected profile sets explicitly.
For the ``default`` profile itself that is flags > env > default > built-ins.
"""

import re
from collections.abc import Mapping
from typing import Literal

from voxlektor.domain.errors import InvalidSetting
from voxlektor.domain.settings import Settings, merge_layers

type Source = Literal["flag", "env", "profile", "default", "builtin"]

PROFILE_KEYS: tuple[str, ...] = ("voice", "lang", "base_lang", "rate", "volume", "steps")
GLOBAL_KEYS: tuple[str, ...] = ("engine", "history", "idle_timeout", "autostart", "floor_grace")
DEFAULT_PROFILE = "default"

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}", re.ASCII)


def valid_profile_name(name: str) -> bool:
    """True for ``^[a-z0-9][a-z0-9_-]{0,31}$`` (no trailing newline)."""
    return _NAME.fullmatch(name) is not None


def _only(layer: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    return {k: v for k, v in layer.items() if k in keys}


def resolve(
    *,
    profile: str,
    profiles: Mapping[str, Mapping[str, object]],
    globals_: Mapping[str, object],
    env: Mapping[str, object],
    flags: Mapping[str, object],
) -> tuple[Settings, dict[str, Source]]:
    """Resolve settings and the source of every profile and global key (precedence: see the
    module doc).

    Global keys come from ``globals_`` (source ``default``, the file), env and flags, never
    from a profile table. Keys unknown to both sets in env or flags are ignored.
    """
    if profile != DEFAULT_PROFILE and profile not in profiles:
        raise InvalidSetting(f"unknown profile {profile!r}")
    known = PROFILE_KEYS + GLOBAL_KEYS
    default_layer = _only(profiles.get(DEFAULT_PROFILE, {}), PROFILE_KEYS)
    selected = _only(profiles.get(profile, {}), PROFILE_KEYS) if profile != DEFAULT_PROFILE else {}
    file_globals = _only(globals_, GLOBAL_KEYS)
    env_layer = _only(env, known)
    flag_layer = _only(flags, known)
    settings = merge_layers(default_layer, file_globals, env_layer, selected, flag_layer)
    sources: dict[str, Source] = {}
    for key in known:
        if key in flag_layer:
            sources[key] = "flag"
        elif key in selected:
            sources[key] = "profile"
        elif key in env_layer:
            sources[key] = "env"
        elif key in default_layer or key in file_globals:
            sources[key] = "default"
        else:
            sources[key] = "builtin"
    return settings, sources
