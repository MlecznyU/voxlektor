"""Lightweight wiring for the CLI: settings, platform, protocol client, autostart, voice catalog,
and the daemon-backed sample player of ``vxl setup``.

Imports nothing heavy (no numpy, MLX, audio libraries, asyncio or runtime), so ``vxl`` starts
fast. Settings precedence: CLI flag > ``VOXLEKTOR_*`` env > ``config.toml`` > defaults; a flag
whose value is ``None`` was not given and does not override anything.

``ClientContext.send`` returns the reply's ``data``; an error reply becomes a ``DomainError``
carrying the wire ``code`` and message; an unreachable daemon is ``ServiceUnavailable``, and
``NotRunning`` (its subclass) when nobody listens on the socket at all.
"""

import os
import platform as host
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import msgspec

from voxlektor.adapters.procs.spawn import ensure_daemon
from voxlektor.adapters.storage.settings_toml import TomlSettingsStore
from voxlektor.adapters.storage.voices_fs import FsVoiceRepository
from voxlektor.adapters.transport.unix_client import request
from voxlektor.app.ports import ModelStore, PlatformInfo, Prompter, SettingsStore
from voxlektor.app.voices import VoiceCatalog
from voxlektor.bootstrap.platform_select import select_platform
from voxlektor.contracts import protocol as p
from voxlektor.domain.errors import (
    DomainError,
    InvalidDuration,
    InvalidSetting,
    ServiceUnavailable,
)
from voxlektor.domain.settings import Settings, merge_layers
from voxlektor.domain.types import Lang

_CONFIG_FILE = "config.toml"
_VOICES_DIR = "voices"
_PING_TIMEOUT_S = 1.0
_SAMPLE_POLL_S = 0.2
_SAMPLE_TIMEOUT_S = 120.0
_NOT_WARM = frozenset({"cold", "loading"})

# Replaced by tests (fake clock); production uses the real ones.
_sleep = time.sleep
_monotonic = time.monotonic


@dataclass(frozen=True, slots=True)
class ClientContext:
    settings: Settings
    platform: PlatformInfo
    store: SettingsStore
    config_path: Path

    def send(self, cmd: str, **args: object) -> dict[str, object]:
        """One request to the daemon; the reply's data, or the reply's error raised."""
        raw = request(self.platform.socket_path(), {"cmd": cmd, **args})
        reply = msgspec.convert(raw, p.Reply)
        if not reply.ok:
            if reply.error is None:
                raise DomainError(p.ErrorCode.internal.value, f"{cmd} failed")
            raise DomainError(reply.error.code.value, reply.error.message)
        return dict(reply.data or {})

    def ensure(self) -> None:
        """Make sure a daemon answers, autostarting one if needed."""
        ensure_daemon(self.platform, self._ping)

    def catalog(self) -> VoiceCatalog:
        """The local voice catalog, with the bundled voices installed."""
        catalog = VoiceCatalog(FsVoiceRepository(self.platform.config_dir() / _VOICES_DIR))
        catalog.ensure_bundled()
        return catalog

    def _ping(self) -> bool:
        try:
            reply = request(self.platform.socket_path(), {"cmd": "status"}, _PING_TIMEOUT_S)
        except ServiceUnavailable:
            return False
        return reply.get("ok") is True


def build_client(flags: Mapping[str, object]) -> ClientContext:
    platform, config_path, store = _locate()
    given = {key: value for key, value in flags.items() if value is not None}
    settings = merge_layers(store.read_file(), store.read_env(), given)
    return ClientContext(settings, platform, store, config_path)


def build_setup_client(warn: Callable[[str], None]) -> tuple[ClientContext, Settings]:
    """``build_client({})`` for ``vxl setup``, which is how a broken config gets repaired.

    An invalid value in ``config.toml`` or in a ``VOXLEKTOR_*`` variable is skipped (its
    default is used) and reported through ``warn``. Also returns the setup's current values:
    the file's valid values only, never the env's (Enter in setup must not persist an env value).
    """
    platform, config_path, store = _locate()
    file_values: dict[str, object] = {}
    for key, value in store.read_file().items():
        if (problem := _invalid({key: value})) is not None:
            warn(f"ignoring invalid {key} in {_CONFIG_FILE}: {problem}")
        else:
            file_values[key] = value
    env_values: dict[str, object] = {}
    for name in sorted(os.environ):
        single = TomlSettingsStore(config_path, {name: os.environ[name]})
        try:
            values = single.read_env()  # empty for names that are no setting
        except InvalidSetting as err:
            warn(f"ignoring invalid {name} in the environment: {err.message}")
            continue
        if (problem := _invalid(values)) is not None:
            warn(f"ignoring invalid {name} in the environment: {problem}")
        else:
            env_values.update(values)
    settings = merge_layers(file_values, env_values)
    return ClientContext(settings, platform, store, config_path), merge_layers(file_values)


def model_store() -> ModelStore:
    """The model store of ``vxl setup`` (the Hugging Face cache; imports huggingface_hub)."""
    from voxlektor.adapters.models.hf_cache import HfModelStore  # noqa: PLC0415  # heavy

    return HfModelStore()


def setup_prompter() -> Prompter:
    """The menus of the interactive ``vxl setup`` (imports questionary)."""
    from voxlektor.adapters.terminal.menus import QuestionaryPrompter  # noqa: PLC0415  # heavy

    return QuestionaryPrompter()


def _locate() -> tuple[PlatformInfo, Path, TomlSettingsStore]:
    platform = select_platform(sys.platform, host.machine(), host.mac_ver()[0])
    config_path = platform.config_dir() / _CONFIG_FILE
    return platform, config_path, TomlSettingsStore(config_path, os.environ)


def _invalid(values: Mapping[str, object]) -> str | None:
    """Why ``values`` is not a valid settings layer, or ``None`` when it is."""
    try:
        merge_layers(values)
    except (InvalidSetting, InvalidDuration) as err:
        return err.message
    return None


class DaemonSamplePlayer:
    """SamplePlayer for setup: an ordinary ``speak`` (``now``) through the daemon, then wait.

    The daemon is started when needed (also with ``autostart = false``: setup wants samples).
    ``play`` polls ``status`` every 0.2 s until nothing plays or waits, at most 120 s, and says
    ``loading the model…`` once while the engine is still cold or loading. When nothing is left
    and the daemon reports ``last_error`` (e.g. the model failed to load), that error is raised
    as ``ServiceUnavailable``.
    """

    def __init__(self, ctx: ClientContext) -> None:
        self._ctx = ctx
        self._announced = False

    def play(self, text: str, voice: str, lang: Lang, rate: float, volume: int) -> None:
        ctx = self._ctx
        ctx.ensure()
        ctx.send(
            "speak",
            text=text,
            voice=voice,
            lang=lang,
            base_lang=lang,
            rate=rate,
            volume=volume,
            now=True,
        )
        deadline = _monotonic() + _SAMPLE_TIMEOUT_S
        while True:
            status = ctx.send("status")
            if status.get("playing") is not True and not status.get("queued"):
                # Done, whatever the state: a failed model load leaves the daemon ``cold``.
                error = status.get("last_error")
                if isinstance(error, str) and error:
                    raise ServiceUnavailable(error)
                return
            if status.get("state") in _NOT_WARM and not self._announced:
                print("loading the model…", file=sys.stderr, flush=True)
                self._announced = True
            if _monotonic() >= deadline:
                raise ServiceUnavailable("the sample did not finish playing in time")
            _sleep(_SAMPLE_POLL_S)
