"""Lightweight wiring for the CLI: settings, platform, protocol client, autostart, voice catalog.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import msgspec

from voxlektor.adapters.procs.spawn import ensure_daemon
from voxlektor.adapters.storage.settings_toml import TomlSettingsStore
from voxlektor.adapters.storage.voices_fs import FsVoiceRepository
from voxlektor.adapters.transport.unix_client import request
from voxlektor.app.ports import PlatformInfo, SettingsStore
from voxlektor.app.voices import VoiceCatalog
from voxlektor.bootstrap.platform_select import select_platform
from voxlektor.contracts import protocol as p
from voxlektor.domain.errors import DomainError, ServiceUnavailable
from voxlektor.domain.settings import Settings, merge_layers

_CONFIG_FILE = "config.toml"
_VOICES_DIR = "voices"
_PING_TIMEOUT_S = 1.0


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
    platform = select_platform(sys.platform, host.machine(), host.mac_ver()[0])
    config_path = platform.config_dir() / _CONFIG_FILE
    store = TomlSettingsStore(config_path, os.environ)
    given = {key: value for key, value in flags.items() if value is not None}
    settings = merge_layers(store.read_file(), store.read_env(), given)
    return ClientContext(settings, platform, store, config_path)
