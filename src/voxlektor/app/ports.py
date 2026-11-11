"""Ports: what the application needs from the outside world."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from voxlektor.contracts.engine import Engine, VoiceRef
from voxlektor.domain.types import ChunkId


class Clock(Protocol):
    def now(self) -> float: ...


class AudioOutput(Protocol):
    """Plays one chunk at a time; completion is reported through callbacks set once.

    ``samples`` is opaque to the application (a numpy array in production). The callbacks may
    be invoked from any thread; the runtime wraps them to hop back onto the loop.
    """

    def set_callbacks(
        self,
        on_done: Callable[[ChunkId], None],
        on_failed: Callable[[ChunkId, str], None],
    ) -> None: ...

    def start(self, chunk: ChunkId, samples: object, sample_rate: int) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def abort(self) -> None: ...


class SettingsStore(Protocol):
    def read_file(self) -> dict[str, object]: ...

    def read_env(self) -> dict[str, object]: ...

    def write_value(self, key: str, value: str) -> None: ...


class PlatformInfo(Protocol):
    def config_dir(self) -> Path: ...

    def runtime_dir(self) -> Path: ...

    def socket_path(self) -> Path: ...

    def log_path(self) -> Path: ...

    def pid_path(self) -> Path: ...


class VoiceRepository(Protocol):
    def names(self) -> list[str]: ...

    def ref(self, name: str) -> VoiceRef: ...

    def read_cache(self, name: str, key: str) -> bytes | None: ...

    def write_cache(self, name: str, key: str, data: bytes) -> None: ...

    def install_bundled(self) -> None:
        """Copy bundled voices that are missing; never overwrite anything."""
        ...


class EngineProvider(Protocol):
    def __call__(self, engine_id: str) -> Engine: ...
