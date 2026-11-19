"""Ports: what the application needs from the outside world."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from voxlektor.app.effects import Cue
from voxlektor.contracts.engine import Engine, ModelInfo, VoiceRef
from voxlektor.domain.types import ChunkId, Lang


class Clock(Protocol):
    def now(self) -> float: ...


class AudioOutput(Protocol):
    """Plays one chunk at a time; completion is reported through callbacks set once.

    ``samples`` is opaque to the application (a numpy array in production). The callbacks may
    be invoked from any thread; the runtime wraps them to hop back onto the loop.

    ``lead_s`` is the pause to put before the chunk, as a MINIMUM gap: silence already heard
    since the previous chunk's last sample (not counting time paused) counts toward it.
    """

    def set_callbacks(
        self,
        on_done: Callable[[ChunkId], None],
        on_failed: Callable[[ChunkId, str], None],
    ) -> None: ...

    def start(
        self, chunk: ChunkId, samples: object, sample_rate: int, *, lead_s: float = 0.0
    ) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def abort(self) -> None: ...


class CuePlayer(Protocol):
    """Plays a short cue next to the reading (also while it is paused); never raises.

    ``volume`` (0..100, 50 = the even level) is the volume of the reading the cue belongs to.
    """

    def play(self, cue: Cue, volume: int = 50) -> None: ...


class SettingsStore(Protocol):
    def read_file(self) -> dict[str, object]:
        """Merged file view: the globals plus the ``default`` profile's keys."""
        ...

    def read_env(self) -> dict[str, object]: ...

    def write_value(self, key: str, value: str) -> None:
        """Write one setting; a profile key goes into the ``default`` profile."""
        ...

    def read_globals(self) -> dict[str, object]:
        """The non-profile settings (``engine``, ``history``, ``[service]`` keys)."""
        ...

    def read_profiles(self) -> dict[str, dict[str, object]]:
        """Every profile's overlay by name; ``default`` is always present."""
        ...

    def write_profile_value(self, profile: str, key: str, value: str) -> None:
        """Set a profile key (``InvalidSetting`` for a bad value or a non-profile key)."""
        ...

    def create_profile(self, profile: str) -> None:
        """Make sure a profile exists (an empty overlay when new); a no-op when it does."""
        ...

    def unset_profile_value(self, profile: str, key: str) -> None:
        """Drop a profile key; ``InvalidSetting`` for an unknown profile."""
        ...

    def remove_profile(self, profile: str) -> None:
        """Delete a profile; ``InvalidSetting`` for ``default`` or an unknown profile."""
        ...

    def setup_completed(self) -> bool: ...

    def mark_setup_completed(self) -> None:
        """Record that ``vxl setup`` finished; keeps every other key and table."""
        ...


@dataclass(frozen=True, slots=True)
class Option:
    """One menu entry of ``Prompter.choose``: ``value`` is returned, ``label`` is shown.

    Re-exported by ``app.setup`` (its documented home): defined here because ``app.setup``
    needs these ports, and pyright reports an import cycle even for typing-only imports.
    """

    value: str
    label: str


class Prompter(Protocol):
    """Asks the user during setup; an answer method raises ``SetupCancelled`` on cancel."""

    def choose(self, question: str, options: list[Option], default: str) -> str: ...

    def number(self, question: str, low: float, high: float, default: float) -> float: ...

    def confirm(self, question: str, default: bool) -> bool: ...

    def say(self, line: str) -> None: ...


class SamplePlayer(Protocol):
    def play(self, text: str, voice: str, lang: Lang, rate: float, volume: int) -> None:
        """Read ``text`` aloud and return once it has been heard."""
        ...


class ModelStore(Protocol):
    def is_present(self, info: ModelInfo) -> bool: ...

    def download(self, info: ModelInfo, progress: Callable[[int, int], None]) -> None:
        """Fetch the model, reporting ``progress(done, total)`` in bytes.

        Raises ``ServiceUnavailable`` when the download fails.
        """
        ...


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
