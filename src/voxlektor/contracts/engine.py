"""Engine SPI v1: what an engine adapter implements and what flows through it."""

from __future__ import annotations

import typing
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

ENGINE_API = 1


class Capability(StrEnum):
    LIVE = "live"
    RENDER = "render"
    CLONE = "clone"
    DESIGN = "design"


@dataclass(frozen=True)
class EngineSpec:
    """What an engine is and can do; ``revision`` pins the model so caches stay valid."""

    id: str
    revision: str
    prompt_format: int
    sample_rate: int
    langs: frozenset[str]
    capabilities: frozenset[Capability]


@dataclass(frozen=True)
class ModelInfo:
    """The model an engine needs, described without importing the engine (setup reads it).

    Published under the ``voxlektor.engine_models`` entry-point group by a light module next
    to the engine. ``repo == ""`` means the engine needs no download; ``hidden`` engines (test
    engines) are left out of the setup menu but may still be chosen by flag.
    """

    engine_id: str
    title: str
    repo: str
    revision: str
    size_bytes: int
    license_name: str
    license_url: str
    commercial_use: bool
    hidden: bool = False


@dataclass(frozen=True)
class VoiceRef:
    """A voice on disk: its name, reference audio and the transcript of that audio."""

    name: str
    ref_wav: Path
    ref_text: str


@dataclass(frozen=True)
class VoicePrompt:
    """Engine-specific, opaque conditioning for one voice (``data`` is the engine's business)."""

    voice: str
    prompt_format: int
    data: object


@dataclass(frozen=True)
class AudioChunk:
    """Mono float32 samples at ``sample_rate``."""

    samples: npt.NDArray[np.float32]
    sample_rate: int


class Engine(Protocol):
    """Every call happens on the engine thread.

    An engine never writes to the file system. The one read it may do is in ``prepare_voice``:
    the reference audio named by ``VoiceRef.ref_wav`` (the engine library opens that file).
    """

    spec: EngineSpec

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def prepare_voice(self, ref: VoiceRef) -> VoicePrompt: ...

    def serialize_prompt(self, p: VoicePrompt) -> bytes: ...

    def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt:
        """Rebuild a prompt from ``serialize_prompt`` output; raise ``ValueError`` on bad data."""
        ...

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]: ...
