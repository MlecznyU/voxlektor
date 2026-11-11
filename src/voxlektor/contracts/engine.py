"""Engine SPI contract (extended by the engine task); for now only ``VoiceRef``."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceRef:
    """A voice on disk: its name, reference audio and the transcript of that audio."""

    name: str
    ref_wav: Path
    ref_text: str
