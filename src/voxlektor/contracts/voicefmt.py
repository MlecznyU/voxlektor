"""Voice format v1: ``voices/<name>/{ref.wav, ref.txt, voice.toml, cache/}``."""

from dataclasses import dataclass
from typing import Literal

VOICE_FORMAT = 1

REF_WAV = "ref.wav"
REF_TXT = "ref.txt"
META = "voice.toml"
CACHE_DIR = "cache"

MAX_REF_SECONDS = 30


@dataclass(frozen=True)
class VoiceMeta:
    """Contents of ``voice.toml``, the source of truth about a voice."""

    format: int
    name: str
    source: Literal["bundled", "clone", "design"]
    lang: str
    created: str


def prompt_cache_name(engine_id: str, revision: str, prompt_format: int) -> str:
    """File name of a disposable prompt cache inside ``cache/``."""
    return f"{engine_id}-{revision[:8]}-p{prompt_format}.bin"
