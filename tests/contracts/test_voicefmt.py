from pathlib import Path

from voxlektor.contracts import voicefmt
from voxlektor.contracts.engine import VoiceRef
from voxlektor.contracts.voicefmt import VoiceMeta, prompt_cache_name


def test_prompt_cache_name() -> None:
    name = prompt_cache_name("omnivoice", "0123456789abcdef", 2)
    assert name == "omnivoice-01234567-p2.bin"


def test_prompt_cache_name_short_revision() -> None:
    assert prompt_cache_name("e", "abc", 1) == "e-abc-p1.bin"


def test_format_constants() -> None:
    assert voicefmt.VOICE_FORMAT == 1
    assert voicefmt.REF_WAV == "ref.wav"
    assert voicefmt.REF_TXT == "ref.txt"
    assert voicefmt.META == "voice.toml"
    assert voicefmt.CACHE_DIR == "cache"
    assert voicefmt.MAX_REF_SECONDS == 30


def test_voice_meta_is_frozen_value() -> None:
    meta = VoiceMeta(format=1, name="v", source="bundled", lang="pl", created="2026-11-10")
    assert meta == VoiceMeta(1, "v", "bundled", "pl", "2026-11-10")


def test_voice_ref_is_frozen_value() -> None:
    ref = VoiceRef(name="v", ref_wav=Path("/x/ref.wav"), ref_text="hi")
    assert ref == VoiceRef("v", Path("/x/ref.wav"), "hi")
