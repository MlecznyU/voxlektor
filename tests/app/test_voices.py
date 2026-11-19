from pathlib import Path

import pytest

from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import UnknownVoice


class _Repo:
    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.installed = 0

    def names(self) -> list[str]:
        return list(self._names)

    def ref(self, name: str) -> VoiceRef:
        return VoiceRef(name, Path(f"/v/{name}/ref.wav"), f"text of {name}")

    def read_cache(self, name: str, key: str) -> bytes | None:
        return None

    def write_cache(self, name: str, key: str, data: bytes) -> None:
        return None

    def install_bundled(self) -> None:
        self.installed += 1


BUNDLED = [
    "v0-default",
    "f1-warm",
    "f2-narrator",
    "f3-bright",
    "m1-calm",
    "m2-narrator",
    "m3-casual",
    "zoe-spark",
]


def test_list_is_sorted() -> None:
    assert VoiceCatalog(_Repo(BUNDLED)).list() == sorted(BUNDLED)
    assert VoiceCatalog(_Repo(BUNDLED)).list() != BUNDLED


def test_resolve_known_voice() -> None:
    ref = VoiceCatalog(_Repo(BUNDLED)).resolve("m1-calm")
    assert ref == VoiceRef("m1-calm", Path("/v/m1-calm/ref.wav"), "text of m1-calm")


def test_resolve_unknown_lists_available() -> None:
    with pytest.raises(UnknownVoice) as info:
        VoiceCatalog(_Repo(BUNDLED)).resolve("nope")
    assert info.value.name == "nope"
    assert info.value.available == tuple(sorted(BUNDLED))
    assert info.value.code == "unknown_voice"


def test_ensure_bundled_delegates() -> None:
    repo = _Repo([])
    VoiceCatalog(repo).ensure_bundled()
    assert repo.installed == 1
