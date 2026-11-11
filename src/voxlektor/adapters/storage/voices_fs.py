"""VoiceRepository on disk per ``contracts.voicefmt``; bundled voices are package data."""

import contextlib
import os
import tempfile
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from voxlektor.contracts import voicefmt
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import UnknownVoice

_BUNDLED_PACKAGE = "voxlektor.bundled_voices"
_TEMP_PREFIX = ".tmp-"


class FsVoiceRepository:
    """Voices live in ``<root>/<name>/``; only files inside ``cache/`` are ever deleted."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def names(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(
            entry.name for entry in self._root.iterdir() if (entry / voicefmt.META).is_file()
        )

    def ref(self, name: str) -> VoiceRef:
        voice_dir = self._voice_dir(name)
        text = (voice_dir / voicefmt.REF_TXT).read_text(encoding="utf-8")
        return VoiceRef(name=name, ref_wav=voice_dir / voicefmt.REF_WAV, ref_text=text)

    def read_cache(self, name: str, key: str) -> bytes | None:
        path = self._cache_file(name, key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            return None

    def write_cache(self, name: str, key: str, data: bytes) -> None:
        path = self._cache_file(name, key)
        _private_dir(path.parent)
        # Temp names start with "." and cache keys may not, so a key never names a temp file.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=_TEMP_PREFIX)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)

    def install_bundled(self) -> None:
        """Copy bundled files that are missing; existing files are never touched."""
        _private_dir(self._root)
        for voice in resources.files(_BUNDLED_PACKAGE).iterdir():
            if voice.is_dir():
                self._install_voice(voice)

    def _install_voice(self, voice: Traversable) -> None:
        target = _private_dir(self._root / voice.name)
        # voice.toml goes last: a voice only becomes visible once all its files are in place.
        items = sorted(voice.iterdir(), key=lambda item: (item.name == voicefmt.META, item.name))
        for item in items:
            if item.is_file() and not item.name.startswith("."):
                _install_file(item, target / item.name)

    def _voice_dir(self, name: str) -> Path:
        names = self.names()
        if name not in names:
            raise UnknownVoice(name, tuple(names))
        return self._root / name

    def _cache_file(self, name: str, key: str) -> Path:
        if not key or key.startswith(".") or Path(key).name != key:
            raise ValueError(f"invalid cache key {key!r}")
        return self._voice_dir(name) / voicefmt.CACHE_DIR / key


def _install_file(item: Traversable, destination: Path) -> None:
    """Stage in the voice's ``cache/`` and link into place only if the target is still absent."""
    if destination.exists():
        return
    staging = _private_dir(destination.parent / voicefmt.CACHE_DIR)
    fd, tmp_name = tempfile.mkstemp(dir=staging, prefix=_TEMP_PREFIX)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(item.read_bytes())
        with contextlib.suppress(FileExistsError):
            os.link(tmp, destination)
    finally:
        tmp.unlink()


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path
