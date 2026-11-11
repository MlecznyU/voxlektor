"""Voice catalog: which voices exist and how a name becomes a ``VoiceRef``."""

from voxlektor.app.ports import VoiceRepository
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import UnknownVoice


class VoiceCatalog:
    def __init__(self, repo: VoiceRepository) -> None:
        self._repo = repo

    def list(self) -> list[str]:
        return sorted(self._repo.names())

    def resolve(self, name: str) -> VoiceRef:
        names = self.list()
        if name not in names:
            raise UnknownVoice(name, tuple(names))
        return self._repo.ref(name)

    def ensure_bundled(self) -> None:
        self._repo.install_bundled()
