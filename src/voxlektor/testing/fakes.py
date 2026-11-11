"""Fakes for tests: a deterministic engine and an in-memory voice repository."""

import time
from collections.abc import Iterable, Iterator

import numpy as np

from voxlektor.adapters.audio.memory_out import MemoryOutput
from voxlektor.contracts.engine import AudioChunk, Capability, EngineSpec, VoicePrompt, VoiceRef
from voxlektor.domain.errors import UnknownVoice

__all__ = ["FakeClock", "FakeEngine", "InMemoryVoiceRepository", "MemoryOutput"]

_TONE_HZ = 220.0
_AMPLITUDE = 0.1


class FakeEngine:
    """Speaks sine tones: ``seconds_per_char`` of audio per character, in 1 s chunks.

    ``synth_calls`` records a call when ``synthesize`` is called; ``fail_on`` (a substring of
    the text) and ``delay_s`` act lazily, when the returned iterator is first advanced.
    """

    PROMPT_FORMAT = 1
    _PROMPT_MAGIC = b"fake-prompt-v1:"

    def __init__(
        self,
        sample_rate: int = 24000,
        seconds_per_char: float = 0.05,
        delay_s: float = 0.0,
        fail_on: str | None = None,
    ) -> None:
        self.spec = EngineSpec(
            id="fake",
            revision="fake0001",
            prompt_format=self.PROMPT_FORMAT,
            sample_rate=sample_rate,
            langs=frozenset({"pl", "en"}),
            capabilities=frozenset({Capability.LIVE, Capability.CLONE}),
        )
        self._seconds_per_char = seconds_per_char
        self._delay_s = delay_s
        self._fail_on = fail_on
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self.prepare_calls = 0
        self.synth_calls: list[tuple[str, str]] = []

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def prepare_voice(self, ref: VoiceRef) -> VoicePrompt:
        self._require_loaded()
        self.prepare_calls += 1
        return VoicePrompt(ref.name, self.PROMPT_FORMAT, ref.ref_text)

    def serialize_prompt(self, p: VoicePrompt) -> bytes:
        if not isinstance(p.data, str):
            raise TypeError("not a fake prompt")
        return self._PROMPT_MAGIC + p.data.encode("utf-8")

    def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt:
        if not data.startswith(self._PROMPT_MAGIC):
            raise ValueError("not a fake prompt")
        try:
            text = data[len(self._PROMPT_MAGIC) :].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("corrupt fake prompt") from exc
        return VoicePrompt(voice, self.PROMPT_FORMAT, text)

    def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]:
        self._require_loaded()
        if lang not in self.spec.langs:
            raise ValueError(f"unsupported language {lang!r}")
        self.synth_calls.append((text, lang))
        return self._chunks(text)

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("engine not loaded")

    def _chunks(self, text: str) -> Iterator[AudioChunk]:
        time.sleep(self._delay_s)
        if self._fail_on is not None and self._fail_on in text:
            raise RuntimeError(f"fake engine failure on {self._fail_on!r}")
        rate = self.spec.sample_rate
        total = max(1, round(len(text) * self._seconds_per_char * rate))
        for start in range(0, total, rate):
            index = np.arange(start, min(start + rate, total), dtype=np.float64)
            tone = _AMPLITUDE * np.sin(2.0 * np.pi * _TONE_HZ * index / rate)
            yield AudioChunk(tone.astype(np.float32), rate)


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, t: float = 0.0) -> None:
        self._t = t

    def now(self) -> float:
        return self._t

    def advance(self, s: float) -> None:
        self._t += s


class InMemoryVoiceRepository:
    """VoiceRepository without a disk; ``cache`` is public so tests can plant corrupt bytes."""

    def __init__(self, refs: Iterable[VoiceRef] = ()) -> None:
        self._refs = {ref.name: ref for ref in refs}
        self.cache: dict[tuple[str, str], bytes] = {}
        self.install_calls = 0

    def names(self) -> list[str]:
        return sorted(self._refs)

    def ref(self, name: str) -> VoiceRef:
        try:
            return self._refs[name]
        except KeyError:
            raise UnknownVoice(name, tuple(self.names())) from None

    def read_cache(self, name: str, key: str) -> bytes | None:
        return self.cache.get((name, key))

    def write_cache(self, name: str, key: str, data: bytes) -> None:
        self.cache[(name, key)] = data

    def install_bundled(self) -> None:
        self.install_calls += 1
