"""Fakes for tests: a deterministic engine, in-memory repositories and the setup fakes."""

import time
from collections.abc import Callable, Iterable, Iterator

import numpy as np

from voxlektor.adapters.audio.memory_out import MemoryOutput
from voxlektor.app.ports import Option
from voxlektor.contracts.engine import (
    ENGINE_API,
    AudioChunk,
    Capability,
    EngineSpec,
    ModelInfo,
    VoicePrompt,
    VoiceRef,
)
from voxlektor.domain.errors import DomainError, UnknownVoice
from voxlektor.domain.types import Lang

__all__ = ["FAKE_MODEL", "FakeClock", "FakeEngine", "InMemoryVoiceRepository", "MemoryOutput"]

_TONE_HZ = 220.0
_AMPLITUDE = 0.1

# The fake engine needs no download (``repo == ""``) and stays out of the setup menu.
FAKE_MODEL = ModelInfo(
    engine_id="fake",
    title="Fake engine (tests only)",
    repo="",
    revision="fake0001",
    size_bytes=0,
    license_name="Apache-2.0",
    license_url="https://github.com/MlecznyU/voxlektor",
    commercial_use=True,
    hidden=True,
)


class FakeEngine:
    """Speaks sine tones: ``seconds_per_char`` of audio per character, in 1 s chunks.

    ``synth_calls`` records a call when ``synthesize`` is called; ``fail_on`` (a substring of
    the text) and ``delay_s`` act lazily, when the returned iterator is first advanced.

    Transient failures: with ``fail_times=None`` every synthesis matching ``fail_on`` fails;
    with ``fail_times=N`` only the first N matching syntheses fail and later ones succeed.
    ``fail_prepare=N`` makes the first N ``prepare_voice`` calls raise (then they succeed).

    ``native_rate=True`` declares ``Capability.RATE`` and divides the audio length by ``rate``;
    otherwise ``rate`` is ignored. ``synth_options`` records ``(rate, steps)`` per call.
    """

    PROMPT_FORMAT = 1
    _PROMPT_MAGIC = b"fake-prompt-v1:"

    def __init__(  # noqa: PLR0913  # the failure knobs are keyword-only additions
        self,
        sample_rate: int = 24000,
        seconds_per_char: float = 0.05,
        delay_s: float = 0.0,
        fail_on: str | None = None,
        *,
        fail_times: int | None = None,
        fail_prepare: int = 0,
        native_rate: bool = False,
    ) -> None:
        capabilities = {Capability.LIVE, Capability.CLONE}
        if native_rate:
            capabilities.add(Capability.RATE)
        self.spec = EngineSpec(
            id="fake",
            revision="fake0001",
            prompt_format=self.PROMPT_FORMAT,
            sample_rate=sample_rate,
            langs=frozenset({"pl", "en"}),
            capabilities=frozenset(capabilities),
            api=ENGINE_API,
        )
        self._native_rate = native_rate
        self._seconds_per_char = seconds_per_char
        self._delay_s = delay_s
        self._fail_on = fail_on
        self._synth_failures_left = fail_times
        self._prepare_failures_left = fail_prepare
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self.prepare_calls = 0
        self.synth_calls: list[tuple[str, str]] = []
        self.synth_options: list[tuple[float, int | None]] = []

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def prepare_voice(self, ref: VoiceRef) -> VoicePrompt:
        self._require_loaded()
        if self._prepare_failures_left > 0:
            self._prepare_failures_left -= 1
            raise RuntimeError("fake prepare failure")
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

    def synthesize(
        self,
        text: str,
        lang: str,
        prompt: VoicePrompt,
        *,
        rate: float = 1.0,
        steps: int | None = None,
    ) -> Iterator[AudioChunk]:
        self._require_loaded()
        if lang not in self.spec.langs:
            raise ValueError(f"unsupported language {lang!r}")
        self.synth_calls.append((text, lang))
        self.synth_options.append((rate, steps))
        return self._chunks(text, rate if self._native_rate else 1.0)

    def _use_up_failure(self) -> bool:
        if self._synth_failures_left is None:
            return True
        if self._synth_failures_left == 0:
            return False
        self._synth_failures_left -= 1
        return True

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("engine not loaded")

    def _chunks(self, text: str, speed: float) -> Iterator[AudioChunk]:
        time.sleep(self._delay_s)
        if self._fail_on is not None and self._fail_on in text and self._use_up_failure():
            raise RuntimeError(f"fake engine failure on {self._fail_on!r}")
        rate = self.spec.sample_rate
        total = max(1, round(len(text) * self._seconds_per_char * rate / speed))
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


class _Enter:
    def __repr__(self) -> str:
        return "ENTER"


ENTER = _Enter()
"""A scripted answer that presses Enter: the question's default is taken."""


class ScriptedPrompter:
    """A Prompter that answers from a script, in order; an exhausted script presses Enter.

    An answer is a value, ``ENTER`` (take the default) or an exception instance to raise
    (``SetupCancelled`` for Esc, ``KeyboardInterrupt`` for Ctrl+C). ``asked`` records
    ``(kind, question, default)`` for every question, ``options`` the options of every
    ``choose`` and ``said`` every line said.
    """

    def __init__(self, answers: Iterable[object] = ()) -> None:
        self._answers = list(answers)
        self.asked: list[tuple[str, str, object]] = []
        self.options: list[list[Option]] = []
        self.said: list[str] = []

    def choose(self, question: str, options: list[Option], default: str) -> str:
        self.options.append(list(options))
        answer = self._next("choose", question, default)
        if not isinstance(answer, str):
            raise TypeError(f"scripted answer {answer!r} for {question!r} is not a str")
        return answer

    def number(self, question: str, low: float, high: float, default: float) -> float:
        answer = self._next("number", question, default)
        if isinstance(answer, bool) or not isinstance(answer, int | float):
            raise TypeError(f"scripted answer {answer!r} for {question!r} is not a number")
        return float(answer)

    def confirm(self, question: str, default: bool) -> bool:
        answer = self._next("confirm", question, default)
        if not isinstance(answer, bool):
            raise TypeError(f"scripted answer {answer!r} for {question!r} is not a bool")
        return answer

    def say(self, line: str) -> None:
        self.said.append(line)

    def _next(self, kind: str, question: str, default: object) -> object:
        self.asked.append((kind, question, default))
        answer = self._answers.pop(0) if self._answers else ENTER
        if isinstance(answer, BaseException):
            raise answer
        return default if answer is ENTER else answer


class FakeSamplePlayer:
    """Records every sample ``(text, voice, lang, rate, volume)`` instead of playing it."""

    def __init__(self) -> None:
        self.played: list[tuple[str, str, Lang, float, int]] = []

    def play(self, text: str, voice: str, lang: Lang, rate: float, volume: int) -> None:
        self.played.append((text, voice, lang, rate, volume))


class FakeModelStore:
    """A model cache in memory: ``present`` engine ids; a download reports ``steps``.

    With ``fail`` set a download raises it (after reporting the steps) and leaves the model
    absent, like an interrupted download.
    """

    def __init__(
        self,
        present: Iterable[str] = (),
        *,
        steps: Iterable[tuple[int, int]] = ((0, 10), (5, 10), (10, 10)),
        fail: DomainError | None = None,
    ) -> None:
        self.present = set(present)
        self._steps = list(steps)
        self._fail = fail
        self.downloads: list[str] = []

    def is_present(self, info: ModelInfo) -> bool:
        return info.engine_id in self.present

    def download(self, info: ModelInfo, progress: Callable[[int, int], None]) -> None:
        self.downloads.append(info.engine_id)
        for done, total in self._steps:
            progress(done, total)
        if self._fail is not None:
            raise self._fail
        self.present.add(info.engine_id)


class InMemorySettingsStore:
    """A SettingsStore in memory: ``file`` and ``env`` layers; ``writes`` in order.

    ``file`` is the merged view (globals + the ``default`` profile); ``profiles`` holds the
    other profiles. Writes store the raw string (no coercion; the TOML store's job).
    """

    _GLOBALS = frozenset({"engine", "history", "idle_timeout", "autostart", "floor_grace"})

    def __init__(
        self,
        file: dict[str, object] | None = None,
        env: dict[str, object] | None = None,
        *,
        completed: bool = False,
        profiles: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.file: dict[str, object] = dict(file or {})
        self.env: dict[str, object] = dict(env or {})
        self.completed = completed
        self.profiles: dict[str, dict[str, object]] = {
            name: dict(table) for name, table in (profiles or {}).items()
        }
        self.writes: list[tuple[str, str]] = []
        self.profile_writes: list[tuple[str, str, str]] = []

    def read_file(self) -> dict[str, object]:
        return dict(self.file)

    def read_env(self) -> dict[str, object]:
        return dict(self.env)

    def write_value(self, key: str, value: str) -> None:
        self.writes.append((key, value))
        self.file[key] = value

    def read_globals(self) -> dict[str, object]:
        return {key: value for key, value in self.file.items() if key in self._GLOBALS}

    def read_profiles(self) -> dict[str, dict[str, object]]:
        default = {k: v for k, v in self.file.items() if k not in self._GLOBALS}
        return {"default": default} | {name: dict(t) for name, t in self.profiles.items()}

    def write_profile_value(self, profile: str, key: str, value: str) -> None:
        self.profile_writes.append((profile, key, value))
        self._table(profile)[key] = value

    def create_profile(self, profile: str) -> None:
        self._table(profile)

    def unset_profile_value(self, profile: str, key: str) -> None:
        self._table(profile).pop(key, None)

    def remove_profile(self, profile: str) -> None:
        del self.profiles[profile]

    def _table(self, profile: str) -> dict[str, object]:
        return self.file if profile == "default" else self.profiles.setdefault(profile, {})

    def setup_completed(self) -> bool:
        return self.completed

    def mark_setup_completed(self) -> None:
        self.completed = True
