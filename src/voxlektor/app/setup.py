"""Setup use case: the guided first run (``run_interactive``) and its silent flag mode.

Steps 2 to 9 of the setup flow, in this order: engine, model (download after the license is
accepted), voice, base language, rate, volume, service, summary. The platform (step 1) is
checked by the entrypoint before. Every question offers the current value as its default, so
Enter keeps it. Nothing is written until the end: then each value goes through
``SettingsStore.write_value`` and the setup is marked completed. A cancel anywhere
(``SetupCancelled``) or a declined model license leaves the config untouched.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from voxlektor.app.ports import ModelStore, Option, Prompter, SamplePlayer, SettingsStore
from voxlektor.contracts.engine import ModelInfo
from voxlektor.domain.errors import DomainError, InvalidDuration, InvalidSetting
from voxlektor.domain.settings import Settings, merge_layers
from voxlektor.domain.types import Lang

# ``Option`` is defined in ``app.ports`` (see there) and belongs to this module's surface.
__all__ = ["Option", "SetupCancelled", "SetupResult", "apply_flags", "run_interactive"]

_ISSUES_URL = "https://github.com/MlecznyU/voxlektor/issues"
_SAMPLES: dict[Lang, str] = {
    "pl": "Dzień dobry, tak brzmi ten głos.",
    "en": "Hello, this is how this voice sounds.",
}
_LANG_LABELS: dict[Lang, str] = {"pl": "Polish", "en": "English"}
_IDLE_CHOICES = (
    ("0", "never unload the model"),
    ("5m", "unload the model after 5 minutes without reading"),
    ("10m", "unload the model after 10 minutes without reading"),
    ("30m", "unload the model after 30 minutes without reading"),
    ("1h", "unload the model after 1 hour without reading"),
)
_FLAG_KEYS = frozenset(
    {"engine", "voice", "base_lang", "rate", "volume", "idle_timeout", "autostart"}
)
_RATE_LOW, _RATE_HIGH = 0.5, 3.0
_VOLUME_LOW, _VOLUME_HIGH = 0, 100
_GB = 1_000_000_000
_PROGRESS_STEP = 10  # percent between two progress lines


@dataclass(frozen=True, slots=True)
class SetupResult:
    """The values written (as config strings) and whether any differs from before."""

    values: dict[str, str]
    changed: bool


class SetupCancelled(DomainError):
    """The user cancelled a question (Esc) or declined the model license; nothing is written."""

    def __init__(self, message: str = "setup cancelled; nothing was changed") -> None:
        super().__init__("setup_cancelled", message)


def run_interactive(  # noqa: PLR0913, PLR0917  # one port per collaborator (frozen interface)
    store: SettingsStore,
    prompter: Prompter,
    player: SamplePlayer,
    models: ModelStore,
    engines: list[ModelInfo],
    voices: list[str],
    current: Settings,
) -> SetupResult:
    """Ask every step (Enter keeps the current value), then write the config and the marker."""
    info = _ask_engine(prompter, engines, current.engine)
    _ensure_model(prompter, models, info)
    voice = _ask_voice(prompter, player, voices, current)
    base_lang = _ask_base_lang(prompter, current.base_lang)
    sample = _Sample(player, _SAMPLES[base_lang], voice, base_lang)
    rate = _ask_rate(prompter, sample, current.rate, current.volume)
    volume = _ask_volume(prompter, sample, rate, current.volume)
    idle = _ask_idle(prompter, current.idle_timeout_s)
    autostart = prompter.confirm(
        "Start the service automatically when you read something?", default=current.autostart
    )
    chosen = _merge(
        {
            "engine": info.engine_id,
            "voice": voice,
            "base_lang": base_lang,
            "rate": rate,
            "volume": volume,
            "idle_timeout": idle,
            "autostart": autostart,
        }
    )
    return _finish(store, current, chosen)


def apply_flags(  # noqa: PLR0913, PLR0917  # frozen interface
    store: SettingsStore,
    models: ModelStore,
    engines: list[ModelInfo],
    current: Settings,
    flags: Mapping[str, object],
    accept_license: bool,
) -> SetupResult:
    """Setup without questions: given flags override the current values (``None`` = not given).

    No samples are played. A model that is not cached is downloaded only with
    ``accept_license``; otherwise nothing is written and ``InvalidSetting`` says why.
    """
    given = {key: value for key, value in flags.items() if value is not None}
    unknown = sorted(set(given) - _FLAG_KEYS)
    if unknown:
        raise InvalidSetting(f"unknown setup option {unknown[0]!r}")
    chosen = _merge({**_typed(current), **given})
    info = _find_engine(engines, chosen.engine)
    if not models.is_present(info):
        if not accept_license:
            raise InvalidSetting("model download needs --accept-model-license")
        models.download(info, _no_progress)
    return _finish(store, current, chosen)


@dataclass(frozen=True, slots=True)
class _Sample:
    """The sample sentence in the chosen voice and language; rate and volume vary."""

    player: SamplePlayer
    text: str
    voice: str
    lang: Lang

    def play(self, rate: float, volume: int) -> None:
        self.player.play(self.text, self.voice, self.lang, rate, volume)


def _ask_engine(prompter: Prompter, engines: list[ModelInfo], current: str) -> ModelInfo:
    if not engines:
        raise InvalidSetting("no engine is installed")
    ids = [info.engine_id for info in engines]
    only = " (the only one so far)" if len(ids) == 1 else ""
    question = (
        f"Engine. Available engines: {', '.join(ids)}{only}. "
        f"Want another engine? Open an issue or a pull request: {_ISSUES_URL}"
    )
    options = [Option(info.engine_id, info.title) for info in engines]
    default = current if current in ids else ids[0]
    return _find_engine(engines, prompter.choose(question, options, default))


def _ensure_model(prompter: Prompter, models: ModelStore, info: ModelInfo) -> None:
    if models.is_present(info):
        return
    prompter.say(
        f"{info.title} needs a one-time download of {info.size_bytes / _GB:.1f} GB "
        "from Hugging Face."
    )
    prompter.say(f"Model license: {info.license_name} ({info.license_url})")
    if not info.commercial_use:
        prompter.say("The model weights are licensed for non-commercial use only.")
    if not prompter.confirm("Accept the model license and download it now?", default=False):
        raise SetupCancelled("setup cancelled: the model license was not accepted")
    models.download(info, _progress_lines(prompter))


def _ask_voice(
    prompter: Prompter, player: SamplePlayer, voices: list[str], current: Settings
) -> str:
    if not voices:
        raise InvalidSetting("no voice is installed")
    options = [Option(name, name) for name in voices]
    candidate = current.voice if current.voice in voices else voices[0]
    text = _SAMPLES[current.base_lang]
    while True:
        candidate = prompter.choose("Voice (you will hear a sample)", options, candidate)
        player.play(text, candidate, current.base_lang, current.rate, current.volume)
        if prompter.confirm(f"Keep the voice {candidate}?", default=True):
            return candidate


def _ask_base_lang(prompter: Prompter, current: Lang) -> Lang:
    options = [Option(lang, f"{lang}: {label}") for lang, label in _LANG_LABELS.items()]
    answer = prompter.choose("Base language (for text whose language is unclear)", options, current)
    return "en" if answer == "en" else "pl"


def _ask_rate(prompter: Prompter, sample: _Sample, rate: float, volume: int) -> float:
    while True:
        rate = prompter.number("Rate (0.5 to 3.0; 1.0 = normal)", _RATE_LOW, _RATE_HIGH, rate)
        sample.play(rate, volume)
        if prompter.confirm("Keep this rate?", default=True):
            return rate


def _ask_volume(prompter: Prompter, sample: _Sample, rate: float, volume: int) -> int:
    while True:
        answer = prompter.number(
            "Volume (0 to 100; 50 = even level)", _VOLUME_LOW, _VOLUME_HIGH, volume
        )
        volume = round(answer)
        sample.play(rate, volume)
        if prompter.confirm("Keep this volume?", default=True):
            return volume


def _ask_idle(prompter: Prompter, current_s: int) -> str:
    current = _duration_text(current_s)
    options = [Option(value, f"{value}: {label}") for value, label in _IDLE_CHOICES]
    if current not in {value for value, _ in _IDLE_CHOICES}:
        options.append(Option(current, f"{current}: keep the current timeout"))
    return prompter.choose("Service: when should the model be unloaded?", options, current)


def _find_engine(engines: list[ModelInfo], engine_id: str) -> ModelInfo:
    for info in engines:
        if info.engine_id == engine_id:
            return info
    available = ", ".join(info.engine_id for info in engines)
    raise InvalidSetting(f"unknown engine {engine_id!r}; available: {available}")


def _merge(values: Mapping[str, object]) -> Settings:
    try:
        return merge_layers(values)
    except InvalidDuration as err:
        raise InvalidSetting(err.message) from None


def _typed(settings: Settings) -> dict[str, object]:
    return {
        "engine": settings.engine,
        "voice": settings.voice,
        "base_lang": settings.base_lang,
        "rate": settings.rate,
        "volume": settings.volume,
        "idle_timeout": settings.idle_timeout_s,
        "autostart": settings.autostart,
    }


def _texts(settings: Settings) -> dict[str, str]:
    """The setup values as ``write_value`` strings."""
    return {
        "engine": settings.engine,
        "voice": settings.voice,
        "base_lang": settings.base_lang,
        "rate": str(settings.rate),
        "volume": str(settings.volume),
        "idle_timeout": _duration_text(settings.idle_timeout_s),
        "autostart": "true" if settings.autostart else "false",
    }


def _finish(store: SettingsStore, current: Settings, chosen: Settings) -> SetupResult:
    values = _texts(chosen)
    for key, value in values.items():
        store.write_value(key, value)
    store.mark_setup_completed()
    return SetupResult(values, changed=values != _texts(current))


def _duration_text(seconds: int) -> str:
    """``600`` -> ``10m``: the largest unit that divides exactly (``0`` stays ``0``)."""
    if seconds == 0:
        return "0"
    for size, unit in ((3600, "h"), (60, "m")):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def _progress_lines(prompter: Prompter) -> Callable[[int, int], None]:
    """A progress callback that says one line per ``_PROGRESS_STEP`` percent reached.

    The total may grow during a download (files add their size when they start); then the
    percentage is said again from its new, lower value instead of waiting to pass the old one.
    """
    last = _ProgressLine()

    def report(done: int, total: int) -> None:
        if last.advance(done, total):
            prompter.say(f"downloading the model: {last.percent}%")

    return report


@dataclass(slots=True)
class _ProgressLine:
    """The last said percentage (``-1``: none yet) and the total it was computed from."""

    percent: int = -1
    total: int = 0

    def advance(self, done: int, total: int) -> bool:
        """Take one report; True when a new line is due."""
        if total <= 0:
            return False
        percent = min(100, done * 100 // total) // _PROGRESS_STEP * _PROGRESS_STEP
        grew, self.total = total > self.total, max(total, self.total)
        if percent == self.percent or (percent < self.percent and not grew):
            return False
        self.percent = percent
        return True


def _no_progress(done: int, total: int) -> None:
    return None
