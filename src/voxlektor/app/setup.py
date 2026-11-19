"""Setup use case: the guided first run (``run_interactive``) and its silent flag mode.

Steps 2 to 9 of the setup flow, in this order: engine, model (download after the license is
accepted), voice, base language, rate, volume, service, summary. The platform (step 1) is
checked by the entrypoint before. Every question offers the current value as its default, so
Enter keeps it. Nothing is written until the end: then each value goes through
``SettingsStore.write_value`` and the setup is marked completed. A cancel anywhere
(``SetupCancelled``) or a declined model license leaves the config untouched.

Profiles (S3): ``run_profile_interactive`` asks one non-default profile's voice, base
language, rate, volume, reading language and quality steps; each question shows the effective
value and where it comes from, and "same as default" removes the profile's own value.
``apply_profile_flags`` stores only the given profile keys (also when equal to default's) in
any profile, creating it when missing.
Engine, model and service stay global: they belong to ``run_interactive`` / ``apply_flags``.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

from voxlektor.app.ports import ModelStore, Option, Prompter, SamplePlayer, SettingsStore
from voxlektor.contracts.engine import ModelInfo
from voxlektor.domain.errors import DomainError, InvalidDuration, InvalidSetting
from voxlektor.domain.profiles import (
    DEFAULT_PROFILE,
    PROFILE_KEYS,
    Source,
    resolve,
    valid_profile_name,
)
from voxlektor.domain.settings import Settings, merge_layers
from voxlektor.domain.types import Lang, LangPref

# ``Option`` is defined in ``app.ports`` (see there) and belongs to this module's surface.
__all__ = [
    "Option",
    "SetupCancelled",
    "SetupResult",
    "apply_flags",
    "apply_profile_flags",
    "run_interactive",
    "run_profile_interactive",
]

_ISSUES_URL = "https://github.com/MlecznyU/voxlektor/issues"
_SAMPLES: dict[Lang, str] = {
    "pl": "Dzień dobry, tak brzmi ten głos.",
    "en": "Hello, this is how this voice sounds.",
}
_LANG_LABELS: dict[Lang, str] = {"pl": "Polish", "en": "English"}
_READ_LANG_LABELS: dict[LangPref, str] = {
    "auto": "detect per sentence",
    "pl": "Polish",
    "en": "English",
}
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
_STEPS_LOW, _STEPS_HIGH = 4, 32
_GB = 1_000_000_000
_PROGRESS_STEP = 10  # percent between two progress lines
_SAME = ":default"  # menu value of "same as default" (no voice or language is named so)
_SET = ":set"  # menu value of "set a value for this profile"
_ASKED = PROFILE_KEYS  # a profile run asks every profile key
_SOURCES: dict[Source, str] = {
    "profile": "set in this profile",
    "default": "from default",
    "builtin": "built-in",
    "env": "from the environment",
    "flag": "from an option",
}


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
    _ensure_present(models, _find_engine(engines, chosen.engine), accept_license)
    return _finish(store, current, chosen)


def run_profile_interactive(
    store: SettingsStore,
    prompter: Prompter,
    player: SamplePlayer,
    voices: list[str],
    profile: str,
) -> SetupResult:
    """Ask one non-default profile's voice, base language, rate, volume, reading language
    (``lang``) and quality steps, then write it.

    Each question shows the effective value and its source; "same as default" removes the
    profile's own value. A missing profile is created (also when it stays empty). Values
    stored in the profile that are invalid are ignored (and repaired when asked). The result
    holds the profile's effective values; ``changed`` says whether its own values changed.
    """
    if profile == DEFAULT_PROFILE:
        raise InvalidSetting("the default profile is set up by the full setup (no --profile)")
    view = _ProfileView.load(store, profile)
    draft = _Draft(view.base, view.base_sources, dict(view.own))
    _ask_profile_voice(prompter, player, voices, draft)
    _ask_profile_base_lang(prompter, draft)
    if _same_or_set(prompter, draft, "rate", "Rate"):
        now = draft.settings()
        draft.values["rate"] = _ask_rate(prompter, _sample_of(player, now), now.rate, now.volume)
    if _same_or_set(prompter, draft, "volume", "Volume"):
        now = draft.settings()
        sample = _sample_of(player, now)
        draft.values["volume"] = _ask_volume(prompter, sample, now.rate, now.volume)
    _ask_profile_lang(prompter, draft)
    if _same_or_set(prompter, draft, "steps", "Quality steps"):
        steps = prompter.number(
            "Steps (4 to 32; fewer = faster)", _STEPS_LOW, _STEPS_HIGH, draft.settings().steps
        )
        draft.values["steps"] = round(steps)
    settings = draft.settings()  # validates before anything is written
    unset = [key for key in _ASKED if key in view.stored and key not in draft.values]
    changed = _save_profile(store, profile, view.own, draft.values, unset)
    return SetupResult(_profile_texts(settings), changed)


def apply_profile_flags(  # noqa: PLR0913, PLR0917  # the frozen apply_flags plus the profile
    store: SettingsStore,
    models: ModelStore,
    engines: list[ModelInfo],
    current: Settings,
    profile: str,
    flags: Mapping[str, object],
    accept_license: bool,
) -> SetupResult:
    """Store only the given profile keys (``None`` = not given) in ``profile``, creating it.

    A key equal to default's value is still stored (an explicit pin). Engine and service
    keys are global and refused here. The model of the current engine must be present, or is
    downloaded with ``accept_license``; then the setup is marked completed.
    """
    given = {key: value for key, value in flags.items() if value is not None}
    unknown = sorted(set(given) - set(PROFILE_KEYS))
    if unknown:
        raise InvalidSetting(f"{unknown[0]!r} is not a profile setting")
    view = _ProfileView.load(store, profile)
    after = {**view.own, **given}
    settings = _merge({**_profile_layer(view.base), **after})
    _ensure_present(models, _find_engine(engines, current.engine), accept_license)
    changed = _save_profile(store, profile, view.own, after, ())
    store.mark_setup_completed()
    return SetupResult(_profile_texts(settings), changed)


@dataclass(frozen=True, slots=True)
class _ProfileView:
    """What a profile run starts from (the file's values only, never the env's)."""

    base: Settings  # the default profile's effective values: what "same as default" gives
    base_sources: Mapping[str, Source]
    own: dict[str, object]  # the profile's valid own values
    stored: Collection[str]  # every key the profile stores, valid or not

    @staticmethod
    def load(store: SettingsStore, profile: str) -> "_ProfileView":
        if not valid_profile_name(profile):
            raise InvalidSetting(
                f"invalid profile name {profile!r}; use a-z, 0-9, '-' and '_' (at most 32, "
                "starting with a letter or digit)"
            )
        tables = store.read_profiles()
        valid = {name: _valid_values(table) for name, table in tables.items()}
        base, sources = resolve(
            profile=DEFAULT_PROFILE, profiles=valid, globals_={}, env={}, flags={}
        )
        own = valid.get(profile, {})
        return _ProfileView(base, sources, own, frozenset(tables.get(profile, {})))


@dataclass(frozen=True, slots=True)
class _Draft:
    """A profile's own values while its questions are answered."""

    base: Settings
    base_sources: Mapping[str, Source]
    values: dict[str, object]

    def settings(self) -> Settings:
        return _merge({**_profile_layer(self.base), **self.values})

    def now(self, key: str) -> str:
        """``now 1.1, set in this profile``: the effective value and where it comes from."""
        value = _profile_texts(self.settings())[key]
        source: Source = "profile" if key in self.values else self.base_sources[key]
        return f"now {value}, {_SOURCES[source]}"

    def take(self, key: str, answer: str, value: object) -> None:
        if answer == _SAME:
            self.values.pop(key, None)
        else:
            self.values[key] = value


def _ask_profile_voice(
    prompter: Prompter, player: SamplePlayer, voices: list[str], draft: _Draft
) -> None:
    if not voices:
        raise InvalidSetting("no voice is installed")
    same = Option(_SAME, f"same as default ({draft.base.voice})")
    options = [same, *(Option(name, name) for name in voices)]
    candidate = next((name for name in voices if name == draft.values.get("voice")), _SAME)
    while True:
        question = f"Voice ({draft.now('voice')}; you will hear a sample)"
        candidate = prompter.choose(question, options, candidate)
        draft.take("voice", candidate, candidate)
        now = draft.settings()
        _sample_of(player, now).play(now.rate, now.volume)
        if prompter.confirm(f"Keep the voice {now.voice}?", default=True):
            return


def _ask_profile_base_lang(prompter: Prompter, draft: _Draft) -> None:
    base = draft.base.base_lang
    options = [Option(_SAME, f"same as default ({base}: {_LANG_LABELS[base]})")]
    options += [Option(lang, f"{lang}: {label}") for lang, label in _LANG_LABELS.items()]
    default = str(draft.values.get("base_lang", _SAME))
    answer = prompter.choose(f"Base language ({draft.now('base_lang')})", options, default)
    draft.take("base_lang", answer, "en" if answer == "en" else "pl")


def _ask_profile_lang(prompter: Prompter, draft: _Draft) -> None:
    options = [Option(_SAME, f"same as default ({draft.base.lang})")]
    options += [Option(lang, f"{lang}: {label}") for lang, label in _READ_LANG_LABELS.items()]
    default = str(draft.values.get("lang", _SAME))
    answer = prompter.choose(f"Reading language ({draft.now('lang')})", options, default)
    draft.take("lang", answer, answer)


def _same_or_set(prompter: Prompter, draft: _Draft, key: str, title: str) -> bool:
    """Ask whether ``key`` follows default or gets its own value; True for its own value."""
    what = title.lower() if title.endswith("s") else f"a {title.lower()}"
    options = [
        Option(_SAME, f"same as default ({_profile_texts(draft.base)[key]})"),
        Option(_SET, f"set {what} for this profile"),
    ]
    default = _SET if key in draft.values else _SAME
    answer = prompter.choose(f"{title} ({draft.now(key)})", options, default)
    if answer == _SAME:
        draft.values.pop(key, None)
        return False
    return True


def _sample_of(player: SamplePlayer, settings: Settings) -> "_Sample":
    lang = settings.base_lang
    return _Sample(player, _SAMPLES[lang], settings.voice, lang)


def _valid_values(table: Mapping[str, object]) -> dict[str, object]:
    """A profile table's profile keys whose values are valid (others are skipped)."""
    valid: dict[str, object] = {}
    for key, value in table.items():
        if key in PROFILE_KEYS and _invalid(key, value) is None:
            valid[key] = value
    return valid


def _invalid(key: str, value: object) -> str | None:
    try:
        merge_layers({key: value})
    except InvalidSetting as err:
        return err.message
    return None


def _save_profile(
    store: SettingsStore,
    profile: str,
    before: Mapping[str, object],
    after: Mapping[str, object],
    unset: Collection[str],
) -> bool:
    """Create the profile, write the keys that are new or changed, drop ``unset``.

    True when the profile's own (valid) values differ from before.
    """
    old, new = _overlay_texts(before), _overlay_texts(after)
    store.create_profile(profile)
    for key, value in new.items():
        if old.get(key) != value:
            store.write_profile_value(profile, key, value)
    for key in unset:
        store.unset_profile_value(profile, key)
    return old != new


def _profile_layer(settings: Settings) -> dict[str, object]:
    return {
        "voice": settings.voice,
        "lang": settings.lang,
        "base_lang": settings.base_lang,
        "rate": settings.rate,
        "volume": settings.volume,
        "steps": settings.steps,
    }


def _profile_texts(settings: Settings) -> dict[str, str]:
    """The profile keys as ``write_profile_value`` strings."""
    return {key: str(value) for key, value in _profile_layer(settings).items()}


def _overlay_texts(overlay: Mapping[str, object]) -> dict[str, str]:
    texts = _profile_texts(_merge(overlay))
    return {key: texts[key] for key in PROFILE_KEYS if key in overlay}


def _ensure_present(models: ModelStore, info: ModelInfo, accept_license: bool) -> None:
    if models.is_present(info):
        return
    if not accept_license:
        raise InvalidSetting("model download needs --accept-model-license")
    models.download(info, _no_progress)


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
