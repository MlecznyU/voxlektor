"""The setup use case through a scripted prompter: no TTY, no audio, no download."""

from dataclasses import replace

import pytest

from voxlektor.app.ports import Option
from voxlektor.app.setup import (
    SetupCancelled,
    SetupResult,
    apply_flags,
    apply_profile_flags,
    run_interactive,
    run_profile_interactive,
)
from voxlektor.contracts.engine import ModelInfo
from voxlektor.domain.errors import InvalidSetting, ServiceUnavailable
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId, VoiceName
from voxlektor.testing.fakes import (
    ENTER,
    FakeModelStore,
    FakeSamplePlayer,
    InMemorySettingsStore,
    ScriptedPrompter,
)

OMNI = ModelInfo(
    engine_id="omnivoice",
    title="OmniVoice (k2-fsa), 0.6B, bf16",
    repo="mlx-community/OmniVoice-bf16",
    revision="8fb0b754",
    size_bytes=1_700_000_000,
    license_name="CC-BY-NC-4.0",
    license_url="https://huggingface.co/k2-fsa/OmniVoice",
    commercial_use=False,
)
OPEN = replace(OMNI, engine_id="open", title="Open engine", commercial_use=True)
FAKE = replace(OMNI, engine_id="fake", title="Fake", repo="", hidden=True)
VOICES = ["f1-calm", "m1-warm", "v0-default"]
CURRENT = Settings()
DEFAULT_VALUES = {
    "engine": "omnivoice",
    "voice": "v0-default",
    "base_lang": "pl",
    "rate": "1.0",
    "volume": "50",
    "idle_timeout": "10m",
    "autostart": "true",
}
PL_SAMPLE = "Dzień dobry, tak brzmi ten głos."
EN_SAMPLE = "Hello, this is how this voice sounds."


def _interactive(  # noqa: PLR0913  # keyword-only overrides of the defaults
    prompter: ScriptedPrompter,
    *,
    store: InMemorySettingsStore | None = None,
    models: FakeModelStore | None = None,
    player: FakeSamplePlayer | None = None,
    engines: list[ModelInfo] | None = None,
    voices: list[str] | None = None,
    current: Settings = CURRENT,
) -> SetupResult:
    return run_interactive(
        store if store is not None else InMemorySettingsStore(),
        prompter,
        player if player is not None else FakeSamplePlayer(),
        models if models is not None else FakeModelStore(present=["omnivoice"]),
        [OMNI] if engines is None else engines,
        VOICES if voices is None else voices,
        current,
    )


def _questions(prompter: ScriptedPrompter) -> list[str]:
    return [question for _, question, _ in prompter.asked]


# ---------------------------------------------------------------- interactive


def test_enter_everywhere_keeps_the_current_values_and_marks_completed() -> None:
    store = InMemorySettingsStore()
    prompter = ScriptedPrompter()
    result = _interactive(prompter, store=store)
    assert result == SetupResult(DEFAULT_VALUES, changed=False)
    assert store.writes == list(DEFAULT_VALUES.items())
    assert store.completed is True
    assert [kind for kind, _, _ in prompter.asked] == [
        "choose",  # engine
        "choose",  # voice
        "confirm",
        "choose",  # base language
        "number",  # rate
        "confirm",
        "number",  # volume
        "confirm",
        "choose",  # idle timeout
        "confirm",  # autostart
    ]
    assert [default for _, _, default in prompter.asked] == [
        "omnivoice",
        "v0-default",
        True,
        "pl",
        1.0,
        True,
        50,
        True,
        "10m",
        True,
    ]


def test_full_happy_path_writes_exactly_the_chosen_keys() -> None:
    store = InMemorySettingsStore(completed=False)
    player = FakeSamplePlayer()
    models = FakeModelStore()
    prompter = ScriptedPrompter(
        [
            "omnivoice",
            True,  # accept the license
            "m1-warm",
            True,
            "en",
            1.25,
            True,
            70,
            True,
            "1h",
            False,
        ]
    )
    result = _interactive(prompter, store=store, models=models, player=player)
    expected = {
        "engine": "omnivoice",
        "voice": "m1-warm",
        "base_lang": "en",
        "rate": "1.25",
        "volume": "70",
        "idle_timeout": "1h",
        "autostart": "false",
    }
    assert result == SetupResult(expected, changed=True)
    assert store.writes == list(expected.items())
    assert store.completed is True
    assert models.downloads == ["omnivoice"]
    assert player.played == [
        (PL_SAMPLE, "m1-warm", "pl", 1.0, 50),  # voice: current language, rate and volume
        (EN_SAMPLE, "m1-warm", "en", 1.25, 50),  # rate: the chosen language
        (EN_SAMPLE, "m1-warm", "en", 1.25, 70),  # volume
    ]


def test_engine_question_is_shown_with_one_engine() -> None:
    prompter = ScriptedPrompter()
    _interactive(prompter)
    engine_question = _questions(prompter)[0]
    assert "Available engines: omnivoice (the only one so far)." in engine_question
    assert (
        "Want another engine? Open an issue or a pull request: "
        "https://github.com/MlecznyU/voxlektor/issues"
    ) in engine_question
    assert prompter.options[0] == [Option("omnivoice", OMNI.title)]


def test_several_engines_are_listed_and_an_unknown_current_falls_back_to_the_first() -> None:
    prompter = ScriptedPrompter()
    models = FakeModelStore(present=["omnivoice", "open"])
    current = replace(CURRENT, engine=EngineId("gone"))
    result = _interactive(prompter, models=models, engines=[OPEN, OMNI], current=current)
    question = _questions(prompter)[0]
    assert "Available engines: open, omnivoice. Want" in question
    assert prompter.asked[0][2] == "open"
    assert result.values["engine"] == "open"
    assert result.changed is True


def test_no_engine_is_a_setting_error() -> None:
    with pytest.raises(InvalidSetting, match="no engine"):
        _interactive(ScriptedPrompter(), engines=[])


def test_absent_model_shows_size_and_license_then_downloads_with_progress() -> None:
    prompter = ScriptedPrompter([ENTER, True])
    models = FakeModelStore(steps=[(0, 0), (0, 100), (5, 100), (40, 100), (45, 100), (100, 100)])
    _interactive(prompter, models=models)
    assert prompter.said == [
        "OmniVoice (k2-fsa), 0.6B, bf16 needs a one-time download of 1.7 GB from Hugging Face.",
        "Model license: CC-BY-NC-4.0 (https://huggingface.co/k2-fsa/OmniVoice)",
        "The model weights are licensed for non-commercial use only.",
        "downloading the model: 0%",
        "downloading the model: 40%",
        "downloading the model: 100%",
    ]
    license_question = prompter.asked[1]
    assert license_question[0] == "confirm"
    assert license_question[2] is False  # the default is "no"


def test_progress_is_redrawn_when_the_total_grows_after_an_early_100_percent() -> None:
    """The library's total grows while files start: an early 100% must not freeze the lines."""
    prompter = ScriptedPrompter([ENTER, True])
    steps = [(10, 10), (10, 10), (10, 100), (30, 100), (55, 100), (55, 200), (200, 200)]
    _interactive(prompter, models=FakeModelStore(steps=steps))
    assert [line for line in prompter.said if line.startswith("downloading")] == [
        "downloading the model: 100%",
        "downloading the model: 10%",
        "downloading the model: 30%",
        "downloading the model: 50%",
        "downloading the model: 20%",
        "downloading the model: 100%",
    ]


def test_commercial_model_has_no_non_commercial_line() -> None:
    prompter = ScriptedPrompter([ENTER, True])
    _interactive(prompter, engines=[OPEN], models=FakeModelStore(steps=[]))
    assert not any("non-commercial" in line for line in prompter.said)


def test_declined_license_cancels_and_writes_nothing() -> None:
    store = InMemorySettingsStore()
    models = FakeModelStore()
    with pytest.raises(SetupCancelled) as info:
        _interactive(ScriptedPrompter([ENTER, False]), store=store, models=models)
    assert info.value.code == "setup_cancelled"
    assert store.writes == []
    assert store.completed is False
    assert models.downloads == []


def test_failed_download_writes_nothing() -> None:
    store = InMemorySettingsStore()
    models = FakeModelStore(fail=ServiceUnavailable("model download failed: offline"))
    with pytest.raises(ServiceUnavailable):
        _interactive(ScriptedPrompter([ENTER, True]), store=store, models=models)
    assert store.writes == []
    assert store.completed is False


def test_present_model_asks_no_license() -> None:
    prompter = ScriptedPrompter()
    _interactive(prompter)
    assert prompter.said == []
    assert not any("license" in question for question in _questions(prompter))


def test_cancel_in_a_menu_writes_nothing() -> None:
    store = InMemorySettingsStore()
    with pytest.raises(SetupCancelled):
        _interactive(ScriptedPrompter([ENTER, ENTER, ENTER, SetupCancelled()]), store=store)
    assert store.writes == []
    assert store.completed is False


def test_ctrl_c_in_a_menu_propagates_and_writes_nothing() -> None:
    store = InMemorySettingsStore()
    with pytest.raises(KeyboardInterrupt):
        _interactive(ScriptedPrompter([KeyboardInterrupt()]), store=store)
    assert store.writes == []


def test_voice_step_plays_a_sample_per_candidate() -> None:
    player = FakeSamplePlayer()
    prompter = ScriptedPrompter([ENTER, "f1-calm", False, "m1-warm", True])
    result = _interactive(prompter, player=player)
    assert [voice for _, voice, _, _, _ in player.played[:2]] == ["f1-calm", "m1-warm"]
    assert prompter.asked[3][2] == "f1-calm"  # the next menu starts at the last candidate
    assert result.values["voice"] == "m1-warm"


def test_unknown_current_voice_starts_at_the_first_voice() -> None:
    prompter = ScriptedPrompter()
    current = replace(CURRENT, voice=VoiceName("gone"))
    result = _interactive(prompter, current=current)
    assert prompter.asked[1][2] == "f1-calm"
    assert result.values["voice"] == "f1-calm"


def test_no_voice_is_a_setting_error() -> None:
    with pytest.raises(InvalidSetting, match="no voice"):
        _interactive(ScriptedPrompter(), voices=[])


def test_rate_step_plays_a_sample_after_each_answer() -> None:
    player = FakeSamplePlayer()
    prompter = ScriptedPrompter([ENTER, ENTER, ENTER, ENTER, 2.0, False, 1.5, True])
    result = _interactive(prompter, player=player)
    assert [rate for _, _, _, rate, _ in player.played[1:3]] == [2.0, 1.5]
    assert result.values["rate"] == "1.5"


def test_volume_step_plays_a_sample_after_each_change() -> None:
    player = FakeSamplePlayer()
    answers = [ENTER] * 6 + [80, False, 30.4, False, ENTER, True]
    result = _interactive(ScriptedPrompter(answers), player=player)
    assert [volume for _, _, _, _, volume in player.played[2:]] == [80, 30, 30]
    assert result.values["volume"] == "30"


def test_out_of_range_answer_is_rejected_before_writing() -> None:
    store = InMemorySettingsStore()
    with pytest.raises(InvalidSetting, match="rate"):
        _interactive(ScriptedPrompter([ENTER, ENTER, ENTER, ENTER, 9.0]), store=store)
    assert store.writes == []


@pytest.mark.parametrize(
    ("seconds", "shown", "extra"),
    [(0, "0", False), (600, "10m", False), (7200, "2h", True), (90, "90s", True)],
)
def test_idle_step_offers_the_current_timeout(seconds: int, shown: str, extra: bool) -> None:
    prompter = ScriptedPrompter()
    result = _interactive(prompter, current=replace(CURRENT, idle_timeout_s=seconds))
    idle_options = prompter.options[-1]
    assert prompter.asked[-2][2] == shown
    assert (Option(shown, f"{shown}: keep the current timeout") in idle_options) is extra
    assert result.values["idle_timeout"] == shown
    assert result.changed is False


# ---------------------------------------------------------------- flag mode


def _flags(
    flags: dict[str, object],
    *,
    store: InMemorySettingsStore | None = None,
    models: FakeModelStore | None = None,
    accept: bool = False,
    current: Settings = CURRENT,
) -> SetupResult:
    return apply_flags(
        store if store is not None else InMemorySettingsStore(),
        models if models is not None else FakeModelStore(present=["omnivoice", "fake"]),
        [FAKE, OMNI],
        current,
        flags,
        accept,
    )


def test_flags_override_and_missing_flags_keep_current_values() -> None:
    store = InMemorySettingsStore()
    result = _flags(
        {
            "engine": "fake",
            "voice": "m1-warm",
            "base_lang": "en",
            "rate": 1.5,
            "volume": 20,
            "idle_timeout": "5m",
            "autostart": False,
            "lang": None,
        },
        store=store,
    )
    assert result.values == {
        "engine": "fake",
        "voice": "m1-warm",
        "base_lang": "en",
        "rate": "1.5",
        "volume": "20",
        "idle_timeout": "5m",
        "autostart": "false",
    }
    assert result.changed is True
    assert store.completed is True
    assert _flags({"volume": None}) == SetupResult(DEFAULT_VALUES, changed=False)


def test_flags_may_choose_a_hidden_engine() -> None:
    models = FakeModelStore(present=["fake"])  # repo-less models are always present
    result = _flags({"engine": "fake"}, models=models)
    assert result.values["engine"] == "fake"
    assert models.downloads == []


def test_absent_model_without_license_flag_fails_and_writes_nothing() -> None:
    store = InMemorySettingsStore()
    models = FakeModelStore(present=[])
    with pytest.raises(InvalidSetting, match="model download needs --accept-model-license"):
        _flags({}, store=store, models=models, accept=False)
    assert store.writes == []
    assert store.completed is False
    assert models.downloads == []


def test_absent_model_with_license_flag_downloads_silently() -> None:
    models = FakeModelStore(present=[])
    result = _flags({}, models=models, accept=True)
    assert models.downloads == ["omnivoice"]
    assert result.changed is False


def test_present_model_needs_no_license() -> None:
    models = FakeModelStore(present=["omnivoice"])
    _flags({"rate": 2.0}, models=models, accept=False)
    assert models.downloads == []


@pytest.mark.parametrize(
    "flags",
    [
        {"engine": "nope"},
        {"voice": ""},
        {"base_lang": "de"},
        {"rate": 9.0},
        {"volume": 101},
        {"volume": 5.5},
        {"idle_timeout": "soon"},
        {"autostart": "maybe"},
        {"history": 3},
    ],
    ids=str,
)
def test_invalid_flags_are_setting_errors_and_write_nothing(flags: dict[str, object]) -> None:
    store = InMemorySettingsStore()
    with pytest.raises(InvalidSetting):
        _flags(flags, store=store, accept=True)
    assert store.writes == []
    assert store.completed is False


# ---------------------------------------------------------------- one profile, interactive

SAME = ":default"  # the menu value of "same as default"
SET = ":set"  # the menu value of "set a value for this profile"


def _profile(
    prompter: ScriptedPrompter,
    store: InMemorySettingsStore,
    *,
    player: FakeSamplePlayer | None = None,
    profile: str = "claude",
    voices: list[str] | None = None,
) -> SetupResult:
    return run_profile_interactive(
        store,
        prompter,
        player if player is not None else FakeSamplePlayer(),
        VOICES if voices is None else voices,
        profile,
    )


def test_profile_questions_show_the_effective_value_and_its_source() -> None:
    store = InMemorySettingsStore(
        {"rate": 1.2, "engine": "omnivoice"}, profiles={"claude": {"voice": "m1-warm"}}
    )
    prompter = ScriptedPrompter()  # Enter everywhere
    result = _profile(prompter, store)
    questions = _questions(prompter)
    kinds = [kind for kind, _, _ in prompter.asked]
    assert kinds == ["choose", "confirm", "choose", "choose", "choose", "choose", "choose"]
    assert questions[0] == "Voice (now m1-warm, set in this profile; you will hear a sample)"
    assert questions[2] == "Base language (now pl, built-in)"
    assert questions[3] == "Rate (now 1.2, from default)"
    assert questions[4] == "Volume (now 50, built-in)"
    assert questions[5] == "Reading language (now auto, built-in)"
    assert questions[6] == "Quality steps (now 12, built-in)"
    assert prompter.options[4] == [
        Option(SAME, "same as default (auto)"),
        Option("auto", "auto: detect per sentence"),
        Option("pl", "pl: Polish"),
        Option("en", "en: English"),
    ]
    assert prompter.options[5] == [
        Option(SAME, "same as default (12)"),
        Option(SET, "set quality steps for this profile"),
    ]
    assert prompter.options[0] == [
        Option(SAME, "same as default (v0-default)"),
        *[Option(voice, voice) for voice in VOICES],
    ]
    assert prompter.options[1][0] == Option(SAME, "same as default (pl: Polish)")
    assert prompter.options[2] == [
        Option(SAME, "same as default (1.2)"),
        Option(SET, "set a rate for this profile"),
    ]
    assert [default for _, _, default in prompter.asked] == [
        "m1-warm",
        True,
        SAME,
        SAME,
        SAME,
        SAME,
        SAME,
    ]
    assert result == SetupResult(
        {
            "voice": "m1-warm",
            "lang": "auto",
            "base_lang": "pl",
            "rate": "1.2",
            "volume": "50",
            "steps": "12",
        },
        changed=False,
    )
    assert store.profile_writes == []
    assert store.writes == []
    assert store.completed is False  # the global setup is not this run's business


def test_same_as_default_removes_the_profile_values() -> None:
    own: dict[str, object] = {"voice": "m1-warm", "rate": 1.5, "steps": 8, "lang": "en"}
    store = InMemorySettingsStore(profiles={"claude": own})
    player = FakeSamplePlayer()
    prompter = ScriptedPrompter([SAME, True, ENTER, SAME, ENTER, SAME, ENTER, ENTER])
    result = _profile(prompter, store, player=player)
    assert prompter.asked[3][2] == SET  # the rate had a profile value
    assert prompter.asked[5][2] == "en"  # so had the language
    assert prompter.asked[6][2] == SET  # and the steps: Enter keeps them
    assert prompter.asked[7] == ("number", "Steps (4 to 32; fewer = faster)", 8)
    assert store.profiles["claude"] == {"steps": 8}
    assert result.values["voice"] == "v0-default"
    assert result.values["rate"] == "1.0"
    assert result.changed is True
    assert player.played == [(PL_SAMPLE, "v0-default", "pl", 1.5, 50)]  # the rate it had then


def test_setting_values_writes_only_those_keys_even_when_equal_to_default() -> None:
    store = InMemorySettingsStore({"voice": "f1-calm"}, profiles={"claude": {}})
    player = FakeSamplePlayer()
    answers = ["f1-calm", True, "en", SET, 1.3, True, SET, 80, False, 70, True]
    result = _profile(ScriptedPrompter(answers), store, player=player)
    assert store.profile_writes == [
        ("claude", "voice", "f1-calm"),  # an explicit pin, although default has the same voice
        ("claude", "base_lang", "en"),
        ("claude", "rate", "1.3"),
        ("claude", "volume", "70"),
    ]
    assert store.writes == []
    assert result.changed is True
    assert player.played == [
        (PL_SAMPLE, "f1-calm", "pl", 1.0, 50),
        (EN_SAMPLE, "f1-calm", "en", 1.3, 50),
        (EN_SAMPLE, "f1-calm", "en", 1.3, 80),
        (EN_SAMPLE, "f1-calm", "en", 1.3, 70),
    ]


def test_a_missing_profile_is_created_empty() -> None:
    store = InMemorySettingsStore()
    result = _profile(ScriptedPrompter(), store, profile="audiobook")
    assert store.profiles == {"audiobook": {}}
    assert result.changed is False


def test_profile_voice_loop_and_a_profile_voice_no_longer_installed() -> None:
    store = InMemorySettingsStore(profiles={"claude": {"voice": "gone"}})
    player = FakeSamplePlayer()
    prompter = ScriptedPrompter(["f1-calm", False, "m1-warm", True])
    result = _profile(prompter, store, player=player)
    assert prompter.asked[0][2] == SAME  # the stored voice is not offered
    assert prompter.asked[0][1] == "Voice (now gone, set in this profile; you will hear a sample)"
    assert prompter.asked[2][1] == (
        "Voice (now f1-calm, set in this profile; you will hear a sample)"
    )
    assert prompter.asked[2][2] == "f1-calm"
    assert [voice for _, voice, _, _, _ in player.played] == ["f1-calm", "m1-warm"]
    assert result.values["voice"] == "m1-warm"


def test_invalid_stored_profile_values_are_ignored_and_repaired() -> None:
    store = InMemorySettingsStore(
        profiles={"claude": {"rate": 9.0, "lang": "de", "steps": 99, "voice": 7}}
    )
    prompter = ScriptedPrompter([ENTER, True, ENTER, SAME])
    result = _profile(prompter, store)
    assert prompter.asked[3][1] == "Rate (now 1.0, built-in)"
    assert prompter.asked[5][1] == "Reading language (now auto, built-in)"
    assert prompter.asked[6][1] == "Quality steps (now 12, built-in)"
    assert store.profiles["claude"] == {}  # every asked key is repaired
    assert result.changed is False


def test_profile_language_and_steps_can_be_set() -> None:
    store = InMemorySettingsStore({"steps": 16}, profiles={"claude": {}})
    answers = [ENTER, True, ENTER, SAME, SAME, "en", SET, 8.4]
    result = _profile(ScriptedPrompter(answers), store)
    assert store.profile_writes == [("claude", "lang", "en"), ("claude", "steps", "8")]
    assert result.values["lang"] == "en"
    assert result.values["steps"] == "8"
    assert result.changed is True


def test_out_of_range_profile_steps_are_rejected_before_writing() -> None:
    store = InMemorySettingsStore()
    with pytest.raises(InvalidSetting, match="steps"):
        _profile(ScriptedPrompter([ENTER, True, ENTER, SAME, SAME, SAME, SET, 40]), store)
    assert store.profiles == {}


def test_profile_setup_needs_voices() -> None:
    with pytest.raises(InvalidSetting, match="no voice"):
        _profile(ScriptedPrompter(), InMemorySettingsStore(), voices=[])


@pytest.mark.parametrize("name", ["default", "Bad Name", ""])
def test_profile_setup_refuses_default_and_invalid_names(name: str) -> None:
    store = InMemorySettingsStore()
    with pytest.raises(InvalidSetting):
        _profile(ScriptedPrompter(), store, profile=name)
    assert store.profiles == {}


def test_cancel_in_a_profile_question_writes_nothing() -> None:
    store = InMemorySettingsStore(profiles={"claude": {"rate": 1.5}})
    with pytest.raises(SetupCancelled):
        _profile(ScriptedPrompter([SAME, True, SetupCancelled()]), store)
    assert store.profiles == {"claude": {"rate": 1.5}}
    assert store.profile_writes == []


def test_out_of_range_profile_rate_is_rejected_before_writing() -> None:
    store = InMemorySettingsStore()
    with pytest.raises(InvalidSetting, match="rate"):
        _profile(ScriptedPrompter([ENTER, True, ENTER, SET, 9.0, True]), store)
    assert store.profiles == {}


# ---------------------------------------------------------------- one profile, flag mode


def _profile_flags(  # noqa: PLR0913  # keyword-only overrides of the defaults
    flags: dict[str, object],
    *,
    store: InMemorySettingsStore | None = None,
    models: FakeModelStore | None = None,
    accept: bool = False,
    profile: str = "claude",
    current: Settings = CURRENT,
) -> SetupResult:
    return apply_profile_flags(
        store if store is not None else InMemorySettingsStore(),
        models if models is not None else FakeModelStore(present=["omnivoice", "fake"]),
        [FAKE, OMNI],
        current,
        profile,
        flags,
        accept,
    )


def test_profile_flags_create_the_profile_and_store_only_the_given_keys() -> None:
    store = InMemorySettingsStore({"rate": 1.2})
    flags: dict[str, object] = {"voice": "v0-default", "rate": 1.0, "volume": None, "steps": 8}
    result = _profile_flags(flags, store=store)
    assert store.profile_writes == [
        ("claude", "voice", "v0-default"),  # equal to the built-in default, still stored
        ("claude", "rate", "1.0"),  # the built-in value, over default's 1.2
        ("claude", "steps", "8"),
    ]
    assert store.writes == []
    assert store.completed is True
    assert result.values == {
        "voice": "v0-default",
        "lang": "auto",
        "base_lang": "pl",
        "rate": "1.0",
        "volume": "50",
        "steps": "8",
    }
    assert result.changed is True


def test_profile_flags_without_keys_create_an_empty_profile() -> None:
    store = InMemorySettingsStore()
    result = _profile_flags({"voice": None}, store=store)
    assert store.profiles == {"claude": {}}
    assert result.changed is False
    assert store.completed is True


def test_profile_flags_again_with_the_same_values_change_nothing() -> None:
    store = InMemorySettingsStore(profiles={"claude": {"lang": "en", "rate": 1.1}})
    result = _profile_flags({"lang": "en", "rate": 1.1}, store=store)
    assert store.profile_writes == []
    assert result.changed is False
    assert result.values["lang"] == "en"


def test_profile_flags_may_target_the_default_profile() -> None:
    store = InMemorySettingsStore({"voice": "m1-warm"})
    result = _profile_flags({"base_lang": "en"}, store=store, profile="default")
    assert store.profile_writes == [("default", "base_lang", "en")]
    assert store.profiles == {}
    assert result.values["voice"] == "m1-warm"


def test_profile_flags_keep_invalid_values_of_keys_not_given() -> None:
    store = InMemorySettingsStore(profiles={"claude": {"rate": 9.0}})
    result = _profile_flags({"volume": 30}, store=store)
    assert store.profiles["claude"] == {"rate": 9.0, "volume": "30"}
    assert result.values["rate"] == "1.0"


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"engine": "fake"}, "'engine' is not a profile setting"),
        ({"idle_timeout": "5m"}, "'idle_timeout' is not a profile setting"),
        ({"rate": 9.0}, "rate"),
        ({"lang": "de"}, "lang"),
        ({"steps": 2}, "steps"),
    ],
    ids=str,
)
def test_invalid_profile_flags_write_nothing(flags: dict[str, object], message: str) -> None:
    store = InMemorySettingsStore()
    with pytest.raises(InvalidSetting, match=message):
        _profile_flags(flags, store=store, accept=True)
    assert store.profiles == {}
    assert store.completed is False


def test_profile_flags_refuse_an_invalid_name() -> None:
    store = InMemorySettingsStore()
    with pytest.raises(InvalidSetting, match="invalid profile name 'Claude'"):
        _profile_flags({}, store=store, profile="Claude")


def test_profile_flags_need_the_license_for_an_absent_model() -> None:
    store = InMemorySettingsStore()
    models = FakeModelStore(present=[])
    with pytest.raises(InvalidSetting, match="--accept-model-license"):
        _profile_flags({"rate": 1.1}, store=store, models=models)
    assert store.profiles == {}
    result = _profile_flags({"rate": 1.1}, store=store, models=models, accept=True)
    assert models.downloads == ["omnivoice"]
    assert result.changed is True
