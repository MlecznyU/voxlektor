"""OmniVoice pace arithmetic: no model, no MLX."""

import pytest

from tests.fixtures.samples import BENCH_PARAGRAPHS_EN, PL_PARAGRAPH_110
from voxlektor.adapters.engines import omnivoice_pace as pace
from voxlektor.adapters.engines.omnivoice_pace import (
    LANG_CORRECTION,
    MARGIN,
    MAX_PACE,
    MIN_SECONDS,
    PACE_RANGE,
    TAIL_S,
    pace_for,
    ref_lang,
    ref_pace,
    speech_weight,
    target_seconds,
)

_COMBINING_ACUTE = chr(0x301)
_NO_BREAK_SPACE = chr(0xA0)


@pytest.mark.parametrize(
    ("text", "weight"),
    [
        ("", 0.0),
        ("abc", 3.0),
        ("zażółć", 6.0),  # precomposed Polish letters are letters
        ("a b", 2.2),
        ("a, b.", 3.2),
        ("2026", 14.0),
        ("e" + _COMBINING_ACUTE, 1.0),  # a combining accent is silent
        ("a\tb\nc", 3.4),  # control characters count as a word boundary
        ("a" + _NO_BREAK_SPACE + "b", 2.2),  # a no-break space is a space
        ("50%", 7.5),
    ],
)
def test_speech_weight_uses_the_rule_estimator_weights(text: str, weight: float) -> None:
    assert speech_weight(text) == pytest.approx(weight)


def test_speech_weight_matches_mlx_audio_on_sample_texts() -> None:
    # Values from mlx_audio's RuleDurationEstimator.calculate_total_weight (pinned library).
    assert speech_weight(PL_PARAGRAPH_110) == pytest.approx(97.3)
    assert speech_weight(BENCH_PARAGRAPHS_EN[0]) == pytest.approx(91.6)


def test_ref_pace_is_weight_per_second_of_the_stripped_transcript() -> None:
    text = "  " + "abcd efgh " * 10 + " "  # weight 10 * 8.2 + 9 * 0.2 = 83.8
    assert ref_pace(text, 6.0) == pytest.approx(83.8 / 6.0)


def test_ref_pace_is_clamped_to_plausible_speech() -> None:
    low, high = PACE_RANGE
    assert ref_pace("a" * 10, 10.0) == low  # 1 per second: a transcript far too short
    assert ref_pace("a" * 300, 10.0) == high  # 30 per second: audio cut, transcript whole


def test_ref_pace_of_a_cut_reference_counts_only_the_kept_part_of_the_transcript() -> None:
    text = "abcd efgh " * 10  # weight 83.8
    assert ref_pace(text, 3.0, kept=0.5) == pytest.approx(83.8 * 0.5 / 3.0)
    with pytest.raises(ValueError, match="kept"):
        ref_pace(text, 3.0, kept=0.0)


@pytest.mark.parametrize(("text", "seconds"), [("", 3.0), ("   ", 3.0), ("abc", 0.0)])
def test_ref_pace_rejects_an_empty_reference(text: str, seconds: float) -> None:
    with pytest.raises(ValueError, match="no speech"):
        ref_pace(text, seconds)


@pytest.mark.parametrize(
    ("text", "lang"),
    [(PL_PARAGRAPH_110, "pl"), (BENCH_PARAGRAPHS_EN[0], "en"), ("npm run dev", "en")],
)
def test_ref_lang_detects_the_reference_language(text: str, lang: str) -> None:
    assert ref_lang(text) == lang


def test_pace_for_applies_the_correction_only_across_languages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pace, "LANG_CORRECTION", {("pl", "en"): 1.2})
    assert pace_for(10.0, "pl", "pl") == 10.0
    assert pace_for(10.0, "pl", "en") == pytest.approx(12.0)
    assert pace_for(10.0, "en", "pl") == 10.0  # no entry: no correction


def test_the_measured_correction_table_covers_both_directions() -> None:
    assert set(LANG_CORRECTION) == {("pl", "en"), ("en", "pl")}
    assert all(0.8 <= ratio <= 1.25 for ratio in LANG_CORRECTION.values())


def test_target_seconds_is_estimate_times_margin_over_rate_plus_tail() -> None:
    text = "a" * 100  # weight 100
    spoken = 100 / 10.0 * MARGIN
    assert target_seconds(text, "pl", 10.0, "pl", 1.0) == pytest.approx(spoken + TAIL_S)
    assert target_seconds(text, "pl", 10.0, "pl", 1.25) == pytest.approx(spoken / 1.25 + TAIL_S)
    assert target_seconds(f"  {text}  ", "pl", 10.0, "pl", 1.0) == pytest.approx(spoken + TAIL_S)


def test_target_seconds_never_asks_for_more_than_the_max_pace() -> None:
    # 15 weight/s * 1.5 / MARGIN is above MAX_PACE: the cap decides.
    capped = target_seconds("a" * 180, "pl", 15.0, "pl", 1.5)
    assert capped == pytest.approx(180 / MAX_PACE + TAIL_S)


def test_target_seconds_has_a_floor() -> None:
    assert target_seconds("a", "pl", 18.0, "pl", 1.5) == MIN_SECONDS


def test_target_seconds_uses_the_language_pace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pace, "LANG_CORRECTION", {("pl", "en"): 1.2})
    same = target_seconds("a" * 100, "pl", 10.0, "pl", 1.0)
    other = target_seconds("a" * 100, "en", 10.0, "pl", 1.0)
    assert other - TAIL_S == pytest.approx((same - TAIL_S) / 1.2)


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_target_seconds_rejects_a_non_positive_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        target_seconds("abc", "pl", 10.0, "pl", rate)
