import pytest

from tests.fixtures.samples import LANG_TABLE, REPEATED_EN_ONE_PL
from voxlektor.domain.text.lang import detect_lang
from voxlektor.domain.types import Lang


@pytest.mark.parametrize(("text", "fallback", "expected"), LANG_TABLE)
def test_detect_lang_table(text: str, fallback: Lang, expected: Lang) -> None:
    assert detect_lang(text, fallback) == expected


def test_detect_lang_counts_distinct_markers_not_repeats() -> None:
    assert detect_lang(REPEATED_EN_ONE_PL, "pl") == "pl"
    assert detect_lang(REPEATED_EN_ONE_PL, "en") == "en"


def test_detect_lang_is_case_insensitive() -> None:
    assert detect_lang("THE And", "pl") == "en"
