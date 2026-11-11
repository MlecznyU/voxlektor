import pytest
from hypothesis import given
from hypothesis import strategies as st

from voxlektor.domain.text.chunking import split_text


def test_short_paragraphs_are_merged() -> None:
    assert split_text("A.\n\nB.") == ["A. B."]


def test_two_long_paragraphs_stay_separate() -> None:
    a, b = "a" * 100, "b" * 100
    assert split_text(f"{a}\n\n{b}") == [a, b]


def test_blank_input_gives_no_chunks() -> None:
    assert split_text("  \n\n ") == []
    assert split_text("") == []


def test_long_unpunctuated_sentence_splits_on_spaces() -> None:
    text = " ".join(["word"] * 140)
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert all(w == "word" for c in chunks for w in c.split())


def test_sentences_are_packed_greedily() -> None:
    s = "x" * 100 + "."
    chunks = split_text(" ".join([s] * 4))
    assert chunks == [" ".join([s] * 2), " ".join([s] * 2)]


def test_sentence_enders_include_ellipsis_char() -> None:
    s = "y" * 150
    assert split_text(f"{s}… {s}! {s}?") == [f"{s}…", f"{s}!", f"{s}?"]


def test_overlong_single_token_is_hard_split() -> None:
    chunks = split_text("z" * 700)
    assert chunks == ["z" * 300, "z" * 300, "z" * 100]


def test_whitespace_is_normalised() -> None:
    assert split_text("a   b\tc\nd") == ["a b c d"]


def test_merge_skipped_when_result_does_not_fit() -> None:
    short, long_ = "s" * 10, "l" * 295
    assert split_text(f"{short}\n\n{long_}") == [short, long_]


def test_custom_limits() -> None:
    assert split_text("aaa bbb ccc", max_chars=7, merge_below=0) == ["aaa bbb", "ccc"]


@pytest.mark.parametrize("max_chars", [0, -1])
def test_invalid_max_chars_raises(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_text("abc", max_chars=max_chars)


def test_negative_merge_below_raises() -> None:
    with pytest.raises(ValueError, match="merge_below"):
        split_text("abc", merge_below=-1)


@given(
    st.text(alphabet="abc def.\n", max_size=1500),
    st.integers(min_value=1, max_value=40),
    st.integers(min_value=0, max_value=20),
)
def test_chunks_fit_and_preserve_content(text: str, max_chars: int, merge_below: int) -> None:
    chunks = split_text(text, max_chars=max_chars, merge_below=merge_below)
    # Tokens longer than max_chars are hard-cut, so no chunk may exceed the limit.
    assert all(len(c) <= max_chars for c in chunks)
    assert all(c for c in chunks)
    assert "".join(text.split()) == "".join("".join(chunks).split())


def test_overlong_token_after_words_flushes_the_words_first() -> None:
    assert split_text("ab " + "z" * 350, max_chars=300) == ["ab", "z" * 300, "z" * 50]


def test_overlong_token_with_exact_multiple_length() -> None:
    assert split_text("z" * 600) == ["z" * 300, "z" * 300]
