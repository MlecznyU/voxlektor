import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.samples import MIXED_PL_EN, READING_SENTENCES, TWO_SHORT_PL
from voxlektor.domain.reading import AheadWindow, Chunk, UnderrunTracker, plan_reading
from voxlektor.domain.text import limits
from voxlektor.domain.text.chunking import split_text
from voxlektor.domain.text.lang import detect_lang
from voxlektor.domain.types import ChunkId, Epoch, Lang, LangPref

EPOCH = Epoch(3)


def _chunk(i: int) -> Chunk:
    return Chunk(ChunkId(i), f"text {i}", "en", 1.0, EPOCH)


def test_ahead_window_returns_first_pending_below_limit() -> None:
    pending = [_chunk(1), _chunk(2)]
    window = AheadWindow(2)
    assert window.next_to_generate(pending, 0) is pending[0]
    assert window.next_to_generate(pending, 1) is pending[0]


def test_ahead_window_returns_none_at_limit() -> None:
    window = AheadWindow(2)
    assert window.next_to_generate([_chunk(1)], 2) is None
    assert window.next_to_generate([_chunk(1)], 3) is None


def test_ahead_window_returns_none_when_nothing_pending() -> None:
    assert AheadWindow(2).next_to_generate([], 0) is None


def test_ahead_window_exposes_its_limit() -> None:
    assert AheadWindow(2).ahead == 2


def test_underrun_counted_only_when_more_expected() -> None:
    tracker = UnderrunTracker()
    assert tracker.count == 0
    tracker.on_queue_empty(more_expected=False)
    assert tracker.count == 0
    tracker.on_queue_empty(more_expected=True)
    tracker.on_queue_empty(more_expected=True)
    assert tracker.count == 2


def test_plan_auto_detects_language_per_chunk() -> None:
    chunks = plan_reading(MIXED_PL_EN, "auto", "en", 1.0, EPOCH, 10)
    assert [c.lang for c in chunks] == ["pl", "en"]
    assert [c.text for c in chunks] == ["Zażółć gęślą jaźń.", "Hello world, this is English."]


def test_plan_ids_rate_and_epoch() -> None:
    chunks = plan_reading(MIXED_PL_EN, "auto", "en", 1.25, EPOCH, 10)
    assert [c.id for c in chunks] == [ChunkId(10), ChunkId(11)]
    assert all(c.rate == 1.25 and c.epoch == EPOCH for c in chunks)


def test_plan_volume() -> None:
    chunks = plan_reading(MIXED_PL_EN, "auto", "en", 1.0, EPOCH, 0, volume=30)
    assert [c.volume for c in chunks] == [30, 30]
    assert all(c.volume == 50 for c in plan_reading(MIXED_PL_EN, "pl", "en", 1.0, EPOCH, 0))
    assert _chunk(1).volume == 50


def test_plan_auto_uses_base_lang_on_ties() -> None:
    assert plan_reading("npm run dev", "auto", "pl", 1.0, EPOCH, 0)[0].lang == "pl"
    assert plan_reading("npm run dev", "auto", "en", 1.0, EPOCH, 0)[0].lang == "en"


def test_plan_auto_merges_short_same_language_chunks() -> None:
    chunks = plan_reading(TWO_SHORT_PL, "auto", "en", 1.0, EPOCH, 0)
    assert len(chunks) == 1
    assert chunks[0].lang == "pl"
    assert chunks[0].text == "Zażółć gęślą jaźń. Właśnie sprawdziłem wszystkie testy."


def test_plan_auto_merges_short_english_chunks() -> None:
    text = "I merged the branch.\n\nI need to go to the store."
    chunks = plan_reading(text, "auto", "pl", 1.0, EPOCH, 0)
    assert [c.text for c in chunks] == ["I merged the branch. I need to go to the store."]
    assert chunks[0].lang == "en"


def test_plan_auto_merges_a_chain_of_three_short_same_language_chunks() -> None:
    text = "I merged the branch.\n\nI need to go to the store.\n\nI merged the branch."
    chunks = plan_reading(text, "auto", "pl", 1.0, EPOCH, 0)
    assert [c.text for c in chunks] == [
        "I merged the branch. I need to go to the store. I merged the branch."
    ]


def test_plan_auto_chain_stops_once_the_accumulated_chunk_reaches_the_threshold() -> None:
    text = "\n\n".join(["I merged the branch."] * 4)  # 20 + 21 + 21 = 62 >= 60 after three
    chunks = plan_reading(text, "auto", "pl", 1.0, EPOCH, 0)
    assert [c.text for c in chunks] == [
        " ".join(["I merged the branch."] * 3),
        "I merged the branch.",
    ]


@settings(max_examples=200)
@given(
    sentences=st.lists(st.sampled_from(READING_SENTENCES), max_size=10),
    separators=st.lists(st.sampled_from([" ", "\n\n"]), min_size=10, max_size=10),
    base_lang=st.sampled_from(["pl", "en"]),
    first_id=st.integers(0, 1000),
)
def test_plan_auto_properties(
    sentences: list[str], separators: list[str], base_lang: Lang, first_id: int
) -> None:
    text = "".join(s + sep for s, sep in zip(sentences, separators, strict=False))
    chunks = plan_reading(text, "auto", base_lang, 1.0, EPOCH, first_id)

    assert [c.id for c in chunks] == [ChunkId(first_id + i) for i in range(len(chunks))]
    assert "".join("".join(c.text.split()) for c in chunks) == "".join(text.split())
    assert all(len(c.text) <= limits.MAX_CHARS for c in chunks)
    # The cap is skipped only when no cut leaves both pieces MERGE_BELOW long, which with
    # ordinary words happens only for a first chunk shorter than FIRST_MAX_CHARS + MERGE_BELOW.
    assert not chunks or len(chunks[0].text) < limits.FIRST_MAX_CHARS + limits.MERGE_BELOW

    # Every chunk is a run of unmerged pieces that all share the chunk's language. The capped
    # first chunk and the one after it (its remainder, same language) are checked as one run.
    runs = [(c.text, c.lang) for c in chunks]
    if len(runs) >= 2 and runs[0][1] == runs[1][1]:
        runs[:2] = [(f"{runs[0][0]} {runs[1][0]}", runs[0][1])]
    pieces = [(p, detect_lang(p, base_lang)) for p in split_text(text, merge_below=0)]
    position = 0
    for run_text, run_lang in runs:
        taken: list[str] = []
        while " ".join(taken) != run_text:
            piece, piece_lang = pieces[position]
            assert piece_lang == run_lang
            taken.append(piece)
            position += 1
    assert position == len(pieces)


_SENTENCES_280 = (
    "The first sentence of this paragraph is short enough to start at once. "
    "The second sentence carries on with a little more detail about the topic at hand. "
    "The third and last sentence closes the paragraph with a few more words to read aloud."
)


def test_first_max_chars_is_smaller_than_max_chars() -> None:
    assert limits.FIRST_MAX_CHARS == 120
    assert limits.FIRST_MAX_CHARS < limits.MAX_CHARS


@pytest.mark.parametrize("lang", ["auto", "en"])
def test_plan_caps_only_the_first_chunk_to_its_first_sentences(lang: LangPref) -> None:
    assert 230 <= len(_SENTENCES_280) <= 300
    chunks = plan_reading(_SENTENCES_280, lang, "pl", 1.0, EPOCH, 7)
    first, second, third = _SENTENCES_280.split(". ")
    assert [c.text for c in chunks] == [f"{first}.", f"{second}. {third}"]
    assert [c.id for c in chunks] == [ChunkId(7), ChunkId(8)]
    assert all(c.lang == "en" for c in chunks)


def test_plan_splits_a_long_first_sentence_on_spaces() -> None:
    sentence = ("word " * 40).strip() + " end."  # about 200 characters, no sentence break
    assert len(sentence) == 204
    chunks = plan_reading(sentence, "en", "pl", 1.0, EPOCH, 0)
    assert len(chunks) == 2
    assert len(chunks[0].text) <= limits.FIRST_MAX_CHARS
    assert f"{chunks[0].text} {chunks[1].text}" == sentence
    assert not chunks[0].text.endswith(" ")


def test_plan_short_text_stays_one_chunk() -> None:
    text = "A short reading that fits in the first chunk as it is."
    assert [c.text for c in plan_reading(text, "auto", "en", 1.0, EPOCH, 0)] == [text]
    exactly = ("x " * 60).strip()[:119] + "."
    assert len(exactly) == limits.FIRST_MAX_CHARS
    assert [c.text for c in plan_reading(exactly, "en", "en", 1.0, EPOCH, 0)] == [exactly]


def test_plan_remainder_joins_the_next_chunk_of_the_same_language_when_it_fits() -> None:
    head = (
        "This opening sentence is a fair bit longer than usual, so that together with the "
        "next one it overflows the first chunk."
    )
    tail = "It is followed by a second one."
    assert len(head) + 1 + len(tail) > limits.FIRST_MAX_CHARS
    nxt = "Then a new paragraph begins here."
    chunks = plan_reading(f"{head} {tail}\n\n{nxt}", "en", "pl", 1.0, EPOCH, 0)
    assert [c.text for c in chunks] == [head, f"{tail} {nxt}"]


def test_plan_remainder_stays_alone_when_the_next_chunk_is_another_language() -> None:
    english = (
        "This opening English sentence is long enough to need the cap on its own here. "
        "And this second English sentence comes right after it in the same paragraph."
    )
    polish = "Zażółć gęślą jaźń, a potem jeszcze jedno polskie zdanie na koniec tego tekstu."
    chunks = plan_reading(f"{english}\n\n{polish}", "auto", "en", 1.0, EPOCH, 0)
    assert [c.lang for c in chunks] == ["en", "en", "pl"]
    assert chunks[2].text == polish


def test_plan_remainder_stays_alone_when_joining_would_exceed_max_chars() -> None:
    first = (
        "This opening sentence is long enough to be capped all by itself in this test. "
        "And here a second sentence follows it within the same opening paragraph."
    )
    big = ("the " * 60).strip() + "."
    chunks = plan_reading(f"{first}\n\n{big}", "en", "pl", 1.0, EPOCH, 0)
    assert len(chunks) == 3
    assert chunks[2].text == big


_LONG_PL = (
    "Wczoraj wieczorem długo rozmawialiśmy o tym, jak działa nowy lektor i dlaczego pierwszy "
    "kawałek tekstu nie musi być wcale, koniec."
)


def _speakable(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


@pytest.mark.parametrize("lang", ["auto", "pl"])
def test_plan_cap_never_leaves_a_short_remainder_of_one_sentence(lang: LangPref) -> None:
    assert limits.FIRST_MAX_CHARS < len(_LONG_PL) <= 140
    # The plain cap would leave a tiny tail ("wcale, koniec.").
    tail = split_text(_LONG_PL, max_chars=limits.FIRST_MAX_CHARS, merge_below=0)[-1]
    assert len(tail) < limits.MERGE_BELOW
    chunks = plan_reading(_LONG_PL, lang, "pl", 1.0, EPOCH, 4)
    assert [c.id for c in chunks] == [ChunkId(4), ChunkId(5)]
    assert f"{chunks[0].text} {chunks[1].text}" == _LONG_PL
    assert len(chunks[0].text) <= limits.FIRST_MAX_CHARS
    assert all(len(c.text) >= limits.MERGE_BELOW for c in chunks)
    assert all(c.lang == "pl" for c in chunks)


def test_plan_cap_prefers_an_earlier_sentence_end_over_a_cut_mid_sentence() -> None:
    first = "This opening sentence is written to be a little over seventy characters."
    second = "A middle one comes in second."
    third = "And the closing words follow it here."
    text = f"{first} {second} {third}"
    assert limits.FIRST_MAX_CHARS < len(text) <= limits.MAX_CHARS
    assert len(first) >= limits.MERGE_BELOW
    assert len(first) + 1 + len(second) <= limits.FIRST_MAX_CHARS  # the plain cap keeps both
    assert len(third) < limits.MERGE_BELOW <= len(second) + 1 + len(third)
    chunks = plan_reading(f"{text}\n\nZażółć gęślą jaźń.", "auto", "en", 1.0, EPOCH, 0)
    assert [c.text for c in chunks] == [first, f"{second} {third}", "Zażółć gęślą jaźń."]


def test_plan_cap_is_skipped_when_no_split_gives_two_reasonable_pieces() -> None:
    text = "x" * 115 + " abc def."  # one huge token: any cut leaves a piece under MERGE_BELOW
    assert len(text) > limits.FIRST_MAX_CHARS
    assert [c.text for c in plan_reading(text, "en", "en", 1.0, EPOCH, 0)] == [text]


@pytest.mark.parametrize("lang", ["auto", "en"])
def test_plan_attaches_a_punctuation_only_chunk_to_the_one_before(lang: LangPref) -> None:
    sentence = "This sentence is long enough to stand on its own as a chunk here."
    assert len(sentence) >= limits.MERGE_BELOW
    chunks = plan_reading(f"{sentence}\n\n!!! ... ---", lang, "pl", 1.0, EPOCH, 2)
    assert [c.text for c in chunks] == [f"{sentence} !!! ... ---"]
    assert [c.id for c in chunks] == [ChunkId(2)]
    assert chunks[0].lang == "en"


def test_plan_attaches_a_leading_punctuation_only_chunk_to_the_one_after() -> None:
    sentence = "This sentence is long enough to stand on its own as a chunk here."
    chunks = plan_reading(f"* * *\n\n{sentence}", "auto", "pl", 1.0, EPOCH, 0)
    assert [(c.text, c.lang) for c in chunks] == [(f"* * * {sentence}", "en")]


def test_plan_drops_punctuation_with_no_neighbour_to_join() -> None:
    assert plan_reading("!!! ...", "auto", "en", 1.0, EPOCH, 0) == []
    assert plan_reading("—", "pl", "en", 1.0, EPOCH, 0) == []
    big = ("the " * 74).strip() + "."  # 296 characters: nothing else fits next to it
    dashes = "-- -- --"
    chunks = plan_reading(f"Hello world.\n\n{big}\n\n{dashes}\n\n{big}", "en", "en", 1.0, EPOCH, 9)
    assert all(_speakable(c.text) for c in chunks)
    assert [c.id for c in chunks] == [ChunkId(9 + i) for i in range(len(chunks))]
    assert dashes not in " ".join(c.text for c in chunks)
    assert [c.text for c in chunks][-1] == big


def test_plan_auto_never_merges_across_languages() -> None:
    chunks = plan_reading(MIXED_PL_EN, "auto", "en", 1.0, EPOCH, 0)
    assert len(chunks) == 2


def test_plan_auto_does_not_merge_when_earlier_chunk_is_long() -> None:
    long_first = ("the " * 16).strip() + "."
    assert len(long_first) >= 60
    chunks = plan_reading(f"{long_first}\n\nHello world.", "auto", "en", 1.0, EPOCH, 0)
    assert [c.text for c in chunks] == [long_first, "Hello world."]


def test_plan_auto_does_not_merge_when_result_exceeds_limit() -> None:
    big = ("the " * 72).strip() + "."
    assert len(big) == 288
    chunks = plan_reading(f"Hello world.\n\n{big}", "auto", "en", 1.0, EPOCH, 0)
    assert [c.text for c in chunks] == ["Hello world.", big]


def test_plan_forced_language_applies_to_every_chunk() -> None:
    chunks = plan_reading(MIXED_PL_EN, "en", "pl", 1.0, EPOCH, 0)
    assert [c.lang for c in chunks] == ["en"]  # split_text defaults merge the short paragraphs
    assert plan_reading(MIXED_PL_EN, "pl", "en", 1.0, EPOCH, 0)[0].lang == "pl"


def test_plan_empty_text_gives_no_chunks() -> None:
    assert plan_reading("  \n\n ", "auto", "en", 1.0, EPOCH, 0) == []
