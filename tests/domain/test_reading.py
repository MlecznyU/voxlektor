from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.samples import MIXED_PL_EN, READING_SENTENCES, TWO_SHORT_PL
from voxlektor.domain.reading import AheadWindow, Chunk, UnderrunTracker, plan_reading
from voxlektor.domain.text.chunking import split_text
from voxlektor.domain.text.lang import detect_lang
from voxlektor.domain.types import ChunkId, Epoch, Lang

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

    # Every chunk is a run of unmerged pieces that all share the chunk's language.
    pieces = [(p, detect_lang(p, base_lang)) for p in split_text(text, merge_below=0)]
    position = 0
    for chunk in chunks:
        taken: list[str] = []
        while " ".join(taken) != chunk.text:
            piece, piece_lang = pieces[position]
            assert piece_lang == chunk.lang
            taken.append(piece)
            position += 1
    assert position == len(pieces)


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
