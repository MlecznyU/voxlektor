"""Reading policy: how text becomes chunks and how far generation runs ahead of playback."""

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Literal

from voxlektor.domain.text import limits
from voxlektor.domain.text.chunking import split_paragraphs, split_text
from voxlektor.domain.text.lang import detect_lang
from voxlektor.domain.types import ChunkId, Epoch, Lang, LangPref

type _Ends = Literal["sentence", "paragraph"]


@dataclass(frozen=True)
class Chunk:
    id: ChunkId
    text: str
    lang: Lang
    rate: float
    epoch: Epoch
    volume: int = 50  # 0..100, 50 = the even level; last, so positional construction is unchanged
    _: KW_ONLY
    ends: Literal["sentence", "paragraph"] = "sentence"  # how the text goes on after this chunk
    steps: int | None = None  # diffusion steps for the engine; None = the engine's default


class AheadWindow:
    """Allows generating at most ``ahead`` chunks that have not been played yet."""

    def __init__(self, ahead: int) -> None:
        self.ahead = ahead

    def next_to_generate(self, pending: Sequence[Chunk], generated_unplayed: int) -> Chunk | None:
        if pending and generated_unplayed < self.ahead:
            return pending[0]
        return None


class UnderrunTracker:
    """Counts the times playback ran dry while more audio was still expected."""

    def __init__(self) -> None:
        self.count = 0

    def on_queue_empty(self, more_expected: bool) -> None:
        if more_expected:
            self.count += 1


def _speakable(text: str) -> bool:
    """True when ``text`` has at least one letter or digit (punctuation alone is not read)."""
    return any(ch.isalnum() for ch in text)


def _plan_paragraph(paragraph: str, lang: LangPref, base_lang: Lang) -> list[tuple[str, Lang]]:
    """Split one paragraph into chunks; with ``lang == "auto"`` each is detected on its own.

    ``split_text`` packs a paragraph greedily, so no two neighbours fit ``MAX_CHARS`` together
    and nothing is left to merge; a chunk without a letter or digit (a hard-cut run of
    punctuation) has nothing to read and is dropped.
    """
    if lang == "auto":
        pieces = split_text(paragraph, merge_below=0)
        return [(t, detect_lang(t, base_lang)) for t in pieces if _speakable(t)]
    forced: Lang = lang
    return [(t, forced) for t in split_text(paragraph) if _speakable(t)]


def _balanced_split(text: str, max_chars: int) -> tuple[str, str] | None:
    """Split ``text`` on a space into a head of at most ``max_chars`` and a rest, both at
    least ``MERGE_BELOW`` long and speakable; prefer a sentence end, then the longest head.
    """
    words = text.split(" ")
    best: tuple[tuple[bool, int], str, str] | None = None
    for k in range(1, len(words)):
        head, rest = " ".join(words[:k]), " ".join(words[k:])
        if (
            limits.MERGE_BELOW <= len(head) <= max_chars
            and len(rest) >= limits.MERGE_BELOW
            and _speakable(head)
            and _speakable(rest)
        ):
            key = (head[-1] in ".!?…", k)
            if best is None or key > best[0]:
                best = (key, head, rest)
    return None if best is None else (best[1], best[2])


def _cap(
    planned: list[tuple[str, Lang, _Ends]], index: int, max_chars: int
) -> list[tuple[str, Lang, _Ends]]:
    """Cut the chunk at ``index`` down to its first piece of at most ``max_chars``.

    The rest of it joins the next chunk when that one is in the same paragraph, shares the
    language and both fit ``MAX_CHARS``. Otherwise it is a chunk of its own (keeping how the
    original chunk ended), so it must be at least ``MERGE_BELOW`` long and speakable: if the
    plain cut leaves less, the split point moves earlier (``_balanced_split``), and if no such
    split exists the chunk is left whole. Other chunks are unchanged.
    """
    if index >= len(planned) or len(planned[index][0]) <= max_chars:
        return planned
    before, (text, lang, ends), later = planned[:index], planned[index], planned[index + 1 :]

    def joins_next(rest: str) -> bool:
        # A chunk ending a sentence is never the last one, so ``later`` is not empty here.
        return (
            ends == "sentence"
            and later[0][1] == lang
            and len(rest) + 1 + len(later[0][0]) <= limits.MAX_CHARS
        )

    head, *tail = split_text(text, max_chars=max_chars, merge_below=0)
    rest = " ".join(tail)
    if _speakable(head) and joins_next(rest):
        nxt_text, _, nxt_ends = later[0]
        joined = (f"{rest} {nxt_text}", lang, nxt_ends)
        return [*before, (head, lang, "sentence"), joined, *later[1:]]
    if not (_speakable(head) and _speakable(rest) and len(rest) >= limits.MERGE_BELOW):
        balanced = _balanced_split(text, max_chars)
        if balanced is None:
            return planned
        head, rest = balanced
    return [*before, (head, lang, "sentence"), (rest, lang, ends), *later]


def plan_reading(  # noqa: PLR0913, PLR0917  # frozen signature (plan Task 6, ruling R19)
    text: str,
    lang: LangPref,
    base_lang: Lang,
    rate: float,
    epoch: Epoch,
    first_id: int,
    *,
    volume: int = 50,
    steps: int | None = None,
    cap_start: bool = True,
) -> list[Chunk]:
    """Split ``text`` into chunks with consecutive ids starting at ``first_id``.

    Paragraphs (blank lines) are planned on their own and never merged: the last chunk of
    each is marked ``ends="paragraph"`` (so also the last chunk of the speak), the others
    ``"sentence"``. With ``lang == "auto"`` every chunk is detected on its own (ties go to
    ``base_lang``). A chunk without any letter or digit is dropped. Every chunk carries
    ``rate``, ``epoch``, ``volume`` and ``steps``.

    With ``cap_start`` (a reading that starts from idle) the first chunk is capped to
    ``FIRST_MAX_CHARS`` and then the second to ``SECOND_MAX_CHARS`` (their first sentences, or
    a long sentence split on spaces) so that the first audio is ready sooner and the second is
    ready before the first has played, never leaving a tiny remainder; see ``_cap``.
    """
    planned: list[tuple[str, Lang, _Ends]] = []
    for paragraph in split_paragraphs(text):
        chunks = _plan_paragraph(paragraph, lang, base_lang)
        last = len(chunks) - 1
        planned.extend(
            (t, chunk_lang, "paragraph" if i == last else "sentence")
            for i, (t, chunk_lang) in enumerate(chunks)
        )
    if cap_start:
        planned = _cap(_cap(planned, 0, limits.FIRST_MAX_CHARS), 1, limits.SECOND_MAX_CHARS)
    return [
        Chunk(
            ChunkId(first_id + i),
            t,
            chunk_lang,
            rate,
            epoch,
            volume,
            ends=ends,
            steps=steps,
        )
        for i, (t, chunk_lang, ends) in enumerate(planned)
    ]
