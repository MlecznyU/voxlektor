"""Reading policy: how text becomes chunks and how far generation runs ahead of playback."""

from collections.abc import Sequence
from dataclasses import dataclass

from voxlektor.domain.text import limits
from voxlektor.domain.text.chunking import split_text
from voxlektor.domain.text.lang import detect_lang
from voxlektor.domain.types import ChunkId, Epoch, Lang, LangPref


@dataclass(frozen=True)
class Chunk:
    id: ChunkId
    text: str
    lang: Lang
    rate: float
    epoch: Epoch
    volume: int = 50  # 0..100, 50 = the even level; last, so positional construction is unchanged


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


def _merge_same_language(pieces: list[tuple[str, Lang]]) -> list[tuple[str, Lang]]:
    """Merge adjacent short chunks of the same language (the split_text rule, per language)."""
    merged: list[tuple[str, Lang]] = []
    for text, lang in pieces:
        if merged:
            previous, previous_lang = merged[-1]
            if (
                previous_lang == lang
                and len(previous) < limits.MERGE_BELOW
                and len(previous) + 1 + len(text) <= limits.MAX_CHARS
            ):
                merged[-1] = (f"{previous} {text}", lang)
                continue
        merged.append((text, lang))
    return merged


def _speakable(text: str) -> bool:
    """True when ``text`` has at least one letter or digit (punctuation alone is not read)."""
    return any(ch.isalnum() for ch in text)


def _attach_unspeakable(planned: list[tuple[str, Lang]]) -> list[tuple[str, Lang]]:
    """Fold every chunk without a letter or digit into a neighbour, taking its language.

    It joins the chunk before it when the result fits ``MAX_CHARS``, else the chunk after it;
    when neither fits (or it has no neighbour) it is dropped, as there is nothing to read.
    """
    result: list[tuple[str, Lang]] = []
    carry = ""  # unspeakable text waiting for the next speakable chunk
    for text, lang in planned:
        if _speakable(text):
            joined = f"{carry} {text}"
            result.append((joined if carry and len(joined) <= limits.MAX_CHARS else text, lang))
            carry = ""
        elif result and len(result[-1][0]) + 1 + len(text) <= limits.MAX_CHARS:
            result[-1] = (f"{result[-1][0]} {text}", result[-1][1])
        else:
            carry = f"{carry} {text}" if carry else text
    return result


def _balanced_split(text: str) -> tuple[str, str] | None:
    """Split ``text`` on a space into a head of at most ``FIRST_MAX_CHARS`` and a rest, both at
    least ``MERGE_BELOW`` long and speakable; prefer a sentence end, then the longest head.
    """
    words = text.split(" ")
    best: tuple[tuple[bool, int], str, str] | None = None
    for k in range(1, len(words)):
        head, rest = " ".join(words[:k]), " ".join(words[k:])
        if (
            limits.MERGE_BELOW <= len(head) <= limits.FIRST_MAX_CHARS
            and len(rest) >= limits.MERGE_BELOW
            and _speakable(head)
            and _speakable(rest)
        ):
            key = (head[-1] in ".!?…", k)
            if best is None or key > best[0]:
                best = (key, head, rest)
    return None if best is None else (best[1], best[2])


def _cap_first(planned: list[tuple[str, Lang]]) -> list[tuple[str, Lang]]:
    """Cut the first chunk down to its first piece of at most ``FIRST_MAX_CHARS``.

    The rest of it joins the next chunk when both share a language and fit ``MAX_CHARS``.
    Otherwise it is a chunk of its own, so it must be at least ``MERGE_BELOW`` long and
    speakable: if the plain cut leaves less, the split point moves earlier
    (``_balanced_split``), and if no such split exists the first chunk is left whole.
    Later chunks are unchanged.
    """
    if not planned or len(planned[0][0]) <= limits.FIRST_MAX_CHARS:
        return planned
    (first, lang), *later = planned

    def joins_next(rest: str) -> bool:
        return (
            bool(later)
            and later[0][1] == lang
            and len(rest) + 1 + len(later[0][0]) <= (limits.MAX_CHARS)
        )

    head, *tail = split_text(first, max_chars=limits.FIRST_MAX_CHARS, merge_below=0)
    rest = " ".join(tail)
    if _speakable(head) and joins_next(rest):
        return [(head, lang), (f"{rest} {later[0][0]}", lang), *later[1:]]
    if not (_speakable(head) and _speakable(rest) and len(rest) >= limits.MERGE_BELOW):
        balanced = _balanced_split(first)
        if balanced is None:
            return planned
        head, rest = balanced
    return [(head, lang), (rest, lang), *later]


def plan_reading(  # noqa: PLR0913, PLR0917  # frozen signature (plan Task 6, ruling R19)
    text: str,
    lang: LangPref,
    base_lang: Lang,
    rate: float,
    epoch: Epoch,
    first_id: int,
    *,
    volume: int = 50,
) -> list[Chunk]:
    """Split ``text`` into chunks with consecutive ids starting at ``first_id``.

    With ``lang == "auto"`` every chunk is detected on its own (ties go to ``base_lang``)
    and short neighbours are merged only when they share a language. Every chunk carries
    ``rate``, ``epoch`` and ``volume``. A chunk without any letter or digit is folded into a
    neighbour (``_attach_unspeakable``). The first chunk is then capped to ``FIRST_MAX_CHARS``
    (its first sentences, or a long first sentence split on spaces) so that the first audio is
    ready sooner, never leaving a tiny remainder; see ``_cap_first``.
    """
    planned: list[tuple[str, Lang]]
    if lang == "auto":
        pieces: list[tuple[str, Lang]] = [
            (t, detect_lang(t, base_lang)) for t in split_text(text, merge_below=0)
        ]
        planned = _merge_same_language(pieces)
    else:
        forced: Lang = lang
        planned = [(t, forced) for t in split_text(text)]
    planned = _cap_first(_attach_unspeakable(planned))
    return [
        Chunk(ChunkId(first_id + i), t, chunk_lang, rate, epoch, volume)
        for i, (t, chunk_lang) in enumerate(planned)
    ]
