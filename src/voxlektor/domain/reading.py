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


def plan_reading(  # noqa: PLR0913, PLR0917  # frozen signature (plan Task 6, ruling R19)
    text: str,
    lang: LangPref,
    base_lang: Lang,
    rate: float,
    epoch: Epoch,
    first_id: int,
) -> list[Chunk]:
    """Split ``text`` into chunks with consecutive ids starting at ``first_id``.

    With ``lang == "auto"`` every chunk is detected on its own (ties go to ``base_lang``)
    and short neighbours are merged only when they share a language.
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
    return [
        Chunk(ChunkId(first_id + i), t, chunk_lang, rate, epoch)
        for i, (t, chunk_lang) in enumerate(planned)
    ]
