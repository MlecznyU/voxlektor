"""Split text into speakable chunks."""

import re

from voxlektor.domain.text import limits

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Split one overlong sentence on spaces; hard-cut only a single oversized token."""
    pieces: list[str] = []
    current = ""
    for token in sentence.split():
        word = token
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    pieces.append(current)  # never empty: the sentence is non-empty and normalised
    return pieces


def _pack(units: list[str], max_chars: int) -> list[str]:
    """Greedily join units with a space while the result stays within max_chars."""
    packed: list[str] = []
    for unit in units:
        if packed and len(packed[-1]) + 1 + len(unit) <= max_chars:
            packed[-1] = f"{packed[-1]} {unit}"
        else:
            packed.append(unit)
    return packed


def _paragraph_chunks(paragraph: str, max_chars: int) -> list[str]:
    units: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(_split_long(sentence, max_chars))
    return _pack(units, max_chars)


def split_text(
    text: str, max_chars: int = limits.MAX_CHARS, merge_below: int = limits.MERGE_BELOW
) -> list[str]:
    """Split text into chunks of at most max_chars (except never inside a word).

    Paragraphs (blank lines) and sentences are boundaries; consecutive chunks
    shorter than merge_below are merged with the next one when the result fits.

    Raises ValueError when max_chars < 1 or merge_below < 0.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}")
    if merge_below < 0:
        raise ValueError(f"merge_below must not be negative, got {merge_below}")
    chunks: list[str] = []
    for raw in _PARAGRAPH.split(text):
        paragraph = " ".join(raw.split())
        if paragraph:
            chunks.extend(_paragraph_chunks(paragraph, max_chars))

    merged: list[str] = []
    for chunk in chunks:
        previous = merged[-1] if merged else ""
        if previous and len(previous) < merge_below and len(previous) + 1 + len(chunk) <= max_chars:
            merged[-1] = f"{previous} {chunk}"
        else:
            merged.append(chunk)
    return merged
