"""Minimal Polish/English language detection with a base-language fallback."""

import re

from voxlektor.domain.types import Lang

_DIACRITICS = frozenset("ąćęłńóśźż")
_PL_MARKERS = frozenset(
    {"się", "nie", "jest", "że", "w", "z", "na", "jak", "ale", "czy", "tak", "mamy"}
)
_EN_MARKERS = frozenset(
    {
        "the",
        "and",
        "is",
        "are",
        "was",
        "of",
        "you",
        "with",
        "this",
        "that",
        "what",
        "it",
        "for",
        "not",
        "be",
        "have",
    }
)
_TOKEN = re.compile(r"[^\W\d_]+")


def detect_lang(text: str, fallback: Lang) -> Lang:
    """Detect "pl" or "en"; ambiguous text (a tie, including 0:0) gets ``fallback``.

    Polish diacritics decide at once; otherwise the number of distinct Polish and
    English marker words is compared.
    """
    lowered = text.lower()
    if any(ch in _DIACRITICS for ch in lowered):
        return "pl"
    tokens = set(_TOKEN.findall(lowered))
    pl_count = len(tokens & _PL_MARKERS)
    en_count = len(tokens & _EN_MARKERS)
    if pl_count > en_count:
        return "pl"
    if en_count > pl_count:
        return "en"
    return fallback
