"""Sample data shared by tests and scripts. Polish strings live only here."""

from voxlektor.domain.types import Lang

# One distinct EN marker repeated, one distinct PL marker: a tie, so the fallback decides.
REPEATED_EN_ONE_PL = "the the the nie"

# (text, fallback, expected language)
LANG_TABLE: tuple[tuple[str, Lang, Lang], ...] = (
    ("Zmergowałem brancha z mainem", "pl", "pl"),
    ("Zmergowałem brancha z mainem", "en", "pl"),
    ("ŁÓDŹ", "en", "pl"),
    ("to nie jest ok", "pl", "pl"),
    ("to nie jest ok", "en", "pl"),
    ("I merged the branch", "pl", "en"),
    ("I merged the branch", "en", "en"),
    ("I need to go to the store", "pl", "en"),
    ("to be or not to be", "pl", "en"),
    ("what to do", "pl", "en"),  # "what" is an English marker, so no fallback
    ("to do", "pl", "pl"),
    ("to do", "en", "en"),
    ("npm run dev", "pl", "pl"),
    ("npm run dev", "en", "en"),
    ("", "pl", "pl"),
    ("", "en", "en"),
)
