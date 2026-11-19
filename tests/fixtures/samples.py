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

# 110 characters; the latency probe of the real-model test (tests/adapters/engines).
PL_PARAGRAPH_110 = (
    "Sprawdziłem testy i wszystko przechodzi. Jedyny problem to ostrzeżenie przy buildzie, "
    "ale nie blokuje niczego."
)

# Three-paragraph texts of a typical conversation reply (copied from the tts-lab benchmark);
# scripts/bench_gaps.py reads them aloud once per voice, rate and language.
BENCH_PARAGRAPHS_PL: tuple[str, ...] = (
    "Sprawdziłem testy i wszystko przechodzi. Jedyny problem to ostrzeżenie przy buildzie, "
    "ale nie blokuje niczego.",
    "Mam dwie opcje. Pierwsza jest szybsza, ale trudniej ją potem rozbudować. Druga zajmie "
    "więcej czasu, za to zostawi nam porządny fundament.",
    "Polecam drugą. Który kierunek wybierasz?",
)
BENCH_PARAGRAPHS_EN: tuple[str, ...] = (
    "I checked the tests and everything passes. The only issue is a build warning, but it "
    "doesn't block anything.",
    "There are two options. The first is faster, but harder to extend later. The second takes "
    "longer, but leaves us a solid foundation.",
    "I'd go with the second. Which way do you want to go?",
)

# A five-paragraph Polish code-review reply with English terms (1220 characters);
# scripts/bench_latency.py reads it one speak per paragraph, as dialogue-box does.
LATENCY_REPLY_PL: tuple[str, ...] = (
    "Przejrzałem twój pull request i ogólnie wygląda dobrze. Zmiany w daemonie są spójne z "
    "architekturą, a testy przechodzą bez błędów. Mam jednak kilka uwag, które warto poprawić "
    "przed scaleniem.",
    "Po pierwsze, w module dialogue-box kolejkujesz każdy akapit osobno, ale nie obsługujesz "
    "sytuacji, w której daemon zwróci błąd w połowie odpowiedzi. Wtedy użytkownik usłyszy tylko "
    "część tekstu i nie dostanie żadnego komunikatu. Proponuję dodać prosty retry oraz czytelny "
    "log.",
    "Po drugie, nowe API do statusu zwraca pole z czasem w milisekundach, a reszta protokołu "
    "używa sekund. To drobiazg, ale łatwo o pomyłkę po stronie klienta. Lepiej trzymać jedną "
    "jednostkę w całym kontrakcie.",
    "Po trzecie, pokrycie testami spadło w warstwie app poniżej wymaganego progu. Brakuje testu "
    "dla ścieżki, w której sesja dostaje polecenie stop w trakcie generowania. Symulator "
    "powinien to łatwo odtworzyć, bez wątków i bez opóźnień. Dopisz jeden przypadek, a gate "
    "znowu będzie zielony.",
    "Poza tym nie widzę większych problemów. Jeśli poprawisz te trzy rzeczy, chętnie zatwierdzę "
    "zmiany. Daj znać, gdybyś chciał, żebym przejrzał kod jeszcze raz po aktualizacji.",
)

# Short paragraphs for plan_reading: one PL then one EN, and two PL ones.
MIXED_PL_EN = "Zażółć gęślą jaźń.\n\nHello world, this is English."
TWO_SHORT_PL = "Zażółć gęślą jaźń.\n\nWłaśnie sprawdziłem wszystkie testy."

# Sentences of both languages for property tests of plan_reading.
READING_SENTENCES: tuple[str, ...] = (
    "Zażółć gęślą jaźń.",
    "Właśnie sprawdziłem wszystkie testy.",
    "Jedyny problem to ostrzeżenie przy buildzie, ale nie blokuje niczego.",
    "Hello world, this is English.",
    "I merged the branch.",
    "I need to go to the store and buy some bread for the weekend.",
    "npm run dev",
)
