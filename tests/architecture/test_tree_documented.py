"""The section 2 map of docs/architecture.md and the package tree match, in both directions."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "voxlektor"
ARCHITECTURE = ROOT / "docs" / "architecture.md"

_BRACES = re.compile(r"^(?P<prefix>.*)\{(?P<names>[^}]*)\}(?P<suffix>.*)$")


def _expand(token: str) -> list[str]:
    match = _BRACES.match(token)
    if match is None:
        return [token]
    names = match["names"].split(",")
    return [f"{match['prefix']}{name}{match['suffix']}" for name in names]


def documented_modules(text: str) -> set[str]:
    """Parse the section 2 code block into package-relative ``.py`` paths."""
    section = text.split("## 2.", 1)[1].split("\n## ", 1)[0]
    block = section.split("```", 2)[1]
    documented: set[str] = set()
    context = ""
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("src/"):
            continue
        indent = len(line) - len(line.lstrip())
        token = stripped.split()[0]
        if indent == 2 and token.endswith("/"):
            context = token
            continue
        prefix = "" if indent == 2 else context
        for path in _expand(token):
            documented.add(f"{prefix}{path}")
    return {path for path in documented if path.endswith(".py")}


def test_every_module_is_in_architecture_map() -> None:
    documented = documented_modules(ARCHITECTURE.read_text())
    missing = sorted(_actual_modules() - documented)
    assert not missing, f"modules missing from docs/architecture.md section 2: {missing}"


# Modules the section 2 code block may name before they exist (a later phase). Keep it explicit:
# the Phase 3 modules are listed in prose below the block, so today nothing is needed here.
LATER_PHASE: frozenset[str] = frozenset()


def _actual_modules(*, packages: bool = False) -> set[str]:
    return {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if packages or path.name != "__init__.py"
    }


def test_every_documented_module_exists() -> None:
    """M9: the map names no module that is gone (or that never was)."""
    documented = documented_modules(ARCHITECTURE.read_text())
    stale = sorted(documented - _actual_modules(packages=True) - LATER_PHASE)
    assert not stale, f"docs/architecture.md section 2 lists modules that do not exist: {stale}"


def test_stale_map_entry_would_be_detected() -> None:
    text = "## 2. Tree\n\n```\n  testing/\n    protocol_fixtures.py  gone\n```\n"
    assert documented_modules(text) - _actual_modules() == {"testing/protocol_fixtures.py"}


def test_parser_understands_the_map_notation() -> None:
    text = (
        "## 2. Tree\n\n```\nsrc/voxlektor/\n"
        "  __main__.py   entry\n"
        "  domain/       PURE\n"
        "    types.py    t\n"
        "    text/chunking.py  c\n"
        "  entrypoints/\n"
        "    cli/{app,speak}.py\n"
        "```\n\n## 3. Next\n"
    )
    assert documented_modules(text) == {
        "__main__.py",
        "domain/types.py",
        "domain/text/chunking.py",
        "entrypoints/cli/app.py",
        "entrypoints/cli/speak.py",
    }


def test_undocumented_module_would_be_detected() -> None:
    text = "## 2. Tree\n\n```\n  domain/\n    types.py  t\n```\n"
    assert "domain/extra.py" not in documented_modules(text)
